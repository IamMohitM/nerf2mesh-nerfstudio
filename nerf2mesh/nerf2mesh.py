import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Type, Optional, Literal, Union
import tqdm
import trimesh
import mcubes
import numpy as np
import raymarching
import nerfacc
import nvdiffrast.torch as dr

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.instant_ngp import NGPModel, InstantNGPModelConfig
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import (
    RGBRenderer,
    AccumulationRenderer,
    DepthRenderer,
)
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerf2mesh.field import Nerf2MeshField
from nerf2mesh.utils import (
    Shading,
    clean_mesh,
    remove_masked_trigs,
    remove_selected_verts,
    decimate_mesh,
    uncontract,
)


@dataclass
class Nerf2MeshModelConfig(InstantNGPModelConfig):
    _target: Type = field(default_factory=lambda: Nerf2MeshModel)

    # occupancy grid
    grid_levels: int = 1
    density_thresh: float = 10
    alpha_thre: float = 0.0
    """Threshold for opacity skipping."""
    cone_angle: float = 0.0  # 04
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: Optional[float] = None
    """Minimum step size for rendering."""

    disable_scene_contraction: bool = True
    # near_plane: float=0.01

    hidden_dim_sigma: int = 32
    hidden_dim_color: int = 64
    num_levels_sigma_encoder: int = 16
    num_levels_color_encoder: int = 16
    n_features_per_level_sigma_encoder: int = 1
    n_features_per_level_color_encoder: int = 2
    base_resolution: int = 16  # starting resolution
    desired_resolution: int = 2048  # ending resolution

    shading_type: Shading = Shading.diffuse

    ## -- Sammpling Parameters
    min_near: float = 0.01  # same as opt.min_near (nerf2mesh)
    max_far: float = 1000  # infinite - same as opt.min_far (nerf2mesh)

    specular_dim: int = 3  # fs in paper input features for specular net

    log2hash_map_size: int = 19

    sigma_layers = 2

    background_color: Literal["random", "black", "white"] = "black"

    lambda_mask: float = 0.1
    lambda_rgb: float = 1.0
    lambda_depth: float = 0.1
    lambda_specular: float = 1e-5

    # coarse mesh params
    coarse_mesh_path: str = "meshes/mesh_0.ply"
    env_reso: int = 256
    fp16: bool = True
    clean_min_f: int = 8
    clean_min_d: int = 5
    visibility_mask_dilation: int = 5
    sdf: bool = False
    mvps: torch.tensor = None


#     # NOTE: all default configs are passed to nerf2mesh field
#     # NOTE: all dervied are computed in nerf2mesh field

#     # TODO: config for stage 1 if needed


class Nerf2MeshModel(NGPModel):
    config: Nerf2MeshModelConfig
    field: Nerf2MeshField

    def populate_modules(self):
        self.mvps = None
        self.image_height = None
        self.image_width = None
        self.glctx = None
        # super().populate_modules()
        self.scene_aabb = torch.nn.Parameter(
            self.scene_box.aabb.flatten(), requires_grad=False
        )

        self.register_buffer("bound", self.scene_box.aabb.max())

        self.field = Nerf2MeshField(
            aabb=self.scene_box.aabb,
            appearance_embedding_dim=0 if self.config.use_appearance_embedding else 32,
            num_images=self.num_train_data,
            base_res=self.config.base_resolution,
            max_res=self.config.desired_resolution,
            spatial_distortion=None,
            log2_hashmap_size=self.config.log2hash_map_size,
            num_layers_sigma=self.config.sigma_layers,
            specular_dim=self.config.specular_dim,
            hidden_dim_sigma=self.config.hidden_dim_sigma,
            hidden_dim_color=self.config.hidden_dim_color,
            num_levels_sigma_encoder=self.config.num_levels_sigma_encoder,
            num_levels_color_encoder=self.config.num_levels_color_encoder,
            n_features_per_level_sigma_encoder=self.config.n_features_per_level_sigma_encoder,
            n_features_per_level_color_encoder=self.config.n_features_per_level_color_encoder,
        )

        if self.config.render_step_size is None:
            # auto step size: ~1000 samples in the base level grid
            self.config.render_step_size = (
                (self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2
            ).sum().sqrt().item() / 1000

        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid, density_fn=self.field.density_fn
        )

        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)

        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # Criterian in nerf2mesh
        self.rgb_loss = torch.nn.MSELoss(reduction="none")

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        def export_coarse_mesh(step: int):
            self.export_stage0(
                save_path=os.path.join(self.config.coarse_mesh_path),
                resolution=512,
                decimate_target=3e5,
                S=self.config.grid_resolution,
            )

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN],
                # update_every_num_iters=1,
                func=export_coarse_mesh,
            )
        )
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {"fields": list(self.field.parameters())}

    # # same as ngp model
    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor | List]:
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.min_near,
                far_plane=self.config.max_far,
                render_step_size=self.config.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )

        field_outputs = self.field(ray_samples, shading=self.config.shading_type)
        packed_info = nerfacc.pack_info(ray_indices, num_rays)

        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0]
        weights = weights[..., None]

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )

        if (specular := field_outputs.get("specular")) is not None:
            specular = self.renderer_rgb(
                rgb=specular,
                weights=weights,
                ray_indices=ray_indices,
                num_rays=num_rays,
            )

        depth = self.renderer_depth(
            weights=weights,
            ray_samples=ray_samples,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        accumulation = self.renderer_accumulation(
            weights=weights, ray_indices=ray_indices, num_rays=num_rays
        )

        # the diff between weights and accumuation
        # weights is individual weight of each point considered on each ray
        # accumulation is the sum of weights of all points considered on each ray
        # therefore, accumulation shape will be equal to number of rays

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],
            "specular_rgb": specular,
            "specular": field_outputs.get("specular"),
        }
        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(
                start_idx, end_idx
            )
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if output_name in ["num_samples_per_ray", "specular"]:
                continue
            try:
                outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
            except AttributeError:
                outputs[output_name] = torch.zeros((image_height, image_width, 3))  # type: ignore
        return outputs

    # almost same as ngp model until rgb_loss
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"][..., :3].to(self.device)
        # image = self.renderer_rgb.blend_background(image)

        pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        loss = self.config.lambda_rgb * self.rgb_loss(pred_rgb, image).mean(-1)

        if self.config.lambda_mask > 0 and batch["image"].size(-1) > 3:
            gt_mask = batch["image"][..., 3:].to(self.device)
            pred_mask = outputs["accumulation"]
            loss += self.config.lambda_mask * self.rgb_loss(
                pred_mask.squeeze(1), gt_mask.squeeze(1)
            )

        if (
            self.config.lambda_specular > 0
            and (specular := outputs.get("specular")) is not None
        ):
            loss += self.config.lambda_specular * (specular**2).sum(-1).mean()

        loss_dict = {"rgb_loss": loss.mean()}
        return loss_dict

    @torch.no_grad()
    def export_stage0(self, save_path, resolution=None, decimate_target=1e5, S=128):
        # only for the inner mesh inside [-1, 1]
        if resolution is None:
            resolution = self.config.grid_resolution

        # TODO: check correctness
        density_thresh = min(
            self.occupancy_grid.occs.mean(), self.config.density_thresh
        )

        # sigmas = np.zeros([resolution] * 3, dtype=np.float32)
        sigmas = torch.zeros([resolution] * 3, dtype=torch.float32, device=self.device)

        if resolution == self.config.grid_resolution:
            # re-map from morton code to regular coords...
            all_indices = torch.arange(
                resolution**3, device=self.device, dtype=torch.int
            )
            all_coords = raymarching.morton3D_invert(all_indices).long()
            sigmas[tuple(all_coords.T)] = self.density_grid[0]
        else:
            # query

            X = torch.linspace(-0.5, 0.5, resolution).split(S)
            Y = torch.linspace(-0.5, 0.5, resolution).split(S)
            Z = torch.linspace(-0.5, 0.5, resolution).split(S)

            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
                        pts = torch.cat(
                            [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                            dim=-1,
                        )  # [S, 3]
                        with torch.cuda.amp.autocast(enabled=self.config.fp16):
                            val = self.field.density_fn(pts.to(self.device))  # [S, 1]
                        sigmas[
                            xi * S : xi * S + len(xs),
                            yi * S : yi * S + len(ys),
                            zi * S : zi * S + len(zs),
                        ] = val.reshape(len(xs), len(ys), len(zs))

            # use the density_grid as a baseline mask (also excluding untrained regions)
            if not self.config.sdf:
                mask = torch.zeros(
                    [self.config.grid_resolution] * 3,
                    dtype=torch.float32,
                    device=self.device,
                )
                all_indices = torch.arange(
                    self.config.grid_resolution**3,
                    device=self.device,
                    dtype=torch.int,
                )
                all_coords = raymarching.morton3D_invert(all_indices).long()
                mask[tuple(all_coords.T)] = self.occupancy_grid.occs
                mask = (
                    F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0),
                        size=[resolution] * 3,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
                mask = mask > density_thresh
                sigmas = sigmas * mask

        sigmas = torch.nan_to_num(sigmas, 0)
        sigmas = sigmas.cpu().numpy()

        if self.config.sdf:
            vertices, triangles = mcubes.marching_cubes(-sigmas, 0)
        else:
            vertices, triangles = mcubes.marching_cubes(sigmas, density_thresh)

        vertices = vertices / (resolution - 0.5) * 2 - 0.5
        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.int32)

        ### visibility test.
        if self.mvps is not None:
            visibility_mask = (
                self.mark_unseen_triangles(
                    vertices, triangles, self.mvps, self.image_height, self.image_width
                )
                .cpu()
                .numpy()
            )
            vertices, triangles = remove_masked_trigs(
                vertices,
                triangles,
                visibility_mask,
                dilation=self.config.visibility_mask_dilation,
            )

        ### reduce floaters by post-processing...
        vertices, triangles = clean_mesh(
            vertices,
            triangles,
            min_f=self.config.clean_min_f,
            min_d=self.config.clean_min_d,
            repair=True,
            remesh=False,
        )

        ### decimation
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = decimate_mesh(
                vertices, triangles, decimate_target, remesh=False
            )

        mesh = trimesh.Trimesh(vertices, triangles, process=False)
        mesh.export(os.path.join(save_path))

        # for the outer mesh [1, inf]
        if self.bound > 1:
            if self.config.sdf:
                # assume background contracted in [-2, 2], process it specially
                sigmas = torch.zeros(
                    [resolution] * 3, dtype=torch.float32, device=self.device
                )
                for xi, xs in enumerate(X):
                    for yi, ys in enumerate(Y):
                        for zi, zs in enumerate(Z):
                            xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
                            pts = 2 * torch.cat(
                                [
                                    xx.reshape(-1, 1),
                                    yy.reshape(-1, 1),
                                    zz.reshape(-1, 1),
                                ],
                                dim=-1,
                            )  # [S, 3]
                            with torch.cuda.amp.autocast(enabled=self.config.fp16):
                                val = self.density(pts.to(self.device))  # [S, 1]
                            sigmas[
                                xi * S : xi * S + len(xs),
                                yi * S : yi * S + len(ys),
                                zi * S : zi * S + len(zs),
                            ] = val.reshape(len(xs), len(ys), len(zs))
                sigmas = torch.nan_to_num(sigmas, 0)
                sigmas = sigmas.cpu().numpy()

                vertices_out, triangles_out = mcubes.marching_cubes(-sigmas, 0)

                vertices_out = vertices_out / (resolution - 1.0) * 2 - 1
                vertices_out = vertices_out.astype(np.float32)
                triangles_out = triangles_out.astype(np.int32)

                _r = 0.5
                vertices_out, triangles_out = remove_selected_verts(
                    vertices_out,
                    triangles_out,
                    f"(x <= {_r}) && (x >= -{_r}) && (y <= {_r}) && (y >= -{_r}) && (z <= {_r} ) && (z >= -{_r})",
                )

                bound = 2
                half_grid_size = bound / resolution

                vertices_out = vertices_out * (bound - half_grid_size)

                # clean mesh
                vertices_out, triangles_out = clean_mesh(
                    vertices_out,
                    triangles_out,
                    min_f=self.config.clean_min_f,
                    min_d=self.config.clean_min_d,
                    repair=False,
                    remesh=False,
                )

                # decimate
                decimate_target *= 2
                if decimate_target > 0 and triangles_out.shape[0] > decimate_target:
                    vertices_out, triangles_out = decimate_mesh(
                        vertices_out,
                        triangles_out,
                        decimate_target,
                        optimalplacement=False,
                    )

                vertices_out = vertices_out.astype(np.float32)
                triangles_out = triangles_out.astype(np.int32)

                # warp back (uncontract)
                vertices_out = uncontract(vertices_out)

                # remove the out-of-AABB region
                xmn, ymn, zmn, xmx, ymx, zmx = self.aabb_train.cpu().numpy().tolist()
                vertices_out, triangles_out = remove_selected_verts(
                    vertices_out,
                    triangles_out,
                    f"(x <= {xmn}) || (x >= {xmx}) || (y <= {ymn}) || (y >= {ymx}) || (z <= {zmn} ) || (z >= {zmx})",
                )

                if self.mvps is not None:
                    visibility_mask = (
                        self.mark_unseen_triangles(
                            vertices_out,
                            triangles_out,
                            self.mvps,
                            self.image_height,
                            self.image_width,
                        )
                        .cpu()
                        .numpy()
                    )
                    vertices_out, triangles_out = remove_masked_trigs(
                        vertices_out,
                        triangles_out,
                        visibility_mask,
                        dilation=self.config.visibility_mask_dilation,
                    )

                print(
                    f"[INFO] exporting outer mesh at cas 1, v = {vertices_out.shape}, f = {triangles_out.shape}"
                )

                # vertices_out, triangles_out = clean_mesh(vertices_out, triangles_out, min_f=self.opt.clean_min_f, min_d=self.opt.clean_min_d, repair=False, remesh=False)
                mesh_out = trimesh.Trimesh(
                    vertices_out, triangles_out, process=False
                )  # important, process=True leads to seg fault...
                mesh_out.export(os.path.join(save_path, f"mesh_1.ply"))

            else:
                reso = self.config.grid_resolution
                target_reso = self.config.env_reso
                decimate_target //= 2  # empirical...

                all_indices = torch.arange(
                    reso**3, device=self.device, dtype=torch.int
                )
                all_coords = raymarching.morton3D_invert(all_indices).cpu().numpy()

                # for each cas >= 1
                for cas in range(1, self.cascade):
                    bound = min(2**cas, self.bound)
                    half_grid_size = bound / target_reso

                    # remap from density_grid
                    occ = torch.zeros(
                        [reso] * 3, dtype=torch.float32, device=self.device
                    )
                    occ[tuple(all_coords.T)] = self.density_grid[cas]

                    # remove the center (before mcubes)
                    # occ[reso // 4 : reso * 3 // 4, reso // 4 : reso * 3 // 4, reso // 4 : reso * 3 // 4] = 0

                    # interpolate the occ grid to desired resolution to control mesh size...
                    occ = (
                        F.interpolate(
                            occ.unsqueeze(0).unsqueeze(0),
                            [target_reso] * 3,
                            mode="trilinear",
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                    occ = torch.nan_to_num(occ, 0)
                    occ = (occ > density_thresh).cpu().numpy()

                    vertices_out, triangles_out = mcubes.marching_cubes(occ, 0.5)

                    vertices_out = (
                        vertices_out / (target_reso - 1.0) * 2 - 1
                    )  # range in [-1, 1]

                    # remove the center (already covered by previous cascades)
                    _r = 0.45
                    vertices_out, triangles_out = remove_selected_verts(
                        vertices_out,
                        triangles_out,
                        f"(x <= {_r}) && (x >= -{_r}) && (y <= {_r}) && (y >= -{_r}) && (z <= {_r} ) && (z >= -{_r})",
                    )
                    if vertices_out.shape[0] == 0:
                        continue

                    vertices_out = vertices_out * (bound - half_grid_size)

                    # remove the out-of-AABB region
                    xmn, ymn, zmn, xmx, ymx, zmx = (
                        self.aabb_train.cpu().numpy().tolist()
                    )
                    xmn += half_grid_size
                    ymn += half_grid_size
                    zmn += half_grid_size
                    xmx -= half_grid_size
                    ymx -= half_grid_size
                    zmx -= half_grid_size
                    vertices_out, triangles_out = remove_selected_verts(
                        vertices_out,
                        triangles_out,
                        f"(x <= {xmn}) || (x >= {xmx}) || (y <= {ymn}) || (y >= {ymx}) || (z <= {zmn} ) || (z >= {zmx})",
                    )

                    # clean mesh
                    vertices_out, triangles_out = clean_mesh(
                        vertices_out,
                        triangles_out,
                        min_f=self.config.clean_min_f,
                        min_d=self.config.clean_min_d,
                        repair=False,
                        remesh=False,
                    )

                    if vertices_out.shape[0] == 0:
                        continue

                    # decimate
                    if decimate_target > 0 and triangles_out.shape[0] > decimate_target:
                        vertices_out, triangles_out = decimate_mesh(
                            vertices_out,
                            triangles_out,
                            decimate_target,
                            optimalplacement=False,
                        )

                    vertices_out = vertices_out.astype(np.float32)
                    triangles_out = triangles_out.astype(np.int32)

                    print(
                        f"[INFO] exporting outer mesh at cas {cas}, v = {vertices_out.shape}, f = {triangles_out.shape}"
                    )

                    if self.mvps is not None:
                        visibility_mask = (
                            self.mark_unseen_triangles(
                                vertices_out,
                                triangles_out,
                                self.mvps,
                                self.image_height,
                                self.image_width
                            )
                            .cpu()
                            .numpy()
                        )
                        vertices_out, triangles_out = remove_masked_trigs(
                            vertices_out,
                            triangles_out,
                            visibility_mask,
                            dilation=self.config.visibility_mask_dilation,
                        )

                    # vertices_out, triangles_out = clean_mesh(vertices_out, triangles_out, min_f=self.opt.clean_min_f, min_d=self.opt.clean_min_d, repair=False, remesh=False)
                    mesh_out = trimesh.Trimesh(
                        vertices_out, triangles_out, process=False
                    )  # important, process=True leads to seg fault...
                    mesh_out.export(os.path.join(save_path, f"mesh_{cas}.ply"))

    @torch.no_grad()
    def mark_unseen_triangles(self, vertices, triangles, mvps, H, W):
        # vertices: coords in world system
        # mvps: [B, 4, 4]
        device = self.device

        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).contiguous().float().to(device)

        if isinstance(triangles, np.ndarray):
            triangles = torch.from_numpy(triangles).contiguous().int().to(device)

        mask = torch.zeros_like(triangles[:, 0])  # [M,], for face.

        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext(output_db=False)

        for mvp in tqdm.tqdm(mvps):
            vertices_clip = (
                torch.matmul(
                    F.pad(vertices, pad=(0, 1), mode="constant", value=1.0),
                    torch.transpose(mvp.to(device), 0, 1),
                )
                .float()
                .unsqueeze(0)
            )  # [1, N, 4]

            # ENHANCE: lower resolution since we don't need that high?
            rast, _ = dr.rasterize(
                self.glctx, vertices_clip, triangles, (H, W)
            )  # [1, H, W, 4]

            # collect the triangle_id (it is offseted by 1)
            trig_id = rast[..., -1].long().view(-1) - 1

            # no need to accumulate, just a 0/1 mask.
            mask[trig_id] += 1  # wrong for duplicated indices, but faster.
            # mask.index_put_((trig_id,), torch.ones(trig_id.shape[0], device=device, dtype=mask.dtype), accumulate=True)

        mask = mask == 0  # unseen faces by all cameras

        print(f"[mark unseen trigs] {mask.sum()} from {mask.shape[0]}")

        return mask  # [N]