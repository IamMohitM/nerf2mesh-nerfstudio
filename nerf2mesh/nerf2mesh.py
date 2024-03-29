import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Type, Optional, Literal, Union
import tqdm
import trimesh
import mcubes
import numpy as np
import nerfacc
import nvdiffrast.torch as dr
import trimesh
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import math
import torch_scatter as TORCH_SCATTER

from nerfstudio.data.scene_box import SceneBox
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.instant_ngp import NGPModel, InstantNGPModelConfig
from nerfstudio.model_components.ray_samplers import VolumetricSampler

from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_normal_consistency, mesh_edge_loss
from nerfstudio.model_components.renderers import (
    RGBRenderer,
    AccumulationRenderer,
    DepthRenderer,
)
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerf2mesh.field import (
    Nerf2MeshField,
    Nerf2MeshFieldHeadNames,
    Nerf2MeshFieldStage1,
)
from nerf2mesh.utils import (
    Shading,
    clean_mesh,
    remove_masked_trigs,
    remove_selected_verts,
    decimate_mesh,
    laplacian_smooth_loss,
    contract,
    decimate_and_refine_mesh
)
from nerf2mesh.sampler import MetaDataUniformSampler

import cv2
import json
from scipy.ndimage import binary_dilation, binary_erosion
import xatlas
from sklearn.neighbors import NearestNeighbors


@dataclass
class Nerf2MeshModelConfig(InstantNGPModelConfig):
    _target: Type = field(default_factory=lambda: Nerf2MeshModel)

    # occupancy grid
    # grid_levels: int = 1
    density_thresh: float = 10
    alpha_thre: float = 0.0
    """Threshold for opacity skipping."""
    cone_angle: float = 0.0  # 04
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: Optional[float] = None
    """Minimum step size for rendering."""
    bound: float = 1
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
    mvps: torch.Tensor = None
    mark_unseen_triangles: bool = True
    contract: bool = False

    # NOTE: all default configs are passed to nerf2mesh field
    # NOTE: all dervied are computed in nerf2mesh field

    # STAGE 1
    stage: int = 0
    enable_offset_nerf_grad: bool = False
    lambda_normal: float = 0#0.001
    lambda_edgelen: float = 0#0.1
    lambda_lap: float = 0.001
    lambda_offsets: float = 0.1
    fine_mesh_path: str = "meshes/stage_1"
    texture_size: float = 4096
    # stage 1 fine mesh refine parameters
    refine: bool = True
    refine_steps: int = 1000
    refine_remesh_size: float = 0.01
    refine_decimate_ratio: float = 0.1
    refine_size: float  = 0.01

class Nerf2MeshModel(NGPModel):
    config: Nerf2MeshModelConfig
    field: Nerf2MeshField

    def populate_modules(self):
        self.all_mvps = None
        self.train_mvp = None
        self.eval_mvp = None
        self.image_height = None
        self.image_width = None
        self.glctx = None
        self.cx = None
        self.cy = None
        # super().populate_modules()
        scene_box = SceneBox(aabb=torch.tensor([ [-self.config.bound] * 3,[self.config.bound] * 3], dtype=torch.float32))
        
        
        self.scene_aabb  = torch.nn.Parameter(
            scene_box.aabb.flatten(), requires_grad=False
        )
        self.bound = self.config.bound
        # self.register_buffer("bound", self.config.bound)

        self.field = Nerf2MeshField(
            aabb=self.scene_box.aabb,
            base_res=self.config.base_resolution,
            max_res=self.config.desired_resolution,
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

        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)

        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # Criterian in nerf2mesh
        self.rgb_loss = torch.nn.MSELoss(reduction="none")

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        self.grid_levels = 1 + math.ceil(math.log2(self.bound))

        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.grid_levels,
        )
        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid, density_fn=self.field.density_fn
        )


    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        def export_coarse_mesh(step: int):
            self.export_stage0(
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
        params = {"fields": list(self.field.parameters())}

        return params

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
            sigmas=field_outputs[Nerf2MeshFieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0]
        weights = weights[..., None]

        rgb = self.renderer_rgb(
            rgb=field_outputs[Nerf2MeshFieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )

        if (
            specular := field_outputs.get(Nerf2MeshFieldHeadNames.SPECULAR)
        ) is not None:
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

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],
            "specular_rgb": specular,
            "specular": field_outputs.get(Nerf2MeshFieldHeadNames.SPECULAR),
        }
        # the diff between weights and accumuation
        # weights is individual weight of each point considered on each ray
        # accumulation is the sum of weights of all points considered on each ray
        # therefore, accumulation shape will be equal to number of rays

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
            if (
                output_name in ["num_samples_per_ray", "specular"]
                or outputs_list is None
            ):
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

        loss = self.config.lambda_rgb * self.rgb_loss(
            pred_rgb.view(-1, 3), image.view(-1, 3)
        ).mean(-1)

        if self.config.lambda_mask > 0 and batch["image"].size(-1) > 3:
            
            gt_mask = batch["image"][..., 3:].to(self.device)
            pred_mask = outputs["accumulation"]
            loss += self.config.lambda_mask * self.rgb_loss(
                pred_mask.view(-1), gt_mask.view(-1)
            )

        if (
            self.config.lambda_specular > 0
            and (specular := outputs.get("specular")) is not None
        ):
            loss += self.config.lambda_specular * (specular**2).sum(-1).mean()

        loss_dict = {"rgb_loss": loss.mean()}

        return loss_dict

    @torch.no_grad()
    def export_stage0(self, resolution: int =None, decimate_target: int =1e5, S: int =128):
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
            sigmas = self.occupancy_grid.occs[:self.occupancy_grid.cells_per_lvl].view_as(sigmas)
        else:
            # query

            X = torch.linspace(-1, 1, resolution).split(S)
            Y = torch.linspace(-1, 1, resolution).split(S)
            Z = torch.linspace(-1, 1, resolution).split(S)

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

        mask = torch.zeros(
                    [self.config.grid_resolution] * 3,
                    dtype=torch.float32,
                    device=self.device,
                )
        mask = self.occupancy_grid.occs[:self.occupancy_grid.cells_per_lvl].view_as(mask)
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

        vertices, triangles = mcubes.marching_cubes(sigmas, density_thresh)

        vertices = vertices / (resolution - 1) * 2 - 1
        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.int32)

        ### visibility test.
        if self.config.mark_unseen_triangles and self.train_mvp is not None:
            visibility_mask = (
                self.mark_unseen_triangles(
                    vertices,
                    triangles,
                    self.train_mvp,
                    self.image_height,
                    self.image_width,
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
        os.makedirs(os.path.dirname(self.config.coarse_mesh_path), exist_ok=True)
        mesh.export(os.path.join(self.config.coarse_mesh_path))

        # for the outer mesh [1, inf]
        if self.bound > 1:

            reso = self.config.grid_resolution
            target_reso = self.config.env_reso
            decimate_target //= 2  # empirical...

            all_indices = torch.arange(
                reso**3, device=self.device, dtype=torch.int
            )

            # for each cas >= 1
            for cas in range(1, self.grid_levels):
                bound = min(2**cas, self.bound)
                half_grid_size = bound / target_reso

                # remap from density_grid
                occ = torch.zeros(
                    [reso] * 3, dtype=torch.float32, device=self.device
                )
                start_index = cas * self.occupancy_grid.cells_per_lvl
                end_index = (cas+1) * self.occupancy_grid.cells_per_lvl
                occ = self.occupancy_grid.occs[start_index:end_index].view_as(occ)

            # remove the center (before mcubes)
                # occ[reso // 4 : reso * 3 // 4, reso // 4 : r  eso * 3 // 4, reso // 4 : reso * 3 // 4] = 0

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
                    self.scene_aabb.cpu().numpy().tolist()
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

            if self.all_mvps is not None:
                visibility_mask = (
                    self.mark_unseen_triangles(
                        vertices_out,
                        triangles_out,
                        self.all_mvps,
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

                # vertices_out, triangles_out = clean_mesh(vertices_out, triangles_out, min_f=self.opt.clean_min_f, min_d=self.opt.clean_min_d, repair=False, remesh=False)
                mesh_out = trimesh.Trimesh(
                    vertices_out, triangles_out, process=False
                )  # important, process=True leads to seg fault...
                mesh_out.export(os.path.join(os.path.dirname(self.config.coarse_mesh_path), f"mesh_{cas}.ply"))

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


@dataclass
class Nerf2MeshStage1ModelConfig(Nerf2MeshModelConfig):
    _target: Type = field(default_factory=lambda: Nerf2MeshStage1Model)
    stage: int = 1
    

class Nerf2MeshStage1Model(NGPModel):
    config: Nerf2MeshStage1ModelConfig

    def populate_modules(self):
        self.all_mvps = None
        self.train_mvp = None
        self.eval_mvp = None
        self.image_height = None
        self.image_width = None
        self.glctx = None
        self.cx = None
        self.cy = None
        # super().populate_modules()
        self.scene_aabb = torch.nn.Parameter(
            self.scene_box.aabb.flatten(), requires_grad=False
        )

        self.register_buffer("bound", self.scene_box.aabb.max())
        self.grid_levels = 1 + math.ceil(math.log2(self.bound))

        self.prev_field = Nerf2MeshField(
            aabb=self.scene_box.aabb,
            base_res=self.config.base_resolution,
            max_res=self.config.desired_resolution,
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

        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)

        self.field = Nerf2MeshFieldStage1(
                prev_field=self.prev_field,
                grid_levels=self.grid_levels,
                mesh_path=os.path.dirname(self.config.coarse_mesh_path),
                train_mvps = self.train_mvp,
                # eval_mvps = self.eval,

            )
        self.sampler = MetaDataUniformSampler()
        
        # Criterian in nerf2mesh
        self.rgb_loss = torch.nn.MSELoss(reduction="none")

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)


    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        
        callbacks = []
        def export_fine_mesh(step: int):
            self.export_stage1(
                path=self.config.fine_mesh_path,
                h0=self.config.texture_size,
                w0=self.config.texture_size,
            )

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN],
                func=export_fine_mesh,
            )
        )

        if self.config.refine:
            self.refine_flag = False
            def refine_mesh(step: int):
                if not self.refine_flag:
                    self.refine_flag = True
                    return
                self.refine_and_decimate()

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    func=refine_mesh,
                    update_every_num_iters=self.config.refine_steps
                )
            )

        return callbacks
    
     # same as ngp model
    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor | List]:
                
        self.current_image = ray_bundle.camera_indices[0].squeeze(0)
        with torch.no_grad():
            ray_bundle.nears = self.config.min_near
            ray_bundle.fars = self.config.max_far
            ray_samples = self.sampler(ray_bundle=ray_bundle, num_samples=1)
        field_outputs = self.field(
            ray_samples=ray_samples,
        )

        outputs = {
            "rgb": field_outputs[Nerf2MeshFieldHeadNames.RGB].view(
                self.image_height, self.image_width, -1
            ),
            "accumulation": field_outputs[Nerf2MeshFieldHeadNames.ACCUMULATION],
        }

        return outputs
    
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"][..., :3].to(self.device)
        # image = self.renderer_rgb.blend_background(image)

        pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        loss = self.config.lambda_rgb * self.rgb_loss(
                pred_rgb.view(-1, 3), image.view(-1, 3)
            ).mean(-1)

        if self.config.lambda_mask > 0 and batch["image"].size(-1) > 3:
            gt_mask = batch["image"][..., 3:].to(self.device)
            pred_mask = outputs["accumulation"]
            loss += self.config.lambda_mask * self.rgb_loss(
                pred_mask.view(-1), gt_mask.view(-1)
            )

        if (
            self.config.lambda_specular > 0
            and (specular := outputs.get("specular")) is not None
        ):
            loss += self.config.lambda_specular * (specular**2).sum(-1).mean()

        if self.config.lambda_lap > 0:
            loss_lap = laplacian_smooth_loss(
                self.field.vertices + self.field.vertices_offsets,
                self.field.triangles,
            )
            loss = loss + self.config.lambda_lap * loss_lap
        
        _mesh = None

        if self.config.lambda_normal > 0:
                if _mesh is None:
                    _mesh = Meshes(verts=[self.field.vertices + self.field.vertices_offsets], faces=[self.field.triangles])
                loss_normal = mesh_normal_consistency(_mesh)
                loss = loss + self.config.lambda_normal * loss_normal
            
        if self.config.lambda_edgelen > 0:
            if _mesh is None:
                _mesh = Meshes(verts=[self.field.vertices + self.field.vertices_offsets], faces=[self.field.triangles])
            loss_edge = mesh_edge_loss(_mesh)
            loss = loss + self.config.lambda_edgelen * loss_edge


        if self.config.lambda_offsets > 0:
            if self.bound > 1:
                abs_offsets_inner = self.field.vertices_offsets[
                    : self.field.v_cumsum[1]
                ].abs()
                abs_offsets_outer = self.field.vertices_offsets[
                    self.field.v_cumsum[1] :
                ].abs()
            else:
                abs_offsets_inner = self.field.vertices_offsets.abs()
            loss_offsets = (abs_offsets_inner**2).sum(-1).mean()

            if self.bound > 1:
                # loss_offsets = loss_offsets + torch.where(abs_offsets_outer > 0.02, abs_offsets_outer * 100, abs_offsets_outer * 0.01).sum(-1).mean()
                loss_offsets = (
                    loss_offsets + 0.1 * (abs_offsets_outer**2).sum(-1).mean()
                )

            loss = loss + self.config.lambda_offsets * loss_offsets

        if self.config.refine:
            self.update_triangles_errors(loss.detach())

        loss_dict = {"rgb_loss": loss.mean()}

        return loss_dict
    
    def get_metrics_dict(self, outputs, batch):
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image[..., :3]).item()
        t: torch.Tensor = outputs['rgb'].detach().cpu()
        im = image[0].detach().cpu()
        # metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict

    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        params = {"fields": list(self.prev_field.parameters())}
        # TODO: add correct implementation
        params["vertices_offsets"] = [self.field.vertices_offsets]

        return params
    
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        return {}
    


    @torch.no_grad()
    def export_stage1(self, path, h0=2048, w0=2048, png_compression_level=3):
        # png_compression_level: 0 is no compression, 9 is max (default will be 3)

        device = self.field.vertices.device
        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext(output_db=False)

        def _export_obj(v, f, h0, w0, ssaa=1, cas=0):
            # v, f: torch Tensor

            v_np = v.cpu().numpy()  # [N, 3]
            f_np = f.cpu().numpy()  # [M, 3]

            print(
                f"[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}"
            )

            # unwrap uv in contracted space
            atlas = xatlas.Atlas()
            atlas.add_mesh(contract(v_np) if self.config.contract else v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 0  # disable merge_chart for faster unwrap...
            pack_options = xatlas.PackOptions()
            # pack_options.blockAlign = True
            # pack_options.bruteForce = False
            atlas.generate(chart_options=chart_options, pack_options=pack_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

            # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

            # render uv maps
            uv = vt * 2.0 - 1.0  # uvs to range [-1, 1]
            uv = torch.cat(
                (uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])),
                dim=-1,
            )  # [N, 4]

            if ssaa > 1:
                h = int(h0 * ssaa)
                w = int(w0 * ssaa)
            else:
                h, w = h0, w0

            rast, _ = dr.rasterize(
                self.glctx, uv.unsqueeze(0), ft, (h, w)
            )  # [1, h, w, 4]
            xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f)  # [1, h, w, 3]
            mask, _ = dr.interpolate(
                torch.ones_like(v[:, :1]).unsqueeze(0), rast, f
            )  # [1, h, w, 1]

            # masked query
            xyzs = xyzs.view(-1, 3)
            mask = (mask > 0).view(-1)

            # TODO: add contract support
            if self.config.contract:
                xyzs = contract(xyzs)

            feats = torch.zeros(h * w, 6, device=device, dtype=torch.float32)

            if mask.any():
                xyzs = xyzs[mask]  # [M, 3]

                # check individual codes
                # if self.config.individual_dim > 0:
                #     ind_code = self.individual_codes[[0]]
                # else:
                #     ind_code = None
                ind_code = None

                # batched inference to avoid OOM
                all_feats = []
                head = 0
                while head < xyzs.shape[0]:
                    tail = min(head + 640000, xyzs.shape[0])
                    with torch.cuda.amp.autocast(enabled=True):
                        all_feats.append(
                            self.prev_field._get_diffuse_color(
                                xyzs[head:tail], ind_code
                            ).float()
                        )
                    head += 640000

                feats[mask] = torch.cat(all_feats, dim=0)

            feats = feats.view(h, w, -1)  # 6 channels
            mask = mask.view(h, w)

            # quantize [0.0, 1.0] to [0, 255]
            feats = feats.detach().cpu().numpy()
            feats = (feats * 255).astype(np.uint8)

            ### NN search as a queer antialiasing ...
            mask = mask.cpu().numpy()

            inpaint_region = binary_dilation(mask, iterations=32)  # pad width
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            feats[tuple(inpaint_coords.T)] = feats[
                tuple(search_coords[indices[:, 0]].T)
            ]

            # do ssaa after the NN search, in numpy
            feats0 = cv2.cvtColor(feats[..., :3], cv2.COLOR_RGB2BGR)  # albedo
            feats1 = cv2.cvtColor(
                feats[..., 3:], cv2.COLOR_RGB2BGR
            )  # visibility features

            if ssaa > 1:
                feats0 = cv2.resize(feats0, (w0, h0), interpolation=cv2.INTER_LINEAR)
                feats1 = cv2.resize(feats1, (w0, h0), interpolation=cv2.INTER_LINEAR)

            os.makedirs(path, exist_ok=True)
            cv2.imwrite(os.path.join(path, f"feat0_{cas}.jpg"), feats0)
            cv2.imwrite(os.path.join(path, f"feat1_{cas}.jpg"), feats1)

            # save obj (v, vt, f /)
            obj_file = os.path.join(path, f"mesh_{cas}.obj")
            mtl_file = os.path.join(path, f"mesh_{cas}.mtl")

            print(f"[INFO] writing obj mesh to {obj_file}")
            with open(obj_file, "w") as fp:
                fp.write(f"mtllib mesh_{cas}.mtl \n")

                print(f"[INFO] writing vertices {v_np.shape}")
                for v in v_np:
                    fp.write(f"v {v[0]} {v[1]} {v[2]} \n")

                print(f"[INFO] writing vertices texture coords {vt_np.shape}")
                for v in vt_np:
                    fp.write(f"vt {v[0]} {1 - v[1]} \n")

                print(f"[INFO] writing faces {f_np.shape}")
                fp.write(f"usemtl defaultMat \n")
                for i in range(len(f_np)):
                    fp.write(
                        f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n"
                    )

            with open(mtl_file, "w") as fp:
                fp.write(f"newmtl defaultMat \n")
                fp.write(f"Ka 1 1 1 \n")
                fp.write(f"Kd 1 1 1 \n")
                fp.write(f"Ks 0 0 0 \n")
                fp.write(f"Tr 1 \n")
                fp.write(f"illum 1 \n")
                fp.write(f"Ns 0 \n")
                fp.write(f"map_Kd feat0_{cas}.jpg \n")

        v = (self.field.vertices + self.field.vertices_offsets).detach()
        f = self.field.triangles.detach()

        for cas in range(self.grid_levels):
            cur_v = v[self.field.v_cumsum[cas] : self.field.v_cumsum[cas + 1]]
            cur_f = (
                f[self.field.f_cumsum[cas] : self.field.f_cumsum[cas + 1]]
                - self.field.v_cumsum[cas]
            )
            _export_obj(cur_v, cur_f, h0, w0, self.field.super_sample, cas)

        # save mlp as json
        params = dict(self.prev_field.specular_net.named_parameters())

        mlp = {}
        for k, p in params.items():
            p_np = p.detach().cpu().numpy().T
            print(f"[INFO] wrting MLP param {k}: {p_np.shape}")
            mlp[k] = p_np.tolist()

        mlp["bound"] = self.bound.item()
        mlp["cascade"] = self.grid_levels

        mlp_file = os.path.join(path, f"mlp.json")
        with open(mlp_file, "w") as fp:
            json.dump(mlp, fp, indent=2)

    @torch.no_grad()
    def refine_and_decimate(self):
        device = self.field.vertices.device

        v = (self.field.vertices + self.field.vertices_offsets).detach().cpu().numpy()
        f = self.field.triangles.detach().cpu().numpy()

        errors = self.field.triangle_errors.cpu().numpy()

        cnt = self.field.triangle_errors_cnt.cpu().numpy()
        cnt_mask = cnt > 0
        errors[cnt_mask] = errors[cnt_mask] / cnt[cnt_mask]

        # only care about the inner mesh
        errors = errors[: self.field.f_cumsum[1]]
        cnt_mask = cnt_mask[: self.field.f_cumsum[1]]

        # find a threshold to decide whether we perform subdivision / decimation.
        thresh_refine = np.percentile(errors[cnt_mask], 90)
        thresh_decimate = np.percentile(errors[cnt_mask], 50)

        mask = np.zeros_like(errors)
        mask[(errors > thresh_refine) & cnt_mask] = 2
        mask[(errors < thresh_decimate) & cnt_mask] = 1

        print(
            f"[INFO] faces to decimate {(mask == 1).sum()}, faces to refine {(mask == 2).sum()}"
        )

        if self.bound <= 1:
            # mesh = trimesh.Trimesh(v, f, process=False)
            # mesh.export(os.path.join(self.opt.workspace, 'mesh_stage0', 'mesh_0_before_updated.ply'))

            v, f = decimate_and_refine_mesh(
                v,
                f,
                mask,
                decimate_ratio=self.config.refine_decimate_ratio,
                refine_size=self.config.refine_size,
                refine_remesh_size=self.config.refine_remesh_size,
            )
            # export
            mesh = trimesh.Trimesh(v, f, process=False)
            mesh.export(
                os.path.join(self.config.fine_mesh_path, "mesh_0_updated.ply")
            )
            v, f = mesh.vertices, mesh.faces

            # fix counters
            self.field.v_cumsum[1:] += v.shape[0] - self.field.v_cumsum[1]
            self.field.f_cumsum[1:] += f.shape[0] - self.field.f_cumsum[1]
        else:
            vertices = []
            triangles = []
            v_cumsum = [0]
            f_cumsum = [0]
            
            for cas in range(self.grid_levels):
                cur_v = v[self.field.v_cumsum[cas] : self.field.v_cumsum[cas + 1]]
                cur_f = (
                    f[self.field.f_cumsum[cas] : self.field.f_cumsum[cas + 1]] - self.field.v_cumsum[cas]
                )

                if cas == 0:
                    cur_v, cur_f = decimate_and_refine_mesh(
                        cur_v,
                        cur_f,
                        mask,
                        decimate_ratio=self.config.refine_decimate_ratio,
                        refine_size=self.config.refine_size,
                        refine_remesh_size=self.config.refine_remesh_size,
                    )

                mesh = trimesh.Trimesh(cur_v, cur_f, process=False)
                mesh.export(
                    os.path.join(
                        self.config.fine_mesh_path, f"mesh_{cas}_updated.ply"
                    )
                )

                vertices.append(mesh.vertices)
                triangles.append(mesh.faces + v_cumsum[-1])

                v_cumsum.append(v_cumsum[-1] + mesh.vertices.shape[0])
                f_cumsum.append(f_cumsum[-1] + mesh.faces.shape[0])

            v = np.concatenate(vertices, axis=0)
            f = np.concatenate(triangles, axis=0)
            self.field.v_cumsum = np.array(v_cumsum)
            self.field.f_cumsum = np.array(f_cumsum)

        self.field.vertices = torch.from_numpy(v).float().contiguous().to(device)  # [N, 3]
        self.field.triangles = torch.from_numpy(f).int().contiguous().to(device)
        self.field.vertices_offsets = torch.nn.Parameter(torch.zeros_like(self.field.vertices))

        self.field.triangle_errors = torch.zeros_like(
            self.field.triangles[:, 0], dtype=torch.float32
        )
        self.field.triangle_errors_cnt = torch.zeros_like(
            self.field.triangles[:, 0], dtype=torch.float32
        )

        CONSOLE.print(
            f"[INFO] update stage1 mesh: {self.field.vertices.shape}, {self.field.triangles.shape}"
        )

    @torch.no_grad()
    def update_triangles_errors(self, loss):
        # loss: [H, W], detached!

        # always call after render_stage1, so self.triangles_errors_id is not None.
        indices = self.field.triangles_errors_id.view(-1).long()
        mask = indices >= 0

        indices = indices[mask].contiguous()
        values = loss.view(-1)[mask].contiguous()

        TORCH_SCATTER.scatter_add(values, indices, out=self.field.triangle_errors)
        TORCH_SCATTER.scatter_add(
            torch.ones_like(values), indices, out=self.field.triangle_errors_cnt
        )

        self.field.triangles_errors_id = None
        