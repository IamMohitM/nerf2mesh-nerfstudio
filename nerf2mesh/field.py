from typing import Dict, Optional, Tuple, Type
from enum import Enum

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.utils.rich_utils import CONSOLE

from torch import Tensor
from torch.nn import functional as F
import torch
from nerfstudio.data.scene_box import SceneBox
from nerf2mesh.utils import Shading
import nvdiffrast.torch.ops as dr
import numpy as np

import trimesh


def scale_img_nhwc(x: torch.tensor, size: Tuple, mag="bilinear", min="bilinear"):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (
        x.shape[1] < size[0] and x.shape[2] < size[1]
    ), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    if (
        x.shape[1] > size[0] and x.shape[2] > size[1]
    ):  # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else:  # Magnification
        if mag == "bilinear" or mag == "bicubic":
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC

def scale_img_hwc(x, size, mag="bilinear", min="bilinear"):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_hw(x, size, mag="bilinear", min="bilinear"):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]

class Nerf2MeshFieldHeadNames(Enum):
    RGB = "rgb"
    DENSITY = "density"
    SPECULAR = "specular"
    DEPTH = "depth"
    ACCUMULATION = "accumulation"

class Nerf2MeshField(Field):
    def __init__(
        self,
        aabb: torch.Tensor,
        num_layers_sigma: int,
        specular_dim: int,
        hidden_dim_sigma: int,
        hidden_dim_color: int,
        num_levels_sigma_encoder: int,
        num_levels_color_encoder: int,
        n_features_per_level_sigma_encoder: int,
        n_features_per_level_color_encoder: int,
        base_res: int,
        max_res: int,
        log2_hashmap_size: int,
        implementation: str = "tcnn",
    ) -> None:
        # super().__init__(aabb = aabb, base_res=base_res, max_res=max_res, log2_hashmap_size = log2_hashmap_size, **kwargs)
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.register_buffer("max_res", torch.tensor(max_res))

        self.encoder = HashEncoding(
            num_levels=num_levels_sigma_encoder,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=n_features_per_level_sigma_encoder,
            implementation=implementation,
            interpolation="linear",
        )

        self.sigma_net = MLP(
            in_dim=self.encoder.get_out_dim(),
            out_dim=1,
            num_layers=num_layers_sigma,
            layer_width=hidden_dim_sigma,
            activation=torch.nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )
        self.density_mlp = torch.nn.Sequential(self.encoder, self.sigma_net)

        self.encoder_color = HashEncoding(
            num_levels=num_levels_color_encoder,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=n_features_per_level_color_encoder,
            implementation=implementation,
            interpolation="linear",
        )

        self.color_net = MLP(
            in_dim=3 + self.encoder_color.get_out_dim(),
            num_layers=3,
            layer_width=hidden_dim_color,
            out_dim=3 + specular_dim,
            activation=torch.nn.ReLU(),
            implementation=implementation,
        )

        self.specular_net = MLP(
            in_dim=specular_dim + 3,
            num_layers=2,
            layer_width=32,
            out_dim=3,
            activation=torch.nn.ReLU(),
            implementation=implementation,
        )

    def forward(
        self,
        ray_samples: RaySamples,
        shading: Shading,
        compute_normals: bool = False,
    ) -> Dict[Nerf2MeshFieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        density, density_embedding = self.get_density(ray_samples)

        field_outputs = self.get_outputs(
            ray_samples, density_embedding=density_embedding, shading=shading
        )
        # field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        field_outputs[Nerf2MeshFieldHeadNames.DENSITY] = density  # type: ignore

        return field_outputs

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Optional[Tensor]]:
        positions = SceneBox.get_normalized_positions(
            ray_samples.frustums.get_positions(), self.aabb
        )
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        h = self.density_mlp(positions)
        # h = torch.cat([positions_flat, h], dim=-1)
        density = trunc_exp(h.to(positions))
        density = density * selector[..., None]
        return density, None

    def _get_diffuse_color(self, x, c=None):
        h = self.encoder_color(x)
        h = torch.cat([x, h], dim=-1)
        # TODO: check when
        if c is not None:
            h = torch.cat(
                [h, c.repeat(x.shape[0], 1) if c.shape[0] == 1 else c], dim=-1
            )
        h = self.color_net(h)
        return torch.sigmoid(h)

    def _get_specular_color(self, d, diffuse_feat):
        specular = self.specular_net(torch.cat([d, diffuse_feat[..., 3:]], dim=-1))
        return torch.sigmoid(specular)

    def get_outputs(
        self,
        ray_samples: RaySamples,
        shading: Shading,
        density_embedding: Tensor = None,
    ) -> Dict[Nerf2MeshFieldHeadNames, Tensor]:
        outputs = {}
        positions = SceneBox.get_normalized_positions(
            ray_samples.frustums.get_positions(), self.aabb
        )
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        diffuse_feat = self._get_diffuse_color(positions)
        diffuse = diffuse_feat[..., :3]
        if shading == Shading.diffuse:
            color = diffuse
        else:
            directions = get_normalized_directions(ray_samples.frustums.directions)
            specular_feat = self._get_specular_color(directions, diffuse_feat)
            if shading == Shading.specular:
                color = specular_feat
            else:
                color = (specular_feat + diffuse).clamp(0, 1)
            outputs[Nerf2MeshFieldHeadNames.SPECULAR] = specular_feat

        outputs[Nerf2MeshFieldHeadNames.RGB] = color
        return outputs
    


class Nerf2MeshFieldStage1(Field):

    def __init__(
            self,
            prev_field: Nerf2MeshField,
            grid_levels: int,
            mesh_path: str,
            super_sample: int = 2,
            pos_gradient_boost: int = 1,
            enable_offset_nerf_grad: bool = False,
    ):
        super().__init__()
        
        self.prev_field = prev_field
        self.mesh_path = mesh_path
        self.glctx = dr.RasterizeGLContext(output_db=False)
        self.super_sample = super_sample
        self.pos_gradient_boost = pos_gradient_boost
        self.grid_levels = grid_levels
        self.device = next(self.prev_field.parameters()).device
        self.enable_offset_nerf_grad = enable_offset_nerf_grad

        vertices = []
        triangles = []
        v_cum_sum = [0]
        f_cumsum = [0]

        # TODO: this will change - we now use a fixed coarse mesh path
        # However, when we have multiple grid levels, there will be a mesh for each grid
        # Therefore, this needs to be updated to load the correct mesh for each grid level
        # assumming one grid level
        for cas in range(self.grid_levels):
            mesh = trimesh.load(
                self.mesh_path,
                force="mesh",
                process=False,
                skip_materials=True,
            )
            vertices.append(mesh.vertices)
            triangles.append(mesh.faces + v_cum_sum[-1])
            v_cum_sum.append(v_cum_sum[-1] + len(mesh.vertices))
            f_cumsum.append(f_cumsum[-1] + len(mesh.faces))

        vertices = np.concatenate(vertices, axis=0)
        triangles = np.concatenate(triangles, axis=0)
        self.v_cum_sum = np.array(v_cum_sum)
        self.f_cumsum = np.array(f_cumsum)

        self.vertices = torch.from_numpy(vertices).float().to(self.device)
        self.triangles = torch.from_numpy(triangles).int().to(self.device)

        # these will be trained
        self.vertices_offsets = torch.nn.Parameter(
            torch.zeros_like(self.vertices, dtype=torch.float32)
        ).to(self.device)

        self.triangle_errors = torch.nn.Parameter(
            torch.zeros_like(self.triangles[:, 0], dtype=torch.float32)
        ).to(self.device)
        self.triangle_errors_cnt = torch.zeros_like(
            self.triangles[:, 0], dtype=torch.float32
        ).to(self.device)
        self.triangles_errors_id = None

        CONSOLE.print(
            f"Loaded coarse mesh: vertices - {self.vertices.shape}, triangles - {self.triangles.shape}"
        )

    #TODO: perhaps dont' add height and width to forward and maybe include in raysamples
    def forward(self, ray_samples: RaySamples, height, width, mvp, compute_normals: bool = False) -> Dict[Nerf2MeshFieldHeadNames, Tensor]:
        
        field_outputs = self.get_outputs(ray_samples, height, width, mvp)
        # field_outputs[FieldHeadNames.DENSITY] = density

        return field_outputs

    def get_outputs(
        self,
        ray_samples: RaySamples,
        height: int, #TODO: add these parameters
        width: int, #TODO: add these parameters,
        mvp: Tensor,
        # bg_color: str = 
    ) -> Dict[Nerf2MeshFieldHeadNames, Tensor]:
        directions = ray_samples.frustums.directions
        prefix = directions.shape[:1]

        if self.super_sample > 1:
            H = int(height * self.super_sample)
            W = int(width * self.super_sample)

            dirs = directions.view(height, width, 3)
            dirs = scale_img_hwc(dirs, (H, W), mag="nearest").view(-1, 3).contiguous()
        else:
            H, W = height, width
            dirs = directions.view(-1, 3).contiguous()

        dirs = get_normalized_directions(dirs) 

        
        results = {}

        vertices = self.vertices + self.vertices_offsets
        vertices_clip = torch.matmul(
            F.pad(vertices, (0, 1), "constant", 1.0),
            torch.transpose(mvp, 0, 1),
        ).float().unsqueeze(0)

        rast, _ = dr.rasterize(self.glctx, vertices_clip, self.triangles, (H, W))

        xyzs, _ = dr.interpolate(
            vertices.unsqueeze(0), rast, self.triangles
        )  # [1, H, W, 3]
        mask, _ = dr.interpolate(
            torch.ones_like(vertices[:, :1]).unsqueeze(0), rast, self.triangles
        )  # [1, H, W, 1]
        mask_flatten = (mask > 0).view(-1).detach()
        xyzs = xyzs.view(-1, 3)

        #TODO: add contraction support for unbounded scenes
        # if self.opt.contract:
        #     xyzs = contract(xyzs)

        rgbs = torch.zeros(H * W, 3, device=self.device, dtype=torch.float32)


        if mask_flatten.any():
            with torch.cuda.amp.autocast(enabled=True):
                diffuse_feat = self.prev_field._get_diffuse_color(xyzs[mask_flatten] if self.enable_offset_nerf_grad else xyzs[mask_flatten].detach())
                diffuse = diffuse_feat[..., :3]
                specular_feat = self.prev_field._get_specular_color(dirs[mask_flatten], diffuse_feat)
                color = (specular_feat + diffuse).clamp(0, 1)
                # outputs[Nerf2MeshFieldHeadNames.SPECULAR] = specular_feat
                mask_rgb = color
        
            rgbs[mask_flatten] = mask_rgb.float()

        rgbs = rgbs.view(1, H, W, 3)
        alphas = mask.float()

        alphas = (
            dr.antialias(
                alphas,
                rast,
                vertices_clip,
                self.triangles,
                pos_gradient_boost=self.pos_gradient_boost,
            )
            .squeeze(0)
            .clamp(0, 1)
        )
        rgbs = (
            dr.antialias(
                rgbs,
                rast,
                vertices_clip,
                self.triangles,
                pos_gradient_boost=self.pos_gradient_boost,
            )
            .squeeze(0)
            .clamp(0, 1)
        )

        image = alphas * rgbs
        depth = alphas * rast[0, :, :, [2]]
        T = 1 - alphas

        # trig_id for updating trig errors - triangle_id is offseted by 1 in nvdiffrast
        trig_id = rast[0, :, :, -1] - 1  # [h, w]

        # ssaa
        if self.super_sample > 1:
            image = scale_img_hwc(image, (height, width))
            depth = scale_img_hwc(depth, (height, width))
            T = scale_img_hwc(T, (height, width))
            trig_id = scale_img_hw(
                trig_id.float(), (height, width), mag="nearest", min="nearest"
            )

        self.triangles_errors_id = trig_id

        #TODO: check if bg color is needed
        # image = image + T * bg_color

        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)

        results[Nerf2MeshFieldHeadNames.DEPTH] = depth
        results[Nerf2MeshFieldHeadNames.RGB] = image
        results[Nerf2MeshFieldHeadNames.ACCUMULATION] = 1 - T

        return results

    def get_param_groups(self):
        return [
            {
                "params": [self.vertices_offsets],
                "lr": 1e-4,
                "weight_decay": 0,
            },
        ]