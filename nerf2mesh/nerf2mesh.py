import math
from nerfstudio.cameras.rays import RayBundle

import torch

from dataclasses import dataclass, field
from typing import Dict, List, Type, Optional, Tuple, Literal

from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.base_model import Model, ModelConfig
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
from torch.nn import Parameter
from nerf2mesh.field import Nerf2MeshField
import nerfacc


@dataclass
class Nerf2MeshModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: Nerf2MeshModel)

    real_bound: float = 1.0

    contract: bool = False
    collider_params: Optional[Dict[str, float]] = None

    # bound: float = 1.0 #TODO derived

    # cascade: float TODO: add cascade in model def derived
    hidden_dim_sigma: int = 32
    hidden_dim_color: int = 64
    # TODO: check if grid_resolution and desired_resolution are the same
    # understand each parameter related to grid
    # occupancy grid
    grid_resolution: int = 128  # same as grid resolution or grid size
    grid_levels: int = 4  # differnet resolution levels for occupancy grid - this is for efficiency (paper appendix) not the encodng levels

    num_levels_sigma_encoder: int = 16
    num_levels_color_encoder: int = 16
    n_features_per_level_sigma_encoder: int = 2
    n_features_per_level_color_encoder: int = 2
    base_resolution: int = 16  # starting resolution
    desired_resolution: int = 2048  # ending resolution

    ## -- Sammpling Parameters
    min_near: float = 0.05  # same as opt.min_near (nerf2mesh)
    min_far: float = 1000  # infinite - same as opt.min_far (nerf2mesh)

    # following parameters copied from instant-ngp nerfstudio
    alpha_thre: float = 0.01
    """Threshold for opacity skipping."""
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: Optional[float] = None
    """Minimum step size for rendering."""

    ## ----------------------

    density_threshold: int = 10

    # aabb_train: torch.Tensor # TODO: derived
    # aabb_infer: torch.Tensor # TODO: derived

    individual_num: int = 500
    individual_dim: int = 0
    specular_dim: int = 3  # fs input features for specular net

    # individual_codes # TODO: derived

    cuda_ray: bool = True

    trainable_density_grid: bool = True
    log2hash_map_size: int = 19

    # density_grid #TODO: derived

    # density_bitfield #TODO: derived

    mean_density: float = 0.0
    iter_density: float = 0.0

    sigma_layers = 2

    background_color: Literal["random", "black", "white"] = "random"

    lambda_mask: float = 0.1
    lambda_rgb: float = 1.0
    lambda_depth: float = 0.1

    # TODO: register_buffer

    # NOTE: all default configs are passed to nerf2mesh field
    # NOTE: all dervied are computed in nerf2mesh field

    # TODO: config for stage 1 if needed


class Nerf2MeshModel(Model):
    config: Nerf2MeshModelConfig
    field: Nerf2MeshField

    # encoder
    # fields
    # renderer - NerfRender
    # sampler

    # #Raybundlers > generate samples  --> field

    # NOTE: to Model I must pass a scenebox and a config

    def __init__(
        self,
        config: Nerf2MeshModelConfig,
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        self.scene_aabb = torch.nn.Parameter(
            self.scene_box.aabb.flatten(), requires_grad=False
        )

        # TODO: add remaining parameters
        self.field = Nerf2MeshField(
            aabb=self.scene_box.aabb,
            num_layers_sigma=self.config.sigma_layers,
            specular_dim=self.config.specular_dim,
            hidden_dim_sigma=self.config.hidden_dim_sigma,
            hidden_dim_color=self.config.hidden_dim_color,
            num_levels_sigma_encoder=self.config.num_levels_sigma_encoder,
            num_levels_color_encoder=self.config.num_levels_color_encoder,
            n_features_per_level_sigma_encoder=self.config.n_features_per_level_sigma_encoder,
            n_features_per_level_color_encoder=self.config.n_features_per_level_color_encoder,
            num_levels=self.config.grid_levels,
            base_res=self.config.base_resolution,
            max_res=self.config.grid_resolution,
            log2_hashmap_size=self.config.log2hash_map_size,
        )

        if self.config.render_step_size is None:
            # auto step size: ~1000 samples in the base level grid
            self.config.render_step_size = (
                (self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2
            ).sum().sqrt().item() / 1000

        # TODO: check if occgrid is used in nerf2mesh
        # TODO: check if occgrid is same as self.opt.trainable_density_grid in nerf2mesh
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
        self.loss = torch.nn.MSELoss(reduction="none")

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_occupancy_grid(step: int):
            self.occupancy_grid.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.field.density_fn(x)
                * self.config.render_step_size,
            )

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
        ]

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        params = []

        # if self.individual_codes is not None:
        #     params.append(
        #         {
        #             "params": self.individual_codes,
        #             "lr": self.opt.lr * 0.1,
        #             "weight_decay": 0,
        #         }
        #     )

        # if self.opt.trainable_density_grid:
        #     params.append(
        #         {"params": self.density_grid, "lr": self.opt.lr, "weight_decay": 0}
        #     )

        # if self.glctx is not None:
        #     params.append(
        #         {
        #             "params": self.vertices_offsets,
        #             "lr": self.opt.lr_vert,
        #             "weight_decay": 0,
        #         }
        #     )

        lr = 0.01

        # TODO: add other param groups from NerfRenderer

        return {"fields": list(self.field.parameters())}

        params.extend(
            [
                {"params": self.field.encoder.parameters(), "lr": lr},
                {"params": self.field.encoder_color.parameters(), "lr": lr},
                {"params": self.field.sigma_net.parameters(), "lr": lr},
                {"params": self.field.color_net.parameters(), "lr": lr},
                {"params": self.field.specular_net.parameters(), "lr": lr},
            ]
        )

        # if self.opt.sdf:
        #     params.append({'params': self.variance, 'lr': lr * 0.1})

        return params

        ...

    # same as ngp model
    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor | List]:
        assert self.field is not None
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.min_near,
                far_plane=self.config.min_far,
                render_step_size=self.config.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )

        field_outputs = self.field(ray_samples)

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
        }
        return outputs

        # composite_rays_train in nerf2mesh returns weights, weights_sum, depth, image
        # weights, rgb, depth, accumulation is computed in NGPmodel get_outputs. Are they the same?
        # computing image is ame

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        # print(outputs)
        metrics_dict = {}
        # copied from instant ngp
        # image = batch["image"].to(self.device)
        # image = self.renderer_rgb.blend_background(image)
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        # TODO: add psnr
        # metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict

    # almost same as ngp model until rgb_loss
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch['image'][..., :3].to(self.device)
        image = self.renderer_rgb.blend_background(image)

        pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        loss = self.loss(pred_rgb, image).mean(-1)

        if self.config.lambda_mask > 0 and batch['image'].size(-1) > 3:
            gt_mask = batch['image'][..., 3:].to(self.device)
            pred_mask = outputs['accumulation']
            loss = loss + self.config.lambda_mask * self.loss(pred_mask.squeeze(1), gt_mask.squeeze(1))

        # TODO: add different losses - lambda_mask, lambda_rgb
        loss_dict = {"rgb_loss": loss.mean()}
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        ...
        """Returns a dictionary of images and metrics to plot. Here you can apply your colormaps."""
