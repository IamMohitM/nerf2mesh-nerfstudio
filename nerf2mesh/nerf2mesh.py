import math
from nerfstudio.cameras.rays import RayBundle

import torch

from dataclasses import dataclass, field
from typing import Dict, List, Type, Optional, Tuple

from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.model_components.ray_samplers import VolumetricSampler
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
    grid_resolution: int = 128  # same as grid resolution or grid size
    grid_levels: int = 16
    base_resolution: int = 16
    desired_resolution: int = 2048

    min_near: float = 0.05

    density_threshold: int = 10

    # aabb_train: torch.Tensor # TODO: derived
    # aabb_infer: torch.Tensor # TODO: derived

    individual_num: int = 500
    individual_dim: int = 0
    specular_dim: int = 0

    # individual_codes # TODO: derived

    cuda_ray: bool = True

    trainable_density_grid: bool = True
    log2hash_map_size: int = 19

    # density_grid #TODO: derived

    # density_bitfield #TODO: derived

    mean_density: float = 0.0
    iter_density: float = 0.0

    sigma_layers = 2

    # TODO: register_buffer

    # NOTE: all default configs are passed to nerf2mesh field
    # NOTE: all dervied are computed in nerf2mesh field

    # TODO: config for stage 1 if needed


@dataclass
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

        self.field = Nerf2MeshField(
            num_layers_sigma=self.config.sigma_layers,
            specular_dim=self.config.specular_dim,
            hidden_dim_sigma=self.config.hidden_dim_sigma,
            hidden_dim_color=self.config.hidden_dim_color,
            num_levels=self.config.grid_levels,
            base_res=self.config.base_resolution,
            max_res=self.config.grid_resolution,
            log2_hashmap_size=self.config.log2hash_map_size,
        )

        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid, density_fn=self.field.density_fn
        )

        # Criterian in nerf2mesh
        self.loss = torch.nn.MSELoss(reduction="none")

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

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor | List]:
        ...
        # return super().get_outputs(ray_bundle)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        ...
        # return super().get_metrics_dict(outputs, batch)

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        ...

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        ...
        """Returns a dictionary of images and metrics to plot. Here you can apply your colormaps."""
