import math

import torch

from dataclasses import dataclass, field
from typing import Dict, List, Type, Optional

from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from torch.nn import Parameter
from nerf2mesh.field import Nerf2MeshField
import nerfacc



@dataclass
class Nerf2MeshModelConfig(ModelConfig):
    _target: Type = field(default_factory = lambda: Nerf2MeshModel)

    real_bound: float = 1.0

    contract: bool = False
    collider_params: Optional[Dict[str, float]] = None

    # bound: float = 1.0 #TODO derived

    # cascade: float TODO: add cascade in model def derived

    # TODO: check if grid_resolution and desired_resolution are the same
    # understand each parameter related to grid
    grid_resolution: int = 128 # same as grid resolution or grid size
    grid_levels: int = 16
    base_resolution: int = 16
    desired_resolution: int = 2048

    min_near: float = 0.05

    density_threshold: int = 10

    # aabb_train: torch.Tensor # TODO: derived
    #aabb_infer: torch.Tensor # TODO: derived


    individual_num : int = 500
    individual_dim: int = 0

    # individual_codes # TODO: derived

    cuda_ray: bool = True

    trainable_density_grid: bool = True

    # density_grid #TODO: derived

    # density_bitfield #TODO: derived

    mean_density: float = 0.0
    iter_density: float = 0.0

    #register_buffer

    # NOTE: all default configs are passed to nerf2mesh field
    # NOTE: all dervied are computed in nerf2mesh field

    # TODO: config for stage 1 if needed




@dataclass
class Nerf2MeshModel(Model):

    config: Nerf2MeshModelConfig
    field: Nerf2MeshField

    #encoder
    #fields
    #renderer - NerfRender
    #sampler


    # #Raybundlers > generate samples  --> field

    # NOTE: to Model I must pass a scenebox and a config

    def populate_modules(self):
        ...

        self.scene_aabb = torch.nn.Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels
        )

        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn)

        # return super().populate_modules()

    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        ...