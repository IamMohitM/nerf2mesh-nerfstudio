from typing import Dict, Optional
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field, FieldConfig
from torch import Tensor
from typing import Tuple
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.activations import trunc_exp
import torch
from nerfstudio.data.scene_box import SceneBox

class Nerf2MeshField(Field):
    def __init__(
        self,
        num_layers_sigma: int,
        num_layers_color: int,
        hidden_dim_sigma: int,
        num_levels: int,
        base_res: int,
        max_res: int,
        log2_hashmap_size,
        features_per_level,
        geom_init: bool = False,
        implementation: str = "tcnn",
    ) -> None:
        super().__init__()

        self.mlp_base_grid = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )

        in_activation = torch.nn.Softplus if geom_init else torch.nn.ReLU
        self.sigma_net = MLP(
            in_dim=self.mlp_base_grid.get_out_dim(),
            out_dim=1,
            num_layers=num_layers_sigma,
            layer_width=hidden_dim_sigma,
            activation=in_activation,
            out_activation=None,
        )
        self.color_net = ...

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Optional[Tensor]]:
        # TODO: check preprocessing if spatial encoding is needed or positions are within the scenebox
        # positions = ray_samples.frustums.get_positions()
        # selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        # positions = positions * selector[..., None]
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        h = self.mlp_base_grid(positions)
        h = torch.cat([positions, h], dim=-1)
        h = self.sigma_net(h)
        #TODO: add trunc exp and density before activation
        
        density = trunc_exp(density_before_activation.to(positions))
        return density

        # return self.sigma_net(*args)
        encoding = ...  # hashencoding

        ...

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Tensor | None = None
    ) -> Dict[FieldHeadNames, Tensor]:
        return super().get_outputs(ray_samples, density_embedding)

    def forward(
        self, ray_samples: RaySamples, compute_normals: bool = False
    ) -> Dict[FieldHeadNames, Tensor]:
        return super().forward(ray_samples, compute_normals)

    ...
