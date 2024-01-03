from typing import Dict, Optional, Tuple
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.field_components.activations import trunc_exp
from torch import Tensor
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.mlp import MLP
import torch
from nerfstudio.data.scene_box import SceneBox
from nerf2mesh.utils import Shading

class Nerf2MeshField(Field):
    ...
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
        **kwargs
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

        #TODO: change in_dim and out_dim(remove self.geo_feat_dim) eventually
        self.sigma_net = MLP(
            in_dim=self.encoder.get_out_dim(),
            out_dim=1,
            num_layers=num_layers_sigma,
            layer_width=hidden_dim_sigma,
            activation=torch.nn.ReLU(),
            out_activation=None,
            implementation=implementation
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
            implementation=implementation
        )

        self.specular_net = MLP(
            in_dim=specular_dim + 3, num_layers=2, layer_width=32, out_dim=3,
             activation=torch.nn.ReLU(),
             implementation=implementation
        )

    def forward(self, ray_samples: RaySamples, shading: Shading, compute_normals: bool = False, ) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        density, density_embedding = self.get_density(ray_samples)
        

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding, shading=shading)
        # field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore
        
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
        self, ray_samples: RaySamples, shading: Shading, density_embedding: Tensor | None = None
    ) -> Dict[FieldHeadNames, Tensor]:
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
        # add support for shading
                color = (specular_feat + diffuse).clamp(0, 1)
            outputs['specular'] = specular_feat

        outputs[FieldHeadNames.RGB] = color
        return outputs
