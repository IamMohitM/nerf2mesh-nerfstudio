from typing import Dict, Optional
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field, get_normalized_directions
from torch import Tensor
from typing import Tuple
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.mlp import MLP
import torch
from nerfstudio.data.scene_box import SceneBox
from nerf2mesh.utils import Shading
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd


class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


trunc_exp = _trunc_exp.apply


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
        num_levels: int,
        base_res: int,
        max_res: int,
        log2_hashmap_size,
        features_per_level: int = 2,
        geom_init: bool = False,
        implementation: str = "tcnn",
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)

        self.encoder = HashEncoding(
            num_levels=num_levels_sigma_encoder,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=n_features_per_level_sigma_encoder,
            implementation=implementation,
        )

        in_activation = torch.nn.Softplus() if geom_init else torch.nn.ReLU()
        self.sigma_net = MLP(
            in_dim=3 + self.encoder.get_out_dim(),
            out_dim=1,
            num_layers=num_layers_sigma,
            layer_width=hidden_dim_sigma,
            activation=in_activation,
            out_activation=None,
        )

        self.encoder_color = HashEncoding(
            num_levels=num_levels_color_encoder,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=n_features_per_level_color_encoder,
            implementation=implementation,
        )

        self.color_net = MLP(
            in_dim=3 + self.encoder_color.get_out_dim(),
            num_layers=3,
            layer_width=hidden_dim_color,
            out_dim=3 + specular_dim,
        )

        self.specular_net = MLP(
            in_dim=specular_dim + 3, num_layers=2, layer_width=32, out_dim=3
        )

    def forward(self, ray_samples: RaySamples, shading: Shading, compute_normals: bool = False, ) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        density, density_embedding = self.get_density(ray_samples)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding, shading=shading)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore
        
        # field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Optional[Tensor]]:
        # TODO: check preprocessing if spatial encoding is needed or positions are within the scenebox
        # positions = ray_samples.frustums.get_positions()
        # selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        # positions = positions * selector[..., None]

        positions = SceneBox.get_normalized_positions(
            ray_samples.frustums.get_positions(), self.aabb
        )
        
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        #TODO:  WHy are number of positions !=0 are much less x in nerf2mesh
        # Both are inputs to encoder
        h = self.encoder(positions)
        h = torch.cat([positions, h], dim=-1)
        # h = trunc_exp(self.sigma_net(h))
        h = self.sigma_net(h)
        # TODO: add trunc exp and density before activation

        density = trunc_exp(h[..., 0])
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

    def _get_specular_color(self, d, diffuse_feat, c=None):
        specular = self.specular_net(torch.cat([d, diffuse_feat[..., 3:]], dim=-1))
        return torch.sigmoid(specular)

    def get_outputs(
        self, ray_samples: RaySamples, shading: Shading, density_embedding: Tensor | None = None
    ) -> Dict[FieldHeadNames, Tensor]:
        # TODO: implement this
        outputs = {}
        positions = SceneBox.get_normalized_positions(
            ray_samples.frustums.get_positions(), self.aabb
        )
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        # d = self.encoder_color(directions_flat)

        # same as geo_feat from nerf2mesh
        diffuse_feat = self._get_diffuse_color(positions)
        diffuse = diffuse_feat[..., :3]

        if shading == Shading.diffuse:
            color = diffuse
        else:
            specular_feat = self._get_specular_color(directions_flat, diffuse_feat)
            if shading == Shading.specular:
                color = specular_feat
            else:
        # add support for shading
                color = (specular_feat + diffuse).clamp(0, 1)

        outputs[FieldHeadNames.RGB] = color.view_as(directions)
        outputs["num_samples_per_batch"] = positions.shape[0]
        # outputs[FieldHeadNames.]
        return outputs
