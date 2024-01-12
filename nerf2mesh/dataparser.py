from dataclasses import dataclass, field
from typing import Type
from nerfstudio.data.dataparsers.instant_ngp_dataparser import (
    InstantNGPDataParserConfig,
    InstantNGP,
)
import torch


@dataclass
class Nerf2MeshDataParserConfig(InstantNGPDataParserConfig):
    _target: Type = field(default_factory=lambda: Nerf2Mesh)
    min_near: float = 0.01
    max_far: float = 1000.0


@dataclass
class Nerf2Mesh(InstantNGP):
    config: Nerf2MeshDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        dataparser_output = super()._generate_dataparser_outputs(split)
        # NOTE: assumes all height, fy, width, fx are the same
        y = dataparser_output.cameras.height[0].item() / (
            2.0 * dataparser_output.cameras.fy[0].item()
        )
        aspect = (
            dataparser_output.cameras.width[0].item()
            / dataparser_output.cameras.height[0].item()
        )
        projection = torch.tensor(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.config.max_far + self.config.min_near)
                    / (self.config.max_far - self.config.min_near),
                    -(2 * self.config.max_far * self.config.min_near)
                    / (self.config.max_far - self.config.min_near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=torch.float,
        )

        new_poses = torch.cat(
            [
                dataparser_output.cameras.camera_to_worlds,
                torch.tensor([[0, 0, 0, 1]], dtype=torch.float)
                .repeat(dataparser_output.cameras.camera_to_worlds.shape[0], 1, 1),
            ],
            dim=1,
        )
        mvps = projection.unsqueeze(0) @ torch.inverse(new_poses)
        dataparser_output.metadata["mvps"] = mvps

        return dataparser_output
