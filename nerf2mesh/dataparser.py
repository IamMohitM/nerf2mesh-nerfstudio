from dataclasses import dataclass, field

from typing import Type
from nerfstudio.data.dataparsers.instant_ngp_dataparser import (
    InstantNGPDataParserConfig,
    InstantNGP,
)


@dataclass
class Nerf2MeshDataParserConfig(InstantNGPDataParserConfig):
    _target: Type = field(default_factory=lambda: Nerf2Mesh)


@dataclass
class Nerf2Mesh(InstantNGP):
    config: Nerf2MeshDataParserConfig
