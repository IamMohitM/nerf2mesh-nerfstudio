from dataclasses import dataclass, field
from typing import Type, Literal
from nerfstudio.engine.trainer import TrainerConfig, Trainer
from nerf2mesh.pipeline import Nerf2MeshPipelineStage1Config
from nerfstudio.configs.base_config import InstantiateConfig


@dataclass
class Nerf2MeshTrainerConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: Nerf2MeshTrainer)
    stage: int = 0


class Nerf2MeshTrainer(Trainer):
    config: Nerf2MeshTrainerConfig
    pipeline_stage1: Nerf2MeshPipelineStage1Config()

    # recursively update one config with another but ignore one specific key
    def update_dict(self, d, u, u_type, ignore_key="_target"):
        for k, v in u.items():
            if isinstance(v, InstantiateConfig):
                d[k] = self.update_dict(
                    d.get(k).__dict__, v.__dict__, v.__class__, ignore_key
                )
            else:
                if k != ignore_key:
                    d[k] = v
        return u_type(**d)

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
        """
        if self.config.stage == 1:
            stage1_pipeline_config = Nerf2MeshPipelineStage1Config()
            d = self.update_dict(
                stage1_pipeline_config.__dict__.copy(),
                self.config.pipeline.__dict__,
                self.config.pipeline.__class__,
                "_target",
            )
            self.config.pipeline = Nerf2MeshPipelineStage1Config(**d.__dict__)
        super().setup(test_mode)
