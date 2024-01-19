from typing import Type, Dict, Any
from dataclasses import dataclass, field
import torch
from nerfstudio.data.datamanagers.base_datamanager import DataManagerConfig
from nerfstudio.pipelines.dynamic_batch import VanillaPipelineConfig, VanillaPipeline
from nerfstudio.utils import profiler
from nerf2mesh.utils import Shading
from nerfstudio.models.base_model import ModelConfig

from nerf2mesh.datamanager import (
    Nerf2MeshDataManagerConfig,
    Nerf2MeshDataManagerStage1Config,
)
from nerf2mesh.nerf2mesh import Nerf2MeshStage1ModelConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp.grad_scaler import GradScaler


@dataclass
class Nerf2MeshPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: Nerf2MeshPipeline)
    stage: int = 0  # training stage 0 - coarse mesh only, 1 - fine mesh only
    diffuse_only: bool = False  # if true, only train diffuse
    initial_diffuse_steps: int = 1000
    datamanager: DataManagerConfig = Nerf2MeshDataManagerConfig()


class Nerf2MeshPipeline(VanillaPipeline):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # NOTE: assumes all height, fy, width, fx are the same for all cameras
        self.model.all_mvps = torch.cat(
            [
                self.datamanager.train_dataset.metadata["mvps"],
                self.datamanager.eval_dataset.metadata["mvps"],
            ],
            dim=0,
        ).to(self.model.device)
        self.model.train_mvp = self.datamanager.train_dataset.metadata["mvps"].to(
            self.model.device
        )
        self.model.field.train_mvp = self.model.train_mvp
        self.model.eval_mvp = self.datamanager.eval_dataset.metadata["mvps"].to(
            self.model.device
        )
        self.model.field.evel_mvp = self.model.eval_mvp
        self.model.image_height = self.datamanager.train_dataset.cameras.image_height[
            0
        ].item()
        self.model.image_width = self.datamanager.train_dataset.cameras.image_width[
            0
        ].item()
        self.model.image_cx = self.datamanager.train_dataset.cameras.cx[0].item()
        self.model.image_cy = self.datamanager.train_dataset.cameras.cy[0].item()

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if (
            self.config.stage == 0 and step < self.config.initial_diffuse_steps
        ) or self.config.diffuse_only:
            self._model.config.shading_type = Shading.diffuse
        else:
            self._model.config.shading_type = Shading.full
        return super().get_train_loss_dict(step)


@dataclass
class Nerf2MeshPipelineStage1Config(Nerf2MeshPipelineConfig):
    _target: Type = field(default_factory=lambda: Nerf2MeshPipelineStage1)
    stage: int = 1  # training stage 0 - coarse mesh only, 1 - fine mesh only
    datamanager: DataManagerConfig = Nerf2MeshDataManagerStage1Config()
    model: ModelConfig = Nerf2MeshStage1ModelConfig()


class Nerf2MeshPipelineStage1(Nerf2MeshPipeline):
    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        self._model.config.shading_type = Shading.full
        return super().get_train_loss_dict(step)

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value
            for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)

        self.model.prev_field.load_state_dict(
            {
                key[len("_model.field.") :]: value
                for key, value in state.items()
                if key.startswith("_model.field")
            }
        )
