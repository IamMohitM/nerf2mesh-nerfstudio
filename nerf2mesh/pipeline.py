from typing import Type
from dataclasses import dataclass, field
import torch

from nerfstudio.pipelines.dynamic_batch import VanillaPipelineConfig, VanillaPipeline
from nerfstudio.utils import profiler
from nerf2mesh.utils import Shading



@dataclass
class Nerf2MeshPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: Nerf2MeshPipeline)
    stage: int = 0 # training stage 0 - coarse mesh only, 1 - fine mesh only
    diffuse_only: bool = False # if true, only train diffuse
    initial_diffuse_steps: int = 1000

class Nerf2MeshPipeline(VanillaPipeline):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        #NOTE: assumes all height, fy, width, fx are the same for all cameras
        self.model.mvps = torch.cat([self.datamanager.train_dataset.metadata["mvps"], self.datamanager.eval_dataset.metadata["mvps"]], dim=0)
        self.model.image_height = self.datamanager.train_dataset.cameras.image_height[0].item()
        self.model.image_width = self.datamanager.train_dataset.cameras.image_width[0].item()

    
    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if (self.config.stage == 0 and step < self.config.initial_diffuse_steps) or self.config.diffuse_only:
            self._model.config.shading_type = Shading.diffuse
        else:
            self._model.config.shading_type = Shading.full
        return super().get_train_loss_dict(step)
    