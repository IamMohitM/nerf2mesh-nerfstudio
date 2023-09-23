from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
from nerfstudio.engine.schedulers import SchedulerConfig, Scheduler

from typing import Type
from dataclasses import dataclass, field


@dataclass
class Nerf2MeshSchedulerConfig(SchedulerConfig):
    _target: Type = field(default_factory=lambda: Nerf2MeshScheduler)
    max_steps: int = 30000


class Nerf2MeshScheduler(Scheduler):
    config: Nerf2MeshSchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        return LambdaLR(
            optimizer,
            lr_lambda=lambda iter: lr_init + 0.99 * (iter / 500)
            if iter <= 500
            else 0.1 ** ((iter - 500) / (self.config.max_steps - 500)), #TODO: The self.config.max_steps should be checked
        )
