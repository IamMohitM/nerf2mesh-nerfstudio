import time
import torch

from dataclasses import dataclass, field
from typing import Type, Literal
from nerfstudio.engine.trainer import TrainerConfig, Trainer
from nerf2mesh.pipeline import Nerf2MeshPipelineStage1Config
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.engine.callbacks import TrainingCallbackLocation
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.misc import step_check

from rich import box, style
from rich.table import Table
from rich.panel import Panel

@dataclass
class Nerf2MeshTrainerConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: Nerf2MeshTrainer)
    stage: int = 0


class Nerf2MeshTrainer(Trainer):
    config: Nerf2MeshTrainerConfig

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

    def train(self) -> None:
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"

        self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
            self.base_dir / "dataparser_transforms.json"
        )

        self._init_viewer_state()
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.max_num_iterations
            step = 0
            for step in range(self._start_step, self._start_step + num_iterations):
                while self.training_state == "paused":
                    time.sleep(0.01)
                with self.train_lock:
                    with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                        self.pipeline.train()

                        # training callbacks before the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                            )

                        # time the forward pass
                        loss, loss_dict, metrics_dict = self.train_iteration(step)

                        # training callbacks after the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                            )
                        
                        # Added this for refining - rest of the function is inherited from Trainer
                        if self.config.stage > 1 and self.pipeline.model.config.refine and step > 0 and step % self.pipeline.model.config.refine_steps==0:
                            self.optimizer = self.setup_optimizers()


                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.world_size
                        * self.pipeline.datamanager.get_train_rays_per_batch()
                        / max(0.001, train_t.duration),
                        step=step,
                        avg_over_steps=True,
                    )

                self._update_viewer_state(step)

                # a batch of train rays
                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                    # The actual memory allocated by Pytorch. This is likely less than the amount
                    # shown in nvidia-smi since some unused memory can be held by the caching
                    # allocator and some context needs to be created on GPU. See Memory management
                    # (https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
                    # for more details about GPU memory management.
                    writer.put_scalar(
                        name="GPU Memory (MB)", scalar=torch.cuda.max_memory_allocated() / (1024**2), step=step
                    )

                # Do not perform evaluation if there are no validation images
                if self.pipeline.datamanager.eval_dataset:
                    self.eval_iteration(step)

                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()

        # save checkpoint at the end of training
        self.save_checkpoint(step)

        # write out any remaining events (e.g., total train time)
        writer.write_out_storage()

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Config File", str(self.config.get_base_dir() / "config.yml"))
        table.add_row("Checkpoint Directory", str(self.checkpoint_dir))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Training Finished :tada:[/bold]", expand=False))

        # after train end callbacks
        for callback in self.callbacks:
            callback.run_callback_at_location(step=step, location=TrainingCallbackLocation.AFTER_TRAIN)

        if not self.config.viewer.quit_on_train_completion:
            self._train_complete_viewer()
