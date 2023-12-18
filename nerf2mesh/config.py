from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManagerConfig,
)
from nerf2mesh.pipeline import Nerf2MeshPipelineConfig
from nerf2mesh.dataparser import Nerf2MeshDataParserConfig
from nerf2mesh.nerf2mesh import Nerf2MeshModelConfig
from nerf2mesh.scheduler import Nerf2MeshSchedulerConfig


max_num_iterations = 30000

nerf2mesh = MethodSpecification(
    config=TrainerConfig(
        method_name="nerf2mesh",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=max_num_iterations,
        mixed_precision=True,
        pipeline=Nerf2MeshPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=Nerf2MeshDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=Nerf2MeshModelConfig(eval_num_rays_per_chunk=8192),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                # TODO: check max_num_iterations same as opt.iters in nerf2mesh
                "scheduler": Nerf2MeshSchedulerConfig(
                    max_steps=max_num_iterations
                ),
            }
        },
        vis="viewer",
    ),
    description="nerf2mesh",
)
