from nerfstudio.plugins.types import MethodSpecification #@IgnoreException
# from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig

from nerf2mesh.trainer import Nerf2MeshTrainerConfig
from nerf2mesh.datamanager import Nerf2MeshDataManagerConfig
from nerf2mesh.dataparser import Nerf2MeshDataParserConfig
from nerf2mesh.pipeline import Nerf2MeshPipelineConfig
from nerf2mesh.nerf2mesh import Nerf2MeshModelConfig
from nerf2mesh.scheduler import Nerf2MeshSchedulerConfig


max_num_iterations = 10000

nerf2mesh = MethodSpecification(
    config=Nerf2MeshTrainerConfig(
        method_name="nerf2mesh",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=max_num_iterations,
        mixed_precision=True,
        pipeline=Nerf2MeshPipelineConfig(
            datamanager=Nerf2MeshDataManagerConfig(
                dataparser=Nerf2MeshDataParserConfig(),
                train_num_rays_per_batch=8192,
                eval_num_rays_per_batch=8192,
            ),
            model=Nerf2MeshModelConfig(eval_num_rays_per_chunk=8192),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": Nerf2MeshSchedulerConfig(max_steps=max_num_iterations),
            },
            "vertices_offsets":
            {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15, weight_decay=0),
                "scheduler": Nerf2MeshSchedulerConfig(max_steps=max_num_iterations)
            }
        },
        vis="viewer",
    ),
    description="nerf2mesh",
)
