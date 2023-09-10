from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManagerConfig,
)
from nerfstudio.data.dataparsers.instant_ngp_dataparser import (
    InstantNGPDataParserConfig,
)
from nerf2mesh.nerf2mesh import Nerf2MeshModelConfig

nerf2mesh = MethodSpecification(
    config=TrainerConfig(
        method_name="nerf2mesh",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=DynamicBatchPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=InstantNGPDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=Nerf2MeshModelConfig(eval_num_rays_per_chunk=8192),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                # TODO: add scheduler config from nerf2mesh
                # "scheduler": ExponentialDecaySchedulerConfig(
                #     lr_final=0.0001, max_steps=200000
                # ),
            }
        },
        vis="viewer",
    ),
    description="nerf2mesh",
)
