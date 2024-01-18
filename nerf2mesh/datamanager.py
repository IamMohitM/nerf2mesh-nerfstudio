from dataclasses import dataclass, field
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager, TDataset
from nerfstudio.data.pixel_samplers import PixelSampler, PatchPixelSamplerConfig, PixelSamplerConfig
from nerfstudio.utils.rich_utils import CONSOLE
from nerf2mesh.dataparser import Nerf2MeshDataParserConfig
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion

from nerf2mesh.ray_generator import SimpleRayGenerator
from nerf2mesh.sampler import AllPixelSamplerConfig

@dataclass
class Nerf2MeshDataManagerConfig(VanillaDataManagerConfig):
    _target: type = field(default_factory=lambda: Nerf2MeshDataManager)
    stage: int = 0
    dataparser: AnnotatedDataParserUnion = Nerf2MeshDataParserConfig()
    train_num_rays_per_batch: int = 8192
    eval_num_rays_per_batch: int = 8192

class Nerf2MeshDataManager(VanillaDataManager):
    config: Nerf2MeshDataManagerConfig
    
    def _get_pixel_sampler(self, dataset: TDataset, num_rays_per_batch: int) -> PixelSampler:
        """Infer pixel sampler to use."""
        if self.config.patch_size > 1 and type(self.config.pixel_sampler) is PixelSamplerConfig:
            return PatchPixelSamplerConfig().setup(
                patch_size=self.config.patch_size, num_rays_per_batch=num_rays_per_batch
            )
        is_equirectangular = (dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value).all()
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
        return self.config.pixel_sampler.setup(
            is_equirectangular=is_equirectangular, num_rays_per_batch=num_rays_per_batch
        )
        
@dataclass
class Nerf2MeshDataManagerStage1Config(Nerf2MeshDataManagerConfig):
    _target: type = field(default_factory=lambda: Nerf2MeshDataStage1Manager)
    pixel_sampler: PixelSamplerConfig = AllPixelSamplerConfig()

class Nerf2MeshDataStage1Manager(Nerf2MeshDataManager):
    config: Nerf2MeshDataManagerConfig

    def setup_train(self):
        super().setup_train()
        self.train_ray_generator = SimpleRayGenerator(
                    self.train_dataset.cameras.to(self.device),
                    self.train_camera_optimizer,
                )
        
    def setup_eval(self):
        super().setup_eval()
        self.eval_ray_generator = SimpleRayGenerator(
                    self.eval_dataset.cameras.to(self.device),
                    self.eval_camera_optimizer,
                )
    
    # def _get_pixel_sampler(self, dataset: TDataset, num_rays_per_batch: int) -> PixelSampler:
    #     """Infer pixel sampler to use."""
        # return AllPixelSamplerConfig().setup()

    

