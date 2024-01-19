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
    

