from dataclasses import dataclass, field
from typing import Literal, Union
import torch
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
    dataparser: AnnotatedDataParserUnion = Nerf2MeshDataParserConfig(eval_mode='all')

class Nerf2MeshDataStage1Manager(Nerf2MeshDataManager):
    config: Nerf2MeshDataManagerConfig

    def __init__(self, config: VanillaDataManagerConfig, device: Union[torch.device | str] = "cpu", test_mode: Literal['test'] | Literal['val'] | Literal['inference'] = "val", world_size: int = 1, local_rank: int = 0, **kwargs):
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)
        index_order = self.train_image_dataloader.cached_collated_batch['image_idx'].argsort().cpu()
        #NOTE: This is done to avoid mismatch of ground truth and predicted images (using mvp for stage 1 )
        self.train_image_dataloader.cached_collated_batch['image_idx'] = self.train_image_dataloader.cached_collated_batch['image_idx'][index_order]
        self.train_image_dataloader.cached_collated_batch['image'] = self.train_image_dataloader.cached_collated_batch['image'][index_order]
        

    def setup_train(self):
        super().setup_train()
        self.train_ray_generator = SimpleRayGenerator(
                    self.train_dataset.cameras.to(self.device),
                )
        
        
    def setup_eval(self):
        super().setup_eval()
        self.eval_ray_generator = SimpleRayGenerator(
                    self.eval_dataset.cameras.to(self.device),
                )
    

