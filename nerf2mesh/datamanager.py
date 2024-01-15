from dataclasses import dataclass, field
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager, TDataset
from nerfstudio.data.pixel_samplers import PixelSampler, PatchPixelSamplerConfig, PixelSamplerConfig
from nerfstudio.utils.rich_utils import CONSOLE

from nerf2mesh.sampler import AllPixelSamplerConfig

@dataclass
class Nerf2MeshDataManagerConfig(VanillaDataManagerConfig):
    _target: type = field(default_factory=lambda: Nerf2MeshDataManager)
    stage: int = 0

class Nerf2MeshDataManager(VanillaDataManager):
    config: Nerf2MeshDataManagerConfig
    
    def _get_pixel_sampler(self, dataset: TDataset, num_rays_per_batch: int) -> PixelSampler:
        """Infer pixel sampler to use."""
        if self.config.stage == 0:
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
        elif self.config.stage == 1:
            return AllPixelSamplerConfig().setup()

    

