from typing import Type, Dict
from dataclasses import dataclass, field
from nerfstudio.data.pixel_samplers import PixelSampler, PixelSamplerConfig

import torch

@dataclass
class AllPixelSamplerConfig(PixelSamplerConfig):
    _target : Type = field(default_factory=lambda: AllPixelSampler)

class AllPixelSampler(PixelSampler):
    config: AllPixelSamplerConfig

    def sample(self, image_batch: Dict):
        device = image_batch["image"][0].device
        num_images, image_height, image_width, _ = image_batch["image"].shape

        image_index = torch.randint(num_images, (1,), device=device)
        i, j= torch.meshgrid(
            torch.arange(image_height, device=device),
            torch.arange(image_width, device=device),
        )
        
        pixels = torch.stack([j, i], dim=-1).reshape(-1, 2)
        pixels = torch.cat((image_index.repeat(pixels.shape[0], 1), pixels), dim=1)

        collated_batch = {}
        collated_batch['image'] = image_batch['image']
        collated_batch['indices'] = pixels

        return collated_batch
