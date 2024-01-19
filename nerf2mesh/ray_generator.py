from jaxtyping import Int
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
import torch
from torch import Tensor

# NOTE: This class is written to replicate nerf2mesh ray generator for stage 1.
# The raygenerator from nerfstudio normalizes the rays, which is not done in nerf2mesh.


class SimpleRayGenerator(RayGenerator):
    def forward(self, ray_indices: Tensor) -> RayBundle:
        c = ray_indices[:, 0].to(self.cameras.device)  # camera indices
        assert (
            len(torch.unique(c)) == 1
        ), "Only one camera is supported for this ray generator"
        y = ray_indices[:, 1].to(self.cameras.device)  # row indices
        x = ray_indices[:, 2].to(self.cameras.device)  # col indices

        current_image_index = c[0].item()

        width = self.cameras.image_width[current_image_index].item()
        height = self.cameras.image_height[current_image_index].item()
        # camera indices

        cx = self.cameras.cx[current_image_index].item()
        cy = self.cameras.cy[current_image_index].item()
        fx = self.cameras.fx[current_image_index].item()
        fy = self.cameras.fy[current_image_index].item()

        y = self.image_coords[..., 0].contiguous().view(-1)
        x = self.image_coords[..., 1].contiguous().view(-1)
        zs = -torch.ones_like(c)
        xs = (x - cx) / fx
        ys = -(y - cy) / fy

        direction_stack = torch.stack((xs, ys, zs), dim=-1)  # [N, 3]

        directions = (
            direction_stack.unsqueeze(1)
            @ self.cameras.camera_to_worlds[c, :3, :3].transpose(-1, -2)
        ).squeeze(1)

        origins = self.cameras.camera_to_worlds[c, :3, 3]

        # pixel area is constant for all pixels -
        pixel_area = torch.tensor(1 / (width * height)).expand_as(
            origins[..., 0]
        )  # .to(self.cameras.device)

        ray_bundle = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area[..., None],
            camera_indices=c[..., None],
            times=None,
            metadata={
                "height": self.cameras.image_height[c],
                "width": self.cameras.image_width[c],
            },
        )

        return ray_bundle
