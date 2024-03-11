from nerfstudio.exporter.marching_cubes import generate_mesh_with_multires_marching_cubes
import torch

def random_sdf_function(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.sin(x[..., 0]) + torch.cos(x[..., 1]) + torch.sin(x[..., 2]), min=-1.0, max=1.0)

def test_mcubes():
    generate_mesh_with_multires_marching_cubes(random_sdf_function, resolution = 512, bounding_box_max=(1.0, 1.0, 1.0), bounding_box_min=(-1.0, )* 3)


if __name__ == "__main__":
    test_mcubes()