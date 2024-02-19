import argparse
import pathlib
from nerf2mesh.config import nerf2mesh
 
def nerf_train(data, output_dir):
    config = nerf2mesh.config
    config.set_timestamp()
    config.stage = 1
    config.load_dir = pathlib.Path("outputs/chair+mesh/nerf2mesh_mark_unseen/chair/nerf2mesh/2024-02-18_225635/nerfstudio_models")
    config.pipeline.model.coarse_mesh_path = "meshes/mesh_0_with_unseenmarked.ply"
    config.pipeline.model.fine_mesh_path = "meshes/"
    config.pipeline.datamanager.data = pathlib.Path(data)
    
    
    config.pipeline.model.mark_unseen_triangles = True
    config.output_dir = output_dir
    config.save_config()
    trainer = config.setup()
    trainer.setup()
    trainer.train()

# if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="path to data", default="~/datasets/nerf_synthetic_small/nerf_synthetic/chair/transforms_train.json")
parser.add_argument("--output_dir", type=str, help="path to output dir", default="outputs/train")
args = parser.parse_args()
nerf_train(data=args.data, output_dir=args.output_dir)