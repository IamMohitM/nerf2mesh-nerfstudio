import argparse
import pathlib
from nerf2mesh.config import nerf2mesh
 
def nerf_train(data, output_dir):
    config = nerf2mesh.config
    config.set_timestamp()
    
    config.pipeline.datamanager.data = pathlib.Path(data)
    config.max_num_iterations=10000
    config.mixed_precision = True
    config.pipeline.model.mark_unseen_triangles = True
    config.output_dir = output_dir
    config.viewer.quit_on_train_completion = True
    checkpoint_dir = config.get_checkpoint_dir()
    config.pipeline.model.coarse_mesh_path = checkpoint_dir.parent / "meshes/mesh_0.ply"
    config.save_config()
    trainer = config.setup()
    trainer.setup()
    trainer.train()
    del trainer # to free up memory

    config.stage = 1
    config.mixed_precision = False
    config.max_num_iterations=20000
    config.load_dir = checkpoint_dir
    config.pipeline.model.fine_mesh_path = config.pipeline.model.coarse_mesh_path.parent
    
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