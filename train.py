import argparse
import pathlib
from nerfstudio.configs.method_configs import method_configs
from nerf2mesh.config import nerf2mesh
 
def nerf_train(data, output_dir):
    config = nerf2mesh.config
    config.set_timestamp()
 
    config.pipeline.datamanager.data = pathlib.Path(data)
    config.output_dir = output_dir
    config.save_config()
    trainer = config.setup()
    trainer.setup()
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to data", default="/home/mohitm/datasets/nerf_synthetic_small/nerf_synthetic/chair/transforms_train.json")
    parser.add_argument("--output_dir", type=str, help="path to output dir", default="outputs/train")
    args = parser.parse_args()
    nerf_train(data=args.data, output_dir=args.output_dir)