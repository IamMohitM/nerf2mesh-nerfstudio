This project is a work in progress

# Aim
This project aims to integrate the [nerf2mesh](https://me.kiui.moe/nerf2mesh/) method suggested by Jiaxiang Tang et al, with NerfStudio. 

# Train

```
NERFSTUDIO_DATAPARSER_CONFIGS=nerf2mesh=nerf2mesh.config:Nerf2MeshDataParserConfig  python3.10 train.py --data path_to_transform.json --output_dir path_to_output_dir
```

You can also use the launch.json in the .vscode for starting debugging sessions.

# TODO
- [x] Implement NerFModel from nerf2mesh - use Nerfstudio API
- [-] Fix coarse mesh - stage 0 bugs
- [] Implement Stage 1 from Nerf2mesh