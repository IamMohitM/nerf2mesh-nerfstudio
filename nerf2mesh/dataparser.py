from dataclasses import dataclass, field
from typing import Type
from pathlib import Path

from nerfstudio.data.dataparsers.instant_ngp_dataparser import (
    InstantNGPDataParserConfig,
    InstantNGP,
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
    get_train_eval_split_all,
)

import imageio
import torch
import numpy as np
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.io import load_from_json


@dataclass
class Nerf2MeshDataParserConfig(InstantNGPDataParserConfig):
    _target: Type = field(default_factory=lambda: Nerf2MeshDataParser)
    min_near: float = 0.01
    max_far: float = 1000.0


@dataclass
class Nerf2MeshDataParser(InstantNGP):
    config: Nerf2MeshDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        meta = load_from_json(self.config.data / f"transforms_{split}.json")
        data_dir = self.config.data

        image_filenames = []
        mask_filenames = []
        poses = []
        num_skipped_image_filenames = 0
        for frame in meta["frames"]:
            fname = data_dir / Path(frame["file_path"])
            # search for png file
            if not fname.exists():
                fname = data_dir / Path(frame["file_path"] + ".png")
            if not fname.exists():
                CONSOLE.log(f"couldn't find {fname} image")
                num_skipped_image_filenames += 1
            else:
                if "w" not in meta:
                    img_0 = imageio.imread(fname)
                    h, w = img_0.shape[:2]
                    meta["w"] = w
                    if "h" in meta:
                        meta_h = meta["h"]
                        assert (
                            meta_h == h
                        ), f"height of image dont not correspond metadata {h} != {meta_h}"
                    else:
                        meta["h"] = h
                image_filenames.append(fname)
                poses.append(np.array(frame["transform_matrix"]))
                if "mask_path" in frame:
                    mask_fname = data_dir / Path(frame["mask_path"])
                    mask_filenames.append(mask_fname)
        if num_skipped_image_filenames >= 0:
            CONSOLE.print(
                f"Skipping {num_skipped_image_filenames} files in dataset split {split}."
            )
        assert (
            len(image_filenames) != 0
        ), """
        No image files found.
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """
        poses = np.array(poses).astype(np.float32)
        poses[:, :3, 3] *= self.config.scene_scale

        # find train and eval indices based on the eval_mode specified
        if self.config.eval_mode == "fraction":
            i_train, i_eval = get_train_eval_split_fraction(
                image_filenames, self.config.train_split_fraction
            )
        elif self.config.eval_mode == "filename":
            i_train, i_eval = get_train_eval_split_filename(image_filenames)
        elif self.config.eval_mode == "interval":
            i_train, i_eval = get_train_eval_split_interval(
                image_filenames, self.config.eval_interval
            )
        elif self.config.eval_mode == "all":
            CONSOLE.log(
                "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
            )
            i_train, i_eval = get_train_eval_split_all(image_filenames)
        else:
            raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")
        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = (
            [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        )

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        distortion_params = camera_utils.get_distortion_params(
            k1=float(meta.get("k1", 0)),
            k2=float(meta.get("k2", 0)),
            k3=float(meta.get("k3", 0)),
            k4=float(meta.get("k4", 0)),
            p1=float(meta.get("p1", 0)),
            p2=float(meta.get("p2", 0)),
        )

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = 0.5 * meta.get("aabb_scale", 1)

        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-aabb_scale, -aabb_scale, -aabb_scale],
                    [aabb_scale, aabb_scale, aabb_scale],
                ],
                dtype=torch.float32,
            )
        )

        fl_x, fl_y = InstantNGP.get_focal_lengths(meta)

        w, h = meta["w"], meta["h"]

        camera_type = CameraType.PERSPECTIVE
        if meta.get("is_fisheye", False):
            camera_type = CameraType.FISHEYE

        cameras = Cameras(
            fx=float(fl_x),
            fy=float(fl_y),
            cx=float(meta.get("cx", 0.5 * w)),
            cy=float(meta.get("cy", 0.5 * h)),
            distortion_params=distortion_params,
            height=int(h),
            width=int(w),
            camera_to_worlds=camera_to_world,
            camera_type=camera_type,
        )

        # TODO(ethan): add alpha background color
        dataparser_output = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=self.config.scene_scale,
        )
        # NOTE: assumes all height, fy, width, fx are the same
        # dataparser_output.scene_box.aabb *= 2.0
        y = dataparser_output.cameras.height[0].item() / (
            2.0 * dataparser_output.cameras.fy[0].item()
        )
        aspect = (
            dataparser_output.cameras.width[0].item()
            / dataparser_output.cameras.height[0].item()
        )
        # projection = torch.tensor(
        #     [
        #         [2 * self.config.min_near, 0, 0, 0],
        #         [0, -2 * self.config.min_near, 0, 0],
        #         [
        #             0,
        #             0,
        #             -(self.config.max_far + self.config.min_near)
        #             / (self.config.max_far - self.config.min_near),
        #             -(2 * self.config.max_far * self.config.min_near)
        #             / (self.config.max_far - self.config.min_near),
        #         ],
        #         [0, 0, -1, 0],
        #     ]
        # )
        projection = torch.tensor(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.config.max_far + self.config.min_near)
                    / (self.config.max_far - self.config.min_near),
                    -(2 * self.config.max_far * self.config.min_near)
                    / (self.config.max_far - self.config.min_near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=torch.float,
        )

        new_poses = torch.cat(
            [
                dataparser_output.cameras.camera_to_worlds,
                torch.tensor([[0, 0, 0, 1]], dtype=torch.float).repeat(
                    dataparser_output.cameras.camera_to_worlds.shape[0], 1, 1
                ),
            ],
            dim=1,
        )
        mvps = projection.unsqueeze(0) @ torch.inverse(new_poses) 
        dataparser_output.metadata["mvps"] = mvps

        return dataparser_output
