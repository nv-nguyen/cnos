from dataclasses import dataclass
import logging, os
import os.path as osp
from tqdm import tqdm
import time
import numpy as np
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import os.path as osp
from src.poses.utils import load_index_level_in_level2
import torch
from src.utils.bbox_utils import CropResizePad
import pytorch_lightning as pl
from src.dataloader.base_bop import BaseBOP
import src.dataloader.hot3d.clip_util as clip_util
from src.utils.inout import load_json
from enum import auto, Enum
import tarfile

pl.seed_everything(2023)

@dataclass
class Target(object):
    im_id: int
    scene_id: int
    device_type: str
    tar_filepath: str
    stream_id: str
    clip_name: str

    def to_json(self):
        return {
            "im_id": im_id,
            "scene_id": scene_id,
            "device_type": device_type,
            "tar_filepath": targets_filepath,
            "stream_id": stream_id,
            "clip_name": clip_name,
        }

class BaseBOPHOT3D(BaseBOP):
    def __init__(
        self,
        root_dir,
        **kwargs,
    ):
        self._root_dir = root_dir
        print(f"root_dir: {root_dir}")

        clip_definitions_filepath = f"{root_dir}/clip_definitions.json"
        targets_filepath = f"{root_dir}/test_targets_bop24.json"
                
        clip_definitions = load_json(clip_definitions_filepath)
        print(f"len clip_definitions: {len(clip_definitions)}")

        target_payload = load_json(targets_filepath)
        targets = []
        for tgt in target_payload:
            im_id = int(tgt["im_id"])
            scene_id=int(tgt["scene_id"])
            device_type = str(clip_definitions[str(scene_id)]["device"])
            clip_name = f"clip-{scene_id:06d}"

            tar_folder = None
            stream_id = None
            if device_type == "Aria":
                tar_folder = f"{root_dir}/test_aria"
                stream_id = "214-1"
            elif device_type == "Quest3":
                tar_folder = f"{root_dir}/test_quest3"
                stream_id = "1201-1"

            tar_filepath = f"{tar_folder}/{clip_name}.tar"
            target_obj = Target(                
                im_id=im_id,
                scene_id=scene_id,
                device_type=device_type,
                tar_filepath=tar_filepath,
                stream_id=stream_id,
                clip_name=clip_name,
            )
            targets += [target_obj]

        self._targets = targets[0:]
        print(f"len targets: {len(self._targets)}")
        
        self._rgb_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, idx):
        st = self._targets[idx]
        tar = tarfile.open(st.tar_filepath, mode="r")
        frame_id = st.im_id
        scene_id = st.scene_id
        stream_id = st.stream_id

        frame_key = f"{frame_id:06d}"
        image_np: np.ndarray = clip_util.load_image(tar, frame_key, stream_id)
        image = Image.fromarray(np.uint8(image_np))
        image = self._rgb_transform(image.convert("RGB"))
        return dict(
            image=image,
            scene_id=scene_id,
            frame_id=frame_id,
        )





if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from omegaconf import DictConfig, OmegaConf
    from torchvision.utils import make_grid, save_image

    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )
    inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )

    dataset = BaseBOPHOT3D(
        root_dir="/home/prithv/workplace/code/hot3d/cnos/cnos/datasets/bop23_challenge/datasets/hot3d",
        processing_config=processing_config,
    )

    for idx in tqdm(range(len(dataset))):
        if idx > 5:
            break
        sample = dataset[idx]
        image_out = inv_rgb_transform(sample["image"])
        save_image(image_out, f"./tmp/hot3d_{idx}.png", nrow=7)
