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

pl.seed_everything(2023)


class BOPTemplate(Dataset):
    def __init__(
        self,
        template_dir,
        obj_ids,
        processing_config,
        level_templates,
        pose_distribution,
        **kwargs,
    ):
        self.template_dir = template_dir
        if obj_ids is None:
            obj_ids = [
                int(obj_id[4:])
                for obj_id in os.listdir(template_dir)
                if osp.isdir(osp.join(template_dir, obj_id))
            ]
            obj_ids = sorted(obj_ids)
            logging.info(f"Found {obj_ids} objects in {self.template_dir}")
        self.obj_ids = obj_ids
        self.processing_config = processing_config
        self.rgb_transform = T.Compose(
            [
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.proposal_processor = CropResizePad(self.processing_config.image_size)
        self.load_template_poses(level_templates, pose_distribution)

    def __len__(self):
        return len(self.obj_ids)

    def load_template_poses(self, level_templates, pose_distribution):
        if pose_distribution == "all":
            self.index_templates = load_index_level_in_level2(level_templates, "all")
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        templates, masks, boxes = [], [], []
        for id_template in self.index_templates:
            image = Image.open(
                f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}/{id_template:06d}.png"
            )
            boxes.append(image.getbbox())

            mask = image.getchannel("A")
            mask = torch.from_numpy(np.array(mask) / 255).float()
            masks.append(mask.unsqueeze(-1))

            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            templates.append(image)

        templates = torch.stack(templates).permute(0, 3, 1, 2)
        masks = torch.stack(masks).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        templates_croped = self.proposal_processor(images=templates, boxes=boxes)
        masks_cropped = self.proposal_processor(images=masks, boxes=boxes)
        return {
            "templates": self.rgb_transform(templates_croped),
            "template_masks": masks_cropped[:, 0, :, :],
        }


class BaseBOPTest(BaseBOP):
    def __init__(
        self,
        root_dir,
        split,
        **kwargs,
    ):
        self.root_dir = root_dir
        self.split = split
        self.load_list_scene(split=split)
        self.load_metaData(reset_metaData=True)
        # shuffle metadata
        self.metaData = self.metaData.sample(frac=1, random_state=2021).reset_index()
        self.rgb_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __getitem__(self, idx):
        rgb_path = self.metaData.iloc[idx]["rgb_path"]
        scene_id = self.metaData.iloc[idx]["scene_id"]
        frame_id = self.metaData.iloc[idx]["frame_id"]
        image = Image.open(rgb_path)
        image = self.rgb_transform(image.convert("RGB"))
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
    dataset = BOPTemplate(
        template_dir="/home/nguyen/Documents/datasets/bop23/datasets/templates_pyrender/lmo",
        obj_ids=None,
        level_templates=0,
        pose_distribution="all",
        processing_config=processing_config,
    )
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        sample["templates"] = inv_rgb_transform(sample["templates"])
        save_image(sample["templates"], f"./tmp/lm_{idx}.png", nrow=7)