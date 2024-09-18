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
        num_imgs_per_obj=50,
        **kwargs,
    ):
        self.template_dir = template_dir
        if obj_ids is None:
            obj_ids = [
                int(obj_id[4:10])
                for obj_id in os.listdir(template_dir)
                if osp.isdir(osp.join(template_dir, obj_id))
            ]
            obj_ids = sorted(np.unique(obj_ids).tolist())
            logging.info(f"Found {obj_ids} objects in {self.template_dir}")
        if "onboarding_static" in template_dir or "onboarding_dynamic" in template_dir:
            self.model_free_onboarding = True
        else:
            self.model_free_onboarding = False
        # for HOT3D, we have black objects so we use gray background
        if "hot3d" in template_dir:
            self.use_gray_background = True
            logging.info("Use gray background for HOT3D")
        else:
            self.use_gray_background = False
        self.num_imgs_per_obj = num_imgs_per_obj  # to avoid memory issue
        self.obj_ids = obj_ids
        self.processing_config = processing_config
        self.rgb_transform = T.Compose(
            [
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.proposal_processor = CropResizePad(self.processing_config.image_size, pad_value=0.5 if self.use_gray_background else 0)
        self.load_template_poses(level_templates, pose_distribution)

    def __len__(self):
        return len(self.obj_ids)

    def load_template_poses(self, level_templates, pose_distribution):
        if pose_distribution == "all":
            self.index_templates = load_index_level_in_level2(level_templates, "all")
        else:
            raise NotImplementedError

    def __getitem__modelbased__(self, idx):
        templates, masks, boxes = [], [], []
        for id_template in self.index_templates:
            image = Image.open(
                f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}/{id_template:06d}.png"
            )
            boxes.append(image.getbbox())

            mask = image.getchannel("A")
            mask = torch.from_numpy(np.array(mask) / 255).float()
            masks.append(mask.unsqueeze(-1))

            if self.use_gray_background:
                gray_image = Image.new("RGB", image.size, (128, 128, 128))
                gray_image.paste(image, mask=image.getchannel("A"))
                image = gray_image.convert("RGB")
            else:
                image = image.convert("RGB")
            image = torch.from_numpy(np.array(image) / 255).float()
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

    def __getitem__modelfree__(self, idx):
        templates, masks, boxes = [], [], []
        static_onboarding = True if "onboarding_static" in self.template_dir else False
        if static_onboarding:
            obj_dirs = [
                f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}_up",
                f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}_down",
            ]
            num_selected_imgs = self.num_imgs_per_obj // 2  # 100 for 2 videos
        else:
            obj_dirs = [
                f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}",
            ]
            num_selected_imgs = self.num_imgs_per_obj
        for obj_dir in obj_dirs:
            obj_rgb_dir = Path(obj_dir) / "rgb"
            obj_mask_dir = Path(obj_dir) / "mask_visib"
            # rgb and mask do not have same extension .png or .jpg
            # list all rgb
            obj_images = sorted(list(obj_rgb_dir.glob("*.png")))
            if len(obj_images) == 0:
                obj_images = sorted(list(obj_rgb_dir.glob("*.jpg")))
            # list all masks
            obj_masks = sorted(list(obj_mask_dir.glob("*.png")))
            if len(obj_masks) == 0:
                obj_masks = sorted(list(obj_mask_dir.glob("*.jpg")))
            assert len(obj_images) == len(
                obj_masks
            ), f"rgb and mask mismatch in {obj_dir}"
            selected_idx = np.random.choice(
                len(obj_images), num_selected_imgs, replace=False
            )
            for idx_img in tqdm(selected_idx):
                image = Image.open(obj_images[idx_img])
                mask = Image.open(obj_masks[idx_img])
                image = np.asarray(image) * np.expand_dims(np.asarray(mask) > 0, -1)
                image = Image.fromarray(image)
                boxes.append(mask.getbbox())

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

    def __getitem__(self, idx):
        if self.model_free_onboarding:
            return self.__getitem__modelfree__(idx)
        else:
            return self.__getitem__modelbased__(idx)


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
        template_dir="/home/nguyen/Documents/datasets/gigaPose_datasets/datasets/templates_pyrender/hot3d",
        obj_ids=None,
        level_templates=0,
        pose_distribution="all",
        processing_config=processing_config,
    )
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        sample["templates"] = inv_rgb_transform(sample["templates"])
        save_image(sample["templates"], f"./tmp/hot3d_{idx}.png", nrow=7)
