import hydra
import numpy as np
import json
import cv2
from pycocotools import mask as mutils
from pycocotools import _mask as coco_mask
from skimage.feature import canny
from skimage.morphology import binary_dilation
import matplotlib.pyplot as plt
import logging
import distinctipy
import torch
from torchvision.ops import masks_to_boxes
from src.utils.inout import load_json
import os
from segment_anything.utils.amg import rle_to_mask
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../../configs", config_name="run_vis")
def visualize(cfg: DictConfig) -> None:
    if cfg.dataset_name in ["hb", "tless"]:
        split = "test_primesense"
    else:
        split = "test"
        
    num_max_objs = 50
    colors = distinctipy.get_colors(num_max_objs)
    
    logging.info("Loading detections...")
    with open(cfg.input_file, 'r') as f:
        dets = json.load(f)
    logging.info(f'Loaded {len(dets)} detections')
    dets = [det for det in dets if det['score'] > cfg.conf_threshold]
    logging.info(f'Keeping only {len(dets)} detections having score > {cfg.conf_threshold}')
    
    
    # sort by (scene_id, frame_id)
    dets = sorted(dets, key=lambda x: (x['scene_id'], x['image_id']))
    list_scene_id_and_frame_id = [(det['scene_id'], det['image_id']) for det in dets]
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    counter = 0
    for scene_id, image_id in tqdm(list_scene_id_and_frame_id):
        img = Image.open(f'{cfg.root_dir}/{cfg.dataset_name}/{split}/{scene_id:06d}/rgb/{image_id:06d}.png')
        rgb = img.copy()
        img = np.array(img)
        masks, object_ids, scores = [], [], []
        for det in dets:
            if det['scene_id'] == scene_id and det['image_id'] == image_id:
                masks.append(rle_to_mask(det['segmentation']))
                object_ids.append(det['category_id']-1)
                scores.append(det['score'])
        # color_map = {obj_id: color for obj_id in np.unique(object_ids)}
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # img = (255*img).astype(np.uint8)
        
        alpha = 0.33
        for mask_idx, mask in enumerate(masks):
            edge = canny(mask)
            edge = binary_dilation(edge, np.ones((2, 2)))
            obj_id = object_ids[mask_idx]
            temp_id = obj_id - 1

            r = int(255*colors[temp_id][0])
            g = int(255*colors[temp_id][1])
            b = int(255*colors[temp_id][2])
            img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
            img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
            img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
            img[edge, :] = 255
        
        scene_dir = f"{cfg.output_dir}/{cfg.dataset_name}{scene_id:06d}"
        os.makedirs(scene_dir, exist_ok=True)
        save_path = f"{scene_dir}/{image_id:06d}.png"
        img = Image.fromarray(np.uint8(img))
        img.save(save_path)
        prediction = Image.open(save_path)
        # concat side by side in PIL
        img = np.array(img)
        concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
        concat.paste(rgb, (0, 0))
        concat.paste(prediction, (img.shape[1], 0))
        concat.save(save_path)
        if counter % 10 == 0:
            logging.info(f"Saving {save_path}")
        counter+=1

if __name__ == "__main__":
    visualize()