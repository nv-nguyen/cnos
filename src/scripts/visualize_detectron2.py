import hydra
import numpy as np
import json
import cv2
import logging
from torchvision.ops import masks_to_boxes
from src.utils.inout import load_json
import os
from segment_anything.utils.amg import rle_to_mask
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from src.utils.visualization_detectron2 import CNOSVisualizer

@hydra.main(version_base=None, config_path="../../configs", config_name="run_vis")
def visualize(cfg: DictConfig) -> None:
    if cfg.dataset_name in ["hb", "tless"]:
        split = "test_primesense"
    else:
        split = "test"
        
    num_max_objs = 50
    object_names = cfg.data.datasets[cfg.dataset_name].obj_names
    object_names = [name for name in object_names]
    
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
    for idx, (scene_id, image_id) in tqdm(enumerate(list_scene_id_and_frame_id)):
        img = Image.open(f'{cfg.root_dir}/{cfg.dataset_name}/{split}/{scene_id:06d}/rgb/{image_id:06d}.png')
        rgb = img.copy()
        img = np.array(img)
        visualizer = CNOSVisualizer(object_names, img_size=img.shape[:2])
        
        masks, object_ids, scores, bboxes = [], [], [], []
        for det in dets:
            if det['scene_id'] == scene_id and det['image_id'] == image_id:
                masks.append(rle_to_mask(det['segmentation']))
                object_ids.append(det['category_id']-1)
                scores.append(det['score'])
                bbox = det['bbox']
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                bboxes.append(bbox)
        masks = np.stack(masks)
        scores = np.array(scores)
        bboxes = np.array(bboxes)
        assert len(masks) == len(object_ids) == len(scores) == len(bboxes)
        assert np.max(object_ids) <= len(object_names)
        object_ids = np.array(object_ids)
        # conver image to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        
        scene_dir = f"{cfg.output_dir}/{cfg.dataset_name}{scene_id:06d}"
        os.makedirs(scene_dir, exist_ok=True)
        save_path = f"{scene_dir}/{image_id:06d}.png"
        
        visualizer.forward(rgb=img, masks=masks, bboxes=bboxes, scores=scores, labels=object_ids, save_path=save_path)
        prediction = Image.open(save_path)
        # concat side by side in PIL
        concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
        concat.paste(rgb, (0, 0))
        concat.paste(prediction, (img.shape[1], 0))
        concat.save(save_path)
        if idx % 10 == 0:
            logging.info(f"Saving {save_path}")
        
if __name__ == "__main__":
    visualize()