import os
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
import distinctipy
import argparse
import torchvision.transforms as T
import cv2
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask
from src.utils.inout import load_json
import src.dataloader.hot3d.clip_util as clip_util
import tarfile
from functools import partial
import multiprocessing

inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )

def draw_visual(rgb, detections, models_info):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # img = (255*img).astype(np.uint8)
    # colors = distinctipy.get_colors(len(detections))
    colors = distinctipy.get_colors(34)
    alpha = 0.33

    for mask_idx, det in enumerate(detections):
        mask = rle_to_mask(det["segmentation"])
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2)))
        obj_id = det["category_id"]
        temp_id = obj_id - 1
        r = int(255*colors[temp_id][0])
        g = int(255*colors[temp_id][1])
        b = int(255*colors[temp_id][2])
        img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
        img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
        img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
        img[edge, :] = 255
        
        h,w,c = img.shape

        coords = np.where(mask)
        coords_list = list(zip(*coords))
        pt_yx = list(coords_list[0])
        pt_yx[0] = min(max(20, pt_yx[0]), h-20)
        pt_yx[1] = min(max(20, pt_yx[1]), w-20)

        score = det["score"]
        obj_name = models_info[str(obj_id)]["name"]
        disp_txt = f"{obj_id}_{obj_name}_{score:.2f}"
        img = cv2.putText(img, 
            text=f"{disp_txt}", 
            org=(pt_yx[1], pt_yx[0]), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1, 
            color=tuple((np.array(colors[temp_id])*128).tolist()),
            thickness=2, 
            lineType=cv2.LINE_AA)
    
    return Image.fromarray(np.uint8(img)), None


def run_visualizer_on_key(unique_key, grouped_results, clip_root_folder, models_info, output_folder, verbose):
    if verbose:
        print(f"processing {unique_key}")
    re = grouped_results[unique_key]
    scene_id = re["scene_id"]
    image_id = re["image_id"]
    unique_key = re["unique_key"]
    detections = re["detections"]

    detections = [x for x in detections if x["score"] > 0.4]

    ## load the image
    device_type = re["device_type"]
    clip_name = f"clip-{scene_id:06d}"

    tar_folder = None
    stream_id = None
    if device_type == "Aria":
        tar_folder = f"{clip_root_folder}/test_aria"
        stream_id = "214-1"
    elif device_type == "Quest3":
        tar_folder = f"{clip_root_folder}/test_quest3"
        stream_id = "1201-1"

    tar_filepath = f"{tar_folder}/{clip_name}.tar"
    tar = tarfile.open(tar_filepath, mode="r")
    frame_key = f"{image_id:06d}"
    image_np: np.ndarray = clip_util.load_image(tar, frame_key, stream_id)
    if image_np.ndim == 2:
        image_np = np.stack([image_np, image_np, image_np], axis=-1)
    rgb = Image.fromarray(np.uint8(image_np))


    rendered_img, concat_img = draw_visual(rgb=rgb, detections=detections, models_info=models_info)

    image_name = f"{scene_id:06d}_{image_id:06d}"
    rendered_img_filepath = f"{output_folder}/{image_name}.png"
    rendered_img.save(rendered_img_filepath)
    if verbose:
        print(f"rendered_img_filepath: {rendered_img_filepath}")
    return True


def run_visualizer(clip_root_folder, results_filepath, object_models_folder, clip_definitions_filepath, output_folder, filter_mode):
    clip_definitions_filepath = f"{clip_root_folder}/clip_definitions.json"            
    clip_definitions = load_json(clip_definitions_filepath)
    print(f"len clip_definitions: {len(clip_definitions)}")

    models_info_filepath = f"{object_models_folder}/models_info.json"
    models_info = load_json(models_info_filepath)
    results_payload = load_json(results_filepath)
    print(f"len results_payload: {len(results_payload)}")

    ## group the results by unique images
    grouped_results = {}
    for re in results_payload:
        scene_id = re["scene_id"]
        image_id = re["image_id"]
        device_type = str(clip_definitions[str(scene_id)]["device"])
        unique_key = f"{scene_id:06d}_{image_id:06d}"
        if unique_key not in grouped_results:
            grouped_results[unique_key] = {
                "scene_id": scene_id,
                "image_id": image_id,
                "device_type": device_type,
                "unique_key": unique_key,
                "detections": []
            }
        grouped_results[unique_key]["detections"] += [re]

    ## execute visualizer on each group of detections    
    unique_keys = sorted(grouped_results.keys())

    if filter_mode == 0:
        pass ## don't filter anything
    elif filter_mode == 1:
        unique_keys = [x for x in unique_keys if grouped_results[x]["device_type"] == "Aria"]
    elif filter_mode == 2:
        print(f"selecting images every 10 frames")
        unique_keys = unique_keys[5::10]    
    unique_keys = unique_keys[0:]
    print(f"after filtering len unique_keys: {len(unique_keys)}")

    num_workers = 32
    with multiprocessing.Pool(processes=int(num_workers)) as pool:
        call_run_visualizer_on_keys = partial(
            run_visualizer_on_key,
            grouped_results=grouped_results,
            clip_root_folder=clip_root_folder,
            models_info=models_info,
            output_folder=output_folder,
            verbose=False,
        )
        num_keys_per_process = int(np.ceil(len(unique_keys) / num_workers))
        num_keys_per_process = min(num_keys_per_process, 5)
        print(f"num_keys_per_process: {num_keys_per_process}")
        mapped_values = list(
            tqdm(
                pool.imap_unordered(call_run_visualizer_on_keys, unique_keys, chunksize=num_keys_per_process),
                total=len(unique_keys)
            )
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_root_folder", nargs="?", help="base folder to the hot3d clip")
    parser.add_argument("--results_filepath", nargs="?", help="file path containing the inference results")
    parser.add_argument("--object_models_folder", nargs="?", help="folder path to the object models (CAD models)")
    parser.add_argument("--clip_definitions_filepath", nargs="?", help="path to clip_definitions.json")
    parser.add_argument("--output_folder", nargs="?", help="folder where visualization results will be saved")
    parser.add_argument('--filter_mode', type=int, default=0, help='different filter modes to render only specific samples')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(f"args: {args}")
    vis_folder = f"{args.output_folder}/prediction_vis"
    os.makedirs(vis_folder, exist_ok=True)    
    print(f"vis_folder: {vis_folder}")

    run_visualizer(
        clip_root_folder=args.clip_root_folder,
        results_filepath=args.results_filepath,
        object_models_folder=args.object_models_folder,
        clip_definitions_filepath=args.clip_definitions_filepath,
        output_folder=vis_folder,
        filter_mode=args.filter_mode
    )