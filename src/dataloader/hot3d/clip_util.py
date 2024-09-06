#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np
import torch
import trimesh

# HT toolkit (https://github.com/facebookresearch/hand_tracking_toolkit)
from hand_tracking_toolkit import (
    camera,
    math_utils,
    visualization,
)
from hand_tracking_toolkit.dataset import (
    decode_hand_pose,
    HandShapeCollection,
    HandSide,
)
from hand_tracking_toolkit.hand_models.mano_hand_model import (
    MANOHandModel,
)
from hand_tracking_toolkit.hand_models.umetrack_hand_model import (
    from_json as from_umetrack_hand_model_json,
)


def get_number_of_frames(tar: Any) -> int:
    """Returns the number of frames in a clip.

    Args:
        tar: File handler of an open tar file with clip data.
    Returns:
        Number of frames in the given tar file.
    """

    max_frame_id = -1
    for x in tar.getnames():
        if x.endswith(".info.json"):
            frame_id = int(x.split(".info.json")[0])
            if frame_id > max_frame_id:
                max_frame_id = frame_id
    return max_frame_id + 1


def load_image(
    tar: Any, frame_key: str, stream_key: str, dtype: Any = np.uint8
) -> np.ndarray:
    """Loads an image from the specified frame and stream of a clip.

    Args:
        tar: File handler of an open tar file with clip data.
        frame_key: Key of the frame from which to load the image.
        stream_key: Key of the stream from which to load the image.
        dtype: Desired type of the loaded image.
    Returns:
        Numpy array with the loaded image.
    """

    file = tar.extractfile(f"{frame_key}.image_{stream_key}.jpg")
    return imageio.imread(file).astype(dtype)


def load_info(tar: Any, frame_key: str) -> Dict[str, Any]:
    """Loads meta info of the specified frame.

    Args:
        tar: File handler of an open tar file with clip data.
        frame_key: Key of the frame from which to load the image.
    Returns:
        Dictionary with meta info of the frame.
    """

    return json.load(tar.extractfile(f"{frame_key}.info.json"))


def load_cameras(
    tar: Any,
    frame_key: str,
) -> Tuple[Dict[str, camera.CameraModel], Dict[str, np.ndarray]]:
    """Loads cameras for all image streams in a specified frame of a clip.

    Args:
        tar: File handler of an open tar file with clip data.
        frame_key: Key of the frame for which to load the cameras.
    Returns:
        A dictionary mapping a stream key to a camera model.
    """

    cameras_raw = json.load(tar.extractfile(f"{frame_key}.cameras.json"))

    cameras = {}
    Ts_device_from_camera = {}
    for stream_key, camera_raw in cameras_raw.items():
        cameras[stream_key] = camera.from_json(camera_raw)
        Ts_device_from_camera[stream_key] = se3_from_dict(
            camera_raw["calibration"]["T_device_from_camera"]
        )

    return cameras, Ts_device_from_camera


def load_object_annotations(
    tar: Any,
    frame_key: str,
) -> Optional[Dict[str, Any]]:
    """Loads object annotations for a specified frame of a clip.

    Args:
        tar: File handler of an open tar file with clip data.
        frame_key: Key of the frame for which to load the annotations.
    Returns:
        A dictionary with object annotations.
    """

    filename = f"{frame_key}.objects.json"
    if filename in tar.getnames():
        return json.load(tar.extractfile(filename))
    else:
        # Annotations are not provided for test clips.
        return None


def load_hand_annotations(
    tar: Any,
    frame_key: str,
) -> Optional[Dict[str, Any]]:
    """Loads hand annotations for a specified frame of a clip.

    Args:
        tar: File handler of an open tar file with clip data.
        frame_key: Key of the frame for which to load the annotations.
    Returns:
        A dictionary with hand annotations. Poses are provided in two
        formats: UmeTrack and MANO.
    """

    filename = f"{frame_key}.hands.json"
    if filename in tar.getnames():
        return json.load(tar.extractfile(filename))
    else:
        # Annotations are not provided for test clips.
        return None


def load_hand_shape(tar: Any) -> Optional[HandShapeCollection]:
    """Loads hand shape for a specified clip.

    Args:
        tar: File handler of an open tar file with clip data.
    Returns:
        Hand shape in two formats: UmeTrack and MANO.
    """

    filename = "__hand_shapes.json__"
    if filename in tar.getnames():
        shape_params_dict = json.load(tar.extractfile(filename))
        return HandShapeCollection(
            mano_beta=torch.tensor(shape_params_dict["mano"]),
            umetrack=from_umetrack_hand_model_json(shape_params_dict["umetrack"]),
        )
    else:
        # Hand shapes are not provided for some test clips.
        return None


def get_hand_meshes(
    hands: Dict[str, Any],
    hand_shape: HandShapeCollection,
    hand_type: str = "umetrack",
    mano_model: Optional[MANOHandModel] = None,
) -> Dict[HandSide, trimesh.Trimesh]:
    """Provides hand meshes of specified shape and poses.

    Args:
        hands: Hand annotations (including hand poses).
        hand_shape: Hand shape.
        hand_type: Hand type ("umetrack" or "mano").
        mano_model: MANO hand model (needs to be provided
            if hand_type == "mano").
    Returns:
        Triangular meshes of left and/or right hands.
    """

    if hand_type == "mano" and mano_model is None:
        raise ValueError("MANO hand model is missing.")

    hand_poses = decode_hand_pose(hands)

    meshes: Dict[HandSide, trimesh.Trimesh] = {}
    for hand_side, hand_pose in hand_poses.items():
        assert mano_model is not None
        _, hand_verts, hand_faces = visualization.get_keypoints_and_mesh(
            hand_pose=hand_pose,
            hand_shape=hand_shape,
            mano_model=mano_model,
            pose_type=hand_type,
        )

        meshes[hand_side] = trimesh.Trimesh(
            vertices=hand_verts,
            faces=hand_faces,
            process=False,
        )

    return meshes


def load_mesh(
    path: str,
) -> trimesh.Trimesh:
    """Loads a 3D mesh model from a specified path.

    Args:
        path: Path to the model to load.
    Returns.
        Loaded mesh.
    """

    # Load the scene.
    scene = trimesh.load_mesh(
        path,
        process=True,
        merge_primitives=True,
        skip_materials=True,
    )

    # Represent the scene by a single mesh.
    mesh = scene.dump(concatenate=True)

    # Make sure there are no large triangles (the rasterizer
    # from hand_tracking_toolkit becomes slow if some triangles
    # are much larger than others)
    mesh = subdivide_mesh(mesh)

    # Clean the mesh.
    mesh.process(validate=True)

    return mesh


def subdivide_mesh(
    mesh: trimesh.Trimesh,
    max_edge: float = 0.005,
    max_iters: int = 50,
    debug: bool = False,
):
    """Subdivides mesh such as all edges are shorter than a threshold.

    Args:
        mesh: Mesh to subdivide.
        max_edge: Maximum allowed edge length in meters (note that this may
            not be reachable if max_iters is too low).
        max_iters: Number of subdivision iterations.
    Returns.
        Subdivided mesh.
    """

    new_vertices, new_faces = trimesh.remesh.subdivide_to_size(
        mesh.vertices,
        mesh.faces,
        max_edge,
        max_iter=max_iters,
    )
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

    if debug:
        print(f"Remeshing: {len(mesh.vertices)} -> {len(new_mesh.vertices)}")

    return new_mesh


def convert_to_pinhole_camera(
    camera_model: camera.CameraModel, focal_scale: float = 1.0
) -> camera.CameraModel:
    """Converts a camera model to a pinhole version.

    Args:
        camera_model: Input camera model.
        focal_scale: Focal scaling factor (can be used to contol
            the portion of an original fisheye image that is seen in
            the resulting pinhole camera).
    Returns:
        Pinhole camera model.
    """

    return camera.PinholePlaneCameraModel(
        width=camera_model.width,
        height=camera_model.height,
        f=[camera_model.f[0] * focal_scale, camera_model.f[1] * focal_scale],
        c=camera_model.c,
        distort_coeffs=[],
        T_world_from_eye=camera_model.T_world_from_eye,
    )


def se3_from_dict(se3_dict: Dict[str, Any]) -> np.ndarray:
    """Converts a dictionary to an 4x4 SE3 transformation matrix.

    Args:
        se3_dict: Dictionary with items "quaternion_wxyz" and
            "translation_xyz".
    Returns:
        4x4 numpy array with a 4x4 SE3 transformation matrix.
    """

    return math_utils.quat_trans_to_matrix(
        *se3_dict["quaternion_wxyz"],
        *se3_dict["translation_xyz"],
    )


def stack_images(images: List[np.ndarray]) -> np.ndarray:
    """Horizontally stack a list of images.

    Args:
        images: List of images to stack.
    Returns:
        Input images horizontally stacked into a single image
        (if the height of the images is different, all are
        resized to the smallest height).
    """

    # Make sure all images have the same height before
    # horizontally stacking them.
    min_image_height = images[0].shape[0]
    all_same_height = True
    for image in images[1:]:
        if image.shape[0] < min_image_height:
            min_image_height = image.shape[0]
            all_same_height = False

    # Potentially resize the images (to the smallest height).
    if not all_same_height:
        for image_id, image in enumerate(images):
            scale = min_image_height / image.shape[0]
            images[image_id] = cv2.resize(
                image, (int(scale * image.shape[1]), min_image_height)
            )

    return np.concatenate(images, axis=1)


def vis_mask_contours(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    """Overlays mask contour on top of an image.

    Args:
        image: Base image.
        mask: Mask whose contour will be overlaid on the image.
        color: Color of the contour.
        thickness: Thickness of the contour.
    Returns:
        Image overlaid with the mask contour.
    """

    contours = cv2.findContours(
        mask.astype(np.uint8),
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )[0]

    return cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)


def encode_binary_mask_rle(mask: np.ndarray) -> Dict[str, Any]:
    """Encodes a binary mask using Run-Length Encoding (RLE).

    Args:
        mask: An np.ndarray with the binary mask.
    Returns:
        The encoded mask.
    """

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    pixels = np.concatenate([[0], mask.flatten(), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return {"height": mask.shape[0], "width": mask.shape[1], "rle": runs}


def decode_binary_mask_rle(data: Dict[str, Any]) -> np.ndarray:
    """Decodes a binary mask that was encoded using `encode_binary_mask_rle`.

    Args:
        data: RLE-encoded mask (output of `encode_binary_mask_rle`).
    Returns:
        The decoded mask represented as an np.ndarray.
    """

    starts = np.asarray(data["rle"][0:][::2]) - 1
    ends = starts + np.asarray(data["rle"][1:][::2])
    mask = np.zeros(data["height"] * data["width"], dtype=np.bool)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = True

    return mask.reshape((data["height"], data["width"]))