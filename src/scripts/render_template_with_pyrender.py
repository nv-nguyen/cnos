import os
import numpy as np
from tqdm import tqdm
import time
from omegaconf import DictConfig, OmegaConf
from functools import partial
from src.poses.utils import get_obj_poses_from_template_level
import multiprocessing
import logging
import os.path as osp
import logging
import hydra


# set level logging
logging.basicConfig(level=logging.INFO)


def call_pyrender(
    idx_obj,
    list_cad_path,
    list_output_dir,
    obj_pose_path,
    disable_output,
    gpus_devices,
):
    output_dir = list_output_dir[idx_obj]
    cad_path = list_cad_path[idx_obj]
    if os.path.exists(
        output_dir
    ):  # remove first to avoid the overlapping folder of blender proc
        os.system("rm -r {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # command = f"blenderproc run ./src/poses/blenderproc.py {cad_path} {obj_pose_path} {output_dir} {gpus_devices}"
    command = f"python -m src.poses.pyrender {cad_path} {obj_pose_path} {output_dir} {gpus_devices}"
    if disable_output:
        command += " true"
    else:
        command += " false"
    os.system(command)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="download",
)
def render(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    root_save_dir = osp.join(cfg.data.root_dir, "templates_pyrender")
    template_poses = get_obj_poses_from_template_level(
        level=cfg.level, pose_distribution="all"
    )
    template_poses[:, :3, 3] *= 0.4  # zoom to object

    bop23_datasets = [
        "lmo",
        "tless",
        "tudl",
        "icbin",
        "itodd",
        "hb",
        "ycbv",
    ]
    if cfg.dataset_name is None:
        datasets = bop23_datasets
    else:
        datasets = [cfg.dataset_name]
    for dataset_name in datasets:
        dataset_save_dir = osp.join(root_save_dir, dataset_name)
        logging.info(f"Rendering templates for {dataset_name}")
        os.makedirs(dataset_save_dir, exist_ok=True)
        obj_pose_path = f"{dataset_save_dir}/template_poses.npy"
        np.save(obj_pose_path, template_poses)

        if dataset_name in ["tless"]:
            cad_dir = os.path.join(cfg.data.root_dir, dataset_name, "models/models_cad")
            if not os.path.exists(cad_dir):
                cad_dir = os.path.join(cfg.data.root_dir, dataset_name, "models_cad")
        else:
            cad_dir = os.path.join(cfg.data.root_dir, dataset_name, "models/models")
            if not os.path.exists(cad_dir):
                cad_dir = os.path.join(cfg.data.root_dir, dataset_name, "models")
        cad_paths = []
        output_dirs = []
        object_ids = sorted(
            [
                int(name[4:][:-4])
                for name in os.listdir(cad_dir)
                if name.endswith(".ply")
            ]
        )
        for object_id in object_ids:
            cad_paths.append(
                os.path.join(
                    cad_dir,
                    "obj_{:06d}.ply".format(object_id),
                )
            )
            output_dirs.append(
                os.path.join(
                    dataset_save_dir,
                    f"obj_{object_id:06d}",
                )
            )
        os.makedirs(dataset_save_dir, exist_ok=True)

        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
        start_time = time.time()
        pool = multiprocessing.Pool(processes=int(cfg.num_workers))
        call_pyrender_with_index = partial(
            call_pyrender,
            list_cad_path=cad_paths,
            list_output_dir=output_dirs,
            obj_pose_path=obj_pose_path,
            disable_output=cfg.disable_output,
            gpus_devices=cfg.gpus,
        )
        mapped_values = list(
            tqdm(
                pool.imap_unordered(call_pyrender_with_index, range(len(object_ids))),
                total=len(object_ids),
            )
        )
        finish_time = time.time()
        logging.info(
            f"Total time to render templates for dataset_name: {finish_time - start_time}"
        )


if __name__ == "__main__":
    render()
