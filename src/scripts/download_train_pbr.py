import logging
import os, sys
import os.path as osp

# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import hydra
from omegaconf import DictConfig, OmegaConf


def run_download(config_dataset: DictConfig) -> None:
    os.makedirs(f"{config_dataset.tmp}", exist_ok=True)
    tmp_path = f"{config_dataset.tmp}/{config_dataset.name}.zip"

    # define command to download RGB
    command = (
        f"wget -O {tmp_path} {config_dataset.source_train_pbr} --no-check-certificate"
    )
    logging.info(f"Running {command}")
    os.system(command)

    unzip_cmd = "unzip {} -d {}".format(tmp_path, config_dataset.target_dir)
    logging.info(f"Running {unzip_cmd}")
    os.system(unzip_cmd)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="download",
)
def download(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    for dataset_name in [
        "lmo",
        "tless",
        "tudl",
        "icbin",
        "itodd",
        "hb",
        "ycbv",
    ]:
        logging.info(f"Downloading {dataset_name}")
        config_dataset = OmegaConf.create(
            {
                "name": dataset_name + "_train_pbr",
                "source_train_pbr": osp.join(
                    cfg.data.source_url, str(cfg.data.datasets[dataset_name].pbr_train)
                ),
                "target_dir": osp.join(cfg.data.root_dir, dataset_name),
                "tmp": osp.join(cfg.data.root_dir, "tmp"),
            }
        )
        run_download(config_dataset)
        logging.info(f"---" * 100)


if __name__ == "__main__":
    download()
