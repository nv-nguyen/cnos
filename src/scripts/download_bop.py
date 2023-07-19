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
    cad_tmp_path = f"{config_dataset.tmp}/{config_dataset.name}_cad.zip"

    # define command to download RGB
    command = f"wget -O {tmp_path} {config_dataset.source_test} --no-check-certificate"
    logging.info(f"Running {command}")
    os.system(command)

    # define command to download CAD models
    command = (
        f"wget -O {cad_tmp_path} {config_dataset.source_cad}  --no-check-certificate"
    )
    logging.info(f"Running {command}")
    os.system(command)

    unzip_cmd = "unzip {} -d {}".format(tmp_path, config_dataset.target_dir)
    logging.info(f"Running {unzip_cmd}")
    os.system(unzip_cmd)

    unzip_cmd = "unzip {} -d {}/models".format(cad_tmp_path, config_dataset.target_dir)
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
                "name": dataset_name,
                "source_test": osp.join(
                    cfg.data.source_url, str(cfg.data.datasets[dataset_name].test)
                ),
                "source_cad": osp.join(
                    cfg.data.source_url,
                    str(cfg.data.datasets[dataset_name].cad),
                ),
                "target_dir": osp.join(cfg.data.root_dir, dataset_name),
                "tmp": osp.join(cfg.data.root_dir, "tmp"),
            }
        )
        run_download(config_dataset)
        logging.info(f"---" * 100)


if __name__ == "__main__":
    download()
