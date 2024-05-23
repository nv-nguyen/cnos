from omegaconf import DictConfig, OmegaConf
import logging
import os
import hydra

# set level logging
logging.basicConfig(level=logging.INFO)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="download",
)
def download(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    logging.info(f"Downloading {cfg.dataset_name}")
    local_dir = os.path.join(cfg.data.root_dir, cfg.dataset_name)

    download_cmd = f"huggingface-cli download bop-benchmark/datasets --include {cfg.dataset_name}/*train_pbr* --local-dir {cfg.data.root_dir} --repo-type=dataset"
    logging.info(f"Running {download_cmd}")
    os.system(download_cmd)
    logging.info(f"Dataset downloaded to {local_dir}")

    # if zip files is splitted into z01, z02, ... parts, merge them first
    files = os.listdir(local_dir)
    if any([f.endswith(".z01") for f in files]):
        logging.info("Merging zip files...")
        name_file = [f for f in files if f.endswith(".z01")][0]
        name = name_file.split(".")[0]
        merge_cmd = f"zip -s0 {local_dir}/{name}.zip --out {local_dir}/{name}_all.zip"
        logging.info(f"Running {merge_cmd}")
        os.system(merge_cmd)
        logging.info("Merging done")

        # remove splitted files
        remove_cmd = f"rm {local_dir}/{name}.zip"
        logging.info(f"Running {remove_cmd}")
        os.system(remove_cmd)
        logging.info(f"Removed {local_dir}/{name}.zip")

    # unzip the dataset
    unzip_cmd = f"unzip {local_dir}/{name}_all.zip -d {local_dir}"
    logging.info(f"Running {unzip_cmd}")
    os.system(unzip_cmd)
    logging.info(f"Dataset downloaded to {local_dir}")


if __name__ == "__main__":
    download()
