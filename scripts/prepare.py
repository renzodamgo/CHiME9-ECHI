"""Prepare ECHI data"""

import logging

import hydra
from omegaconf import DictConfig

from signal_tools import segment_signal_dir


@hydra.main(version_base=None, config_path="../config", config_name="prepare")
def main(cfg: DictConfig) -> None:
    logging.info("Preparing the ECHI dataset")

    signal_dir = f"{cfg.paths.echi}/ref/dev/"
    csv_dir = f"{cfg.paths.echi}/metadata/ref/dev/"
    ha_segment_dir = f"{cfg.paths.ref_segment_dir}/ha"
    aria_segment_dir = f"{cfg.paths.ref_segment_dir}/aria"

    logging.info(f"Segmenting hearing aid reference signals into {ha_segment_dir}")
    segment_signal_dir(signal_dir, csv_dir, ha_segment_dir, filter="*ha*P*")

    logging.info(f"Segmenting aria reference signals into {aria_segment_dir}")
    segment_signal_dir(signal_dir, csv_dir, aria_segment_dir, filter="*aria*P*")


if __name__ == "__main__":
    main()
