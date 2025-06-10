"""Prepare ECHI data"""

import logging

import hydra
from omegaconf import DictConfig

from signal_tools import segment_signal_dir


def prepare(cfg):
    logging.info("Preparing the ECHI dataset")

    logging.info(f"Segmenting hearing aid reference signals into {cfg.ha_segment_dir}")
    segment_signal_dir(cfg.signal_dir, cfg.csv_dir, cfg.ha_segment_dir, filter="*ha*P*")

    logging.info(f"Segmenting aria reference signals into {cfg.aria_segment_dir}")
    segment_signal_dir(
        cfg.signal_dir, cfg.csv_dir, cfg.aria_segment_dir, filter="*aria*P*"
    )


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    prepare(cfg.prepare)


if __name__ == "__main__":
    main()
