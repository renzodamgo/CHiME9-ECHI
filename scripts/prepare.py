"""Prepare ECHI data"""

import logging

import hydra
from omegaconf import DictConfig

from signal_tools import segment_signal_dir


@hydra.main(version_base=None, config_path="../config", config_name="prepare")
def main(cfg: DictConfig) -> None:
    logging.info("Preparing the ECHI dataset")

    # Segment the reference signals
    signal_dir = "/Volumes/ECHI1/chime9_echi/ref/dev/"
    csv_dir = "/Volumes/ECHI1/chime9_echi/metadata/ref/dev/"
    output_dir = "output"
    segment_signal_dir(signal_dir, csv_dir, output_dir, filter="*ha*")


if __name__ == "__main__":
    main()
