"""Prepare ECHI data"""

import logging

import hydra
from omegaconf import DictConfig

from signal_tools import segment_signal_dir


def prepare(cfg):
    logging.info("Preparing the ECHI dataset")

    signal_dir = cfg.ref_signal_dir.format(dataset=cfg.dataset)

    for device in cfg.devices:
        output_dir = cfg.ref_segment_dir.format(dataset=cfg.dataset, device=device)
        logging.info(f"Segmenting {device} reference signals into {output_dir}")
        segment_info_dir = cfg.segment_info_dir.format(dataset=cfg.dataset)
        segment_signal_dir(
            signal_dir=signal_dir,
            segment_info_dir=segment_info_dir,
            output_dir=output_dir,
            filter=f"*{device}*P*",
        )


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    prepare(cfg.prepare)


if __name__ == "__main__":
    main()
