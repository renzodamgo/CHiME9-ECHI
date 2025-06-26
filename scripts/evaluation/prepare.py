"""Prepare ECHI submission for evaluation"""

import logging

import hydra
from omegaconf import DictConfig

from evaluation.segment_signals import get_session_tuples, segment_all_signals


def prepare(cfg):
    logging.info("Running preparation for evaluation")
    # Segment the enhanced signals

    session_tuples = get_session_tuples(
        cfg.sessions_file, cfg.devices, datasets=[cfg.dataset]
    )

    segment_all_signals(
        signal_template=cfg.enhanced_signal,
        output_dir_template=cfg.segment_dir,
        segment_info_file=cfg.segment_info_file,
        seg_sample_rate=cfg.segment_sample_rate,
        session_tuples=session_tuples,
    )


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    prepare(cfg.prepare)


if __name__ == "__main__":
    main()
