"""Setup the ECHI data for use in experiments"""

import logging

import hydra
from omegaconf import DictConfig

from evaluation.segment_signals import get_session_tuples, segment_all_signals


def setup(cfg):
    logging.info("Preparing the ECHI dataset")

    session_tuples = get_session_tuples(
        cfg.sessions_file, cfg.devices, datasets=cfg.dataset
    )

    segment_all_signals(
        signal_template=cfg.input_signal_file,
        output_dir_template=cfg.output_segment_dir,
        segment_info_file=cfg.segment_info_file,
        session_tuples=session_tuples,
        seg_sample_rate=cfg.segment_sample_rate,
    )


@hydra.main(
    version_base=None, config_path="../../config/evaluation", config_name="main_eval"
)
def main(cfg: DictConfig) -> None:
    setup(cfg.setup)


if __name__ == "__main__":
    main()
