"""Setup the ECHI data for use in experiments"""

import logging

import hydra
from omegaconf import DictConfig

from train.signal_prep import get_session_tuples, segment_all_signals


def unpack(cfg):
    logging.info("Preparing the ECHI dataset")

    session_tuples = get_session_tuples(
        cfg.sessions_file, cfg.devices, datasets=cfg.dataset
    )

    directories = [
        [cfg.noisy_input_file, cfg.noisy_output_dir],
        [cfg.ref_input_file, cfg.ref_output_dir],
    ]

    for in_files, out_dir in directories:
        segment_all_signals(
            signal_template=in_files,
            output_dir_template=out_dir,
            segment_info_file=cfg.segment_info_file,
            session_tuples=session_tuples,
            seg_sample_rate=cfg.segment_sample_rate,
        )


@hydra.main(version_base=None, config_path="../config", config_name="main_eval")
def main(cfg: DictConfig) -> None:
    unpack(cfg.setup)


if __name__ == "__main__":
    main()
