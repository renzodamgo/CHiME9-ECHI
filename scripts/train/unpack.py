"""Setup the ECHI data for use in experiments"""

import logging

import hydra
from omegaconf import DictConfig

from shared.core_utils import get_session_tuples
from train.signal_prep import segment_all_signals, resample_rainbow


def unpack(cfg):
    logging.info("Preparing the ECHI dataset")

    session_tuples = get_session_tuples(
        cfg.sessions_file, cfg.devices, datasets=cfg.dataset
    )

    resample_rainbow(
        cfg.rainbow_input_file,
        cfg.rainbow_output_file,
        cfg.model_sample_rate,
        session_tuples,
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
            save_sample_rate=cfg.model_sample_rate,
            seg_sample_rate=cfg.ref_sample_rate,
        )


@hydra.main(version_base=None, config_path="../../config/train", config_name="main_ha")
def main(cfg: DictConfig) -> None:
    unpack(cfg.unpack)


if __name__ == "__main__":
    main()
