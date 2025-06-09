"""Evaluate ECHI submission using Versa Scorer"""

import logging

import hydra
import torch
import yaml
from omegaconf import DictConfig

from signal_tools import segment_signal_dir


def evaluate(enhanced, reference, score_config, output_file, use_gpu):
    """Run the evaluation wih versa"""
    from versa.scorer_shared import (  # TODO: here for testing - move to top
        audio_loader_setup,
        list_scoring,
        load_score_modules,
        load_summary,
    )

    if use_gpu:
        GPU_RANK = 0
        torch.cuda.set_device(GPU_RANK)
        logging.info(f"using device: cuda:{GPU_RANK}")

    enhanced_files = audio_loader_setup(enhanced, io="dir")

    # find reference file

    reference_files = audio_loader_setup(reference, io="dir")

    # Get and divide list
    if len(enhanced_files) == 0:
        raise FileNotFoundError("Not found any generated audio files.")
    if len(enhanced_files) > len(reference_files):
        raise ValueError(
            "#groundtruth files are less than #generated files "
            f"(#gen={len(enhanced_files)} vs. #gt={len(reference_files)}). "
            "Please check the groundtruth directory."
        )

    logging.info("The number of utterances = %d" % len(enhanced_files))

    with open(score_config, "r", encoding="utf-8") as f:
        score_config = yaml.full_load(f)

    score_modules = load_score_modules(
        score_config,
        use_gt=(True if reference_files is not None else False),
        use_gt_text=False,
        use_gpu=use_gpu,
    )

    if len(score_modules) > 0:
        score_info = list_scoring(
            enhanced_files,
            score_modules,
            reference_files,
            text_info=None,
            output_file=output_file,
            io="dir",
        )
        logging.info("Summary: {}".format(load_summary(score_info)))
    else:
        logging.info("No utterance-level scoring function is provided.")


@hydra.main(version_base=None, config_path="../config", config_name="evaluate")
def main(cfg: DictConfig):
    logging.info("Running evaluate")

    signal_dir = cfg.submission
    csv_dir = f"{cfg.paths.echi}/metadata/ref/dev/"
    ref_segment_dir = cfg.paths.ref_segment_dir

    ha_segment_dir = f"{cfg.paths.scratch}/segments/ha"
    logging.info(f"Hearing aid signals into {ha_segment_dir}")
    segment_signal_dir(signal_dir, csv_dir, ha_segment_dir, filter="*ha*P*")

    aria_segment_dir = f"{cfg.paths.scratch}/segments/aria"
    logging.info(f"Aria signals into {aria_segment_dir}")
    segment_signal_dir(signal_dir, csv_dir, aria_segment_dir, filter="*aria*P*")

    logging.info("Evaluating hearing aid segments")
    evaluate(
        f"{ref_segment_dir}/ha",
        ha_segment_dir,
        cfg.score_config,
        cfg.output_file,
        cfg.use_gpu,
    )

    logging.info("Evaluating aria segments")
    evaluate(
        f"{ref_segment_dir}/aria",
        aria_segment_dir,
        cfg.score_config,
        cfg.output_file,
        cfg.use_gpu,
    )


if __name__ == "__main__":
    main()
