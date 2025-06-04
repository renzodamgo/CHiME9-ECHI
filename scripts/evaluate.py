"""Evaluate ECHI submission using Versa Scorer"""

import logging

import hydra
import torch
import yaml
from omegaconf import DictConfig
from versa.scorer_shared import (
    audio_loader_setup,
    list_scoring,
    load_score_modules,
    load_summary,
)


@hydra.main(version_base=None, config_path="../config", config_name="evaluate")
def main(cfg: DictConfig):
    io_type = "dir"
    # In case of using `local` backend, all GPU will be visible to all process.
    if cfg.use_gpu:
        GPU_RANK = 0
        torch.cuda.set_device(GPU_RANK)
        logging.info(f"using device: cuda:{GPU_RANK}")

    enhanced_files = audio_loader_setup(cfg.enhanced, io_type)

    # find reference file

    reference_files = audio_loader_setup(cfg.reference, io_type)

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

    with open(cfg.score_config, "r", encoding="utf-8") as f:
        score_config = yaml.full_load(f)

    score_modules = load_score_modules(
        score_config,
        use_gt=(True if reference_files is not None else False),
        use_gt_text=False,
        use_gpu=cfg.use_gpu,
    )

    if len(score_modules) > 0:
        score_info = list_scoring(
            enhanced_files,
            score_modules,
            reference_files,
            text_info=None,
            output_file=cfg.output_file,
            io="dir",
        )
        logging.info("Summary: {}".format(load_summary(score_info)))
    else:
        logging.info("No utterance-level scoring function is provided.")


if __name__ == "__main__":
    main()
