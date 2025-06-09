"""Evaluate ECHI submission using Versa Scorer"""

import glob
import logging
import os
import tempfile

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

from signal_tools import segment_signal_dir


def load_audio_files(directory: str, decimate_factor: int = 1):
    """Load audio files from a directory.

    Uses the Versa soundfile loader and adds a feature that allows decimating
    the file list by a given factor to allow processing only a subset of files.
    """

    file_list = glob.glob(os.path.join(directory, "*.wav"))
    file_list.sort()
    file_list = file_list[::decimate_factor]  # Decimate the list
    tmp_filelist = tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".scp"
    ).name

    with open(tmp_filelist, "w") as f:
        for file in file_list:
            name = os.path.basename(file)
            f.write(f"{name} {file}\n")

    files = audio_loader_setup(tmp_filelist, io="soundfile")
    return files


def evaluate(
    enhanced: str,
    reference: str,
    score_config,
    output_file,
    use_gpu,
    decimate_factor: int = 1,
):
    """Run the evaluation wih versa"""

    if use_gpu:
        GPU_RANK = 0
        torch.cuda.set_device(GPU_RANK)
        logging.info(f"using device: cuda:{GPU_RANK}")

    enhanced_files = load_audio_files(enhanced, decimate_factor)

    reference_files = load_audio_files(reference, decimate_factor)

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
            io="soundfile",
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

    for device in ["ha", "aria"]:
        segment_dir = f"{cfg.paths.scratch}/segments/{device}"
        logging.info(f"Segment {device} signals into {segment_dir}")
        segment_signal_dir(signal_dir, csv_dir, segment_dir, filter=f"*{device}*P*")

        logging.info(f"Evaluating {device} segments")
        evaluate(
            f"{ref_segment_dir}/{device}",
            segment_dir,
            cfg.score_config,
            cfg.output_file.format(device=device),
            cfg.use_gpu,
            decimate_factor=cfg.decimate_factor,
        )


if __name__ == "__main__":
    main()
