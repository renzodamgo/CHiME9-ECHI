"""Evaluate ECHI submission using Versa Scorer"""

import glob
import logging
import os
import tempfile
from pathlib import Path

import hydra
import torch
import yaml
from omegaconf import DictConfig

from signal_tools import get_session_tuples, segment_signal_dir


def load_audio_files(
    directory: str,
    selection=None,
    decimate_factor: int = 1,
):
    """Load audio files from a directory.

    Uses the Versa soundfile loader and adds a feature that allows decimating
    the file list by a given factor to allow processing only a subset of files.
    """

    from versa.scorer_shared import audio_loader_setup

    file_list = glob.glob(os.path.join(directory, "*.wav"))

    if selection is not None:
        file_list = [
            file for file in file_list if Path(file).name.startswith(selection)
        ]

    file_list.sort()
    print(file_list)
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


def evaluate_device(
    enhanced: str,
    reference: str,
    session_device_pid_tuples,
    score_config,
    results_file,
    use_gpu,
    decimate_factor: int = 1,
):
    """Run the evaluation wih versa"""

    from versa.scorer_shared import list_scoring, load_score_modules, load_summary

    if use_gpu:
        GPU_RANK = 0
        torch.cuda.set_device(GPU_RANK)
        logging.info(f"using device: cuda:{GPU_RANK}")

    # Selection set so that only loads audio files that are part of the
    # list of session, device, pids that are to be evaluated.
    selection = tuple(f"{s}.{d}.{p}" for s, d, p in session_device_pid_tuples)

    enhanced_files = load_audio_files(enhanced, selection, decimate_factor)
    reference_files = load_audio_files(reference, selection, decimate_factor)

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
            output_file=results_file,
            io="soundfile",
        )
        logging.info("Summary: {}".format(load_summary(score_info)))
    else:
        logging.info("No utterance-level scoring function is provided.")


def evaluate(cfg):
    logging.info("Running evaluate")

    signal_dir = cfg.enhanced_dir

    for device in cfg.devices:
        session_device_pid_tuples = get_session_tuples(
            cfg.sessions_file, [device], datasets=[cfg.dataset]
        )

        segment_dir = cfg.segment_dir.format(device=device)
        logging.info(f"Segment {device} signals into {segment_dir}")
        segment_signal_dir(signal_dir, cfg.csv_dir, segment_dir, filter=f"*{device}*P*")

        logging.info(f"Evaluating {device} segments")
        ref_segment_dir = cfg.ref_segment_dir.format(dataset=cfg.dataset, device=device)
        evaluate_device(
            ref_segment_dir,
            segment_dir,
            session_device_pid_tuples,
            cfg.score_config,
            cfg.results_file.format(device=device),
            cfg.use_gpu,
            decimate_factor=cfg.decimate_factor,
        )


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    evaluate(cfg.evaluate)


if __name__ == "__main__":
    main()
