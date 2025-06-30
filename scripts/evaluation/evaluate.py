"""Evaluate ECHI submission using Versa Scorer"""

import glob
import itertools
import logging
import os
import tempfile
from pathlib import Path

import hydra
import torch
import yaml
from omegaconf import DictConfig

from evaluation.segment_signals import get_session_tuples


def load_audio_files(
    file_list: list[str],
    selection=None,
    batch=(1, 1),
):
    """Load audio files from a directory.

    Uses the Versa soundfile loader and adds a feature that allows decimating
    the file list by a given factor to allow processing only a subset of files.
    """

    from versa.scorer_shared import audio_loader_setup

    if selection is not None:
        file_list = [
            file for file in file_list if Path(file).name.startswith(selection)
        ]

    file_list.sort()

    file_list = file_list[batch[0] - 1 :: batch[1]]  # Form list for batch
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
    signal_id,
    session_device_pid_tuples,
    score_config,
    results_file,
    use_gpu,
    batch=(1, 1),
):
    """Run the evaluation wih versa"""

    from versa.scorer_shared import list_scoring, load_score_modules, load_summary

    if use_gpu:
        GPU_RANK = 0
        torch.cuda.set_device(GPU_RANK)
        logging.info(f"using device: cuda:{GPU_RANK}")

    # Selection set so that only loads audio files that are part of the
    # list of session, device, pids that are to be evaluated.
    selection = tuple(
        signal_id.format(session=s, device=d, pid=p)
        for s, d, p in session_device_pid_tuples
    )

    # Find all the enhanced and reference files...
    enhanced_file_list = glob.glob(os.path.join(enhanced, "*.wav"))
    reference_file_list = glob.glob(os.path.join(reference, "*.wav"))

    # ...keep only enhanced files for which we have a reference file
    reference_names = set(os.path.basename(f) for f in reference_file_list)
    enhanced_file_list = [
        f for f in enhanced_file_list if os.path.basename(f) in reference_names
    ]

    # If the enhanced_file_list is shorter than the reference_file_list
    # then report and error, throw away the extra reference files and continue.
    if len(enhanced_file_list) < len(reference_file_list):
        logging.error(
            "The number of enhanced files is less than the number of reference files. "
            "This may lead to missing scores for some reference files."
        )
        enhanced_names = set(os.path.basename(f) for f in enhanced_file_list)
        reference_file_list = [
            f for f in reference_file_list if os.path.basename(f) in enhanced_names
        ]

    # Now that the lists are the same length, load the audio files
    enhanced_files = load_audio_files(enhanced_file_list, selection, batch)
    reference_files = load_audio_files(reference_file_list, selection, batch)

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


def validate_batch_param(batch):
    """Validate batch parameter."""
    if len(batch) != 2 or batch[0] > batch[1]:
        raise ValueError(
            "Batch parameter must be a tuple of two integers (i, N) "
            "where index i <= batch size N. Got: {}".format(batch)
        )


def add_batch_to_results_file_name(results_file, batch):
    """Add batch information to the results file name."""
    if batch == (1, 1):  # No batching, return original file name
        return results_file
    batch_str = f"{batch[0]}_{batch[1]}"
    base, ext = os.path.splitext(results_file)
    results_file_with_batch = f"{base}.batch_{batch_str}{ext}"
    return results_file_with_batch


def evaluate(cfg):
    logging.info("Running evaluate")

    batch = (cfg.batch, cfg.n_batches)  # ie., i of N
    validate_batch_param(batch)

    for device, segment_type in itertools.product(cfg.devices, cfg.segment_types):
        logging.info(f"Evaluating {device} with {segment_type} segments")

        results_file = cfg.results_file.format(
            dataset=cfg.dataset, device=device, segment_type=segment_type
        )
        results_file = add_batch_to_results_file_name(results_file, batch)

        # Create directory if it does not exist
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        session_device_pid_tuples = get_session_tuples(
            cfg.sessions_file, [device], datasets=[cfg.dataset]
        )

        segment_dir = cfg.segment_dir.format(
            dataset=cfg.dataset, device=device, segment_type=segment_type
        )
        ref_segment_dir = cfg.ref_segment_dir.format(
            dataset=cfg.dataset, device=device, segment_type=segment_type
        )
        try:
            evaluate_device(
                segment_dir,
                ref_segment_dir,
                cfg.signal_id,
                session_device_pid_tuples,
                cfg.score_config,
                results_file,
                cfg.use_gpu,
                batch=batch,
            )
        except Exception as e:
            logging.error(
                f"Error evaluating {device} with {segment_type} segments: {e}"
            )
            continue


@hydra.main(
    version_base=None, config_path="../../config/evaluation", config_name="main"
)
def main(cfg: DictConfig):
    evaluate(cfg.evaluate)


if __name__ == "__main__":
    main()
