"""Mock up of the inference stage.

Processes the dev or eval set by simply producing single channel versions of the
noisy inputs. The output signals are in the correct format for evaluation but
no speaker extraction or enhancement is being performed so results will be poor.
"""

import csv
import itertools
import logging
import os

import hydra
import numpy as np
import soundfile as sf
from librosa import resample
from omegaconf import DictConfig
from tqdm import tqdm

POSITIONS = ["pos1", "pos2", "pos3", "pos4"]


def dummy_inference_for_session(
    signal_filename,
    enhanced_signal_filename_template,
    dataset,
    session,
    device,
    pids,
    required_samplerate,
):
    # For the dummy inference the output for each speaker is the same
    # so we just write one output file and symlink it for each PID
    outfile = enhanced_signal_filename_template.format(
        session=session, dataset=dataset, device=device, pid="xxx"
    )

    if os.path.exists(outfile):
        logging.info(f"Output file {outfile} already exists, skipping processing.")
        return

    logging.info(f"Processing session {session} for device {device} with PIDs {pids}")

    with open(signal_filename, "rb") as f:
        signal, samplerate = sf.read(f)

    if device == "ha":
        # For ha, we just average front left and right channels
        processed_signal = np.mean(signal[:, 0:2], axis=1)
    elif device == "aria":
        # For aria, we just take the nose-bridge channel
        processed_signal = signal[:, 3]
    else:
        raise ValueError(f"Unknown device {device}")

    if samplerate != required_samplerate:
        processed_signal = resample(
            processed_signal, orig_sr=samplerate, target_sr=required_samplerate
        )

    # For the dummy inference the output for each speaker is the same
    # so we just write one output file and symlink it for each PID

    outfile = enhanced_signal_filename_template.format(
        session=session, dataset=dataset, device=device, pid="xxx"
    )
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "wb") as f:
        sf.write(f, processed_signal, required_samplerate)

    for pid in pids:
        outfile_pid = enhanced_signal_filename_template.format(
            session=session, dataset=dataset, device=device, pid=pid
        )
        outfile_name = os.path.basename(outfile)
        os.symlink(outfile_name, outfile_pid)


def dummy_inference(cfg):
    logging.info("Processing the noisy signals")

    with open(cfg.sessions_file, "r") as f:
        sessions = list(csv.DictReader(f))

    # Only use the session for the specified dataset
    sessions = [s for s in sessions if s["session"].startswith(cfg.dataset)]

    session_device_pairs = list(itertools.product(sessions, cfg.devices))
    for session, device in tqdm(session_device_pairs, desc="Processing sessions..."):
        device_pos = "pos" + session[f"{device}_pos"]
        pids = [session[pos] for pos in POSITIONS if pos != device_pos]
        noisy_signal_filename = cfg.noisy_signal.format(
            session=session["session"], dataset=cfg.dataset, device=device
        )
        dummy_inference_for_session(
            noisy_signal_filename,
            cfg.enhanced_signal,
            cfg.dataset,
            session["session"],
            device,
            pids,
            cfg.sample_rate,
        )


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    dummy_inference(cfg.inference)


if __name__ == "__main__":
    main()
