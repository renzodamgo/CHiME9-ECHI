"""Mock up of the enhance stage.

Processes the dev or eval set by simply producing single channel versions of the
noisy inputs. The output signals are in the correct format for evaluation but
no speaker extraction or enhancement is being performed so results will be poor.
"""

import logging
import os

import hydra
import numpy as np
import soundfile as sf
from librosa import resample
from omegaconf import DictConfig
from tqdm import tqdm

from evaluation.segment_signals import get_session_tuples


def dummy_enhance_for_session(
    signal_filename_template,
    enhanced_signal_filename_template,
    dataset,
    session,
    device,
    pid,
    required_samplerate,
):
    signal_filename = signal_filename_template.format(
        session=session, dataset=dataset, device=device
    )

    # For the dummy enhance the output for each speaker is the same
    # so we just write one output file and symlink it for each PID
    outfile = enhanced_signal_filename_template.format(
        dataset=dataset, session=session, device=device, pid="xxx"
    )

    logging.info(f"Processing session {session} for device {device} with PID {pid}")

    if not os.path.exists(outfile):
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

        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, "wb") as f:
            sf.write(f, processed_signal, required_samplerate)

    outfile_pid = enhanced_signal_filename_template.format(
        dataset=dataset, session=session, device=device, pid=pid
    )
    if not os.path.exists(outfile_pid):
        os.symlink(os.path.basename(outfile), outfile_pid)


def dummy_enhance(cfg):
    logging.info("Processing the noisy signals")

    session_device_pid_tuples = get_session_tuples(
        cfg.sessions_file, cfg.devices, datasets=[cfg.dataset]
    )

    for session, device, pid in tqdm(
        session_device_pid_tuples, desc="Processing sessions..."
    ):
        dummy_enhance_for_session(
            cfg.noisy_signal,
            cfg.enhanced_signal,
            cfg.dataset,
            session,
            device,
            pid,
            cfg.sample_rate,
        )


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    dummy_enhance(cfg.inference)


if __name__ == "__main__":
    main()
