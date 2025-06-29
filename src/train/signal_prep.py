"""Tools for handling signals"""

import csv
import logging
from pathlib import Path

import soxr
import numpy as np
import soundfile as sf
from tqdm import tqdm


def read_wav_files_and_sum(wav_files) -> tuple[np.ndarray, int]:
    """Read a list of wav files and return their sum."""

    sum_signal = None
    fs_set = set()
    for file in wav_files:
        with open(file, "rb") as f:
            signal, fs = sf.read(f)
            fs_set.add(fs)
            if sum_signal is not None:
                if len(signal) != len(sum_signal):
                    # pad the short with zeros
                    if len(signal) < len(sum_signal):
                        signal = np.pad(signal, (0, len(sum_signal) - len(signal)))
                    else:
                        sum_signal = np.pad(
                            sum_signal, (0, len(signal) - len(sum_signal))
                        )
                sum_signal += signal
            else:
                sum_signal = signal
    if len(fs_set) != 1:
        raise ValueError(f"Inconsistent sampling rates found: {fs_set}")
    if sum_signal is None:
        raise ValueError(f"No wav files found: {wav_files}")
    fs = fs_set.pop()

    return sum_signal, fs


def wav_file_name(output_dir: Path, stem: str, index: int) -> Path:
    """Construct the wav file name based on session, device, and pid."""
    return Path(output_dir) / f"{stem}.{index:03g}.wav"


def segment_signal(
    wav_file: Path | list[Path],
    csv_file: Path,
    output_dir: Path,
    save_sample_rate: int,
    seg_sample_rate: int,
) -> None:
    """Extract speech segments from a signal"""
    logging.debug(f"Segmenting {wav_file} {csv_file}")
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(csv_file, "r") as f:
        segments = list(csv.DictReader(f, fieldnames=["index", "start", "end"]))

    # check if any files missing:
    files_missing = False
    for segment in segments:
        expected_files = wav_file_name(
            output_dir, Path(csv_file).stem, int(segment["index"])
        )
        if not expected_files.exists():
            files_missing = True
            break
    if not files_missing:
        logging.debug(f"All segments already exist in {output_dir}")
        return

    if isinstance(wav_file, list):
        signal, fs = read_wav_files_and_sum(wav_file)
    else:
        with open(wav_file, "rb") as f:
            signal, fs = sf.read(f)

    if fs != save_sample_rate:
        signal = soxr.resample(signal, fs, save_sample_rate)

    logging.debug(f"Will generate {len(segments)} segments from {wav_file}")
    sample_scalar = save_sample_rate / seg_sample_rate

    for segment in segments:
        index = int(segment["index"])
        start_sample = int(int(segment["start"]) * sample_scalar)
        end_sample = int(int(segment["end"]) * sample_scalar)

        output_file = wav_file_name(output_dir, Path(csv_file).stem, index)
        if output_file.exists():
            logging.debug(f"Segment {output_file} already exists, skipping")
            continue
        if end_sample > len(signal):
            logging.warning(f"Segment {output_file} exceeds signal length. Skipping.")
            continue
        signal_segment = signal[start_sample:end_sample]

        with open(output_file, "wb") as f:
            sf.write(f, signal_segment, samplerate=save_sample_rate)


def csv_to_pid_wav(name: str) -> str:
    """Replace .wav with .csv"""
    return ".".join(name.split(".")[:-1]) + ".csv"


def segment_all_signals(
    signal_template,
    output_dir_template,
    segment_info_file,
    session_tuples,
    save_sample_rate,
    seg_sample_rate,
):
    for session, device, pid in tqdm(session_tuples):
        dataset = session.split("_")[0]
        # Segment the reference signal for this PID
        output_dir = output_dir_template.format(
            dataset=dataset, device=device, segment_type="individual"
        )

        logging.debug(f"Segmenting {device}, {pid} reference signals into {output_dir}")
        wav_file = signal_template.format(
            dataset=dataset, session=session, device=device, pid=pid
        )
        csv_file = segment_info_file.format(
            dataset=dataset, session=session, device=device, pid=pid
        )

        if not Path(csv_file).exists():
            logging.warning(f"WARNING: csv file not found at {csv_file}")
            continue

        segment_signal(
            wav_file, csv_file, output_dir, save_sample_rate, seg_sample_rate
        )


def resample_rainbow(
    signal_template,
    output_template,
    save_sample_rate,
    session_tuples,
):
    dataset_pids = list(set((x.split("_")[0], y) for x, _, y in session_tuples))

    for dataset, pid in dataset_pids:
        sig_file = Path(signal_template.format(dataset=dataset, pid=pid))
        if not sig_file.exists():
            logging.warning(f"Rainbow file {sig_file} not found")
            continue

        output_file = Path(output_template.format(dataset=dataset, pid=pid))
        if output_file.exists():
            continue

        with open(sig_file, "rb") as file:
            signal, fs = sf.read(file)

        if fs != save_sample_rate:
            signal = soxr.resample(signal, fs, save_sample_rate)

        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True)

        with open(output_file, "wb") as f:
            sf.write(f, signal, samplerate=save_sample_rate)
