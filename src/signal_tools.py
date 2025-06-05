"""Tools for handling signals"""

import csv
import logging
from pathlib import Path
from typing import Callable, Optional

import soundfile as sf
from tqdm import tqdm


def segment_signal(wav_file: Path, csv_file: Path, output_dir: Path) -> None:
    """Extract speech segments from a signal"""
    logging.debug(f"Segmenting {wav_file} {csv_file}")
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(wav_file, "rb") as f:
        signal, fs = sf.read(f)

    with open(csv_file, "r") as f:
        segments = list(csv.DictReader(f, fieldnames=["start", "end"]))

    for index, segment in enumerate(segments, start=1):
        output_file = Path(output_dir) / f"{csv_file.stem}.{index:03g}.wav"
        start_sample = int(segment["start"])
        end_sample = int(segment["end"])
        signal_segment = signal[start_sample:end_sample]
        with open(output_file, "wb") as f:
            sf.write(f, signal_segment, samplerate=fs)


def csv_to_wav(name: str) -> str:
    """Replace .csv with .wav"""
    return ".".join(name.split(".")[:-1]) + ".wav"


def segment_signal_dir(
    signals_dir: Path | str,
    csv_dir: Path | str,
    output_dir: Path | str,
    filter: str = "*",
    translate: Optional[Callable[[str], str]] = None,
) -> None:
    """Extract speech segments from all signals in a directory"""
    logging.info("Segmenting signals...")
    if translate is None:
        translate = csv_to_wav

    csv_files = list(Path(csv_dir).glob(filter))
    wav_files = [
        Path(signals_dir) / translate(str(csv_file.name)) for csv_file in csv_files
    ]
    n_files = len(wav_files)

    for wav_file in wav_files:
        if not wav_file.exists():
            logging.error(f"Missing wav file: {wav_file}")

    for wav_file, csv_file in tqdm(
        zip(wav_files, csv_files), desc="Segmenting...", total=n_files
    ):
        segment_signal(wav_file, csv_file, Path(output_dir))
        segment_signal(wav_file, csv_file, Path(output_dir))
