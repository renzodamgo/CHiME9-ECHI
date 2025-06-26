"""Tools for handling signals"""

import csv
import itertools
import logging
from pathlib import Path
import torch
import torchaudio

import numpy as np
import soundfile as sf
from tqdm import tqdm

POSITIONS = ["pos1", "pos2", "pos3", "pos4"]


def get_session_tuples(session_file, devices, datasets):
    """Get session tuples for the specified datasets and devices."""
    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(devices, str):
        devices = [devices]

    sessions = []

    for ds in datasets:
        with open(session_file.format(dataset=ds), "r") as f:
            sessions += list(csv.DictReader(f))

    session_device_pid_tuples = []

    for device, session in itertools.product(devices, sessions):
        device_pos = "pos" + session[f"{device}_pos"]

        if device_pos not in POSITIONS:
            logging.warning(f"Device {device} not found for session {session}")
            continue

        pids = [session[pos] for pos in POSITIONS if pos != device_pos]
        for pid in pids:
            session_device_pid_tuples.append((session["session"], device, pid))

    return session_device_pid_tuples


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
    wav_file: Path | list[Path], csv_file: Path, output_dir: Path, seg_sample_rate: int
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

    logging.debug(f"Will generate {len(segments)} segments from {wav_file}")
    sample_scalar = fs / seg_sample_rate

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
        if output_file.name[:2] == "._":
            print(wav_file)
        with open(output_file, "wb") as f:
            sf.write(f, signal_segment, samplerate=fs)


def csv_to_pid_wav(name: str) -> str:
    """Replace .wav with .csv"""
    return ".".join(name.split(".")[:-1]) + ".csv"


def segment_all_signals(
    signal_template,
    output_dir_template,
    segment_info_file,
    session_tuples,
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

        segment_signal(wav_file, csv_file, output_dir, seg_sample_rate)


def get_rms(signal: torch.Tensor) -> torch.Tensor:
    """
    Calculate the RMS of a signal.
    Args:
        signal (torch.Tensor): The input signal.
    Returns:
        torch.Tensor: The RMS of the signal.
    """
    return torch.sqrt(torch.mean(signal**2))


def rms_normalize(signal: torch.Tensor, target_rms: float) -> torch.Tensor:
    """
    Normalize the RMS of a signal to 1.
    Args:
        signal (torch.Tensor): The input signal.
    Returns:
        torch.Tensor: The normalized signal.
    """
    rms = get_rms(signal)
    if rms > 0:
        return signal * target_rms / rms
    else:
        return signal


class AudioPrep:
    def __init__(
        self,
        output_channels: int,
        input_sr: int,
        output_sr: int,
        output_rms: float,
        device: str,
    ):
        """
        Initialize the AudioPrep class.

        Args:
            output_channels (int): Number of output audio channels.
            input_sr (int): Sample rate of the input audio.
            output_sr (int): Desired sample rate for the output audio.
            output_rms (float): Target RMS value for output normalization.
            device (str): Device to run the resampler on (e.g., 'cpu' or 'cuda').

        Attributes:
            output_channels (int): Number of output channels.
            input_sr (int): Input sample rate.
            output_sr (int): Output sample rate.
            output_rms (float): Target output RMS value.
            resampler (torchaudio.transforms.Resample): Resampler for converting audio from input_sr to output_sr.
        """
        self.output_channels = output_channels
        self.input_sr = input_sr
        self.output_sr = output_sr
        self.output_rms = output_rms
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=input_sr, new_freq=output_sr
        ).to(device)

    def process(self, audio: torch.Tensor, fs: int):
        """
        Processes an audio tensor by ensuring correct shape, resampling, and RMS normalization.
        Args:
            audio (torch.Tensor): Input audio tensor. shape: (batch x channels x samples)
            fs (int): Sample rate of the input audio.
        Returns:
            torch.Tensor: Processed audio tensor with correct shape, sample rate, and RMS normalization. shape: (batch (x channels) x samples)
        Behavior:
            - If the input audio is 1D, it is unsqueezed to add a channel dimension.
            - If the input sample rate matches `self.input_sr`, the audio is resampled.
            - If the input sample rate does not match `self.output_sr`, an error is logged.
            - If `self.output_rms` is non-zero, the audio is RMS-normalized to this value.
        """

        if audio.ndim == 1:
            # Assume shape = (samples)
            audio = audio.unsqueeze(0)
        elif audio.ndim > 2:
            logging.error(f"Too many dimensions in audio!!\naudio.shape={audio.shape}")

        if self.output_channels > audio.shape[0]:
            logging.error(
                f"Invalid # of channels. Requested {self.output_channels} channels but audio only has {audio.shape[0]}"
            )
        elif self.output_channels == 1:
            audio = audio[0, :]
        else:
            audio = audio[: self.output_channels, :]

        if fs == self.input_sr:
            audio = self.resampler(audio)
        elif fs != self.output_sr:
            logging.error(
                f"Unexpected sample rate:\nExpected input: {self.input_sr}Hz\nExpected output: {self.output_sr}Hz\nGiven input: {fs}Hz"
            )

        if self.output_rms != 0:
            audio = rms_normalize(audio, self.output_rms)

        return audio


def match_length(audio0, audio1):
    """
    Pads the shorter of two audio tensors along the last dimension so that both have the same length.
    Parameters:
        audio0 (torch.Tensor): The first audio tensor.
        audio1 (torch.Tensor): The second audio tensor.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the two audio tensors, both padded to the same length along the last dimension.
    """

    if audio0.shape[-1] > audio1.shape[-1]:
        pad_len = audio0.shape[-1] - audio1.shape[-1]
        audio1 = torch.nn.functional.pad(audio1, (0, pad_len))
    elif audio1.shape[-1] > audio0.shape[-1]:
        pad_len = audio1.shape[-1] - audio0.shape[-1]
        audio0 = torch.nn.functional.pad(audio0, (0, pad_len))
    return audio0, audio1


def pad_samples(audio: torch.Tensor, samples: int):
    """
    Pads the input audio tensor with a specified number of zeros at the end.
    Args:
        audio (torch.Tensor): The input audio tensor to be padded.
        samples (int): The number of zero samples to pad at the end of the audio tensor.
    Returns:
        torch.Tensor: The padded audio tensor. If samples is 0, returns the original tensor.
    """

    if samples == 0:
        return audio
    audio = torch.nn.functional.pad(audio, (0, samples), mode="constant", value=0.0)
    return audio


def pad_tolength(audio: torch.Tensor, target_length: int):
    """
    Pads the input audio tensor to the specified target length.
    If the target length is less than or equal to the current length of the audio tensor,
    the original audio is returned. If the target length is greater, the audio is padded
    with zeros (or as defined by `pad_samples`) to reach the target length.
    Args:
        audio (torch.Tensor): The input audio tensor to be padded.
        target_length (int): The desired length of the output tensor.
    Returns:
        torch.Tensor: The padded audio tensor if padding is needed, otherwise the original tensor.
    Raises:
        Logs an error if the target length is shorter than the audio length.
    """

    if target_length < audio.shape[-1]:
        logging.error("Target length shorter than audio len")
        return audio
    elif target_length == audio.shape[-1]:
        return audio
    else:
        return pad_samples(audio, target_length - audio.shape[-1])


def combine_audio_list(audio: list[torch.Tensor]):
    lens = [x.shape[-1] for x in audio]
    if len(set(lens)) == 1:
        return torch.stack(audio), torch.tensor(lens)
    max_len = max(lens)
    new_audio = []
    for x in audio:
        new_audio.append(pad_tolength(x, max_len))
    return torch.stack(new_audio), torch.tensor(lens)


class STFTWrapper(torch.nn.Module):
    def __init__(
        self, n_fft=1024, hop_length=256, win_length=None, window=None, device="cpu"
    ):
        super(STFTWrapper, self).__init__()

        self.device = device

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.window = torch.hann_window(self.win_length).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        do_reshape = False
        if x.ndim == 3:
            do_reshape = True
            batch, chan, samp = x.shape
            x = x.reshape(batch * chan, samp)

        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        X = torch.view_as_real(X)

        if do_reshape:
            _, F, T, comp = X.shape
            X = X.reshape(batch, chan, F, T, comp)

        return X

    def inverse(self, X: torch.Tensor) -> torch.Tensor:
        if not X.is_complex():
            X = X.contiguous()
            X = torch.view_as_complex(X)

        do_reshape = X.ndim == 4
        if do_reshape:
            # Multichannel input
            batch, chan, freq, frames = X.shape
            X = X.reshape(batch * chan, freq, frames)

        x = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
        )

        if do_reshape:
            x = x.reshape(batch, chan, -1)

        return x
