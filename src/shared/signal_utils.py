import torch
import torchaudio
import logging


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
            X = torch.complex(X[..., 0], X[..., 1])

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
