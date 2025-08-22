import torch
import torchaudio
import logging
from typing import Optional, Union

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


def prep_audio(
    audio: torch.Tensor,
    sample_rate: int,
    target_channels: int,
    target_sr: int,
    target_rms: float,
    batched: bool = True,
):
    input_ndim = audio.ndim

    # Ensure second dimension is the channel dimension
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
        assert not batched, "Batched audio must be 2D or 3D"

    if audio.ndim == 2:
        audio = audio.unsqueeze(int(batched))

    if audio.ndim > 3:
        raise ValueError(
            f"Audio cannot have more than 3 dimensions. Found shape {audio.shape}"
        )

    # Audio shape: [batch, audio_channels, samples]

    # Strip excess channels
    if target_channels > audio.shape[1]:
        logging.warning(
            f"Unexpected number of audio channels. Found {audio.shape[1]} channels in audio, but want to return {target_channels}"
        )
    else:
        audio = audio[:, :target_channels, :]

    # Resample
    if sample_rate != target_sr:
        audio = torchaudio.functional.resample(audio, sample_rate, target_sr)

    # Normalize
    if target_rms > 0.0:
        audio = rms_normalize(audio, target_rms)

    # Restore to original shape
    if batched:
        # input was 2D or 3D. Squeeze channel dim if necessary
        if input_ndim == 2:
            audio = audio.squeeze(1)
    else:
        # input was 1D or 2D. Squeeze batch dim
        audio = audio.squeeze(0)
        if input_ndim == 1:
            # Squeeze channel dim
            audio = audio.squeeze(0)

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

    def inverse(self, X: torch.Tensor, lengths: Optional[Union[int, torch.Tensor]] = None) -> torch.Tensor:
        """
        Inverse STFT that:
        • accepts [..., F, T] or [..., T, F] (RI or complex)
        • accepts per-sample output lengths to make time-domain signals match exactly

        Args:
        X: complex (or RI) STFT with last two dims (F,T) or (T,F)
        lengths: None | int | Tensor matching the 'front' dims of X (e.g., [B,K])
        """
        # ---- complexify if RI ----
        if not X.is_complex():
            X = X.contiguous()
            X = torch.complex(X[..., 0], X[..., 1])

        # ---- normalize layout to [*, F, T] ----
        n_freqs = self.n_fft // 2 + 1
        if X.size(-2) == n_freqs and X.size(-1) != n_freqs:
            C_FT = X  # [*, F, T]
        elif X.size(-1) == n_freqs:
            perm = list(range(X.ndim))
            perm[-2], perm[-1] = perm[-1], perm[-2]  # [*, T, F] -> [*, F, T]
            C_FT = X.permute(*perm).contiguous()
        else:
            raise RuntimeError(f"Cannot find freq bins among last two dims: {tuple(X.shape)} (expected {n_freqs}).")

        front_shape = C_FT.shape[:-2]
        F, T = C_FT.shape[-2], C_FT.shape[-1]
        C_flat = C_FT.reshape(-1, F, T)  # [N, F, T], N = prod(front_shape)

        # ---- window on correct device/dtype ----
        win = self.window
        if win is not None:
            target_dtype = C_flat.real.dtype  # float32 for complex64, etc.
            if win.device != C_flat.device or win.dtype != target_dtype:
                win = win.to(device=C_flat.device, dtype=target_dtype)

        # ---- handle lengths ----
        if lengths is None:
            x = torch.istft(
                C_flat, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                window=win, center=True, normalized=False, onesided=True, return_complex=False
            )
        else:
            if isinstance(lengths, int):
                # same length for all items
                x = torch.istft(
                    C_flat, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                    window=win, center=True, normalized=False, onesided=True, length=int(lengths),
                    return_complex=False
                )
            else:
                # tensor of lengths matching front dims
                L = lengths.reshape(-1).to(device=C_flat.device, dtype=torch.long)  # [N]
                assert L.numel() == C_flat.size(0), f"lengths {tuple(L.shape)} must match front dims {front_shape}"
                # Per-item iSTFT with given length
                xs = []
                for i in range(C_flat.size(0)):
                    xs.append(torch.istft(
                        C_flat[i], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                        window=win, center=True, normalized=False, onesided=True, length=int(L[i]),
                        return_complex=False
                    ))
                x = torch.stack(xs, dim=0)  # [N, Tw]

        return x.reshape(*front_shape, -1)