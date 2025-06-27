import torch
from librosa import resample


def process_session(
    noisy_audio: torch.Tensor,
    noisy_fs: int,
    spkid_audio: torch.Tensor,
    spkid_fs: int,
    target_sr: int,
):
    output = resample(
        noisy_audio[0].detach().cpu().numpy(), orig_sr=noisy_fs, target_sr=target_sr
    )
    return torch.from_numpy(output).unsqueeze(0)
