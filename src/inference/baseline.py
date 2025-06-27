import torch
from omegaconf import OmegaConf, DictConfig
import json
from pathlib import Path
from typing import Callable

from shared.core_utils import get_model
from shared.signal_utils import AudioPrep, STFTWrapper
from shared.CausalMCxTFGridNet import MCxTFGridNet


def enhance(
    noisy_audio: torch.Tensor,
    noisy_fs: int,
    spkid_audio: torch.Tensor,
    spkid_fs: int,
    noisy_prep: AudioPrep,
    spkid_prep: AudioPrep,
    stft: STFTWrapper,
    model: MCxTFGridNet,
):
    noisy_audio = noisy_prep.process(noisy_audio, noisy_fs)
    spkid_audio = spkid_prep.process(spkid_audio, spkid_fs)
    spkid_stft = stft(spkid_audio)

    # spkid_stft = spkid_stft.transpose(0, 1)
    spkid_stft = spkid_stft.unsqueeze(0)
    spkid_lens = torch.tensor([spkid_stft.shape[2]])

    duration = noisy_audio.shape[-1]

    output = torch.zeros(duration)

    window_time = 60
    overlap = 1 / 8

    window_samples = window_time * noisy_prep.output_sr
    window_samples -= (window_samples - stft.n_fft) % stft.hop_length
    overlap_samples = int(window_samples * overlap)
    stride = window_samples - overlap_samples

    print("window", window_samples // noisy_prep.output_sr)
    print("stride", stride / noisy_prep.output_sr)

    model.eval()
    with torch.no_grad():
        for start in range(0, noisy_audio.shape[-1], stride):
            print(start)
            end = start + window_samples
            if end > duration:
                end = duration + 1

            snippet = noisy_audio[:, start:end]
            snippet = stft(snippet)
            snippet = snippet.unsqueeze(0)

            den_snippet = model(snippet, spkid_stft, spkid_lens)

            den_snippet = den_snippet.squeeze(0)
            den_snippet = stft.inverse(den_snippet)
            den_snippet = den_snippet.squeeze(0).squeeze(0)

            if start > 0:
                den_snippet[:overlap_samples] *= 0.5
            if end != duration + 1:
                den_snippet[-overlap_samples:] *= 0.5

            output[start:end] += den_snippet
            break

    return output


def get_process(exp_dir: Path) -> tuple[Callable, dict]:

    # Get model
    model, cfg = find_model(exp_dir)

    # Get kwargs
    kwargs = load_kwargs(cfg.model)
    kwargs["model"] = model

    return enhance, kwargs


def find_model(exp_dir: Path):

    exp_dir = exp_dir / "train"

    with open(exp_dir / "train_log.json", "r") as file:
        train_log = json.load(file)

    cfg = OmegaConf.load(exp_dir / "hydra/.hydra/config.yaml")

    best_log = min(train_log, key=lambda x: x["val_loss"])
    best_epoch = best_log["epoch"]
    ckpt_path = exp_dir / f"checkpoints/epoch{str(best_epoch).zfill(3)}.pt"

    model = get_model(cfg.model, ckpt_path)
    return model, cfg


def load_kwargs(model_cfg: DictConfig):

    noisy_prepper = AudioPrep(
        output_channels=model_cfg.input.channels,
        input_sr=48000,
        output_sr=model_cfg.input.sample_rate,
        output_rms=model_cfg.input.rms,
        device="cpu",
    )
    spk_prepper = AudioPrep(
        output_channels=1,
        input_sr=48000,
        output_sr=model_cfg.input.sample_rate,
        output_rms=model_cfg.input.rms,
        device="cpu",
    )
    stft = STFTWrapper(**model_cfg.input.stft)

    return {"noisy_prep": noisy_prepper, "spkid_prep": spk_prepper, "stft": stft}
