"""Setup the ECHI data for use in experiments"""

import logging

import hydra
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import soxr
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from shared.core_utils import get_session_tuples, get_device
from enhancement.registry import enhancement_options


def enhance_all_sessions(cfg, enhance_args):
    logging.info("Preparing the ECHI dataset")

    torch_device = get_device()

    session_tuples = get_session_tuples(
        cfg.sessions_file, cfg.device, datasets=cfg.dataset
    )
    enhancement = enhancement_options[enhance_args.name]
    enhancement = enhancement(**enhance_args.args, torch_device=torch_device)

    for session, device, pid in tqdm(session_tuples):
        dataset = session.split("_")[0]

        noisy_fpath = cfg.noisy_signal.format(
            dataset=dataset, session=session, device=device, pid=pid
        )
        rainbow_fpath = cfg.rainbow_signal.format(
            dataset=dataset, session=session, device=device, pid=pid
        )

        noisy_audio, noisy_fs = torchaudio.load(noisy_fpath)
        rainbow_audio, rainbow_fs = torchaudio.load(rainbow_fpath)

        noisy_audio = noisy_audio.to(torch_device)
        rainbow_audio = rainbow_audio.to(torch_device)

        output = enhancement.process_session(
            device_audio=noisy_audio,
            device_fs=noisy_fs,
            spkid_audio=rainbow_audio,
            spkid_fs=rainbow_fs,
        )

        enhanced_fpath = Path(
            cfg.enhanced_signal.format(
                dataset=dataset, session=session, device=device, pid=pid
            )
        )

        if not enhanced_fpath.parent.exists():
            enhanced_fpath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()

        output = np.squeeze(output)

        if output.ndim > 1:
            raise ValueError(f"Output has too many channels: {output.shape}")

        if cfg.model_sample_rate != cfg.ref_sample_rate:
            output = soxr.resample(output, cfg.model_sample_rate, cfg.ref_sample_rate)

        with open(enhanced_fpath, "wb") as file:
            sf.write(file, output, cfg.ref_sample_rate, subtype=cfg.bitdepth)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    enhance_all_sessions(cfg.inference)
