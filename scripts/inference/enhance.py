"""Setup the ECHI data for use in experiments"""

import logging

import hydra
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from shared.core_utils import get_session_tuples
from inference import get_enhance_fn


def enhance_all_sessions(cfg):
    logging.info("Preparing the ECHI dataset")

    session_tuples = get_session_tuples(
        cfg.sessions_file, cfg.devices, datasets=cfg.dataset
    )
    enhance_fn, kwargs = get_enhance_fn(Path(cfg.base_dir).parent)

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

        output = enhance_fn(
            noisy_audio=noisy_audio,
            noisy_fs=noisy_fs,
            spkid_audio=rainbow_audio,
            spkid_fs=rainbow_fs,
            **kwargs,
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

        with open(enhanced_fpath, "wb") as file:
            sf.write(file, output, cfg.sample_rate, subtype=cfg.bitdepth)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    enhance_all_sessions(cfg.inference)
