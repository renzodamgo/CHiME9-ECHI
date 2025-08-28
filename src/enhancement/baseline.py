import torch
from omegaconf import OmegaConf
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import logging
from torch.amp import autocast
import soundfile as sf

from enhancement.registry import register_enhancement
from shared.core_utils import get_model
from shared.signal_utils import STFTWrapper, prep_audio


@register_enhancement("baseline")
class Baseline:
    def __init__(
        self,
        inference_dir: str,
        config_path: str,
        ckpt_path: str,
        audio_device: str,
        window_size: int,
        stride: int,
        torch_device: str,
    ):
        self.train_dir = Path(inference_dir).parent / f"train_{audio_device}"

        self.model_cfg = OmegaConf.load(config_path)

        self.stft = STFTWrapper(**self.model_cfg.input.stft, device=torch_device)
        self.stft = self.stft.to(torch_device)
        logging.info("ENCHANCEMENT INITIALIZED")
        logging.info(f"stft: {self.stft.n_fft}, {self.stft.hop_length}")
        logging.info(f"stft device: {self.stft.device}")
        logging.info(f"checkpoint path: {ckpt_path}")

        self.model = get_model(self.model_cfg, ckpt_path)
        self.model = self.model.to(torch_device)
        self.model.eval()

        self.window_samples = window_size * self.model_cfg.input.sample_rate
        rem = (self.window_samples - self.stft.n_fft) % self.stft.hop_length
        if rem > 0:
            self.window_samples += self.stft.hop_length - rem

        self.stride_samples = stride * self.model_cfg.input.sample_rate

        self.olap_samples = self.window_samples - self.stride_samples
        if self.olap_samples < 0:
            raise ValueError(
                f"Stride must be smaller than window size! Window: {window_size}, stride: {stride}"
            )
        elif self.olap_samples > 0:
            self.crossfade = torch.hann_window(
                self.olap_samples * 2, device=torch_device
            )

    def get_train_config(self):
        return OmegaConf.load(self.train_dir / "hydra/.hydra/config.yaml")

    def process_session(
        self,
        device_audio: torch.Tensor,
        device_fs: int,
        spkid_audio: torch.Tensor,
        spkid_fs: int,
    ) -> torch.Tensor:
        """
        Baseline-style enhancement with a single enrollment (Rainbow).
        - Computes enrollment STFT once outside the sliding loop.
        - Keeps model inputs 5D with K=1: spkid_input -> [1,1,T,F,2], spkid_lens -> [1,1].
        Returns mono enhanced waveform [T].
        """
        for d in range(torch.cuda.device_count()):
            with torch.cuda.device(d):
                torch.cuda.reset_peak_memory_stats(d)

        device_audio = prep_audio(
            device_audio,
            device_fs,
            self.model_cfg.input.channels,
            self.model_cfg.input.sample_rate,
            self.model_cfg.input.rms,
            batched=False,
        )  # [C,T] on same device as input

        sample_rate = self.model_cfg.input.sample_rate

        logging.info(f"device_audio shape: {device_audio.shape}")

        # ----- Prep single enrollment (Rainbow) once -----
        # Accept [T] or [1,T]
        if spkid_audio.ndim == 2 and spkid_audio.shape[0] == 1:
            spkid_audio = spkid_audio.squeeze(0)  # -> [T]

        spkid_audio = prep_audio(
            spkid_audio,  # [T]
            spkid_fs,
            1,  # mono
            self.model_cfg.input.sample_rate,
            self.model_cfg.input.rms,
            batched=False,
        )  # -> [1,T] (mono) on current device
        logging.info(f"spkid_audio shape: {spkid_audio.shape}")

        ## STFT of enrollment once → normalize to [1,1,F,T,2], lens [1,1]
        spkid_tf = self.stft(spkid_audio)  # returns [F, T, 2] or [1, F, T, 2]

        # --- START OF FIX ---
        # The model requires the speaker input to have the shape [B, K, T, F, 2].
        # We must explicitly permute the dimensions to ensure Time comes before Frequency.

        if spkid_tf.ndim == 3 and spkid_tf.shape[-1] == 2:
            # Input is [F, T, 2], e.g., [65, 8336, 2]
            # Permute to [T, F, 2]
            spkid_tf_swapped = spkid_tf.permute(1, 0, 2)
            # Add Batch and K dimensions -> [1, 1, T, F, 2]
            spkid_input = spkid_tf_swapped.unsqueeze(0).unsqueeze(0)

        elif spkid_tf.ndim == 4 and spkid_tf.shape[-1] == 2:
            # Input is [B, F, T, 2]
            # Permute to [B, T, F, 2]
            spkid_tf_swapped = spkid_tf.permute(0, 2, 1, 3)
            # Add K dimension -> [B, 1, T, F, 2]
            spkid_input = spkid_tf_swapped.unsqueeze(1)

        else:
            raise ValueError(
                f"Unexpected STFT(spkid) shape: {tuple(spkid_tf.shape)}; expected [F,T,2] or [B,F,T,2]"
            )
        T_frames = spkid_tf.shape[1]  # T is axis 1 in [F,T,2]
        spkid_lens = torch.tensor(
            [[spkid_input.shape[2]]], dtype=torch.long, device=spkid_input.device
        )

        # Log shape of spkid_input and spkid_lens
        logging.info(f"Corrected spkid_input shape: {spkid_input.shape}")
        logging.info(f"Corrected spkid_lens shape: {spkid_lens.shape}")

        logging.info(
            f"enroll packed (F-first)={tuple(spkid_input.shape)} lens_frames={int(spkid_lens.item())}"
        )
        logging.info(
            f"enroll packed={tuple(spkid_input.shape)} lens={int(spkid_lens.item())}"
        )
        logging.info(f"spkid_tf shape: {spkid_tf.shape}")
        # spkid_input = spkid_tf.unsqueeze(0)  # [1, 1, T, F, 2]
        spkid_lens = torch.tensor(
            [[spkid_input.shape[2]]], dtype=torch.long, device=spkid_input.device
        )  # [1,1]

        # log shape of spkid_input and spkid_lens
        logging.info(f"spkid_input shape: {spkid_input.shape}")
        logging.info(f"spkid_lens shape: {spkid_lens.shape}")

        # ----- Sliding-window OLA enhancement -----
        duration = device_audio.shape[-1]
        output = torch.zeros(duration, device=device_audio.device)
        logging.info(f"output shape: {output.shape}")

        for start in tqdm(range(0, duration, self.stride_samples)):
            end = min(start + self.window_samples, duration)
            window_size = end - start

            snippet = device_audio[..., start:end]
            snippet = prep_audio(
                snippet,
                self.model_cfg.input.sample_rate,
                self.model_cfg.input.channels,
                self.model_cfg.input.sample_rate,
                0.0,
                batched=False,
            )  # [C, Tw]

            # Pad to avoid STFT truncation at the tail
            rem = (window_size - self.stft.n_fft) % self.stft.hop_length
            if rem > 0:
                pad_samples = self.stft.hop_length - rem
                snippet = torch.nn.functional.pad(snippet, (0, pad_samples))

            # STFT mixture -> [1, M, T, F, 2]
            # mix_tf = self.stft(snippet)  # often [C, T, F, 2] or [C, 2, T, F]
            # if mix_tf.ndim == 4 and mix_tf.shape[-1] == 2:  # [C, T, F, 2]
            #     mix_tf = mix_tf.unsqueeze(0)  # [1, C, T, F, 2]
            # elif mix_tf.ndim == 4 and mix_tf.shape[1] == 2:  # [C, 2, T, F]
            #     mix_tf = mix_tf.permute(2, 3, 1, 0)  # [T, F, 2, C]
            #     mix_tf = mix_tf.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, T, F, 2]
            # else:
            #     raise ValueError(f"Unexpected STFT(mix) shape: {tuple(mix_tf.shape)}")

            # snippet: [C, Tw]
            mix_tf = self.stft(snippet)  # [C, F, T, 2]
            if mix_tf.ndim == 4:  # unbatched -> add batch
                mix_tf = mix_tf.unsqueeze(0)  # [1, C, F, T, 2]
            elif mix_tf.ndim == 5:
                # already [B, C, F, T, 2]; keep as-is
                pass
            else:
                raise ValueError(f"Unexpected STFT(mix) shape: {tuple(mix_tf.shape)}")

            logging.info(f"mix_tf shape (F-first): {tuple(mix_tf.shape)}")

            # Forward (K=1) -> [1,1,T,F] complex
            logging.info("FORWARD PASS:")
            self.model.eval()
            self.model.train()

            # autocast to bfloat16
            with torch.inference_mode(), autocast("cuda", dtype=torch.bfloat16):
                den_c = self.model(mix_tf, spkid_input, spkid_lens)
            
            logging.info(f"Model output shape: {den_c.shape}")  # Should be [1, 1, T, F]
            den_c = den_c[:, 0]  # -> [1,T,F] complex
            logging.info(f"After speaker selection shape: {den_c.shape}")  # Should be [1, T, F]

            # iSTFT and trim to window_size
            den_wav = self.stft.inverse(den_c).squeeze(0).squeeze(0)  # [Tw’]
            den_wav = den_wav[:window_size]

            # Crossfade overlaps
            if start > 0 and den_wav.shape[-1] > self.olap_samples:
                den_wav[: self.olap_samples] *= self.crossfade[: self.olap_samples]
            if end < duration and den_wav.shape[-1] > self.olap_samples:
                den_wav[-self.olap_samples :] *= self.crossfade[-self.olap_samples :]

            output[start:end] += den_wav

            logging.info(
                f"mix_tf mean|max: {mix_tf.abs().mean().item():.3e} | {mix_tf.abs().max().item():.3e}"
            )
            logging.info(
                f"den_c  mean|max: {den_c.abs().mean().item():.3e} | {den_c.abs().max().item():.3e}"
            )

        logging.info(
            f"output shape: {output.shape}, output min: {output.min().item()}, output max: {output.max().item()}, output mean: {output.mean().item()}"
        )

        logging.info(
            f"device_fs: {device_fs}, spkid_fs: {spkid_fs}, sample_rate: {sample_rate}"
        )

        # x_c = self.stft(device_audio)
        # x_rec = self.stft.inverse(x_c)
        # sf.write("debug_recon.wav", x_rec.cpu().numpy(), sample_rate)
        # sf.write("check.wav", output.cpu().numpy(), sample_rate)

        for d in range(torch.cuda.device_count()):
            with torch.cuda.device(d):
                torch.cuda.reset_peak_memory_stats(d)

        return output


def _fmt(bytes_):  # helper
    return f"{bytes_ / (1024**2):.1f} MiB"


def log_vram(prefix="", device=None):
    if device is None:
        device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    alloc = torch.cuda.memory_allocated(device)
    rsvd = torch.cuda.memory_reserved(device)
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_rsvd = torch.cuda.max_memory_reserved(device)
    logging.info(
        f"{prefix} GPU{device}: alloc={_fmt(alloc)}, reserved={_fmt(rsvd)}, "
        f"peak_alloc={_fmt(peak_alloc)}, peak_reserved={_fmt(peak_rsvd)}"
    )
