#!/usr/bin/env python3
"""Multi-speaker version of joint_ha_uni enhancement"""

import torch
from omegaconf import OmegaConf
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import logging
from torch.amp import autocast
import soundfile as sf

from enhancement.registry import register_enhancement
from shared.core_utils import get_model
from shared.signal_utils import STFTWrapper, prep_audio


@register_enhancement("joint_ha_uni_multispk")
class JointHaUniMultiSpeaker:
    """Multi-speaker enhancement using Universal GridNet (3 speakers simultaneously)"""
    
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
        
        self.stft = STFTWrapper(**self.model_cfg.model.input.stft, device=torch_device)
        self.stft = self.stft.to(torch_device)
        
        logging.info("UNIVERSAL GRIDNET MULTI-SPEAKER ENHANCEMENT INITIALIZED")
        logging.info(f"stft: {self.stft.n_fft}, {self.stft.hop_length}")
        logging.info(f"stft device: {self.stft.device}")
        logging.info(f"checkpoint path: {ckpt_path}")

        self.model = get_model(self.model_cfg.model, ckpt_path)
        self.model = self.model.to(torch_device)
        self.model.eval()

        self.window_samples = window_size * self.model_cfg.model.input.sample_rate
        rem = (self.window_samples - self.stft.n_fft) % self.stft.hop_length
        if rem > 0:
            self.window_samples += self.stft.hop_length - rem

        self.stride_samples = stride * self.model_cfg.model.input.sample_rate

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

    def process_multi_speaker_session(
        self,
        device_audio: torch.Tensor,
        device_fs: int,
        spkid_audios: List[torch.Tensor],
        spkid_fs: int,
    ) -> List[torch.Tensor]:
        """
        Multi-speaker Universal GridNet enhancement.
        
        Args:
            device_audio: [C,T] 4-channel HA device audio
            device_fs: Sample rate of device audio
            spkid_audios: List of [T] or [1,T] speaker enrollments (3 speakers)
            spkid_fs: Sample rate of speaker enrollments
            
        Returns:
            List of 3 enhanced audio tensors [T]
        """
        K = len(spkid_audios)  # Should be 3
        if K != 3:
            logging.warning(f"Expected 3 speakers, got {K}. Model trained on 3 speakers.")
        
        for d in range(torch.cuda.device_count()):
            with torch.cuda.device(d):
                torch.cuda.reset_peak_memory_stats(d)

        # Add batch dimension for device audio preprocessing  
        device_audio_batched = device_audio.unsqueeze(0)  # [1, C, T]
        
        device_audio = prep_audio(
            device_audio_batched,
            device_fs,
            self.model_cfg.model.input.channels,
            self.model_cfg.model.input.sample_rate,
            self.model_cfg.model.input.rms,
            batched=True,  # Match training preprocessing
        )
        
        device_audio = device_audio.squeeze(0)  # [C, T]
        sample_rate = self.model_cfg.model.input.sample_rate

        logging.info(f"device_audio shape: {device_audio.shape}")

        # ----- Prep multiple enrollments (K speakers) -----
        processed_enrollments = []
        
        for i, spkid_audio in enumerate(spkid_audios):
            # Ensure proper dimensions for batched processing
            if spkid_audio.ndim == 1:
                spkid_audio = spkid_audio.unsqueeze(0)  # [T] -> [1,T]
            elif spkid_audio.ndim == 2 and spkid_audio.shape[0] > 1:
                spkid_audio = spkid_audio[0:1]  # Take first channel: [C,T] -> [1,T]

            spkid_audio = prep_audio(
                spkid_audio,  # [1,T]
                spkid_fs,
                1,  # mono
                self.model_cfg.model.input.sample_rate,
                self.model_cfg.model.input.rms,
                batched=True,  # Match training preprocessing
            )
            
            processed_enrollments.append(spkid_audio)  # [1,T]
            logging.info(f"spkid_audio[{i}] shape: {spkid_audio.shape}")

        # Use Universal GridNet's native multi-speaker support
        # Convert each enrollment to STFT individually, then stack properly
        spkid_stfts = []
        spkid_lengths = []
        
        for i, enroll in enumerate(processed_enrollments):
            # STFT: [1, T] -> [1, F, T_stft, 2]
            enroll_stft = self.stft(enroll)  # [1, F, T_stft, 2]
            spkid_stfts.append(enroll_stft)
            spkid_lengths.append(enroll_stft.shape[2])  # T_stft dimension
            logging.info(f"Speaker {i} STFT shape: {enroll_stft.shape}")
        
        # Pad STFT representations to same length
        max_stft_length = max(spkid_lengths)
        padded_stfts = []
        
        for i, stft in enumerate(spkid_stfts):
            current_length = stft.shape[2]  # T dimension
            if current_length < max_stft_length:
                # Pad in time dimension: [1, F, T, 2] -> [1, F, T_max, 2]
                pad_length = max_stft_length - current_length
                padded_stft = torch.nn.functional.pad(stft, (0, 0, 0, pad_length))
                logging.info(f"Padded speaker {i} STFT: {current_length} -> {max_stft_length}")
            else:
                padded_stft = stft
            padded_stfts.append(padded_stft)
        
        # Stack to create multi-speaker input: [K, F, T_max, 2] -> [1, K, T_max, F, 2] 
        spkid_stacked = torch.stack(padded_stfts, dim=1)  # [1, K, F, T_max, 2]
        spkid_input = spkid_stacked.permute(0, 1, 3, 2, 4)  # [1, K, T_max, F, 2]
        
        # Speaker lengths for all K speakers (STFT frame lengths)
        spkid_lens = torch.tensor([spkid_lengths], dtype=torch.long, device=spkid_input.device)  # [1, K]

        logging.info(f"Universal GridNet - spkid_input shape: {spkid_input.shape}")
        logging.info(f"Universal GridNet - spkid_lens shape: {spkid_lens.shape}")

        # ----- Sliding-window OLA enhancement -----
        duration = device_audio.shape[-1]
        outputs = [torch.zeros(duration, device=device_audio.device) for _ in range(K)]
        
        logging.info(f"Processing {K} speakers simultaneously...")

        for start in tqdm(range(0, duration, self.stride_samples)):
            end = min(start + self.window_samples, duration)
            window_size = end - start

            snippet = device_audio[..., start:end]  # Already preprocessed - no prep_audio needed
            
            # Pad to avoid STFT truncation at the tail
            rem = (window_size - self.stft.n_fft) % self.stft.hop_length
            if rem > 0:
                pad_samples = self.stft.hop_length - rem
                snippet = torch.nn.functional.pad(snippet, (0, pad_samples))

            # STFT mixture -> [1, C, F, T, 2]
            mix_tf = self.stft(snippet)  # [C, F, T, 2]
            if mix_tf.ndim == 4:
                mix_tf = mix_tf.unsqueeze(0)  # [1, C, F, T, 2]

            # Multi-speaker Universal GridNet forward pass
            logging.info("MULTI-SPEAKER UNIVERSAL GRIDNET FORWARD PASS:")
            self.model.eval()

            with torch.inference_mode(), autocast("cuda", dtype=torch.bfloat16):
                den_c_all = self.model(mix_tf, spkid_input, spkid_lens)  # [1, K, T, F] complex
            
            logging.info(f"Universal GridNet multi-speaker output shape: {den_c_all.shape}")

            # Process each speaker's output - den_c_all is already complex
            for k in range(K):
                den_c_k = den_c_all[:, k:k+1]  # [1, 1, T, F] complex for speaker k
                
                # iSTFT and trim to window_size
                den_wav_k = self.stft.inverse(den_c_k).squeeze(0).squeeze(0)  # [Tw']
                den_wav_k = den_wav_k[:window_size]
                
                # Apply gentle high-frequency smoothing (same as single-speaker version)
                if den_wav_k.numel() > 2:
                    den_wav_smooth = torch.zeros_like(den_wav_k)
                    den_wav_smooth[0] = den_wav_k[0]
                    den_wav_smooth[1:-1] = 0.25 * den_wav_k[:-2] + 0.5 * den_wav_k[1:-1] + 0.25 * den_wav_k[2:]
                    den_wav_smooth[-1] = den_wav_k[-1]
                    den_wav_k = 0.8 * den_wav_k + 0.2 * den_wav_smooth

                # Crossfade overlaps
                if start > 0 and den_wav_k.shape[-1] > self.olap_samples:
                    den_wav_k[: self.olap_samples] *= self.crossfade[: self.olap_samples]
                if end < duration and den_wav_k.shape[-1] > self.olap_samples:
                    den_wav_k[-self.olap_samples :] *= self.crossfade[self.olap_samples :]

                outputs[k][start:end] += den_wav_k

        # Log final stats for each speaker
        for k, output_k in enumerate(outputs):
            logging.info(f"Speaker {k} output stats: "
                        f"mean={output_k.mean().item():.6f}, "
                        f"std={output_k.std().item():.6f}, "
                        f"range=[{output_k.min().item():.6f}, {output_k.max().item():.6f}]")

        for d in range(torch.cuda.device_count()):
            with torch.cuda.device(d):
                torch.cuda.reset_peak_memory_stats(d)

        return outputs