#!/usr/bin/env python3
"""Test single sample enhancement with noise fixes"""

import logging
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from omegaconf import OmegaConf
import torch.nn.functional as F

from src.enhancement.joint_ha_uni import JointHaUni
from src.shared.core_utils import get_device

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')

def apply_noise_reduction(audio, sr=16000):
    """Apply high-frequency noise reduction"""
    
    # 1. High-frequency rolloff filter (low-pass at 6kHz)
    nyquist = sr // 2
    cutoff = 6000  # 6kHz cutoff
    normalized_cutoff = cutoff / nyquist
    
    # Simple butterworth-style filter approximation
    # Create frequency domain filter
    n_fft = 1024
    freqs = torch.fft.rfftfreq(n_fft, 1/sr)
    
    # Design low-pass filter
    filter_response = 1.0 / (1.0 + (freqs / cutoff) ** 4)  # 4th order rolloff
    filter_response = filter_response.to(audio.device)
    
    # Apply in chunks to avoid memory issues
    chunk_size = sr * 2  # 2 second chunks
    filtered_audio = torch.zeros_like(audio)
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < n_fft:
            filtered_audio[i:i+len(chunk)] = chunk
            continue
            
        # Apply filter via FFT
        chunk_fft = torch.fft.rfft(chunk, n=n_fft)
        filtered_fft = chunk_fft * filter_response[:len(chunk_fft)]
        filtered_chunk = torch.fft.irfft(filtered_fft, n=n_fft)
        
        filtered_audio[i:i+len(chunk)] = filtered_chunk[:len(chunk)]
    
    # 2. Gentle temporal smoothing (reduce transient noise)
    if len(audio) > 4:
        kernel = torch.tensor([0.1, 0.8, 0.1], device=audio.device)
        kernel = kernel / kernel.sum()
        
        # Pad audio for convolution
        padded = F.pad(filtered_audio.unsqueeze(0).unsqueeze(0), (1, 1), mode='replicate')
        smoothed = F.conv1d(padded, kernel.view(1, 1, 3))
        filtered_audio = smoothed.squeeze(0).squeeze(0)
    
    return filtered_audio

def test_single_enhancement_fixed():
    """Test Universal GridNet with noise reduction fixes"""
    
    # Configuration
    torch_device = get_device()
    print(f"Using device: {torch_device}")
    
    # Sample paths based on actual data structure
    noisy_dir = Path("data/chime9_echi/ha/dev/")
    rainbow_dir = Path("data/chime9_echi/participant/dev/")
    
    noisy_files = list(noisy_dir.glob("*.wav"))
    rainbow_files = list(rainbow_dir.glob("*.wav"))
    
    if not noisy_files or not rainbow_files:
        print("❌ No audio files found")
        return
        
    noisy_file = noisy_files[0]
    rainbow_file = rainbow_files[0]
    session = noisy_file.stem.replace('.ha', '')
    
    noisy_path = str(noisy_file)
    rainbow_path = str(rainbow_file)
    output_path = f"test_enhanced_FIXED_{session}_{rainbow_file.stem}.wav"
    
    print(f"Using noisy sample: {noisy_path}")
    print(f"Using rainbow sample: {rainbow_path}")
    
    # Enhancement configuration with FIXES
    enhance_config = {
        'inference_dir': 'data/working_dir/experiments/ha-joint-uni/',
        'config_path': 'data/working_dir/experiments/ha-joint-uni/train_ha/hydra/.hydra/config.yaml',
        'ckpt_path': 'data/working_dir/experiments/ha-joint-uni/train_ha/checkpoints/ha-joint-uni_048.pt',
        'audio_device': 'ha',
        'window_size': 10,  # Smaller window for testing
        'stride': 8,        # 2 second overlap
    }
    
    print("🔧 Initializing Universal GridNet enhancement...")
    enhancement = JointHaUni(**enhance_config, torch_device=torch_device)
    
    # PATCH: Override RMS normalization in model config
    original_rms = enhancement.model_cfg.model.input.rms
    print(f"Original RMS: {original_rms}")
    enhancement.model_cfg.model.input.rms = 0.05  # Increase from 0.01 to 0.05
    print(f"🔧 Fixed RMS normalization: {original_rms} → {enhancement.model_cfg.model.input.rms}")
    
    print("✅ Enhancement initialized with fixes")
    
    print(f"🎵 Loading audio files...")
    # Load audio
    noisy_audio, noisy_fs = torchaudio.load(noisy_path)
    rainbow_audio, rainbow_fs = torchaudio.load(rainbow_path)
    
    print(f"Noisy audio shape: {noisy_audio.shape}, fs: {noisy_fs}")
    print(f"Rainbow audio shape: {rainbow_audio.shape}, fs: {rainbow_fs}")
    
    # Move to device
    noisy_audio = noisy_audio.to(torch_device)
    rainbow_audio = rainbow_audio.to(torch_device)
    
    print("🚀 Running Universal GridNet enhancement with fixes...")
    with torch.inference_mode():
        enhanced_audio = enhancement.process_session(
            device_audio=noisy_audio,
            device_fs=noisy_fs,
            spkid_audio=rainbow_audio,
            spkid_fs=rainbow_fs,
        )
    
    print(f"✅ Enhancement completed!")
    print(f"Enhanced audio shape: {enhanced_audio.shape}")
    print(f"Enhanced audio stats BEFORE noise reduction: mean={enhanced_audio.mean():.6f}, std={enhanced_audio.std():.6f}")
    print(f"Enhanced audio range BEFORE: [{enhanced_audio.min():.6f}, {enhanced_audio.max():.6f}]")
    
    # Apply noise reduction
    print("🔧 Applying noise reduction...")
    enhanced_audio_clean = apply_noise_reduction(enhanced_audio, sr=16000)
    
    print(f"Enhanced audio stats AFTER noise reduction: mean={enhanced_audio_clean.mean():.6f}, std={enhanced_audio_clean.std():.6f}")
    print(f"Enhanced audio range AFTER: [{enhanced_audio_clean.min():.6f}, {enhanced_audio_clean.max():.6f}]")
    
    # Convert to numpy and save
    if isinstance(enhanced_audio_clean, torch.Tensor):
        enhanced_audio_clean = enhanced_audio_clean.detach().cpu().numpy()
    
    # Save output
    sf.write(output_path, enhanced_audio_clean, 16000)
    print(f"💾 Enhanced audio saved to: {output_path}")
    
    # Quality check
    if enhanced_audio_clean.std() < 1e-6:
        print("⚠️  WARNING: Output has very low variation")
    else:
        print("✅ Output appears to have reasonable variation")
        
    # Compare with original
    rms_ratio = enhanced_audio_clean.std() / enhanced_audio.detach().cpu().numpy().std()
    print(f"📊 Noise reduction effect: {rms_ratio:.2f}x quieter")

if __name__ == "__main__":
    test_single_enhancement_fixed()