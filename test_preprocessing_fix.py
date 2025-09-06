#!/usr/bin/env python3
"""Test Universal GridNet with fixed preprocessing to match training"""

import logging
import torch
import torchaudio
import soundfile as sf
from pathlib import Path

from src.enhancement.joint_ha_uni import JointHaUni
from src.shared.core_utils import get_device

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')

def test_preprocessing_fix():
    """Test Universal GridNet with corrected preprocessing"""
    
    torch_device = get_device()
    print(f"Using device: {torch_device}")
    
    # Sample paths
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
    output_path = f"test_enhanced_PREPROCESSING_FIXED_{session}_{rainbow_file.stem}.wav"
    
    print(f"Using noisy sample: {noisy_path}")
    print(f"Using rainbow sample: {rainbow_path}")
    
    # Enhancement configuration
    enhance_config = {
        'inference_dir': 'data/working_dir/experiments/ha-joint-uni/',
        'config_path': 'data/working_dir/experiments/ha-joint-uni/train_ha/hydra/.hydra/config.yaml',
        'ckpt_path': 'data/working_dir/experiments/ha-joint-uni/train_ha/checkpoints/ha-joint-uni_048.pt',
        'audio_device': 'ha',
        'window_size': 10,  # 10 second windows for testing
        'stride': 8,        # 2 second overlap
    }
    
    print("🔧 Initializing Universal GridNet with FIXED preprocessing...")
    enhancement = JointHaUni(**enhance_config, torch_device=torch_device)
    print("✅ Enhancement initialized")
    
    # Load audio
    noisy_audio, noisy_fs = torchaudio.load(noisy_path)
    rainbow_audio, rainbow_fs = torchaudio.load(rainbow_path)
    
    print(f"Noisy audio shape: {noisy_audio.shape}, fs: {noisy_fs}")
    print(f"Rainbow audio shape: {rainbow_audio.shape}, fs: {rainbow_fs}")
    
    # Move to device
    noisy_audio = noisy_audio.to(torch_device)
    rainbow_audio = rainbow_audio.to(torch_device)
    
    print("🚀 Running Universal GridNet with FIXED preprocessing...")
    with torch.inference_mode():
        enhanced_audio = enhancement.process_session(
            device_audio=noisy_audio,
            device_fs=noisy_fs,
            spkid_audio=rainbow_audio,
            spkid_fs=rainbow_fs,
        )
    
    print(f"✅ Enhancement completed with FIXED preprocessing!")
    print(f"Enhanced audio shape: {enhanced_audio.shape}")
    print(f"Enhanced audio stats: mean={enhanced_audio.mean():.6f}, std={enhanced_audio.std():.6f}")
    print(f"Enhanced audio range: [{enhanced_audio.min():.6f}, {enhanced_audio.max():.6f}]")
    
    # Convert to numpy and save
    if isinstance(enhanced_audio, torch.Tensor):
        enhanced_audio = enhanced_audio.detach().cpu().numpy()
    
    # Save output
    sf.write(output_path, enhanced_audio, 16000)
    print(f"💾 Enhanced audio saved to: {output_path}")
    
    # Quality assessment
    rms_value = enhanced_audio.std()
    print(f"📊 Output RMS: {rms_value:.6f}")
    
    if rms_value < 1e-6:
        print("⚠️  WARNING: Output has very low variation")
    elif rms_value > 0.1:
        print("⚠️  WARNING: Output may be too loud")
    else:
        print("✅ Output RMS appears reasonable")
        
    # Quick spectral analysis
    fft = torch.fft.rfft(torch.from_numpy(enhanced_audio))
    magnitude_db = 20 * torch.log10(torch.abs(fft) + 1e-10)
    
    high_freq_start = len(magnitude_db) // 2
    high_freq_energy = magnitude_db[high_freq_start:].mean().item()
    low_freq_energy = magnitude_db[:high_freq_start].mean().item()
    
    print(f"🎵 Spectral Analysis:")
    print(f"   High-freq energy: {high_freq_energy:.2f} dB")
    print(f"   Low-freq energy: {low_freq_energy:.2f} dB")
    print(f"   HF/LF ratio: {high_freq_energy - low_freq_energy:.2f} dB")
    
    if high_freq_energy - low_freq_energy > 10:
        print("⚠️  WARNING: High-frequency noise detected")
    else:
        print("✅ Frequency balance looks good")

if __name__ == "__main__":
    test_preprocessing_fix()