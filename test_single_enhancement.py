#!/usr/bin/env python3
"""Test single sample enhancement with Universal GridNet"""

import logging
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from omegaconf import OmegaConf

from src.enhancement.joint_ha_uni import JointHaUni
from src.shared.core_utils import get_device

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')

def test_single_enhancement():
    """Test Universal GridNet on a single dev sample"""
    
    # Configuration
    torch_device = get_device()
    print(f"Using device: {torch_device}")
    
    # Sample paths based on actual data structure
    # Use first available dev sample
    noisy_dir = Path("data/chime9_echi/ha/dev/")
    rainbow_dir = Path("data/chime9_echi/participant/dev/")
    
    noisy_files = list(noisy_dir.glob("*.wav"))
    if not noisy_files:
        print("❌ No HA dev samples found")
        return
        
    noisy_file = noisy_files[0]  # Use first available
    session = noisy_file.stem.replace('.ha', '')  # e.g., dev_02
    
    # Find corresponding participant ID from sessions file or use first available
    rainbow_files = list(rainbow_dir.glob("*.wav"))
    if not rainbow_files:
        print("❌ No participant samples found")
        return
        
    rainbow_file = rainbow_files[0]  # Use first available participant
    
    noisy_path = str(noisy_file)
    rainbow_path = str(rainbow_file)
    output_path = f"test_enhanced_{session}_{rainbow_file.stem}.wav"
    
    print(f"Using noisy sample: {noisy_path}")
    print(f"Using rainbow sample: {rainbow_path}")
    
    # Check if files exist
    if not Path(noisy_path).exists():
        print(f"❌ Noisy file not found: {noisy_path}")
        return
    
    if not Path(rainbow_path).exists():
        print(f"❌ Rainbow file not found: {rainbow_path}")
        return
    
    # Enhancement configuration
    enhance_config = {
        'inference_dir': 'data/working_dir/experiments/ha-joint-uni/',
        'config_path': 'data/working_dir/experiments/ha-joint-uni/train_ha/hydra/.hydra/config.yaml',
        'ckpt_path': 'data/working_dir/experiments/ha-joint-uni/train_ha/checkpoints/ha-joint-uni_048.pt',
        'audio_device': 'ha',
        'window_size': 10,  # Smaller window for testing (10 seconds)
        'stride': 8,        # 2 second overlap
    }
    
    print("🔧 Initializing Universal GridNet enhancement...")
    try:
        enhancement = JointHaUni(**enhance_config, torch_device=torch_device)
        print("✅ Enhancement initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize enhancement: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"🎵 Loading audio files...")
    try:
        # Load audio
        noisy_audio, noisy_fs = torchaudio.load(noisy_path)
        rainbow_audio, rainbow_fs = torchaudio.load(rainbow_path)
        
        print(f"Noisy audio shape: {noisy_audio.shape}, fs: {noisy_fs}")
        print(f"Rainbow audio shape: {rainbow_audio.shape}, fs: {rainbow_fs}")
        
        # Move to device
        noisy_audio = noisy_audio.to(torch_device)
        rainbow_audio = rainbow_audio.to(torch_device)
        
        print("✅ Audio loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load audio: {e}")
        return
    
    print("🚀 Running Universal GridNet enhancement...")
    try:
        # Process with Universal GridNet
        with torch.inference_mode():
            enhanced_audio = enhancement.process_session(
                device_audio=noisy_audio,
                device_fs=noisy_fs,
                spkid_audio=rainbow_audio,
                spkid_fs=rainbow_fs,
            )
        
        print(f"✅ Enhancement completed!")
        print(f"Enhanced audio shape: {enhanced_audio.shape}")
        print(f"Enhanced audio stats: mean={enhanced_audio.mean():.6f}, std={enhanced_audio.std():.6f}")
        print(f"Enhanced audio range: [{enhanced_audio.min():.6f}, {enhanced_audio.max():.6f}]")
        
        # Convert to numpy and save
        if isinstance(enhanced_audio, torch.Tensor):
            enhanced_audio = enhanced_audio.detach().cpu().numpy()
        
        # Save output
        sf.write(output_path, enhanced_audio, 16000)
        print(f"💾 Enhanced audio saved to: {output_path}")
        
        # Basic quality check
        if enhanced_audio.std() < 1e-6:
            print("⚠️  WARNING: Output has very low variation - possible silence or constant output")
        else:
            print("✅ Output appears to have reasonable variation")
            
    except Exception as e:
        print(f"❌ Enhancement failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_enhancement()