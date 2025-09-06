#!/usr/bin/env python3
"""Test true multi-speaker enhancement (all 3 speakers simultaneously)"""

import logging
import torch
import torchaudio
import soundfile as sf
from pathlib import Path

from src.enhancement.joint_ha_uni_multispk import JointHaUniMultiSpeaker
from src.shared.core_utils import get_device

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')

def test_true_multispk_enhancement():
    """Test true multi-speaker enhancement (all 3 simultaneously)"""
    
    torch_device = get_device()
    print(f"Using device: {torch_device}")
    
    # Sample paths
    noisy_dir = Path("data/chime9_echi/ha/dev/")
    rainbow_dir = Path("data/chime9_echi/participant/dev/")
    
    noisy_files = list(noisy_dir.glob("*.wav"))
    rainbow_files = list(rainbow_dir.glob("*.wav"))
    
    if not noisy_files or len(rainbow_files) < 3:
        print("❌ Need noisy file and at least 3 participant files")
        return
        
    noisy_file = noisy_files[0]
    # Use first 3 participant files for testing
    speaker_files = rainbow_files[:3]
    
    session = noisy_file.stem.replace('.ha', '')
    
    print(f"🎵 Session: {session}")
    print(f"📁 Noisy file: {noisy_file}")
    print(f"👥 Speaker files: {[f.name for f in speaker_files]}")
    
    # Enhancement configuration
    enhance_config = {
        'inference_dir': 'data/working_dir/experiments/ha-joint-uni/',
        'config_path': 'data/working_dir/experiments/ha-joint-uni/train_ha/hydra/.hydra/config.yaml',
        'ckpt_path': 'data/working_dir/experiments/ha-joint-uni/train_ha/checkpoints/ha-joint-uni_048.pt',
        'audio_device': 'ha',
        'window_size': 10,
        'stride': 8,
        'torch_device': torch_device,
    }
    
    print("🔧 Initializing TRUE multi-speaker Universal GridNet...")
    enhancement = JointHaUniMultiSpeaker(**enhance_config)
    print("✅ Multi-speaker enhancement initialized")
    
    # Load audio files
    print("🎵 Loading audio files...")
    noisy_audio, noisy_fs = torchaudio.load(str(noisy_file))
    
    speaker_audios = []
    for spk_file in speaker_files:
        spk_audio, spk_fs = torchaudio.load(str(spk_file))
        speaker_audios.append(spk_audio)
    
    print(f"📊 Noisy audio: {noisy_audio.shape}, {noisy_fs} Hz")
    for i, spk in enumerate(speaker_audios):
        print(f"   Speaker {i+1}: {spk.shape}, {spk_fs} Hz")
    
    # Move to device
    noisy_audio = noisy_audio.to(torch_device)
    speaker_audios = [spk.to(torch_device) for spk in speaker_audios]
    
    print("🚀 Running TRUE multi-speaker enhancement (3 speakers simultaneously)...")
    
    try:
        with torch.inference_mode():
            enhanced_speakers = enhancement.process_multi_speaker_session(
                device_audio=noisy_audio,
                device_fs=noisy_fs,
                spkid_audios=speaker_audios,
                spkid_fs=48000,
            )
        
        print(f"✅ TRUE multi-speaker enhancement completed!")
        print(f"📊 Enhanced {len(enhanced_speakers)} speakers simultaneously")
        
        # Save outputs
        for i, enhanced_audio in enumerate(enhanced_speakers):
            speaker_id = speaker_files[i].stem
            output_path = f"enhanced_TRUE_MULTISPK_{session}_speaker{i+1}_{speaker_id}.wav"
            
            if isinstance(enhanced_audio, torch.Tensor):
                enhanced_audio_cpu = enhanced_audio.detach().cpu().numpy()
            else:
                enhanced_audio_cpu = enhanced_audio
            
            sf.write(output_path, enhanced_audio_cpu, 16000)
            
            print(f"💾 Speaker {i+1} ({speaker_id}):")
            print(f"   Shape: {enhanced_audio.shape}")
            print(f"   Stats: mean={enhanced_audio.mean():.6f}, std={enhanced_audio.std():.6f}")
            print(f"   Range: [{enhanced_audio.min():.6f}, {enhanced_audio.max():.6f}]")
            print(f"   Saved: {output_path}")
        
        print(f"\n🎉 TRUE Multi-Speaker Enhancement Complete!")
        print(f"   All 3 speakers processed simultaneously in single forward pass")
        print(f"   Maximum efficiency with Universal GridNet architecture")
        
    except Exception as e:
        print(f"❌ Multi-speaker enhancement failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_true_multispk_enhancement()