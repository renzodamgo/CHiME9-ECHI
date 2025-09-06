#!/usr/bin/env python3
"""Check audio file channel configuration"""

import torch
import torchaudio
from pathlib import Path

def check_audio_files():
    """Check channel configuration of audio files"""
    
    print("🔍 AUDIO FILE CHANNEL ANALYSIS")
    print("=" * 50)
    
    # Check HA device files
    ha_dir = Path("data/chime9_echi/ha/dev/")
    ha_files = list(ha_dir.glob("*.wav"))
    
    print(f"\n📊 HA DEVICE FILES:")
    for i, file in enumerate(ha_files[:5]):  # Check first 5
        try:
            audio, sr = torchaudio.load(str(file))
            channels, samples = audio.shape
            duration = samples / sr
            
            print(f"  {file.name}:")
            print(f"    Channels: {channels}")
            print(f"    Sample Rate: {sr} Hz") 
            print(f"    Duration: {duration:.2f}s")
            print(f"    Shape: {audio.shape}")
            
            if channels != 4:
                print(f"    ⚠️  Expected 4 channels, got {channels}")
            else:
                print(f"    ✅ Correct 4-channel format")
                
        except Exception as e:
            print(f"    ❌ Error loading {file.name}: {e}")
        print()
    
    # Check participant/rainbow files  
    rainbow_dir = Path("data/chime9_echi/participant/dev/")
    rainbow_files = list(rainbow_dir.glob("*.wav"))
    
    print(f"\n🌈 PARTICIPANT/RAINBOW FILES:")
    for i, file in enumerate(rainbow_files[:3]):  # Check first 3
        try:
            audio, sr = torchaudio.load(str(file))
            channels, samples = audio.shape
            duration = samples / sr
            
            print(f"  {file.name}:")
            print(f"    Channels: {channels}")
            print(f"    Sample Rate: {sr} Hz")
            print(f"    Duration: {duration:.2f}s") 
            print(f"    Shape: {audio.shape}")
            
            if channels != 1:
                print(f"    ⚠️  Expected 1 channel (mono), got {channels}")
            else:
                print(f"    ✅ Correct mono format")
                
        except Exception as e:
            print(f"    ❌ Error loading {file.name}: {e}")
        print()
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print(f"  Total HA files checked: {min(len(ha_files), 5)}")
    print(f"  Total Rainbow files checked: {min(len(rainbow_files), 3)}")
    
    # Check model expectation vs reality
    print(f"\n🎯 MODEL EXPECTATIONS:")
    print(f"  Device audio (HA): 4 channels expected")
    print(f"  Speaker audio (Rainbow): 1 channel expected")
    
    print(f"\n🚨 CHANNEL MISMATCH ANALYSIS:")
    print(f"  The warning 'Found 1 channels in audio, but want to return 4' suggests:")
    print(f"  1. Some HA files are mono instead of 4-channel")
    print(f"  2. Or there's a bug in channel handling")
    print(f"  3. Check if correct audio files are being loaded")

if __name__ == "__main__":
    check_audio_files()