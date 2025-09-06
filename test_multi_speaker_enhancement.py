#!/usr/bin/env python3
"""Test multi-speaker enhancement with Universal GridNet (3 speakers)"""

import logging
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
import pandas as pd

from src.enhancement.joint_ha_uni import JointHaUni
from src.shared.core_utils import get_device

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')

def load_session_metadata(session_id):
    """Load session metadata to get participant info"""
    # Look for session metadata file
    metadata_files = list(Path("data/chime9_echi/metadata/").glob("sessions.*.csv"))
    
    for metadata_file in metadata_files:
        try:
            df = pd.read_csv(metadata_file)
            session_row = df[df['session'] == session_id]
            if not session_row.empty:
                row = session_row.iloc[0]
                return {
                    'session': session_id,
                    'ha_pos': row['ha_pos'],
                    'aria_pos': row.get('aria_pos', None),
                    'pos1': row['pos1'],
                    'pos2': row['pos2'], 
                    'pos3': row['pos3'],
                    'pos4': row['pos4']
                }
        except Exception as e:
            continue
    
    print(f"⚠️  Metadata not found for {session_id}, using default participants")
    # Fallback: use available participant files
    return None

def get_participant_enrollments(session_metadata):
    """Get 3 participant enrollments (excluding HA wearer)"""
    participant_dir = Path("data/chime9_echi/participant/dev/")
    
    if session_metadata:
        # Get all 4 participants
        all_participants = [
            session_metadata['pos1'],
            session_metadata['pos2'], 
            session_metadata['pos3'],
            session_metadata['pos4']
        ]
        ha_pos = session_metadata['ha_pos']
        
        # Remove HA wearer (they don't need to be separated)
        participants_to_enhance = []
        for i, pid in enumerate(all_participants, 1):
            if i != ha_pos:  # Skip HA wearer position
                participants_to_enhance.append(pid)
        
        print(f"👥 Session participants: {all_participants}")
        print(f"🎧 HA wearer at position {ha_pos}: {all_participants[ha_pos-1]}")
        print(f"🎯 Enhancing 3 speakers: {participants_to_enhance}")
        
    else:
        # Fallback: use first 3 available participants
        participant_files = list(participant_dir.glob("*.wav"))
        participants_to_enhance = [f.stem for f in participant_files[:3]]
        print(f"🎯 Using first 3 available participants: {participants_to_enhance}")
    
    # Load enrollment audios
    enrollments = []
    enrollment_info = []
    
    for pid in participants_to_enhance:
        enrollment_path = participant_dir / f"{pid}.wav"
        if enrollment_path.exists():
            try:
                audio, sr = torchaudio.load(str(enrollment_path))
                enrollments.append(audio)
                enrollment_info.append({
                    'participant_id': pid,
                    'path': str(enrollment_path),
                    'shape': audio.shape,
                    'duration': audio.shape[-1] / sr
                })
                print(f"  ✅ {pid}: {audio.shape}, {audio.shape[-1]/sr:.2f}s")
            except Exception as e:
                print(f"  ❌ Failed to load {pid}: {e}")
        else:
            print(f"  ❌ Missing enrollment: {enrollment_path}")
    
    return enrollments, enrollment_info

def test_multi_speaker_enhancement():
    """Test Universal GridNet with 3-speaker enhancement"""
    
    torch_device = get_device()
    print(f"Using device: {torch_device}")
    
    # Choose a dev session
    noisy_dir = Path("data/chime9_echi/ha/dev/")
    noisy_files = list(noisy_dir.glob("*.wav"))
    
    if not noisy_files:
        print("❌ No HA dev samples found")
        return
        
    noisy_file = noisy_files[0]  # Use first available
    session_id = noisy_file.stem.replace('.ha', '')  # e.g., "dev_05"
    
    print(f"🎵 Processing session: {session_id}")
    print(f"📁 Noisy file: {noisy_file}")
    
    # Load session metadata 
    session_metadata = load_session_metadata(session_id)
    
    # Get 3 participant enrollments
    enrollments, enrollment_info = get_participant_enrollments(session_metadata)
    
    if len(enrollments) < 3:
        print(f"❌ Need 3 enrollments, got {len(enrollments)}")
        return
    
    # Enhancement configuration
    enhance_config = {
        'inference_dir': 'data/working_dir/experiments/ha-joint-uni/',
        'config_path': 'data/working_dir/experiments/ha-joint-uni/train_ha/hydra/.hydra/config.yaml',
        'ckpt_path': 'data/working_dir/experiments/ha-joint-uni/train_ha/checkpoints/ha-joint-uni_048.pt',
        'audio_device': 'ha',
        'window_size': 10,  # 10 second windows
        'stride': 8,        # 2 second overlap
    }
    
    print("🔧 Initializing Universal GridNet for 3-speaker enhancement...")
    enhancement = JointHaUni(**enhance_config, torch_device=torch_device)
    
    # Load noisy mixture
    noisy_audio, noisy_fs = torchaudio.load(str(noisy_file))
    print(f"📊 Noisy mixture: {noisy_audio.shape}, {noisy_fs} Hz")
    
    # Move to device
    noisy_audio = noisy_audio.to(torch_device)
    enrollments = [e.to(torch_device) for e in enrollments]
    
    # Process each speaker separately (current single-speaker mode)
    print("\n🚀 Running 3-speaker enhancement (sequential processing)...")
    
    enhanced_speakers = []
    
    for i, (enrollment, info) in enumerate(zip(enrollments, enrollment_info)):
        print(f"\n--- Enhancing Speaker {i+1}: {info['participant_id']} ---")
        
        try:
            with torch.inference_mode():
                enhanced_audio = enhancement.process_session(
                    device_audio=noisy_audio,
                    device_fs=noisy_fs,
                    spkid_audio=enrollment,
                    spkid_fs=48000,  # All enrollments are 48kHz
                )
            
            enhanced_speakers.append(enhanced_audio)
            
            # Save individual enhanced speaker
            output_path = f"enhanced_3spk_{session_id}_speaker{i+1}_{info['participant_id']}.wav"
            
            if isinstance(enhanced_audio, torch.Tensor):
                enhanced_audio_cpu = enhanced_audio.detach().cpu().numpy()
            else:
                enhanced_audio_cpu = enhanced_audio
            
            sf.write(output_path, enhanced_audio_cpu, 16000)
            
            print(f"✅ Speaker {i+1} enhanced successfully")
            print(f"   Shape: {enhanced_audio.shape}")
            print(f"   Stats: mean={enhanced_audio.mean():.6f}, std={enhanced_audio.std():.6f}")
            print(f"   Range: [{enhanced_audio.min():.6f}, {enhanced_audio.max():.6f}]")
            print(f"   Saved: {output_path}")
            
        except Exception as e:
            print(f"❌ Failed to enhance speaker {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Multi-speaker enhancement completed!")
    print(f"   Processed {len(enhanced_speakers)} speakers")
    print(f"   Session: {session_id}")
    
    # Summary
    if session_metadata:
        print(f"\n📋 Session Summary:")
        print(f"   HA wearer: {session_metadata['pos1'] if session_metadata['ha_pos']==1 else session_metadata['pos2'] if session_metadata['ha_pos']==2 else session_metadata['pos3'] if session_metadata['ha_pos']==3 else session_metadata['pos4']} (pos {session_metadata['ha_pos']})")
        print(f"   Enhanced speakers: {[info['participant_id'] for info in enrollment_info]}")

if __name__ == "__main__":
    test_multi_speaker_enhancement()