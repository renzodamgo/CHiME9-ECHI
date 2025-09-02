#!/usr/bin/env python3

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import argparse
from collections import Counter

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train.echi import ECHIJoint, collate_fn_joint

def test_union_loading(config_params):
    """Test the union-based loading implementation."""
    print("🔍 Testing Union-Based Data Loading")
    print("=" * 50)
    
    # Create dataset with debug mode
    dataset = ECHIJoint(
        subset="train",
        audio_device="ha",
        noisy_signal=config_params['noisy_signal'],
        ref_signal=config_params['ref_signal'], 
        rainbow_signal=config_params['rainbow_signal'],
        sessions_file=config_params['sessions_file'],
        segments_file=config_params['segments_file'],
        debug=True  # Limit to 10 samples for testing
    )
    
    print(f"📊 Dataset size: {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("❌ No samples found in dataset!")
        return
    
    # Analyze segment diversity
    session_segments = Counter()
    speaker_activity_patterns = Counter()
    
    for i in range(min(len(dataset), 30)):  # Check first 30 samples
        try:
            sample = dataset[i]
            sample_id = sample['id']
            
            # Extract session and segment info from ID
            parts = sample_id.split('_')
            if len(parts) >= 3:
                session = parts[0]
                segment = parts[2] if parts[2].startswith('seg') else 'unknown'
                session_segments[f"{session}_{segment}"] += 1
            
            # Analyze speaker activity pattern
            active_mask = sample['speaker_active_mask']
            pattern = tuple(active_mask.tolist())
            speaker_activity_patterns[pattern] += 1
            
            print(f"Sample {i:2d}: {sample_id}")
            print(f"  Active speakers: {active_mask.tolist()}")
            print(f"  Target shape: {sample['target_all'].shape}")
            print(f"  Speaker ID shape: {sample['spkid_all'].shape}")
            
            # Check for silent speakers (should have zero targets)
            for k in range(len(active_mask)):
                if not active_mask[k]:
                    target_energy = (sample['target_all'][k] ** 2).mean().item()
                    print(f"  Speaker {k} (silent): target energy = {target_energy:.6f}")
            
        except Exception as e:
            print(f"❌ Error loading sample {i}: {e}")
            continue
    
    print("\n📈 SEGMENT DIVERSITY ANALYSIS:")
    print("-" * 30)
    print(f"Unique session-segments: {len(session_segments)}")
    for seg, count in session_segments.most_common(10):
        print(f"  {seg}: {count} samples")
    
    print("\n👥 SPEAKER ACTIVITY PATTERNS:")
    print("-" * 30)
    for pattern, count in speaker_activity_patterns.items():
        active_speakers = sum(pattern)
        print(f"  {active_speakers}/3 active {pattern}: {count} samples")
    
    # Test data loader
    print("\n🔄 Testing DataLoader with collate_fn_joint:")
    print("-" * 40)
    
    try:
        loader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False, 
            collate_fn=collate_fn_joint,
            num_workers=0  # Single process for debugging
        )
        
        batch = next(iter(loader))
        print(f"✅ Batch loaded successfully!")
        print(f"  Batch size: {len(batch['id'])}")
        print(f"  Noisy shape: {batch['noisy'].shape}")
        print(f"  Target all shape: {batch['target_all'].shape}") 
        print(f"  Speaker ID all shape: {batch['spkid_all'].shape}")
        print(f"  Speaker active mask shape: {batch['speaker_active_mask'].shape}")
        print(f"  Active mask sample: {batch['speaker_active_mask'][0].tolist()}")
        
    except Exception as e:
        print(f"❌ DataLoader error: {e}")
    
    print("\n✅ Union loading test completed!")

def main():
    parser = argparse.ArgumentParser(description='Test union-based data loading')
    
    args = parser.parse_args()
    
    # Use hardcoded paths based on the config structure
    config_params = {
        'noisy_signal': 'data/working_dir/train_segments/{dataset}/ha/{session}.ha.{pid}.{segment}.wav',
        'ref_signal': 'data/working_dir/train_segments/{dataset}/ha_ref/{session}.ha.{pid}.{segment}.wav', 
        'rainbow_signal': 'data/working_dir/participant/{dataset}/{pid}.wav',
        'sessions_file': 'data/chime9_echi/metadata/sessions.{dataset}.csv',
        'segments_file': 'data/chime9_echi/metadata/ref/{dataset}/{session}.ha.{pid}.csv'
    }
    
    print("📝 Using config:")
    for key, value in config_params.items():
        print(f"  {key}: {value}")
    print()
    
    test_union_loading(config_params)

if __name__ == "__main__":
    main()