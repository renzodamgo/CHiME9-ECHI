#!/usr/bin/env python3

import sys
from pathlib import Path
from collections import Counter

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train.echi import ECHIJoint

def check_all_sessions():
    """
    Check which sessions are actually loaded across the entire dataset.
    """
    print("🔍 CHECKING ALL SESSIONS IN FULL DATASET")
    print("=" * 60)
    
    config_params = {
        'noisy_signal': 'data/working_dir/train_segments/{dataset}/ha/{session}.ha.{pid}.{segment}.wav',
        'ref_signal': 'data/working_dir/train_segments/{dataset}/ha_ref/{session}.ha.{pid}.{segment}.wav', 
        'rainbow_signal': 'data/working_dir/participant/{dataset}/{pid}.wav',
        'sessions_file': 'data/chime9_echi/metadata/sessions.{dataset}.csv',
        'segments_file': 'data/chime9_echi/metadata/ref/{dataset}/{session}.ha.{pid}.csv'
    }
    
    dataset = ECHIJoint(
        subset='train',
        audio_device='ha',
        debug=False,
        **config_params
    )
    
    print(f"📊 Total dataset size: {len(dataset)}")
    
    # Sample across the entire dataset to find all sessions
    sessions_counter = Counter()
    sample_indices = [
        0, 100, 500, 1000, 2000, 3000, 4000, 4900,  # Strategic sampling points
        len(dataset)//4, len(dataset)//2, len(dataset)*3//4, len(dataset)-1
    ]
    
    print(f"\n🔍 Sampling at strategic points in dataset:")
    
    for idx in sample_indices:
        if idx < len(dataset):
            sample_id = dataset.manifest[idx]['id']
            session = sample_id.split('_ha_seg')[0]
            sessions_counter[session] += 1
            print(f"  Index {idx:4d}: {sample_id} -> Session: {session}")
    
    print(f"\n📈 SESSIONS FOUND IN STRATEGIC SAMPLING:")
    print(f"  Unique sessions: {len(sessions_counter)}")
    print(f"  Sessions: {sorted(sessions_counter.keys())}")
    
    # Find where each session starts and ends
    print(f"\n🔍 Finding session boundaries...")
    
    current_session = None
    session_boundaries = {}
    
    # Sample every 200th item to find boundaries efficiently
    for idx in range(0, len(dataset), 200):
        sample_id = dataset.manifest[idx]['id']
        session = sample_id.split('_ha_seg')[0]
        
        if session != current_session:
            if current_session is not None:
                session_boundaries[current_session]['end'] = idx - 200
            
            session_boundaries[session] = {'start': idx, 'end': len(dataset)}
            current_session = session
    
    print(f"\n📋 SESSION BOUNDARIES (approximate):")
    for session in sorted(session_boundaries.keys()):
        start = session_boundaries[session]['start'] 
        end = session_boundaries[session].get('end', len(dataset))
        size = end - start
        print(f"  {session}: indices {start:4d}-{end:4d} (≈{size} samples)")
    
    # Check specific sessions from your training logs
    target_sessions = ['train_07', 'train_11', 'train_16']
    print(f"\n🎯 CHECKING TARGET SESSIONS FROM YOUR TRAINING:")
    
    found_targets = {}
    for session in target_sessions:
        if session in session_boundaries:
            start = session_boundaries[session]['start']
            sample_id = dataset.manifest[start]['id'] if start < len(dataset) else "N/A"
            found_targets[session] = {'index': start, 'sample': sample_id}
            print(f"  ✅ {session} found at index {start}: {sample_id}")
        else:
            print(f"  ❌ {session} NOT found in dataset")
    
    # Calculate where these sessions would appear in training
    print(f"\n📊 TRAINING IMPLICATIONS:")
    if found_targets:
        min_index = min(info['index'] for info in found_targets.values())
        print(f"  First target session appears at index: {min_index}")
        print(f"  This would be reached after: {min_index} samples")
        
        # Estimate batch and epoch
        batch_size = 1  # Based on your config
        batches_to_reach = min_index // batch_size
        print(f"  Batches to reach first target: {batches_to_reach}")
        
        if len(dataset) > 0:
            epoch_fraction = min_index / len(dataset)
            print(f"  Fraction of first epoch: {epoch_fraction:.2%}")
    
    return sessions_counter, session_boundaries

if __name__ == "__main__":
    check_all_sessions()