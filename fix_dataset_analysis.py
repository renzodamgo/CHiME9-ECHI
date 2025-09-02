#!/usr/bin/env python3

import sys
from pathlib import Path
from collections import Counter

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train.echi import ECHIJoint

def analyze_actual_dataset():
    """
    Proper analysis of dataset diversity with correct ID parsing.
    """
    print("🔍 CORRECTED DATASET DIVERSITY ANALYSIS")
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
    
    if len(dataset) == 0:
        print("❌ No samples found!")
        return
    
    # Analyze first 100 samples with correct parsing
    print(f"\n🔍 Analyzing diversity (first 100 samples)...")
    
    sessions = set()
    segments_per_session = Counter()
    raw_ids = []
    
    for i in range(min(len(dataset), 100)):
        # Get the raw ID from manifest
        sample_id = dataset.manifest[i]['id']
        raw_ids.append(sample_id)
        
        print(f"Sample {i:2d}: {sample_id}")
        
        # Parse session correctly - ID format should be like "train_01_ha_seg042"
        if '_ha_seg' in sample_id:
            # Split on '_ha_seg' to get session part
            session_part = sample_id.split('_ha_seg')[0]
            segment_part = 'seg' + sample_id.split('_ha_seg')[1]
        else:
            # Fallback parsing
            parts = sample_id.split('_')
            session_part = '_'.join(parts[:2]) if len(parts) >= 2 else sample_id
            segment_part = parts[-1] if len(parts) > 2 else 'unknown'
        
        sessions.add(session_part)
        segments_per_session[session_part] += 1
        
        # Show first few for debugging
        if i < 10:
            print(f"  Parsed session: '{session_part}', segment: '{segment_part}'")
    
    print(f"\n📈 DIVERSITY RESULTS:")
    print(f"  Unique sessions found: {len(sessions)}")
    print(f"  Sessions: {sorted(sessions)}")
    
    print(f"\n📋 SEGMENTS PER SESSION:")
    for session, count in segments_per_session.most_common(10):
        print(f"  {session}: {count} segments")
    
    # Check if we're getting the problematic sessions from your saved files
    problem_sessions = {'train_07', 'train_11', 'train_16'}
    found_sessions = set()
    
    # Extract session numbers correctly
    for session in sessions:
        if session.startswith('train_'):
            found_sessions.add(session)
    
    print(f"\n🚨 COMPARISON WITH YOUR SAVED FILES:")
    print(f"  Your saved files show: train_07, train_11, train_16")
    print(f"  Dataset shows: {sorted(found_sessions)}")
    
    overlap = problem_sessions.intersection(found_sessions)
    print(f"  Overlap: {sorted(overlap)}")
    
    if found_sessions == problem_sessions:
        print(f"  ❌ CONFIRMED: Still limited to same 3 sessions!")
    elif len(overlap) > 0 and len(found_sessions) > 3:
        print(f"  ⚠️  PARTIAL: Contains problem sessions but has more diversity")
    else:
        print(f"  ✅ GOOD: Different sessions than the problematic ones")
    
    # Check manifest details
    print(f"\n🔧 MANIFEST ANALYSIS:")
    print(f"  First few manifest entries:")
    for i in range(min(3, len(dataset.manifest))):
        entry = dataset.manifest[i]
        print(f"    {i}: session='{entry.get('session', 'N/A')}', id='{entry['id']}'")
    
    return sessions, segments_per_session

if __name__ == "__main__":
    analyze_actual_dataset()