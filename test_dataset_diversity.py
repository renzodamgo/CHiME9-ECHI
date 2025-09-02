#!/usr/bin/env python3

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train.echi import ECHIJoint

def test_dataset_diversity():
    """
    Test if the union-based loading is actually working in the current codebase.
    This will reveal if we're still limited to 3 sessions or if we get full diversity.
    """
    print("🔍 TESTING DATASET DIVERSITY WITH UNION-BASED LOADING")
    print("=" * 60)
    
    # Test configuration matching your training setup
    config_params = {
        'noisy_signal': 'data/working_dir/train_segments/{dataset}/ha/{session}.ha.{pid}.{segment}.wav',
        'ref_signal': 'data/working_dir/train_segments/{dataset}/ha_ref/{session}.ha.{pid}.{segment}.wav', 
        'rainbow_signal': 'data/working_dir/participant/{dataset}/{pid}.wav',
        'sessions_file': 'data/chime9_echi/metadata/sessions.{dataset}.csv',
        'segments_file': 'data/chime9_echi/metadata/ref/{dataset}/{session}.ha.{pid}.csv'
    }
    
    print("📝 Testing with debug=False (full dataset)...")
    
    dataset = ECHIJoint(
        subset='train',
        audio_device='ha',
        debug=False,  # Full dataset - no limitations
        **config_params
    )
    
    dataset_size = len(dataset)
    print(f"📊 Dataset size with debug=False: {dataset_size}")
    
    if dataset_size == 0:
        print("❌ ERROR: No samples found in dataset!")
        return
    
    # Analyze session diversity
    print(f"\n🔍 Analyzing session diversity (checking first {min(dataset_size, 200)} samples)...")
    
    sessions = set()
    segments_per_session = {}
    
    for i in range(min(dataset_size, 200)):
        sample_id = dataset.manifest[i]['id']
        session = sample_id.split('_')[0]
        sessions.add(session)
        
        if session not in segments_per_session:
            segments_per_session[session] = set()
        
        # Extract segment info
        parts = sample_id.split('_')
        if len(parts) >= 3:
            segment = parts[2]  # seg042, seg169, etc.
            segments_per_session[session].add(segment)
    
    print(f"📈 DIVERSITY ANALYSIS:")
    print(f"  Total unique sessions: {len(sessions)}")
    print(f"  Unique sessions found: {sorted(sessions)}")
    
    # Check if we're still limited to the 3 problematic sessions
    problem_sessions = {'train_07', 'train_11', 'train_16'}
    found_problem_sessions = problem_sessions.intersection(sessions)
    
    print(f"\n🚨 PROBLEM SESSION CHECK:")
    print(f"  Expected problem sessions: {sorted(problem_sessions)}")
    print(f"  Found problem sessions: {sorted(found_problem_sessions)}")
    print(f"  Are we LIMITED to just the 3 problem sessions? {sessions == problem_sessions}")
    
    if sessions == problem_sessions:
        print("  ❌ ISSUE CONFIRMED: Still limited to 3 sessions despite union loading!")
        print("  This suggests the union implementation is not working correctly.")
    else:
        print("  ✅ SUCCESS: Found more than 3 sessions - union loading is working!")
    
    # Show segments per session
    print(f"\n📋 SEGMENTS PER SESSION (first few):")
    for session in sorted(list(sessions)[:5]):
        if session in segments_per_session:
            segments = sorted(segments_per_session[session])
            print(f"  {session}: {len(segments)} segments - {segments[:5]}{'...' if len(segments) > 5 else ''}")
    
    # Test with debug=True for comparison
    print(f"\n🔄 Testing with debug=True (limited dataset)...")
    
    debug_dataset = ECHIJoint(
        subset='train',
        audio_device='ha',
        debug=True,  # Limited dataset
        **config_params
    )
    
    debug_size = len(debug_dataset)
    print(f"📊 Dataset size with debug=True: {debug_size}")
    
    # Summary
    print(f"\n🎯 SUMMARY:")
    print(f"  Debug=False dataset size: {dataset_size}")
    print(f"  Debug=True dataset size: {debug_size}")
    print(f"  Unique sessions (debug=False): {len(sessions)}")
    print(f"  Union loading working correctly: {'✅ YES' if len(sessions) > 3 else '❌ NO'}")
    
    if len(sessions) <= 3:
        print(f"\n🛠️  TROUBLESHOOTING:")
        print(f"  1. Check if files exist for other sessions")
        print(f"  2. Verify segment CSV files are present")
        print(f"  3. Check if union logic is actually running")
        print(f"  4. Verify training script is using updated code")

if __name__ == "__main__":
    test_dataset_diversity()