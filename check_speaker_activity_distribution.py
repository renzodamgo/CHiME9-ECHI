#!/usr/bin/env python3

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import argparse
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train.echi import ECHIJoint, collate_fn_joint

def analyze_speaker_activity_distribution(min_active_speakers=1, max_samples=500):
    """
    Analyze the distribution of speaker activity patterns in the dataset.
    
    Args:
        min_active_speakers: Minimum number of active speakers required per segment
        max_samples: Maximum number of samples to analyze
    """
    print(f"🔍 Analyzing Speaker Activity Distribution (min_active={min_active_speakers})")
    print("=" * 70)
    
    # Temporarily modify the ECHIJoint class to use different min_active values
    config_params = {
        'noisy_signal': 'data/working_dir/train_segments/{dataset}/ha/{session}.ha.{pid}.{segment}.wav',
        'ref_signal': 'data/working_dir/train_segments/{dataset}/ha_ref/{session}.ha.{pid}.{segment}.wav', 
        'rainbow_signal': 'data/working_dir/participant/{dataset}/{pid}.wav',
        'sessions_file': 'data/chime9_echi/metadata/sessions.{dataset}.csv',
        'segments_file': 'data/chime9_echi/metadata/ref/{dataset}/{session}.ha.{pid}.csv'
    }
    
    # Create a custom dataset class that allows configurable min_active_speakers
    class ECHIJointAnalysis(ECHIJoint):
        def __init__(self, min_active_speakers=1, **kwargs):
            self.min_active_speakers = min_active_speakers
            super().__init__(**kwargs)
        
        def make_manifest(self):
            self.manifest = []
            end = False
            total_segments_checked = 0
            segments_with_activity = defaultdict(int)  # Track activity patterns
            
            for meta in self.metadata:
                try:
                    device_pos = int(meta[f"{self.audio_device}_pos"])
                except ValueError:
                    continue
                
                # PIDs seated at the other 3 positions
                pids = [meta[f"pos{i}"] for i in range(1, 5) if i != device_pos]
                
                # Load segments CSVs for each PID
                seg_lists = []
                for pid in pids:
                    seg_csv = self.segments_file.format(
                        dataset=self.subset,
                        session=meta["session"],
                        device=self.audio_device,
                        pid=pid,
                    )
                    if not Path(seg_csv).exists():
                        seg_lists = []  # force skip
                        break
                    with open(seg_csv, "r") as f:
                        import csv
                        segs = list(csv.DictReader(f, fieldnames=["index", "start", "end"]))
                    seg_lists.append({int(s["index"]): s for s in segs})
                
                if len(seg_lists) != len(pids):
                    continue
                
                # Use union to get all available segments
                all_idxs = set()
                for d in seg_lists:
                    all_idxs |= set(d.keys())
                if not all_idxs:
                    continue
                
                for idx in sorted(all_idxs):
                    total_segments_checked += 1
                    
                    # Check speaker activity for this segment
                    seg_ok = True
                    entry = {
                        "id": f'{meta["session"]}_{self.audio_device}_seg{idx:03d}',
                        "session": meta["session"],
                        "device": self.audio_device,
                        "idx": idx,
                        "pids": pids,
                        "noisy": None,
                        "target_all": [],
                        "spkid_all": [],
                        "speaker_active_mask": [],
                    }
                    
                    has_noisy = False
                    active_speakers = 0
                    
                    for j, pid in enumerate(pids):
                        noisy_path = self.signal_paths["noisy"].format(
                            dataset=self.subset,
                            session=meta["session"],
                            device=self.audio_device,
                            pid=pid,
                            segment=str(idx).zfill(3),
                        )
                        ref_path = self.signal_paths["target"].format(
                            dataset=self.subset,
                            session=meta["session"],
                            device=self.audio_device,
                            pid=pid,
                            segment=str(idx).zfill(3),
                        )
                        spk_path = self.signal_paths["spkid"].format(
                            dataset=self.subset, pid=pid
                        )
                        
                        # Check if this speaker has an active segment
                        speaker_has_segment = (idx in seg_lists[j] and 
                                             Path(noisy_path).exists() and 
                                             Path(ref_path).exists() and 
                                             Path(spk_path).exists())
                        
                        if speaker_has_segment:
                            entry["target_all"].append(ref_path)
                            entry["speaker_active_mask"].append(True)
                            active_speakers += 1
                            
                            if entry["noisy"] is None:
                                entry["noisy"] = noisy_path
                            has_noisy = True
                        else:
                            entry["target_all"].append(None)
                            entry["speaker_active_mask"].append(False)
                        
                        # Always need speaker ID for enrollment
                        if Path(spk_path).exists():
                            entry["spkid_all"].append(spk_path)
                        else:
                            seg_ok = False
                            break
                    
                    # Track activity patterns regardless of filtering
                    activity_pattern = tuple(entry["speaker_active_mask"])
                    segments_with_activity[activity_pattern] += 1
                    
                    # Apply minimum active speakers filter
                    if not has_noisy or active_speakers < self.min_active_speakers:
                        seg_ok = False
                    
                    if seg_ok:
                        self.manifest.append(entry)
                    
                    # Stop early if we've collected enough samples
                    if len(self.manifest) >= max_samples:
                        end = True
                        break
                        
                if end:
                    break
            
            # Print analysis results
            print(f"📊 TOTAL SEGMENTS ANALYZED: {total_segments_checked}")
            print(f"📊 SEGMENTS PASSING FILTER (≥{self.min_active_speakers} active): {len(self.manifest)}")
            print(f"📊 FILTER EFFICIENCY: {len(self.manifest)/total_segments_checked*100:.1f}%")
            print()
            print("SPEAKER ACTIVITY DISTRIBUTION (all segments):")
            print("-" * 50)
            
            for pattern, count in sorted(segments_with_activity.items(), key=lambda x: sum(x[0]), reverse=True):
                active_count = sum(pattern)
                percentage = count / total_segments_checked * 100
                status = "✅ INCLUDED" if active_count >= self.min_active_speakers else "❌ FILTERED"
                print(f"  {active_count}/3 speakers {pattern}: {count:4d} segments ({percentage:5.1f}%) {status}")
    
    # Test with different minimum active speaker requirements
    results = {}
    
    for min_active in [1, 2, 3]:
        print(f"\n{'='*70}")
        print(f"TESTING WITH MIN_ACTIVE_SPEAKERS = {min_active}")
        print(f"{'='*70}")
        
        dataset = ECHIJointAnalysis(
            min_active_speakers=min_active,
            subset="train",
            audio_device="ha",
            noisy_signal=config_params['noisy_signal'],
            ref_signal=config_params['ref_signal'], 
            rainbow_signal=config_params['rainbow_signal'],
            sessions_file=config_params['sessions_file'],
            segments_file=config_params['segments_file'],
            debug=False  # Don't limit samples artificially
        )
        
        results[min_active] = {
            'dataset_size': len(dataset),
            'manifest': dataset.manifest[:10] if dataset.manifest else []  # Store first 10 samples
        }
        
        print(f"🎯 FINAL DATASET SIZE: {len(dataset)} samples")
        
        # Test batch loading efficiency
        if len(dataset) > 0:
            print("\n🔄 Testing DataLoader Performance:")
            print("-" * 40)
            
            try:
                loader = DataLoader(
                    dataset, 
                    batch_size=4, 
                    shuffle=False, 
                    collate_fn=collate_fn_joint,
                    num_workers=0
                )
                
                batch = next(iter(loader))
                print(f"✅ Batch loaded successfully!")
                print(f"  Batch size: {len(batch['id'])}")
                print(f"  Speaker active mask shape: {batch['speaker_active_mask'].shape}")
                
                # Analyze activity patterns in this batch
                for i in range(len(batch['id'])):
                    active_pattern = batch['speaker_active_mask'][i].tolist()
                    active_count = sum(active_pattern)
                    print(f"  Sample {i}: {active_count}/3 active {active_pattern}")
                    
            except Exception as e:
                print(f"❌ DataLoader error: {e}")
        
        print(f"\n{'='*70}")
    
    # Summary comparison
    print("\n🎯 SUMMARY COMPARISON:")
    print("=" * 50)
    for min_active in [1, 2, 3]:
        size = results[min_active]['dataset_size']
        print(f"Min {min_active} active speakers: {size:,} samples")
    
    # Calculate relative sizes
    if results[1]['dataset_size'] > 0:
        print("\nRelative dataset sizes:")
        base_size = results[1]['dataset_size']
        for min_active in [1, 2, 3]:
            size = results[min_active]['dataset_size']
            percentage = size / base_size * 100
            print(f"  {min_active} speakers: {percentage:5.1f}% of single-speaker dataset")
    
    return results

def create_distribution_plot(results, output_file="speaker_activity_distribution.png"):
    """Create a visualization of dataset sizes by minimum active speakers."""
    min_speakers = [1, 2, 3]
    dataset_sizes = [results[k]['dataset_size'] for k in min_speakers]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(min_speakers, dataset_sizes, 
                   color=['lightblue', 'orange', 'lightgreen'],
                   alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, size in zip(bars, dataset_sizes):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{size:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Minimum Active Speakers Required')
    plt.ylabel('Dataset Size (number of samples)')
    plt.title('CHiME-9 ECHI Dataset Size vs Speaker Activity Requirements')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(min_speakers)
    
    # Add percentage annotations
    if dataset_sizes[0] > 0:
        for i, size in enumerate(dataset_sizes[1:], 1):
            pct = size / dataset_sizes[0] * 100
            plt.annotate(f'{pct:.1f}%', 
                        xy=(i+1, size), 
                        xytext=(i+1, size + dataset_sizes[0]*0.05),
                        ha='center', fontsize=10, color='red',
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Distribution plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Check speaker activity distribution in CHiME-9 ECHI dataset')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='Maximum number of samples to analyze per configuration')
    parser.add_argument('--plot', action='store_true',
                       help='Create distribution visualization plot')
    
    args = parser.parse_args()
    
    print("🎯 CHiME-9 ECHI Speaker Activity Distribution Analysis")
    print("=" * 60)
    print("This script analyzes how dataset size changes based on")
    print("minimum active speaker requirements.")
    print()
    
    # Run analysis
    results = analyze_speaker_activity_distribution(max_samples=args.max_samples)
    
    if args.plot and results:
        create_distribution_plot(results)
    
    print("\n✅ Analysis completed!")
    print("\nKey insights:")
    print("- Higher min_active requirements reduce dataset size")
    print("- Use min_active=2 for balanced training (avoids single-speaker dominance)")
    print("- Use min_active=1 for maximum data diversity")

if __name__ == "__main__":
    main()