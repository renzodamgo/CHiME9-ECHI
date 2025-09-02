#!/usr/bin/env python3
"""
Calculate data coverage for rotating sample selection over 100 epochs.
"""
import random


def calculate_coverage_100_epochs():
    """Calculate how much of the dataset is covered over 100 epochs"""
    
    # Dataset parameters (matching actual training data)
    total_samples = 4944  # Training dataset size
    
    # Parameters from the implementation
    pool_size = min(20, total_samples // 50)  # 20 samples per epoch pool
    samples_per_epoch = min(6, pool_size)  # 6 samples saved per epoch
    
    print(f"Dataset Analysis for 100 Epochs:")
    print(f"=" * 50)
    print(f"Total training samples: {total_samples}")
    print(f"Pool size per epoch: {pool_size}")
    print(f"Samples saved per epoch: {samples_per_epoch}")
    print()
    
    # Simulate 100 epochs
    all_covered_samples = set()
    epoch_data = []
    
    for epoch in range(100):
        # Replicate the exact logic from update_epoch_samples
        random.seed(42 + epoch)
        
        # Generate random sample indices for this epoch
        sample_indices = random.sample(range(total_samples), pool_size)
        
        # Select samples for this epoch, rotating through the pool
        epoch_start_idx = (epoch * samples_per_epoch) % pool_size
        epoch_sample_indices = []
        for i in range(samples_per_epoch):
            idx = sample_indices[(epoch_start_idx + i) % pool_size]
            epoch_sample_indices.append(idx)
        
        # Track coverage
        epoch_samples = set(epoch_sample_indices)
        all_covered_samples.update(epoch_samples)
        
        epoch_data.append({
            'epoch': epoch,
            'samples_this_epoch': len(epoch_samples),
            'unique_so_far': len(all_covered_samples),
            'coverage_percent': (len(all_covered_samples) / total_samples) * 100
        })
    
    # Print milestone coverage
    milestones = [1, 5, 10, 20, 50, 100]
    print("Coverage Milestones:")
    print("-" * 50)
    for milestone in milestones:
        if milestone <= 100:
            data = epoch_data[milestone - 1]
            print(f"After {milestone:2d} epoch(s): {data['unique_so_far']:4d} samples ({data['coverage_percent']:5.2f}%)")
    
    print()
    
    # Final statistics
    final_coverage = len(all_covered_samples)
    coverage_percent = (final_coverage / total_samples) * 100
    
    print("Final Coverage Analysis:")
    print("=" * 50)
    print(f"Samples covered over 100 epochs: {final_coverage:,}")
    print(f"Total dataset size: {total_samples:,}")
    print(f"Coverage percentage: {coverage_percent:.2f}%")
    print()
    
    # Additional insights
    total_sample_saves = 100 * samples_per_epoch  # 600 total saves
    unique_ratio = final_coverage / total_sample_saves
    print("Efficiency Metrics:")
    print("-" * 30)
    print(f"Total sample saves: {total_sample_saves}")
    print(f"Unique samples ratio: {unique_ratio:.3f}")
    print(f"Average reuse per sample: {total_sample_saves / final_coverage:.2f}x")
    print()
    
    # Comparison with old approach
    old_coverage = 3  # Old approach only ever saved 3 fixed samples
    improvement = final_coverage / old_coverage
    print("Comparison with Original Approach:")
    print("-" * 35)
    print(f"Original approach coverage: {old_coverage} samples (0.06%)")
    print(f"New approach coverage: {final_coverage} samples ({coverage_percent:.2f}%)")
    print(f"Improvement factor: {improvement:.0f}x more diverse")
    
    return final_coverage, coverage_percent


if __name__ == "__main__":
    calculate_coverage_100_epochs()