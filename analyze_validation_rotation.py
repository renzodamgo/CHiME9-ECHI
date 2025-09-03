#!/usr/bin/env python3
"""
Analyze validation sample rotation strategy and its implications.
"""
import random


def analyze_validation_strategies():
    """Compare different validation sampling strategies"""
    
    # Validation dataset parameters (from logs: 2094 dev samples)
    val_total_samples = 2094
    
    print("🔍 VALIDATION SAMPLE STRATEGY ANALYSIS")
    print("=" * 60)
    print(f"Validation dataset size: {val_total_samples} samples")
    print()
    
    # Strategy 1: Current rotating approach
    print("📊 STRATEGY 1: ROTATING VALIDATION SAMPLES")
    print("-" * 40)
    
    pool_size = min(20, val_total_samples // 50)  # Same logic as training
    samples_per_epoch = min(6, pool_size)
    
    # Calculate coverage over 100 epochs
    all_covered_samples = set()
    for epoch in range(100):
        random.seed(42 + epoch)  # Same seed logic
        sample_indices = random.sample(range(val_total_samples), pool_size)
        epoch_start_idx = (epoch * samples_per_epoch) % pool_size
        epoch_sample_indices = []
        for i in range(samples_per_epoch):
            idx = sample_indices[(epoch_start_idx + i) % pool_size]
            epoch_sample_indices.append(idx)
        all_covered_samples.update(epoch_sample_indices)
    
    rotating_coverage = len(all_covered_samples)
    rotating_percent = (rotating_coverage / val_total_samples) * 100
    
    print(f"Pool size per epoch: {pool_size}")
    print(f"Samples saved per epoch: {samples_per_epoch}")
    print(f"Coverage over 100 epochs: {rotating_coverage} samples ({rotating_percent:.2f}%)")
    print()
    
    # Strategy 2: Fixed validation samples
    print("📌 STRATEGY 2: FIXED VALIDATION SAMPLES")
    print("-" * 40)
    fixed_samples = 6  # Same number as rotating, but fixed
    fixed_coverage = fixed_samples
    fixed_percent = (fixed_coverage / val_total_samples) * 100
    
    print(f"Fixed samples (never change): {fixed_samples}")
    print(f"Coverage over 100 epochs: {fixed_coverage} samples ({fixed_percent:.2f}%)")
    print()
    
    # Strategy 3: Hybrid approach
    print("🎯 STRATEGY 3: HYBRID APPROACH")
    print("-" * 40)
    core_fixed = 3  # Keep 3 consistent samples for direct comparison
    rotating_additional = 3  # Add 3 rotating samples for diversity
    
    # Calculate rotating part coverage (smaller pool)
    hybrid_covered = set()
    for epoch in range(100):
        random.seed(42 + epoch)
        sample_indices = random.sample(range(val_total_samples), 15)  # Smaller pool
        epoch_indices = sample_indices[:rotating_additional]
        hybrid_covered.update(epoch_indices)
    
    hybrid_total_coverage = core_fixed + len(hybrid_covered)
    hybrid_percent = (hybrid_total_coverage / val_total_samples) * 100
    
    print(f"Core fixed samples: {core_fixed} (for consistent tracking)")
    print(f"Additional rotating: {rotating_additional} per epoch")
    print(f"Total coverage over 100 epochs: {hybrid_total_coverage} samples ({hybrid_percent:.2f}%)")
    print()
    
    # Pros and Cons Analysis
    print("⚖️  STRATEGY COMPARISON")
    print("=" * 60)
    
    strategies = [
        {
            'name': 'Rotating Validation',
            'coverage': rotating_coverage,
            'percent': rotating_percent,
            'pros': [
                'High diversity - see many validation scenarios',
                'Better generalization assessment',
                'Catches edge cases in validation data',
                'Consistent with training approach'
            ],
            'cons': [
                'Harder to track consistent progress',
                'Validation curves may be noisier',
                'Cannot directly compare same samples over time',
                'May mask overfitting to specific samples'
            ]
        },
        {
            'name': 'Fixed Validation',
            'coverage': fixed_coverage,
            'percent': fixed_percent,
            'pros': [
                'Perfect for tracking learning progress',
                'Smooth validation curves',
                'Direct epoch-to-epoch comparison',
                'Standard practice in ML'
            ],
            'cons': [
                'Very limited diversity (0.29% coverage)',
                'May not represent full validation distribution',
                'Could overfit to specific samples',
                'Missing edge cases'
            ]
        },
        {
            'name': 'Hybrid Approach',
            'coverage': hybrid_total_coverage,
            'percent': hybrid_percent,
            'pros': [
                'Best of both worlds',
                'Consistent tracking + diversity',
                'Balanced coverage and comparison',
                'Robust validation assessment'
            ],
            'cons': [
                'Slightly more complex implementation',
                'Need to track two types of samples',
                'Moderate increase in storage'
            ]
        }
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"{i}. {strategy['name'].upper()}")
        print(f"   Coverage: {strategy['coverage']} samples ({strategy['percent']:.2f}%)")
        print(f"   ✅ Pros:")
        for pro in strategy['pros']:
            print(f"      • {pro}")
        print(f"   ❌ Cons:")
        for con in strategy['cons']:
            print(f"      • {con}")
        print()
    
    # Recommendation
    print("🎯 RECOMMENDATION")
    print("=" * 30)
    print("HYBRID APPROACH is recommended because:")
    print("• Maintains consistent progress tracking (like standard ML practice)")
    print("• Adds diversity for robust validation assessment")  
    print("• Balances the trade-offs of both pure approaches")
    print("• Provides richer insights without losing comparability")
    print()
    print("Implementation: 3 fixed + 3 rotating validation samples per epoch")


if __name__ == "__main__":
    analyze_validation_strategies()