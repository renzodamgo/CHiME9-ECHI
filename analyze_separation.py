#!/usr/bin/env python3
"""
Quick analysis script to check speaker separation from the logs you provided.
"""

import torch
import logging

def analyze_separation_from_logs():
    """Analyze the speaker separation metrics from the provided logs."""
    
    print("🔍 SPEAKER SEPARATION ANALYSIS FROM LOGS")
    print("=" * 50)
    
    # Data from your logs
    speaker_stats = {
        "Speaker 0": {
            "spec_mag_mean": 0.045075,
            "wav_std": 0.003391,
            "wav_min": -0.025982,
            "wav_max": 0.022069,
            "wav_range": 0.022069 - (-0.025982)
        },
        "Speaker 1": {
            "spec_mag_mean": 0.055159,
            "wav_std": 0.004511,
            "wav_min": -0.035759,
            "wav_max": 0.031054,
            "wav_range": 0.031054 - (-0.035759)
        },
        "Speaker 2": {
            "spec_mag_mean": 0.053123,
            "wav_std": 0.003826,
            "wav_min": -0.029926,
            "wav_max": 0.027333,
            "wav_range": 0.027333 - (-0.029926)
        }
    }
    
    print("📊 Individual Speaker Statistics:")
    for spk, stats in speaker_stats.items():
        print(f"  {spk}:")
        print(f"    Spectral Magnitude Mean: {stats['spec_mag_mean']:.6f}")
        print(f"    Waveform Std Dev: {stats['wav_std']:.6f}")
        print(f"    Amplitude Range: {stats['wav_range']:.6f}")
        print()
    
    # Analysis
    spec_means = [stats["spec_mag_mean"] for stats in speaker_stats.values()]
    wav_stds = [stats["wav_std"] for stats in speaker_stats.values()]
    wav_ranges = [stats["wav_range"] for stats in speaker_stats.values()]
    
    spec_mean_std = torch.tensor(spec_means).std().item()
    wav_std_std = torch.tensor(wav_stds).std().item()
    wav_range_std = torch.tensor(wav_ranges).std().item()
    
    print("🎯 SEPARATION ANALYSIS:")
    print(f"  Spectral Magnitude Diversity: {spec_mean_std:.6f}")
    print(f"  Waveform Std Diversity: {wav_std_std:.6f}")
    print(f"  Amplitude Range Diversity: {wav_range_std:.6f}")
    print()
    
    # Separation quality assessment
    separation_indicators = []
    
    # 1. Spectral diversity (different frequency content)
    if spec_mean_std > 0.005:  # Good threshold
        separation_indicators.append("✅ Good spectral diversity between speakers")
    else:
        separation_indicators.append("⚠️  Low spectral diversity - speakers might be too similar")
    
    # 2. Amplitude diversity (different energy levels)
    if wav_std_std > 0.0005:  # Reasonable threshold
        separation_indicators.append("✅ Good amplitude diversity between speakers")
    else:
        separation_indicators.append("⚠️  Low amplitude diversity - similar energy levels")
    
    # 3. Dynamic range diversity
    if wav_range_std > 0.005:  # Reasonable threshold
        separation_indicators.append("✅ Good dynamic range diversity")
    else:
        separation_indicators.append("⚠️  Similar dynamic ranges across speakers")
    
    print("🚨 SEPARATION QUALITY INDICATORS:")
    for indicator in separation_indicators:
        print(f"  {indicator}")
    print()
    
    # Overall assessment
    positive_indicators = sum(1 for ind in separation_indicators if ind.startswith("✅"))
    
    print("🏆 OVERALL ASSESSMENT:")
    if positive_indicators >= 2:
        print("  🎉 GOOD SPEAKER SEPARATION detected!")
        print("  The model appears to be successfully separating speakers into distinct outputs.")
        print("  Each speaker shows different spectral and amplitude characteristics.")
    elif positive_indicators == 1:
        print("  🤔 MODERATE SPEAKER SEPARATION detected.")
        print("  Some separation is occurring, but there's room for improvement.")
    else:
        print("  🚨 POOR SPEAKER SEPARATION detected!")
        print("  The outputs are too similar - possible collapse issue.")
    
    print()
    print("📝 RECOMMENDATIONS:")
    print("  1. Monitor cross-speaker correlation during training")
    print("  2. Check that speaker embeddings are distinct")
    print("  3. Ensure loss function is promoting separation")
    print("  4. Verify data loading provides different targets per speaker")

if __name__ == "__main__":
    analyze_separation_from_logs()