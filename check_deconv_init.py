#!/usr/bin/env python3
"""
Check the default initialization of ConvTranspose2d layer to understand
why channels 2-6 might have poor performance.
"""

import torch
import torch.nn as nn
import numpy as np

def analyze_deconv_initialization():
    """Analyze default initialization of ConvTranspose2d"""
    
    print("🔍 ANALYZING DECONV LAYER INITIALIZATION")
    print("=" * 50)
    
    # Simulate the deconv layer from GridNet
    emb_dim = 48  # From model config
    n_srcs = 3    # 3 speakers
    ks = (3, 3)   # kernel size
    
    # Create the deconv layer (same as line 110)
    deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=(1, 1))
    
    print(f"📐 Layer configuration:")
    print(f"   Input channels: {emb_dim}")
    print(f"   Output channels: {n_srcs * 2} (2 per speaker)")
    print(f"   Kernel size: {ks}")
    
    # Check weight initialization
    weight = deconv.weight.data  # [in_channels, out_channels, H, W]
    bias = deconv.bias.data      # [out_channels]
    
    print(f"\n📊 Weight tensor shape: {weight.shape}")
    print(f"📊 Bias tensor shape: {bias.shape}")
    
    # Analyze per-output-channel statistics
    print(f"\n🎯 PER-SPEAKER CHANNEL ANALYSIS:")
    print(f"   Speaker 0 channels: 0-1")
    print(f"   Speaker 1 channels: 2-3") 
    print(f"   Speaker 2 channels: 4-5")
    
    for spk in range(n_srcs):
        ch_start = spk * 2
        ch_end = ch_start + 2
        
        # Weight statistics for this speaker's channels
        spk_weights = weight[:, ch_start:ch_end, :, :]  # [48, 2, 3, 3]
        spk_bias = bias[ch_start:ch_end]                # [2]
        
        print(f"\n   Speaker {spk} (channels {ch_start}:{ch_end}):")
        print(f"     Weight mean: {spk_weights.mean().item():.6f}")
        print(f"     Weight std:  {spk_weights.std().item():.6f}")
        print(f"     Weight min:  {spk_weights.min().item():.6f}")
        print(f"     Weight max:  {spk_weights.max().item():.6f}")
        print(f"     Bias mean:   {spk_bias.mean().item():.6f}")
        print(f"     Bias std:    {spk_bias.std().item():.6f}")
    
    # Check if initialization is uniform across channels
    print(f"\n🔍 INITIALIZATION UNIFORMITY CHECK:")
    
    # Compute per-output-channel statistics
    channel_means = []
    channel_stds = []
    
    for ch in range(n_srcs * 2):
        ch_weights = weight[:, ch, :, :]  # [48, 3, 3]
        channel_means.append(ch_weights.mean().item())
        channel_stds.append(ch_weights.std().item())
    
    channel_means = np.array(channel_means)
    channel_stds = np.array(channel_stds)
    
    print(f"   Channel means: {channel_means}")
    print(f"   Channel stds:  {channel_stds}")
    
    # Check if means are similar across channels
    mean_variation = channel_means.std()
    std_variation = channel_stds.std()
    
    print(f"\n   Mean variation across channels: {mean_variation:.6f}")
    print(f"   Std variation across channels:  {std_variation:.6f}")
    
    # Flags
    uniform_means = mean_variation < 0.01
    uniform_stds = std_variation < 0.01
    
    print(f"\n✅ Uniform initialization:")
    print(f"   Means uniform: {'YES' if uniform_means else 'NO'} (variation: {mean_variation:.6f})")
    print(f"   Stds uniform:  {'YES' if uniform_stds else 'NO'} (variation: {std_variation:.6f})")
    
    # Test with multiple seeds to see if this is consistent
    print(f"\n🎲 TESTING INITIALIZATION CONSISTENCY:")
    
    mean_vars = []
    std_vars = []
    
    for seed in range(5):
        torch.manual_seed(seed)
        test_deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=(1, 1))
        test_weight = test_deconv.weight.data
        
        test_channel_means = []
        test_channel_stds = []
        
        for ch in range(n_srcs * 2):
            ch_weights = test_weight[:, ch, :, :]
            test_channel_means.append(ch_weights.mean().item())
            test_channel_stds.append(ch_weights.std().item())
        
        test_channel_means = np.array(test_channel_means)
        test_channel_stds = np.array(test_channel_stds)
        
        mean_vars.append(test_channel_means.std())
        std_vars.append(test_channel_stds.std())
        
        print(f"   Seed {seed}: mean_var={test_channel_means.std():.6f}, std_var={test_channel_stds.std():.6f}")
    
    print(f"\n📈 SUMMARY ACROSS SEEDS:")
    print(f"   Average mean variation: {np.mean(mean_vars):.6f} ± {np.std(mean_vars):.6f}")
    print(f"   Average std variation:  {np.mean(std_vars):.6f} ± {np.std(std_vars):.6f}")
    
    consistent_init = np.mean(mean_vars) < 0.01 and np.mean(std_vars) < 0.01
    
    print(f"\n🎯 CONCLUSION:")
    if consistent_init:
        print("   ✅ Initialization appears uniform across channels")
        print("   ❓ Problem likely elsewhere (gradient flow, loss weighting, etc.)")
    else:
        print("   ❌ Initialization is NOT uniform across channels")
        print("   🚨 This could explain speaker performance differences!")
    
    # Check what initialization method PyTorch uses
    print(f"\n📚 PYTORCH DEFAULT INITIALIZATION:")
    print(f"   ConvTranspose2d uses: Kaiming Uniform initialization")
    print(f"   Formula: U(-√k, √k) where k = 1/(in_channels * kernel_area)")
    
    k = 1.0 / (emb_dim * ks[0] * ks[1])
    bound = np.sqrt(k)
    print(f"   k = 1/({emb_dim} * {ks[0]} * {ks[1]}) = {k:.6f}")
    print(f"   bound = √k = ±{bound:.6f}")
    print(f"   Expected std ≈ {bound/np.sqrt(3):.6f}")

if __name__ == "__main__":
    analyze_deconv_initialization()