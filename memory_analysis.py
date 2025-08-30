#!/usr/bin/env python3
"""
Memory analysis for multi-speaker GridNet training with 80GB GPU.
"""

import torch

def analyze_memory_requirements():
    """Calculate memory requirements for different batch sizes."""
    
    print("🧠 MEMORY ANALYSIS FOR 80GB GPU")
    print("=" * 60)
    
    # Model configuration from your setup
    config = {
        "n_srcs": 3,  # 3 speakers
        "emb_dim": 64,
        "n_layers": 3,
        "lstm_hidden_units": 64,
        "attn_n_head": 2,
        "attn_qk_output_channel": 64,
        "sample_rate": 16000,
        "n_fft": 128,
        "hop_length": 64,
        "max_audio_length": 4.0,  # seconds
    }
    
    # Calculate dimensions
    max_samples = int(config["sample_rate"] * config["max_audio_length"])  # 64000
    stft_frames = (max_samples - config["n_fft"]) // config["hop_length"] + 1  # ~999
    n_freqs = config["n_fft"] // 2 + 1  # 65
    
    print(f"📏 Audio Dimensions:")
    print(f"  Max samples per speaker: {max_samples:,}")
    print(f"  STFT frames (T): {stft_frames}")
    print(f"  STFT frequencies (F): {n_freqs}")
    print(f"  Number of speakers (K): {config['n_srcs']}")
    print()
    
    def estimate_memory_per_batch(batch_size):
        """Estimate memory usage per batch in GB."""
        B, K, T, F = batch_size, config["n_srcs"], stft_frames, n_freqs
        
        # 1. Input tensors (bfloat16 = 2 bytes)
        noisy_stft = B * 1 * T * F * 2 * 2  # [B, M, T, F, 2] complex
        spk_stft = B * K * T * F * 2 * 2    # [B, K, T, F, 2] complex
        targets = B * K * max_samples * 4    # [B, K, Tw] float32
        enrollments = B * K * max_samples * 4  # [B, K, Tr] float32
        
        input_memory = (noisy_stft + spk_stft + targets + enrollments) / (1024**3)
        
        # 2. Model parameters (~10M parameters estimated)
        model_params = 10_000_000 * 4 / (1024**3)  # float32
        
        # 3. Forward pass activations (intermediate tensors)
        # GridNet blocks with LSTM and attention
        hidden_features = B * config["emb_dim"] * T * F * 2  # Main features (bfloat16)
        lstm_hidden = B * K * config["lstm_hidden_units"] * T * 4  # LSTM states
        attention_weights = B * K * config["attn_n_head"] * T * T * 2  # Attention matrices
        speaker_embeddings = B * K * config["emb_dim"] * 4  # Speaker embeddings
        
        forward_memory = (hidden_features + lstm_hidden + attention_weights + speaker_embeddings) / (1024**3)
        
        # 4. Gradients (same size as model params)
        gradient_memory = model_params
        
        # 5. Optimizer states (Adam: 2x model params)
        optimizer_memory = model_params * 2
        
        # 6. Mixed precision overhead
        mixed_precision_overhead = forward_memory * 0.3
        
        total_memory = (input_memory + model_params + forward_memory + 
                       gradient_memory + optimizer_memory + mixed_precision_overhead)
        
        return {
            "total": total_memory,
            "input": input_memory,
            "model": model_params,
            "forward": forward_memory,
            "gradients": gradient_memory,
            "optimizer": optimizer_memory,
            "mixed_precision": mixed_precision_overhead,
        }
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 12, 16, 20, 24, 32]
    available_vram = 80.0  # GB
    safety_margin = 0.85   # Use 85% of VRAM for safety
    usable_vram = available_vram * safety_margin
    
    print(f"🎯 BATCH SIZE ANALYSIS (Available: {available_vram}GB, Usable: {usable_vram:.1f}GB)")
    print("-" * 60)
    
    optimal_batch_sizes = []
    
    for bs in batch_sizes:
        mem = estimate_memory_per_batch(bs)
        fits = mem["total"] <= usable_vram
        
        print(f"Batch Size {bs:2d}: {mem['total']:5.1f}GB {'✅' if fits else '❌'}")
        print(f"  ├─ Inputs:     {mem['input']:4.1f}GB")
        print(f"  ├─ Model:      {mem['model']:4.1f}GB") 
        print(f"  ├─ Forward:    {mem['forward']:4.1f}GB")
        print(f"  ├─ Gradients:  {mem['gradients']:4.1f}GB")
        print(f"  └─ Optimizer:  {mem['optimizer']:4.1f}GB")
        
        if fits:
            optimal_batch_sizes.append(bs)
        print()
    
    # Recommendations
    print("🚀 RECOMMENDATIONS:")
    
    if optimal_batch_sizes:
        max_batch = max(optimal_batch_sizes)
        conservative = max_batch // 2 if max_batch > 2 else 1
        
        print(f"  🎯 OPTIMAL (Max): batch_size={max_batch}")
        print(f"  🛡️  CONSERVATIVE: batch_size={conservative}")
        print(f"  ⚡ CURRENT:      batch_size=1 (very conservative)")
        print()
        
        speedup_max = max_batch / 1
        speedup_conservative = conservative / 1
        
        print(f"📈 EXPECTED SPEEDUPS:")
        print(f"  Max batch size: {speedup_max:.1f}x faster")
        print(f"  Conservative:   {speedup_conservative:.1f}x faster")
        print()
        
        print(f"⚙️  CONFIGURATION SUGGESTIONS:")
        print(f"  For dataloading.yaml:")
        print(f"    train: batch_size: {max_batch}  # Maximum")
        print(f"    dev:   batch_size: {conservative}     # Conservative")
        print()
        print(f"  Or conservative:")
        print(f"    train: batch_size: {conservative}  # Conservative") 
        print(f"    dev:   batch_size: {conservative}     # Conservative")
        
    else:
        print("  ❌ No batch size fits in available VRAM!")
        print("  Consider:")
        print("    - Reducing max audio length")
        print("    - Using gradient checkpointing")
        print("    - Reducing model parameters")
    
    print()
    print("⚠️  NOTES:")
    print("  - Estimates are conservative and may vary")
    print("  - Mixed precision training reduces memory usage")
    print("  - Peak memory occurs during backward pass")
    print("  - Start with conservative batch size and increase gradually")

if __name__ == "__main__":
    analyze_memory_requirements()