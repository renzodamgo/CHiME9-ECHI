#!/usr/bin/env python3
"""Compare audio quality between training samples and enhancement output"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_audio_file(filepath):
    """Analyze audio file and return statistics"""
    if not Path(filepath).exists():
        print(f"❌ File not found: {filepath}")
        return None
        
    try:
        audio, sr = torchaudio.load(filepath)
        
        # Convert to mono if needed
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        audio_np = audio.numpy().flatten()
        
        stats = {
            'filepath': filepath,
            'sample_rate': sr,
            'duration': len(audio_np) / sr,
            'shape': audio.shape,
            'mean': float(np.mean(audio_np)),
            'std': float(np.std(audio_np)),
            'min': float(np.min(audio_np)),
            'max': float(np.max(audio_np)),
            'rms': float(np.sqrt(np.mean(audio_np**2))),
            'dynamic_range': float(np.max(audio_np) - np.min(audio_np)),
            'zero_crossings': int(np.sum(np.diff(np.sign(audio_np)) != 0)),
        }
        
        # Spectral analysis (basic)
        fft = np.fft.rfft(audio_np)
        magnitude_db = 20 * np.log10(np.abs(fft) + 1e-10)
        
        stats['spectral_centroid'] = float(np.sum(magnitude_db * np.arange(len(magnitude_db))) / np.sum(magnitude_db))
        stats['high_freq_energy'] = float(np.mean(magnitude_db[len(magnitude_db)//2:]))  # Upper half
        stats['low_freq_energy'] = float(np.mean(magnitude_db[:len(magnitude_db)//2]))   # Lower half
        stats['spectral_rolloff'] = float(np.percentile(magnitude_db, 95))
        
        return stats, audio_np
        
    except Exception as e:
        print(f"❌ Error analyzing {filepath}: {e}")
        return None

def compare_audio_sets():
    """Compare training samples with enhancement output"""
    
    # Training sample files (epoch 48)
    base_path = "data/working_dir/experiments/ha-joint-uni/train_ha/train_samples/"
    sample_base = "epoch048_train_16_ha_seg027"
    
    training_files = {
        'noisy': f"{base_path}{sample_base}_noisy.wav",
        'processed_spk0': f"{base_path}{sample_base}_proc_spk0.wav",
        'processed_spk1': f"{base_path}{sample_base}_proc_spk1.wav", 
        'processed_spk2': f"{base_path}{sample_base}_proc_spk2.wav",
        'target_spk0': f"{base_path}{sample_base}_target_spk0.wav",
        'target_spk1': f"{base_path}{sample_base}_target_spk1.wav",
        'target_spk2': f"{base_path}{sample_base}_target_spk2.wav",
    }
    
    # Enhancement output - updated to use the FIXED version
    enhancement_file = "test_enhanced_PREPROCESSING_FIXED_dev_05_P189.wav"
    
    print("🔍 AUDIO QUALITY COMPARISON")
    print("=" * 60)
    
    # Analyze all files
    results = {}
    audio_data = {}
    
    # Training samples
    print("\n📊 TRAINING SAMPLES (Epoch 48):")
    for name, filepath in training_files.items():
        result = analyze_audio_file(filepath)
        if result:
            stats, audio = result
            results[name] = stats
            audio_data[name] = audio
            print(f"\n{name.upper()}:")
            print(f"  Duration: {stats['duration']:.2f}s")
            print(f"  RMS: {stats['rms']:.6f}")
            print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
            print(f"  Dynamic Range: {stats['dynamic_range']:.6f}")
            print(f"  High-freq energy: {stats['high_freq_energy']:.2f} dB")
            print(f"  Low-freq energy: {stats['low_freq_energy']:.2f} dB")
    
    # Enhancement output
    print(f"\n🚀 ENHANCEMENT OUTPUT:")
    result = analyze_audio_file(enhancement_file)
    if result:
        stats, audio = result
        results['enhancement'] = stats
        audio_data['enhancement'] = audio
        print(f"\nENHANCEMENT:")
        print(f"  Duration: {stats['duration']:.2f}s") 
        print(f"  RMS: {stats['rms']:.6f}")
        print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"  Dynamic Range: {stats['dynamic_range']:.6f}")
        print(f"  High-freq energy: {stats['high_freq_energy']:.2f} dB")
        print(f"  Low-freq energy: {stats['low_freq_energy']:.2f} dB")
    
    # Comparison analysis
    print(f"\n🔍 NOISE ANALYSIS:")
    print("=" * 60)
    
    if 'enhancement' in results:
        enh_stats = results['enhancement']
        
        # Compare with training processed outputs
        proc_files = [k for k in results.keys() if k.startswith('processed_')]
        if proc_files:
            print(f"\n📈 Enhancement vs Training Processed Outputs:")
            
            avg_proc_rms = np.mean([results[k]['rms'] for k in proc_files])
            avg_proc_high_freq = np.mean([results[k]['high_freq_energy'] for k in proc_files])
            avg_proc_range = np.mean([results[k]['dynamic_range'] for k in proc_files])
            
            print(f"  RMS - Enhancement: {enh_stats['rms']:.6f}, Avg Processed: {avg_proc_rms:.6f}")
            print(f"  High-freq - Enhancement: {enh_stats['high_freq_energy']:.2f} dB, Avg Processed: {avg_proc_high_freq:.2f} dB")
            print(f"  Dynamic Range - Enhancement: {enh_stats['dynamic_range']:.6f}, Avg Processed: {avg_proc_range:.6f}")
            
            # Noise indicators
            rms_ratio = enh_stats['rms'] / avg_proc_rms if avg_proc_rms > 0 else float('inf')
            hf_diff = enh_stats['high_freq_energy'] - avg_proc_high_freq
            
            print(f"\n⚠️  NOISE INDICATORS:")
            print(f"  RMS Ratio (enh/train): {rms_ratio:.2f} {'❌ HIGH' if rms_ratio > 2 else '✅ OK'}")
            print(f"  High-freq difference: {hf_diff:.2f} dB {'❌ NOISY' if hf_diff > 10 else '✅ OK'}")
            
        # Compare with targets (clean reference)
        target_files = [k for k in results.keys() if k.startswith('target_')]
        if target_files:
            print(f"\n🎯 Enhancement vs Clean Targets:")
            
            avg_target_rms = np.mean([results[k]['rms'] for k in target_files])
            avg_target_high_freq = np.mean([results[k]['high_freq_energy'] for k in target_files])
            
            print(f"  RMS - Enhancement: {enh_stats['rms']:.6f}, Avg Target: {avg_target_rms:.6f}")
            print(f"  High-freq - Enhancement: {enh_stats['high_freq_energy']:.2f} dB, Avg Target: {avg_target_high_freq:.2f} dB")
            
            target_rms_ratio = enh_stats['rms'] / avg_target_rms if avg_target_rms > 0 else float('inf')
            target_hf_diff = enh_stats['high_freq_energy'] - avg_target_high_freq
            
            print(f"\n🎯 QUALITY vs CLEAN REFERENCE:")
            print(f"  RMS Ratio (enh/clean): {target_rms_ratio:.2f}")
            print(f"  High-freq difference: {target_hf_diff:.2f} dB")

if __name__ == "__main__":
    compare_audio_sets()