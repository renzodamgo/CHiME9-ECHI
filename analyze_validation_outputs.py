#!/usr/bin/env python3
"""
Script to analyze processed outputs vs target speakers from HA joint training validation samples.
Computes SI-SDR and STOI metrics across epochs and speakers.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm
import argparse

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("Warning: PESQ not available. Install with: pip install pesq")

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    print("Warning: STOI not available. Install with: pip install pystoi")


def compute_si_sdr(reference, estimation):
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    
    Args:
        reference: clean reference signal
        estimation: enhanced/separated signal
    
    Returns:
        SI-SDR in dB
    """
    # Ensure same length
    min_len = min(len(reference), len(estimation))
    reference = reference[:min_len]
    estimation = estimation[:min_len]
    
    # Remove DC component
    reference = reference - np.mean(reference)
    estimation = estimation - np.mean(estimation)
    
    # Compute optimal scaling factor
    alpha = np.dot(estimation, reference) / np.dot(reference, reference)
    
    # Scale reference
    reference_scaled = alpha * reference
    
    # Compute SI-SDR
    signal_power = np.sum(reference_scaled ** 2)
    noise_power = np.sum((estimation - reference_scaled) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    si_sdr = 10 * np.log10(signal_power / noise_power)
    return si_sdr


def compute_stoi_metric(reference, estimation, fs=16000):
    """Compute STOI metric if available"""
    if not STOI_AVAILABLE:
        return None
    
    # Ensure same length
    min_len = min(len(reference), len(estimation))
    reference = reference[:min_len]
    estimation = estimation[:min_len]
    
    try:
        stoi_score = stoi(reference, estimation, fs, extended=False)
        return stoi_score
    except Exception as e:
        print(f"Error computing STOI: {e}")
        return None


def compute_pesq_metric(reference, estimation, fs=16000):
    """Compute PESQ metric if available"""
    if not PESQ_AVAILABLE:
        return None
    
    # Ensure same length
    min_len = min(len(reference), len(estimation))
    reference = reference[:min_len]
    estimation = estimation[:min_len]
    
    try:
        if fs == 16000:
            pesq_score = pesq(fs, reference, estimation, 'wb')
        else:
            pesq_score = pesq(fs, reference, estimation, 'nb')
        return pesq_score
    except Exception as e:
        print(f"Error computing PESQ: {e}")
        return None


def parse_filename(filename):
    """
    Parse validation sample filename to extract metadata
    
    Format: epoch{XX}_{dev_set}_{segment}_{type}_spk{X}.wav
    """
    pattern = r'epoch(\d+)_(.+?)_(.+?)_(target|proc)_spk(\d+)\.wav'
    match = re.match(pattern, filename)
    
    if match:
        epoch, dev_set, segment, file_type, speaker = match.groups()
        return {
            'epoch': int(epoch),
            'dev_set': dev_set,
            'segment': segment,
            'type': file_type,
            'speaker': int(speaker),
            'filename': filename
        }
    return None


def load_audio_safe(filepath, target_sr=16000):
    """Safely load audio file with error handling"""
    try:
        audio, sr = librosa.load(filepath, sr=target_sr)
        if len(audio) == 0:
            return None, sr
        return audio, sr
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, target_sr


def analyze_validation_samples(val_samples_dir, output_dir="validation_analysis"):
    """
    Main function to analyze validation samples
    """
    val_samples_path = Path(val_samples_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all wav files
    wav_files = list(val_samples_path.glob("*.wav"))
    print(f"Found {len(wav_files)} wav files")
    
    # Parse filenames
    parsed_files = []
    for wav_file in wav_files:
        parsed = parse_filename(wav_file.name)
        if parsed:
            parsed['filepath'] = wav_file
            parsed_files.append(parsed)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(parsed_files)
    print(f"Successfully parsed {len(df)} files")
    
    if len(df) == 0:
        print("No valid files found!")
        return
    
    # Find unique segments and epochs
    segments = df[df['type'] == 'target'][['dev_set', 'segment']].drop_duplicates()
    epochs = sorted(df['epoch'].unique())
    speakers = sorted(df['speaker'].unique())
    
    print(f"Analysis summary:")
    print(f"  Epochs: {len(epochs)} ({min(epochs)}-{max(epochs)})")
    print(f"  Segments: {len(segments)}")
    print(f"  Speakers: {speakers}")
    
    # Compute metrics for each comparison
    results = []
    
    for _, segment_info in tqdm(segments.iterrows(), total=len(segments), desc="Processing segments"):
        dev_set = segment_info['dev_set']
        segment = segment_info['segment']
        
        # Get target files for this segment (only from epoch 0)
        target_files = df[
            (df['dev_set'] == dev_set) & 
            (df['segment'] == segment) & 
            (df['type'] == 'target') & 
            (df['epoch'] == 0)
        ]
        
        for _, target_row in target_files.iterrows():
            speaker = target_row['speaker']
            target_path = target_row['filepath']
            
            # Load target audio
            target_audio, target_sr = load_audio_safe(target_path)
            if target_audio is None:
                continue
            
            # Find corresponding processed files across epochs
            proc_files = df[
                (df['dev_set'] == dev_set) & 
                (df['segment'] == segment) & 
                (df['type'] == 'proc') & 
                (df['speaker'] == speaker)
            ]
            
            for _, proc_row in proc_files.iterrows():
                epoch = proc_row['epoch']
                proc_path = proc_row['filepath']
                
                # Load processed audio
                proc_audio, proc_sr = load_audio_safe(proc_path)
                if proc_audio is None:
                    continue
                
                # Compute metrics
                si_sdr = compute_si_sdr(target_audio, proc_audio)
                stoi_score = compute_stoi_metric(target_audio, proc_audio, target_sr)
                pesq_score = compute_pesq_metric(target_audio, proc_audio, target_sr)
                
                result = {
                    'epoch': epoch,
                    'dev_set': dev_set,
                    'segment': segment,
                    'speaker': speaker,
                    'si_sdr': si_sdr,
                    'stoi': stoi_score,
                    'pesq': pesq_score,
                    'target_length': len(target_audio),
                    'proc_length': len(proc_audio)
                }
                results.append(result)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_csv_path = output_path / "validation_metrics.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to: {results_csv_path}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("="*50)
    
    if len(results_df) > 0:
        for metric in ['si_sdr', 'stoi', 'pesq']:
            if metric in results_df.columns and results_df[metric].notna().any():
                print(f"\n{metric.upper()} Statistics:")
                print(f"  Mean: {results_df[metric].mean():.3f}")
                print(f"  Std:  {results_df[metric].std():.3f}")
                print(f"  Min:  {results_df[metric].min():.3f}")
                print(f"  Max:  {results_df[metric].max():.3f}")
        
        # Show trends by epoch
        epoch_stats = results_df.groupby('epoch').agg({
            'si_sdr': ['mean', 'std'],
            'stoi': ['mean', 'std'],
            'pesq': ['mean', 'std']
        }).round(3)
        
        print("\nTrends by Epoch:")
        print(epoch_stats.head(10))
    
    return results_df, output_path


def create_visualizations(results_df, output_path):
    """Create visualization plots"""
    if len(results_df) == 0:
        print("No results to visualize")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. SI-SDR across epochs
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # SI-SDR by epoch
    if 'si_sdr' in results_df.columns:
        epoch_si_sdr = results_df.groupby('epoch')['si_sdr'].agg(['mean', 'std']).reset_index()
        axes[0, 0].plot(epoch_si_sdr['epoch'], epoch_si_sdr['mean'], 'b-o', linewidth=2, markersize=4)
        axes[0, 0].fill_between(epoch_si_sdr['epoch'], 
                               epoch_si_sdr['mean'] - epoch_si_sdr['std'],
                               epoch_si_sdr['mean'] + epoch_si_sdr['std'], alpha=0.3)
        axes[0, 0].set_title('SI-SDR vs Epoch')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('SI-SDR (dB)')
        axes[0, 0].grid(True)
    
    # STOI by epoch
    if 'stoi' in results_df.columns and results_df['stoi'].notna().any():
        epoch_stoi = results_df.groupby('epoch')['stoi'].agg(['mean', 'std']).reset_index()
        axes[0, 1].plot(epoch_stoi['epoch'], epoch_stoi['mean'], 'g-o', linewidth=2, markersize=4)
        axes[0, 1].fill_between(epoch_stoi['epoch'], 
                               epoch_stoi['mean'] - epoch_stoi['std'],
                               epoch_stoi['mean'] + epoch_stoi['std'], alpha=0.3)
        axes[0, 1].set_title('STOI vs Epoch')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('STOI')
        axes[0, 1].grid(True)
    
    # SI-SDR by speaker
    if 'si_sdr' in results_df.columns:
        sns.boxplot(data=results_df, x='speaker', y='si_sdr', ax=axes[1, 0])
        axes[1, 0].set_title('SI-SDR by Speaker')
        axes[1, 0].set_ylabel('SI-SDR (dB)')
    
    # SI-SDR by segment
    if 'si_sdr' in results_df.columns:
        segment_labels = [f"{row['dev_set']}_{row['segment']}" for _, row in 
                         results_df[['dev_set', 'segment']].drop_duplicates().iterrows()]
        results_df['segment_label'] = results_df['dev_set'] + '_' + results_df['segment']
        sns.boxplot(data=results_df, x='segment_label', y='si_sdr', ax=axes[1, 1])
        axes[1, 1].set_title('SI-SDR by Segment')
        axes[1, 1].set_ylabel('SI-SDR (dB)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_path = output_path / "validation_metrics_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plots saved to: {plot_path}")
    
    # 2. Heatmap of SI-SDR by epoch and segment
    if 'si_sdr' in results_df.columns:
        pivot_data = results_df.pivot_table(
            values='si_sdr', 
            index='epoch', 
            columns='segment_label', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0)
        plt.title('SI-SDR Heatmap: Epoch vs Segment')
        plt.ylabel('Epoch')
        plt.xlabel('Segment')
        plt.tight_layout()
        
        heatmap_path = output_path / "si_sdr_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Heatmap saved to: {heatmap_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze validation outputs from HA joint training")
    parser.add_argument("--val_dir", type=str, 
                       default="data/working_dir/experiments/ha-joint/train_ha/val_samples",
                       help="Path to validation samples directory")
    parser.add_argument("--output_dir", type=str, default="validation_analysis",
                       help="Output directory for results")
    parser.add_argument("--no_plots", action="store_true", 
                       help="Skip generating plots")
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.val_dir):
        print(f"Error: Directory {args.val_dir} does not exist!")
        return
    
    print(f"Analyzing validation samples in: {args.val_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Run analysis
    results_df, output_path = analyze_validation_samples(args.val_dir, args.output_dir)
    
    # Create visualizations
    if not args.no_plots and len(results_df) > 0:
        create_visualizations(results_df, output_path)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()