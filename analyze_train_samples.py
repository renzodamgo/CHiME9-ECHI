#!/usr/bin/env python3
"""
Script to analyze training samples in the CHiME9-ECHI experiment directory.
Analyzes amplitude trends, audio quality, and silent audio detection across epochs.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import soundfile as sf
from pathlib import Path
import re
from typing import Dict, List, Tuple
import argparse


def parse_filename(filepath: str) -> Dict[str, str]:
    """Parse training sample filename to extract metadata."""
    filename = os.path.basename(filepath)
    
    # Pattern: epoch000_train_08_ha_seg004_proc_spk0.wav
    pattern = r'epoch(\d+)_train_(\d+)_ha_seg(\d+)_proc_spk(\d+)\.wav'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'epoch': int(match.group(1)),
            'train_id': match.group(2),
            'segment': match.group(3),
            'speaker': int(match.group(4)),
            'scene_id': f"{match.group(2)}_ha_seg{match.group(3)}",
            'filepath': filepath,
            'filename': filename
        }
    else:
        return None


def analyze_spectrogram(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """Analyze spectrogram for information loss detection."""
    try:
        # Compute STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512, win_length=2048)
        magnitude = np.abs(stft)
        
        # Spectral statistics
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
        
        # High-frequency content analysis (detect information loss)
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        # Energy distribution across frequency bands
        total_energy = np.sum(magnitude**2)
        if total_energy > 0:
            low_freq_energy = np.sum(magnitude[freq_bins < 1000]**2) / total_energy  # 0-1kHz
            mid_freq_energy = np.sum(magnitude[(freq_bins >= 1000) & (freq_bins < 4000)]**2) / total_energy  # 1-4kHz
            high_freq_energy = np.sum(magnitude[freq_bins >= 4000]**2) / total_energy  # 4kHz+
        else:
            low_freq_energy = mid_freq_energy = high_freq_energy = 0.0
        
        # Spectral entropy (measure of spectral complexity)
        power_spectrum = np.mean(magnitude**2, axis=1)
        power_spectrum = power_spectrum / (np.sum(power_spectrum) + 1e-12)  # Normalize
        spectral_entropy = -np.sum(power_spectrum * np.log2(power_spectrum + 1e-12))
        
        # Peak-to-average ratio in frequency domain
        spectral_par = np.max(power_spectrum) / (np.mean(power_spectrum) + 1e-12)
        
        # High-frequency rolloff detection (detect low-pass filtering effects)
        nyquist = sr / 2
        high_freq_threshold = 0.8 * nyquist  # 80% of Nyquist frequency
        high_freq_mask = freq_bins >= high_freq_threshold
        if np.any(high_freq_mask):
            high_freq_power = np.mean(magnitude[high_freq_mask]**2)
            total_power = np.mean(magnitude**2)
            high_freq_ratio = high_freq_power / (total_power + 1e-12)
        else:
            high_freq_ratio = 0.0
        
        # Detect potential information loss indicators
        is_low_complexity = spectral_entropy < 8.0  # Low spectral entropy
        is_missing_highs = high_freq_ratio < 0.01   # Very little high-frequency content
        is_overly_smooth = spectral_flatness < 0.1  # Very non-flat spectrum (overly tonal)
        
        return {
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff,
            'spectral_flatness': spectral_flatness,
            'low_freq_energy': low_freq_energy,
            'mid_freq_energy': mid_freq_energy,
            'high_freq_energy': high_freq_energy,
            'spectral_entropy': spectral_entropy,
            'spectral_par': spectral_par,
            'high_freq_ratio': high_freq_ratio,
            'is_low_complexity': is_low_complexity,
            'is_missing_highs': is_missing_highs,
            'is_overly_smooth': is_overly_smooth,
        }
        
    except Exception as e:
        print(f"Warning: Spectral analysis failed: {e}")
        return {
            'spectral_centroid': 0.0, 'spectral_bandwidth': 0.0, 'spectral_rolloff': 0.0,
            'spectral_flatness': 0.0, 'low_freq_energy': 0.0, 'mid_freq_energy': 0.0,
            'high_freq_energy': 0.0, 'spectral_entropy': 0.0, 'spectral_par': 0.0,
            'high_freq_ratio': 0.0, 'is_low_complexity': False, 'is_missing_highs': False,
            'is_overly_smooth': False
        }


def analyze_audio_file(filepath: str) -> Dict[str, float]:
    """Analyze a single audio file and return statistics."""
    try:
        # Load audio file
        audio, sr = librosa.load(filepath, sr=None)
        
        # Basic statistics
        duration = len(audio) / sr
        rms = np.sqrt(np.mean(audio**2))
        max_abs = np.max(np.abs(audio))
        std = np.std(audio)
        
        # Energy-based metrics
        energy = np.sum(audio**2)
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # Silence detection
        silence_threshold = 0.001
        is_mostly_silent = rms < silence_threshold
        
        # Dynamic range
        dynamic_range = 20 * np.log10(max_abs / (rms + 1e-8))
        
        # Spectral analysis for information loss detection
        spectral_stats = analyze_spectrogram(audio, sr)
        
        result = {
            'duration': duration,
            'rms': rms,
            'max_abs': max_abs,
            'std': std,
            'energy': energy,
            'zero_crossing_rate': zero_crossing_rate,
            'is_mostly_silent': is_mostly_silent,
            'dynamic_range': dynamic_range,
            'file_size': os.path.getsize(filepath)
        }
        
        # Add spectral statistics
        result.update(spectral_stats)
        
        return result
    
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return None


def analyze_experiment_directory(exp_dir: str) -> pd.DataFrame:
    """Analyze all training samples in the experiment directory."""
    
    # Find all training sample files
    pattern = os.path.join(exp_dir, "train_samples", "epoch*_train_*_ha_seg*_proc_spk*.wav")
    audio_files = glob.glob(pattern)
    
    print(f"Found {len(audio_files)} training sample files")
    
    if not audio_files:
        print(f"No training samples found in {exp_dir}/train_samples/")
        return pd.DataFrame()
    
    # Analyze each file
    results = []
    for i, filepath in enumerate(audio_files):
        if i % 10 == 0:
            print(f"Analyzing file {i+1}/{len(audio_files)}: {os.path.basename(filepath)}")
        
        # Parse filename
        metadata = parse_filename(filepath)
        if metadata is None:
            print(f"Could not parse filename: {filepath}")
            continue
        
        # Analyze audio
        audio_stats = analyze_audio_file(filepath)
        if audio_stats is None:
            continue
        
        # Combine metadata and statistics
        result = {**metadata, **audio_stats}
        results.append(result)
    
    return pd.DataFrame(results)


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create visualizations of the analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. RMS amplitude trends across epochs
    plt.figure(figsize=(12, 8))
    
    # Plot by speaker
    for speaker in sorted(df['speaker'].unique()):
        speaker_data = df[df['speaker'] == speaker]
        epoch_rms = speaker_data.groupby('epoch')['rms'].agg(['mean', 'std']).reset_index()
        
        plt.errorbar(epoch_rms['epoch'], epoch_rms['mean'], 
                    yerr=epoch_rms['std'], label=f'Speaker {speaker}',
                    marker='o', capsize=5, alpha=0.8)
    
    plt.xlabel('Epoch')
    plt.ylabel('RMS Amplitude')
    plt.title('RMS Amplitude Trends Across Training Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to see small values
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rms_trends_by_epoch.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Silent audio detection across epochs
    plt.figure(figsize=(12, 6))
    silence_by_epoch = df.groupby('epoch')['is_mostly_silent'].agg(['sum', 'count']).reset_index()
    silence_by_epoch['silent_percentage'] = (silence_by_epoch['sum'] / silence_by_epoch['count']) * 100
    
    bars = plt.bar(silence_by_epoch['epoch'], silence_by_epoch['silent_percentage'], 
                   alpha=0.7, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage of Silent Samples (%)')
    plt.title('Silent Audio Detection Across Epochs')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'silent_audio_trends.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribution of amplitudes by epoch (boxplot)
    plt.figure(figsize=(14, 8))
    
    # Select subset of epochs for clarity
    epochs_to_plot = sorted(df['epoch'].unique())[::2]  # Every other epoch
    df_subset = df[df['epoch'].isin(epochs_to_plot)]
    
    sns.boxplot(data=df_subset, x='epoch', y='rms', hue='speaker')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('RMS Amplitude (log scale)')
    plt.title('Distribution of RMS Amplitudes by Epoch and Speaker')
    plt.legend(title='Speaker', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rms_distribution_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap of RMS by scene and epoch
    plt.figure(figsize=(16, 10))
    
    # Create pivot table for heatmap
    pivot_data = df.pivot_table(values='rms', index='scene_id', columns='epoch', aggfunc='mean')
    
    # Plot heatmap
    sns.heatmap(pivot_data, cmap='viridis', cbar_kws={'label': 'RMS Amplitude'}, 
                fmt='.4f', square=False)
    plt.xlabel('Epoch')
    plt.ylabel('Scene ID')
    plt.title('RMS Amplitude Heatmap: Scenes vs Epochs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rms_heatmap_scenes_epochs.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. File size trends (detect corrupted files)
    plt.figure(figsize=(12, 6))
    
    epoch_filesize = df.groupby('epoch')['file_size'].agg(['mean', 'std']).reset_index()
    plt.errorbar(epoch_filesize['epoch'], epoch_filesize['mean'], 
                yerr=epoch_filesize['std'], marker='o', capsize=5)
    
    plt.xlabel('Epoch')
    plt.ylabel('File Size (bytes)')
    plt.title('Audio File Size Trends Across Epochs')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'filesize_trends.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Spectral analysis - Information loss detection
    create_spectral_visualizations(df, output_dir)
    
    print(f"Visualizations saved to: {output_dir}")


def create_spectral_visualizations(df: pd.DataFrame, output_dir: str):
    """Create spectral analysis visualizations to detect information loss."""
    
    # 6a. Spectral centroid trends (frequency content shift)
    plt.figure(figsize=(12, 8))
    
    for speaker in sorted(df['speaker'].unique()):
        speaker_data = df[df['speaker'] == speaker]
        epoch_centroid = speaker_data.groupby('epoch')['spectral_centroid'].agg(['mean', 'std']).reset_index()
        
        plt.errorbar(epoch_centroid['epoch'], epoch_centroid['mean'], 
                    yerr=epoch_centroid['std'], label=f'Speaker {speaker}',
                    marker='o', capsize=5, alpha=0.8)
    
    plt.xlabel('Epoch')
    plt.ylabel('Spectral Centroid (Hz)')
    plt.title('Spectral Centroid Trends Across Training Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectral_centroid_trends.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6b. High-frequency content loss detection
    plt.figure(figsize=(12, 6))
    hf_loss_by_epoch = df.groupby('epoch')['is_missing_highs'].agg(['sum', 'count']).reset_index()
    hf_loss_by_epoch['hf_loss_percentage'] = (hf_loss_by_epoch['sum'] / hf_loss_by_epoch['count']) * 100
    
    bars = plt.bar(hf_loss_by_epoch['epoch'], hf_loss_by_epoch['hf_loss_percentage'], 
                   alpha=0.7, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage of High-Freq Loss (%)')
    plt.title('High-Frequency Information Loss Across Epochs')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'high_freq_loss_trends.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6c. Spectral entropy trends (complexity loss)
    plt.figure(figsize=(12, 8))
    
    for speaker in sorted(df['speaker'].unique()):
        speaker_data = df[df['speaker'] == speaker]
        epoch_entropy = speaker_data.groupby('epoch')['spectral_entropy'].agg(['mean', 'std']).reset_index()
        
        plt.errorbar(epoch_entropy['epoch'], epoch_entropy['mean'], 
                    yerr=epoch_entropy['std'], label=f'Speaker {speaker}',
                    marker='o', capsize=5, alpha=0.8)
    
    plt.xlabel('Epoch')
    plt.ylabel('Spectral Entropy (bits)')
    plt.title('Spectral Complexity (Entropy) Trends Across Training Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectral_entropy_trends.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6d. Frequency band energy distribution
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each frequency band
    bands = ['low_freq_energy', 'mid_freq_energy', 'high_freq_energy']
    band_names = ['Low Freq (0-1kHz)', 'Mid Freq (1-4kHz)', 'High Freq (4kHz+)']
    colors = ['blue', 'green', 'red']
    
    for i, (band, name, color) in enumerate(zip(bands, band_names, colors)):
        plt.subplot(2, 2, i+1)
        
        for speaker in sorted(df['speaker'].unique()):
            speaker_data = df[df['speaker'] == speaker]
            epoch_band = speaker_data.groupby('epoch')[band].mean().reset_index()
            
            plt.plot(epoch_band['epoch'], epoch_band[band], 
                    label=f'Speaker {speaker}', marker='o', alpha=0.8)
        
        plt.xlabel('Epoch')
        plt.ylabel('Energy Fraction')
        plt.title(f'{name} Energy Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 6e. Combined information loss indicators
    plt.subplot(2, 2, 4)
    info_loss_by_epoch = df.groupby('epoch').agg({
        'is_missing_highs': 'mean',
        'is_low_complexity': 'mean',
        'is_overly_smooth': 'mean'
    }).reset_index()
    
    plt.plot(info_loss_by_epoch['epoch'], info_loss_by_epoch['is_missing_highs'] * 100, 
             label='Missing Highs', marker='o', color='red')
    plt.plot(info_loss_by_epoch['epoch'], info_loss_by_epoch['is_low_complexity'] * 100, 
             label='Low Complexity', marker='s', color='orange')
    plt.plot(info_loss_by_epoch['epoch'], info_loss_by_epoch['is_overly_smooth'] * 100, 
             label='Overly Smooth', marker='^', color='purple')
    
    plt.xlabel('Epoch')
    plt.ylabel('Percentage of Samples (%)')
    plt.title('Information Loss Indicators')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'frequency_band_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_report(df: pd.DataFrame, output_dir: str):
    """Generate a summary report of the analysis."""
    
    report_lines = [
        "=" * 60,
        "CHiME9-ECHI Training Samples Analysis Report",
        "=" * 60,
        "",
        f"Total samples analyzed: {len(df)}",
        f"Epochs covered: {df['epoch'].min()} to {df['epoch'].max()}",
        f"Speakers: {sorted(df['speaker'].unique())}",
        f"Unique scenes: {df['scene_id'].nunique()}",
        "",
        "AMPLITUDE ANALYSIS:",
        "-" * 20,
    ]
    
    # Overall statistics
    report_lines.extend([
        f"Overall RMS range: {df['rms'].min():.6f} to {df['rms'].max():.6f}",
        f"Mean RMS: {df['rms'].mean():.6f}",
        f"Median RMS: {df['rms'].median():.6f}",
        ""
    ])
    
    # Silent audio analysis
    silent_count = df['is_mostly_silent'].sum()
    silent_percentage = (silent_count / len(df)) * 100
    report_lines.extend([
        "SILENT AUDIO DETECTION:",
        "-" * 25,
        f"Silent samples (RMS < 0.001): {silent_count} / {len(df)} ({silent_percentage:.1f}%)",
        ""
    ])
    
    # Epoch-by-epoch analysis
    report_lines.extend([
        "EPOCH-BY-EPOCH TRENDS:",
        "-" * 23,
    ])
    
    for epoch in sorted(df['epoch'].unique()):
        epoch_data = df[df['epoch'] == epoch]
        silent_in_epoch = epoch_data['is_mostly_silent'].sum()
        total_in_epoch = len(epoch_data)
        mean_rms = epoch_data['rms'].mean()
        
        report_lines.append(
            f"Epoch {epoch:2d}: {total_in_epoch} samples, "
            f"Mean RMS: {mean_rms:.6f}, "
            f"Silent: {silent_in_epoch}/{total_in_epoch} ({silent_in_epoch/total_in_epoch*100:.1f}%)"
        )
    
    # Speaker-specific analysis
    report_lines.extend([
        "",
        "SPEAKER-SPECIFIC ANALYSIS:",
        "-" * 26,
    ])
    
    for speaker in sorted(df['speaker'].unique()):
        speaker_data = df[df['speaker'] == speaker]
        silent_count = speaker_data['is_mostly_silent'].sum()
        total_count = len(speaker_data)
        mean_rms = speaker_data['rms'].mean()
        
        report_lines.append(
            f"Speaker {speaker}: {total_count} samples, "
            f"Mean RMS: {mean_rms:.6f}, "
            f"Silent: {silent_count}/{total_count} ({silent_count/total_count*100:.1f}%)"
        )
    
    # Spectral analysis
    report_lines.extend([
        "",
        "SPECTRAL ANALYSIS:",
        "-" * 18,
    ])
    
    # Overall spectral statistics
    report_lines.extend([
        f"Mean Spectral Centroid: {df['spectral_centroid'].mean():.0f} Hz",
        f"Mean Spectral Bandwidth: {df['spectral_bandwidth'].mean():.0f} Hz", 
        f"Mean Spectral Entropy: {df['spectral_entropy'].mean():.2f} bits",
        f"Mean High-Freq Ratio: {df['high_freq_ratio'].mean():.4f}",
        ""
    ])
    
    # Information loss indicators
    hf_loss_count = df['is_missing_highs'].sum()
    hf_loss_percentage = (hf_loss_count / len(df)) * 100
    
    low_complexity_count = df['is_low_complexity'].sum()
    low_complexity_percentage = (low_complexity_count / len(df)) * 100
    
    overly_smooth_count = df['is_overly_smooth'].sum()
    overly_smooth_percentage = (overly_smooth_count / len(df)) * 100
    
    report_lines.extend([
        "INFORMATION LOSS DETECTION:",
        "-" * 27,
        f"High-frequency loss: {hf_loss_count} / {len(df)} ({hf_loss_percentage:.1f}%)",
        f"Low spectral complexity: {low_complexity_count} / {len(df)} ({low_complexity_percentage:.1f}%)",
        f"Overly smooth spectrum: {overly_smooth_count} / {len(df)} ({overly_smooth_percentage:.1f}%)",
        ""
    ])
    
    # Frequency band energy distribution
    report_lines.extend([
        "FREQUENCY BAND ENERGY DISTRIBUTION:",
        "-" * 35,
        f"Low Freq (0-1kHz):  {df['low_freq_energy'].mean():.3f} ± {df['low_freq_energy'].std():.3f}",
        f"Mid Freq (1-4kHz):  {df['mid_freq_energy'].mean():.3f} ± {df['mid_freq_energy'].std():.3f}",
        f"High Freq (4kHz+):  {df['high_freq_energy'].mean():.3f} ± {df['high_freq_energy'].std():.3f}",
        ""
    ])
    
    # Problem detection
    report_lines.extend([
        "",
        "POTENTIAL ISSUES DETECTED:",
        "-" * 26,
    ])
    
    if silent_percentage > 20:
        report_lines.append(f"⚠️  HIGH SILENT AUDIO RATE: {silent_percentage:.1f}% of samples are mostly silent")
    
    # Check for decreasing amplitude trends
    epoch_means = df.groupby('epoch')['rms'].mean()
    if len(epoch_means) > 2:
        trend_slope = np.polyfit(epoch_means.index, epoch_means.values, 1)[0]
        if trend_slope < -0.0001:
            report_lines.append("⚠️  DECREASING AMPLITUDE TREND: RMS amplitude is decreasing across epochs")
    
    # Check for very small files
    small_files = df[df['file_size'] < 1000]  # Less than 1KB
    if len(small_files) > 0:
        report_lines.append(f"⚠️  SMALL FILES DETECTED: {len(small_files)} files are suspiciously small (<1KB)")
    
    # Check for spectral information loss
    if hf_loss_percentage > 30:
        report_lines.append(f"⚠️  HIGH-FREQUENCY INFORMATION LOSS: {hf_loss_percentage:.1f}% of samples missing high-frequency content")
    
    if low_complexity_percentage > 40:
        report_lines.append(f"⚠️  LOW SPECTRAL COMPLEXITY: {low_complexity_percentage:.1f}% of samples have low spectral entropy")
    
    # Check for decreasing spectral complexity trends
    if len(df['epoch'].unique()) > 2:
        epoch_entropy_means = df.groupby('epoch')['spectral_entropy'].mean()
        entropy_trend_slope = np.polyfit(epoch_entropy_means.index, epoch_entropy_means.values, 1)[0]
        if entropy_trend_slope < -0.1:
            report_lines.append("⚠️  DECREASING SPECTRAL COMPLEXITY: Spectral entropy is decreasing across epochs")
        
        # Check for decreasing high-frequency content
        epoch_hf_means = df.groupby('epoch')['high_freq_ratio'].mean()
        hf_trend_slope = np.polyfit(epoch_hf_means.index, epoch_hf_means.values, 1)[0]
        if hf_trend_slope < -0.001:
            report_lines.append("⚠️  DECREASING HIGH-FREQUENCY CONTENT: High-frequency ratio is decreasing across epochs")
    
    # Check for unbalanced frequency distribution
    mean_high_freq = df['high_freq_energy'].mean()
    if mean_high_freq < 0.1:
        report_lines.append(f"⚠️  LOW HIGH-FREQUENCY ENERGY: Only {mean_high_freq:.1%} of energy in high frequencies (should be >10%)")
    
    # Save report
    report_text = "\n".join(report_lines)
    
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nDetailed report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze CHiME9-ECHI training samples')
    parser.add_argument('--exp_dir', 
                       default='data/working_dir/experiments/ha-joint32/train_ha',
                       help='Path to experiment directory')
    parser.add_argument('--output_dir',
                       default='training_samples_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Check if experiment directory exists
    if not os.path.exists(args.exp_dir):
        print(f"Error: Experiment directory does not exist: {args.exp_dir}")
        return
    
    # Analyze samples
    print(f"Analyzing training samples in: {args.exp_dir}")
    df = analyze_experiment_directory(args.exp_dir)
    
    if df.empty:
        print("No valid samples found for analysis.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save raw data
    csv_path = os.path.join(args.output_dir, 'training_samples_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved to: {csv_path}")
    
    # Generate visualizations
    print("Creating visualizations...")
    create_visualizations(df, args.output_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()