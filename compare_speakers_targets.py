#!/usr/bin/env python3
"""
Compare model-generated speakers with targets to calculate SI-SDR and spectral loss per epoch.
"""

import os
import argparse
import numpy as np
import torch
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def sisdr_loss(estimate, target, eps=1e-8):
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    Args:
        estimate: Model output [B, T] or [T]
        target: Ground truth [B, T] or [T]
        eps: Small epsilon for numerical stability
    
    Returns:
        SI-SDR in dB
    """
    if estimate.dim() == 1:
        estimate = estimate.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    
    # Zero-mean signals
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)
    
    # Compute scaling factor
    num = torch.sum(estimate * target, dim=-1, keepdim=True)
    den = torch.sum(target * target, dim=-1, keepdim=True) + eps
    s = num / den
    
    # Compute SI-SDR
    s_target = s * target
    e_noise = estimate - s_target
    
    sisdr = 20 * torch.log10(
        torch.norm(s_target, dim=-1) / (torch.norm(e_noise, dim=-1) + eps) + eps
    )
    
    return sisdr.mean().item()

def spectral_loss(estimate, target, n_fft=1024, hop_length=256):
    """
    Calculate spectral loss between estimate and target.
    
    Args:
        estimate: Model output [T]
        target: Ground truth [T]
        n_fft: FFT size
        hop_length: Hop length for STFT
    
    Returns:
        Spectral loss (L1 loss in magnitude domain)
    """
    # Compute STFT
    estimate_stft = torch.stft(
        estimate, n_fft=n_fft, hop_length=hop_length, 
        return_complex=True, normalized=True
    )
    target_stft = torch.stft(
        target, n_fft=n_fft, hop_length=hop_length,
        return_complex=True, normalized=True
    )
    
    # Magnitude spectrograms
    estimate_mag = torch.abs(estimate_stft)
    target_mag = torch.abs(target_stft)
    
    # L1 loss
    spec_loss = torch.nn.functional.l1_loss(estimate_mag, target_mag)
    
    return spec_loss.item()

def load_audio_safe(file_path, target_sr=16000):
    """
    Safely load audio file.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
    
    Returns:
        Tuple of (waveform, sample_rate) or (None, None) if failed
    """
    try:
        waveform, sr = torchaudio.load(file_path)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        return waveform.squeeze(0), target_sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def parse_filename(filename):
    """
    Parse training sample filename to extract epoch, scene, segment, and speaker info.
    
    Expected format: epoch{XX}_{scene}_{segment}_{type}_spk{X}.wav
    """
    try:
        parts = filename.replace('.wav', '').split('_')
        epoch = int(parts[0].replace('epoch', ''))
        scene = parts[1]
        segment = '_'.join(parts[2:-2])  # Handle multi-part segment names
        audio_type = parts[-2]  # 'proc' or 'target'
        speaker = int(parts[-1].replace('spk', ''))
        
        return {
            'epoch': epoch,
            'scene': scene, 
            'segment': segment,
            'type': audio_type,
            'speaker': speaker,
            'filename': filename
        }
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return None

def compare_speakers_targets(exp_dir, output_dir=None):
    """
    Compare model-generated speakers with their targets.
    
    Args:
        exp_dir: Path to experiment directory containing train_samples/
        output_dir: Output directory for results
    """
    # Setup paths
    samples_dir = Path(exp_dir) / 'train_samples'
    if not samples_dir.exists():
        print(f"Training samples directory not found: {samples_dir}")
        return
    
    if output_dir is None:
        output_dir = Path(exp_dir) / 'speaker_target_comparison'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Analyzing speaker-target comparison in: {exp_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find all audio files
    audio_files = list(samples_dir.glob('*.wav'))
    print(f"Found {len(audio_files)} audio files")
    
    # Parse filenames and organize data
    parsed_files = []
    for file_path in audio_files:
        parsed = parse_filename(file_path.name)
        if parsed:
            parsed['file_path'] = file_path
            parsed_files.append(parsed)
    
    # Separate target and processed files
    target_files = {(item['scene'], item['segment'], item['speaker']): item['file_path'] 
                   for item in parsed_files if item['type'] == 'target'}
    
    proc_files = [(item['epoch'], item['scene'], item['segment'], item['speaker'], item['file_path']) 
                  for item in parsed_files if item['type'] == 'proc']
    
    print(f"Found {len(target_files)} unique target references")
    print(f"Found {len(proc_files)} processed samples")
    
    # Calculate metrics for each comparison
    results = []
    
    print("Comparing processed outputs with targets...")
    for epoch, scene, segment, speaker, proc_path in tqdm(proc_files):
        
        # Look for corresponding target file (targets are from epoch 0)
        target_key = (scene, segment, speaker)
        if target_key not in target_files:
            continue
            
        target_path = target_files[target_key]
        
        # Load audio files
        proc_audio, _ = load_audio_safe(proc_path)
        target_audio, _ = load_audio_safe(target_path)
        
        if proc_audio is None or target_audio is None:
            continue
        
        # Ensure same length (truncate to shorter)
        min_len = min(len(proc_audio), len(target_audio))
        proc_audio = proc_audio[:min_len]
        target_audio = target_audio[:min_len]
        
        # Skip if too short
        if min_len < 1000:  # Less than ~60ms at 16kHz
            continue
        
        # Calculate metrics
        try:
            sisdr = sisdr_loss(proc_audio, target_audio)
            l_spec = spectral_loss(proc_audio, target_audio)
            
            # Audio statistics
            proc_rms = torch.sqrt(torch.mean(proc_audio ** 2)).item()
            target_rms = torch.sqrt(torch.mean(target_audio ** 2)).item()
            
            results.append({
                'epoch': epoch,
                'scene': scene,
                'segment': segment,
                'speaker': speaker,
                'sisdr_db': sisdr,
                'l_spec': l_spec,
                'proc_rms': proc_rms,
                'target_rms': target_rms,
                'length_sec': min_len / 16000.0,
                'proc_file': proc_path.name,
                'target_file': target_path.name
            })
        except Exception as e:
            print(f"Error calculating metrics for epoch {epoch}, speaker {speaker}: {e}")
            continue
    
    if not results:
        print("No valid comparisons found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save raw data
    csv_path = output_dir / 'speaker_target_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved to: {csv_path}")
    
    # Generate analysis and visualizations
    generate_analysis(df, output_dir)
    
    print(f"Analysis complete! Results saved to: {output_dir}")

def generate_analysis(df, output_dir):
    """Generate analysis plots and summary statistics."""
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. SI-SDR and Spectral Loss over epochs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # SI-SDR by epoch
    epoch_sisdr = df.groupby('epoch')['sisdr_db'].agg(['mean', 'std', 'count']).reset_index()
    ax1.plot(epoch_sisdr['epoch'], epoch_sisdr['mean'], 'o-', linewidth=2, markersize=6)
    ax1.fill_between(epoch_sisdr['epoch'], 
                     epoch_sisdr['mean'] - epoch_sisdr['std'],
                     epoch_sisdr['mean'] + epoch_sisdr['std'], 
                     alpha=0.3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('SI-SDR (dB)')
    ax1.set_title('SI-SDR Performance Over Training')
    ax1.grid(True, alpha=0.3)
    
    # Spectral loss by epoch
    epoch_lspec = df.groupby('epoch')['l_spec'].agg(['mean', 'std']).reset_index()
    ax2.plot(epoch_lspec['epoch'], epoch_lspec['mean'], 'o-', linewidth=2, markersize=6, color='orange')
    ax2.fill_between(epoch_lspec['epoch'],
                     epoch_lspec['mean'] - epoch_lspec['std'],
                     epoch_lspec['mean'] + epoch_lspec['std'],
                     alpha=0.3, color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Spectral Loss')
    ax2.set_title('Spectral Loss Over Training')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_over_epochs.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Speaker-specific analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # SI-SDR by speaker
    df.boxplot(column='sisdr_db', by='speaker', ax=ax1)
    ax1.set_title('SI-SDR Distribution by Speaker')
    ax1.set_xlabel('Speaker')
    ax1.set_ylabel('SI-SDR (dB)')
    
    # Spectral loss by speaker
    df.boxplot(column='l_spec', by='speaker', ax=ax2)
    ax2.set_title('Spectral Loss Distribution by Speaker')
    ax2.set_xlabel('Speaker')
    ax2.set_ylabel('Spectral Loss')
    
    # RMS comparison by speaker
    speakers = df['speaker'].unique()
    x_pos = np.arange(len(speakers))
    proc_rms_means = [df[df['speaker'] == spk]['proc_rms'].mean() for spk in speakers]
    target_rms_means = [df[df['speaker'] == spk]['target_rms'].mean() for spk in speakers]
    
    width = 0.35
    ax3.bar(x_pos - width/2, proc_rms_means, width, label='Processed', alpha=0.8)
    ax3.bar(x_pos + width/2, target_rms_means, width, label='Target', alpha=0.8)
    ax3.set_xlabel('Speaker')
    ax3.set_ylabel('RMS')
    ax3.set_title('RMS Comparison: Processed vs Target')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'Speaker {s}' for s in speakers])
    ax3.legend()
    
    # Correlation plot
    ax4.scatter(df['target_rms'], df['proc_rms'], alpha=0.6, c=df['speaker'], cmap='tab10')
    ax4.plot([0, df['target_rms'].max()], [0, df['target_rms'].max()], 'r--', alpha=0.8)
    ax4.set_xlabel('Target RMS')
    ax4.set_ylabel('Processed RMS')
    ax4.set_title('RMS Correlation: Processed vs Target')
    
    plt.suptitle('')  # Remove automatic title
    plt.tight_layout()
    plt.savefig(output_dir / 'speaker_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Training progression heatmap
    pivot_sisdr = df.pivot_table(values='sisdr_db', index='speaker', columns='epoch', aggfunc='mean')
    pivot_lspec = df.pivot_table(values='l_spec', index='speaker', columns='epoch', aggfunc='mean')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # SI-SDR heatmap
    sns.heatmap(pivot_sisdr, annot=False, cmap='RdYlGn', ax=ax1, cbar_kws={'label': 'SI-SDR (dB)'})
    ax1.set_title('SI-SDR by Speaker and Epoch')
    ax1.set_ylabel('Speaker')
    
    # Spectral Loss heatmap
    sns.heatmap(pivot_lspec, annot=False, cmap='RdYlGn_r', ax=ax2, cbar_kws={'label': 'Spectral Loss'})
    ax2.set_title('Spectral Loss by Speaker and Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Speaker')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_progression_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate summary report
    generate_summary_report(df, output_dir)

def generate_summary_report(df, output_dir):
    """Generate text summary report."""
    
    report = []
    report.append("=" * 60)
    report.append("SPEAKER-TARGET COMPARISON ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Overall statistics
    report.append(f"Total comparisons: {len(df)}")
    report.append(f"Epochs covered: {df['epoch'].min()} to {df['epoch'].max()}")
    report.append(f"Speakers: {sorted(df['speaker'].unique())}")
    report.append(f"Scenes: {sorted(df['scene'].unique())}")
    report.append("")
    
    # SI-SDR Analysis
    report.append("SI-SDR ANALYSIS:")
    report.append("-" * 20)
    report.append(f"Overall Mean SI-SDR: {df['sisdr_db'].mean():.2f} ± {df['sisdr_db'].std():.2f} dB")
    report.append(f"SI-SDR Range: {df['sisdr_db'].min():.2f} to {df['sisdr_db'].max():.2f} dB")
    
    # SI-SDR by speaker
    for speaker in sorted(df['speaker'].unique()):
        spk_data = df[df['speaker'] == speaker]
        report.append(f"Speaker {speaker}: {spk_data['sisdr_db'].mean():.2f} ± {spk_data['sisdr_db'].std():.2f} dB ({len(spk_data)} samples)")
    report.append("")
    
    # Spectral Loss Analysis
    report.append("SPECTRAL LOSS ANALYSIS:")
    report.append("-" * 25)
    report.append(f"Overall Mean L_spec: {df['l_spec'].mean():.4f} ± {df['l_spec'].std():.4f}")
    report.append(f"L_spec Range: {df['l_spec'].min():.4f} to {df['l_spec'].max():.4f}")
    
    # Spectral loss by speaker
    for speaker in sorted(df['speaker'].unique()):
        spk_data = df[df['speaker'] == speaker]
        report.append(f"Speaker {speaker}: {spk_data['l_spec'].mean():.4f} ± {spk_data['l_spec'].std():.4f}")
    report.append("")
    
    # Training progression
    report.append("TRAINING PROGRESSION:")
    report.append("-" * 22)
    
    # First vs Last epoch comparison
    first_epoch = df['epoch'].min()
    last_epoch = df['epoch'].max()
    
    first_sisdr = df[df['epoch'] == first_epoch]['sisdr_db'].mean()
    last_sisdr = df[df['epoch'] == last_epoch]['sisdr_db'].mean()
    sisdr_improvement = last_sisdr - first_sisdr
    
    first_lspec = df[df['epoch'] == first_epoch]['l_spec'].mean()
    last_lspec = df[df['epoch'] == last_epoch]['l_spec'].mean()
    lspec_improvement = first_lspec - last_lspec
    
    report.append(f"SI-SDR Improvement: {sisdr_improvement:+.2f} dB (from {first_sisdr:.2f} to {last_sisdr:.2f})")
    report.append(f"Spectral Loss Improvement: {lspec_improvement:+.4f} (from {first_lspec:.4f} to {last_lspec:.4f})")
    report.append("")
    
    # Best and worst performing samples
    report.append("PERFORMANCE EXTREMES:")
    report.append("-" * 21)
    
    best_sisdr = df.loc[df['sisdr_db'].idxmax()]
    worst_sisdr = df.loc[df['sisdr_db'].idxmin()]
    
    report.append(f"Best SI-SDR: {best_sisdr['sisdr_db']:.2f} dB")
    report.append(f"  Epoch {best_sisdr['epoch']}, Speaker {best_sisdr['speaker']}, {best_sisdr['scene']}_{best_sisdr['segment']}")
    
    report.append(f"Worst SI-SDR: {worst_sisdr['sisdr_db']:.2f} dB")
    report.append(f"  Epoch {worst_sisdr['epoch']}, Speaker {worst_sisdr['speaker']}, {worst_sisdr['scene']}_{best_sisdr['segment']}")
    report.append("")
    
    # Issues and recommendations
    report.append("POTENTIAL ISSUES:")
    report.append("-" * 17)
    
    if df['sisdr_db'].mean() < 0:
        report.append("⚠️  LOW SI-SDR: Average SI-SDR is negative, indicating poor separation")
    
    if df['l_spec'].mean() > 0.1:
        report.append("⚠️  HIGH SPECTRAL LOSS: Large spectral differences between processed and target")
    
    # Check for decreasing performance
    epoch_means = df.groupby('epoch')['sisdr_db'].mean()
    if len(epoch_means) > 1 and epoch_means.iloc[-1] < epoch_means.iloc[0]:
        report.append("⚠️  DECREASING SI-SDR: Performance is degrading over training")
    
    # Speaker imbalance
    speaker_counts = df['speaker'].value_counts()
    if speaker_counts.max() / speaker_counts.min() > 2:
        report.append("⚠️  SPEAKER IMBALANCE: Uneven number of samples per speaker")
    
    report.append("")
    report.append("Analysis generated by speaker-target comparison script")
    report.append("")
    
    # Save report
    with open(output_dir / 'comparison_report.txt', 'w') as f:
        f.write('\n'.join(report))

def main():
    parser = argparse.ArgumentParser(description='Compare model speakers with targets for SI-SDR and spectral loss')
    parser.add_argument('--exp_dir', type=str, required=True,
                       help='Path to experiment directory containing train_samples/')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    compare_speakers_targets(args.exp_dir, args.output_dir)

if __name__ == '__main__':
    main()