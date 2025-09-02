#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import json
from collections import defaultdict
import argparse

def parse_log_file(log_path):
    """Parse training log to extract speaker-specific metrics."""
    metrics_by_epoch = defaultdict(lambda: defaultdict(list))
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    current_epoch = None
    for line in lines:
        # Extract epoch number
        epoch_match = re.search(r'Epoch (\d+)', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue
            
        if current_epoch is None:
            continue
            
        # Extract SI-SDR per speaker
        sisdr_match = re.search(r'sisdr_per_spk.*?\[([-\d\., ]+)\]', line)
        if sisdr_match:
            sisdr_values = [float(x.strip()) for x in sisdr_match.group(1).split(',') if x.strip()]
            for i, val in enumerate(sisdr_values):
                metrics_by_epoch[current_epoch][f'speaker_{i}_sisdr'].append(val)
        
        # Extract RMS per speaker  
        rms_match = re.search(r's_hat_rms_per_spk.*?\[([-\d\., ]+)\]', line)
        if rms_match:
            rms_values = [float(x.strip()) for x in rms_match.group(1).split(',') if x.strip()]
            for i, val in enumerate(rms_values):
                metrics_by_epoch[current_epoch][f'speaker_{i}_rms'].append(val)
                
        # Extract separation quality metrics
        sep_score_match = re.search(r'separation_quality_score[\'\"]: ([\d\.-]+)', line)
        if sep_score_match:
            metrics_by_epoch[current_epoch]['separation_score'].append(float(sep_score_match.group(1)))
            
        # Extract cross-speaker correlation
        corr_match = re.search(r'cross_speaker_corr_mean[\'\"]: ([\d\.-]+)', line)
        if corr_match:
            metrics_by_epoch[current_epoch]['cross_correlation'].append(float(corr_match.group(1)))
    
    return metrics_by_epoch

def analyze_speaker_degradation(metrics_by_epoch):
    """Analyze speaker-specific degradation patterns."""
    analysis = {
        'epochs': sorted(metrics_by_epoch.keys()),
        'speaker_trends': {},
        'degradation_detected': False,
        'problematic_speakers': []
    }
    
    # Calculate trends for each speaker
    for epoch in analysis['epochs']:
        epoch_data = metrics_by_epoch[epoch]
        
        # Find speakers in this epoch
        speakers = set()
        for key in epoch_data.keys():
            if key.startswith('speaker_') and key.endswith('_sisdr'):
                speaker_id = int(key.split('_')[1])
                speakers.add(speaker_id)
        
        for speaker_id in speakers:
            if speaker_id not in analysis['speaker_trends']:
                analysis['speaker_trends'][speaker_id] = {
                    'sisdr_trend': [],
                    'rms_trend': [],
                    'epochs': []
                }
            
            sisdr_key = f'speaker_{speaker_id}_sisdr'
            rms_key = f'speaker_{speaker_id}_rms'
            
            if sisdr_key in epoch_data and epoch_data[sisdr_key]:
                avg_sisdr = np.mean(epoch_data[sisdr_key])
                analysis['speaker_trends'][speaker_id]['sisdr_trend'].append(avg_sisdr)
                analysis['speaker_trends'][speaker_id]['epochs'].append(epoch)
                
            if rms_key in epoch_data and epoch_data[rms_key]:
                avg_rms = np.mean(epoch_data[rms_key])
                analysis['speaker_trends'][speaker_id]['rms_trend'].append(avg_rms)
    
    # Detect degradation patterns
    for speaker_id, trends in analysis['speaker_trends'].items():
        if len(trends['sisdr_trend']) >= 3:
            # Check if SI-SDR is consistently decreasing
            recent_sisdr = trends['sisdr_trend'][-3:]
            if len(recent_sisdr) == 3 and recent_sisdr[0] > recent_sisdr[1] > recent_sisdr[2]:
                analysis['degradation_detected'] = True
                analysis['problematic_speakers'].append(speaker_id)
                print(f"🚨 DEGRADATION DETECTED for Speaker {speaker_id}")
                print(f"   Recent SI-SDR trend: {recent_sisdr}")
    
    return analysis

def create_speaker_visualizations(metrics_by_epoch, analysis, output_dir="debug_plots"):
    """Create visualization plots for speaker analysis."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. SI-SDR trends per speaker
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for speaker_id, trends in analysis['speaker_trends'].items():
        if trends['sisdr_trend']:
            color = colors[speaker_id % len(colors)]
            linestyle = '--' if speaker_id in analysis['problematic_speakers'] else '-'
            linewidth = 3 if speaker_id in analysis['problematic_speakers'] else 2
            
            plt.plot(trends['epochs'], trends['sisdr_trend'], 
                    color=color, linestyle=linestyle, linewidth=linewidth,
                    marker='o', markersize=6, 
                    label=f'Speaker {speaker_id}' + (' (DEGRADING)' if speaker_id in analysis['problematic_speakers'] else ''))
    
    plt.xlabel('Epoch')
    plt.ylabel('SI-SDR (dB)')
    plt.title('Speaker-Specific SI-SDR Trends Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sisdr_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. RMS amplitude trends per speaker  
    plt.figure(figsize=(12, 8))
    for speaker_id, trends in analysis['speaker_trends'].items():
        if trends['rms_trend']:
            color = colors[speaker_id % len(colors)]
            linestyle = '--' if speaker_id in analysis['problematic_speakers'] else '-'
            
            plt.plot(trends['epochs'], trends['rms_trend'],
                    color=color, linestyle=linestyle, linewidth=2,
                    marker='s', markersize=5,
                    label=f'Speaker {speaker_id}' + (' (DEGRADING)' if speaker_id in analysis['problematic_speakers'] else ''))
    
    plt.xlabel('Epoch')
    plt.ylabel('RMS Amplitude')
    plt.title('Speaker-Specific RMS Amplitude Trends')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rms_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Separation quality over time
    if any('separation_score' in metrics_by_epoch[epoch] for epoch in analysis['epochs']):
        plt.figure(figsize=(10, 6))
        epochs_with_sep = []
        sep_scores = []
        
        for epoch in analysis['epochs']:
            if 'separation_score' in metrics_by_epoch[epoch] and metrics_by_epoch[epoch]['separation_score']:
                epochs_with_sep.append(epoch)
                sep_scores.append(np.mean(metrics_by_epoch[epoch]['separation_score']))
        
        plt.plot(epochs_with_sep, sep_scores, 'g-', linewidth=3, marker='D', markersize=8)
        plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Good Separation (>0.7)')
        plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Poor Separation (<0.3)')
        
        plt.xlabel('Epoch')
        plt.ylabel('Separation Quality Score')
        plt.title('Overall Speaker Separation Quality Over Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/separation_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Plots saved to {output_dir}/")
    return output_dir

def generate_debug_report(analysis, metrics_by_epoch, output_file="speaker_debug_report.txt"):
    """Generate a comprehensive debug report."""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CHIME-9 ECHI SPEAKER DEGRADATION DEBUG REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Analysis Period: Epochs {min(analysis['epochs'])} - {max(analysis['epochs'])}\n")
        f.write(f"Total Epochs Analyzed: {len(analysis['epochs'])}\n")
        f.write(f"Degradation Detected: {'YES' if analysis['degradation_detected'] else 'NO'}\n")
        f.write(f"Problematic Speakers: {analysis['problematic_speakers']}\n\n")
        
        f.write("SPEAKER-SPECIFIC ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        
        for speaker_id in sorted(analysis['speaker_trends'].keys()):
            trends = analysis['speaker_trends'][speaker_id]
            f.write(f"\nSpeaker {speaker_id}:\n")
            
            if trends['sisdr_trend']:
                initial_sisdr = trends['sisdr_trend'][0]
                final_sisdr = trends['sisdr_trend'][-1]
                sisdr_change = final_sisdr - initial_sisdr
                f.write(f"  SI-SDR: {initial_sisdr:.2f} → {final_sisdr:.2f} (Δ{sisdr_change:+.2f} dB)\n")
                
            if trends['rms_trend']:
                initial_rms = trends['rms_trend'][0]
                final_rms = trends['rms_trend'][-1]
                rms_ratio = final_rms / initial_rms if initial_rms > 0 else float('inf')
                f.write(f"  RMS: {initial_rms:.6f} → {final_rms:.6f} (×{rms_ratio:.2f})\n")
                
            f.write(f"  Status: {'🚨 DEGRADING' if speaker_id in analysis['problematic_speakers'] else '✅ Stable'}\n")
        
        # Recent metrics summary
        f.write(f"\nRECENT METRICS (Last 3 Epochs):\n")
        f.write("-" * 40 + "\n")
        recent_epochs = sorted(analysis['epochs'])[-3:]
        for epoch in recent_epochs:
            f.write(f"\nEpoch {epoch}:\n")
            epoch_data = metrics_by_epoch[epoch]
            
            for key, values in epoch_data.items():
                if values and 'speaker' in key:
                    avg_val = np.mean(values)
                    f.write(f"  {key}: {avg_val:.4f}\n")
    
    print(f"📋 Debug report saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Debug CHiME-9 ECHI speaker degradation issues')
    parser.add_argument('--log_path', type=str, required=True, 
                       help='Path to training log file')
    parser.add_argument('--output_dir', type=str, default='debug_analysis',
                       help='Output directory for plots and reports')
    parser.add_argument('--tail_lines', type=int, default=1000,
                       help='Number of lines to analyze from end of log')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Parse log file
    print(f"🔍 Analyzing training log: {args.log_path}")
    
    # If log is large, only analyze recent lines
    log_path = Path(args.log_path)
    if log_path.stat().st_size > 50 * 1024 * 1024:  # 50MB
        print(f"📄 Large log detected, analyzing last {args.tail_lines} lines...")
        with open(log_path, 'r') as f:
            lines = f.readlines()[-args.tail_lines:]
        
        temp_log = args.output_dir + '/recent_log.txt'
        with open(temp_log, 'w') as f:
            f.writelines(lines)
        log_path = temp_log
    
    metrics_by_epoch = parse_log_file(log_path)
    
    if not metrics_by_epoch:
        print("❌ No metrics found in log file. Check log format.")
        return
    
    print(f"📊 Found metrics for {len(metrics_by_epoch)} epochs")
    
    # Analyze speaker degradation
    analysis = analyze_speaker_degradation(metrics_by_epoch)
    
    # Create visualizations
    plot_dir = f"{args.output_dir}/plots"
    create_speaker_visualizations(metrics_by_epoch, analysis, plot_dir)
    
    # Generate report
    report_file = f"{args.output_dir}/speaker_debug_report.txt"
    generate_debug_report(analysis, metrics_by_epoch, report_file)
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Degradation Detected: {'YES' if analysis['degradation_detected'] else 'NO'}")
    print(f"Problematic Speakers: {analysis['problematic_speakers']}")
    print(f"Total Speakers Tracked: {len(analysis['speaker_trends'])}")
    print(f"Epochs Analyzed: {len(analysis['epochs'])}")
    print(f"\n📁 All outputs saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()