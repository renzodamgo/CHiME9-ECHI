#!/usr/bin/env python3
"""
Evaluation Report Generator for CHiME9-ECHI Quick Test HA Results

Analyzes results from quick_test_ha evaluation and creates comprehensive visualizations.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Any

def load_results_data(results_file: str) -> pd.DataFrame:
    """Load and parse the results JSON file into a DataFrame."""
    data = []
    
    with open(results_file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    df = pd.DataFrame(data)
    
    # Extract session, device, participant from key
    key_parts = df['key'].str.split('.', expand=True)
    df['session'] = key_parts[0]
    df['device'] = key_parts[1] 
    df['participant'] = key_parts[2]
    df['segment_num'] = key_parts[3]
    df['time_range'] = key_parts[4]
    
    return df

def load_session_metadata(sessions_file: str) -> pd.DataFrame:
    """Load session metadata."""
    return pd.read_csv(sessions_file)

def compute_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics for all metrics."""
    metrics = ['sdr', 'sir', 'sar', 'si_snr', 'ci_sdr', 'pesq', 'stoi']
    
    stats = {}
    for metric in metrics:
        if metric in df.columns:
            # Filter out infinite values for statistics
            finite_values = df[metric][np.isfinite(df[metric])]
            stats[metric] = {
                'mean': finite_values.mean(),
                'std': finite_values.std(),
                'median': finite_values.median(),
                'min': finite_values.min(),
                'max': finite_values.max(),
                'count': len(finite_values),
                'total_count': len(df[metric])
            }
    
    return stats

def create_metric_distribution_plots(df: pd.DataFrame, output_dir: Path):
    """Create distribution plots for all metrics."""
    metrics = ['sdr', 'si_snr', 'ci_sdr', 'pesq', 'stoi']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            finite_values = df[metric][np.isfinite(df[metric])]
            
            axes[i].hist(finite_values, bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{metric.upper()} Distribution')
            axes[i].set_xlabel(f'{metric.upper()} (dB)' if metric != 'stoi' and metric != 'pesq' else metric.upper())
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Add mean line
            mean_val = finite_values.mean()
            axes[i].axvline(mean_val, color='red', linestyle='--', 
                           label=f'Mean: {mean_val:.2f}')
            axes[i].legend()
    
    # Hide unused subplot
    axes[5].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_session_comparison_plot(df: pd.DataFrame, output_dir: Path):
    """Create box plots comparing metrics across sessions."""
    metrics = ['sdr', 'si_snr', 'pesq', 'stoi']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            finite_df = df[np.isfinite(df[metric])]
            
            sns.boxplot(data=finite_df, x='session', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric.upper()} by Session')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'session_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_participant_performance_plot(df: pd.DataFrame, output_dir: Path):
    """Create scatter plots of performance per participant."""
    # Calculate mean performance per participant
    participant_stats = df.groupby('participant').agg({
        'sdr': lambda x: x[np.isfinite(x)].mean(),
        'si_snr': lambda x: x[np.isfinite(x)].mean(),
        'pesq': lambda x: x[np.isfinite(x)].mean(),
        'stoi': lambda x: x[np.isfinite(x)].mean()
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    metrics = ['sdr', 'si_snr', 'pesq', 'stoi']
    
    for i, metric in enumerate(metrics):
        y_vals = participant_stats[metric].dropna()
        x_vals = range(len(y_vals))
        
        axes[i].scatter(x_vals, y_vals, alpha=0.7)
        axes[i].set_title(f'Mean {metric.upper()} per Participant')
        axes[i].set_xlabel('Participant Index')
        axes[i].set_ylabel(f'{metric.upper()}')
        axes[i].grid(True, alpha=0.3)
        
        # Add trend line
        if len(y_vals) > 1:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            axes[i].plot(x_vals, p(x_vals), "r--", alpha=0.8, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'participant_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create correlation heatmap between metrics."""
    metrics = ['sdr', 'si_snr', 'ci_sdr', 'pesq', 'stoi']
    
    # Filter to finite values only
    corr_data = df[metrics].copy()
    for col in corr_data.columns:
        corr_data[col] = corr_data[col].replace([np.inf, -np.inf], np.nan)
    
    correlation_matrix = corr_data.corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Correlation Matrix Between Metrics')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_text_report(df: pd.DataFrame, stats: Dict, output_dir: Path):
    """Generate a comprehensive text report."""
    report_lines = [
        "=" * 80,
        "CHiME9-ECHI Quick Test HA Evaluation Report",
        "=" * 80,
        "",
        f"Total audio segments evaluated: {len(df)}",
        f"Unique participants: {df['participant'].nunique()}",
        f"Sessions covered: {', '.join(sorted(df['session'].unique()))}",
        "",
        "SUMMARY STATISTICS",
        "-" * 40,
        ""
    ]
    
    for metric, metric_stats in stats.items():
        if metric == 'sir':  # Skip SIR as it's all infinity
            continue
            
        report_lines.extend([
            f"{metric.upper()}:",
            f"  Mean: {metric_stats['mean']:.3f}",
            f"  Std:  {metric_stats['std']:.3f}",
            f"  Min:  {metric_stats['min']:.3f}",
            f"  Max:  {metric_stats['max']:.3f}",
            f"  Count: {metric_stats['count']}/{metric_stats['total_count']} finite values",
            ""
        ])
    
    # Session-wise performance
    report_lines.extend([
        "SESSION-WISE PERFORMANCE",
        "-" * 40,
        ""
    ])
    
    for session in sorted(df['session'].unique()):
        session_data = df[df['session'] == session]
        report_lines.extend([
            f"{session}:",
            f"  Segments: {len(session_data)}",
            f"  Mean SDR: {session_data['sdr'][np.isfinite(session_data['sdr'])].mean():.3f} dB",
            f"  Mean SI-SNR: {session_data['si_snr'][np.isfinite(session_data['si_snr'])].mean():.3f} dB",
            f"  Mean PESQ: {session_data['pesq'][np.isfinite(session_data['pesq'])].mean():.3f}",
            f"  Mean STOI: {session_data['stoi'][np.isfinite(session_data['stoi'])].mean():.3f}",
            ""
        ])
    
    # Top/Bottom performers
    finite_sdr = df[np.isfinite(df['sdr'])]
    if len(finite_sdr) > 0:
        best_idx = finite_sdr['sdr'].idxmax()
        worst_idx = finite_sdr['sdr'].idxmin()
        
        report_lines.extend([
            "PERFORMANCE EXTREMES",
            "-" * 40,
            "",
            f"Best SDR: {df.loc[best_idx, 'sdr']:.3f} dB ({df.loc[best_idx, 'key']})",
            f"Worst SDR: {df.loc[worst_idx, 'sdr']:.3f} dB ({df.loc[worst_idx, 'key']})",
            ""
        ])
    
    # Write report
    with open(output_dir / 'evaluation_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))

def main():
    parser = argparse.ArgumentParser(description='Generate evaluation report with plots')
    parser.add_argument('--results_file', default='data/working_dir/experiments/quick_test_ha/evaluation/results/results.dev.ha.summed.batch_1_50.json',
                        help='Path to results JSON file')
    parser.add_argument('--sessions_file', default='data/chime9_echi/metadata/sessions.dev.csv',
                        help='Path to sessions CSV file')
    parser.add_argument('--output_dir', default='evaluation_report_output',
                        help='Directory to save report and plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Loading evaluation results...")
    df = load_results_data(args.results_file)
    
    print("Computing summary statistics...")
    stats = compute_summary_stats(df)
    
    print("Creating distribution plots...")
    create_metric_distribution_plots(df, output_dir)
    
    print("Creating session comparison plots...")
    create_session_comparison_plot(df, output_dir)
    
    print("Creating participant performance plots...")
    create_participant_performance_plot(df, output_dir)
    
    print("Creating correlation heatmap...")
    create_correlation_heatmap(df, output_dir)
    
    print("Generating text report...")
    generate_text_report(df, stats, output_dir)
    
    print(f"Report generated successfully in {output_dir}/")
    print("Files created:")
    for file in output_dir.glob('*'):
        print(f"  - {file.name}")

if __name__ == '__main__':
    main()