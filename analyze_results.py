#!/usr/bin/env python3
"""
Audio Source Separation Results Analysis Script

Analyzes the performance of the Universal GridNet model by comparing
processed outputs to target reference files.

Usage:
    python analyze_results.py --data_dir data/working_dir/experiments/ha-joint-uni/train_ha/train_samples/
"""

import argparse
import os
import glob
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import soundfile as sf
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Audio quality metrics
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

try:
    from fast_bss_eval import bss_eval_sources_no_permutation
    BSS_AVAILABLE = True
except ImportError:
    BSS_AVAILABLE = False
    print("Warning: fast_bss_eval not available. Install with: pip install fast_bss_eval")


class AudioAnalyzer:
    """Analyzes audio separation results by comparing processed vs target files."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.results = []
        
    def parse_filename(self, filename: str) -> Optional[Dict[str, str]]:
        """
        Parse filename to extract metadata.
        Format: epoch{N}_{train|dev}_{ID}_ha_seg{N}_{type}_spk{N}.wav or epoch{N}_{train|dev}_{ID}_ha_seg{N}_{type}.wav
        """
        # Pattern for files with speaker ID (proc_spk0, target_spk0)
        pattern_with_spk = r'epoch(\d+)_(train|dev)_(\d+)_ha_seg(\d+)_(proc|target)_spk(\d+)\.wav'
        match = re.match(pattern_with_spk, filename)
        
        if match:
            epoch, split, data_id, segment, file_type, speaker_id = match.groups()
            return {
                'epoch': int(epoch),
                'split': split,  # 'train' or 'dev'
                'data_id': int(data_id),  
                'segment': int(segment),
                'type': file_type,  # 'proc', 'target'
                'speaker_id': int(speaker_id),
                'base_name': f"epoch{epoch}_{split}_{data_id}_ha_seg{segment}"
            }
            
        # Pattern for files without speaker ID (noisy)
        pattern_no_spk = r'epoch(\d+)_(train|dev)_(\d+)_ha_seg(\d+)_(noisy)\.wav'
        match = re.match(pattern_no_spk, filename)
        
        if match:
            epoch, split, data_id, segment, file_type = match.groups()
            return {
                'epoch': int(epoch),
                'split': split,  # 'train' or 'dev'
                'data_id': int(data_id),  
                'segment': int(segment),
                'type': file_type,  # 'noisy'
                'speaker_id': None,
                'base_name': f"epoch{epoch}_{split}_{data_id}_ha_seg{segment}"
            }
            
        return None
        
    def find_file_groups(self) -> Dict[str, Dict[str, List[str]]]:
        """Group files by base name and organize by type."""
        file_groups = defaultdict(lambda: defaultdict(list))
        
        for wav_file in self.data_dir.glob('*.wav'):
            parsed = self.parse_filename(wav_file.name)
            if not parsed:
                continue
                
            base_name = parsed['base_name']
            file_type = parsed['type']
            
            if file_type == 'noisy':
                file_groups[base_name]['noisy'] = str(wav_file)
            elif file_type in ['proc', 'target']:
                file_groups[base_name][file_type].append({
                    'path': str(wav_file),
                    'speaker_id': parsed['speaker_id'],
                    'epoch': parsed['epoch'],
                    'split': parsed['split'],
                    'data_id': parsed['data_id'],
                    'segment': parsed['segment']
                })
                
        return dict(file_groups)
        
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return data and sample rate."""
        try:
            data, sr = sf.read(filepath)
            return data, sr
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None
            
    def calculate_snr(self, target: np.ndarray, processed: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio in dB."""
        try:
            # Align lengths
            min_len = min(len(target), len(processed))
            target = target[:min_len]
            processed = processed[:min_len]
            
            signal_power = np.mean(target ** 2)
            noise_power = np.mean((target - processed) ** 2)
            
            if noise_power == 0:
                return float('inf')
            
            snr = 10 * np.log10(signal_power / noise_power)
            return snr
        except Exception:
            return np.nan
            
    def calculate_si_sdr(self, target: np.ndarray, processed: np.ndarray) -> float:
        """Calculate Scale-Invariant Signal-to-Distortion Ratio."""
        try:
            # Align lengths
            min_len = min(len(target), len(processed))
            target = target[:min_len]  
            processed = processed[:min_len]
            
            # Zero-mean
            target = target - np.mean(target)
            processed = processed - np.mean(processed)
            
            # Scale-invariant target
            alpha = np.dot(processed, target) / np.dot(target, target)
            scaled_target = alpha * target
            
            # SI-SDR calculation
            signal_power = np.sum(scaled_target ** 2)
            noise_power = np.sum((processed - scaled_target) ** 2)
            
            if noise_power == 0:
                return float('inf')
                
            si_sdr = 10 * np.log10(signal_power / noise_power)
            return si_sdr
        except Exception:
            return np.nan
            
    def calculate_pesq(self, target: np.ndarray, processed: np.ndarray, sr: int) -> float:
        """Calculate PESQ score."""
        if not PESQ_AVAILABLE:
            return np.nan
            
        try:
            # PESQ requires specific sample rates
            if sr not in [8000, 16000]:
                return np.nan
                
            # Align lengths
            min_len = min(len(target), len(processed))
            target = target[:min_len]
            processed = processed[:min_len]
            
            # PESQ expects values in [-1, 1]
            target = np.clip(target, -1, 1)
            processed = np.clip(processed, -1, 1)
            
            mode = 'wb' if sr == 16000 else 'nb'
            score = pesq(sr, target, processed, mode)
            return score
        except Exception:
            return np.nan
            
    def calculate_stoi(self, target: np.ndarray, processed: np.ndarray, sr: int) -> float:
        """Calculate STOI score."""
        if not STOI_AVAILABLE:
            return np.nan
            
        try:
            # Align lengths
            min_len = min(len(target), len(processed))
            target = target[:min_len]
            processed = processed[:min_len]
            
            score = stoi(target, processed, sr, extended=False)
            return score
        except Exception:
            return np.nan
            
    def calculate_bss_metrics(self, targets: List[np.ndarray], processed: List[np.ndarray]) -> Dict[str, float]:
        """Calculate BSS evaluation metrics (SDR, SIR, SAR)."""
        if not BSS_AVAILABLE:
            return {'sdr': np.nan, 'sir': np.nan, 'sar': np.nan}
            
        try:
            # Convert to numpy arrays and align
            targets = np.array(targets)
            processed = np.array(processed)
            
            # Ensure same length
            min_len = min(targets.shape[1], processed.shape[1])
            targets = targets[:, :min_len]
            processed = processed[:, :min_len]
            
            # Calculate BSS metrics
            sdr, sir, sar = bss_eval_sources_no_permutation(processed, targets)
            
            return {
                'sdr': np.mean(sdr),
                'sir': np.mean(sir), 
                'sar': np.mean(sar)
            }
        except Exception:
            return {'sdr': np.nan, 'sir': np.nan, 'sar': np.nan}
            
    def analyze_sample_group(self, base_name: str, files: Dict[str, any]) -> List[Dict]:
        """Analyze a single sample group (one base name with all its speakers)."""
        results = []
        
        # Load noisy reference if available
        noisy_path = files.get('noisy')
        if noisy_path:
            noisy_audio, noisy_sr = self.load_audio(noisy_path)
        else:
            noisy_audio, noisy_sr = None, None
            
        # Get processed and target files
        proc_files = files.get('proc', [])
        target_files = files.get('target', [])
        
        # Sort by speaker ID
        proc_files.sort(key=lambda x: x['speaker_id'])
        target_files.sort(key=lambda x: x['speaker_id'])
        
        # Ensure we have matching proc and target files
        if len(proc_files) != len(target_files):
            print(f"Warning: Mismatch in number of processed ({len(proc_files)}) and target ({len(target_files)}) files for {base_name}")
            return results
            
        # Load all target and processed audio for BSS evaluation
        all_targets = []
        all_processed = []
        
        # Analyze each speaker
        for proc_info, target_info in zip(proc_files, target_files):
            if proc_info['speaker_id'] != target_info['speaker_id']:
                print(f"Warning: Speaker ID mismatch for {base_name}")
                continue
                
            # Load audio files
            proc_audio, proc_sr = self.load_audio(proc_info['path'])
            target_audio, target_sr = self.load_audio(target_info['path'])
            
            if proc_audio is None or target_audio is None:
                continue
                
            if proc_sr != target_sr:
                print(f"Warning: Sample rate mismatch for {base_name} spk{proc_info['speaker_id']}")
                continue
                
            # Store for BSS evaluation
            all_targets.append(target_audio)
            all_processed.append(proc_audio)
            
            # Calculate individual metrics
            snr = self.calculate_snr(target_audio, proc_audio)
            si_sdr = self.calculate_si_sdr(target_audio, proc_audio)
            pesq_score = self.calculate_pesq(target_audio, proc_audio, proc_sr)
            stoi_score = self.calculate_stoi(target_audio, proc_audio, proc_sr)
            
            # Compile results for this speaker
            result = {
                'base_name': base_name,
                'epoch': proc_info['epoch'],
                'split': proc_info['split'],
                'data_id': proc_info['data_id'],
                'segment': proc_info['segment'],
                'speaker_id': proc_info['speaker_id'],
                'sample_rate': proc_sr,
                'target_length': len(target_audio),
                'processed_length': len(proc_audio),
                'snr_db': snr,
                'si_sdr_db': si_sdr,
                'pesq': pesq_score,
                'stoi': stoi_score,
            }
            
            # Add noisy reference metrics if available
            if noisy_audio is not None and noisy_sr == proc_sr:
                noisy_snr = self.calculate_snr(target_audio, noisy_audio)
                result['noisy_snr_db'] = noisy_snr
                result['snr_improvement_db'] = snr - noisy_snr
                
            results.append(result)
            
        # Calculate BSS metrics for the whole group
        if len(all_targets) >= 2 and len(all_processed) >= 2:
            bss_metrics = self.calculate_bss_metrics(all_targets, all_processed)
            
            # Add BSS metrics to all results for this group
            for result in results:
                result.update(bss_metrics)
                
        return results
        
    def analyze_all(self) -> pd.DataFrame:
        """Analyze all file groups and return results DataFrame."""
        print(f"Analyzing files in {self.data_dir}")
        
        file_groups = self.find_file_groups()
        print(f"Found {len(file_groups)} sample groups")
        
        all_results = []
        
        for base_name, files in file_groups.items():
            print(f"Processing {base_name}...")
            group_results = self.analyze_sample_group(base_name, files)
            all_results.extend(group_results)
            
        if not all_results:
            print("No results found!")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_results)
        
        print(f"\nAnalysis complete! Processed {len(all_results)} individual speaker separations "
              f"across {len(file_groups)} sample groups.")
        
        return df
        
    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics."""
        if df.empty:
            print("No data to summarize")
            return
            
        print("\n" + "="*80)
        print("UNIVERSAL GRIDNET SEPARATION ANALYSIS SUMMARY")
        print("="*80)
        
        # Overall statistics
        print(f"\nDataset Overview:")
        print(f"  • Total speaker separations analyzed: {len(df)}")
        print(f"  • Unique epochs: {df['epoch'].nunique()}")
        print(f"  • Unique training segments: {df['base_name'].nunique()}")
        print(f"  • Sample rate: {df['sample_rate'].iloc[0]} Hz")
        
        # Metrics summary
        metrics = ['snr_db', 'si_sdr_db', 'pesq', 'stoi']
        available_metrics = [m for m in metrics if m in df.columns and not df[m].isna().all()]
        
        if available_metrics:
            print(f"\nAudio Quality Metrics:")
            print("-" * 60)
            
            for metric in available_metrics:
                values = df[metric].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    median_val = values.median()
                    
                    if metric == 'snr_db':
                        print(f"  SNR:          {mean_val:6.2f} ± {std_val:4.2f} dB  (median: {median_val:6.2f} dB)")
                    elif metric == 'si_sdr_db':
                        print(f"  SI-SDR:       {mean_val:6.2f} ± {std_val:4.2f} dB  (median: {median_val:6.2f} dB)")
                    elif metric == 'pesq':
                        print(f"  PESQ:         {mean_val:6.3f} ± {std_val:4.3f}     (median: {median_val:6.3f})")
                    elif metric == 'stoi':
                        print(f"  STOI:         {mean_val:6.3f} ± {std_val:4.3f}     (median: {median_val:6.3f})")
                        
        # BSS metrics
        bss_metrics = ['sdr', 'sir', 'sar']
        available_bss = [m for m in bss_metrics if m in df.columns and not df[m].isna().all()]
        
        if available_bss:
            print(f"\nBSS Evaluation Metrics:")
            print("-" * 60)
            
            for metric in available_bss:
                values = df[metric].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    
                    if metric == 'sdr':
                        print(f"  SDR:          {mean_val:6.2f} ± {std_val:4.2f} dB")
                    elif metric == 'sir':
                        print(f"  SIR:          {mean_val:6.2f} ± {std_val:4.2f} dB")
                    elif metric == 'sar':
                        print(f"  SAR:          {mean_val:6.2f} ± {std_val:4.2f} dB")
                        
        # Improvement over noisy input
        if 'snr_improvement_db' in df.columns and not df['snr_improvement_db'].isna().all():
            improvement = df['snr_improvement_db'].dropna()
            if len(improvement) > 0:
                mean_imp = improvement.mean()
                print(f"\nImprovement over noisy input:")
                print("-" * 60)
                print(f"  SNR improvement: {mean_imp:6.2f} dB")
                
        # Per-epoch analysis
        if df['epoch'].nunique() > 1:
            print(f"\nPer-Epoch Analysis (SI-SDR):")
            print("-" * 60)
            epoch_stats = df.groupby('epoch')['si_sdr_db'].agg(['mean', 'std', 'count']).round(2)
            for epoch, stats in epoch_stats.iterrows():
                print(f"  Epoch {epoch:2d}: {stats['mean']:6.2f} ± {stats['std']:4.2f} dB  (n={stats['count']})")
        
        # Per-speaker analysis
        if 'speaker_id' in df.columns:
            print(f"\nPer-Speaker Analysis (SI-SDR):")
            print("-" * 60)
            speaker_stats = df.groupby('speaker_id')['si_sdr_db'].agg(['mean', 'std', 'count']).round(2)
            for speaker, stats in speaker_stats.iterrows():
                print(f"  Speaker {speaker}: {stats['mean']:6.2f} ± {stats['std']:4.2f} dB  (n={stats['count']})")
            
            # Speaker performance across epochs (if multiple epochs)
            if df['epoch'].nunique() > 3:  # Only if we have enough epochs
                print(f"\nSpeaker Performance Trends:")
                print("-" * 60)
                speaker_epoch_stats = df.groupby(['speaker_id', 'epoch'])['si_sdr_db'].mean().unstack(level=0)
                
                # Calculate trends (improvement from first to last epoch)
                first_epochs = df[df['epoch'] <= df['epoch'].quantile(0.25)]
                last_epochs = df[df['epoch'] >= df['epoch'].quantile(0.75)]
                
                first_perf = first_epochs.groupby('speaker_id')['si_sdr_db'].mean()
                last_perf = last_epochs.groupby('speaker_id')['si_sdr_db'].mean()
                improvement = last_perf - first_perf
                
                for speaker in sorted(df['speaker_id'].unique()):
                    if speaker in improvement.index:
                        trend = "↗" if improvement[speaker] > 1 else "↘" if improvement[speaker] < -1 else "→"
                        print(f"  Speaker {speaker}: {improvement[speaker]:+5.2f} dB {trend} "
                              f"(early: {first_perf[speaker]:5.2f} → late: {last_perf[speaker]:5.2f})")
                        
        # Data split analysis (if available)
        if 'split' in df.columns and df['split'].nunique() > 1:
            print(f"\nData Split Analysis (SI-SDR):")
            print("-" * 60)
            split_stats = df.groupby('split')['si_sdr_db'].agg(['mean', 'std', 'count']).round(2)
            for split, stats in split_stats.iterrows():
                print(f"  {split.capitalize():5}: {stats['mean']:6.2f} ± {stats['std']:4.2f} dB  (n={stats['count']})")
                
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Analyze audio separation results')
    parser.add_argument('--data_dir', 
                       default='data/working_dir/experiments/ha-joint-uni/train_ha/train_samples/',
                       help='Directory containing audio files')
    parser.add_argument('--output', '-o',
                       help='Output CSV file to save results')
    parser.add_argument('--summary_only', action='store_true',
                       help='Only print summary, don\'t save detailed results')
                       
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} does not exist")
        return 1
        
    # Run analysis
    analyzer = AudioAnalyzer(args.data_dir)
    results_df = analyzer.analyze_all()
    
    if results_df.empty:
        print("No results to analyze")
        return 1
        
    # Print summary
    analyzer.print_summary(results_df)
    
    # Save results if requested
    if not args.summary_only:
        output_file = args.output or f"{args.data_dir.rstrip('/')}_analysis_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
        
    return 0


if __name__ == '__main__':
    exit(main())