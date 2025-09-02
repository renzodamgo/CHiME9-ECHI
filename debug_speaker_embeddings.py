#!/usr/bin/env python3

import sys
from pathlib import Path
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def analyze_rainbow_audio_files(dataset_type="train", max_files=20):
    """
    Analyze the rainbow passage audio files to check for speaker distinctiveness.
    """
    print("🌈 ANALYZING RAINBOW PASSAGE AUDIO FILES")
    print("=" * 60)
    
    rainbow_dir = Path(f"data/working_dir/participant/{dataset_type}")
    
    if not rainbow_dir.exists():
        print(f"❌ Rainbow directory not found: {rainbow_dir}")
        return None
    
    rainbow_files = list(rainbow_dir.glob("*.wav"))[:max_files]
    
    if not rainbow_files:
        print(f"❌ No rainbow audio files found in {rainbow_dir}")
        return None
    
    print(f"📁 Found {len(rainbow_files)} rainbow audio files")
    
    audio_stats = {}
    spectral_features = {}
    
    for i, audio_file in enumerate(rainbow_files):
        pid = audio_file.stem
        print(f"\n🎤 Analyzing {pid} ({i+1}/{len(rainbow_files)})")
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_file)
            waveform = waveform.squeeze(0)  # Remove channel dim if mono
            
            # Basic audio statistics
            duration = len(waveform) / sr
            rms = torch.sqrt(torch.mean(waveform**2)).item()
            peak = torch.max(torch.abs(waveform)).item()
            
            audio_stats[pid] = {
                'duration': duration,
                'rms': rms,
                'peak': peak,
                'sample_rate': sr,
                'length': len(waveform)
            }
            
            print(f"  Duration: {duration:.2f}s, RMS: {rms:.6f}, Peak: {peak:.6f}")
            
            # Spectral analysis
            if len(waveform) > sr:  # At least 1 second
                # Take middle 1 second for analysis
                start_idx = len(waveform) // 2 - sr // 2
                segment = waveform[start_idx:start_idx + sr]
                
                # Compute spectrum
                fft = torch.fft.rfft(segment)
                magnitude = torch.abs(fft)
                
                # Spectral centroid
                freqs = torch.linspace(0, sr/2, len(magnitude))
                spectral_centroid = torch.sum(freqs * magnitude) / (torch.sum(magnitude) + 1e-8)
                
                # Spectral rolloff (95% energy point)
                cumulative_energy = torch.cumsum(magnitude**2, dim=0)
                total_energy = cumulative_energy[-1]
                rolloff_idx = torch.where(cumulative_energy >= 0.95 * total_energy)[0]
                spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
                
                # Spectral bandwidth
                spectral_bandwidth = torch.sqrt(torch.sum(((freqs - spectral_centroid)**2) * magnitude) / (torch.sum(magnitude) + 1e-8))
                
                spectral_features[pid] = {
                    'centroid': spectral_centroid.item(),
                    'rolloff': spectral_rolloff.item(), 
                    'bandwidth': spectral_bandwidth.item(),
                    'rms_db': 20 * np.log10(rms + 1e-8)
                }
                
                print(f"  Spectral centroid: {spectral_centroid:.0f} Hz")
                print(f"  Spectral rolloff: {spectral_rolloff:.0f} Hz")
                print(f"  RMS dB: {spectral_features[pid]['rms_db']:.1f} dB")
                
        except Exception as e:
            print(f"  ❌ Error analyzing {pid}: {e}")
            continue
    
    # Analysis summary
    print(f"\n📊 RAINBOW AUDIO ANALYSIS SUMMARY")
    print("=" * 50)
    
    if audio_stats:
        durations = [stats['duration'] for stats in audio_stats.values()]
        rms_values = [stats['rms'] for stats in audio_stats.values()]
        
        print(f"Duration range: {min(durations):.1f}s - {max(durations):.1f}s (avg: {np.mean(durations):.1f}s)")
        print(f"RMS range: {min(rms_values):.6f} - {max(rms_values):.6f}")
        
        # Check for potential issues
        if max(durations) / min(durations) > 2:
            print("⚠️  WARNING: Large duration variation detected!")
        
        if max(rms_values) / min(rms_values) > 10:
            print("⚠️  WARNING: Large volume variation detected!")
    
    if spectral_features:
        centroids = [feat['centroid'] for feat in spectral_features.values()]
        print(f"Spectral centroid range: {min(centroids):.0f} - {max(centroids):.0f} Hz")
        
        if max(centroids) - min(centroids) < 200:
            print("⚠️  WARNING: Low spectral diversity - speakers may sound similar!")
        
        # Identify most distinctive speakers
        print(f"\n🎭 SPEAKER DISTINCTIVENESS RANKING:")
        print("-" * 40)
        
        # Sort by spectral centroid (voice pitch indicator)
        sorted_speakers = sorted(spectral_features.items(), key=lambda x: x[1]['centroid'])
        
        for i, (pid, features) in enumerate(sorted_speakers):
            print(f"{i+1:2d}. {pid}: centroid={features['centroid']:.0f}Hz, "
                  f"rolloff={features['rolloff']:.0f}Hz, rms={features['rms_db']:.1f}dB")
    
    return audio_stats, spectral_features

def check_speaker_embedding_diversity(audio_stats, spectral_features, output_dir="debug_embeddings"):
    """
    Create visualizations to understand speaker diversity issues.
    """
    if not spectral_features:
        print("No spectral features available for visualization")
        return
        
    Path(output_dir).mkdir(exist_ok=True)
    
    # Extract features for plotting
    pids = list(spectral_features.keys())
    centroids = [spectral_features[pid]['centroid'] for pid in pids]
    rolloffs = [spectral_features[pid]['rolloff'] for pid in pids]
    bandwidths = [spectral_features[pid]['bandwidth'] for pid in pids]
    rms_dbs = [spectral_features[pid]['rms_db'] for pid in pids]
    
    # 1. Spectral characteristics scatter plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(centroids, rolloffs, alpha=0.7, s=100)
    for i, pid in enumerate(pids):
        plt.annotate(pid, (centroids[i], rolloffs[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    plt.xlabel('Spectral Centroid (Hz)')
    plt.ylabel('Spectral Rolloff (Hz)')
    plt.title('Speaker Voice Characteristics')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.scatter(centroids, rms_dbs, alpha=0.7, s=100, c=bandwidths, cmap='viridis')
    plt.colorbar(label='Spectral Bandwidth (Hz)')
    plt.xlabel('Spectral Centroid (Hz)')
    plt.ylabel('RMS Level (dB)')
    plt.title('Voice Characteristics vs Volume')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.hist(centroids, bins=10, alpha=0.7, edgecolor='black')
    plt.xlabel('Spectral Centroid (Hz)')
    plt.ylabel('Count')
    plt.title('Distribution of Voice Pitch')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.hist(rms_dbs, bins=10, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel('RMS Level (dB)')
    plt.ylabel('Count')
    plt.title('Distribution of Volume Levels')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/speaker_characteristics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Similarity matrix
    n_speakers = len(pids)
    similarity_matrix = np.zeros((n_speakers, n_speakers))
    
    # Normalize features for similarity computation
    features_array = np.array([centroids, rolloffs, bandwidths, rms_dbs]).T
    features_normalized = (features_array - features_array.mean(axis=0)) / (features_array.std(axis=0) + 1e-8)
    
    for i in range(n_speakers):
        for j in range(n_speakers):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Cosine similarity
                dot_product = np.dot(features_normalized[i], features_normalized[j])
                norm_i = np.linalg.norm(features_normalized[i])
                norm_j = np.linalg.norm(features_normalized[j])
                similarity_matrix[i, j] = dot_product / (norm_i * norm_j + 1e-8)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, 
                xticklabels=pids, yticklabels=pids,
                annot=True, fmt='.3f', cmap='RdYlBu_r',
                center=0.5, vmin=0, vmax=1)
    plt.title('Speaker Acoustic Similarity Matrix\n(High values = Similar voices)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/speaker_similarity_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find most similar speaker pairs
    print(f"\n🔍 MOST SIMILAR SPEAKER PAIRS:")
    print("-" * 40)
    
    similar_pairs = []
    for i in range(n_speakers):
        for j in range(i+1, n_speakers):
            similarity = similarity_matrix[i, j]
            similar_pairs.append((pids[i], pids[j], similarity))
    
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for i, (pid1, pid2, sim) in enumerate(similar_pairs[:5]):
        status = "🚨 VERY SIMILAR" if sim > 0.95 else "⚠️ SIMILAR" if sim > 0.85 else "✅ DISTINCT"
        print(f"{i+1}. {pid1} vs {pid2}: {sim:.3f} {status}")
    
    print(f"\n📊 Plots saved to {output_dir}/")
    return similarity_matrix, similar_pairs

def suggest_speaker_selection_strategy(similar_pairs, spectral_features):
    """
    Suggest strategies to improve speaker distinctiveness.
    """
    print(f"\n💡 SPEAKER SELECTION RECOMMENDATIONS:")
    print("=" * 50)
    
    # Find most distinctive speakers
    pids = list(spectral_features.keys())
    centroids = [spectral_features[pid]['centroid'] for pid in pids]
    
    # Sort by spectral centroid to find diverse voices
    sorted_by_pitch = sorted(zip(pids, centroids), key=lambda x: x[1])
    
    # Recommend diverse triplets
    if len(sorted_by_pitch) >= 3:
        low_pitch = sorted_by_pitch[0][0]
        high_pitch = sorted_by_pitch[-1][0]
        mid_pitch = sorted_by_pitch[len(sorted_by_pitch)//2][0]
        
        print(f"🎯 RECOMMENDED SPEAKER COMBINATION:")
        print(f"   Low pitch:  {low_pitch} ({sorted_by_pitch[0][1]:.0f} Hz)")
        print(f"   Mid pitch:  {mid_pitch} ({sorted_by_pitch[len(sorted_by_pitch)//2][1]:.0f} Hz)")
        print(f"   High pitch: {high_pitch} ({sorted_by_pitch[-1][1]:.0f} Hz)")
    
    print(f"\n🛠️  POTENTIAL SOLUTIONS:")
    print("1. **Data Augmentation**: Apply pitch shifting, speed changes to increase diversity")
    print("2. **Speaker Filtering**: Remove acoustically similar speakers from training")  
    print("3. **Embedding Regularization**: Add speaker distinctiveness loss")
    print("4. **Model Architecture**: Increase speaker embedding dimension")
    print("5. **Training Strategy**: Use contrastive learning for speaker embeddings")

def main():
    parser = argparse.ArgumentParser(description='Debug CHiME-9 ECHI speaker embedding issues')
    parser.add_argument('--dataset', type=str, default='train', 
                       help='Dataset subset to analyze (train/dev)')
    parser.add_argument('--max_files', type=int, default=20,
                       help='Maximum number of audio files to analyze')
    parser.add_argument('--output_dir', type=str, default='debug_embeddings',
                       help='Output directory for plots and analysis')
    
    args = parser.parse_args()
    
    print("🔍 CHiME-9 ECHI Speaker Embedding Debug Analysis")
    print("=" * 60)
    
    # Analyze rainbow audio files
    audio_stats, spectral_features = analyze_rainbow_audio_files(
        dataset_type=args.dataset, 
        max_files=args.max_files
    )
    
    if not audio_stats:
        print("❌ No audio data available for analysis")
        return
    
    # Create visualizations and similarity analysis
    similarity_matrix, similar_pairs = check_speaker_embedding_diversity(
        audio_stats, spectral_features, args.output_dir
    )
    
    # Suggest improvements
    suggest_speaker_selection_strategy(similar_pairs, spectral_features)
    
    print(f"\n✅ Analysis completed! Check {args.output_dir}/ for visualizations")

if __name__ == "__main__":
    main()