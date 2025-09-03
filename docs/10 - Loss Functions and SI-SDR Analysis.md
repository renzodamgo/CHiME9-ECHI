# 10 - Loss Functions and SI-SDR Analysis

## Overview

The CHiME9-ECHI training pipeline uses a sophisticated joint loss function combining STFT-domain separation loss with Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) time-domain loss. This document analyzes the mathematical foundations, implementation details, and rationale behind the choice of SI-SDR as the primary optimization target.

## Joint Loss Architecture

### Loss Function Composition

The joint loss combines two complementary components:

```python
loss = stft_weight * L_sep + sisdr_weight * sisdr_loss
```

**Default Configuration:**
- `stft_weight = 0.0` (STFT loss disabled by default)
- `sisdr_weight = 1.0` (Primary focus on SI-SDR)

This configuration prioritizes perceptual quality (SI-SDR) over frequency-domain reconstruction accuracy.

### 1. STFT-Domain Separation Loss

```python
# Compute magnitude error in frequency domain
error_mag = torch.abs(S_hat_c - Y_ref_c)  # [B, K, T, F]
n_freqs = error_mag.shape[-1]
freq_weights = torch.ones(n_freqs, device=S_hat_c.device)  # Uniform weighting
L_sep = (error_mag * freq_weights.view(1, 1, 1, -1)).mean()
```

**Characteristics:**
- **Domain:** Complex STFT coefficients
- **Metric:** L1 magnitude error between predicted and target spectra
- **Weighting:** Uniform across frequencies (prevents high-frequency suppression)
- **Purpose:** Ensures spectral reconstruction accuracy

### 2. SI-SDR Time-Domain Loss (Primary Component)

```python
# Convert to time domain and compute SI-SDR
s_hat_wav = stft.inverse(S_hat_c, lengths=tgt_lens)  # [B, K, Tw']
sisdr_per_spk = _sisdr(s_hat_wav_matched, y_wav_matched)  # [B, K]
sisdr_loss = -_compute_balanced_sisdr_loss(sisdr_per_spk, active_mask=active_mask)
```

**Characteristics:**
- **Domain:** Time-domain waveforms
- **Metric:** Scale-Invariant Signal-to-Distortion Ratio
- **Balancing:** Equal weighting across active speakers
- **Purpose:** Perceptual quality optimization

## Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

### Mathematical Foundation

SI-SDR is defined as the ratio between target signal energy and distortion energy, with scale invariance:

```python
def _sisdr(x, s, eps=1e-8):
    # x: estimated signal [B,K,T], s: target signal [B,K,T]
    # Remove DC component (zero-mean)
    x_zm = x - x.mean(dim=-1, keepdim=True)
    s_zm = s - s.mean(dim=-1, keepdim=True)
    
    # Optimal scaling factor α = (x·s) / ||s||²
    t = (torch.sum(x_zm * s_zm, dim=-1, keepdim=True) 
         / (torch.sum(s_zm**2, dim=-1, keepdim=True) + eps)) * s_zm
    
    # Distortion: e = x - α·s
    e = x_zm - t
    
    # SI-SDR in dB: 10·log₁₀(||α·s||² / ||e||²)
    return 10 * torch.log10(
        (torch.sum(t**2, dim=-1) + eps) / (torch.sum(e**2, dim=-1) + eps)
    )
```

**Key Properties:**
- **Scale Invariance:** Optimal scaling α automatically computed
- **DC Removal:** Zero-mean normalization prevents bias
- **Distortion Focus:** Measures residual error after optimal alignment
- **Perceptual Relevance:** Correlates well with human perception

### Why SI-SDR for Speech Separation?

#### 1. **Scale Invariance**
Neural networks often produce outputs at different scales than targets. SI-SDR automatically finds the optimal scaling factor, making it robust to amplitude variations:

```
SI-SDR = 10 · log₁₀(||α·s_target||² / ||α·s_target - ŝ||²)
```

#### 2. **Perceptual Correlation**
Unlike MSE or L1 losses that treat all samples equally, SI-SDR focuses on signal quality relative to noise/distortion, which correlates better with human perception.

#### 3. **Gradient Properties**
SI-SDR provides meaningful gradients for optimization:
- Higher values indicate better separation quality
- Differentiable with respect to model parameters
- Stable gradients across different signal amplitudes

#### 4. **Multi-Speaker Suitability**
For speech separation tasks:
- Each speaker's SI-SDR computed independently
- Balanced aggregation prevents speaker hierarchy collapse
- Compatible with speaker activity masking

### Comparison with Alternative Losses

| Loss Function | Domain | Scale Inv. | Perceptual | Multi-Speaker | Gradient Quality |
|---------------|---------|------------|------------|---------------|------------------|
| **SI-SDR** | Time | ✅ Yes | ✅ High | ✅ Excellent | ✅ Stable |
| MSE | Time | ❌ No | ❌ Low | ⚠️ Moderate | ✅ Stable |
| L1 | Time | ❌ No | ❌ Low | ⚠️ Moderate | ⚠️ Sparse |
| STFT L1 | Frequency | ❌ No | ⚠️ Moderate | ✅ Good | ✅ Stable |
| PESQ | Time | ✅ Yes | ✅ High | ❌ Poor | ❌ Non-diff |

## Balanced SI-SDR Loss

### Speaker Balancing Strategy

```python
def _compute_balanced_sisdr_loss(sisdr_per_spk, active_mask=None):
    if active_mask is not None:
        # Only compute loss for active speakers
        sisdr_masked = sisdr_per_spk * active_mask.float()  # [B, K]
        active_count = active_mask.sum(dim=-1, keepdim=True).float()  # [B, 1]
        active_count = torch.clamp(active_count, min=1.0)  # Avoid division by zero
        
        # Average over active speakers only
        balanced_sisdr_per_sample = sisdr_masked.sum(dim=-1) / active_count.squeeze(-1)
    else:
        # Equal weighting for all speakers
        equal_weights = torch.ones_like(sisdr_per_spk) / sisdr_per_spk.size(-1)
        balanced_sisdr_per_sample = (sisdr_per_spk * equal_weights).sum(dim=-1)
    
    return balanced_sisdr_per_sample.mean()  # Global balanced SI-SDR
```

### Preventing Speaker Hierarchy Collapse

**Problem:** Without balancing, models may:
- Focus on easy speakers while abandoning difficult ones
- Create speaker hierarchies where some outputs degrade
- Converge to sub-optimal local minima

**Solution:** Equal weighting strategy:
- All active speakers receive equal optimization attention
- Inactive speakers (silent periods) properly masked out
- Balanced improvement across the entire speaker set

### Active Speaker Masking

```python
# Extract activity mask from batch
active_mask = batch.get("speaker_active_mask", None)  # [B, K] boolean
if active_mask is not None:
    active_mask = active_mask.to(S_hat_c.device)
    
# Apply mask in loss computation
sisdr_loss = -_compute_balanced_sisdr_loss(sisdr_per_spk, active_mask=active_mask)
```

**Benefits:**
- **Silent Speaker Handling:** Inactive speakers don't contribute to loss
- **Dynamic Activity:** Accommodates varying speaker participation
- **Training Stability:** Prevents optimization of silent targets

## Speaker Separation Quality Analysis

### Comprehensive Separation Metrics

The loss function includes extensive separation quality analysis:

```python
def analyze_speaker_separation(s_hat_wav, y_wav):
    stats = {}
    
    # 1. Cross-speaker correlation (lower is better)
    cross_correlations = []
    for i in range(K):
        for j in range(i + 1, K):
            corr = pearson_correlation(s_hat_wav[b, i], s_hat_wav[b, j])
            cross_correlations.append(abs(corr))
    
    # 2. Speaker distinctness: L2 distance between outputs
    pairwise_distances = []
    for i in range(K):
        for j in range(i + 1, K):
            dist = torch.norm(s_hat_wav[b, i] - s_hat_wav[b, j])
            pairwise_distances.append(dist)
    
    # 3. Energy distribution balance
    speaker_energies = [(spk_wav ** 2).mean() for spk_wav in s_hat_wav]
    energy_std = torch.tensor(speaker_energies).std()
    
    # 4. Spectral diversity analysis
    spectral_centroid_diff = compute_spectral_centroids(s_hat_wav)
    
    return stats
```

### Separation Quality Scoring

```python
# Composite separation quality score (0-1, higher is better)
separation_score = (
    (1.0 - cross_speaker_corr_mean) +          # Low correlation = good
    min(speaker_l2_distance_mean / 10.0, 1.0) + # High distance = good  
    (1.0 / (1.0 + speaker_energy_std))          # Balanced energy = good
) / 3.0

# Quality thresholds and logging
if separation_score < 0.3:
    logging.warning("🚨 POOR SPEAKER SEPARATION DETECTED!")
elif separation_score > 0.7:
    logging.info("✅ Good speaker separation detected")
```

## Output Collapse Detection

### Collapse Detection Metrics

```python
def check_collapse(s_hat_wav, y_wav):
    # Pairwise differences between speakers
    if K >= 2:
        d01 = torch.mean(torch.abs(s_hat_wav[:, 0] - s_hat_wav[:, 1]))
        stats["mean_|s0-s1|"] = d01
    
    # Target correlation for quality assessment
    def _corr(a, b):
        num = (a * b).sum()
        den = a.pow(2).sum().sqrt() * b.pow(2).sum().sqrt() + 1e-12
        return (num / den).item()
    
    stats["corr_k0"] = _corr(s_hat_wav[0, 0], y_wav[0, 0])
    return stats
```

**Warning Signs of Collapse:**
- `mean_|s0-s1|` → 0 (speakers becoming identical)
- `cross_speaker_corr_mean` → 1 (high correlation between speakers)
- `speaker_l2_distance_mean` → 0 (speakers converging)

## Loss Configuration and Hyperparameters

### Default Configuration

```yaml
# config/train/train.yaml - Loss section
loss:
    name: joint_loss
    kwargs:
        # STFT parameters
        fft_size: 1024
        hop_size: 256
        win_length: 1024
        window: hann_window
        
        # Loss component weights (implicitly)
        # stft_weight: 0.0    (disabled)
        # sisdr_weight: 1.0   (primary)
```

### Training Implementation

```python
# In train_script_multi.py
loss_fn = joint_loss
loss, stats = loss_fn(
    S_hat_c=model_output,      # [B, K, T, F] complex
    Y_ref_c=target_stft,       # [B, K, T, F] complex  
    batch=batch_data,
    stft=stft_module,
    weights=(0.0, 1.0)         # (stft_weight, sisdr_weight)
)
```

## Training Dynamics and Monitoring

### Key Metrics Tracked

```python
stats = {
    "loss": float(loss.detach()),                    # Total joint loss
    "L_sep": float(L_sep.detach()),                  # STFT separation loss
    "sisdr_loss": float(sisdr_loss.detach()),        # SI-SDR loss (negative)
    "sisdr_db": float(balanced_sisdr.detach()),      # Actual SI-SDR in dB
    
    # Per-speaker analysis
    "sisdr_per_spk": [float(sisdr_per_spk[0, k]) for k in range(K)],
    "s_hat_rms_per_spk": [...],                      # RMS per speaker
    "y_ref_rms_per_spk": [...],                      # Target RMS per speaker
    
    # Separation quality
    "cross_speaker_corr_mean": float(...),           # Cross-speaker correlation
    "speaker_l2_distance_mean": float(...),          # Speaker distinctness
    "separation_quality_score": float(...),          # Composite score
}
```

### Training Progress Indicators

**Healthy Training:**
- SI-SDR values increasing (less negative)
- `separation_quality_score` > 0.7
- `cross_speaker_corr_mean` < 0.3
- Balanced per-speaker SI-SDR improvements

**Warning Signs:**
- SI-SDR stagnation or degradation
- High cross-speaker correlation (> 0.5)
- Large variation in per-speaker performance
- Output collapse indicators

## Advanced Loss Variations

### Alternative Balancing Strategies

The current implementation uses equal weighting, but alternative approaches exist:

```python
# Current: Equal weighting (implemented)
equal_weights = torch.ones_like(sisdr_per_spk) / sisdr_per_spk.size(-1)

# Alternative 1: Inverse performance weighting (not used)
inverse_weights = torch.softmax(-sisdr_per_spk.detach(), dim=-1)

# Alternative 2: Adaptive weighting (not used)
adaptive_weights = compute_adaptive_weights(sisdr_per_spk, epoch, speaker_history)
```

**Rationale for Equal Weighting:**
- Prevents abandonment of difficult speakers
- Ensures balanced improvement across all speakers
- Avoids training instability from adaptive schemes
- Simple and interpretable optimization objective

## Computational Efficiency

### Memory and Performance Considerations

```python
# Efficient SI-SDR computation
def _sisdr(x, s, eps=1e-8):
    # Memory-efficient zero-mean normalization
    x_zm = x - x.mean(dim=-1, keepdim=True)  # In-place possible
    s_zm = s - s.mean(dim=-1, keepdim=True)  # In-place possible
    
    # Vectorized operations for [B, K, T] tensors
    numerator = torch.sum(x_zm * s_zm, dim=-1, keepdim=True)    # [B, K, 1]
    denominator = torch.sum(s_zm**2, dim=-1, keepdim=True)      # [B, K, 1]
    
    # Broadcasting and log computation
    return 10 * torch.log10((target_power + eps) / (error_power + eps))
```

**Optimization Features:**
- Batch processing for [B, K, T] tensors
- In-place operations where possible  
- Numerical stability with epsilon terms
- GPU-optimized tensor operations

## Summary

The CHiME9-ECHI loss function design prioritizes perceptual quality through SI-SDR optimization while maintaining training stability via balanced multi-speaker learning. Key advantages:

1. **Perceptual Relevance:** SI-SDR correlates with human perception better than spectral losses
2. **Scale Invariance:** Robust to amplitude variations in neural network outputs
3. **Multi-Speaker Balance:** Equal weighting prevents speaker hierarchy collapse
4. **Activity Awareness:** Proper handling of silent speakers through masking
5. **Quality Monitoring:** Comprehensive separation analysis and collapse detection
6. **Training Stability:** Stable gradients and interpretable optimization objectives

The focus on SI-SDR over STFT loss (`weights=(0.0, 1.0)`) reflects the priority on end-to-end perceptual quality for hearing aid applications, where time-domain reconstruction quality directly impacts user experience.