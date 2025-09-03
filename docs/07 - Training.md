# 07 - Training Process

This document provides a comprehensive overview of the CHiME9-ECHI training process, focusing on the multi-speaker architecture, input processing pipeline, and output generation in `train_script_multi.py` and `joint_multi.py`.

## Table of Contents

1. [Training Overview](#training-overview)
2. [Input Processing Pipeline](#input-processing-pipeline)
3. [STFT Transformation](#stft-transformation)
4. [Model Forward Pass](#model-forward-pass)
5. [Loss Computation](#loss-computation)
6. [Output Generation](#output-generation)
7. [Training Loop Architecture](#training-loop-architecture)
8. [Validation Process](#validation-process)
9. [Optimization Strategies](#optimization-strategies)
10. [Monitoring & Debugging](#monitoring--debugging)

## Training Overview

The CHiME9-ECHI training system is designed for multi-speaker target speaker extraction using a speaker-conditional approach. The training process transforms audio inputs through STFT domain processing and produces separated spectrograms for each target speaker.

### Key Components

- **Input**: Multi-channel noisy mixture + speaker enrollments (rainbow passages)
- **Processing**: STFT domain separation with speaker conditioning
- **Output**: Complex spectrograms for each target speaker
- **Loss**: Joint STFT + SI-SDR loss with speaker activity masking

### Architecture Flow

```
[Noisy Mixture] + [Rainbow Enrollments] 
    ↓ (Audio Preprocessing)
[Normalized Audio Tensors]
    ↓ (STFT Transform)
[Complex Spectrograms]
    ↓ (Model Forward Pass)
[Separated Complex Spectrograms]
    ↓ (Loss Computation)
[Joint Loss (STFT + SI-SDR)]
```

## Input Processing Pipeline

The training system processes three types of inputs from the ECHIJoint dataset:

### 1. Noisy Mixture Signal

**Source**: Multi-channel hearing aid recordings
```python
# Shape: [B, C, Tw] where C=4 channels, Tw=time samples
noisy = batch["noisy"].to(device, non_blocking=True)

# Preprocessing pipeline
noisy = prep_audio(
    noisy, 
    batch["fs"],          # Original sample rate (48kHz)
    input_channels=4,     # Target channels  
    input_sr=16000,       # Target sample rate
    input_rms=0.01,       # Target RMS level
    batched=True
)
```

**Processing Steps**:
1. **Resampling**: 48kHz → 16kHz for computational efficiency
2. **Channel handling**: Maintain 4-channel structure
3. **RMS normalization**: Standardize signal levels to 0.01
4. **Tensor formatting**: Ensure proper dimensions for STFT

### 2. Speaker Enrollments (Rainbow Passages)

**Source**: Full rainbow passage recordings per speaker
```python
# Shape: [B, K, Tr] where K=speakers, Tr=enrollment length
spk_all = batch["spkid_all"].to(device, non_blocking=True)

# Efficient batch processing
B, K, T_spk = spk_all.shape
spk_all = spk_all.view(-1, T_spk).unsqueeze(1)  # [B*K, 1, T_spk]

# Apply same preprocessing as noisy signal
spk_all = prep_audio(spk_all, batch["fs"], 1, input_sr, input_rms, True)
spk_all = spk_all.squeeze(1).view(B, K, -1)  # [B, K, T_spk']
```

**Key Features**:
- **Full passages**: Complete rainbow recordings (not segmented)
- **Speaker identity**: Provides rich speaker characteristics
- **Batch processing**: Efficient processing of multiple speakers
- **Length preservation**: Original lengths tracked for attention masking

### 3. Target Signals

**Source**: Clean reference signals per active speaker
```python
# Shape: [B, K, Tw] where K matches number of speakers
targ_all = batch["target_all"].to(device, non_blocking=True)

# Note: Silent speakers have zero tensors as targets
# Active speakers have clean reference audio
# Speaker activity tracked in batch["speaker_active_mask"]
```

**Speaker Activity Handling**:
```python
# Example for 3-speaker scenario with 2 active speakers
speaker_active_mask = [True, False, True]  # Speakers 0,2 active
target_all = [
    clean_audio_speaker_0,    # Active speaker
    torch.zeros(Tw),          # Silent speaker (zero target)  
    clean_audio_speaker_2     # Active speaker
]
```

## STFT Transformation

The system converts time-domain signals to frequency-domain representations for processing.

### Configuration

```python
# STFT parameters (from config/train/model.yaml)
stft_config = {
    "n_fft": 128,           # FFT window size
    "win_length": 128,      # Window length
    "hop_length": 64,       # Hop size (50% overlap)
    "window": "hann"        # Window function
}

stft = STFTWrapper(**stft_config, device=device)
```

### Transformation Pipeline

```python
# 1. Transform noisy mixture
noisy_tf = stft(noisy)  # [B, M, T, F, 2] where 2=[real, imag]

# 2. Transform speaker enrollments  
spk_all_tf = stft(spk_all)  # [B, K, F, T, 2]

# 3. Permute for model input format
spk_all_for_model = spk_all_tf.permute(0, 1, 3, 2, 4).contiguous()  # [B, K, T, F, 2]

# 4. Adjust speaker lengths for STFT frames
spk_lens_all = (batch["spkid_lens_all"] - stft.n_fft) // stft.hop_length

# 5. Transform target references for loss computation
Y_ref_tf = stft(targ_all)  # [B, K, 2, T, F]  
Y_ref_c = torch.complex(Y_ref_tf[..., 0], Y_ref_tf[..., 1])  # Complex format
Y_ref_c = Y_ref_c.permute(0, 1, 3, 2).contiguous()  # [B, K, T, F] complex
```

### STFT Domain Advantages

- **Frequency selectivity**: Better modeling of spectral patterns
- **Computational efficiency**: Parallel frequency processing
- **Speaker characteristics**: Frequency-domain speaker features
- **Separation quality**: Direct spectral masking capabilities

## Model Forward Pass

The MCxTFGridNet model processes the STFT inputs through speaker-conditional pathways.

### Architecture Overview

```python
# Model configuration (n_srcs=3 processing chains)
model = MCxTFGridNet(
    n_srcs=3,                    # 3 speaker processing chains
    n_imics=4,                   # 4 input microphone channels  
    n_layers=3,                  # 3 TFGridNet layers
    lstm_hidden_units=128,       # LSTM hidden size
    emb_dim=64                   # Embedding dimension
)
```

### Processing Flow

```python
# Forward pass with mixed precision
with autocast("cuda", dtype=torch.bfloat16):
    S_hat_c = model(noisy_tf, spk_all_for_model, spk_lens_all)
```

**Internal Model Processing**:

1. **Speaker Embedding Extraction**:
   ```python
   # Extract speaker embeddings from rainbow passages
   spk_feat = spk_all_for_model.reshape(B * K, 2, T, F)
   spk_feat = model.spk_conv(spk_feat)  # [BK, C, T, F]
   
   # Attention pooling for speaker embeddings  
   speaker_embeddings = model.aux_enc(spk_feat, spk_lens_all)  # [BK, C]
   speaker_embeddings = speaker_embeddings.view(B, K, -1)  # [B, K, C]
   ```

2. **Speaker-Conditional Processing**:
   ```python
   for k in range(K):  # For each speaker
       spk_emb = speaker_embeddings[:, k]  # [B, C]
       
       # Use dedicated processing chain (or shared if k >= n_srcs)
       chain_idx = min(k, model.n_srcs - 1)
       
       # Speaker-conditional mixture processing
       z_k = model.speaker_conditional_conv(mixture_features, spk_emb)
       
       # Process through speaker-specific layers
       for layer in range(model.n_layers):
           z_k = model.speaker_fusions[chain_idx][layer](spk_emb, z_k)
           z_k = model.speaker_gridnets[chain_idx][layer](z_k)
       
       # Generate output spectrogram
       output_k = model.speaker_output_heads[chain_idx](z_k)
   ```

3. **Output Generation**:
   ```python
   # Stack all speaker outputs: [B, K, 2, T, F]
   out_ri = torch.stack(speaker_outputs, dim=1)
   
   # Convert to complex format: [B, K, T, F] complex
   S_hat_c = torch.complex(out_ri[..., 0], out_ri[..., 1])
   ```

### Processing Chain Limitations

With current `n_srcs=3` configuration:
- **Speaker 0 → Chain 0** (dedicated)
- **Speaker 1 → Chain 1** (dedicated)  
- **Speaker 2 → Chain 2** (dedicated)
- **Speaker 3+ → Chain 2** (shared - potential bottleneck)

## Loss Computation

The training uses a joint loss combining STFT-domain and time-domain objectives.

### Joint Loss Function (`joint_multi.py`)

```python
def joint_loss(S_hat_c, Y_ref_c, batch, stft, weights=(0.0, 1.0)):
    """
    Joint loss combining STFT separation loss with SI-SDR time-domain loss
    
    Args:
        S_hat_c: [B, K, T, F] complex - model estimates
        Y_ref_c: [B, K, T, F] complex - target references  
        batch: batch data with speaker activity masks
        stft: STFT wrapper for inverse transform
        weights: (stft_weight, sisdr_weight) - default focuses on SI-SDR
    """
```

### Loss Components

#### 1. STFT-Domain Separation Loss

```python
# L_sep: Frequency-domain separation quality
error_mag = torch.abs(S_hat_c - Y_ref_c)  # [B, K, T, F]
freq_weights = torch.ones(n_freqs, device=S_hat_c.device)  # Uniform weighting
L_sep = (error_mag * freq_weights.view(1, 1, 1, -1)).mean()
```

**Purpose**: Direct spectral matching between predicted and target spectrograms

#### 2. SI-SDR Time-Domain Loss  

```python
# Convert to time domain for perceptual quality assessment
s_hat_wav = stft.inverse(S_hat_c, lengths=batch["target_lens_all"])
y_wav = batch["target_all"]

# Compute SI-SDR per speaker: [B, K] (higher is better)
sisdr_per_spk = _sisdr(s_hat_wav, y_wav)

# Apply speaker activity masking
active_mask = batch["speaker_active_mask"]  # [B, K] boolean
sisdr_loss = -_compute_balanced_sisdr_loss(sisdr_per_spk, active_mask=active_mask)
```

**SI-SDR Computation**:
```python
def _sisdr(x, s, eps=1e-8):
    """Scale-Invariant Signal-to-Distortion Ratio"""
    # Zero-mean signals
    x_zm = x - x.mean(dim=-1, keepdim=True)
    s_zm = s - s.mean(dim=-1, keepdim=True)
    
    # Target scaling
    alpha = torch.sum(x_zm * s_zm, dim=-1, keepdim=True) / (torch.sum(s_zm**2, dim=-1, keepdim=True) + eps)
    target = alpha * s_zm
    
    # Noise component  
    noise = x_zm - target
    
    # SI-SDR in dB (higher is better)
    return 10 * torch.log10((torch.sum(target**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps))
```

#### 3. Speaker Activity Masking

```python
def _compute_balanced_sisdr_loss(sisdr_per_spk, active_mask):
    """Compute loss only for active speakers"""
    if active_mask is not None:
        # Zero out inactive speakers
        sisdr_masked = sisdr_per_spk * active_mask.float()  # [B, K]
        active_count = active_mask.sum(dim=-1, keepdim=True).float()  # [B, 1]
        
        # Average over active speakers only
        balanced_sisdr = sisdr_masked.sum(dim=-1) / active_count.squeeze(-1)
    else:
        # Equal weighting for all speakers
        balanced_sisdr = sisdr_per_spk.mean(dim=-1)
        
    return balanced_sisdr.mean()
```

### Final Loss Combination

```python
# Default weights: (stft_weight=0.0, sisdr_weight=1.0)
loss = stft_weight * L_sep + sisdr_weight * sisdr_loss

# Current configuration focuses on SI-SDR for perceptual quality
```

## Output Generation

The model produces complex spectrograms that are converted back to time-domain audio.

### Spectrogram Output Format

```python
# Model output: Complex STFT coefficients
S_hat_c = model(noisy_tf, spk_all_for_model, spk_lens_all)  # [B, K, T, F] complex

# Where:
# B = batch size (typically 1)
# K = number of speakers (typically 3)  
# T = time frames (depends on segment length)
# F = frequency bins (65 for n_fft=128)
```

### Time-Domain Conversion

```python
# Convert to waveform using inverse STFT
target_lens = batch["target_lens_all"]  # [B, K] - original lengths
s_hat_wav = stft.inverse(S_hat_c, lengths=target_lens)  # [B, K, T_wav]

# Shape: [B, K, T_wav] where T_wav is time samples
```

### Sample Saving

```python
# Save samples for monitoring training progress
if epoch % 2 == 0:  # Save every other epoch
    s_hat_wav_cpu = s_hat_wav.detach().cpu()
    
    # Save per-speaker outputs
    for k in range(num_speakers):
        save_path = f"epoch{epoch:03d}_{scene_id}_proc_spk{k}.wav"
        gromit.save_sample(
            s_hat_wav_cpu[b, k],
            model_cfg.input.sample_rate,
            split="train", 
            epoch=epoch,
            scene=scene_id,
            suffix=f"proc_spk{k}"
        )
```

## Training Loop Architecture

The training process in `train_script_multi.py` follows a structured approach:

### Epoch-Level Operations

```python
for epoch in range(train_cfg.epochs):
    # 1. Update sample selection for monitoring diversity
    trainsaves = update_epoch_samples(trainset, "train", epoch, debug)
    devsaves = update_epoch_samples(devset, "dev", epoch, debug)
    
    # 2. Training phase
    model.train()
    for batch_idx, batch in enumerate(trainset):
        # ... training batch processing
        
    # 3. Validation phase  
    model.eval()
    with torch.no_grad():
        for batch in devset:
            # ... validation batch processing
            
    # 4. Checkpointing and reporting
    if epoch % checkpoint_interval == 0:
        save_checkpoint(model, optimizer, epoch)
```

### Batch-Level Processing

```python
# Training batch processing
for batch_idx, batch in enumerate(trainset):
    # 1. Data preparation
    noisy = batch["noisy"].to(device)
    spk_all = batch["spkid_all"].to(device) 
    targ_all = batch["target_all"].to(device)
    
    # 2. Audio preprocessing  
    noisy = prep_audio(noisy, batch["fs"], input_channels, input_sr, input_rms, True)
    spk_all = prep_audio_speakers(spk_all, batch["fs"], input_sr, input_rms)
    
    # 3. STFT transformation
    noisy_tf = stft(noisy)
    spk_all_tf = stft(spk_all) 
    Y_ref_tf = stft(targ_all)
    
    # 4. Model forward pass
    optimizer.zero_grad()
    with autocast("cuda", dtype=torch.bfloat16):
        S_hat_c = model(noisy_tf, spk_all_for_model, spk_lens_all)
        loss, stats = joint_loss(S_hat_c, Y_ref_c, batch, stft)
    
    # 5. Backward pass and optimization
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    
    # 6. Metrics tracking and sample saving
    update_training_metrics(loss, stats)
    save_training_samples(S_hat_c, batch, epoch)
```

### Mixed Precision Training

```python
# Automatic Mixed Precision (AMP) for memory efficiency
scaler = GradScaler()

with autocast("cuda", dtype=torch.bfloat16):
    # Forward pass in reduced precision
    S_hat_c = model(noisy_tf, spk_all_for_model, spk_lens_all)
    loss, stats = joint_loss(S_hat_c, Y_ref_c, batch, stft)

# Backward pass with gradient scaling
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
```

## Validation Process

Validation follows similar processing but with key differences:

### Validation-Specific Features

```python
def validate(epoch, model, devset, ...):
    model.eval()
    
    # Reset epoch metrics
    gromit.val_loss.reset(epoch)
    gromit.val_stoi.reset(epoch)
    
    with torch.no_grad():
        for batch in devset:
            # Same preprocessing and forward pass as training
            S_hat_c = model(noisy_tf, spk_all_for_model, spk_lens_all)
            
            # Validation loss computation
            val_loss, val_stats = joint_loss(S_hat_c, Y_ref_c, batch, stft)
            
            # STOI computation for perceptual quality
            s_hat_wav = stft.inverse(S_hat_c, lengths=target_lens)
            for b in range(B):
                for k in range(K):
                    if target_lens[b, k] >= min_stoi_len:
                        stoi_score = stoi_fn(s_hat_wav[b,k,:L], y_wav[b,k,:L])
                        gromit.val_stoi.update(-stoi_score[0])
            
            # Sample saving (checkpoints only)
            if do_checkpoint:
                save_validation_samples(S_hat_c, batch, epoch)
```

### STOI Evaluation

Short-Time Objective Intelligibility (STOI) provides perceptual quality assessment:

```python
# STOI computation per speaker pair
min_stoi_len = int(math.ceil(7680 * model_cfg.input.sample_rate / 10000.0))  # ~0.768s

for b in range(B):
    for k in range(K):
        L = int(target_lens[b, k])
        if L >= min_stoi_len:
            proc = s_hat_wav[b, k, :L].unsqueeze(0)
            targ = y_wav[b, k, :L].unsqueeze(0) 
            stoi_score = stoi_fn(proc, targ)  # NegSTOILoss (negated)
            gromit.val_stoi.update(-stoi_score[0])  # Store positive STOI
```

## Optimization Strategies

### Memory Optimization

```python
# 1. Conservative batch size
batch_size = 1  # Safe for 80GB A100

# 2. Mixed precision training
autocast("cuda", dtype=torch.bfloat16)

# 3. Gradient checkpointing (if needed)
# torch.utils.checkpoint.checkpoint(model_layer, input)

# 4. Memory cleanup
optimizer.zero_grad(set_to_none=True)  # Release gradient memory
```

### Gradient Management

```python
# Gradient clipping for training stability  
torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)

# Gradient statistics monitoring
with torch.no_grad():
    grad_sq = sum(p.grad.detach().pow(2).sum().item() 
                  for p in model.parameters() if p.grad is not None)
    stats["grad_norm"] = max(grad_sq**0.5, 1e-8)
```

### Learning Rate Scheduling

```python
if do_lrschedule:
    # Step on validation loss (ReduceLROnPlateau)
    lr_scheduler.step(gromit.val_loss.get_average())
```

## Monitoring & Debugging

### Training Metrics

The system tracks comprehensive metrics for training analysis:

```python
# Loss components
stats = {
    "loss": float(loss.detach()),
    "L_sep": float(L_sep.detach()),                    # STFT loss  
    "sisdr_loss": float(sisdr_loss.detach()),         # SI-SDR loss
    "sisdr_db": float(balanced_sisdr.detach()),       # Actual SI-SDR in dB
    
    # Per-speaker metrics
    "sisdr_per_spk": [float(sisdr_per_spk[0, k]) for k in range(K)],
    
    # Training diagnostics
    "grad_norm": gradient_norm,
    "param_norm": parameter_norm, 
    "lr": current_learning_rate,
    
    # Memory usage
    "vram_alloc_MB": torch.cuda.memory_allocated() / 1024**2,
    "vram_reserved_MB": torch.cuda.memory_reserved() / 1024**2,
}
```

### Speaker Separation Analysis

```python
def analyze_speaker_separation(s_hat_wav, y_wav):
    """Enhanced diagnostics for speaker separation quality"""
    
    # Cross-speaker correlation (lower is better)
    cross_correlations = []
    for i in range(K):
        for j in range(i + 1, K):
            corr = pearson_correlation(s_hat_wav[:, i], s_hat_wav[:, j])
            cross_correlations.append(abs(corr))
    
    # Speaker distinctness (higher is better)
    pairwise_distances = []
    for i in range(K):
        for j in range(i + 1, K):
            dist = torch.norm(s_hat_wav[:, i] - s_hat_wav[:, j])
            pairwise_distances.append(dist)
    
    # Energy distribution balance
    speaker_energies = [(s_hat_wav[:, k] ** 2).mean() for k in range(K)]
    
    return {
        "cross_speaker_corr_mean": mean(cross_correlations),
        "speaker_l2_distance_mean": mean(pairwise_distances), 
        "speaker_energies": speaker_energies,
        "separation_quality_score": compute_separation_score(...)
    }
```

### Debug Logging

```python
# Enrollment audio verification
if spk_all.shape[1] > 1:
    diff_0_1 = torch.mean(torch.abs(spk_all[:, 0, :] - spk_all[:, 1, :])).item()
    logging.info(f"🎤 Enrollment difference (spk0 vs spk1): {diff_0_1:.6f}")
    if diff_0_1 < 1e-6:
        logging.warning("⚠️  Identical enrollment audio detected!")

# Shape monitoring  
logging.info(f"BEFORE prep_audio - noisy: {noisy.shape}, spk_all: {spk_all.shape}")
logging.info(f"AFTER prep_audio - noisy: {noisy.shape}, spk_all: {spk_all.shape}")
logging.info(f"STFT shapes - noisy_tf: {noisy_tf.shape}, spk_all_tf: {spk_all_tf.shape}")
```

### Sample Rotation Monitoring

```python
# Enhanced sample diversity tracking
logging.info(f"=== EPOCH {epoch} TRAINING SAMPLE SELECTION ===")
logging.info(f"Selected 6 samples from pool of 20")
logging.info(f"Sample IDs: {sample_ids}")

logging.info(f"=== EPOCH {epoch} HYBRID VALIDATION SAMPLE SELECTION ===") 
logging.info(f"Fixed samples (3): {fixed_samples}")
logging.info(f"Rotating samples (3): {rotating_samples}")
```

## Configuration Summary

### Current Training Configuration

```yaml
# Model architecture
n_srcs: 3                    # 3 speaker processing chains
n_layers: 3                  # 3 TFGridNet layers  
lstm_hidden_units: 128       # LSTM hidden size
emb_dim: 64                  # Speaker embedding dimension

# STFT parameters
n_fft: 128                   # 65 frequency bins
hop_length: 64               # 50% overlap
win_length: 128              # Window size

# Training parameters  
batch_size: 1                # Conservative memory usage
num_workers: 4               # I/O parallelism
learning_rate: 1e-3          # Adam optimizer
clip_grad_norm: 10.0         # Gradient clipping

# Loss configuration
sisdr_weight: 1.0            # Focus on SI-SDR quality
stft_weight: 0.0             # No direct STFT loss

# Sample monitoring
samples_per_epoch: 6         # Training diversity
validation_fixed: 3          # Consistent progress tracking
validation_rotating: 3       # Additional diversity
```

This training system provides robust multi-speaker target extraction with comprehensive monitoring, efficient memory usage, and high-quality separation through joint STFT and time-domain optimization.

---

The training process successfully transforms multi-channel noisy mixtures and speaker enrollments into high-quality separated spectrograms, with extensive monitoring and debugging capabilities for optimal model development.