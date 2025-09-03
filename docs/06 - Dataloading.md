# 06 - Data Loading System

This document provides a comprehensive overview of the CHiME9-ECHI data loading system, including dataset organization, multi-speaker handling, and batch processing optimizations.

## Table of Contents

1. [Overview](#overview)
2. [Dataset Architecture](#dataset-architecture)
3. [ECHIJoint Dataset Class](#echijoint-dataset-class)
4. [Data Loading Configuration](#data-loading-configuration)
5. [Batch Processing & Collation](#batch-processing--collation)
6. [Multi-Speaker Handling](#multi-speaker-handling)
7. [Sample Rotation System](#sample-rotation-system)
8. [Performance Optimizations](#performance-optimizations)
9. [Troubleshooting](#troubleshooting)

## Overview

The CHiME9-ECHI data loading system is designed for multi-speaker target speaker extraction training. It handles:

- **Multi-speaker segments** with variable number of active speakers (2-3 speakers typical)
- **Speaker enrollment** via rainbow passage recordings
- **Dynamic sample selection** with rotation across epochs
- **Efficient batch processing** with padding and cropping strategies
- **Memory-optimized loading** for large-scale training

## Dataset Architecture

### Data Organization

The dataset follows a hierarchical structure:

```
data/chime9_echi/
├── ha/                          # Device recordings (hearing aids)
│   ├── train/                   # Training segments
│   └── dev/                     # Development segments
├── ref/                         # Reference clean recordings
│   ├── train/
│   └── dev/
├── participant/                 # Rainbow passage enrollments
│   ├── train/
│   └── dev/
└── metadata/                    # Session and segment information
    ├── sessions.train.csv
    ├── sessions.dev.csv
    └── ref/                     # Per-speaker segment metadata
```

### Signal Types

Three types of audio signals are used:

1. **Noisy Signal** (`noisy_signal`): Multi-channel mixture recordings
   - Path: `data/working_dir/train_segments/{dataset}/{device}/{session}.{device}.{pid}.{segment}.wav`
   - Channels: 4 (for hearing aids)
   - Sample rate: 48kHz → 16kHz (resampled)

2. **Target Signal** (`ref_signal`): Clean reference for active speakers
   - Path: `data/working_dir/train_segments/{dataset}/{device}_ref/{session}.{device}.{pid}.{segment}.wav`
   - Channels: 1
   - Sample rate: 16kHz

3. **Speaker Enrollment** (`rainbow_signal`): Rainbow passage recordings
   - Path: `data/working_dir/participant/{dataset}/{pid}.wav`
   - Channels: 1
   - Sample rate: 48kHz → 16kHz (resampled)
   - Duration: Full rainbow passage (not segmented)

## ECHIJoint Dataset Class

The `ECHIJoint` class in `src/train/echi.py` handles multi-speaker training data.

### Key Features

- **Speaker-centric organization**: Groups segments by active speakers
- **Union-based segment selection**: Uses all available segments (not intersection)
- **Active speaker tracking**: Maintains masks for which speakers are active
- **Quality filtering**: Requires minimum 2 active speakers and 1-second duration

### Manifest Creation Process

```python
def make_manifest(self):
    """Creates training manifest with multi-speaker segments"""
    
    # 1. Process each session
    for meta in self.metadata:
        device_pos = int(meta[f"{self.audio_device}_pos"])
        pids = [meta[f"pos{i}"] for i in range(1, 5) if i != device_pos]
        
        # 2. Load segment information for each speaker
        seg_lists = []
        for pid in pids:
            segments = load_segments_csv(pid)
            seg_lists.append(segments)
        
        # 3. Use UNION of all segments (not intersection)
        all_segments = union_of_segments(seg_lists)
        
        # 4. For each segment, determine active speakers
        for segment_idx in all_segments:
            active_speakers = []
            for i, pid in enumerate(pids):
                if segment_exists(pid, segment_idx):
                    active_speakers.append(i)
            
            # 5. Quality filtering
            if len(active_speakers) >= 2:  # Minimum speakers
                manifest_entry = create_entry(segment_idx, pids, active_speakers)
                self.manifest.append(manifest_entry)
```

### Data Structure

Each manifest entry contains:

```python
entry = {
    "id": "train_07_ha_seg042",           # Unique identifier
    "session": "train_07",                # Session name
    "device": "ha",                       # Device type
    "idx": 42,                           # Segment index
    "pids": ["P001", "P002", "P003"],     # All speaker PIDs
    "noisy": "path/to/noisy.wav",         # Noisy mixture path
    "target_all": [                       # Per-speaker targets
        "path/to/target_P001.wav",       # Active speaker
        None,                             # Inactive speaker (silent)
        "path/to/target_P003.wav"        # Active speaker
    ],
    "spkid_all": [                        # Rainbow enrollments
        "path/to/P001_rainbow.wav",
        "path/to/P002_rainbow.wav", 
        "path/to/P003_rainbow.wav"
    ],
    "speaker_active_mask": [True, False, True]  # Activity mask
}
```

## Data Loading Configuration

Configuration is managed in `config/train/dataloading.yaml`:

```yaml
# Device and signal configuration
device: ha                              # Device type (hearing aids)
noisy_signal: ${..paths.train_input_file}
ref_signal: ${..paths.train_target_file}
rainbow_signal: ${..paths.train_rainbow_file}

# Metadata files
sessions_file: ${..paths.sessions_file}
segments_file: ${..paths.segment_info_file}

# Multi-speaker training
joint_for: ["train", "dev"]              # Use ECHIJoint for these splits

# Sample rates (before preprocessing)
signal:
  noisy_sr: 48000
  ref_sr: 16000
  spkid_sr: 48000

# DataLoader settings
loader:
  train:
    batch_size: 1                        # Conservative for memory
    num_workers: 4                       # Parallel loading
    shuffle: true                        # Randomize order
    pin_memory: true                     # GPU transfer optimization
  dev:
    batch_size: 1
    num_workers: 4
    shuffle: false                       # Deterministic validation
    pin_memory: true
```

### Memory Optimization Guidelines

The configuration includes VRAM optimization notes:

- **Current settings**: Conservative for 80GB A100
- **Higher memory**: `batch_size: 4-16`
- **Lower memory**: `batch_size: 1`, reduce `num_workers`

## Batch Processing & Collation

The `collate_fn_joint` function in `src/train/echi.py` handles batch creation:

### Processing Pipeline

```python
def collate_fn_joint(batch: list[dict]):
    """
    Converts individual samples into batched tensors
    
    Input: List of samples from ECHIJoint.__getitem__()
    Output: Batched tensors ready for model training
    """
    
    # 1. Extract batch metadata
    ids = [x["id"] for x in batch]
    fs = batch[0]["fs"]
    
    # 2. Random cropping (4-second maximum)
    MAX_TRAIN_SECS = 4.0
    max_samples = int(MAX_TRAIN_SECS * fs)
    
    for x in batch:
        if x["target_all"].size(1) > max_samples:
            start = random.randint(0, Tw - max_samples)
            x["noisy"] = x["noisy"][..., start:start+max_samples]
            x["target_all"] = x["target_all"][..., start:start+max_samples]
    
    # 3. Padding and tensor creation
    # Noisy: [B, C, Tw] with length tracking
    noisy_padded, noisy_lens = combine_audio_list([x["noisy"] for x in batch])
    
    # Targets: [B, K, Tw] with per-speaker lengths
    K = batch[0]["target_all"].size(0)  # Number of speakers
    target_all = pad_to_batch(batch, "target_all")
    target_lens_all = get_lengths(batch, "target_all")
    
    # Speaker enrollments: [B, K, Tr] with variable lengths
    spkid_all = pad_to_batch(batch, "spkid_all") 
    spkid_lens_all = get_original_lengths(batch, "spkid_lens_all")
    
    # Activity masks: [B, K] boolean
    speaker_active_mask = torch.stack([x["speaker_active_mask"] for x in batch])
    
    return {
        "id": ids,
        "fs": fs,
        "noisy": noisy_padded,              # [B, C, Tw]
        "noisy_lens": noisy_lens,           # [B]
        "target_all": target_all,           # [B, K, Tw]
        "target_lens_all": target_lens_all, # [B, K]
        "spkid_all": spkid_all,             # [B, K, Tr]
        "spkid_lens_all": spkid_lens_all,   # [B, K]
        "speaker_active_mask": speaker_active_mask  # [B, K]
    }
```

### Key Processing Steps

1. **Random Cropping**: Limits training segments to 4 seconds
2. **Consistent Cropping**: Same crop applied to noisy and all targets
3. **Padding**: Handles variable-length sequences within batches
4. **Length Tracking**: Preserves original lengths for loss computation

## Multi-Speaker Handling

### Speaker Activity Management

The system handles variable numbers of active speakers:

```python
# Example: 3-speaker setup with 2 active
speaker_active_mask = [True, False, True]  # Speakers 0,2 active; 1 silent
target_all = [
    target_speaker_0,    # Real target audio
    zeros_tensor,        # Silent speaker (zero target)
    target_speaker_2     # Real target audio
]
```

### Silent Speaker Processing

Silent speakers are handled intelligently:

- **Target**: Zero tensor as placeholder
- **Enrollment**: Full rainbow passage still provided
- **Loss masking**: Silent speakers excluded from SI-SDR computation
- **Learning**: Model learns both positive (separate) and negative (ignore) patterns

### Processing Chain Assignment

With `n_srcs=3` configuration:

```python
# Model processing chain assignment
chain_idx = min(speaker_idx, n_srcs - 1)

# Current mapping:
# Speaker 0 → Chain 0 (dedicated)
# Speaker 1 → Chain 1 (dedicated) 
# Speaker 2 → Chain 2 (dedicated)
# Speaker 3+ → Chain 2 (shared) - potential bottleneck
```

## Sample Rotation System

The enhanced sample rotation system provides diversity while maintaining consistency.

### Training Sample Rotation

**Full rotation** for maximum diversity:

```python
def update_epoch_samples(dataset, "train", epoch):
    # Configuration
    pool_size = min(20, data_len // 50)      # ~1% of dataset
    samples_per_epoch = 6                    # 6 samples per epoch
    
    # Epoch-based rotation
    random.seed(42 + epoch)
    sample_indices = random.sample(range(data_len), pool_size)
    selected_indices = rotate_through_pool(epoch, samples_per_epoch, pool_size)
    
    return [dataset[i]["id"] for i in selected_indices]

# Results over 100 epochs:
# - Training: 571 unique samples (11.55% coverage)
# - High diversity for robust training
```

### Validation Sample Rotation

**Hybrid approach** for stable progress tracking:

```python
def update_epoch_samples(dataset, "dev", epoch):
    # Fixed samples (consistent progress tracking)
    fixed_indices = [data_len//6, data_len//2, data_len*5//6]  # 3 fixed
    fixed_samples = [dataset[i]["id"] for i in fixed_indices]
    
    # Rotating samples (diversity assessment) 
    rotating_pool = 15  # Smaller pool for validation
    rotating_samples = select_rotating_samples(epoch, 3)
    
    return fixed_samples + rotating_samples

# Results over 100 epochs:
# - Validation: ~18 unique samples (0.86% coverage)  
# - 3 fixed + 3 rotating per epoch
# - Balanced: consistency + diversity
```

### Benefits

- **Training**: Maximum diversity prevents overfitting
- **Validation**: Smooth curves for progress monitoring
- **Reproducible**: Epoch-based seeding ensures consistent results
- **Efficient**: No dataset recreation, just sample ID updates

## Performance Optimizations

### Memory Management

1. **Conservative Batch Sizes**: `batch_size=1` for safety
2. **Efficient Padding**: Only pad to batch maximum, not global maximum
3. **Random Cropping**: 4-second limit reduces memory usage
4. **Pin Memory**: Faster GPU transfers with `pin_memory=True`

### Loading Optimizations

1. **Parallel Workers**: `num_workers=4` for I/O parallelism
2. **Tensor Reuse**: Efficient padding operations
3. **Length Caching**: Avoid repeated length computations
4. **Lazy Loading**: Audio loaded only when needed

### Data Preprocessing

```python
def prep_audio(audio, sample_rate, target_channels, target_sr, target_rms):
    """Efficient audio preprocessing pipeline"""
    
    # 1. Channel handling
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)  # Add channel dimension
    
    # 2. Resampling (48kHz → 16kHz)
    if sample_rate != target_sr:
        audio = torchaudio.transforms.Resample(sample_rate, target_sr)(audio)
    
    # 3. Channel selection/expansion
    if audio.shape[0] != target_channels:
        audio = handle_channel_mismatch(audio, target_channels)
    
    # 4. RMS normalization
    audio = rms_normalize(audio, target_rms)
    
    return audio
```

## Troubleshooting

### Common Issues

1. **Dataset Size Mismatch**
   ```
   ERROR: Dataset length: 0
   ```
   - Check file paths in `config/paths.yaml`
   - Verify segment CSV files exist
   - Ensure minimum 2 active speakers per segment

2. **Memory Issues**
   ```
   CUDA out of memory
   ```
   - Reduce `batch_size` to 1
   - Decrease `num_workers`
   - Check for memory leaks in custom transforms

3. **Inconsistent Speaker Counts**
   ```
   AssertionError: Inconsistent K across batch
   ```
   - All samples in batch must have same number of speakers
   - Check manifest creation logic
   - Verify speaker ID consistency

4. **Audio Loading Failures**
   ```
   FileNotFoundError: Audio file not found
   ```
   - Check path templates in `paths.yaml`
   - Verify preprocessing completed successfully
   - Check file permissions

### Performance Monitoring

Monitor these metrics during training:

```python
# Log in training loop
logging.info(f"Dataset length: {len(dataset)}")
logging.info(f"Batch processing time: {batch_time:.2f}s")
logging.info(f"Memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")

# Sample selection logs
logging.info(f"Epoch {epoch}: Selected samples {sample_ids}")
logging.info(f"Fixed validation samples: {fixed_samples}")
logging.info(f"Rotating validation samples: {rotating_samples}")
```

### Debug Mode

Enable debug mode for detailed analysis:

```python
# In dataset initialization
debug = True  # Limits dataset to 50 samples
ECHIJoint(split, device, ..., debug=debug)

# Results in:
# - Faster iteration for testing
# - Detailed logging of manifest creation
# - Reduced memory usage
```

## Configuration Examples

### High-Memory Training (80GB+ GPU)

```yaml
loader:
  train:
    batch_size: 8
    num_workers: 8
  dev:
    batch_size: 4
    num_workers: 4
```

### Low-Memory Training (<24GB GPU)

```yaml
loader:
  train:
    batch_size: 1
    num_workers: 2
  dev:
    batch_size: 1
    num_workers: 2
```

### Fast Development Iteration

```yaml
# In main config
debug: true

# Results in:
# - 50 training samples max
# - 3 validation samples per epoch
# - Faster epoch completion
```

---

This data loading system provides a robust foundation for multi-speaker target speaker extraction training, with optimizations for memory usage, training diversity, and validation consistency.