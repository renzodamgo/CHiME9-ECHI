# 09 - Dataset Analysis

## Overview

The CHiME9-ECHI dataset is designed for enhancing conversational speech in noisy environments, specifically targeting hearing aid applications. It contains multi-speaker conversation recordings captured in a simulated cafeteria environment with four participants seated around a table.

## Dataset Structure

### Main Directory Organization

```
/home/damian/CHiME9-ECHI/data/chime9_echi/
├── aria/           # Project Aria glasses audio (7-channel, 48kHz)
│   ├── train/      # 30 training sessions
│   └── dev/        # 10 development sessions
├── ha/             # Hearing aid audio (4-channel, 48kHz)
│   ├── train/      # 30 training sessions  
│   └── dev/        # 10 development sessions
├── ct/             # Close-talk microphones (1 channel per participant)
├── ref/            # Reference signals for evaluation
├── tracker/        # Head tracking data (CSV files, 250Hz)
├── participant/    # Rainbow Passage recordings for speaker ID
│   ├── train/      # 118 participant recordings
│   └── dev/        # 40 participant recordings
└── metadata/       # Session and segmentation metadata
    ├── ref/        # Speech segment annotations
    │   ├── train/  # Per-session CSV files with time segments
    │   └── dev/
    └── sessions.{train|dev}.csv  # Session participant mapping
```

### Working Directory Structure

```
/home/damian/CHiME9-ECHI/data/working_dir/train_segments/
├── train/
│   └── ha/         # 17,517 segmented training files (4-channel, 16kHz)
└── dev/
    └── ha/         # 6,215 segmented development files (4-channel, 16kHz)
```

## Audio File Specifications

### Format and Technical Details

| Audio Type | Channels | Sample Rate | Format | File Size |
|------------|----------|-------------|--------|-----------|
| Hearing Aid | 4 | 48kHz | WAV | ~802MB per session |
| Aria Glasses | 7 | 48kHz | WAV | Similar to HA |
| Rainbow Passages | 1 | 48kHz | 16-bit PCM WAV | ~2.5-2.9MB each |
| Training Segments | 4 | 16kHz | WAV | ~192KB per segment |

### Channel Configuration

**Hearing Aid Audio (4-channel):**
- Channel 0: Left ear, front microphone
- Channel 1: Left ear, rear microphone  
- Channel 2: Right ear, front microphone
- Channel 3: Right ear, rear microphone

**Aria Glasses Audio (7-channel):**
- 7-microphone array with known geometry
- Spatial audio capture for directional processing

## Dataset Statistics

### Training Set
- **Sessions:** 30 (train_01 to train_30)
- **Speakers:** 118 unique participants
- **Duration:** 18 hours total
- **Segments:** 17,517 processed audio segments
- **Rainbow Passages:** 118 speaker identification recordings
- **Background:** LibriSpeech audio only

### Development Set
- **Sessions:** 10 (dev_02 to dev_12)
- **Speakers:** 40 unique participants
- **Duration:** 6 hours total
- **Segments:** 6,215 processed audio segments
- **Rainbow Passages:** 40 speaker identification recordings
- **Background:** Mix of LibriSpeech and EARS spontaneous speech

### Storage Requirements
- **Development set:** ~23.6GB compressed (`chime9_echi.dev.v1_0.tar.gz`)
- **Training set part 1:** ~30GB compressed (`chime9_echi.train_pt1.v1_0.tar.gz`)
- **Training set part 2:** ~37GB compressed (`chime9_echi.train_pt2.v1_0.tar.gz`)
- **Total uncompressed:** ~150GB+ estimated

## Speaker Information

### Participant Data
- Each participant has unique ID format: P001, P002, etc.
- Rainbow Passage recordings stored in `participant/{dataset}/{pid}.wav`
- Used for speaker embedding extraction and identification
- Consistent voice characteristics across sessions

### Session Mapping
Sessions mapped in `sessions.{dataset}.csv` with columns:
- `session`: Session identifier (train_01, dev_02, etc.)
- `aria_pos`, `ha_pos`: Position numbers for device wearers
- `pos1-pos4`: Participant IDs for each table position (P001, P002, etc.)

Example session mapping:
```csv
session,aria_pos,ha_pos,pos1,pos2,pos3,pos4
train_01,1,2,P001,P002,P003,P004
dev_02,3,1,P119,P120,P121,P122
```

## Speech Segmentation

### Reference Segments
Speech segments defined in `metadata/ref/{dataset}/{session}.{device}.{pid}.csv`:
- **Format:** `index,start_sample,end_sample`
- **Example:** `1,335447,359959` (segment 1: samples 335447-359959 at 48kHz)
- **Density:** ~280 segments per participant per session
- **Coverage:** Variable speech activity across participants

### Segment Processing Pipeline
1. **Source:** Full session recordings at 48kHz
2. **Segmentation:** Extract speech segments using reference timestamps
3. **Filtering:** Remove segments shorter than 1 second
4. **Downsampling:** Convert from 48kHz to 16kHz for training efficiency
5. **Cropping:** Random 4-second clips during training
6. **Output:** Both noisy input and clean reference versions

## Recording Environment

### Simulated Cafeteria Setup
- **Configuration:** 4 participants seated around a table
- **Speakers:** 18 loudspeakers positioned around recording room
  - 14 speakers simulate up to 7 simultaneous conversations
  - 4 ambient speakers for background noise from WHAM! dataset
- **Sound Events:** Additional cafeteria sounds from FSD50K
- **Variability:** Speaker repositioning and room configuration changes

### Device Positioning
- **Hearing Aids:** Worn by participants at specified table positions
- **Aria Glasses:** Worn by participants at specified positions
- **Close-talk Mics:** Individual microphones per participant
- **Head Tracking:** 6DOF position and orientation data at 250Hz

## Dataset Loading and Processing

### ECHIJoint Dataset Class
The `ECHIJoint` dataset class provides:

```python
# Key features of dataset loading
- Union-based loading for maximum diversity
- Multi-speaker training with K=3 speakers per sample
- Speaker activity masking for silent participants  
- Random cropping to 4-second segments during training
- Speaker embedding integration via Rainbow Passages
- Active speaker detection and handling
```

### Data Augmentation
- **Random Cropping:** 4-second maximum segments from longer recordings
- **Speaker Selection:** Random K=3 speakers from available participants
- **Activity Masking:** Handle silent speakers with appropriate targets
- **Sample Rotation:** Epoch-based rotation for training diversity

## Multi-Speaker Characteristics

### Speaker Activity Patterns
- **Simultaneous Speech:** Multiple speakers active in same segment
- **Silent Periods:** Speakers may be inactive during segments
- **Variable Participation:** Not all 4 participants speak in every segment
- **Model Handling:** Active speaker masking in loss computation

### Target Generation
```python
# Per-speaker target extraction
target_all: [B, K, Tw]  # Clean speech per enrolled speaker
target_lens_all: [B, K]  # Valid lengths per speaker
speaker_active_mask: [B, K]  # Boolean activity mask
```

## Quality Control and Validation

### Data Integrity Checks
- **File Completeness:** All expected audio files present
- **Format Validation:** Consistent sampling rates and channel counts
- **Segment Alignment:** Proper timestamp alignment across devices
- **Clock Drift:** Compensation between Aria and other recording systems

### Missing Data Handling
- **Training Sessions:** Some Aria data missing (train_16, train_28, train_29)
- **Graceful Degradation:** Dataset loading continues with available modalities
- **Error Logging:** Missing files logged but training proceeds

## Usage Constraints and Licensing

### Data Use Agreement
- **Scope:** Academic and research use only under CHiME-9 ECHI agreement
- **Redistribution:** Raw data redistribution prohibited
- **Citation:** Required for any publications using the dataset
- **Commercial Use:** Requires separate licensing agreements

### Access Requirements
- **Registration:** CHiME-9 challenge registration required
- **Download:** Separate download links for train/dev splits
- **Verification:** Dataset integrity checks recommended after download

## Integration with Training Pipeline

### Sample Selection Strategy
- **Training:** Rotating sample selection achieving 11.55% coverage over 100 epochs
- **Validation:** Hybrid approach with 3 fixed + 3 rotating samples
- **Reproducibility:** Epoch-based seeding for consistent sample rotation

### Performance Metrics
- **Coverage:** Training samples rotated across epochs for maximum diversity
- **Stability:** Validation uses consistent samples for progress tracking
- **Scalability:** Efficient loading for large-scale multi-speaker training

## Summary

The CHiME9-ECHI dataset provides a comprehensive multi-modal resource for hearing aid speech enhancement research. With 40 sessions, 158 unique speakers, and 24 hours of conversational audio in realistic noisy environments, it supports robust development of multi-speaker separation and enhancement algorithms. The careful organization of segmented data, speaker embeddings, and metadata enables sophisticated training strategies while maintaining computational efficiency.