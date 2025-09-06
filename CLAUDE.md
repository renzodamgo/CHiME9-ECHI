# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

CHiME-9 ECHI (Enhancing Conversation to address Hearing Impairment) baseline system for speech enhancement and speaker separation in multi-speaker conversations with cafeteria-like noise backgrounds.

## Environment Setup

Activate the conda environment and set PYTHONPATH:

```bash
conda activate echi_recipe
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

For persistent PYTHONPATH, add the export to `~/.bashrc` or `~/.zshrc`.

## Core Commands

### Training
```bash
# Train HA (hearing aid) model
python run_train.py --config-name main_ha device=ha shared.exp_name=ha-joint

# Train with joint loading for specific splits
python run_train.py --config-name main_ha device=ha dataloading.joint_for=[train]

# Resume from checkpoint
python run_train.py --config-name main_ha train.resume_from_checkpoint=path/to/checkpoint
```

### Enhancement
```bash
# Run enhancement pipeline
python run_enhancement.py device=ha

# Custom enhancement
python run_enhancement.py device=ha enhancement=joint_ha_uni
```

### Evaluation
```bash
# Evaluate enhanced audio
python run_evaluation.py evaluate.devices='[ha]' report.devices='[ha]' report.segment_types='[individual,summed]'

# Custom evaluation config
python run_evaluation.py evaluate.score_config=config/evaluation/metrics.yaml
```

## Architecture Overview

### Core Models
- **UniversalMCxTFGridNet** (`src/shared/UniversalMCxTFGridNet.py`) - Universal multi-channel TF-GridNet with speaker-conditional processing
- **CausalMCxTFGridNet** (`src/shared/CausalMCxTFGridNet.py`) - Causal version for real-time processing

### Enhancement Systems
- **joint_ha_uni** (`src/enhancement/joint_ha_uni.py`) - Universal GridNet enhancement for hearing aids
- **joint_ha_uni_multispk** - Multi-speaker variant with speaker embedding conditioning
- **baseline** (`src/enhancement/baseline.py`) - Baseline enhancement system
- **passthrough** (`src/enhancement/passthrough.py`) - No-op enhancement for testing

### Configuration Structure
- `config/train/` - Training configurations (main_ha.yaml, main_aria.yaml)
- `config/paths.yaml` - Data path definitions
- `config/shared.yaml` - Shared parameters across experiments

### Training Components  
- **Enhanced ECHI Joint** (`src/train/enhanced_echi_joint.py`) - Joint training with enhanced preprocessing
- **Multi-speaker training** (`src/train/joint_multi.py`) - Multi-speaker joint training
- **Losses** (`src/train/losses.py`) - Training loss functions

## Project Structure

- `src/shared/` - Core model implementations (UniversalMCxTFGridNet, signal utilities)
- `src/enhancement/` - Enhancement systems and registry
- `src/train/` - Training scripts and datasets  
- `src/evaluation/` - Evaluation and scoring utilities
- `config/` - Hydra configuration files
- `data/chime9_echi/` - Dataset location (default)
- `data/working_dir/experiments/` - Training outputs and checkpoints

## Analysis and Debugging

### Audio Analysis
```bash
# Analyze training/validation separation results
python analyze_results.py --data_dir data/working_dir/experiments/ha-joint-uni/train_ha/train_samples/ --output train_analysis_results.csv
python analyze_results.py --data_dir data/working_dir/experiments/ha-joint-uni/train_ha/val_samples/ --output val_analysis_results.csv

# Summary only (no CSV output)
python analyze_results.py --data_dir <path> --summary_only

# Single enhancement testing
python test_single_enhancement.py
python test_multi_speaker_enhancement.py
```

### Debugging Tools
```bash
# Speaker analysis and embeddings
python debug_speaker_analysis.py
python debug_speaker_embeddings.py

# Dataset validation
python test_dataset_diversity.py
python check_full_dataset_sessions.py

# Training diagnostics
python memory_analysis.py
python shape_analysis.py
```

## Key Dependencies

**Core ML/Audio:**
- torch, torchaudio - Deep learning framework
- soundfile, librosa - Audio I/O and processing  
- hydra-core, omegaconf - Configuration management
- versa-speech-audio-toolkit - Speech processing utilities

**Audio Quality Metrics:**
- pesq, pystoi - Perceptual quality metrics
- fast_bss_eval - Blind source separation evaluation
- pysepm - Speech enhancement performance measures

**Utilities:**
- wandb - Experiment tracking
- numpy, pandas - Data manipulation
- tqdm - Progress bars

See `environment.yaml` and `pyproject.toml` for complete dependencies.

## Hydra Configuration

The project uses Hydra for configuration management:

- **Main configs:** `config/train/main_ha.yaml`, `config/train/main_aria.yaml`  
- **Override from CLI:** `python run_train.py device=ha shared.exp_name=my-experiment`
- **Config composition:** Automatically combines paths, shared, model, dataloading configs
- **Output dirs:** Set via `train_dir` in config, defaults to `${paths.train_ha_dir}`

## Multi-Speaker Enhancement

The system supports both single and multi-speaker enhancement:

- **Speaker embeddings** - Conditional processing based on speaker identity
- **Joint training** - Train on single + multi-speaker data simultaneously  
- **Universal models** - Single model handles variable speaker counts
- **Real-time processing** - Causal models for low-latency applications

## Checkpoint Management

- **Automatic saving:** Checkpoints saved during training in `train_dir`
- **Resuming:** Use `train.resume_from_checkpoint=path/to/checkpoint.pt`
- **Best model selection:** Based on validation metrics during training
- **Inference loading:** Models loaded via `get_model()` utility in enhancement

## Recent Performance (ha-joint-uni experiment)

### Training Results
- 468 speaker separations across 156 sample groups (26 epochs)
- SI-SDR: -29.50 ± 13.98 dB, SNR improvement: 8.06 dB
- Best performance at epoch 48: -22.15 ± 9.29 dB

### Validation Results  
- SI-SDR: -40.00 ± 10.63 dB, SNR improvement: 11.09 dB
- Performance gap indicates some overfitting as expected