# Claude Code Configuration

This file contains configuration and helpful commands for Claude Code.

## Environment Setup

To activate the conda environment for this project:

```bash
conda activate echi_recipe
```

## Project Structure

- `src/shared/UniversalMCxTFGridNet.py` - Universal Multi-Speaker TFGridNet implementation
- `data/working_dir/experiments/ha-joint-uni/train_ha/` - Training experiment results
- `analyze_results.py` - Audio separation analysis script

## Analysis Scripts

Run audio separation analysis:

```bash
# Analyze training samples
python analyze_results.py --data_dir data/working_dir/experiments/ha-joint-uni/train_ha/train_samples/ --output train_analysis_results.csv

# Analyze validation samples  
python analyze_results.py --data_dir data/working_dir/experiments/ha-joint-uni/train_ha/val_samples/ --output val_analysis_results.csv

# Summary only (no CSV output)
python analyze_results.py --data_dir <path> --summary_only
```

## Dependencies

The project uses these key packages:
- torch, torchaudio
- soundfile, librosa
- numpy, pandas
- pesq, pystoi (for audio quality metrics)
- fast_bss_eval (for BSS evaluation)

## Results Summary

### Training Results (train_samples/)
- 468 speaker separations across 156 sample groups (26 epochs)
- SI-SDR: -29.50 ± 13.98 dB
- SNR improvement: 8.06 dB over noisy input
- Best performance at epoch 48: -22.15 ± 9.29 dB

### Validation Results (val_samples/)  
- 468 speaker separations across 156 sample groups (26 epochs)
- SI-SDR: -40.00 ± 10.63 dB
- SNR improvement: 11.09 dB over noisy input
- Performance more stable but lower than training (expected overfitting)