#!/usr/bin/env python3
"""
Utility script to resume Universal GridNet training from a checkpoint.

Usage:
    python resume_training.py --checkpoint /path/to/checkpoint.pt
    python resume_training.py --experiment ha-joint-uni --epoch 48
    python resume_training.py --latest-checkpoint /path/to/experiment/dir/
"""

import argparse
import logging
from pathlib import Path
import torch
from omegaconf import OmegaConf
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "scripts/train"))

from train_script_multi import run
from shared.core_utils import get_device

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the latest checkpoint in a directory."""
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Sort by modification time, return latest
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    return latest_checkpoint


def find_experiment_checkpoint(exp_name: str, epoch: int) -> Path:
    """Find a specific checkpoint for an experiment."""
    exp_dir = Path("data/working_dir/experiments") / exp_name / "train_ha"
    checkpoint_dir = exp_dir / "checkpoints"
    
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    checkpoint_file = checkpoint_dir / f"{exp_name}_{str(epoch).zfill(3)}.pt"
    if not checkpoint_file.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_file}")
    
    return checkpoint_file


def load_experiment_config(checkpoint_path: Path) -> tuple:
    """Load the original training configuration for a checkpoint."""
    # Try to find the config from the experiment directory
    checkpoint_dir = checkpoint_path.parent
    exp_dir = checkpoint_dir.parent
    
    # Look for hydra config
    hydra_config = exp_dir / "hydra" / ".hydra" / "config.yaml"
    if hydra_config.exists():
        cfg = OmegaConf.load(hydra_config)
        return cfg.dataloading, cfg.model, cfg.train, str(exp_dir), cfg.debug, cfg.wandb.entity, cfg.wandb.project
    
    # Fallback - use standard ha-joint-uni config
    logging.warning(f"Config not found at {hydra_config}, using default ha-joint-uni config")
    config_path = Path("config/train/main_ha.yaml")
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        return cfg.dataloading, cfg.model, cfg.train, str(exp_dir), False, None, None
    
    raise ValueError("Could not find training configuration")


def main():
    parser = argparse.ArgumentParser(description="Resume Universal GridNet training from checkpoint")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Path to specific checkpoint file")
    group.add_argument("--experiment", type=str, help="Experiment name (requires --epoch)")
    group.add_argument("--latest-checkpoint", type=str, help="Path to experiment directory (uses latest checkpoint)")
    
    parser.add_argument("--epoch", type=int, help="Specific epoch number (used with --experiment)")
    parser.add_argument("--config", type=str, help="Override config file path")
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logging.error(f"Checkpoint file not found: {checkpoint_path}")
            return 1
            
    elif args.experiment:
        if args.epoch is None:
            logging.error("--epoch is required when using --experiment")
            return 1
        try:
            checkpoint_path = find_experiment_checkpoint(args.experiment, args.epoch)
        except ValueError as e:
            logging.error(f"Error finding experiment checkpoint: {e}")
            return 1
            
    elif args.latest_checkpoint:
        checkpoint_dir = Path(args.latest_checkpoint)
        if checkpoint_dir.is_dir():
            # Look for checkpoints subdirectory
            if (checkpoint_dir / "checkpoints").exists():
                checkpoint_dir = checkpoint_dir / "checkpoints"
        
        try:
            checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        except ValueError as e:
            logging.error(f"Error finding latest checkpoint: {e}")
            return 1
    
    logging.info(f"🎯 Resuming from checkpoint: {checkpoint_path}")
    
    # Verify checkpoint can be loaded
    try:
        checkpoint = torch.load(checkpoint_path, map_location=get_device())
        if 'epoch' in checkpoint:
            logging.info(f"📊 Checkpoint contains epoch {checkpoint['epoch']}")
            logging.info(f"📦 Checkpoint keys: {list(checkpoint.keys())}")
        else:
            logging.info("📦 Legacy checkpoint format (model state only)")
    except Exception as e:
        logging.error(f"❌ Failed to load checkpoint: {e}")
        return 1
    
    # Load configuration
    try:
        if args.config:
            cfg = OmegaConf.load(args.config)
            data_cfg = cfg.dataloading
            model_cfg = cfg.model
            train_cfg = cfg.train
            exp_dir = cfg.train_dir
            debug = cfg.get('debug', False)
            wandb_entity = cfg.wandb.get('entity', None)
            wandb_project = cfg.wandb.get('project', None)
        else:
            data_cfg, model_cfg, train_cfg, exp_dir, debug, wandb_entity, wandb_project = load_experiment_config(checkpoint_path)
    except Exception as e:
        logging.error(f"❌ Failed to load configuration: {e}")
        return 1
    
    logging.info("🚀 Starting resumed training...")
    logging.info(f"   Experiment dir: {exp_dir}")
    logging.info(f"   Model: {model_cfg.name}")
    logging.info(f"   Training epochs: {train_cfg.epochs}")
    
    # Start training
    try:
        run(
            data_cfg,
            model_cfg, 
            train_cfg,
            exp_dir,
            debug,
            wandb_entity,
            wandb_project,
            str(checkpoint_path)
        )
        logging.info("✅ Resumed training completed successfully!")
        return 0
        
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())