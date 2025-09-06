#!/usr/bin/env python3
"""
Context-Aware Multi-Speaker Training Entry Point

This script trains the enhanced MultiSpeakerContextGridNet with:
- Multi-speaker context awareness
- Contrastive speaker learning
- Context-aware FiLM conditioning
- Separation difficulty adaptation

Usage:
    python run_train_context.py
    python run_train_context.py device=ha shared.exp_name=my-context-experiment
    python run_train_context.py --config-name multi_context_ha
"""

import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path
import sys
import os

# Add src and scripts to path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "scripts/train"))

# Import the context-aware training script
from train_multi_context import run

logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
)


@hydra.main(version_base=None, config_path="config/train", config_name="multi_context_ha")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for context-aware multi-speaker training.
    
    Args:
        cfg: Hydra configuration combining all components
    """
    logging.info("🌟 CONTEXT-AWARE MULTI-SPEAKER TRAINING STARTING")
    logging.info("="*60)
    
    # Log key configuration details
    logging.info("📊 Configuration Summary:")
    logging.info(f"   Model: {cfg.model.name}")
    logging.info(f"   Experiment: {cfg.shared.exp_name}")
    logging.info(f"   Device: {cfg.device}")
    logging.info(f"   Training epochs: {cfg.train.epochs}")
    logging.info(f"   Learning rate: {cfg.train.lr}")
    
    # Context-specific settings
    logging.info("🎭 Context Features:")
    logging.info(f"   Context layers: {cfg.model.context_layers}")
    logging.info(f"   Context heads: {cfg.model.context_heads}")
    logging.info(f"   Contrastive temp: {cfg.train.contrastive_temperature}")
    
    # Loss weights
    logging.info("⚖️  Loss Weights:")
    for loss_type, weight in cfg.train.loss_weights.items():
        logging.info(f"   {loss_type}: {weight}")
    
    logging.info("="*60)
    
    # Extract configuration components
    data_cfg = cfg.dataloading
    model_cfg = cfg.model
    train_cfg = cfg.train
    
    # Set up output directory
    output_dir = Path(cfg.train_dir) / cfg.shared.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"📁 Output directory: {output_dir}")
    
    # Extract optional wandb settings
    wandb_entity = getattr(cfg.wandb, 'entity', None)
    wandb_project = getattr(cfg.wandb, 'project', None)
    
    # Check for resumption
    resume_checkpoint = getattr(cfg.train, 'resume_from_checkpoint', None)
    if resume_checkpoint:
        logging.info(f"🔄 Will resume from checkpoint: {resume_checkpoint}")
    
    try:
        # Run the context-aware training
        run(
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            exp_dir=str(output_dir),
            debug=cfg.debug,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            resume_from_checkpoint=resume_checkpoint,
        )
        
        logging.info("✅ CONTEXT-AWARE TRAINING COMPLETED SUCCESSFULLY!")
        logging.info(f"📁 Results saved to: {output_dir}")
        
    except KeyboardInterrupt:
        logging.info("⚠️  Training interrupted by user")
        return
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()