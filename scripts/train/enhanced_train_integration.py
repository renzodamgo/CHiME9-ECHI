"""
Enhanced training script integration for active speaker masking.

This shows how to integrate the enhanced ECHIJoint dataset with active speaker detection
into your existing training pipeline.
"""

import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import logging
from torch_stoi import NegSTOILoss
from torch.utils.data.dataloader import DataLoader

# Enhanced imports
from train.enhanced_echi_joint import EnhancedECHIJoint, collate_fn_joint_enhanced
from train.echi import ECHIJoint, collate_fn_joint, ECHI, collate_fn
from train.joint_multi import joint_loss
from shared.core_utils import get_model, get_device
from train.losses import get_loss, get_lrmethod
from shared.signal_utils import STFTWrapper, match_length, prep_audio
from torch.amp import autocast


def get_enhanced_dataset(split: str, data_cfg: DictConfig, debug: bool):
    """
    Enhanced dataset factory with energy-based active speaker detection.
    
    New config options:
    - data_cfg.use_enhanced_detection: Enable energy-based validation
    - data_cfg.energy_threshold_db: dB threshold for speech activity  
    - data_cfg.min_speech_duration: Minimum seconds of speech for active classification
    """
    # Decide which dataset to use
    joint_for = set(getattr(data_cfg, "joint_for", []))  # e.g., ["train"]
    use_joint = split in joint_for
    use_enhanced = getattr(data_cfg, "use_enhanced_detection", False)

    logging.info(f"=== ENHANCED DATASET SETUP ({split}) ===")
    logging.info(f"joint_for: {joint_for}")
    logging.info(f"use_joint: {use_joint}")
    logging.info(f"use_enhanced_detection: {use_enhanced}")

    if use_joint:
        if use_enhanced:
            logging.info(f"Creating EnhancedECHIJoint dataset for {split}")
            data = EnhancedECHIJoint(
                split,
                data_cfg.device,
                data_cfg.noisy_signal,
                data_cfg.ref_signal,
                data_cfg.rainbow_signal,
                data_cfg.sessions_file,
                data_cfg.segments_file,
                debug,
                energy_threshold_db=getattr(data_cfg, "energy_threshold_db", -40),
                min_speech_duration=getattr(data_cfg, "min_speech_duration", 0.1),
                validate_energy=True
            )
            chosen_collate = collate_fn_joint_enhanced
        else:
            logging.info(f"Creating standard ECHIJoint dataset for {split}")
            data = ECHIJoint(
                split,
                data_cfg.device,
                data_cfg.noisy_signal,
                data_cfg.ref_signal,
                data_cfg.rainbow_signal,
                data_cfg.sessions_file,
                data_cfg.segments_file,
                debug,
            )
            chosen_collate = collate_fn_joint
    else:
        logging.info(f"Creating standard ECHI dataset for {split}")
        data = ECHI(
            split,
            data_cfg.device,
            data_cfg.noisy_signal,
            data_cfg.ref_signal,
            data_cfg.rainbow_signal,
            data_cfg.sessions_file,
            data_cfg.segments_file,
            debug,
        )
        chosen_collate = collate_fn

    data_len = len(data)
    logging.info(f"Dataset length: {data_len}")
    
    # Log enhanced dataset statistics
    if use_joint and use_enhanced and hasattr(data, 'get_activity_statistics'):
        # Let dataset collect some initial statistics
        for i in range(min(10, len(data))):
            _ = data[i]  # Process a few samples to gather stats
        
        stats = data.get_activity_statistics() 
        logging.info(f"Initial activity stats (first 10 samples):")
        logging.info(f"  Avg active speakers per sample: {stats.get('avg_energy_active_per_sample', 0):.2f}")
        logging.info(f"  Correction rate: {stats.get('correction_rate', 0)*100:.1f}%")

    loader = DataLoader(
        data,
        **data_cfg.loader[split],
        collate_fn=chosen_collate,
    )

    logging.info(f"Loader config: {data_cfg.loader[split]}")
    logging.info(f"Collate function: {chosen_collate.__name__}")

    return loader, data


def enhanced_training_step(model, batch, stft, loss_weights, device):
    """
    Enhanced training step with active speaker masking.
    
    Key improvements:
    1. Uses speaker_active_mask from batch
    2. Logs activity statistics 
    3. Optional confidence-weighted loss
    """
    # Move batch to device
    for key in ["noisy", "target_all", "spkid_all", "speaker_active_mask"]:
        if key in batch:
            batch[key] = batch[key].to(device)
    
    # Optional: move confidence scores
    if "speaker_confidence_scores" in batch:
        batch["speaker_confidence_scores"] = batch["speaker_confidence_scores"].to(device)

    # Forward pass through model
    noisy = batch["noisy"]  # [B, C, T]
    spkid_all = batch["spkid_all"]  # [B, K, Tr]  
    spkid_lens = batch["spkid_lens_all"]  # [B, K]

    # Convert to STFT domain
    noisy_stft = stft(noisy, lengths=batch["noisy_lens"])  # [B, C, T', F, 2]
    spkid_stft = stft(spkid_all.view(-1, spkid_all.shape[-1]), 
                     lengths=spkid_lens.view(-1))  # [BK, 1, Tr', F, 2]
    
    # Reshape speaker enrollments: [BK, 1, Tr', F, 2] -> [B, K, Tr', F, 2]
    B, K = spkid_all.shape[:2]
    spkid_stft = spkid_stft.view(B, K, *spkid_stft.shape[1:])

    # Model forward pass  
    separated_stft = model(noisy_stft, spkid_stft, spkid_lens)  # [B, K, T', F] complex

    # Target STFT
    target_all = batch["target_all"]  # [B, K, T]
    target_stft = stft(target_all.view(-1, target_all.shape[-1]),
                      lengths=batch["target_lens_all"].view(-1))  # [BK, 1, T', F, 2]
    target_stft = target_stft.view(B, K, *target_stft.shape[1:]).squeeze(2)  # [B, K, T', F, 2]
    target_stft_complex = torch.complex(target_stft[..., 0], target_stft[..., 1])  # [B, K, T', F]

    # Enhanced joint loss with active speaker masking
    loss, stats = joint_loss(
        separated_stft, 
        target_stft_complex,
        batch,  # Contains speaker_active_mask
        stft,
        weights=loss_weights
    )
    
    # Enhanced logging
    active_mask = batch["speaker_active_mask"]  # [B, K]
    n_active_total = active_mask.sum().item()
    n_total_speakers = active_mask.numel()
    
    stats["active_speakers_ratio"] = n_active_total / max(n_total_speakers, 1)
    stats["n_active_speakers"] = n_active_total
    stats["n_total_speakers"] = n_total_speakers
    
    # Optional: confidence-weighted statistics
    if "speaker_confidence_scores" in batch:
        confidence = batch["speaker_confidence_scores"]  # [B, K]
        active_confidence = confidence[active_mask]
        if len(active_confidence) > 0:
            stats["avg_active_confidence"] = active_confidence.mean().item()
            stats["min_active_confidence"] = active_confidence.min().item()
    
    return loss, stats


def enhanced_training_loop(model, train_loader, val_loader, optimizer, scheduler, 
                         stft, loss_weights, device, epochs, log_interval=50):
    """
    Enhanced training loop with active speaker detection monitoring.
    """
    model.train()
    
    # Enhanced statistics tracking
    epoch_stats = {
        "total_batches": 0,
        "total_active_speakers": 0,
        "total_possible_speakers": 0,
        "activity_corrections": 0
    }
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_stats = {k: 0 for k in epoch_stats}  # Reset
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            # Enhanced training step
            with autocast("cuda"):
                loss, batch_stats = enhanced_training_step(
                    model, batch, stft, loss_weights, device
                )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            epoch_stats["total_batches"] += 1
            epoch_stats["total_active_speakers"] += batch_stats.get("n_active_speakers", 0)
            epoch_stats["total_possible_speakers"] += batch_stats.get("n_total_speakers", 0)
            
            # Enhanced logging
            if batch_idx % log_interval == 0:
                active_ratio = batch_stats.get("active_speakers_ratio", 0)
                avg_confidence = batch_stats.get("avg_active_confidence", None)
                
                log_msg = f"Batch {batch_idx}: Loss={loss.item():.4f}, Active={active_ratio:.2%}"
                if avg_confidence is not None:
                    log_msg += f", Confidence={avg_confidence:.3f}"
                    
                pbar.set_postfix_str(log_msg)
            
        # Epoch summary
        avg_loss = epoch_loss / max(epoch_stats["total_batches"], 1)
        overall_active_ratio = (epoch_stats["total_active_speakers"] / 
                              max(epoch_stats["total_possible_speakers"], 1))
        
        logging.info(f"=== EPOCH {epoch+1} SUMMARY ===")
        logging.info(f"Average Loss: {avg_loss:.4f}")
        logging.info(f"Active Speaker Ratio: {overall_active_ratio:.2%}")
        logging.info(f"Total Batches: {epoch_stats['total_batches']}")
        
        # Log enhanced dataset statistics if available
        if hasattr(train_loader.dataset, 'log_activity_summary'):
            train_loader.dataset.log_activity_summary()
        
        scheduler.step()


# Example configuration for enhanced training
def get_enhanced_config_example():
    """
    Example configuration showing new enhanced parameters.
    """
    config_additions = {
        "data": {
            "use_enhanced_detection": True,      # Enable energy-based validation
            "energy_threshold_db": -35,          # Slightly higher than default -40dB
            "min_speech_duration": 0.15,        # Require 150ms minimum speech
            "joint_for": ["train"]               # Use joint dataset for training
        },
        "training": {
            "log_activity_stats": True,          # Log detailed activity statistics
            "validate_energy_interval": 100,    # Validate energy every N batches  
            "confidence_weighting": False       # Whether to use confidence scores in loss
        }
    }
    
    logging.info("Enhanced configuration options:")
    logging.info("1. use_enhanced_detection: Energy-based validation")
    logging.info("2. energy_threshold_db: Speech activity threshold")
    logging.info("3. min_speech_duration: Minimum active speech duration")
    logging.info("4. Additional activity statistics and logging")
    
    return config_additions


if __name__ == "__main__":
    print("Enhanced Training Integration for Active Speaker Detection")
    print("\nKey Features:")
    print("✅ Energy-based speaker activity validation")
    print("✅ Active speaker masking in joint loss")
    print("✅ Enhanced statistics and logging")
    print("✅ Confidence scores for activity decisions")
    print("✅ Runtime validation and correction")
    
    print("\nTo use in your training:")
    print("1. Replace get_dataset() with get_enhanced_dataset()")
    print("2. Add enhanced config parameters")
    print("3. Use enhanced_training_step() for active masking")
    print("4. Monitor activity statistics in logs")
    
    # Show example config
    get_enhanced_config_example()