"""
Example of how to use Universal GridNet in your existing train_script_multi.py

Key Changes:
1. Change model config to use "universal" instead of "baseline"
2. The ECHIJoint dataset already provides speaker_active_mask
3. The joint_loss already uses active_mask properly
4. Universal model handles any number of speakers automatically
"""

import torch
import hydra
from omegaconf import DictConfig
import logging

# Your existing imports work as-is
from train.joint_multi import joint_loss
from train.echi import ECHIJoint, collate_fn_joint
from shared.core_utils import get_model, get_device  # Updated to support universal model
from shared.signal_utils import STFTWrapper
from torch.amp import autocast


@hydra.main(version_base=None, config_path="../../checkpoints", config_name="ha_universal_config")
def train_universal_gridnet(cfg: DictConfig) -> None:
    """Train Universal GridNet - drop-in replacement for train_script_multi.py"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    device = get_device()
    
    # Load Universal GridNet model (automatically selected via config.name = "universal")
    model = get_model(cfg)
    model = model.to(device)
    
    logging.info("🌟 UNIVERSAL GRIDNET TRAINING STARTED")
    logging.info(f"Model: {cfg.name}")  # Should be "universal"
    logging.info(f"Device: {device}")
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Your existing data loading works perfectly!
    # ECHIJoint already provides speaker_active_mask
    data_cfg = {
        "device": "ha",
        "noisy_signal": "data/ha/train/train_{session}.ha.wav",
        "ref_signal": "data/ref/train/train_{session}.ha.{pid}.wav", 
        "rainbow_signal": "data/participant/train/{pid}.wav",
        "sessions_file": "data/metadata/sessions.train.csv",
        "segments_file": "data/metadata/ref/train_{session}.ha.{pid}.csv",
        "joint_for": ["train"]  # Use ECHIJoint for multi-speaker training
    }
    
    # Create dataset - no changes needed!
    train_dataset = ECHIJoint(
        "train", 
        data_cfg["device"],
        data_cfg["noisy_signal"],
        data_cfg["ref_signal"], 
        data_cfg["rainbow_signal"],
        data_cfg["sessions_file"],
        data_cfg["segments_file"],
        debug=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn_joint  # Handles speaker_active_mask automatically
    )
    
    # STFT wrapper
    stft = STFTWrapper(**cfg.input.stft).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop - mostly unchanged!
    model.train()
    
    for epoch in range(10):  # Example: 10 epochs
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Move batch to device
            for key in ["noisy", "target_all", "spkid_all", "speaker_active_mask"]:
                if key in batch:
                    batch[key] = batch[key].to(device)
            
            # STFT transforms
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
            
            with autocast("cuda"):
                # Universal GridNet forward pass - handles any K automatically!
                separated_stft = model(noisy_stft, spkid_stft, spkid_lens)  # [B, K, T', F] complex
                
                # Target STFT
                target_all = batch["target_all"]  # [B, K, T]
                target_stft = stft(target_all.view(-1, target_all.shape[-1]),
                                  lengths=batch["target_lens_all"].view(-1))  # [BK, 1, T', F, 2]
                target_stft = target_stft.view(B, K, *target_stft.shape[1:]).squeeze(2)  # [B, K, T', F, 2]
                target_stft_complex = torch.complex(target_stft[..., 0], target_stft[..., 1])  # [B, K, T', F]
                
                # Joint loss with active speaker masking (no changes needed!)
                loss, stats = joint_loss(
                    separated_stft, 
                    target_stft_complex,
                    batch,  # Contains speaker_active_mask automatically
                    stft,
                    weights=(0.0, 1.0)  # Pure SI-SDR loss
                )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Enhanced logging
            if batch_idx % 10 == 0:
                active_mask = batch["speaker_active_mask"]  # [B, K]
                n_active = active_mask.sum().item()
                n_total = active_mask.numel()
                active_ratio = n_active / max(n_total, 1)
                
                logging.info(f"Epoch {epoch}, Batch {batch_idx}: "
                           f"Loss={loss.item():.4f}, "
                           f"SI-SDR={stats.get('sisdr_db', 0):.2f}dB, "
                           f"Active={active_ratio:.1%} ({n_active}/{n_total})")
        
        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"=== EPOCH {epoch} COMPLETE ===")
        logging.info(f"Average Loss: {avg_loss:.4f}")
        
        # Log Universal GridNet specific info
        if hasattr(model, 'num_spk'):
            logging.info(f"Universal model supports: {model.num_spk} speakers")
    
    logging.info("🎉 UNIVERSAL GRIDNET TRAINING COMPLETED!")


if __name__ == "__main__":
    train_universal_gridnet()