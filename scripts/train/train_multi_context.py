import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import logging
import math
import random
from torch_stoi import NegSTOILoss
from torch.utils.data.dataloader import DataLoader
from train.multi_context_loss import contrastive_multi_speaker_loss, log_separation_metrics
from train.echi import ECHIJoint, collate_fn_joint
from shared.core_utils import get_model, get_device
from train.losses import get_loss, get_lrmethod
from train.gromit import Gromit
from shared.signal_utils import STFTWrapper, match_length, prep_audio
from torch.amp import autocast, GradScaler

# Import the new architecture
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.MultiSpeakerContextGridNet import MultiSpeakerContextGridNet

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_context_aware_model(model_cfg):
    """Create MultiSpeakerContextGridNet model."""
    model = MultiSpeakerContextGridNet(
        n_imics=model_cfg.input.channels,
        n_layers=model_cfg.n_layers,
        lstm_hidden_units=model_cfg.lstm_hidden_units,
        attn_n_head=model_cfg.attn_n_head,
        attn_qk_output_channel=model_cfg.attn_qk_output_channel,
        emb_dim=model_cfg.emb_dim,
        emb_ks=model_cfg.emb_ks,
        emb_hs=model_cfg.emb_hs,
        activation=model_cfg.activation,
        eps=model_cfg.eps,
        context_layers=getattr(model_cfg, 'context_layers', 3),
        context_heads=getattr(model_cfg, 'context_heads', 4),
    ).to(get_device())
    
    return model


def get_dataset(split, data_cfg, debug):
    """Get dataset for multi-context training - matches train_script_multi.py exactly."""
    from train.enhanced_echi_joint import EnhancedECHIJoint
    
    logging.info(f"=== CONTEXT-AWARE DATASET SETUP ({split}) ===")
    logging.info(f"Creating EnhancedECHIJoint dataset for {split}")
    
    # Use same parameter structure as working train_script_multi.py
    dataset = EnhancedECHIJoint(
        split,  # subset parameter
        data_cfg.device,  # audio_device parameter
        data_cfg.noisy_signal,
        data_cfg.ref_signal,
        data_cfg.rainbow_signal,
        data_cfg.sessions_file,
        data_cfg.segments_file,
        debug,
        # Enhanced parameters - make them more permissive to avoid filtering all data
        validate_energy=True,
        energy_threshold_db=-45,  # More permissive threshold
        min_speech_duration=0.05   # Shorter minimum duration
    )
    
    data_len = len(dataset)
    logging.info(f"Dataset length: {data_len}")
    
    if data_len == 0:
        logging.error(f"❌ No samples found in {split} dataset!")
        logging.error("   This might be due to:")
        logging.error("   1. Missing training data preparation")
        logging.error("   2. Too strict energy validation parameters")
        logging.error("   3. Incorrect file paths")
        raise ValueError(f"Empty {split} dataset")
    
    loader_cfg = getattr(data_cfg.loader, split)
    saves = list(range(min(5, data_len)))  # Save first 5 scenes
    
    dataloader = DataLoader(
        dataset,
        batch_size=loader_cfg.batch_size,
        num_workers=loader_cfg.num_workers,
        shuffle=loader_cfg.shuffle,
        collate_fn=collate_fn_joint,
        pin_memory=True,
        drop_last=True
    )
    
    logging.info(f"📊 {split.upper()} Dataset:")
    logging.info(f"   Size: {len(dataset)} samples")
    logging.info(f"   Batch size: {loader_cfg.batch_size}")
    logging.info(f"   Num workers: {loader_cfg.num_workers}")
    
    return dataloader, saves


def train_epoch(
    epoch,
    model,
    trainset,
    stft,
    stoi_fn,
    gromit,
    model_cfg,
    device,
    debug,
    optimizer,
    input_channels,
    input_sr,
    input_rms,
    trainsaves,
    scaler,
    loss_config=None
):
    """Enhanced training epoch with context-aware loss."""
    logging.info(f"=== CONTEXT-AWARE TRAINING epoch {epoch} ===")
    model.train()
    
    gromit.train_loss.reset(epoch)
    gromit.train_sisdr.reset(epoch)
    
    # Loss configuration
    if loss_config is None:
        loss_config = {
            'weights': {
                'sisdr': 1.0,
                'contrastive': 0.5,
                'separation': 0.3,
                'distinctiveness': 0.2
            },
            'temperature': 0.1
        }
    
    loader = tqdm(trainset, desc="Training loop") if debug else trainset
    
    # Progressive difficulty: start with easier examples
    # (This could be enhanced with curriculum learning)
    
    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        
        # Move data to device
        noisy = batch["noisy"].to(device)      # [B, M, T] mixture
        target_all = batch["target_all"].to(device)  # [B, K, T] all speakers
        spk_all = batch["spkid_all"].to(device)      # [B, K, T] all enrollments
        spkid_lens_all = batch["spkid_lens_all"].to(device)  # [B, K] lengths
        
        B, M, T_mix = noisy.shape
        B, K, T_tgt = target_all.shape
        B, K, T_spk = spk_all.shape
        
        # Ensure compatible lengths
        min_length = min(T_mix, T_tgt)
        noisy = noisy[:, :, :min_length]
        target_all = target_all[:, :, :min_length]
        
        # Convert to STFT domain
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            # Mixture STFT: [B, M, F, T, 2] or [B, M, T, F, 2]
            noisy_tf = stft(noisy)
            
            # Speaker enrollments STFT
            spk_all_tf = stft(spk_all)  # [B,K,F,T,2]  
            spk_all_for_model = spk_all_tf.permute(0, 1, 3, 2, 4).contiguous()  # [B,K,T,F,2]
            
            # Adjust lengths for STFT frames (following train_script_multi pattern)
            spk_lens_all = (
                batch["spkid_lens_all"].to(device) - stft.n_fft
            ) // stft.hop_length
            
            # Forward pass
            s_hat_tf = model(noisy_tf, spk_all_for_model, spk_lens_all)  # [B, K, T, F] complex
            
            # Convert back to waveform domain
            s_hat_wav = stft.istft(s_hat_tf)  # [B, K, T] 
            
            # Match target length
            s_hat_wav = match_length(s_hat_wav, target_all)
            
        # Compute context-aware loss
        # Get context info from model's auxiliary encoder (if available)
        context_info = getattr(model, '_last_context_info', {
            'num_speakers': K,
            'separation_difficulty': torch.zeros(B, device=device),
            'speaker_similarities': torch.eye(K, device=device).unsqueeze(0).expand(B, -1, -1),
        })
        
        loss, loss_components = contrastive_multi_speaker_loss(
            s_hat_wav, target_all, context_info, 
            loss_weights=loss_config['weights'],
            temperature=loss_config['temperature']
        )
        
        # Backward pass with mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        gromit.train_loss.update(loss)
        
        # Compute SI-SDR for monitoring (following train_script_multi pattern)
        with torch.no_grad():
            # Using the SI-SDR from loss_components if available
            if 'sisdr' in loss_components:
                sisdr_db = loss_components['sisdr'].item()
                gromit.train_sisdr.update(torch.tensor(sisdr_db))
        
        # Detailed logging every N batches
        if batch_idx % 100 == 0:
            log_separation_metrics(loss_components, epoch, batch_idx, split="train")
        
        # Save samples periodically
        if batch_idx % 500 == 0 and trainsaves:
            save_context_aware_samples(
                s_hat_wav.detach().cpu(), batch["id"], trainsaves, 
                gromit, model_cfg, "train", epoch, batch=batch
            )
    
    logging.info(f"✅ Training epoch {epoch} completed")
    logging.info(f"   Average loss: {gromit.train_loss.get_average():.6f}")
    logging.info(f"   Average SI-SDR: {gromit.train_sisdr.get_average():.4f} dB")


def validate_epoch(
    epoch,
    model,
    devset,
    stft,
    stoi_fn,
    gromit,
    model_cfg,
    device,
    debug,
    do_checkpoint,
    lr_scheduler,
    do_lrschedule,
    optimizer,
    input_channels,
    input_sr,
    input_rms,
    devsaves,
    stats,
    scaler,
    loss_config=None
):
    """Context-aware validation."""
    logging.info(f"=== CONTEXT-AWARE VALIDATION epoch {epoch} ===")
    model.eval()
    
    gromit.val_loss.reset(epoch)
    gromit.val_sisdr.reset(epoch)
    
    if loss_config is None:
        loss_config = {
            'weights': {
                'sisdr': 1.0,
                'contrastive': 0.5,
                'separation': 0.3,
                'distinctiveness': 0.2
            },
            'temperature': 0.1
        }
    
    loader = tqdm(devset, desc="Validation loop") if debug else devset
    
    all_metrics = []
    
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                # Move data to device  
                noisy = batch["noisy"].to(device)
                target_all = batch["target_all"].to(device)
                spk_all = batch["spkid_all"].to(device)
                spkid_lens_all = batch["spkid_lens_all"].to(device)
                
                B, M, T_mix = noisy.shape
                B, K, T_tgt = target_all.shape
                
                # Ensure compatible lengths
                min_length = min(T_mix, T_tgt)
                noisy = noisy[:, :, :min_length]
                target_all = target_all[:, :, :min_length]
                
                # Convert to STFT
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    noisy_tf = stft(noisy)
                    spk_all_tf = stft(spk_all)  # [B,K,F,T,2]
                    spk_all_for_model = spk_all_tf.permute(0, 1, 3, 2, 4).contiguous()  # [B,K,T,F,2]
                    
                    spk_lens_all = (
                        batch["spkid_lens_all"].to(device) - stft.n_fft
                    ) // stft.hop_length
                    
                    # Forward pass
                    s_hat_tf = model(noisy_tf, spk_all_for_model, spk_lens_all)
                    s_hat_wav = stft.istft(s_hat_tf)
                    s_hat_wav = match_length(s_hat_wav, target_all)
                
                # Compute loss and metrics
                context_info = getattr(model, '_last_context_info', {
                    'num_speakers': K,
                    'separation_difficulty': torch.zeros(B, device=device),
                    'speaker_similarities': torch.eye(K, device=device).unsqueeze(0).expand(B, -1, -1),
                })
                
                loss, loss_components = contrastive_multi_speaker_loss(
                    s_hat_wav, target_all, context_info,
                    loss_weights=loss_config['weights'],
                    temperature=loss_config['temperature']
                )
                
                gromit.val_loss.update(loss)
                
                # SI-SDR (following train_script_multi pattern)
                if 'sisdr' in loss_components:
                    sisdr_db = loss_components['sisdr'].item()
                    gromit.val_sisdr.update(torch.tensor(sisdr_db))
                
                all_metrics.append(loss_components)
                
                # Log periodically  
                if batch_idx % 50 == 0:
                    log_separation_metrics(loss_components, epoch, batch_idx, split="val")
                
                # Save samples
                if batch_idx % 200 == 0 and devsaves:
                    save_context_aware_samples(
                        s_hat_wav.detach().cpu(), batch["id"], devsaves,
                        gromit, model_cfg, "val", epoch
                    )
    
    except Exception as e:
        logging.error(f"Validation error: {e}")
        raise
    
    # Aggregate validation metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if isinstance(all_metrics[0][key], torch.Tensor):
                values = [m[key].item() if m[key].numel() == 1 else m[key] for m in all_metrics]
                if all(isinstance(v, (int, float)) for v in values):
                    avg_metrics[key] = sum(values) / len(values)
        
        log_separation_metrics(avg_metrics, epoch, 0, split="val_summary")
    
    # Learning rate scheduling
    if do_lrschedule:
        lr_scheduler.step(gromit.val_loss.get_average())
    
    # Checkpoint
    gromit.epoch_report(
        epoch, do_checkpoint, model, optimizer.param_groups[0]["lr"], stats,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler if do_lrschedule else None,
        scaler=scaler
    )
    
    logging.info(f"✅ Validation epoch {epoch} completed")
    logging.info(f"   Average loss: {gromit.val_loss.get_average():.6f}")
    logging.info(f"   Average SI-SDR: {gromit.val_sisdr.get_average():.4f} dB")


def save_context_aware_samples(s_hat_wav_cpu, scenes_in_batch, scenes_to_save, 
                              gromit, model_cfg, split, epoch, batch=None):
    """Save processed samples with context information."""
    if not scenes_to_save:
        return
    
    for b_idx, scene in enumerate(scenes_in_batch):
        if scene in scenes_to_save:
            num_speakers = s_hat_wav_cpu.shape[1]
            
            # Save each separated speaker
            for k_idx in range(num_speakers):
                spk_audio = s_hat_wav_cpu[b_idx, k_idx]
                
                gromit.save_sample(
                    spk_audio,
                    model_cfg.input.sample_rate,
                    split,
                    epoch,
                    scene,
                    f"context_spk{k_idx}",
                )
            
            # Save targets and noisy if available (every 2 epochs)
            if batch is not None and epoch % 2 == 0:
                if "target_all" in batch:
                    target_all = batch["target_all"].detach().cpu()
                    for k_idx in range(num_speakers):
                        gromit.save_sample(
                            target_all[b_idx, k_idx],
                            model_cfg.input.sample_rate,
                            split,
                            epoch,
                            scene,
                            f"target_spk{k_idx}",
                        )
                
                if "noisy" in batch:
                    noisy = batch["noisy"].detach().cpu()
                    gromit.save_sample(
                        noisy[b_idx, 0],  # First microphone
                        model_cfg.input.sample_rate,
                        split,
                        epoch,
                        scene,
                        "noisy",
                    )


def run(
    data_cfg,
    model_cfg,
    train_cfg,
    exp_dir,
    debug,
    wandb_entity=None,
    wandb_project=None,
    resume_from_checkpoint=None,
):
    """Run context-aware multi-speaker training."""
    logging.info("=== CONTEXT-AWARE MULTI-SPEAKER TRAINING ===")
    logging.info(f"Device: {get_device()}")
    logging.info(f"Debug mode: {debug}")
    logging.info(f"Experiment directory: {exp_dir}")
    logging.info(f"Model: MultiSpeakerContextGridNet")
    logging.info(f"Training epochs: {train_cfg.epochs}")
    logging.info("🌟 Enhanced Features:")
    logging.info("   ✅ Multi-speaker context awareness")
    logging.info("   ✅ Contrastive speaker learning")
    logging.info("   ✅ Context-aware FiLM conditioning")
    logging.info("   ✅ Separation difficulty adaptation")
    
    device = get_device()
    
    # Training helper
    gromit = Gromit(
        train_cfg.epochs,
        "context_contrastive",  # Custom loss name
        train_cfg.exp_name,
        exp_dir,
        debug,
        wandb_entity,
        wandb_project,
    )
    
    # STFT setup
    assert model_cfg.input.type == "stft", "Context training requires STFT input"
    stft = STFTWrapper(**model_cfg.input.stft, device=device)
    
    # Data loading
    trainset, trainsaves = get_dataset("train", data_cfg, debug)
    devset, devsaves = get_dataset("dev", data_cfg, debug)
    
    # Model setup - use our enhanced model
    model = get_context_aware_model(model_cfg)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    stoi_fn = NegSTOILoss(model_cfg.input.sample_rate).to(device)
    ckpt_interval = train_cfg.checkpoint_interval
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Learning rate scheduling
    do_lrschedule = train_cfg.schedule_lr is not None
    if do_lrschedule:
        lr_scheduler = get_lrmethod(
            train_cfg.schedule_lr.name, optimizer, train_cfg.schedule_lr.params
        )
    
    # Loss configuration
    loss_config = {
        'weights': getattr(train_cfg, 'loss_weights', {
            'sisdr': 1.0,
            'contrastive': 0.5,
            'separation': 0.3,
            'distinctiveness': 0.2
        }),
        'temperature': getattr(train_cfg, 'contrastive_temperature', 0.1)
    }
    
    # Checkpoint resumption
    start_epoch = 0
    if resume_from_checkpoint is not None:
        logging.info(f"🔄 Resuming training from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            # Try to load, handle architecture differences gracefully
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                logging.info("✅ Model state loaded (some parameters may be new)")
            except Exception as e:
                logging.warning(f"⚠️  Could not load all model parameters: {e}")
                logging.info("🔧 Continuing with randomly initialized parameters")
        
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logging.info("✅ Optimizer state loaded")
            except Exception as e:
                logging.warning(f"⚠️  Could not load optimizer state: {e}")
        
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"📍 Resuming from epoch {start_epoch}")
    
    gromit.start_training()
    
    # Training loop
    for epoch in range(start_epoch, train_cfg.epochs):
        logging.info(f"🚀 Starting epoch {epoch}")
        
        # Training
        train_epoch(
            epoch, model, trainset, stft, stoi_fn, gromit, model_cfg,
            device, debug, optimizer, 
            model_cfg.input.channels, model_cfg.input.sample_rate, 
            model_cfg.input.rms, trainsaves, scaler, loss_config
        )
        
        # Validation
        do_checkpoint = (epoch % ckpt_interval) == 0
        validate_epoch(
            epoch, model, devset, stft, stoi_fn, gromit, model_cfg,
            device, debug, do_checkpoint, lr_scheduler, do_lrschedule,
            optimizer, model_cfg.input.channels, model_cfg.input.sample_rate,
            model_cfg.input.rms, devsaves, {}, scaler, loss_config
        )
    
    logging.info("✅ Context-aware training completed successfully!")


if __name__ == "__main__":
    # This would be called by hydra in the actual setup
    pass