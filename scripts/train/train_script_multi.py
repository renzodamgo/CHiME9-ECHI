import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import logging
import math
from torch_stoi import NegSTOILoss
from torch.utils.data.dataloader import DataLoader
from train.joint_multi import joint_loss
from train.echi import ECHIJoint, collate_fn_joint
from shared.core_utils import get_model, get_device
from train.losses import get_loss, get_lrmethod
from train.gromit import Gromit
from shared.signal_utils import STFTWrapper, match_length, prep_audio
from torch.amp import autocast, GradScaler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_dataset(split: str, data_cfg: DictConfig, debug: bool):
    """
    Always use ECHIJoint dataset for multi-speaker training.
    """
    logging.info(f"=== DATASET SETUP ({split}) ===")
    logging.info(f"Creating ECHIJoint dataset for {split}")

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

    data_len = len(data)
    logging.info(f"Dataset length: {data_len}")
    samples = [data.__getitem__(i * data_len // 5)["id"] for i in range(1, 4)]
    logging.info(f"Sample IDs: {samples}")

    loader = DataLoader(
        data,
        **data_cfg.loader[split],
        collate_fn=collate_fn_joint,
    )

    logging.info(f"Loader config: {data_cfg.loader[split]}")
    logging.info(f"Collate function: {collate_fn_joint.__name__}")

    return loader, samples


def save_samples_for_scenes(
    s_hat_wav_cpu: torch.Tensor,
    scenes_in_batch: list,
    scenes_to_save: list,
    gromit,
    model_cfg,
    split: str,
    epoch: int,
    batch=None,
    save_targets_and_noisy: bool = False,
):
    """
    Save processed audio samples for specified scenes.
    
    Args:
        s_hat_wav_cpu: Processed waveforms [B, K, T] on CPU
        scenes_in_batch: Scene IDs in current batch
        scenes_to_save: Scene IDs that should be saved
        gromit: Gromit logger instance
        model_cfg: Model configuration
        split: "train" or "dev"
        epoch: Current epoch
        batch: Batch data (needed if save_targets_and_noisy=True)
        save_targets_and_noisy: Whether to also save target and noisy audio
    """
    if not scenes_to_save:
        return
        
    if save_targets_and_noisy and batch is not None:
        noisy_wav = batch["noisy"].detach().cpu()
        target_wav = batch["target_all"].detach().cpu()
    
    for b_idx, scene in enumerate(scenes_in_batch):
        if scene in scenes_to_save:
            num_speakers = s_hat_wav_cpu.shape[1]  # Get K
            
            # Save processed audio for each speaker
            for k_idx in range(num_speakers):
                spk_audio = s_hat_wav_cpu[b_idx, k_idx]
                logging.info(
                    f"DEBUG: {split} Speaker {k_idx} stats - min={spk_audio.min():.6f}, max={spk_audio.max():.6f}, mean={spk_audio.mean():.6f}, std={spk_audio.std():.6f}"
                )
                
                gromit.save_sample(
                    spk_audio,
                    model_cfg.input.sample_rate,
                    split,
                    epoch,
                    scene,
                    f"proc_spk{k_idx}",
                )
                
                # Save target audio if requested
                if save_targets_and_noisy and epoch == 0:
                    gromit.save_sample(
                        target_wav[b_idx, k_idx],
                        model_cfg.input.sample_rate,
                        split,
                        epoch,
                        scene,
                        f"target_spk{k_idx}",
                    )
            
            # Save noisy audio if requested (once per scene)
            if save_targets_and_noisy and epoch == 0:
                gromit.save_sample(
                    noisy_wav[b_idx, 0],  # Use first microphone channel
                    model_cfg.input.sample_rate,
                    split,
                    epoch,
                    scene,
                    "noisy",
                )


def validate(
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
):
    """Multi-speaker validation only."""
    logging.info(f"=== VALIDATION epoch {epoch} ===")
    model.eval()

    gromit.val_loss.reset(epoch)
    gromit.val_stoi.reset(epoch)

    loader = tqdm(devset, desc="Validation loop") if debug else devset
    with torch.no_grad():
        for batch in loader:
            # Multi-speaker validation path
            noisy = batch["noisy"].to(device, non_blocking=True)  # [B,C,Tw]
            spk_all = batch["spkid_all"].to(device, non_blocking=True)  # [B,K,Tr]
            targ_all = batch["target_all"].to(device, non_blocking=True)  # [B,K,Tw]

            logging.info(
                f"VAL BEFORE prep_audio - noisy: {noisy.shape}, spk_all: {spk_all.shape}, targ_all: {targ_all.shape}"
            )

            # Apply prep_audio (same as training)
            noisy = prep_audio(
                noisy, batch["fs"], input_channels, input_sr, input_rms, True
            )

            # Process speaker embeddings efficiently
            B, K, T_spk = spk_all.shape
            spk_all = spk_all.view(-1, T_spk).unsqueeze(1)  # [B*K, 1, T_spk]
            spk_all = prep_audio(spk_all, batch["fs"], 1, input_sr, input_rms, True)
            spk_all = spk_all.squeeze(1).view(B, K, -1)  # [B, K, T_spk']

            logging.info(
                f"VAL AFTER prep_audio - noisy: {noisy.shape}, spk_all: {spk_all.shape}"
            )

            # STFT transformation
            noisy_tf = stft(noisy)  # [B,M,T,F,2]
            spk_all_tf = stft(spk_all)  # [B,K,F,T,2]
            spk_all_for_model = spk_all_tf.permute(
                0, 1, 3, 2, 4
            ).contiguous()  # [B,K,T,F,2]

            logging.info(
                f"VAL STFT shapes - noisy_tf: {noisy_tf.shape}, spk_all_tf: {spk_all_tf.shape}"
            )

            spk_lens_all = (
                batch["spkid_lens_all"].to(device) - stft.n_fft
            ) // stft.hop_length  # [B,K]

            # Forward pass
            S_hat_c = model(
                noisy_tf, spk_all_for_model, spk_lens_all
            )  # [B,K,T,F] (complex)

            # Build Y_ref_c in the same domain
            Y_ref_tf = stft(targ_all)  # [B,K,2,T,F]
            Y_ref_c = (
                torch.complex(Y_ref_tf[..., 0], Y_ref_tf[..., 1])
                .permute(0, 1, 3, 2)
                .contiguous()
            )  # [B,K,T,F]

            # Loss computation
            val_loss, val_stats = joint_loss(
                S_hat_c, Y_ref_c, batch, stft, weights=(1.0, 1.0), adaptive_weighting=True
            )
            gromit.val_loss.update(val_loss.detach())
            gromit.val_l_sep.update(torch.tensor(val_stats["L_sep"]))
            gromit.val_si_sdr.update(torch.tensor(val_stats["SI_SDR"]))

            # DEBUG: Check STFT magnitude before inverse transform
            S_hat_mag = torch.abs(S_hat_c)
            logging.info(
                f"DEBUG: S_hat_c STFT magnitude - min={S_hat_mag.min():.6f}, max={S_hat_mag.max():.6f}, mean={S_hat_mag.mean():.6f}, shape={S_hat_c.shape}"
            )

            # STOI computation
            target_lens = batch["target_lens_all"].to(device)

            # DEBUG: Check lengths parameter
            logging.info(
                f"DEBUG: target_lens_all shape={target_lens.shape}, values={target_lens}"
            )

            # DEBUG: Check S_hat_c before inverse transform
            logging.info(
                f"DEBUG: S_hat_c before inverse - shape={S_hat_c.shape}, is_complex={S_hat_c.is_complex()}"
            )
            if S_hat_c.is_complex():
                logging.info(
                    f"DEBUG: S_hat_c complex parts - real_min={S_hat_c.real.min():.6f}, real_max={S_hat_c.real.max():.6f}, imag_min={S_hat_c.imag.min():.6f}, imag_max={S_hat_c.imag.max():.6f}"
                )

            # DEBUG: Check individual speaker spectrograms
            for k_idx in range(S_hat_c.shape[1]):
                spk_spec = S_hat_c[0, k_idx]  # [T, F]
                spec_mag = torch.abs(spk_spec)
                logging.info(
                    f"DEBUG: S_hat_c speaker {k_idx} spectrogram - shape={spk_spec.shape}, mag_min={spec_mag.min():.6f}, mag_max={spec_mag.max():.6f}, mag_mean={spec_mag.mean():.6f}"
                )

            s_hat_wav = stft.inverse(S_hat_c, lengths=target_lens)  # [B,K,T]

            # DEBUG: Check s_hat_wav stats
            logging.info(
                f"DEBUG: s_hat_wav stats - min={s_hat_wav.min():.6f}, max={s_hat_wav.max():.6f}, mean={s_hat_wav.mean():.6f}, shape={s_hat_wav.shape}"
            )
            y_wav = targ_all  # [B,K,T]
            min_stoi_len = int(math.ceil(7680 * model_cfg.input.sample_rate / 10000.0))

            B, K = s_hat_wav.shape[:2]
            for b in range(B):
                for k in range(K):
                    L = int(batch["target_lens_all"][b, k])
                    L = min(L, s_hat_wav.size(-1), y_wav.size(-1))
                    if L >= min_stoi_len:
                        proc = s_hat_wav[b, k, :L].unsqueeze(0).contiguous()
                        targ = y_wav[b, k, :L].unsqueeze(0).contiguous()
                        try:
                            stoi_score = stoi_fn(proc, targ)
                            gromit.val_stoi.update(-stoi_score[0])
                        except RuntimeError:
                            pass

            # Save samples only when checkpointing
            if do_checkpoint or epoch == 0:
                logging.info(f"SAVING VAL SAMPLES FOR EPOCH {epoch}")
                s_hat_wav_cpu = s_hat_wav.detach().cpu()
                scenes_in_batch = batch["id"]
                scenes_to_save = list(set(scenes_in_batch) & set(devsaves))
                
                save_samples_for_scenes(
                    s_hat_wav_cpu=s_hat_wav_cpu,
                    scenes_in_batch=scenes_in_batch,
                    scenes_to_save=scenes_to_save,
                    gromit=gromit,
                    model_cfg=model_cfg,
                    split="dev",
                    epoch=epoch,
                    batch=batch,
                    save_targets_and_noisy=True,
                )

    # LR scheduling
    if do_lrschedule:
        lr_scheduler.step(gromit.val_loss.get_average())

    gromit.epoch_report(
        epoch, do_checkpoint, model, optimizer.param_groups[0]["lr"], stats
    )


def run(
    data_cfg,
    model_cfg,
    train_cfg,
    exp_dir,
    debug,
    wandb_entity=None,
    wandb_project=None,
):
    logging.info("=== MULTI-SPEAKER ONLY TRAINING (OPTIMIZED) ===")
    logging.info(f"Device: {get_device()}")
    logging.info(f"Debug mode: {debug}")
    logging.info(f"Experiment directory: {exp_dir}")
    logging.info(f"Model input type: {model_cfg.input.type}")
    logging.info(f"Model input channels: {model_cfg.input.channels}")
    logging.info(f"Model input sample rate: {model_cfg.input.sample_rate}")
    logging.info(f"Training epochs: {train_cfg.epochs}")
    logging.info(f"🚀 Training batch size: {data_cfg.loader.train.batch_size}")
    logging.info(f"🚀 Validation batch size: {data_cfg.loader.dev.batch_size}")
    logging.info("🚀 Mixed precision training: ENABLED")
    logging.info("🚀 Model compilation: ENABLED (if supported)")

    device = get_device()

    # Training helper
    gromit = Gromit(
        train_cfg.epochs,
        train_cfg.loss.name,
        train_cfg.exp_name,
        exp_dir,
        debug,
        wandb_entity,
        wandb_project,
    )

    # STFT setup (always required for multi-speaker)
    assert model_cfg.input.type == "stft", "Multi-speaker training requires STFT input"
    stft = STFTWrapper(**model_cfg.input.stft, device=device)

    # Data loading
    trainset, trainsaves = get_dataset("train", data_cfg, debug)
    devset, devsaves = get_dataset("dev", data_cfg, debug)

    # Model setup
    model = get_model(model_cfg, None)
    
    # Model compilation for faster training (PyTorch 2.0+)
    try:
        model = torch.compile(model, mode='default')
        logging.info("✅ Model compilation enabled")
    except Exception as e:
        logging.warning(f"⚠️ Model compilation failed: {e}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    stoi_fn = NegSTOILoss(model_cfg.input.sample_rate).to(device)
    ckpt_interval = train_cfg.checkpoint_interval
    
    # Mixed precision training setup
    scaler = GradScaler()

    # LR scheduling
    do_lrschedule = train_cfg.schedule_lr is not None
    if do_lrschedule:
        lr_scheduler = get_lrmethod(
            train_cfg.schedule_lr.name, optimizer, train_cfg.schedule_lr.params
        )

    # Config shortcuts
    input_channels = model_cfg.input.channels
    input_sr = model_cfg.input.sample_rate
    input_rms = model_cfg.input.rms

    model.to(device)
    gromit.start_training()

    # Training loop
    for epoch in range(train_cfg.epochs):
        logging.info(f"=== EPOCH {epoch}/{train_cfg.epochs-1} START ===")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model.train()

        loader = tqdm(trainset, desc="Training loop") if debug else trainset

        try:
            num_batches = len(loader)
        except TypeError:
            num_batches = None

        for batch_idx, batch in enumerate(loader, start=1):
            global_step = (epoch * (num_batches or 0)) + (batch_idx - 1)
            bn = f"{batch_idx}" + (f"/{num_batches}" if num_batches else "")

            logging.info(
                f"=== BATCH DEBUG (epoch {epoch} | batch {bn} | global {global_step}) ==="
            )

            # Multi-speaker training path only
            noisy = batch["noisy"].to(device, non_blocking=True)  # [B, C, Tw]
            spk_all = batch["spkid_all"].to(device, non_blocking=True)  # [B, K, Tr]
            targ_all = batch["target_all"].to(device, non_blocking=True)  # [B, K, Tw]

            logging.info(
                f"BEFORE prep_audio - noisy: {noisy.shape}, spk_all: {spk_all.shape}, targ_all: {targ_all.shape}"
            )

            # Apply prep_audio preprocessing
            noisy = prep_audio(
                noisy, batch["fs"], input_channels, input_sr, input_rms, True
            )

            # Process speaker embeddings efficiently
            B, K, T_spk = spk_all.shape
            spk_all = spk_all.view(-1, T_spk).unsqueeze(1)  # [B*K, 1, T_spk]
            spk_all = prep_audio(spk_all, batch["fs"], 1, input_sr, input_rms, True)
            spk_all = spk_all.squeeze(1).view(B, K, -1)  # [B, K, T_spk']

            logging.info(
                f"AFTER prep_audio - noisy: {noisy.shape}, spk_all: {spk_all.shape}, targ_all: {targ_all.shape}"
            )
            logging.info(
                f"Sample rates - batch fs: {batch['fs']}, target sr: {input_sr}"
            )

            # STFT transformation
            noisy_tf = stft(noisy)  # → [B, M, T, F, 2]
            spk_all_tf = stft(spk_all)  #  [B,K,F,T,2]
            logging.info(
                f"STFT shapes - noisy_tf: {noisy_tf.shape}, spk_all_tf: {spk_all_tf.shape}"
            )

            # Permute for model input
            spk_all_for_model = spk_all_tf.permute(0, 1, 3, 2, 4).contiguous()
            logging.info(f"spk_all_for_model after permute: {spk_all_for_model.shape}")

            assert spk_all_for_model.shape[-1] == 2 and spk_all_for_model.shape[
                -2
            ] == getattr(
                stft, "n_freqs", stft.n_fft // 2 + 1
            ), f"Expected [B,K,T,F,2], got {spk_all_for_model.shape}"

            # Speaker length adjustment
            spk_lens_all = (
                batch["spkid_lens_all"].to(device) - stft.n_fft
            ) // stft.hop_length
            logging.info(f"spkid_lens_all original: {batch['spkid_lens_all']}")
            logging.info(f"spk_lens_all after STFT adjustment: {spk_lens_all}")

            # Build reference targets
            Y_ref_tf = stft(targ_all)  # [B, K, 2, T, F]
            Y_ref_c = (
                torch.complex(Y_ref_tf[..., 0], Y_ref_tf[..., 1])
                .permute(0, 1, 3, 2)
                .contiguous()
            )

            # Forward pass
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16):
                S_hat_c = model(noisy_tf, spk_all_for_model, spk_lens_all)
                loss, stats = joint_loss(
                    S_hat_c, Y_ref_c, batch, stft, weights=(1.0, 1.0), adaptive_weighting=True
                )

            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)

            # Statistics logging
            with torch.no_grad():
                grad_sq = sum(
                    p.grad.detach().pow(2).sum().item()
                    for p in model.parameters()
                    if p.grad is not None
                )
                param_sq = sum(
                    p.detach().pow(2).sum().item() for p in model.parameters()
                )
                stats["grad_norm"] = grad_sq**0.5
                stats["param_norm"] = param_sq**0.5
                stats["lr"] = optimizer.param_groups[0]["lr"]

                if torch.cuda.is_available():
                    stats["vram_alloc_MB"] = torch.cuda.memory_allocated() / 1024**2
                    stats["vram_reserved_MB"] = torch.cuda.memory_reserved() / 1024**2
                    stats["vram_max_alloc_MB"] = (
                        torch.cuda.max_memory_allocated() / 1024**2
                    )

            scaler.step(optimizer)
            scaler.update()
            gromit.train_loss.update(loss.detach())
            gromit.train_l_sep.update(torch.tensor(stats["L_sep"]))
            gromit.train_si_sdr.update(torch.tensor(stats["SI_SDR"]))

            # Sample saving (simplified)
            if epoch % 2 == 0 or epoch == 0:
                with torch.no_grad():
                    s_hat_wav = (
                        stft.inverse(
                            S_hat_c, lengths=batch["target_lens_all"].to(device)
                        )
                        .detach()
                        .cpu()
                    )
                    scenes_in_batch = batch["id"]
                    scenes_to_save = list(set(scenes_in_batch) & set(trainsaves))

                    save_samples_for_scenes(
                        s_hat_wav_cpu=s_hat_wav,
                        scenes_in_batch=scenes_in_batch,
                        scenes_to_save=scenes_to_save,
                        gromit=gromit,
                        model_cfg=model_cfg,
                        split="train",
                        epoch=epoch,
                        batch=batch,
                        save_targets_and_noisy=True,
                    )

        # Checkpointing
        do_checkpoint = (epoch % ckpt_interval == 0 and epoch > 0) or (
            (epoch + 1) == train_cfg.epochs
        )

        # Validation
        validate(
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
        )


@hydra.main(version_base=None, config_path="../../config/train", config_name="main_ha")
def main(cfg: DictConfig) -> None:
    run(
        cfg.dataloading,
        cfg.model,
        cfg.train,
        cfg.train_dir,
        cfg.debug,
        cfg.wandb.entity,
        cfg.wandb.project,
    )


if __name__ == "__main__":
    main()
