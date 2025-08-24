import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import logging
import math
from torch_stoi import NegSTOILoss
from torch.utils.data.dataloader import DataLoader
from train.joint_multi import joint_loss
from train.echi import ECHI, collate_fn
from train.echi import ECHIJoint, collate_fn_joint
from shared.core_utils import get_model, get_device
from train.losses import get_loss, get_lrmethod
from train.gromit import Gromit
from shared.signal_utils import STFTWrapper, match_length, prep_audio
from torch.amp import autocast

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(666)


def log_detailed_stats(stats, model, optimizer):
    with torch.no_grad():
        grad_sq = 0.0
        param_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_sq += p.grad.detach().pow(2).sum().item()
            param_sq += p.detach().pow(2).sum().item()
        stats["grad_norm"] = grad_sq**0.5
        stats["param_norm"] = param_sq**0.5

        # LR from the first param group (adjust if you use multiple groups)
        stats["lr"] = optimizer.param_groups[0]["lr"]

        # VRAM
        if torch.cuda.is_available():
            stats["vram_alloc_MB"] = torch.cuda.memory_allocated() / 1024**2
            stats["vram_reserved_MB"] = torch.cuda.memory_reserved() / 1024**2
            stats["vram_max_alloc_MB"] = torch.cuda.max_memory_allocated() / 1024**2
    logging.info(
        "Stats: %s",
        {k: (float(v) if isinstance(v, torch.Tensor) else v) for k, v in stats.items()},
    )


def save_sample(
    sample_rate: int,
    processed: torch.Tensor,
    batch_scenes: list,
    save_scenes: list,
    split: str,
    epoch: int,
    noisy: torch.Tensor,
    target: torch.Tensor,
    gromit: Gromit,
):
    saves = list(set(batch_scenes) & set(save_scenes))
    if not saves:
        return None

    processed = processed.detach().cpu()
    if epoch == 0:
        noisy = noisy.detach().cpu()
        target = target.detach().cpu()
    for i, scene in enumerate(batch_scenes):
        if scene in save_scenes:
            gromit.save_sample(
                processed[i],
                sample_rate,
                split,
                epoch,
                scene,
                "proc",
            )
            if epoch == 0:
                gromit.save_sample(
                    noisy[i],
                    sample_rate,
                    split,
                    epoch,
                    scene,
                    "noisy",
                )
                gromit.save_sample(
                    target[i],
                    sample_rate,
                    split,
                    epoch,
                    scene,
                    "target",
                )


def check_lengths(
    scene: list[str],
    processed: torch.Tensor,
    target: torch.Tensor,
    split: str,
    do_stft: bool,
):
    use_val = True
    if processed.shape[-1] != target.shape[-1]:
        len_diff = abs(processed.shape[-1] - target.shape[-1])
        if not do_stft and len_diff > 1000:
            # Difference not due to stft
            logging.error(
                f"Time samples mismatch ({split}). Batch: {scene}. Proc: {processed.shape[-1]}. Targ: {target.shape[-1]}"
            )
            use_val = False
        processed, target = match_length(processed, target)
    return processed, target, use_val


def get_dataset(split: str, data_cfg: DictConfig, debug: bool):
    """
    If `split` is listed in data_cfg.joint_for (e.g., ["train"]),
    we use ECHIJoint + collate_fn_joint; otherwise the classic ECHI.
    """

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

    data_len = len(data)
    samples = [data.__getitem__(i * data_len // 5)["id"] for i in range(1, 4)]

    loader = DataLoader(
        data,
        **data_cfg.loader[split],
        collate_fn=chosen_collate,  # <- switches automatically
    )

    return loader, samples


def run(
    data_cfg,
    model_cfg,
    train_cfg,
    exp_dir,
    debug,
    wandb_entity=None,
    wandb_project=None,
):

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

    # Model and training bits and bobs

    if model_cfg.input.type == "stft":
        do_stft = True
        stft = STFTWrapper(**model_cfg.input.stft, device=device)
    elif model_cfg.input.type != "wave":
        logging.error(f"Unrecognised model input type {model_cfg.input.type}")
    else:
        do_stft = False

    trainset, trainsaves = get_dataset("train", data_cfg, debug)
    devset, devsaves = get_dataset("dev", data_cfg, debug)

    model = get_model(model_cfg, None)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    stoi_fn = NegSTOILoss(model_cfg.input.sample_rate).to(device)
    ckpt_interval = train_cfg.checkpoint_interval

    do_lrschedule = train_cfg.schedule_lr is not None
    if do_lrschedule:
        lr_scheduler = get_lrmethod(
            train_cfg.schedule_lr.name, optimizer, train_cfg.schedule_lr.params
        )

    model.to(device)

    gromit.start_training()

    # Train this fine chap
    for epoch in range(train_cfg.epochs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model.train()

        if debug:
            loader = tqdm(trainset, desc="Training loop")
        else:
            loader = trainset

        try:
            num_batches = len(loader)
        except TypeError:
            num_batches = None

        for batch_idx, batch in enumerate(loader, start=1):
            global_step = (epoch * (num_batches or 0)) + (batch_idx - 1)
            bn = f"{batch_idx}" + (f"/{num_batches}" if num_batches else "")
            # Log batch keys to understand data structure
            logging.info(
                f"=== BATCH DEBUG (epoch {epoch} | batch {bn} | global {global_step}) ==="
            )
            # logging.info(f"Batch keys: {list(batch.keys())}")
            # logging.info(f"Batch ID: {batch.get('id', 'N/A')}")

            # Prep
            noisy = batch["noisy"].to(device, non_blocking=True)  # [B, C, Tw]
            spk_all = batch["spkid_all"].to(device, non_blocking=True)  # [B, K, Tr]
            targ_all = batch["target_all"].to(device, non_blocking=True)  # [B, K, Tw]

            logging.info(f"noisy shape: {noisy.shape}")
            logging.info(f"spk_all shape: {spk_all.shape}")
            logging.info(f"targ_all shape: {targ_all.shape}")

            noisy_tf = stft(noisy)  # → [B, M, T, F, 2]
            spk_all_tf = stft(spk_all)  #  [B,K,F,T,2]
            spk_all_for_model = spk_all_tf.permute(0, 1, 3, 2, 4).contiguous()
            assert spk_all_for_model.shape[-1] == 2 and spk_all_for_model.shape[
                -2
            ] == getattr(
                stft, "n_freqs", stft.n_fft // 2 + 1
            ), f"Expected [B,K,T,F,2], got {spk_all_for_model.shape}"

            spk_lens_all = (
                batch["spkid_lens_all"].to(device) - stft.n_fft
            ) // stft.hop_length  # [B, K]

            # reference complex mixture (pick mic 0)
            X_ref_c = torch.complex(
                noisy_tf[:, 0, ..., 0], noisy_tf[:, 0, ..., 1]
            )  # [B, F, T], complex
            X_ref_c = X_ref_c.permute(0, 2, 1).contiguous()  # [B, T, F], complex
            Y_ref_tf = stft(targ_all)  # [B, K, 2, T, F]
            # Build complex from last-dim RI, then permute to [B, K, T, F] to match S_hat_c
            Y_ref_c = torch.complex(
                Y_ref_tf[..., 0], Y_ref_tf[..., 1]
            )  # [B, K, F, T], complex
            Y_ref_c = Y_ref_c.permute(0, 1, 3, 2).contiguous()  # [B, K, T, F], complex

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16):
                S_hat_c = model(noisy_tf, spk_all_for_model, spk_lens_all)
                loss, stats = joint_loss(
                    S_hat_c, Y_ref_c, batch, stft, weights=(1.0, 0.5)
                )
                # def joint_loss(S_hat_c, Y_ref_c, batch, stft, weights=(1.0, 1.0)):

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
            # --- end added block ---
            optimizer.step()

            gromit.train_loss.update(loss.detach())

            log_detailed_stats(stats, model, optimizer)

        do_checkpoint = (epoch % ckpt_interval == 0 and epoch > 0) or (
            (epoch + 1) == train_cfg.epochs
        )

        validate(
            epoch,
            model,
            devset,
            stft,
            stoi_fn,
            gromit,
            model_cfg,
            device,
            do_stft,
            debug,
            do_checkpoint,
        )

        # step LR *every epoch* using current val loss
        if do_lrschedule:
            lr_scheduler.step(gromit.val_loss.get_average())

        gromit.epoch_report(
            epoch, do_checkpoint, model, optimizer.param_groups[0]["lr"]
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
    do_stft,
    debug,
    do_checkpoint,
):
    # --- VALIDATION: run every epoch ---
    logging.info(f"=== VALIDATION epoch {epoch} ===")
    model.eval()

    # (recommended) reset epoch metrics
    gromit.val_loss.reset(epoch)
    gromit.val_stoi.reset(epoch)

    loader = tqdm(devset, desc="Validation loop") if debug else devset
    with torch.no_grad():
        for batch in loader:

            # ===== MULTI-SPEAKER PATH (mirrors joint_loss) =====
            noisy = batch["noisy"].to(device, non_blocking=True)  # [B,C,Tw]
            spk_all = batch["spkid_all"].to(device, non_blocking=True)  # [B,K,Tr]
            targ_all = batch["target_all"].to(device, non_blocking=True)  # [B,K,Tw]

            assert do_stft, "Joint validation expects STFT path."
            noisy_tf = stft(noisy)  # [B,M,T,F,2]
            spk_all_tf = stft(spk_all)  # [B,K,F,T,2]
            spk_all_for_model = spk_all_tf.permute(
                0, 1, 3, 2, 4
            ).contiguous()  # [B,K,T,F,2]

            spk_lens_all = (
                batch["spkid_lens_all"].to(device) - stft.n_fft
            ) // stft.hop_length  # [B,K]

            # Forward
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

            # Loss exactly like training joint_loss
            val_loss, _ = joint_loss(S_hat_c, Y_ref_c, batch, stft, weights=(1.0, 0.5))
            gromit.val_loss.update(val_loss.detach())

            # === STOI per (b,k), using per-speaker valid lengths ===
            s_hat_wav = stft.inverse(
                S_hat_c, lengths=batch["target_lens_all"].to(device)
            )  # [B,K,T]
            y_wav = targ_all  # [B,K,T]
            min_stoi_len = int(
                math.ceil(7680 * model_cfg.input.sample_rate / 10000.0)
            )  # ~0.768s@10kHz

            B, K = s_hat_wav.shape[:2]
            for b in range(B):
                for k in range(K):
                    L = int(batch["target_lens_all"][b, k])
                    L = min(L, s_hat_wav.size(-1), y_wav.size(-1))
                    if L >= min_stoi_len:
                        proc = s_hat_wav[b, k, :L].unsqueeze(0).contiguous()
                        targ = y_wav[b, k, :L].unsqueeze(0).contiguous()
                        try:
                            stoi_score = stoi_fn(proc, targ)  # NegSTOILoss
                            gromit.val_stoi.update(-stoi_score[0])
                        except RuntimeError:
                            pass  # skip pathological clips
            # Only save samples when checkpointing (avoid file explosion)
            if do_checkpoint:
                k0 = 0
                gromit.save_sample(
                    s_hat_wav[:, k0].detach().cpu(),
                    model_cfg.input.sample_rate,
                    "dev",
                    epoch,
                    batch["id"],
                    "proc_k0",
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


# Add logging for debugging tensor shapes
# logging.info(f"=== MULTI-SPEAKER TRAINING DEBUG ===")
# logging.info(f"noisy shape: {noisy.shape}")
# logging.info(f"spk_all shape: {spk_all.shape}")
# logging.info(f"targ_all shape: {targ_all.shape}")
# logging.info(f"noisy_tf shape: {noisy_tf.shape}")
# logging.info(f"spk_all_tf shape: {spk_all_tf.shape}")
# logging.info(f"spk_lens_all shape: {spk_lens_all.shape}")
# logging.info(f"spk_all_for_model shape: {spk_all_for_model.shape}")
# logging.info(f"Model input channels: {model_cfg.input.channels}")
# logging.info(f"STFT n_fft: {stft.n_fft}, hop_length: {stft.hop_length}")
# logging.info(f"Device: {device}")
