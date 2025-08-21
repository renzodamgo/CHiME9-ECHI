import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import logging
from torch_stoi import NegSTOILoss
from torch.utils.data.dataloader import DataLoader
from train.joint_multi import joint_loss
from train.echi import ECHI, collate_fn
from train.echi import ECHIJoint, collate_fn_joint
from shared.core_utils import get_model, get_device
from train.losses import get_loss, get_lrmethod
from train.gromit import Gromit
from shared.signal_utils import STFTWrapper, match_length, prep_audio

torch.manual_seed(666)


def get_dataset(split: str, data_cfg: DictConfig, debug: bool):
    """
    If `split` is listed in data_cfg.joint_for (e.g., ["train"]),
    we use ECHIJoint + collate_fn_joint; otherwise the classic ECHI.
    """
    # Decide which dataset to use
    joint_for = set(getattr(data_cfg, "joint_for", []))  # e.g., ["train"]
    use_joint = split in joint_for

    if use_joint:
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
    samples = [data.__getitem__(i * data_len // 5)["id"] for i in range(1, 4)]

    loader = DataLoader(
        data,
        **data_cfg.loader[split],
        collate_fn=chosen_collate,  # <- switches automatically
    )

    return loader, samples


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
    loss_fn = get_loss(train_cfg.loss.name, train_cfg.loss.kwargs)
    stoi_fn = NegSTOILoss(model_cfg.input.sample_rate).to(device)
    ckpt_interval = train_cfg.checkpoint_interval

    do_lrschedule = train_cfg.schedule_lr is not None
    if do_lrschedule:
        lr_scheduler = get_lrmethod(
            train_cfg.schedule_lr.name, optimizer, train_cfg.schedule_lr.params
        )

    input_channels = model_cfg.input.channels
    input_sr = model_cfg.input.sample_rate
    input_rms = model_cfg.input.rms

    model.to(device)
    loss_fn.to(device)

    gromit.start_training()

    # Train this fine chap
    for epoch in range(train_cfg.epochs):
        model.train()

        if debug:
            loader = tqdm(trainset, desc="Training loop")
        else:
            loader = trainset

        for batch in loader:
            # Log batch keys to understand data structure
            logging.info(f"=== BATCH DEBUG ===")
            logging.info(f"Batch keys: {list(batch.keys())}")
            logging.info(f"Batch ID: {batch.get('id', 'N/A')}")

            multi = (
                ("spkid_all" in batch)
                and ("target_all" in batch)
                and ("spkid_lens_all" in batch)
            )
            logging.info(f"Multi-speaker mode: {multi}")
            if multi:
                # Prep
                noisy = batch["noisy"].to(device, non_blocking=True)  # [B, C, Tw]
                spk_all = batch["spkid_all"].to(device, non_blocking=True)  # [B, K, Tr]
                targ_all = batch["target_all"].to(
                    device, non_blocking=True
                )  # [B, K, Tw]

                noisy_tf = stft(noisy)  # → [B, M, T, F, 2]
                spk_all_tf = stft(spk_all)  # → [B, K, 2, T, F]
                spk_lens_all = (
                    batch["spkid_lens_all"].to(device) - stft.n_fft
                ) // stft.hop_length  # [B, K]

                # reference complex mixture (pick mic 0)
                X_ref_c = torch.complex(
                    noisy_tf[:, 0, :, :, 0], noisy_tf[:, 0, :, :, 1]
                )  # [B, T, F]
                Y_ref_tf = stft(targ_all)  # [B, K, 2, T, F]
                Y_ref_c = torch.complex(
                    Y_ref_tf[:, :, 0], Y_ref_tf[:, :, 1]
                )  # [B, K, T, F]

                # Model (multi-speaker forward we added in MCxTFGridNet)
                spk_all_for_model = spk_all_tf.permute(
                    0, 1, 3, 4, 2
                ).contiguous()  # [B,K,T,F,2]

                # Add logging for debugging tensor shapes
                logging.info(f"=== MULTI-SPEAKER TRAINING DEBUG ===")
                logging.info(f"noisy shape: {noisy.shape}")
                logging.info(f"spk_all shape: {spk_all.shape}")
                logging.info(f"targ_all shape: {targ_all.shape}")
                logging.info(f"noisy_tf shape: {noisy_tf.shape}")
                logging.info(f"spk_all_tf shape: {spk_all_tf.shape}")
                logging.info(f"spk_lens_all shape: {spk_lens_all.shape}")
                logging.info(f"spk_all_for_model shape: {spk_all_for_model.shape}")
                logging.info(f"Model input channels: {model_cfg.input.channels}")
                logging.info(f"STFT n_fft: {stft.n_fft}, hop_length: {stft.hop_length}")
                logging.info(f"Device: {device}")

                S_hat_c = model(
                    noisy_tf, spk_all_for_model, spk_lens_all
                )  # [B, K, T, F] (complex)

                # Joint loss (use the tiny helper from earlier message or inline your own)
                loss, stats = joint_loss(
                    S_hat_c=S_hat_c,
                    X_ref_c=X_ref_c,
                    Y_ref_c=Y_ref_c,
                    y_wav=targ_all,
                    istft_fn=lambda C: stft.inverse(C),
                    lambda_mix=0.1,
                    lambda_xtalk=0.1,
                    use_sisdr=True,
                )

                processed = stft.inverse(S_hat_c)  # [B, K, Tw] for preview/saving

            else:
                noisy = batch["noisy"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                spk_id = batch["spkid"].to(device, non_blocking=True)

                noisy = prep_audio(
                    noisy, batch["fs"], input_channels, input_sr, input_rms, True
                )
                spk_id = prep_audio(spk_id, batch["fs"], 1, input_sr, input_rms, True)

                if do_stft:
                    noisy = stft(noisy)
                    spk_id = stft(spk_id)
                    batch["spkid_lens"] = (
                        batch["spkid_lens"] - stft.n_fft
                    ) // stft.hop_length

                # Add logging for debugging single-speaker path
                logging.info(f"=== SINGLE-SPEAKER TRAINING DEBUG ===")
                logging.info(f"noisy shape: {noisy.shape}")
                logging.info(f"targets shape: {targets.shape}")
                logging.info(f"spk_id shape: {spk_id.shape}")
                logging.info(f"spkid_lens shape: {batch['spkid_lens'].shape}")
                logging.info(f"Model input channels: {model_cfg.input.channels}")
                logging.info(
                    f"STFT n_fft: {stft.n_fft if do_stft else 'N/A'}, hop_length: {stft.hop_length if do_stft else 'N/A'}"
                )
                logging.info(f"Device: {device}")

                optimizer.zero_grad()

                processed = model(noisy, spk_id, batch["spkid_lens"]).squeeze(1)

                if do_stft:
                    processed = stft.inverse(processed)

                processed, targets, use_val = check_lengths(
                    batch["id"], processed, targets, "train", do_stft
                )
                if not use_val:
                    continue

                loss = loss_fn(processed, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.clip_grad_norm
                )
                optimizer.step()

                gromit.train_loss.update(loss.detach())

                save_sample(
                    model_cfg.input.sample_rate,
                    processed,
                    batch["id"],
                    trainsaves,
                    "train",
                    epoch,
                    batch["noisy"],
                    batch["target"],
                    gromit,
                )

        do_checkpoint = (epoch % ckpt_interval == 0) or (
            (epoch + 1) == train_cfg.epochs
        )

        if do_checkpoint:
            model.eval()
            if debug:
                loader = tqdm(devset, desc="Validation loop")
            else:
                loader = devset

            with torch.no_grad():
                for batch in loader:

                    noisy = batch["noisy"].to(device, non_blocking=True)
                    targets = batch["target"].to(device, non_blocking=True)
                    spk_id = batch["spkid"].to(device, non_blocking=True)

                    noisy = prep_audio(
                        noisy, batch["fs"], input_channels, input_sr, input_rms, True
                    )
                    spk_id = prep_audio(
                        spk_id, batch["fs"], 1, input_sr, input_rms, True
                    )

                    if do_stft:
                        noisy = stft(noisy)
                        spk_id = stft(spk_id)
                        batch["spkid_lens"] = (
                            batch["spkid_lens"] - stft.n_fft
                        ) // stft.hop_length

                    processed = model(noisy, spk_id, batch["spkid_lens"]).squeeze(1)

                    if do_stft:
                        processed = stft.inverse(processed)

                    processed, targets, use_val = check_lengths(
                        batch["id"], processed, targets, "val", do_stft
                    )

                    loss = loss_fn(processed, targets)
                    stoi_processed = processed.reshape(-1, processed.shape[-1])
                    for length, proc, targ in zip(
                        batch["noisy_lens"], stoi_processed, targets
                    ):
                        stoi_score = stoi_fn(
                            proc[:length].unsqueeze(0), targ[:length].unsqueeze(0)
                        )
                        gromit.val_stoi.update(-stoi_score[0])

                    gromit.val_loss.update(loss.detach())

                    save_sample(
                        model_cfg.input.sample_rate,
                        processed,
                        batch["id"],
                        devsaves,
                        "dev",
                        epoch,
                        batch["noisy"],
                        batch["target"],
                        gromit,
                    )

            if do_lrschedule:
                lr_scheduler.step(gromit.val_loss.get_average())

        gromit.epoch_report(
            epoch,
            do_checkpoint,
            model,
            optimizer.param_groups[0]["lr"],
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
