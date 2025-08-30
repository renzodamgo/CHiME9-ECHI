import torch
from pathlib import Path
import wandb
import logging
import numpy as np
import torchaudio

from train.file_utils import read_json, write_json


class LossTracker:
    def __init__(self):
        self.loss = torch.tensor(0.0)
        self.steps = 0
        self.history = []

    def update(self, loss: torch.Tensor):
        self.loss += loss.to(self.loss.device)
        self.steps += 1

    def get_average(self) -> float:
        return self.loss.item() / self.steps if self.steps > 0 else 0

    def reset(self, epoch):
        self.history.append((epoch, self.get_average()))
        self.loss = torch.tensor(0.0)
        self.steps = 0


class Gromit:
    def __init__(
        self,
        epochs,
        loss_name,
        exp_name,
        output_path,
        debug,
        wandb_entity,
        wandb_project,
    ):
        self.debug = debug
        self.epochs = epochs
        self.loss_name = loss_name
        self.exp_name = exp_name
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project

        self.use_wandb = (
            (not self.debug)
            and (self.wandb_entity is not None)
            and (self.wandb_project is not None)
        )

        self.train_loss = LossTracker()
        self.val_loss = LossTracker()
        self.val_stoi = LossTracker()
        self.train_l_sep = LossTracker()
        self.train_si_sdr = LossTracker()
        self.val_l_sep = LossTracker()
        self.val_si_sdr = LossTracker()

        self.output_path = Path(output_path)
        self.json_name = self.output_path / "train_log.json"

        self.train_sampledir = self.output_path / "train_samples"
        self.dev_sampledir = self.output_path / "val_samples"
        self.train_sampledir.mkdir(parents=True, exist_ok=True)
        self.dev_sampledir.mkdir(parents=True, exist_ok=True)

        self.ckpt_dir = self.output_path / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def start_training(self):
        write_json(self.json_name, [])

        run_name = self.output_path.name

        if self.use_wandb:
            wandb.init(
                entity=self.wandb_entity,
                project=self.wandb_project,
                name=run_name,
                dir=self.output_path,
            )

        # logging.info("Training")

    def epoch_report(
        self, epoch, do_ckpt, model: torch.nn.Module, lr: float, stats: dict, delim=" "
    ):

        train_log = read_json(self.json_name)
        new_log = {
            "epoch": epoch,
            "train_loss": np.nan,
            "val_loss": np.nan,
            "val_stoi": np.nan,
            "lr": lr,
            **stats,
        }

        wlog = {}

        train_loss = self.train_loss.get_average()
        train_l_sep = self.train_l_sep.get_average()
        train_si_sdr = self.train_si_sdr.get_average()
        out_string = f"{delim}Epoch {epoch+1} report"
        out_string += f"{delim}Train loss: {train_loss:.6f}"
        out_string += f"{delim}Train L_sep: {train_l_sep:.6f}"
        out_string += f"{delim}Train SI-SDR: {train_si_sdr:.6f}"
        self.train_loss.reset(epoch)
        self.train_l_sep.reset(epoch)
        self.train_si_sdr.reset(epoch)

        wlog[f"train_{self.loss_name}"] = train_loss
        wlog["train_L_sep"] = train_l_sep
        wlog["train_SI_SDR"] = train_si_sdr
        new_log["train_loss"] = train_loss

        val_loss = self.val_loss.get_average()
        val_stoi = self.val_stoi.get_average()
        val_l_sep = self.val_l_sep.get_average()
        val_si_sdr = self.val_si_sdr.get_average()
        out_string += f"{delim}Val loss: {val_loss:.6f}"
        out_string += f"{delim}Val stoi: {val_stoi:.6f}"
        out_string += f"{delim}Val L_sep: {val_l_sep:.6f}"
        out_string += f"{delim}Val SI-SDR: {val_si_sdr:.6f}"
        self.val_loss.reset(epoch)
        self.val_stoi.reset(epoch)
        self.val_l_sep.reset(epoch)
        self.val_si_sdr.reset(epoch)

        wlog[f"val_{self.loss_name}"] = val_loss
        wlog["val_stoi"] = val_stoi
        wlog["val_L_sep"] = val_l_sep
        wlog["val_SI_SDR"] = val_si_sdr
        wlog["lr"] = lr
        new_log["val_loss"] = val_loss
        new_log["val_stoi"] = val_stoi

        if do_ckpt:
            ckpt_path = self.get_ckpt_path(epoch, self.exp_name)
            torch.save(model.state_dict(), ckpt_path)
            # logging.info(f"model dict: {model.state_dict().keys()}")
            # logging.info(f"SAVED CHECKPOINT {epoch} to {ckpt_path}")

        out_string += f"{delim}LR: {lr}"

        # logging.info(out_string)
        train_log.append(new_log)
        write_json(self.json_name, train_log)

        if self.use_wandb:
            wandb.log(wlog)

    def save_sample(
        self,
        sample: torch.Tensor,
        fs: int,
        split: str,
        epoch: int,
        scene: str,
        audio_type: str,
    ):
        if split == "train":
            audio_dir = self.train_sampledir
        elif split == "dev":
            audio_dir = self.dev_sampledir
        else:
            raise ValueError(f"Unknown split: {split}")

        audio_fname = f"{scene}_{audio_type}.wav"
        # if audio type contains proc add epoch
        audio_fname = f"epoch{str(epoch).zfill(3)}_" + audio_fname
        audio_path = str(audio_dir / audio_fname)

        # DEBUG: Log sample stats before saving
        import logging
        # logging.info(f"DEBUG: save_sample - {audio_fname} - shape={sample.shape}, min={sample.min():.6f}, max={sample.max():.6f}, mean={sample.mean():.6f}")

        if sample.ndim == 1:
            sample = sample.unsqueeze(0)
        elif sample.ndim == 3:
            sample = sample.squeeze(0)

        # DEBUG: Log sample stats after reshaping
        # logging.info(f"DEBUG: save_sample after reshape - {audio_fname} - shape={sample.shape}, min={sample.min():.6f}, max={sample.max():.6f}")

        torchaudio.save(audio_path, sample, fs)

        # DEBUG: Log file path and size
        import os
        file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        # logging.info(f"DEBUG: saved {audio_path} - file_size={file_size} bytes")

    def get_ckpt_path(self, epoch, exp_name):
        return self.ckpt_dir / f"{exp_name}_{str(epoch).zfill(3)}.pt"
