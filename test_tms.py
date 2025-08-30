import torchaudio
import torch

wav, sr = torchaudio.load("data/working_dir/experiments/ha-joint-6/train_ha/train_samples/epoch006_train_08_ha_seg004_proc_spk2.wav")
tgt, _ = torchaudio.load("data/working_dir/experiments/ha-joint-6/train_ha/train_samples/epoch000_train_08_ha_seg004_target_spk2.wav")

def stats(x):
    return x.mean().item(), x.abs().max().item(), x.pow(2).mean().sqrt().item()

print("Proc (mean, peak, rms):", stats(wav))
print("Target (mean, peak, rms):", stats(tgt))
