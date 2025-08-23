# train/losses/joint_multi.py
import torch
import torch.nn.functional as F
import logging


def _l1_complex(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Both complex, same layout; crop to common [T,F] just in case
    T = min(a.size(-2), b.size(-2))
    Freq = min(a.size(-1), b.size(-1))
    if a.size(-2) != T or b.size(-2) != T or a.size(-1) != Freq or b.size(-1) != Freq:
        a = a[..., :T, :Freq]
        b = b[..., :T, :Freq]
    return F.l1_loss(a.real, b.real) + F.l1_loss(a.imag, b.imag)


def _sisdr(x: torch.Tensor, s: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x,s: [..., T] (same length)
    s_energy = (s * s).sum(-1, keepdim=True) + eps
    proj = (x * s).sum(-1, keepdim=True) * s / s_energy
    e = x - proj
    return 10.0 * torch.log10(((proj * proj).sum(-1) + eps) / ((e * e).sum(-1) + eps))


def _vad_from_tf(Yc, thr_db=-40.0, eps=1e-10):
    # Yc: [B,K,T,F] complex
    p = (Yc.real**2 + Yc.imag**2).mean(dim=-1)  # [B,K,T]
    db = 10.0 * torch.log10(p + eps)
    return db > thr_db  # bool [B,K,T]


def joint_loss(
    S_hat_c: torch.Tensor,  # [B,K,T,F] complex, model output
    Y_ref_c: torch.Tensor,  # [B,K,T,F] complex, reference target STFT
    batch: dict,
    stft,  # your STFT wrapper (with inverse(X, lengths=...))
    weights=(1.0, 1.0),  # e.g., (w_sep, w_time)
):
    B, K = S_hat_c.shape[:2]

    # --- 1) STFT-domain separation loss (shape-safe) ---
    L_sep = 0.0
    for k in range(K):
        L_sep = L_sep + _l1_complex(S_hat_c[:, k], Y_ref_c[:, k])
    L_sep = L_sep / K

    # --- 2) Time-domain SI-SDR using exact lengths ---
    # Use target_lens_all (samples) so iSTFT returns exact per-(B,K) lengths
    tgt_lens = batch["target_lens_all"]  # [B,K], on CPU or GPU
    s_hat_wav = stft.inverse(S_hat_c, lengths=tgt_lens)  # [B,K,Tw_true]
    y_wav = batch["target_all"].to(s_hat_wav.device)  # [B,K,Tw_true]

    sisdr = _sisdr(s_hat_wav, y_wav).mean()  # scalar

    # --- Weighted sum ---
    w_sep, w_time = weights
    loss = w_sep * L_sep + w_time * (-sisdr)  # maximize SI-SDR => minimize -SI-SDR

    stats = {
        "loss": loss.detach(),
        "L_sep": L_sep.detach(),
        "SI_SDR": sisdr.detach(),
        "w_sep": w_sep,
        "w_time": w_time,
    }
    return loss, stats


# def joint_loss(S_hat_c, Y_ref_c, batch, stft, weights=(1.0, 1.0)):
#     # STFT-domain L1 on complex spectra (avg over B,K,T,F)
#     L_sep = torch.abs(S_hat_c - Y_ref_c).mean()

#     # Time-domain SI-SDR in fp32 with exact per-(B,K) lengths
#     with torch.cuda.amp.autocast(enabled=False):
#         s_hat_wav = stft.inverse(S_hat_c.float(), lengths=batch["target_lens_all"])
#         y_wav = batch["target_all"].to(s_hat_wav.device).float()
#         sisdr = _sisdr(s_hat_wav, y_wav).mean()

#     w_sep, w_time = weights
#     loss = w_sep * L_sep + w_time * (-sisdr)

#     stats = {
#         "loss": loss.detach(),
#         "L_sep": L_sep.detach(),
#         "SI_SDR": sisdr.detach(),
#     }
#     logging.info(f"Stats: {stats}")
#     return loss, stats
