import itertools
import torch
import torch.nn.functional as F
import logging


def _sisdr(x, s, eps=1e-8):
    # x,s: [B,K,T] or [B,1,T]; returns [B,K]
    x_zm = x - x.mean(dim=-1, keepdim=True)
    s_zm = s - s.mean(dim=-1, keepdim=True)
    t = (
        torch.sum(x_zm * s_zm, dim=-1, keepdim=True)
        / (torch.sum(s_zm**2, dim=-1, keepdim=True) + eps)
    ) * s_zm
    e = x_zm - t
    return 10 * torch.log10(
        (torch.sum(t**2, dim=-1) + eps) / (torch.sum(e**2, dim=-1) + eps)
    )


def joint_loss(S_hat_c, Y_ref_c, batch, stft, weights=(1.0, 1.0)):
    """
    Speaker-aware (no PIT) joint loss.

    Args:
        S_hat_c: [B, K, T, F] complex — model estimate per enrolled speaker
        Y_ref_c: [B, K, T, F] complex — clean reference STFT per speaker
        batch:
            target_all: [B, K, Tw]  — clean waveforms
            target_lens_all: [B, K] — clean lengths (samples)
        stft: STFT wrapper with .inverse(C, lengths=...) returning wave
        weights: (w_sep, w_time) for STFT L1 and SI-SDR terms
    Returns:
        loss (scalar), stats (dict)
    """
    assert torch.is_complex(S_hat_c) and torch.is_complex(Y_ref_c)
    B, K = S_hat_c.shape[:2]
    w_sep, w_time = weights

    # --- 1) STFT-domain separation loss (index-aligned, no permutation) ---
    # L1 over complex plane (equivalently L1 over RI if you prefer)
    L_sep = torch.abs(S_hat_c - Y_ref_c).mean()

    # --- 2) Time-domain SI-SDR (index-aligned, no PIT) ---
    # Use exact per-(B,K) sample lengths so iSTFT returns the right shapes
    tgt_lens = batch["target_lens_all"].to(S_hat_c.device)  # [B,K] samples
    y_wav = batch["target_all"].to(S_hat_c.device)  # [B,K,Tw]

    # iSTFT expects [*, F, T] internally; your wrapper handles [B,K,T,F] 𝒞
    s_hat_wav = stft.inverse(S_hat_c, lengths=tgt_lens)  # [B,K,Tw’] (matched)
    sisdr = _sisdr(s_hat_wav, y_wav).mean()

    # --- 3) Combine ---
    loss = w_sep * L_sep + w_time * (-sisdr)

    # --- 4) Diagnostics (distinctness and correlation) ---
    stats = {
        "loss": float(loss.detach()),
        "L_sep": float(L_sep.detach()),
        "SI_SDR": float(sisdr.detach()),
        "S_hat_c": tuple(S_hat_c.shape),
        "Y_ref_c": tuple(Y_ref_c.shape),
        "s_hat_wav": tuple(s_hat_wav.shape),
        "y_wav": tuple(y_wav.shape),
    }

    # Optional: check output streams are not collapsing
    col_stats = check_collapse(s_hat_wav, y_wav)
    stats.update(col_stats)
    return loss, stats


def check_collapse(s_hat_wav, y_wav):
    """
    Check if the output streams are collapsing.

    Args:
        s_hat_wav: [B,K,T]
        y_wav: [B,K,T]

    Returns:
        stats: dict
    """
    B, K, T = s_hat_wav.shape
    stats = {}
    with torch.no_grad():
        if K >= 2:
            d01 = torch.mean(torch.abs(s_hat_wav[:, 0] - s_hat_wav[:, 1])).item()
            stats["mean_|s0-s1|"] = d01
        if K >= 3:
            d12 = torch.mean(torch.abs(s_hat_wav[:, 1] - s_hat_wav[:, 2])).item()
            stats["mean_|s1-s2|"] = d12

        # quick cosine corr for k=0 on first batch item
        def _corr(a, b):
            num = (a * b).sum()
            den = a.pow(2).sum().sqrt() * b.pow(2).sum().sqrt() + 1e-12
            return (num / den).item()

        k0 = 0
        stats["corr_k0"] = (
            _corr(s_hat_wav[0, k0], y_wav[0, k0]) if B > 0 and K > 0 else 0.0
        )
    return stats
