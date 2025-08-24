import itertools
import torch
import torch.nn.functional as F
import logging


def _l1_complex(a, b):
    T = min(a.size(-2), b.size(-2))
    Freq = min(a.size(-1), b.size(-1))
    if a.size(-2) != T or b.size(-2) != T or a.size(-1) != Freq or b.size(-1) != Freq:
        a = a[..., :T, :Freq]
        b = b[..., :T, :Freq]
    return F.l1_loss(a.real, b.real) + F.l1_loss(a.imag, b.imag)


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


def _permute(x, perms):  # x: [B,K,...]
    B, K = x.shape[:2]
    return torch.stack([x[b, list(perms[b])].contiguous() for b in range(B)], dim=0)


def _pit_best_perm(pred_wav, ref_wav):
    # pred_wav, ref_wav: [B,K,T]
    B, K, T = pred_wav.shape
    perms = list(itertools.permutations(range(K)))
    best = []
    for b in range(B):
        # pairwise SI-SDR [K,K]
        S = []
        for i in range(K):
            # broadcast to [1,K,T] for ref_wav[b]
            s = _sisdr(
                pred_wav[b, i][None, None, :].expand(1, K, T), ref_wav[b][None, :, :]
            ).squeeze(
                0
            )  # [K]
            S.append(s)
        S = torch.stack(S, dim=0)  # [K,K]
        scores = [S[range(K), p].sum() for p in perms]
        j = int(torch.argmax(torch.stack(scores)))
        best.append(perms[j])
    return best  # list of tuples per B


# def joint_loss(S_hat_c, Y_ref_c, batch, stft, weights=(1.0, 1.0)):
#     """
#     S_hat_c: [B,K,T,F] complex (or RI convertible before call)
#     Y_ref_c: [B,K,T,F] complex
#     """
#     B, K = S_hat_c.shape[:2]

#     # Time-domain signals
#     s_hat_wav = stft.inverse(S_hat_c, lengths=batch["target_lens_all"])  # [B,K,T]
#     y_wav = batch["target_all"].to(s_hat_wav.device)  # [B,K,T]

#     # PIT: find best permutation by SI-SDR
#     best_perm = _pit_best_perm(s_hat_wav, y_wav)
#     y_wav_aligned = _permute(y_wav, best_perm)
#     Y_ref_c_aligned = _permute(Y_ref_c, best_perm)

#     # Losses
#     L_sep = torch.abs(S_hat_c - Y_ref_c_aligned).mean()  # STFT L1
#     sisdr = _sisdr(s_hat_wav, y_wav_aligned).mean()  # SI-SDR on aligned

#     w_sep, w_time = weights
#     loss = w_sep * L_sep + w_time * (-sisdr)
#     diff01 = torch.mean(torch.abs(s_hat_wav[:, 0] - s_hat_wav[:, 1])).item()
#     diff12 = torch.mean(torch.abs(s_hat_wav[:, 1] - s_hat_wav[:, 2])).item()
#     logging.info(f"Mean |s0-s1|: {diff01:.3e} | |s1-s2|: {diff12:.3e}")

#     # Are K outputs distinct?
#     diff01 = torch.mean(torch.abs(s_hat_wav[:, 0] - s_hat_wav[:, 1])).item()
#     diff12 = torch.mean(torch.abs(s_hat_wav[:, 1] - s_hat_wav[:, 2])).item()
#     logging.info(f"Mean |s0-s1|: {diff01:.3e} | |s1-s2|: {diff12:.3e}")

# Correlation with aligned refs (should trend upward over training)
# def _corr(a, b):
#     num = (a * b).sum()
#     den = a.pow(2).sum().sqrt() * b.pow(2).sum().sqrt() + 1e-12
#     return (num / den).item()

# for k in range(s_hat_wav.shape[1]):
#     logging.info(f"corr(k={k}): {_corr(s_hat_wav[0,k], y_wav_aligned[0,k]):.3f}")

# stats = {
#     "loss": float(loss.detach()),
#     "L_sep": float(L_sep.detach()),
#     "SI_SDR": float(sisdr.detach()),
#     "S_hat_c": S_hat_c.shape,
#     "Y_ref_c": Y_ref_c.shape,
#     "s_hat_wav": s_hat_wav.shape,
#     "y_wav": y_wav.shape,
#     "y_wav_aligned": y_wav_aligned.shape,
#     "Y_ref_c_aligned": Y_ref_c_aligned.shape,
# }
# logging.info(f"Stats: {stats}")
# return loss, stats


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

    # # If your _sisdr supports lengths, pass them; else we’ll crop pairwise.
    # def _crop_to_lengths(x, L):
    #     # x: [B,K,T], L: [B,K]
    #     # returns list of (x_bk[:L], y_bk[:L]) assembled back with padding-safe mean
    #     xs, masks = [], []
    #     T = x.size(-1)
    #     L_clamped = L.clamp_max(T)
    #     arange = torch.arange(T, device=x.device)[None, None, :]
    #     mask = (arange < L_clamped[..., None]).float()  # [B,K,T]
    #     return mask

    # Try preferred: _sisdr(x, y, lengths=...)
    # try:
    sisdr = _sisdr(s_hat_wav, y_wav).mean()
    # except TypeError:
    #     # Fallback: mask invalid tail before computing SI-SDR if your impl supports masks
    #     mask = _crop_to_lengths(y_wav, tgt_lens)  # [B,K,T]
    #     # If your _sisdr doesn’t support masks either, you can manually crop per (B,K).
    #     sisdr = _sisdr(s_hat_wav * mask, y_wav * mask).mean()

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
