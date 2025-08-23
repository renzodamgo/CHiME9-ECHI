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


def joint_loss(S_hat_c, Y_ref_c, batch, stft, weights=(1.0, 1.0)):
    """
    S_hat_c: [B,K,T,F] complex (or RI convertible before call)
    Y_ref_c: [B,K,T,F] complex
    """
    B, K = S_hat_c.shape[:2]

    # Time-domain signals
    s_hat_wav = stft.inverse(S_hat_c, lengths=batch["target_lens_all"])  # [B,K,T]
    y_wav = batch["target_all"].to(s_hat_wav.device)  # [B,K,T]

    # PIT: find best permutation by SI-SDR
    best_perm = _pit_best_perm(s_hat_wav, y_wav)
    y_wav_aligned = _permute(y_wav, best_perm)
    Y_ref_c_aligned = _permute(Y_ref_c, best_perm)

    # Losses
    L_sep = torch.abs(S_hat_c - Y_ref_c_aligned).mean()  # STFT L1
    sisdr = _sisdr(s_hat_wav, y_wav_aligned).mean()  # SI-SDR on aligned

    w_sep, w_time = weights
    loss = w_sep * L_sep + w_time * (-sisdr)
    diff01 = torch.mean(torch.abs(s_hat_wav[:, 0] - s_hat_wav[:, 1])).item()
    diff12 = torch.mean(torch.abs(s_hat_wav[:, 1] - s_hat_wav[:, 2])).item()
    logging.info(f"Mean |s0-s1|: {diff01:.3e} | |s1-s2|: {diff12:.3e}")

    # Are K outputs distinct?
    diff01 = torch.mean(torch.abs(s_hat_wav[:, 0] - s_hat_wav[:, 1])).item()
    diff12 = torch.mean(torch.abs(s_hat_wav[:, 1] - s_hat_wav[:, 2])).item()
    logging.info(f"Mean |s0-s1|: {diff01:.3e} | |s1-s2|: {diff12:.3e}")

    # Correlation with aligned refs (should trend upward over training)
    def _corr(a, b):
        num = (a * b).sum()
        den = a.pow(2).sum().sqrt() * b.pow(2).sum().sqrt() + 1e-12
        return (num / den).item()

    for k in range(s_hat_wav.shape[1]):
        logging.info(f"corr(k={k}): {_corr(s_hat_wav[0,k], y_wav_aligned[0,k]):.3f}")

    stats = {
        "loss": float(loss.detach()),
        "L_sep": float(L_sep.detach()),
        "SI_SDR": float(sisdr.detach()),
        "S_hat_c": S_hat_c.shape,
        "Y_ref_c": Y_ref_c.shape,
        "s_hat_wav": s_hat_wav.shape,
        "y_wav": y_wav.shape,
        "y_wav_aligned": y_wav_aligned.shape,
        "Y_ref_c_aligned": Y_ref_c_aligned.shape,
    }
    logging.info(f"Stats: {stats}")
    return loss, stats
