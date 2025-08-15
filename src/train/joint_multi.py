# train/losses/joint_multi.py
import torch
import torch.nn.functional as F

def _l1_complex(a, b):
    # a,b: complex [..., T, F]
    return F.l1_loss(a.real, b.real) + F.l1_loss(a.imag, b.imag)

def _vad_from_tf(Yc, thr_db=-40.0, eps=1e-10):
    # Yc: [B,K,T,F] complex
    p = (Yc.real**2 + Yc.imag**2).mean(dim=-1)          # [B,K,T]
    db = 10.0 * torch.log10(p + eps)
    return (db > thr_db)                                 # bool [B,K,T]

def joint_loss(
    S_hat_c,      # [B,K,T,F] complex  (model outputs, ref channel)
    X_ref_c,      # [B,  T,F] complex  (mixture, same ref channel)
    Y_ref_c,      # [B,K,T,F] complex  (GT per speaker, ref channel)
    y_wav=None,   # [B,K, Tw] waveform (optional, only if you also want SI-SDR)
    istft_fn=None,
    lambda_mix=0.1,
    lambda_xtalk=0.1,
    use_sisdr=False,
):
    B, K, T, F = S_hat_c.shape

    # (1) separation (spec L1 per speaker)
    L_sep = 0.0
    for k in range(K):
        L_sep = L_sep + _l1_complex(S_hat_c[:, k], Y_ref_c[:, k])
    L_sep = L_sep / K

    # (2) mixture consistency on complex TF (sum_k S_hat ≈ X_ref)
    L_mix = _l1_complex(S_hat_c.sum(dim=1), X_ref_c)

    # (3) VAD-gated cross-talk (penalize energy of ŝ_k when k is silent)
    vad = _vad_from_tf(Y_ref_c).float()                 # [B,K,T]
    silent = (1.0 - vad)
    est_e = (S_hat_c.real**2 + S_hat_c.imag**2).mean(dim=-1)  # [B,K,T]
    L_xt = (silent * est_e).sum() / silent.sum().clamp_min(1.0)

    # (4) optional: add SI-SDR on waveforms
    if use_sisdr and (y_wav is not None) and (istft_fn is not None):
        s_hat_wav = istft_fn(S_hat_c)                   # [B,K, Tw]
        # tiny SI-SDR impl
        def _sisdr(x, s, eps=1e-8):
            s_energy = (s**2).sum(-1, keepdim=True) + eps
            proj = (x*s).sum(-1, keepdim=True) * s / s_energy
            e = x - proj
            return 10*torch.log10(((proj**2).sum(-1)+eps)/((e**2).sum(-1)+eps))
        sisdr = _sisdr(s_hat_wav, y_wav).mean()
        L_sep = L_sep - sisdr                            # maximize SI-SDR

    total = L_sep + lambda_mix*L_mix + lambda_xtalk*L_xt
    stats = {
        "L_sep": float(L_sep.detach()),
        "L_mix": float(L_mix.detach()),
        "L_xtalk": float(L_xt.detach()),
        "L_total": float(total.detach()),
    }
    return total, stats
