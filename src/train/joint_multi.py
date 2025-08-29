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


def joint_loss(S_hat_c, Y_ref_c, batch, stft, weights=(1.0, 1.0), adaptive_weighting=True, 
               amplitude_aware=True, amplitude_loss_weight=0.5):
    """

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

    # --- 2) Time-domain SI-SDR  ---
    # Use exact per-(B,K) sample lengths so iSTFT returns the right shapes
    tgt_lens = batch["target_lens_all"].to(S_hat_c.device)  # [B,K] samples
    y_wav = batch["target_all"].to(S_hat_c.device)  # [B,K,Tw]

    # iSTFT expects [*, F, T] internally; your wrapper handles [B,K,T,F] 𝒞
    s_hat_wav = stft.inverse(S_hat_c, lengths=tgt_lens)  # [B,K,Tw’] (matched)
    sisdr = _sisdr(s_hat_wav, y_wav).mean()

    # --- 3) Combine with adaptive weighting and target-proportional scaling ---
    # First compute per-speaker RMS for proportional weighting (before main amplitude analysis)
    temp_s_hat_rms_per_spk = torch.sqrt(torch.mean(s_hat_wav**2, dim=-1) + 1e-8)  # [B, K]
    temp_y_ref_rms_per_spk = torch.sqrt(torch.mean(y_wav**2, dim=-1) + 1e-8)      # [B, K]
    
    # Target-proportional weighting: boost SI-SDR loss weight for louder targets
    if amplitude_aware:
        # Scale SI-SDR importance based on target amplitude
        # Louder targets get more emphasis on audio quality (SI-SDR)
        amplitude_weights = torch.clamp(temp_y_ref_rms_per_spk * 50.0, min=0.5, max=3.0)  # [B, K]
        mean_amplitude_weight = amplitude_weights.mean()
        
        # Apply target-proportional weighting to SI-SDR
        proportional_w_time = w_time * mean_amplitude_weight
    else:
        proportional_w_time = w_time
    
    if adaptive_weighting:
        # Normalize weights based on typical scales to ensure balanced contribution
        # L_sep: typically 0.02-0.5, scale factor ~2
        # SI-SDR: typically -40 to 0 dB, scale factor ~20
        # This ensures both losses contribute roughly equally when weights are (1.0, 1.0)
        normalized_L_sep = L_sep * 2.0  # Scale L_sep up
        normalized_sisdr = (-sisdr) / 20.0  # Scale SI-SDR down
        loss = w_sep * normalized_L_sep + proportional_w_time * normalized_sisdr
    else:
        normalized_L_sep = L_sep
        normalized_sisdr = (-sisdr)
        loss = w_sep * L_sep + proportional_w_time * (-sisdr)
    
    # --- 4) Amplitude-aware loss components ---
    # Compute per-speaker and global amplitude statistics
    s_hat_rms_per_spk = torch.sqrt(torch.mean(s_hat_wav**2, dim=-1, keepdim=True) + 1e-8)  # [B, K, 1]
    y_ref_rms_per_spk = torch.sqrt(torch.mean(y_wav**2, dim=-1, keepdim=True) + 1e-8)      # [B, K, 1]
    s_hat_rms = torch.sqrt(torch.mean(s_hat_wav**2) + 1e-8)  # Global RMS
    y_ref_rms = torch.sqrt(torch.mean(y_wav**2) + 1e-8)      # Global RMS
    
    amplitude_loss = 0.0
    silence_penalty = 0.0
    
    if amplitude_aware:
        # 4a) Amplitude Ratio Loss - penalize deviations from target amplitude
        # Only apply to speakers with sufficient target amplitude
        active_speakers_mask = (y_ref_rms_per_spk.squeeze(-1) > 0.001)  # [B, K]
        if active_speakers_mask.any():
            # Relative amplitude error for active speakers only
            amplitude_ratio_error = torch.abs(s_hat_rms_per_spk - y_ref_rms_per_spk) / (y_ref_rms_per_spk + 1e-8)
            amplitude_loss = (amplitude_ratio_error * active_speakers_mask.unsqueeze(-1)).mean()
        
        # 4b) Dynamic Anti-Silence Penalty (scaled by target amplitude)
        # Stronger penalty for loud targets that produce quiet outputs
        for b in range(B):
            for k in range(K):
                target_rms = y_ref_rms_per_spk[b, k, 0]
                output_rms = s_hat_rms_per_spk[b, k, 0]
                
                # Only penalize if target is loud enough but output is too quiet
                if target_rms > 0.005 and output_rms < 0.5 * target_rms:
                    # Scale penalty by how loud the target should be
                    penalty_scale = torch.clamp(target_rms * 20.0, min=0.1, max=2.0)
                    amplitude_deficit = target_rms - output_rms
                    silence_penalty += penalty_scale * amplitude_deficit
        
        silence_penalty = silence_penalty / (B * K)  # Normalize by number of speakers
        
        # Add amplitude losses to total loss
        loss = loss + amplitude_loss_weight * amplitude_loss + silence_penalty
    else:
        # Original static anti-silence penalty (backward compatibility)
        if y_ref_rms > 0.01 and s_hat_rms < 0.001:
            silence_penalty = 0.1 * (0.001 - s_hat_rms)
            loss = loss + silence_penalty

    # --- 4) Diagnostics (distinctness and correlation) ---
    stats = {
        "loss": float(loss.detach()),
        "L_sep": float(L_sep.detach()),
        "SI_SDR": float(sisdr.detach()),
        "S_hat_c": tuple(S_hat_c.shape),
        "Y_ref_c": tuple(Y_ref_c.shape),
        "s_hat_wav": tuple(s_hat_wav.shape),
        "y_wav": tuple(y_wav.shape),
        
        # Add amplitude monitoring to detect silence convergence
        "s_hat_rms": float(s_hat_rms.detach()),
        "s_hat_max_abs": float(torch.max(torch.abs(s_hat_wav)).detach()),
        "y_ref_rms": float(y_ref_rms.detach()),
        "silence_penalty": float(silence_penalty) if isinstance(silence_penalty, torch.Tensor) else silence_penalty,
        
        # Add comprehensive amplitude analysis
        "s_hat_rms_per_spk": [float(s_hat_rms_per_spk[0, k, 0].detach()) for k in range(K)] if B > 0 else [],
        "y_ref_rms_per_spk": [float(y_ref_rms_per_spk[0, k, 0].detach()) for k in range(K)] if B > 0 else [],
        "amplitude_ratio_error": float(amplitude_loss) if isinstance(amplitude_loss, torch.Tensor) else amplitude_loss,
        
        # Add loss component analysis with proportional weighting
        "L_sep_contribution": float((w_sep * (normalized_L_sep if adaptive_weighting else L_sep)).detach()),
        "SI_SDR_contribution": float((proportional_w_time * (normalized_sisdr if adaptive_weighting else (-sisdr))).detach()),
        "proportional_weight_applied": float(proportional_w_time / w_time) if w_time > 0 else 1.0,
        
        # Amplitude-aware loss components
        "amplitude_loss": float(amplitude_loss) if isinstance(amplitude_loss, torch.Tensor) else amplitude_loss,
        "amplitude_loss_contribution": float(amplitude_loss_weight * amplitude_loss) if isinstance(amplitude_loss, torch.Tensor) else 0.0,
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
