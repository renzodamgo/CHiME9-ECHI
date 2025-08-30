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


def analyze_speaker_separation(s_hat_wav, y_wav):
    """
    Analyze speaker separation quality in the output.
    
    Args:
        s_hat_wav: [B, K, T] predicted separated waveforms
        y_wav: [B, K, T] target separated waveforms
        
    Returns:
        dict: Speaker separation metrics
    """
    B, K, T = s_hat_wav.shape
    stats = {}
    
    with torch.no_grad():
        # 1. Cross-speaker correlation analysis (lower is better for separation)
        cross_correlations = []
        for b in range(min(B, 1)):  # Only analyze first batch for efficiency
            for i in range(K):
                for j in range(i + 1, K):
                    s_i = s_hat_wav[b, i]  # [T]
                    s_j = s_hat_wav[b, j]  # [T]
                    
                    # Pearson correlation
                    s_i_mean = s_i.mean()
                    s_j_mean = s_j.mean()
                    numerator = ((s_i - s_i_mean) * (s_j - s_j_mean)).sum()
                    denominator = (((s_i - s_i_mean) ** 2).sum().sqrt() * 
                                   ((s_j - s_j_mean) ** 2).sum().sqrt() + 1e-8)
                    corr = (numerator / denominator).item()
                    cross_correlations.append(abs(corr))
        
        stats["cross_speaker_corr_mean"] = sum(cross_correlations) / len(cross_correlations) if cross_correlations else 0.0
        stats["cross_speaker_corr_max"] = max(cross_correlations) if cross_correlations else 0.0
        
        # 2. Speaker distinctness: pairwise L2 distance between outputs
        pairwise_distances = []
        for b in range(min(B, 1)):
            for i in range(K):
                for j in range(i + 1, K):
                    dist = torch.norm(s_hat_wav[b, i] - s_hat_wav[b, j]).item()
                    pairwise_distances.append(dist)
        
        stats["speaker_l2_distance_mean"] = sum(pairwise_distances) / len(pairwise_distances) if pairwise_distances else 0.0
        stats["speaker_l2_distance_min"] = min(pairwise_distances) if pairwise_distances else 0.0
        
        # 3. Energy distribution across speakers
        speaker_energies = []
        for b in range(min(B, 1)):
            energies = []
            for k in range(K):
                energy = (s_hat_wav[b, k] ** 2).mean().item()
                energies.append(energy)
            speaker_energies.append(energies)
        
        if speaker_energies:
            energies = speaker_energies[0]
            stats["speaker_energy_std"] = torch.tensor(energies).std().item()
            stats["speaker_energy_ratio"] = max(energies) / (min(energies) + 1e-8)
            stats["speaker_energies"] = energies
        
        # 4. Spectral diversity (frequency content differences)
        if K >= 2:
            try:
                # Simple spectral centroid difference
                for b in range(min(B, 1)):
                    s1_fft = torch.fft.rfft(s_hat_wav[b, 0])
                    s2_fft = torch.fft.rfft(s_hat_wav[b, 1])
                    
                    # Spectral centroids
                    freqs = torch.linspace(0, 1, s1_fft.size(0), device=s_hat_wav.device)
                    
                    s1_mag = torch.abs(s1_fft)
                    s2_mag = torch.abs(s2_fft)
                    
                    s1_centroid = (freqs * s1_mag).sum() / (s1_mag.sum() + 1e-8)
                    s2_centroid = (freqs * s2_mag).sum() / (s2_mag.sum() + 1e-8)
                    
                    stats["spectral_centroid_diff"] = abs(s1_centroid - s2_centroid).item()
            except:
                stats["spectral_centroid_diff"] = 0.0
        
        # 5. Separation quality indicator
        # Good separation: low cross-correlation, high L2 distance, balanced energy
        separation_score = 0.0
        if cross_correlations and pairwise_distances:
            # Lower correlation = better (subtract from 1)
            corr_score = 1.0 - stats["cross_speaker_corr_mean"]
            # Higher distance = better (normalize by typical range)
            dist_score = min(stats["speaker_l2_distance_mean"] / 10.0, 1.0)
            # More balanced energy = better (lower std is better)
            energy_balance_score = 1.0 / (1.0 + stats.get("speaker_energy_std", 1.0))
            
            separation_score = (corr_score + dist_score + energy_balance_score) / 3.0
        
        stats["separation_quality_score"] = separation_score
        
        # 6. Log warning if poor separation detected
        if separation_score < 0.3:
            logging.warning(f"🚨 POOR SPEAKER SEPARATION DETECTED!")
            logging.warning(f"   Cross-correlation: {stats['cross_speaker_corr_mean']:.4f} (should be < 0.3)")
            logging.warning(f"   L2 distance: {stats['speaker_l2_distance_mean']:.4f} (should be > 1.0)")
            logging.warning(f"   Energy std: {stats.get('speaker_energy_std', 0):.4f} (should be < 0.5)")
            logging.warning(f"   Separation score: {separation_score:.4f} (should be > 0.7)")
        elif separation_score > 0.7:
            logging.info(f"✅ Good speaker separation detected (score: {separation_score:.4f})")
    
    return stats


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
    # Enhanced with frequency-aware weighting to preserve high-frequency content
    error_mag = torch.abs(S_hat_c - Y_ref_c)  # [B, K, T, F]

    if amplitude_aware:
        # Frequency-aware weighting: emphasize high frequencies to prevent filtering
        F = error_mag.shape[-1]
        freq_weights = torch.linspace(1.0, 2.5, F, device=S_hat_c.device)  # Higher weight for high freq
        freq_weights = freq_weights.view(1, 1, 1, -1)  # [1, 1, 1, F]
        weighted_error = error_mag * freq_weights
        L_sep = weighted_error.mean()
    else:
        L_sep = error_mag.mean()

    # --- 2) Time-domain SI-SDR (per-speaker) ---
    # Use exact per-(B,K) sample lengths so iSTFT returns the right shapes
    tgt_lens = batch["target_lens_all"].to(S_hat_c.device)  # [B,K] samples
    y_wav = batch["target_all"].to(S_hat_c.device)  # [B,K,Tw]

    # iSTFT expects [*, F, T] internally; your wrapper handles [B,K,T,F] 𝒞
    s_hat_wav = stft.inverse(S_hat_c, lengths=tgt_lens)  # [B,K,Tw'] (matched)
    sisdr_per_spk = _sisdr(s_hat_wav, y_wav)  # [B, K] - keep per-speaker SI-SDR

    # --- 3) Combine with adaptive weighting and per-speaker amplitude scaling ---
    # Compute per-speaker RMS for amplitude-based weighting
    temp_s_hat_rms_per_spk = torch.sqrt(torch.mean(s_hat_wav**2, dim=-1) + 1e-8)  # [B, K]
    temp_y_ref_rms_per_spk = torch.sqrt(torch.mean(y_wav**2, dim=-1) + 1e-8)      # [B, K]

    # Per-speaker amplitude weighting for SI-SDR
    if amplitude_aware:
        # Per-speaker amplitude weights: louder targets get more emphasis
        amplitude_weights = torch.clamp(temp_y_ref_rms_per_spk * 50.0, min=0.5, max=3.0)  # [B, K]

        # Apply per-speaker weighting to SI-SDR and compute weighted average
        weighted_sisdr_per_spk = sisdr_per_spk * amplitude_weights  # [B, K]
        sisdr = weighted_sisdr_per_spk.mean()  # Global weighted SI-SDR

        # For backward compatibility, store the effective weight multiplier
        proportional_w_time = w_time * amplitude_weights.mean()
    else:
        # Standard averaging without amplitude weighting
        amplitude_weights = torch.ones_like(temp_y_ref_rms_per_spk)  # [B, K] all 1.0
        weighted_sisdr_per_spk = sisdr_per_spk  # No weighting applied
        sisdr = sisdr_per_spk.mean()
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

    # --- 4) Enhanced Speaker Separation Diagnostics ---
    separation_stats = analyze_speaker_separation(s_hat_wav, y_wav)
    
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

        # Add per-speaker SI-SDR monitoring
        "sisdr_per_spk": [float(sisdr_per_spk[0, k].detach()) for k in range(K)] if B > 0 else [],
        "amplitude_weights": [float(amplitude_weights[0, k].detach()) for k in range(K)] if B > 0 else [],
        "weighted_sisdr_per_spk": [float(weighted_sisdr_per_spk[0, k].detach()) for k in range(K)] if B > 0 else [],

        # Add loss component analysis with proportional weighting
        "L_sep_contribution": float((w_sep * (normalized_L_sep if adaptive_weighting else L_sep)).detach()),
        "SI_SDR_contribution": float((proportional_w_time * (normalized_sisdr if adaptive_weighting else (-sisdr))).detach()),
        "proportional_weight_applied": float(proportional_w_time / w_time) if w_time > 0 else 1.0,
        "frequency_weighted_L_sep": amplitude_aware,

        # Amplitude-aware loss components
        "amplitude_loss": float(amplitude_loss) if isinstance(amplitude_loss, torch.Tensor) else amplitude_loss,
        "amplitude_loss_contribution": float(amplitude_loss_weight * amplitude_loss) if isinstance(amplitude_loss, torch.Tensor) else 0.0,
    }
    
    # Add speaker separation analysis
    stats.update(separation_stats)

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
