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
                spk_wav = s_hat_wav[b, k]  # [T]
                energy = (spk_wav ** 2).mean().item()
                energies.append(energy)

            speaker_energies.append(energies)

        # Store debug info for logging every 50 batches
        if speaker_energies:
            stats["debug_speaker_waveforms"] = []
            stats["debug_target_waveforms"] = []
            for k in range(K):
                # Model output analysis
                spk_wav = s_hat_wav[0, k]  # First batch only
                output_debug = {
                    "speaker": k,
                    "shape": tuple(spk_wav.shape),
                    "min": float(spk_wav.min()),
                    "max": float(spk_wav.max()),
                    "mean": float(spk_wav.mean()),
                    "std": float(spk_wav.std()),
                    "energy": float((spk_wav ** 2).mean()),
                    "rms": float((spk_wav ** 2).mean() ** 0.5)
                }
                stats["debug_speaker_waveforms"].append(output_debug)

                # Target audio analysis (from y_wav)
                target_wav = y_wav[0, k]  # First batch only
                target_debug = {
                    "speaker": k,
                    "shape": tuple(target_wav.shape),
                    "min": float(target_wav.min()),
                    "max": float(target_wav.max()),
                    "mean": float(target_wav.mean()),
                    "std": float(target_wav.std()),
                    "energy": float((target_wav ** 2).mean()),
                    "rms": float((target_wav ** 2).mean() ** 0.5),
                    "is_silent": float((target_wav ** 2).mean()) < 1e-6
                }
                stats["debug_target_waveforms"].append(target_debug)

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


def _compute_balanced_sisdr_loss(sisdr_per_spk, active_mask=None):
    """
    Compute speaker-balanced SI-SDR loss to prevent hierarchy collapse.

    Uses equal weighting to encourage balanced improvement across all speakers
    instead of focusing on worst performers, which can lead to training instability.

    Args:
        sisdr_per_spk: [B, K] per-speaker SI-SDR values (higher is better)
        active_mask: [B, K] boolean mask indicating which speakers are active (optional)

    Returns:
        balanced_sisdr: scalar balanced SI-SDR loss
    """
    if sisdr_per_spk.numel() == 1:
        # Single speaker case - no balancing needed
        return sisdr_per_spk.squeeze()

    if active_mask is not None:
        # Only compute loss for active speakers
        # Zero out inactive speakers and normalize by active count
        sisdr_masked = sisdr_per_spk * active_mask.float()  # [B, K]
        active_count = active_mask.sum(dim=-1, keepdim=True).float()  # [B, 1]
        active_count = torch.clamp(active_count, min=1.0)  # Avoid division by zero
        
        # Average over active speakers only
        balanced_sisdr_per_sample = sisdr_masked.sum(dim=-1) / active_count.squeeze(-1)  # [B]
    else:
        # Use equal weighting for all speakers to encourage balanced improvement
        # This prevents the model from abandoning good speakers to chase impossible cases
        equal_weights = torch.ones_like(sisdr_per_spk) / sisdr_per_spk.size(-1)  # [B, K]

        # Apply equal weighting - all speakers get equal optimization attention
        balanced_sisdr_per_sample = (sisdr_per_spk * equal_weights).sum(dim=-1)  # [B]

    return balanced_sisdr_per_sample.mean()  # Global balanced SI-SDR


def joint_loss(S_hat_c, Y_ref_c, batch, stft, weights=(0.0, 1.0), adaptive_weighting=False,
               amplitude_aware=True, amplitude_loss_weight=1.0):
    """
    Joint loss combining STFT separation loss with SI-SDR time-domain loss.

    Args:
        S_hat_c: [B, K, T, F] complex — model estimate per enrolled speaker
        Y_ref_c: [B, K, T, F] complex — clean reference STFT per speaker
        batch: dict containing target_all: [B, K, Tw] and target_lens_all: [B, K]
        stft: STFT wrapper for inverse transform
        weights: (stft_weight, sisdr_weight) for the two loss components
    Returns:
        loss (scalar), stats (dict)
    """
    assert torch.is_complex(S_hat_c) and torch.is_complex(Y_ref_c)
    B, K = S_hat_c.shape[:2]
    stft_weight, sisdr_weight = weights

    # 1. STFT-domain separation loss (uniform frequency weighting)
    error_mag = torch.abs(S_hat_c - Y_ref_c)  # [B, K, T, F]
    n_freqs = error_mag.shape[-1]
    freq_weights = torch.ones(n_freqs, device=S_hat_c.device)  # Uniform weighting to prevent high-freq suppression
    L_sep = (error_mag * freq_weights.view(1, 1, 1, -1)).mean()

    # 2. SI-SDR loss (time-domain separation quality)
    tgt_lens = batch["target_lens_all"].to(S_hat_c.device)  # [B, K] samples
    y_wav = batch["target_all"].to(S_hat_c.device)  # [B, K, Tw]
    s_hat_wav = stft.inverse(S_hat_c, lengths=tgt_lens)  # [B, K, Tw']

    # Match lengths for SI-SDR computation
    min_len = min(s_hat_wav.size(-1), y_wav.size(-1))
    s_hat_wav_matched = s_hat_wav[..., :min_len]
    y_wav_matched = y_wav[..., :min_len]

    # Compute SI-SDR per speaker: [B, K] -> higher is better
    sisdr_per_spk = _sisdr(s_hat_wav_matched, y_wav_matched)

    # Extract speaker activity mask from batch
    active_mask = batch.get("speaker_active_mask", None)  # [B, K] boolean
    if active_mask is not None:
        active_mask = active_mask.to(S_hat_c.device)

    # Convert to loss (negate since SI-SDR higher = better, but we minimize loss)
    sisdr_loss = -_compute_balanced_sisdr_loss(sisdr_per_spk, active_mask=active_mask)

    # 3. Combine STFT loss and SI-SDR loss
    loss = stft_weight * L_sep + sisdr_weight * sisdr_loss

    # Enhanced speaker separation diagnostics
    separation_stats = analyze_speaker_separation(s_hat_wav, y_wav)

    # Compute amplitude statistics for monitoring (kept for diagnostics)
    s_hat_rms_per_spk = torch.sqrt(torch.mean(s_hat_wav**2, dim=-1) + 1e-8)  # [B, K]
    y_ref_rms_per_spk = torch.sqrt(torch.mean(y_wav**2, dim=-1) + 1e-8)     # [B, K]
    s_hat_rms = torch.sqrt(torch.mean(s_hat_wav**2) + 1e-8)  # Global RMS
    y_ref_rms = torch.sqrt(torch.mean(y_wav**2) + 1e-8)      # Global RMS

    stats = {
        "loss": float(loss.detach()),
        "L_sep": float(L_sep.detach()),
        "sisdr_loss": float(sisdr_loss.detach()),
        "sisdr_db": float(_compute_balanced_sisdr_loss(sisdr_per_spk).detach()),  # Actual SI-SDR in dB
        "S_hat_c": tuple(S_hat_c.shape),
        "Y_ref_c": tuple(Y_ref_c.shape),
        "s_hat_wav": tuple(s_hat_wav.shape),
        "y_wav": tuple(y_wav.shape),

        # Amplitude monitoring (kept for diagnostics)
        "s_hat_rms": float(s_hat_rms.detach()),
        "s_hat_max_abs": float(torch.max(torch.abs(s_hat_wav)).detach()),
        "y_ref_rms": float(y_ref_rms.detach()),

        # Per-speaker amplitude analysis
        "s_hat_rms_per_spk": [float(s_hat_rms_per_spk[0, k].detach()) for k in range(K)] if B > 0 else [],
        "y_ref_rms_per_spk": [float(y_ref_rms_per_spk[0, k].detach()) for k in range(K)] if B > 0 else [],

        # Per-speaker SI-SDR values
        "sisdr_per_spk": [float(sisdr_per_spk[0, k].detach()) for k in range(K)] if B > 0 else [],

        # Loss component contributions
        "L_sep_contribution": float((stft_weight * L_sep).detach()),
        "sisdr_loss_contribution": float((sisdr_weight * sisdr_loss).detach()),
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
