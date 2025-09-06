import torch
import torch.nn.functional as F
import logging
from typing import Dict, Tuple, Optional


def _sisdr(x, s, eps=1e-8):
    """Scale-Invariant Signal-to-Distortion Ratio."""
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


def contrastive_multi_speaker_loss(
    s_hat_wav: torch.Tensor,
    y_wav: torch.Tensor, 
    context_info: Dict,
    loss_weights: Dict = None,
    temperature: float = 0.1,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, Dict]:
    """
    Enhanced contrastive multi-speaker loss that encourages:
    1. High quality separation (SI-SDR)
    2. Speaker distinctiveness (contrastive learning)
    3. Context-aware difficulty adaptation
    4. Inter-speaker suppression
    
    Args:
        s_hat_wav: [B, K, T] predicted separated waveforms
        y_wav: [B, K, T] target separated waveforms
        context_info: Dict containing multi-speaker context from aux encoder
        loss_weights: Dict with loss component weights
        temperature: Temperature for contrastive learning
        eps: Small epsilon for numerical stability
        
    Returns:
        total_loss: Combined loss
        loss_components: Dict with individual loss components and metrics
    """
    if loss_weights is None:
        loss_weights = {
            'sisdr': 1.0,
            'contrastive': 0.5, 
            'separation': 0.3,
            'distinctiveness': 0.2
        }
    
    B, K, T = s_hat_wav.shape
    device = s_hat_wav.device
    
    # 1. Base SI-SDR Loss (primary signal quality)
    sisdr_loss = -_sisdr(s_hat_wav, y_wav).mean()
    
    # 2. Contrastive Speaker Separation Loss
    contrastive_loss = _contrastive_speaker_loss(s_hat_wav, y_wav, temperature, eps)
    
    # 3. Context-Aware Separation Loss  
    separation_loss = _context_aware_separation_loss(s_hat_wav, y_wav, context_info, eps)
    
    # 4. Speaker Distinctiveness Loss
    distinctiveness_loss = _speaker_distinctiveness_loss(s_hat_wav, context_info, eps)
    
    # 5. Difficulty-Adaptive Weighting
    if context_info['num_speakers'] > 1:
        difficulty = context_info['separation_difficulty']  # [B]
        difficulty_weight = 1.0 + 0.5 * difficulty  # Higher weight for harder cases
        difficulty_weight = difficulty_weight.mean()  # Average across batch
    else:
        difficulty_weight = torch.tensor(1.0, device=device)
    
    # Combine losses with adaptive weighting
    total_loss = (
        loss_weights['sisdr'] * sisdr_loss +
        loss_weights['contrastive'] * contrastive_loss * difficulty_weight +
        loss_weights['separation'] * separation_loss * difficulty_weight +
        loss_weights['distinctiveness'] * distinctiveness_loss
    )
    
    # Compute additional metrics for monitoring
    metrics = _compute_separation_metrics(s_hat_wav, y_wav, context_info)
    
    loss_components = {
        'total_loss': total_loss,
        'sisdr_loss': sisdr_loss,
        'contrastive_loss': contrastive_loss,
        'separation_loss': separation_loss,
        'distinctiveness_loss': distinctiveness_loss,
        'difficulty_weight': difficulty_weight,
        **metrics
    }
    
    return total_loss, loss_components


def _contrastive_speaker_loss(s_hat_wav: torch.Tensor, y_wav: torch.Tensor, temperature: float, eps: float) -> torch.Tensor:
    """
    Contrastive loss that encourages:
    - High similarity between predicted and target for same speaker (positive pairs)
    - Low similarity between predicted and target for different speakers (negative pairs)
    """
    B, K, T = s_hat_wav.shape
    
    if K <= 1:
        return torch.tensor(0.0, device=s_hat_wav.device)
    
    contrastive_losses = []
    
    for b in range(B):
        # For each batch, create positive and negative pairs
        batch_losses = []
        
        for k in range(K):
            # Positive pair: predicted_k vs target_k
            pred_k = s_hat_wav[b, k]  # [T]
            target_k = y_wav[b, k]   # [T]
            
            # Compute similarities with all targets
            similarities = []
            for j in range(K):
                target_j = y_wav[b, j]  # [T]
                
                # Cosine similarity between predicted_k and target_j
                pred_norm = pred_k / (pred_k.norm() + eps)
                target_norm = target_j / (target_j.norm() + eps)
                sim = torch.sum(pred_norm * target_norm)
                similarities.append(sim / temperature)
            
            similarities = torch.stack(similarities)  # [K]
            
            # Contrastive loss: maximize similarity to correct target (k), minimize to others
            positive_sim = similarities[k]
            contrastive_loss_k = -positive_sim + torch.logsumexp(similarities, dim=0)
            batch_losses.append(contrastive_loss_k)
        
        if batch_losses:
            batch_loss = torch.stack(batch_losses).mean()
            contrastive_losses.append(batch_loss)
    
    if contrastive_losses:
        return torch.stack(contrastive_losses).mean()
    else:
        return torch.tensor(0.0, device=s_hat_wav.device)


def _context_aware_separation_loss(s_hat_wav: torch.Tensor, y_wav: torch.Tensor, context_info: Dict, eps: float) -> torch.Tensor:
    """
    Context-aware loss that penalizes cross-speaker interference based on similarity.
    Higher penalty when speakers are more similar (harder to separate).
    """
    B, K, T = s_hat_wav.shape
    device = s_hat_wav.device
    
    if K <= 1 or context_info['num_speakers'] <= 1:
        return torch.tensor(0.0, device=device)
    
    speaker_similarities = context_info['speaker_similarities']  # [B, K, K]
    separation_losses = []
    
    for b in range(B):
        batch_losses = []
        
        for k1 in range(K):
            for k2 in range(K):
                if k1 == k2:
                    continue  # Skip self-comparison
                
                # Get predicted output for speaker k1
                pred_k1 = s_hat_wav[b, k1]  # [T]
                
                # Get target for speaker k2 (should be minimally present in pred_k1)
                target_k2 = y_wav[b, k2]   # [T]
                
                # Measure unwanted presence of speaker k2 in output k1
                # Use correlation as a measure of interference
                pred_norm = pred_k1 - pred_k1.mean()
                target_norm = target_k2 - target_k2.mean()
                
                correlation = torch.sum(pred_norm * target_norm) / (
                    torch.sqrt(torch.sum(pred_norm**2)) * 
                    torch.sqrt(torch.sum(target_norm**2)) + eps
                )
                
                # Weight by speaker similarity - more similar speakers should have stronger penalty
                similarity_weight = speaker_similarities[b, k1, k2]
                
                # Penalty for high correlation (interference)
                interference_penalty = torch.clamp(correlation, min=0.0) * similarity_weight
                batch_losses.append(interference_penalty)
        
        if batch_losses:
            batch_loss = torch.stack(batch_losses).mean()
            separation_losses.append(batch_loss)
    
    if separation_losses:
        return torch.stack(separation_losses).mean()
    else:
        return torch.tensor(0.0, device=device)


def _speaker_distinctiveness_loss(s_hat_wav: torch.Tensor, context_info: Dict, eps: float) -> torch.Tensor:
    """
    Encourages the separated outputs to be maximally distinct from each other.
    This helps prevent speaker assignment errors and promotes clean separation.
    """
    B, K, T = s_hat_wav.shape
    device = s_hat_wav.device
    
    if K <= 1:
        return torch.tensor(0.0, device=device)
    
    distinctiveness_losses = []
    
    for b in range(B):
        batch_similarities = []
        
        # Compute pairwise similarities between separated outputs
        for k1 in range(K):
            for k2 in range(k1 + 1, K):  # Only upper triangle
                pred_k1 = s_hat_wav[b, k1]  # [T]
                pred_k2 = s_hat_wav[b, k2]  # [T]
                
                # Normalize
                pred_k1_norm = pred_k1 / (pred_k1.norm() + eps)
                pred_k2_norm = pred_k2 / (pred_k2.norm() + eps)
                
                # Cosine similarity
                similarity = torch.sum(pred_k1_norm * pred_k2_norm)
                batch_similarities.append(similarity)
        
        if batch_similarities:
            # Penalty for high similarity (want low similarity = high distinctiveness)
            avg_similarity = torch.stack(batch_similarities).mean()
            distinctiveness_loss = torch.clamp(avg_similarity, min=0.0)  # Only penalize positive correlations
            distinctiveness_losses.append(distinctiveness_loss)
    
    if distinctiveness_losses:
        return torch.stack(distinctiveness_losses).mean()
    else:
        return torch.tensor(0.0, device=device)


def _compute_separation_metrics(s_hat_wav: torch.Tensor, y_wav: torch.Tensor, context_info: Dict) -> Dict:
    """Compute additional metrics for monitoring separation quality."""
    B, K, T = s_hat_wav.shape
    
    with torch.no_grad():
        # Base SI-SDR per speaker
        sisdr_per_speaker = _sisdr(s_hat_wav, y_wav)  # [B, K]
        
        metrics = {
            'sisdr_mean': sisdr_per_speaker.mean(),
            'sisdr_std': sisdr_per_speaker.std(),
            'sisdr_min': sisdr_per_speaker.min(),
            'sisdr_max': sisdr_per_speaker.max(),
        }
        
        if K > 1:
            # Cross-speaker correlation (lower is better)
            correlations = []
            for b in range(min(B, 2)):  # Analyze first 2 batches for efficiency
                for k1 in range(K):
                    for k2 in range(k1 + 1, K):
                        pred_k1 = s_hat_wav[b, k1]
                        pred_k2 = s_hat_wav[b, k2]
                        
                        # Pearson correlation
                        mean_k1 = pred_k1.mean()
                        mean_k2 = pred_k2.mean()
                        numerator = ((pred_k1 - mean_k1) * (pred_k2 - mean_k2)).sum()
                        denominator = (
                            ((pred_k1 - mean_k1) ** 2).sum().sqrt() *
                            ((pred_k2 - mean_k2) ** 2).sum().sqrt() + 1e-8
                        )
                        corr = (numerator / denominator).item()
                        correlations.append(abs(corr))  # Use absolute correlation
            
            if correlations:
                metrics.update({
                    'cross_speaker_corr_mean': sum(correlations) / len(correlations),
                    'cross_speaker_corr_max': max(correlations),
                })
            
            # Speaker energy balance
            energies = []
            for b in range(min(B, 1)):  # First batch only
                batch_energies = []
                for k in range(K):
                    energy = (s_hat_wav[b, k] ** 2).mean().item()
                    batch_energies.append(energy)
                energies.append(batch_energies)
            
            if energies:
                flat_energies = [e for batch in energies for e in batch]
                energy_std = torch.tensor(flat_energies).std().item() if len(flat_energies) > 1 else 0.0
                energy_ratio = max(flat_energies) / (min(flat_energies) + 1e-8) if flat_energies else 1.0
                
                metrics.update({
                    'speaker_energy_std': energy_std,
                    'speaker_energy_ratio': energy_ratio,
                })
            
            # Context information
            if 'separation_difficulty' in context_info:
                metrics['separation_difficulty_mean'] = context_info['separation_difficulty'].mean().item()
            
            if 'speaker_similarities' in context_info:
                similarities = context_info['speaker_similarities']
                # Exclude diagonal (self-similarities)
                off_diagonal = similarities * (1 - torch.eye(K, device=similarities.device))
                metrics['speaker_similarity_mean'] = off_diagonal.sum() / (K * (K - 1)) if K > 1 else 0.0
    
    return metrics


def log_separation_metrics(metrics: Dict, epoch: int, batch_idx: int, split: str = "train"):
    """Log separation metrics in a readable format."""
    logging.info(f"=== CONTEXT-AWARE SEPARATION ANALYSIS [{split.upper()}] E{epoch}B{batch_idx} ===")
    
    # Base metrics
    logging.info(f"🎯 SI-SDR Quality: {metrics.get('sisdr_mean', 0):.3f} ± {metrics.get('sisdr_std', 0):.3f} dB")
    logging.info(f"📊 SI-SDR Range: [{metrics.get('sisdr_min', 0):.3f}, {metrics.get('sisdr_max', 0):.3f}] dB")
    
    # Multi-speaker specific metrics
    if 'cross_speaker_corr_mean' in metrics:
        logging.info(f"📈 Cross-Speaker Correlation: {metrics['cross_speaker_corr_mean']:.4f} (max: {metrics.get('cross_speaker_corr_max', 0):.4f})")
    
    if 'speaker_energy_ratio' in metrics:
        logging.info(f"⚖️  Energy Balance - Std: {metrics.get('speaker_energy_std', 0):.6f}, Ratio: {metrics['speaker_energy_ratio']:.2f}")
    
    # Context metrics
    if 'separation_difficulty_mean' in metrics:
        logging.info(f"🎭 Separation Difficulty: {metrics['separation_difficulty_mean']:.4f}")
    
    if 'speaker_similarity_mean' in metrics:
        logging.info(f"🔗 Speaker Similarity: {metrics['speaker_similarity_mean']:.4f}")
    
    # Loss components
    loss_components = ['sisdr_loss', 'contrastive_loss', 'separation_loss', 'distinctiveness_loss']
    loss_info = []
    for component in loss_components:
        if component in metrics:
            loss_info.append(f"{component.replace('_loss', '').upper()}: {metrics[component]:.4f}")
    
    if loss_info:
        logging.info(f"📉 Loss Components: {' | '.join(loss_info)}")
    
    if 'difficulty_weight' in metrics:
        logging.info(f"🎲 Difficulty Weight: {metrics['difficulty_weight']:.3f}")