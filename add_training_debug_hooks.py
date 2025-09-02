#!/usr/bin/env python3
"""
Quick debugging hooks to add to your existing training code.
These are lightweight additions that work with your current debug logging.
"""

import torch
import logging
import numpy as np
from typing import Dict, List


def debug_gridnet_lstm_states(model, log_interval: int = 50):
    """
    Add LSTM state analysis to your GridNet blocks.
    Call this ONCE during model initialization.
    """
    
    # Counter for logging frequency
    if not hasattr(model, '_debug_step_count'):
        model._debug_step_count = 0
    
    def create_lstm_debug_hook(block_idx: int, rnn_type: str):
        def hook(module, input, output):
            model._debug_step_count += 1
            
            if model._debug_step_count % log_interval == 0:
                hidden_states, final_states = output
                
                # Analyze hidden state statistics
                h_mean = hidden_states.mean().item()
                h_std = hidden_states.std().item()
                h_max = hidden_states.abs().max().item()
                
                # Analyze cell state statistics (for LSTM)
                if final_states is not None:
                    _, cell_states = final_states
                    c_mean = cell_states.mean().item()
                    c_std = cell_states.std().item()
                    c_max = cell_states.abs().max().item()
                else:
                    c_mean = c_std = c_max = 0.0
                
                logging.info(f"🧠 LSTM DEBUG Block {block_idx} ({rnn_type}):")
                logging.info(f"   Hidden: mean={h_mean:.4f}, std={h_std:.4f}, max={h_max:.4f}")
                logging.info(f"   Cell:   mean={c_mean:.4f}, std={c_std:.4f}, max={c_max:.4f}")
                
                # Detect LSTM saturation
                if h_std < 0.01:
                    logging.warning(f"⚠️  LSTM SATURATION in block {block_idx} {rnn_type}: std={h_std:.4f}")
                
                # Detect LSTM explosion
                if h_max > 10.0:
                    logging.warning(f"⚠️  LSTM EXPLOSION in block {block_idx} {rnn_type}: max={h_max:.4f}")
        
        return hook
    
    # Add hooks to all GridNet LSTM layers
    if hasattr(model, 'speaker_gridnets'):
        block_idx = 0
        for speaker_blocks in model.speaker_gridnets:
            for gridnet_block in speaker_blocks:
                gridnet_block.intra_rnn.register_forward_hook(
                    create_lstm_debug_hook(block_idx, "intra")
                )
                gridnet_block.inter_rnn.register_forward_hook(
                    create_lstm_debug_hook(block_idx, "inter")
                )
                block_idx += 1
    elif hasattr(model, 'gridnets'):
        for i, gridnet_block in enumerate(model.gridnets):
            gridnet_block.intra_rnn.register_forward_hook(
                create_lstm_debug_hook(i, "intra")
            )
            gridnet_block.inter_rnn.register_forward_hook(
                create_lstm_debug_hook(i, "inter")
            )
    
    logging.info("🔧 LSTM debug hooks registered for GridNet blocks")


def debug_attention_weights(model, log_interval: int = 50):
    """
    Add attention weight analysis to GridNet self-attention.
    Call this ONCE during model initialization.
    """
    
    def create_attention_debug_hook(block_idx: int):
        def hook(module, input, output):
            if hasattr(model, '_debug_step_count') and model._debug_step_count % log_interval == 0:
                
                # The output should be the attention-weighted features
                if isinstance(output, torch.Tensor) and output.dim() == 4:
                    B, C, T, F = output.shape
                    
                    # Analyze attention output distribution
                    attn_mean = output.mean().item()
                    attn_std = output.std().item()
                    attn_max = output.abs().max().item()
                    
                    logging.info(f"🎯 ATTENTION DEBUG Block {block_idx}:")
                    logging.info(f"   Output: mean={attn_mean:.4f}, std={attn_std:.4f}, max={attn_max:.4f}")
                    
                    # Detect attention collapse (all features become similar)
                    if attn_std < 0.01:
                        logging.warning(f"⚠️  ATTENTION COLLAPSE in block {block_idx}: std={attn_std:.4f}")
                    
                    # Analyze per-frequency attention (check if certain frequencies dominate)
                    freq_attention = output.mean(dim=(0, 2))  # [C, F] - average over batch and time
                    freq_variance = freq_attention.var(dim=0).mean().item()  # Variance across channels per freq
                    
                    logging.info(f"   Freq variance: {freq_variance:.4f}")
                    
                    if freq_variance < 0.01:
                        logging.warning(f"⚠️  FREQUENCY ATTENTION COLLAPSE in block {block_idx}")
        
        return hook
    
    # Add hooks to GridNet attention modules
    if hasattr(model, 'speaker_gridnets'):
        block_idx = 0
        for speaker_blocks in model.speaker_gridnets:
            for gridnet_block in speaker_blocks:
                if hasattr(gridnet_block, 'attn_concat_proj'):
                    gridnet_block.attn_concat_proj.register_forward_hook(
                        create_attention_debug_hook(block_idx)
                    )
                block_idx += 1
    elif hasattr(model, 'gridnets'):
        for i, gridnet_block in enumerate(model.gridnets):
            # Hook the final attention output
            if hasattr(gridnet_block, 'attn_concat_proj'):
                gridnet_block.attn_concat_proj.register_forward_hook(
                    create_attention_debug_hook(i)
                )
    
    logging.info("🔧 Attention debug hooks registered for GridNet blocks")


def analyze_speaker_channel_bias(outputs: torch.Tensor, step: int, log_interval: int = 25):
    """
    Analyze per-speaker channel outputs for bias detection.
    
    Args:
        outputs: [B, K, T, F] complex model outputs
        step: Current training step
        log_interval: How often to log analysis
    """
    
    if step % log_interval != 0:
        return
    
    if not torch.is_complex(outputs):
        logging.warning("⚠️  Expected complex outputs for channel bias analysis")
        return
    
    B, K, T, F = outputs.shape
    
    logging.info("📊 CHANNEL BIAS ANALYSIS:")
    
    # Compute per-speaker statistics
    speaker_stats = []
    
    for k in range(K):
        spk_output = outputs[:, k, :, :]  # [B, T, F]
        spk_magnitude = spk_output.abs()
        
        # Compute statistics
        rms = torch.sqrt(torch.mean(spk_magnitude**2)).item()
        peak = spk_magnitude.max().item()
        mean = spk_magnitude.mean().item()
        std = spk_magnitude.std().item()
        
        # Compute silent ratio (very low energy)
        silent_threshold = 1e-6
        silent_ratio = (spk_magnitude < silent_threshold).float().mean().item()
        
        speaker_stats.append({
            'speaker': k,
            'rms': rms,
            'peak': peak,
            'mean': mean,
            'std': std,
            'silent_ratio': silent_ratio
        })
        
        logging.info(f"   Speaker {k}: RMS={rms:.6f}, Peak={peak:.4f}, "
                    f"Mean={mean:.6f}, Silent={silent_ratio:.1%}")
        
        # Detect issues
        if silent_ratio > 0.3:
            logging.warning(f"⚠️  Speaker {k} HIGH SILENT RATIO: {silent_ratio:.1%}")
        
        if rms < 1e-5:
            logging.warning(f"⚠️  Speaker {k} VERY LOW RMS: {rms:.2e}")
    
    # Compute relative performance
    if len(speaker_stats) > 1:
        rms_values = [s['rms'] for s in speaker_stats]
        max_rms = max(rms_values)
        min_rms = min(rms_values)
        
        if max_rms > 0:
            rms_ratio = max_rms / (min_rms + 1e-8)
            
            logging.info(f"   RMS ratio (max/min): {rms_ratio:.2f}")
            
            if rms_ratio > 10.0:
                logging.warning(f"⚠️  HIGH RMS IMBALANCE: ratio={rms_ratio:.2f}")
                
                # Identify dominant and weak speakers
                max_idx = rms_values.index(max_rms)
                min_idx = rms_values.index(min_rms)
                
                logging.warning(f"⚠️  Dominant speaker: {max_idx} (RMS={max_rms:.6f})")
                logging.warning(f"⚠️  Weakest speaker: {min_idx} (RMS={min_rms:.6f})")


def debug_film_conditioning_per_speaker(film_layers: List, embeddings: torch.Tensor,
                                       K: int, step: int, log_interval: int = 50):
    """
    Analyze FiLM conditioning effectiveness per speaker.

    Args:
        film_layers: List of FiLM layer modules
        embeddings: [BK, C] speaker embeddings
        K: Number of speakers
        step: Current training step
        log_interval: How often to log analysis
    """

    if step % log_interval != 0:
        return

    BK, C = embeddings.shape
    B = BK // K
    
    if BK % K != 0:
        logging.warning(f"⚠️  Embeddings shape {embeddings.shape} not divisible by {K} speakers")
        return
    
    embeddings_reshaped = embeddings.view(B, K, C)  # [B, K, C]
    
    logging.info("🎭 FILM CONDITIONING PER-SPEAKER ANALYSIS:")
    
    # Analyze per-speaker embedding quality
    for k in range(K):
        spk_embeddings = embeddings_reshaped[:, k, :]  # [B, C]
        
        emb_mean = spk_embeddings.mean().item()
        emb_std = spk_embeddings.std().item()
        emb_norm = spk_embeddings.norm(dim=1).mean().item()
        
        logging.info(f"   Speaker {k} embedding: mean={emb_mean:.4f}, "
                    f"std={emb_std:.4f}, norm={emb_norm:.4f}")
        
        if emb_std < 0.01:
            logging.warning(f"⚠️  Speaker {k} EMBEDDING COLLAPSE: std={emb_std:.4f}")
    
    # Analyze inter-speaker similarity
    logging.info("Analyze inter-speaker similarity:")
    for i in range(K):
        for j in range(i+1, K):
            emb_i = embeddings_reshaped[0, i, :]  # First batch sample
            emb_j = embeddings_reshaped[0, j, :]
            
            similarity = torch.cosine_similarity(emb_i, emb_j, dim=0).item()
            
            logging.info(f"   Speaker {i} vs {j} similarity: {similarity:.4f}")
            
            if similarity > 0.95:
                logging.warning(f"⚠️  HIGH SIMILARITY: Speaker {i} vs {j} = {similarity:.4f}")
            elif similarity < -0.95:
                logging.warning(f"⚠️  ANTI-CORRELATION: Speaker {i} vs {j} = {similarity:.4f}")
    
    # Analyze FiLM parameter effectiveness
    for layer_idx, film_layer in enumerate(film_layers):
        # Get gamma and beta for each speaker
        gammas = film_layer.gamma_fc(embeddings)  # [BK, C]
        betas = film_layer.beta_fc(embeddings)   # [BK, C]
        
        gammas_reshaped = gammas.view(B, K, C)  # [B, K, C]
        betas_reshaped = betas.view(B, K, C)    # [B, K, C]
        
        logging.info(f"   FiLM Layer {layer_idx}:")
        
        for k in range(K):
            gamma_k = gammas_reshaped[:, k, :]  # [B, C]
            beta_k = betas_reshaped[:, k, :]    # [B, C]
            
            gamma_range = (gamma_k.max() - gamma_k.min()).item()
            beta_range = (beta_k.max() - beta_k.min()).item()
            
            logging.info(f"     Speaker {k}: gamma_range={gamma_range:.4f}, "
                        f"beta_range={beta_range:.4f}")
            
            if gamma_range < 0.1:
                logging.warning(f"⚠️  Speaker {k} GAMMA collapse in layer {layer_idx}: "
                              f"range={gamma_range:.4f}")
            
            if beta_range < 0.1:
                logging.warning(f"⚠️  Speaker {k} BETA collapse in layer {layer_idx}: "
                              f"range={beta_range:.4f}")


# Example integration code
def add_debug_to_training_loop():
    """
    Example of how to integrate these debugging functions into your training loop.
    """
    
    example_code = '''
# Add to your training script initialization:
debug_gridnet_lstm_states(model, log_interval=50)
debug_attention_weights(model, log_interval=50)

# Add to your training loop:
for step, batch in enumerate(dataloader):
    # ... your existing forward pass ...
    outputs = model(mixture, enrollments, lengths)
    
    # Add debugging analysis
    analyze_speaker_channel_bias(outputs, step, log_interval=25)
    debug_film_conditioning_per_speaker(model.fusions, speaker_embeddings, step, log_interval=50)
    
    # ... your existing backward pass and optimization ...
    '''
    
    print("🔧 Integration Example:")
    print(example_code)


if __name__ == "__main__":
    print("🔬 Training Debug Hooks for Speaker Hierarchy Collapse")
    print("="*60)
    print("This module provides lightweight debugging hooks to add to your existing training code.")
    print()
    add_debug_to_training_loop()