#!/usr/bin/env python3
"""
Comprehensive debugging suite for Speaker Hierarchy Collapse analysis in MCxTFGridNet.

This script provides detailed monitoring of:
1. LSTM gradient flow in GridNet blocks
2. Self-attention weight distribution  
3. Speaker embedding evolution
4. FiLM conditioning effectiveness
5. Channel-specific output analysis
"""

import torch
import torch.nn as nn
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class HierarchyCollapseDebugger:
    """
    Comprehensive debugging class for analyzing speaker hierarchy collapse
    during MCxTFGridNet training.
    """
    
    def __init__(self, model, log_interval: int = 50):
        self.model = model
        self.log_interval = log_interval
        self.step_count = 0
        
        # Storage for analysis
        self.gradient_history = defaultdict(list)
        self.attention_history = defaultdict(list)
        self.embedding_history = defaultdict(list)
        self.output_history = defaultdict(list)
        self.film_history = defaultdict(list)
        
        # Register hooks
        self._register_gradient_hooks()
        self._register_attention_hooks()
        
        logging.info("🔬 HierarchyCollapseDebugger initialized")
    
    def _register_gradient_hooks(self):
        """Register backward hooks for gradient flow analysis"""
        
        def create_gradient_hook(name, speaker_idx=None):
            def hook(module, grad_input, grad_output):
                if self.step_count % self.log_interval == 0 and grad_output[0] is not None:
                    grad = grad_output[0]
                    
                    # Gradient statistics
                    grad_stats = {
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'norm': grad.norm().item(),
                        'max_abs': grad.abs().max().item(),
                        'step': self.step_count
                    }
                    
                    if speaker_idx is not None:
                        key = f"{name}_speaker_{speaker_idx}"
                    else:
                        key = name
                        
                    self.gradient_history[key].append(grad_stats)
                    
                    # Log critical gradient flow issues
                    if grad_stats['norm'] < 1e-8:
                        logging.warning(f"⚠️  VANISHING GRADIENTS in {key}: norm={grad_stats['norm']:.2e}")
                    elif grad_stats['norm'] > 100:
                        logging.warning(f"⚠️  EXPLODING GRADIENTS in {key}: norm={grad_stats['norm']:.2e}")
            
            return hook
        
        # Hook GridNet LSTM layers for gradient flow analysis
        for i, gridnet_block in enumerate(self.model.gridnets):
            # Intra-RNN (bidirectional LSTM)
            gridnet_block.intra_rnn.register_backward_hook(
                create_gradient_hook(f"gridnet_{i}_intra_lstm")
            )
            
            # Inter-RNN (unidirectional LSTM) 
            gridnet_block.inter_rnn.register_backward_hook(
                create_gradient_hook(f"gridnet_{i}_inter_lstm")
            )
        
        # Hook deconv layer for per-speaker channel analysis
        self.model.deconv.register_backward_hook(
            create_gradient_hook("deconv_output")
        )
        
        # Hook FiLM layers
        for i, film_layer in enumerate(self.model.fusions):
            film_layer.gamma_fc.register_backward_hook(
                create_gradient_hook(f"film_{i}_gamma")
            )
            film_layer.beta_fc.register_backward_hook(
                create_gradient_hook(f"film_{i}_beta")
            )
    
    def _register_attention_hooks(self):
        """Register forward hooks for attention weight analysis"""
        
        def create_attention_hook(block_idx):
            def hook(module, input, output):
                if self.step_count % self.log_interval == 0:
                    # Attention weights from GridNet self-attention
                    # This captures the attention weights after softmax
                    attn_weights = output  # This should be attention weights
                    
                    if isinstance(attn_weights, torch.Tensor) and attn_weights.dim() >= 3:
                        # Analyze attention distribution
                        attention_stats = {
                            'entropy': self._compute_attention_entropy(attn_weights),
                            'max_weight': attn_weights.max().item(),
                            'concentration': self._compute_attention_concentration(attn_weights),
                            'step': self.step_count
                        }
                        
                        self.attention_history[f"gridnet_{block_idx}_attention"].append(attention_stats)
                        
                        # Log attention concentration issues
                        if attention_stats['concentration'] > 0.8:
                            logging.warning(f"⚠️  HIGH ATTENTION CONCENTRATION in GridNet block {block_idx}: {attention_stats['concentration']:.3f}")
            
            return hook
        
        # Hook attention modules in GridNet blocks
        for i, gridnet_block in enumerate(self.model.gridnets):
            # Hook the attention computation in GridNet blocks
            # Note: The exact module depends on the GridNet implementation
            if hasattr(gridnet_block, 'attn_concat_proj'):
                gridnet_block.attn_concat_proj.register_forward_hook(
                    create_attention_hook(i)
                )
    
    def analyze_lstm_states(self, gridnet_outputs: List[torch.Tensor]):
        """
        Analyze LSTM hidden states for bias toward specific speakers.
        
        Args:
            gridnet_outputs: List of outputs from each GridNet block [B, C, T, F]
        """
        if self.step_count % self.log_interval != 0:
            return
        
        for block_idx, output in enumerate(gridnet_outputs):
            # Analyze feature activation patterns
            B, C, T, F = output.shape
            
            # Compute activation statistics
            activation_stats = {
                'mean_activation': output.mean().item(),
                'std_activation': output.std().item(),
                'sparsity': (output.abs() < 1e-6).float().mean().item(),
                'step': self.step_count
            }
            
            # Check for feature collapse (all features becoming similar)
            if activation_stats['std_activation'] < 0.01:
                logging.warning(f"⚠️  FEATURE COLLAPSE in GridNet block {block_idx}: std={activation_stats['std_activation']:.4f}")
            
            # Store for analysis
            self.gradient_history[f"gridnet_{block_idx}_activations"].append(activation_stats)
    
    def analyze_speaker_embeddings(self, embeddings: torch.Tensor, speaker_ids: List[int]):
        """
        Analyze speaker embedding quality and discriminability.
        
        Args:
            embeddings: [BK, C] speaker embeddings
            speaker_ids: List of speaker IDs for the batch
        """
        if self.step_count % self.log_interval != 0:
            return
        
        B = len(speaker_ids)
        K = len(set(speaker_ids))
        C = embeddings.shape[1]
        
        # Reshape embeddings per speaker
        embeddings_per_speaker = embeddings.view(B, K, C)  # [B, K, C]
        
        embedding_analysis = {}
        
        for k in range(K):
            spk_embeddings = embeddings_per_speaker[:, k, :]  # [B, C]
            
            # Embedding quality metrics
            embedding_stats = {
                'mean': spk_embeddings.mean().item(),
                'std': spk_embeddings.std().item(),
                'norm': spk_embeddings.norm(dim=1).mean().item(),
                'step': self.step_count
            }
            
            # Check for embedding collapse
            if embedding_stats['std'] < 0.01:
                logging.warning(f"⚠️  SPEAKER {k} EMBEDDING COLLAPSE: std={embedding_stats['std']:.4f}")
            
            self.embedding_history[f"speaker_{k}"].append(embedding_stats)
            embedding_analysis[f"speaker_{k}"] = embedding_stats
        
        # Inter-speaker similarity analysis
        if K > 1:
            similarities = {}
            for i in range(K):
                for j in range(i+1, K):
                    emb_i = embeddings_per_speaker[0, i, :]  # First batch sample
                    emb_j = embeddings_per_speaker[0, j, :]
                    
                    similarity = torch.cosine_similarity(emb_i, emb_j, dim=0).item()
                    similarities[f"speaker_{i}_vs_{j}"] = similarity
                    
                    if similarity > 0.95:
                        logging.warning(f"⚠️  HIGH SPEAKER SIMILARITY: Speaker {i} vs {j} = {similarity:.3f}")
                    elif similarity < -0.95:
                        logging.warning(f"⚠️  SPEAKER ANTI-CORRELATION: Speaker {i} vs {j} = {similarity:.3f}")
            
            # Store similarity analysis
            for key, value in similarities.items():
                self.embedding_history[key].append({
                    'similarity': value,
                    'step': self.step_count
                })
    
    def analyze_film_conditioning(self, film_outputs: List[Tuple[torch.Tensor, torch.Tensor]]):
        """
        Analyze FiLM layer conditioning effectiveness.
        
        Args:
            film_outputs: List of (gamma, beta) pairs from each FiLM layer
        """
        if self.step_count % self.log_interval != 0:
            return
        
        for layer_idx, (gamma, beta) in enumerate(film_outputs):
            # Analyze FiLM parameter variation
            gamma_stats = {
                'mean_abs': gamma.abs().mean().item(),
                'std': gamma.std().item(),
                'range': (gamma.max() - gamma.min()).item(),
                'step': self.step_count
            }
            
            beta_stats = {
                'mean_abs': beta.abs().mean().item(),
                'std': beta.std().item(),
                'range': (beta.max() - beta.min()).item(),
                'step': self.step_count
            }
            
            # Check for conditioning collapse
            if gamma_stats['range'] < 0.1:
                logging.warning(f"⚠️  FiLM GAMMA collapse in layer {layer_idx}: range={gamma_stats['range']:.4f}")
            
            if beta_stats['range'] < 0.1:
                logging.warning(f"⚠️  FiLM BETA collapse in layer {layer_idx}: range={beta_stats['range']:.4f}")
            
            # Store analysis
            self.film_history[f"layer_{layer_idx}_gamma"].append(gamma_stats)
            self.film_history[f"layer_{layer_idx}_beta"].append(beta_stats)
    
    def analyze_channel_outputs(self, model_outputs: torch.Tensor, n_speakers: int = 3):
        """
        Analyze per-speaker channel outputs for hierarchy collapse.
        
        Args:
            model_outputs: [B, K, T, F] complex outputs
            n_speakers: Number of target speakers
        """
        if self.step_count % self.log_interval != 0:
            return
        
        B, K, T, F = model_outputs.shape
        
        for k in range(min(K, n_speakers)):
            spk_output = model_outputs[:, k, :, :]  # [B, T, F]
            
            # Convert to magnitude if complex
            if torch.is_complex(spk_output):
                spk_magnitude = spk_output.abs()
            else:
                spk_magnitude = spk_output.abs()
            
            # Output quality metrics
            output_stats = {
                'rms': torch.sqrt(torch.mean(spk_magnitude**2)).item(),
                'max_magnitude': spk_magnitude.max().item(),
                'mean_magnitude': spk_magnitude.mean().item(),
                'silent_ratio': (spk_magnitude < 1e-6).float().mean().item(),
                'step': self.step_count
            }
            
            # Detect output collapse
            if output_stats['silent_ratio'] > 0.3:
                logging.warning(f"⚠️  SPEAKER {k} OUTPUT COLLAPSE: {output_stats['silent_ratio']:.1%} silent")
            
            if output_stats['rms'] < 1e-4:
                logging.warning(f"⚠️  SPEAKER {k} WEAK OUTPUT: RMS={output_stats['rms']:.2e}")
            
            # Store analysis
            self.output_history[f"speaker_{k}"].append(output_stats)
    
    def step(self):
        """Increment step counter for logging"""
        self.step_count += 1
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention weights (higher = more distributed)"""
        # Flatten attention weights and normalize
        attn_flat = attention_weights.flatten()
        attn_prob = torch.softmax(attn_flat, dim=0)
        
        # Compute entropy
        entropy = -torch.sum(attn_prob * torch.log(attn_prob + 1e-8))
        return entropy.item()
    
    def _compute_attention_concentration(self, attention_weights: torch.Tensor) -> float:
        """Compute attention concentration (0 = uniform, 1 = concentrated)"""
        attn_flat = attention_weights.flatten()
        attn_prob = torch.softmax(attn_flat, dim=0)
        
        # Concentration as max probability
        concentration = attn_prob.max().item()
        return concentration
    
    def generate_collapse_report(self, save_path: str = "hierarchy_collapse_report.md"):
        """Generate comprehensive analysis report"""
        
        report_lines = [
            "# Speaker Hierarchy Collapse Analysis Report",
            f"## Generated at step {self.step_count}",
            "",
            "## Executive Summary",
        ]
        
        # Analyze gradient flow issues
        gradient_issues = []
        for key, history in self.gradient_history.items():
            if history:
                latest = history[-1]
                if latest['norm'] < 1e-6:
                    gradient_issues.append(f"- **{key}**: Vanishing gradients (norm={latest['norm']:.2e})")
                elif latest['norm'] > 50:
                    gradient_issues.append(f"- **{key}**: Exploding gradients (norm={latest['norm']:.2e})")
        
        if gradient_issues:
            report_lines.extend([
                "",
                "### 🚨 Gradient Flow Issues Detected:",
                *gradient_issues
            ])
        
        # Analyze embedding collapse
        embedding_issues = []
        for key, history in self.embedding_history.items():
            if 'speaker_' in key and 'vs' not in key and history:
                latest = history[-1]
                if latest['std'] < 0.01:
                    embedding_issues.append(f"- **{key}**: Embedding collapse (std={latest['std']:.4f})")
        
        if embedding_issues:
            report_lines.extend([
                "",
                "### 🎤 Speaker Embedding Issues:",
                *embedding_issues
            ])
        
        # Analyze output collapse
        output_issues = []
        for key, history in self.output_history.items():
            if 'speaker_' in key and history:
                latest = history[-1]
                if latest['silent_ratio'] > 0.3:
                    output_issues.append(f"- **{key}**: Output collapse ({latest['silent_ratio']:.1%} silent)")
                elif latest['rms'] < 1e-4:
                    output_issues.append(f"- **{key}**: Weak output (RMS={latest['rms']:.2e})")
        
        if output_issues:
            report_lines.extend([
                "",
                "### 📊 Output Quality Issues:",
                *output_issues
            ])
        
        # Save report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logging.info(f"📄 Collapse analysis report saved to {save_path}")


def integrate_debugger_with_training():
    """
    Example integration with training loop.
    Add this to your training script.
    """
    
    # Initialize debugger
    # debugger = HierarchyCollapseDebugger(model, log_interval=25)
    
    # In training loop:
    """
    for batch_idx, batch in enumerate(dataloader):
        # ... normal forward pass ...
        outputs = model(mixture, enrollments, lengths)
        
        # Debug analysis
        debugger.step()
        debugger.analyze_channel_outputs(outputs, n_speakers=3)
        debugger.analyze_speaker_embeddings(speaker_embeddings, speaker_ids)
        # debugger.analyze_film_conditioning(film_outputs)  # if available
        
        # ... backward pass and optimization ...
        loss.backward()
        
        # Generate report periodically
        if batch_idx % 500 == 0:
            debugger.generate_collapse_report(f"collapse_report_step_{batch_idx}.md")
    """


if __name__ == "__main__":
    print("🔬 Speaker Hierarchy Collapse Debugger")
    print("="*50)
    print("This module provides comprehensive debugging tools for analyzing")
    print("speaker hierarchy collapse in MCxTFGridNet training.")
    print()
    print("Integration steps:")
    print("1. Import this module in your training script")
    print("2. Initialize HierarchyCollapseDebugger with your model")  
    print("3. Call debugger methods in your training loop")
    print("4. Generate periodic analysis reports")
    print()
    print("See integrate_debugger_with_training() for example usage.")