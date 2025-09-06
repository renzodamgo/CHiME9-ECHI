import math
from typing import Tuple, Any, Optional
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

# Import base components from UniversalMCxTFGridNet
from .UniversalMCxTFGridNet import (
    LayerNormalization, GridNetV3Block, SpeakerConditionalConv2d, 
    FiLM as BaseFiLM, HALF_PRECISION_DTYPES
)


class MultiSpeakerContextAuxEncoder(nn.Module):
    """
    Multi-Speaker Context-Aware Auxiliary Encoder.
    
    Unlike the standard auxiliary encoder that processes speakers independently,
    this encoder understands the full multi-speaker context:
    
    1. Individual speaker characteristics (like standard aux encoder)
    2. Inter-speaker relationships and conflicts  
    3. Separation difficulty estimation
    4. Dynamic speaker count adaptation
    
    Key Features:
    - Cross-attention between speaker enrollments
    - Context-aware embeddings that understand "separate X from Y,Z"
    - Single-speaker compatible (graceful degradation)
    - Progressive speaker difficulty estimation
    """
    
    def __init__(self, emb_dim=48, num_layers=3, n_head=4, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.n_head = n_head
        
        # Individual speaker processing (maintains compatibility)
        self.individual_encoder = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, kernel_size=(3, 3), padding=(1, 1)),
            LayerNormalization(emb_dim),
            nn.PReLU(),
            nn.Conv2d(emb_dim, emb_dim, kernel_size=(3, 3), padding=(1, 1)),
            LayerNormalization(emb_dim),
            nn.PReLU()
        )
        
        # Global average pooling to get speaker embeddings
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Multi-speaker context understanding
        self.context_attention_layers = nn.ModuleList([
            MultiSpeakerAttentionBlock(emb_dim, n_head, dropout) 
            for _ in range(num_layers)
        ])
        
        # Speaker relationship analysis
        self.relationship_analyzer = SpeakerRelationshipAnalyzer(emb_dim)
        
        # Separation difficulty estimator
        self.difficulty_estimator = SeparationDifficultyEstimator(emb_dim)
        
        # Context-aware embedding projection
        self.context_projection = nn.Linear(emb_dim * 2, emb_dim)  # Individual + context
        
        # Compatibility projection for single-speaker mode
        self.single_speaker_projection = nn.Linear(emb_dim, emb_dim)
        
        logging.info("🎭 MultiSpeakerContextAuxEncoder initialized")
        logging.info(f"   Embedding dimension: {emb_dim}")
        logging.info(f"   Context attention layers: {num_layers}")
        logging.info(f"   Attention heads: {n_head}")
        
    def forward(self, spk_features, spk_lens, batch_size, num_speakers):
        """
        Process speaker features with multi-speaker context awareness.
        
        Args:
            spk_features: [B, C, T, F] or [B, K, C, T, F] speaker features
            spk_lens: [B] or [B, K] speaker lengths
            batch_size: Batch size
            num_speakers: Number of speakers (K)
            
        Returns:
            speaker_embeddings: [B, K, C] context-aware speaker embeddings
            context_info: Dict with separation context information
        """
        if num_speakers == 1:
            # Single speaker mode - use individual processing only
            return self._process_single_speaker(spk_features, spk_lens, batch_size)
        else:
            # Multi-speaker mode - full context processing
            return self._process_multi_speakers(spk_features, spk_lens, batch_size, num_speakers)
    
    def _process_single_speaker(self, spk_features, spk_lens, batch_size):
        """Process single speaker (maintains compatibility with existing enhancement)."""
        # spk_features: [B, C, T, F]
        individual_features = self.individual_encoder(spk_features)  # [B, C, T, F]
        pooled = self.global_pool(individual_features).squeeze(-1).squeeze(-1)  # [B, C]
        
        # Apply single-speaker projection
        embedding = self.single_speaker_projection(pooled)  # [B, C]
        
        # Expand to match multi-speaker format: [B, 1, C]
        speaker_embeddings = embedding.unsqueeze(1)
        
        context_info = {
            'num_speakers': 1,
            'separation_difficulty': torch.zeros(batch_size, device=spk_features.device),
            'speaker_similarities': torch.zeros(batch_size, 1, 1, device=spk_features.device),
            'context_weights': torch.ones(batch_size, 1, device=spk_features.device)
        }
        
        return speaker_embeddings, context_info
    
    def _process_multi_speakers(self, spk_features, spk_lens, batch_size, num_speakers):
        """Process multiple speakers with full context awareness."""
        # spk_features: [B, K, C, T, F]
        B, K, C, T, F = spk_features.shape
        
        # Process each speaker individually first
        individual_embeddings = []
        for k in range(K):
            spk_k = spk_features[:, k]  # [B, C, T, F]
            individual_features = self.individual_encoder(spk_k)  # [B, C, T, F]
            pooled = self.global_pool(individual_features).squeeze(-1).squeeze(-1)  # [B, C]
            individual_embeddings.append(pooled)
        
        individual_embeddings = torch.stack(individual_embeddings, dim=1)  # [B, K, C]
        
        # Apply cross-speaker attention to understand relationships
        context_embeddings = individual_embeddings
        for attention_layer in self.context_attention_layers:
            context_embeddings = attention_layer(context_embeddings, spk_lens)
        
        # Analyze speaker relationships
        speaker_similarities = self.relationship_analyzer(individual_embeddings)  # [B, K, K]
        
        # Estimate separation difficulty
        separation_difficulty = self.difficulty_estimator(
            individual_embeddings, speaker_similarities
        )  # [B]
        
        # Combine individual and context information
        # Concatenate individual and context embeddings
        combined_embeddings = torch.cat([
            individual_embeddings,  # [B, K, C]
            context_embeddings      # [B, K, C]
        ], dim=-1)  # [B, K, 2*C]
        
        # Project to final embedding space
        speaker_embeddings = self.context_projection(combined_embeddings)  # [B, K, C]
        
        # Generate context weights based on difficulty
        context_weights = torch.softmax(
            -separation_difficulty.unsqueeze(-1).expand(-1, K), dim=-1
        )  # [B, K] - higher weight for harder separations
        
        context_info = {
            'num_speakers': K,
            'separation_difficulty': separation_difficulty,
            'speaker_similarities': speaker_similarities,
            'context_weights': context_weights,
            'individual_embeddings': individual_embeddings,
            'context_embeddings': context_embeddings
        }
        
        return speaker_embeddings, context_info


class MultiSpeakerAttentionBlock(nn.Module):
    """Cross-attention block for understanding speaker relationships."""
    
    def __init__(self, emb_dim, n_head=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            emb_dim, n_head, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 4, emb_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, embeddings, spk_lens):
        """
        Args:
            embeddings: [B, K, C] speaker embeddings
            spk_lens: [B, K] speaker lengths (for masking)
        Returns:
            context_embeddings: [B, K, C] context-aware embeddings
        """
        B, K, C = embeddings.shape
        
        # Create attention mask for variable speaker lengths
        # For now, assume all speakers are valid (can be enhanced later)
        attn_mask = None
        
        # Self-attention across speakers
        attn_out, attn_weights = self.attention(
            embeddings, embeddings, embeddings, 
            attn_mask=attn_mask
        )
        embeddings = self.norm1(embeddings + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(embeddings)
        embeddings = self.norm2(embeddings + ffn_out)
        
        return embeddings


class SpeakerRelationshipAnalyzer(nn.Module):
    """Analyzes relationships and similarities between speakers."""
    
    def __init__(self, emb_dim):
        super().__init__()
        self.similarity_projection = nn.Linear(emb_dim, emb_dim // 2)
        
    def forward(self, speaker_embeddings):
        """
        Args:
            speaker_embeddings: [B, K, C]
        Returns:
            similarities: [B, K, K] pairwise speaker similarities
        """
        B, K, C = speaker_embeddings.shape
        
        # Project to similarity space
        projected = self.similarity_projection(speaker_embeddings)  # [B, K, C//2]
        
        # Compute pairwise similarities
        similarities = torch.bmm(projected, projected.transpose(-2, -1))  # [B, K, K]
        
        # Normalize by embedding dimension
        similarities = similarities / math.sqrt(projected.size(-1))
        
        # Apply softmax to get similarity scores
        similarities = F.softmax(similarities, dim=-1)
        
        return similarities


class SeparationDifficultyEstimator(nn.Module):
    """Estimates how difficult it will be to separate the given speakers."""
    
    def __init__(self, emb_dim):
        super().__init__()
        self.difficulty_network = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),  # Pairwise features
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1)  # Single difficulty score
        )
        
    def forward(self, speaker_embeddings, speaker_similarities):
        """
        Args:
            speaker_embeddings: [B, K, C]
            speaker_similarities: [B, K, K]
        Returns:
            difficulty: [B] separation difficulty score per batch
        """
        B, K, C = speaker_embeddings.shape
        
        # Compute pairwise difficulty features
        difficulty_scores = []
        
        for k1 in range(K):
            for k2 in range(k1 + 1, K):  # Only upper triangle
                # Concatenate speaker embeddings for pairwise analysis
                pair_features = torch.cat([
                    speaker_embeddings[:, k1],  # [B, C]
                    speaker_embeddings[:, k2]   # [B, C]
                ], dim=-1)  # [B, 2*C]
                
                # Get similarity score for this pair
                similarity = speaker_similarities[:, k1, k2]  # [B]
                
                # Predict difficulty for this pair
                pair_difficulty = self.difficulty_network(pair_features).squeeze(-1)  # [B]
                
                # Weight by similarity (more similar = harder to separate)
                weighted_difficulty = pair_difficulty * similarity
                difficulty_scores.append(weighted_difficulty)
        
        if difficulty_scores:
            # Average difficulty across all pairs
            overall_difficulty = torch.stack(difficulty_scores, dim=1).mean(dim=1)  # [B]
        else:
            # Single speaker case
            overall_difficulty = torch.zeros(B, device=speaker_embeddings.device)
        
        return overall_difficulty


class ContextAwareFiLM(BaseFiLM):
    """
    Enhanced FiLM layer that uses multi-speaker context for conditioning.
    
    Unlike standard FiLM that only considers individual speaker embeddings,
    this version incorporates:
    - Speaker separation context
    - Inter-speaker relationship information  
    - Dynamic difficulty-based conditioning strength
    """
    
    def __init__(self, feature_dim, cond_dim):
        super().__init__(feature_dim, cond_dim)
        
        # Additional context processing
        self.context_gamma_fc = nn.Linear(cond_dim, feature_dim)
        self.context_beta_fc = nn.Linear(cond_dim, feature_dim)
        
        # Context fusion weights
        self.context_fusion = nn.Parameter(torch.zeros(1))
        
    def forward(self, cond, x, context_info=None):
        """
        Apply context-aware speaker conditioning to features.
        
        Args:
            cond: [B, cond_dim] speaker embedding
            x: [B, C, T, F] feature tensor
            context_info: Dict with multi-speaker context (optional)
        Returns:
            conditioned features: [B, C, T, F]
        """
        # Standard FiLM conditioning
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]
        
        # Apply base conditioning
        output = gamma * x + beta
        
        # Add context conditioning if available
        if context_info is not None and context_info['num_speakers'] > 1:
            # Get separation difficulty for adaptive conditioning
            difficulty = context_info['separation_difficulty']  # [B]
            
            # Generate context-aware modulation
            context_gamma = self.context_gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            context_beta = self.context_beta_fc(cond).unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
            
            # Scale context conditioning by difficulty
            difficulty_weight = difficulty.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
            context_gamma = context_gamma * difficulty_weight
            context_beta = context_beta * difficulty_weight
            
            # Fuse standard and context conditioning
            fusion_weight = torch.sigmoid(self.context_fusion)
            output = (1 - fusion_weight) * output + fusion_weight * (context_gamma * x + context_beta)
        
        return output


class MultiSpeakerContextGridNet(nn.Module):
    """
    Enhanced Multi-Speaker Context-Aware GridNet.
    
    Key Improvements over UniversalMCxTFGridNet:
    1. Multi-speaker context awareness during training
    2. Inter-speaker relationship understanding
    3. Context-aware FiLM conditioning
    4. Separation difficulty adaptation
    5. Maintains single-speaker enhancement compatibility
    
    Training Mode: Learns from full multi-speaker context
    Enhancement Mode: Uses learned context knowledge for single-speaker enhancement
    """
    
    def __init__(
        self,
        n_imics=1,
        n_layers=6,
        lstm_hidden_units=192,
        attn_n_head=4,
        attn_qk_output_channel=4,
        emb_dim=48,
        emb_ks=4,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
        context_layers=3,
        context_heads=4,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_imics = n_imics
        self.emb_dim = emb_dim
        
        # Mixture processing setup (same as Universal)
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        
        # Speaker feature extraction
        self.spk_conv = nn.Sequential(
            nn.Conv2d(2, emb_dim//2, ks, padding=padding),
            LayerNormalization(emb_dim//2, eps=eps),
            nn.PReLU(),
            nn.Conv2d(emb_dim//2, emb_dim, ks, padding=padding),
            LayerNormalization(emb_dim, eps=eps),
            nn.PReLU(),
        )
        
        # Multi-speaker context-aware auxiliary encoder
        self.aux_enc = MultiSpeakerContextAuxEncoder(
            emb_dim=emb_dim,
            num_layers=context_layers,
            n_head=context_heads
        )
        
        # Speaker-conditional mixture processing
        self.speaker_conditional_conv = SpeakerConditionalConv2d(
            in_channels=n_imics * 2,
            out_channels=emb_dim,
            kernel_size=ks,
            padding=padding,
            conditioning_dim=emb_dim,
            eps=eps
        )
        
        # Shared GridNet blocks (same weights for all speakers)
        self.shared_gridnets = nn.ModuleList(
            [
                GridNetV3Block(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    qk_output_channel=attn_qk_output_channel,
                    activation=activation,
                    eps=eps,
                )
                for _ in range(n_layers)
            ]
        )
        
        # Context-aware FiLM conditioning layers
        self.shared_fusions = nn.ModuleList(
            [ContextAwareFiLM(emb_dim, emb_dim) for _ in range(n_layers)]
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, ks, padding=padding),
            LayerNormalization(emb_dim, eps=eps),
            nn.PReLU(),
            nn.Conv2d(emb_dim, 2, (1, 1)),  # Output real/imaginary
        )
        
        # Debugging
        self._forward_count = 0
        self._gradient_log_interval = 50
        
        # Log model architecture
        logging.info("🌟 MULTI-SPEAKER CONTEXT GRIDNET INITIALIZED")
        logging.info("🎯 Key Features:")
        logging.info("   ✅ Multi-speaker context awareness")
        logging.info("   ✅ Inter-speaker relationship modeling")
        logging.info("   ✅ Context-aware FiLM conditioning")
        logging.info("   ✅ Separation difficulty adaptation")
        logging.info("   ✅ Single-speaker enhancement compatible")
        logging.info(f"📊 Architecture Details:")
        logging.info(f"   - Embedding dimension: {emb_dim}")
        logging.info(f"   - Number of layers: {n_layers}")
        logging.info(f"   - Context attention layers: {context_layers}")
        logging.info(f"   - Context attention heads: {context_heads}")
        
        # Parameter count
        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"   - Total parameters: {total_params:,}")
        
    def forward(self, spec: torch.Tensor, spk: torch.Tensor, spk_lens: torch.Tensor):
        """
        Context-aware forward pass supporting both training and enhancement modes.
        
        Training Mode (Multi-speaker):
            spec: [B, M, T, F, 2] mixture 
            spk: [B, K, T, F, 2] K speaker enrollments
            spk_lens: [B, K] speaker lengths
            Returns: [B, K, T, F] complex separated outputs
            
        Enhancement Mode (Single-speaker):  
            spec: [B, M, T, F, 2] mixture
            spk: [B, T, F, 2] single enrollment
            spk_lens: [B] single length
            Returns: [B, 1, T, F] complex enhanced output
        """
        assert spec.size(-1) == 2, spec.shape
        B, M, D2, D3, RI = spec.shape
        assert RI == 2
        
        # Log preprocessing info
        if self._forward_count % self._gradient_log_interval == 0:
            logging.info("🎯 CONTEXT-AWARE GRIDNET ANALYSIS:")
            logging.info(f"   Mixture spec shape: [B={B}, M={M}, D2={D2}, D3={D3}, RI={RI}]")
            if spk.ndim == 5:
                logging.info(f"   Multi-speaker mode: {spk.shape[1]} speakers")
            else:
                logging.info("   Single-speaker enhancement mode")
        
        # Decide which axis is F vs T
        if D2 <= D3:
            T, F = D3, D2
            feat = (
                spec.permute(0, 1, 4, 3, 2)  # [B, M, 2, T, F]
                .contiguous()
                .view(B, M * 2, T, F)  # [B, 2*M, T, F]
            )
        else:
            T, F = D2, D3
            feat = (
                spec.permute(0, 1, 4, 2, 3)  # [B, M, 2, T, F]
                .contiguous()
                .view(B, M * 2, T, F)  # [B, 2*M, T, F]
            )
        
        n_batch, mics, n_frames, n_freqs = B, M, T, F
        assert mics == self.n_imics
        
        mixture_features = feat  # [B, 2*M, T, F]
        self._forward_count += 1
        
        # Handle both single and multi-speaker cases
        if spk.ndim == 4:
            # Single speaker enhancement mode
            return self._process_single_speaker(mixture_features, spk, spk_lens, n_frames, n_freqs)
        elif spk.ndim == 5:
            # Multi-speaker training mode
            return self._process_multi_speakers(mixture_features, spk, spk_lens, n_frames, n_freqs)
        else:
            raise ValueError(f"spk must be 4D or 5D, got {spk.ndim}")
    
    def _process_single_speaker(self, mixture_features, spk, spk_lens, n_frames, n_freqs):
        """Process single speaker using context-aware processing."""
        B = mixture_features.shape[0]
        
        # Extract speaker features
        spk_feat = spk.permute(0, 3, 1, 2)  # [B, 2, T, F]  
        spk_feat = self.spk_conv(spk_feat)  # [B, C, T, F]
        
        # Get context-aware speaker embedding (single-speaker mode)
        speaker_embeddings, context_info = self.aux_enc(spk_feat, spk_lens, B, 1)
        speaker_embedding = speaker_embeddings.squeeze(1)  # [B, C]
        
        # Log embedding quality
        if self._forward_count % self._gradient_log_interval == 0:
            logging.info("🎤 CONTEXT-AWARE SPEAKER EMBEDDING (Single):")
            logging.info(f"   Embedding shape: {speaker_embedding.shape}")
            logging.info(f"   Embedding mean: {speaker_embedding.mean().item():.4f}")
            logging.info(f"   Embedding std: {speaker_embedding.std().item():.4f}")
            logging.info(f"   Context info: {list(context_info.keys())}")
        
        # Context-aware separation processing
        separated_audio = self._context_aware_separation_chain(
            mixture_features, speaker_embedding, context_info
        )
        
        # Convert to complex format
        re = separated_audio[:, 0].to(torch.float32)
        im = separated_audio[:, 1].to(torch.float32)
        output = torch.complex(re, im).unsqueeze(1)  # [B, 1, T, F]
        
        return output
    
    def _process_multi_speakers(self, mixture_features, spk, spk_lens, n_frames, n_freqs):
        """Process multiple speakers with full context awareness."""
        B, K = spk.shape[:2]
        
        # Extract features for all speakers: [B, K, T, F, 2] -> [B, K, C, T, F]
        spk_feat = spk.permute(0, 1, 4, 2, 3)  # [B, K, 2, T, F]
        spk_feat_list = []
        for k in range(K):
            feat_k = self.spk_conv(spk_feat[:, k])  # [B, C, T, F]
            spk_feat_list.append(feat_k)
        spk_features = torch.stack(spk_feat_list, dim=1)  # [B, K, C, T, F]
        
        # Get multi-speaker context-aware embeddings
        speaker_embeddings, context_info = self.aux_enc(spk_features, spk_lens, B, K)
        
        # Log multi-speaker context
        if self._forward_count % self._gradient_log_interval == 0:
            logging.info("🎭 MULTI-SPEAKER CONTEXT ANALYSIS:")
            logging.info(f"   Number of speakers: {K}")
            logging.info(f"   Embedding shape: {speaker_embeddings.shape}")
            logging.info(f"   Separation difficulty: {context_info['separation_difficulty'].mean().item():.4f}")
            logging.info(f"   Average similarity: {context_info['speaker_similarities'].mean().item():.4f}")
        
        # Process each speaker with full context awareness
        speaker_outputs = []
        for k in range(K):
            spk_emb = speaker_embeddings[:, k]  # [B, C]
            
            # Context-aware separation for speaker k
            separated_k = self._context_aware_separation_chain(
                mixture_features, spk_emb, context_info, speaker_idx=k
            )
            speaker_outputs.append(separated_k)
        
        # Stack outputs: [B, K, 2, T, F]
        out_ri = torch.stack(speaker_outputs, dim=1)
        
        # Convert to complex format
        re = out_ri[:, :, 0].to(torch.float32)
        im = out_ri[:, :, 1].to(torch.float32)
        output = torch.complex(re, im)  # [B, K, T, F]
        
        return output
    
    def _context_aware_separation_chain(self, mixture_features, speaker_embedding, context_info, speaker_idx=None):
        """
        Core separation processing with context awareness.
        
        Args:
            mixture_features: [B, 2*M, T, F]
            speaker_embedding: [B, C] 
            context_info: Dict with separation context
            speaker_idx: Optional speaker index for multi-speaker mode
        Returns:
            separated_features: [B, 2, T, F]
        """
        # Speaker-conditional mixture processing
        z = self.speaker_conditional_conv(mixture_features, speaker_embedding)  # [B, C, T, F]
        
        # Process through shared layers with context-aware conditioning
        for i in range(self.n_layers):
            # Context-aware FiLM conditioning
            z = self.shared_fusions[i](speaker_embedding, z, context_info)
            
            # Shared GridNet processing
            z = self.shared_gridnets[i](z)
        
        # Output head
        output = self.output_head(z)  # [B, 2, T, F]
        
        return output
    
    @property
    def num_spk(self):
        return -1  # Dynamic speaker count support


# Export all classes
__all__ = [
    'MultiSpeakerContextAuxEncoder',
    'MultiSpeakerAttentionBlock', 
    'SpeakerRelationshipAnalyzer',
    'SeparationDifficultyEstimator', 
    'ContextAwareFiLM',
    'MultiSpeakerContextGridNet'
]