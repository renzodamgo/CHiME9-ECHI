import math
from typing import Tuple, Any
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

HALF_PRECISION_DTYPES: tuple[Any, ...]
if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


class SpeakerConditionalConv2d(nn.Module):
    """
    Speaker-conditional convolution that conditions mixture processing on speaker embedding.
    This allows each speaker to have their own mixture analysis from the start.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, conditioning_dim, eps=1e-5):
        super().__init__()
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conditioning_proj = nn.Linear(conditioning_dim, out_channels)
        self.norm = LayerNormalization(out_channels, eps=eps)

    def forward(self, mixture_features, speaker_embedding):
        """
        Args:
            mixture_features: [B, C_in, T, F] mixture spectrogram
            speaker_embedding: [B, conditioning_dim] speaker embedding
        Returns:
            conditioned_features: [B, C_out, T, F] speaker-conditioned mixture features
        """
        # Base mixture processing
        base_features = self.base_conv(mixture_features)  # [B, C_out, T, F]

        # Speaker conditioning
        speaker_condition = self.conditioning_proj(speaker_embedding)  # [B, C_out]
        speaker_condition = speaker_condition.unsqueeze(-1).unsqueeze(-1)  # [B, C_out, 1, 1]

        # Apply conditioning via element-wise multiplication (gating)
        conditioned_features = base_features * (1.0 + speaker_condition)

        # Normalize
        conditioned_features = self.norm(conditioned_features)

        return conditioned_features


class UniversalMCxTFGridNet(nn.Module):
    """Universal Multi-Speaker TFGridNet with Speaker-Agnostic Architecture.

    This is an improved version of MCxTFGridNet that eliminates speaker positional bias
    by using a single shared processing chain for all speakers. Each speaker gets the
    same processing quality regardless of input order, making it ideal for general-purpose
    multi-speaker separation models.

    Key Improvements over Original GridNet:
    - Single shared processing chain (no speaker-specific chains)
    - Perfect speaker order agnosticism
    - Maximum parameter efficiency through shared learning
    - Dynamic K speaker support without retraining
    - Consistent separation quality across all speakers

    Args:
        n_imics: number of microphones channels (only fixed-array geometry supported).
        n_layers: number of TFGridNetV3 blocks.
        lstm_hidden_units: number of hidden units in LSTM.
        attn_n_head: number of heads in self-attention
        attn_qk_output_channel: output channels of point-wise conv2d for getting key and query
        emb_dim: embedding dimension
        emb_ks: kernel size for unfolding and deconv1D
        emb_hs: hop size for unfolding and deconv1D
        activation: activation function to use in the whole TFGridNetV3 model
        eps: small epsilon for normalization layers.
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
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_imics = n_imics
        self.emb_dim = emb_dim

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)

        # Speaker-conditional mixture encoder (shared across all speakers)
        self.speaker_conditional_conv = SpeakerConditionalConv2d(
            in_channels=2 * n_imics,
            out_channels=emb_dim,
            kernel_size=ks,
            padding=padding,
            conditioning_dim=emb_dim,
            eps=eps
        )

        # Shared speaker enrollment encoder
        self.spk_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2, out_channels=emb_dim, kernel_size=(3, 3), padding=(1, 1)
            ),
            LayerNormalization(emb_dim, eps=eps),
        )

        # Shared auxiliary encoder for speaker embedding extraction
        self.aux_enc = AuxEncoder(emb_dim)

        # UNIVERSAL ARCHITECTURE: Single shared processing chain for ALL speakers
        # This eliminates speaker positional bias and ensures consistent quality
        self.shared_fusions = nn.ModuleList([])
        self.shared_gridnets = nn.ModuleList([])

        for _ in range(n_layers):
            # FiLM conditioning layers (shared weights, speaker-specific conditioning)
            self.shared_fusions.append(FiLM(emb_dim, emb_dim))
            
            # GridNet processing blocks (shared weights)
            self.shared_gridnets.append(
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
            )

        # Single shared output head (all speakers use the same reconstruction weights)
        self.output_head = nn.ConvTranspose2d(emb_dim, 2, ks, padding=padding)

        # Debug counters
        self._forward_count = 0
        self._gradient_log_interval = 50

        # Log model architecture
        logging.info("UNIVERSAL SPEAKER-AGNOSTIC MODEL INITIALIZED")
        logging.info("🌟 Key Features:")
        logging.info("   ✅ Single shared processing chain for all speakers")
        logging.info("   ✅ Perfect speaker order agnosticism")
        logging.info("   ✅ Dynamic K speaker support")
        logging.info("   ✅ Maximum parameter efficiency")
        logging.info(f"📊 Architecture Details:")
        logging.info(f"   - Embedding dimension: {emb_dim}")
        logging.info(f"   - Number of layers: {n_layers}")
        logging.info(f"   - Shared FiLM layers: {len(self.shared_fusions)}")
        logging.info(f"   - Shared GridNet blocks: {len(self.shared_gridnets)}")
        
        # Parameter count comparison
        total_params = sum(p.numel() for p in self.parameters())
        shared_params = sum(p.numel() for p in self.shared_fusions.parameters()) + \
                      sum(p.numel() for p in self.shared_gridnets.parameters()) + \
                      sum(p.numel() for p in self.output_head.parameters())
        logging.info(f"   - Total parameters: {total_params:,}")
        logging.info(f"   - Shared processing parameters: {shared_params:,} ({shared_params/total_params*100:.1f}%)")

    def forward(self, spec: torch.Tensor, spk: torch.Tensor, spk_lens: torch.Tensor):
        """
        Universal forward pass supporting any number of speakers with consistent quality.
        
        Args:
            spec: [B, M, T, F, 2]  mixture (M = n_imics)
            spk : [B, T, F, 2] OR [B, K, T, F, 2] enrollment(s)
            spk_lens: [B] or [B, K] STFT frame lengths for enroll(s)

        Returns:
            if spk.ndim == 4 (single spk): [B, 1, T, F] complex
            if spk.ndim == 5 (K spk):      [B, K, T, F] complex
        """
        assert spec.size(-1) == 2, spec.shape
        B, M, D2, D3, RI = spec.shape
        assert RI == 2

        # Log preprocessing info for debugging
        if self._forward_count % self._gradient_log_interval == 0:
            logging.info("🔍 UNIVERSAL GRIDNET ANALYSIS:")
            logging.info(f"   Mixture spec shape: [B={B}, M={M}, D2={D2}, D3={D3}, RI={RI}]")

        # Decide which axis is F vs T (F is usually the smaller one, e.g., 65)
        if D2 <= D3:
            # spec: [B, M, F, T, 2]  (common in your logs)
            T, F = D3, D2
            feat = (
                spec.permute(0, 1, 4, 3, 2)  # [B, M, 2, T, F]
                .contiguous()
                .view(B, M * 2, T, F)  # [B, 2*M, T, F]
            )
        else:
            # spec: [B, M, T, F, 2]
            T, F = D2, D3
            feat = (
                spec.permute(0, 1, 4, 2, 3)  # [B, M, 2, T, F]
                .contiguous()
                .view(B, M * 2, T, F)  # [B, 2*M, T, F]
            )

        n_batch, mics, n_frames, n_freqs = B, M, T, F
        assert mics == self.n_imics

        # Store mixture features for universal speaker-conditional processing
        mixture_features = feat  # [B, 2*M, T, F]
        self._forward_count += 1

        # Handle both single and multi-speaker cases with universal processing
        if spk.ndim == 4:
            # Single speaker case: [B, T, F, 2] -> [B, 1, T, F] output
            return self._process_single_speaker(mixture_features, spk, spk_lens, n_frames, n_freqs)
        elif spk.ndim == 5:
            # Multi-speaker case: [B, K, T, F, 2] -> [B, K, T, F] output  
            return self._process_multi_speakers(mixture_features, spk, spk_lens, n_frames, n_freqs)
        else:
            raise ValueError(f"spk must be 4D or 5D, got {spk.ndim}")

    def _process_single_speaker(self, mixture_features, spk, spk_lens, n_frames, n_freqs):
        """Process single speaker using universal shared chain."""
        B = mixture_features.shape[0]
        
        # Extract speaker embedding
        spk_feat = spk.permute(0, 3, 1, 2)  # [B, 2, T, F]
        spk_feat = self.spk_conv(spk_feat)  # [B, C, T, F]
        speaker_embedding, _ = self.aux_enc(spk_feat, spk_lens, B, 1)  # [B, C] - single speaker case

        # Log speaker embedding quality
        if self._forward_count % self._gradient_log_interval == 0:
            logging.info("🎤 UNIVERSAL SPEAKER EMBEDDING (Single):")
            logging.info(f"   Embedding shape: {speaker_embedding.shape}")
            logging.info(f"   Embedding mean: {speaker_embedding.mean().item():.4f}")
            logging.info(f"   Embedding std: {speaker_embedding.std().item():.4f}")
            
            if speaker_embedding.std() < 0.01:
                logging.warning("⚠️  Speaker embedding collapse detected!")

        # Universal processing: same shared chain for any speaker
        separated_audio = self._universal_separation_chain(mixture_features, speaker_embedding)

        # Convert to complex format: [B, 2, T, F] -> [B, T, F] complex
        re = separated_audio[:, 0].to(torch.float32)
        im = separated_audio[:, 1].to(torch.float32)
        output = torch.complex(re, im)  # [B, T, F] complex64
        
        return output.unsqueeze(1)  # [B, 1, T, F] for consistency

    def _process_multi_speakers(self, mixture_features, spk, spk_lens, n_frames, n_freqs):
        """Process multiple speakers using universal shared chain for each."""
        B, K, T, F, _ = spk.shape

        # Extract all speaker embeddings
        spk_feat = spk.permute(0, 1, 4, 2, 3).reshape(B * K, 2, T, F)  # [BK, 2, T, F]
        spk_feat = self.spk_conv(spk_feat)  # [BK, C, T, F]

        # Handle speaker lengths
        if spk_lens.ndim == 1:
            spk_lens = spk_lens.unsqueeze(1).expand(B, K).reshape(B * K)
        else:
            spk_lens = spk_lens.reshape(B * K)

        speaker_embeddings, _ = self.aux_enc(spk_feat, spk_lens, B, K)  # [BK, C]
        speaker_embeddings = speaker_embeddings.view(B, K, -1)  # [B, K, C]

        # Process each speaker with the SAME universal chain
        speaker_outputs = []
        
        for k in range(K):
            spk_emb = speaker_embeddings[:, k]  # [B, C]
            
            if self._forward_count % self._gradient_log_interval == 0 and k == 0:
                logging.info(f"🌟 UNIVERSAL PROCESSING: All {K} speakers use identical shared chain")
                logging.info(f"   Speaker {k} embedding norm: {spk_emb.norm(dim=1).mean().item():.4f}")
            
            # Universal separation: same shared weights for ALL speakers
            separated_k = self._universal_separation_chain(mixture_features, spk_emb)
            speaker_outputs.append(separated_k)

        # Stack all speaker outputs: [B, K, 2, T, F]
        out_ri = torch.stack(speaker_outputs, dim=1)

        # Convert to complex format
        re = out_ri[:, :, 0].to(torch.float32)
        im = out_ri[:, :, 1].to(torch.float32)
        output = torch.complex(re, im)  # [B, K, T, F] complex64
        
        return output

    def _universal_separation_chain(self, mixture_features, speaker_embedding):
        """
        The core universal separation chain used by ALL speakers.
        
        This is the key innovation: same weights, different conditioning.
        Every speaker gets identical processing quality regardless of order.
        """
        # Speaker-conditional mixture processing
        z = self.speaker_conditional_conv(mixture_features, speaker_embedding)  # [B, C, T, F]

        # Shared processing chain with speaker-specific conditioning
        for i in range(self.n_layers):
            # FiLM conditioning: same weights, speaker-specific scaling/bias
            z = self.shared_fusions[i](speaker_embedding, z)
            
            # GridNet processing: same weights for all speakers
            z = self.shared_gridnets[i](z)

        # Shared output head: same reconstruction weights for all speakers  
        output = self.output_head(z)  # [B, 2, T, F]
        
        return output

    @property 
    def num_spk(self):
        """Dynamic speaker support - can handle any number of speakers."""
        return float('inf')  # Universal model supports unlimited speakers


class GridNetV3Block(nn.Module):
    """GridNetV3 processing block for time-frequency modeling."""
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        hidden_channels,
        n_head=4,
        qk_output_channel=4,
        activation="prelu",
        eps=1e-5,
    ):
        super().__init__()
        assert activation == "prelu"
        activation_fn = torch.nn.PReLU()

        in_channels = emb_dim * emb_ks

        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        if emb_ks == emb_hs:
            self.intra_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
            )

        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=False
        )
        if emb_ks == emb_hs:
            self.inter_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.inter_linear = nn.ConvTranspose1d(
                hidden_channels, emb_dim, emb_ks, stride=emb_hs
            )

        # use constant E not to be dependent on the number of frequency bins
        E = qk_output_channel
        assert emb_dim % n_head == 0

        self.add_module("attn_conv_Q", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_Q",
            AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps),
        )

        self.add_module("attn_conv_K", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_K",
            AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps),
        )

        self.add_module(
            "attn_conv_V", nn.Conv2d(emb_dim, n_head * emb_dim // n_head, 1)
        )
        self.add_module(
            "attn_norm_V",
            AllHeadPReLULayerNormalization4DC((n_head, emb_dim // n_head), eps=eps),
        )

        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                activation_fn,
                LayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps),
            ),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        """GridNetV3Block Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape

        olp = self.emb_ks - self.emb_hs
        T = (
            math.ceil((old_T + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )
        Q = (
            math.ceil((old_Q + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )

        x = x.permute(0, 2, 3, 1)  # [B, old_T, old_Q, C]
        x = F.pad(x, (0, 0, olp, Q - old_Q - olp, olp, T - old_T - olp))  # [B, T, Q, C]

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, T, Q, C]
        if self.emb_ks == self.emb_hs:
            intra_rnn = intra_rnn.view([B * T, -1, self.emb_ks * C])  # [BT, Q//I, I*C]
            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, Q//I, H]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, Q//I, I*C]
            intra_rnn = intra_rnn.view([B, T, Q, C])
        else:
            intra_rnn = intra_rnn.view([B * T, Q, C])  # [BT, Q, C]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, C, Q]
            intra_rnn = F.unfold(
                intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )  # [BT, C*I, -1]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*I]

            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]

            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
            intra_rnn = intra_rnn.view([B, T, C, Q])
            intra_rnn = intra_rnn.transpose(-2, -1)  # [B, T, Q, C]
        intra_rnn = intra_rnn + input_  # [B, T, Q, C]

        intra_rnn = intra_rnn.transpose(1, 2)  # [B, Q, T, C]

        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, Q, T, C]
        if self.emb_ks == self.emb_hs:
            inter_rnn = inter_rnn.view([B * Q, -1, self.emb_ks * C])  # [BQ, T//I, I*C]
            inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BQ, T//I, H]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, T//I, I*C]
            inter_rnn = inter_rnn.view([B, Q, T, C])
        else:
            inter_rnn = inter_rnn.view(B * Q, T, C)  # [BQ, T, C]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, C, T]
            inter_rnn = F.unfold(
                inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )  # [BQ, C*I, -1]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, -1, C*I]

            inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BQ, -1, H]

            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, H, -1]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, C, T]
            inter_rnn = inter_rnn.view([B, Q, C, T])
            inter_rnn = inter_rnn.transpose(-2, -1)  # [B, Q, T, C]
        inter_rnn = inter_rnn + input_  # [B, Q, T, C]

        inter_rnn = inter_rnn.permute(0, 3, 2, 1)  # [B, C, T, Q]

        inter_rnn = inter_rnn[..., olp : olp + old_T, olp : olp + old_Q]
        batch = inter_rnn

        Q = self["attn_norm_Q"](self["attn_conv_Q"](batch))  # [B, n_head, C, T, Q]
        K = self["attn_norm_K"](self["attn_conv_K"](batch))  # [B, n_head, C, T, Q]
        V = self["attn_norm_V"](self["attn_conv_V"](batch))  # [B, n_head, C, T, Q]
        Q = Q.view(-1, *Q.shape[2:])  # [B*n_head, C, T, Q]
        K = K.view(-1, *K.shape[2:])  # [B*n_head, C, T, Q]
        V = V.view(-1, *V.shape[2:])  # [B*n_head, C, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]

        K = K.transpose(2, 3)
        K = K.contiguous().view([B * self.n_head, -1, old_T])  # [B', C*Q, T]

        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K) / (emb_dim**0.5)  # [B', T, T]

        causal_mask = (
            torch.tril(torch.ones(attn_mat.shape[-1], attn_mat.shape[-1]))
            .bool()
            .to(attn_mat.device)
        )
        attn_mat = attn_mat.masked_fill(~causal_mask, float("-inf"))

        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.contiguous().view(
            [B, self.n_head * emb_dim, old_T, old_Q]
        )  # [B, C, T, Q])
        batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

        out = batch + inter_rnn
        return out


class LayerNormalization(nn.Module):
    def __init__(self, input_dim, dim=1, total_dim=4, eps=1e-5):
        super().__init__()
        self.dim = dim if dim >= 0 else total_dim + dim
        param_size = [1 if ii != self.dim else input_dim for ii in range(total_dim)]
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x):
        if x.ndim - 1 < self.dim:
            raise ValueError(
                f"Expect x to have {self.dim + 1} dimensions, but got {x.ndim}"
            )
        if x.dtype in HALF_PRECISION_DTYPES:
            dtype = x.dtype
            x = x.float()
        else:
            dtype = None
        mu_ = x.mean(dim=self.dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat.to(dtype=dtype) if dtype else x_hat


class AllHeadPReLULayerNormalization4DC(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2, input_dimension
        H, E = input_dimension
        param_size = [1, H, E, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E

    def forward(self, x):
        assert x.ndim == 4
        B, _, T, F = x.shape
        x = x.view([B, self.H, self.E, T, F])
        x = self.act(x)  # [B,H,E,T,F]
        stat_dim = (2,)
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,H,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,H,1,T,1]
        x = ((x - mu_) / std_) * self.gamma + self.beta  # [B,H,E,T,F]
        return x


class AuxEncoder(nn.Module):
    """Universal auxiliary encoder for speaker embedding extraction."""
    
    def __init__(self, emb_dim):
        super(AuxEncoder, self).__init__()
        k1, k2 = (1, 3), (1, 3)
        self.d_feat = emb_dim

        self.aux_enc = nn.ModuleList(
            [
                EnUnetModule(emb_dim, emb_dim, (1, 5), k2, scale=4),
                EnUnetModule(emb_dim, emb_dim, k1, k2, scale=3),
                EnUnetModule(emb_dim, emb_dim, k1, k2, scale=2),
                EnUnetModule(emb_dim, emb_dim, k1, k2, scale=1),
            ]
        )
        self.out_conv = nn.Linear(emb_dim, emb_dim)

        # Attention Pooling mechanism
        self.attention = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.Tanh(),
            nn.Linear(emb_dim // 2, 1)
        )

    def forward(
        self, auxs: torch.Tensor, aux_lengths: torch.Tensor, B: int = None, K: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        aux_lengths = (((aux_lengths // 3) // 3) // 3) // 3
        auxs = auxs.transpose(2, 3)
        for i in range(len(self.aux_enc)):
            auxs = self.aux_enc[i](auxs)  # [BK, C, T, F]

        # Attention Pooling
        # auxs is [BK, C, T, F]
        BK, C, T, n_freqs = auxs.shape

        # Reshape for attention: [BK, C, T*F] -> [BK, T*F, C]
        x = auxs.view(BK, C, T * n_freqs).transpose(1, 2)

        # Get attention scores
        attn_weights = self.attention(x).squeeze(-1) # [BK, T*F]
        attn_weights = F.softmax(attn_weights, dim=-1) # [BK, T*F]

        # Apply attention weights
        # (BK, 1, T*F) @ (BK, T*F, C) -> (BK, 1, C)
        weighted_avg = torch.bmm(attn_weights.unsqueeze(1), x)
        auxs = weighted_avg.squeeze(1) # [BK, C]

        auxs = self.out_conv(auxs)
        return auxs, None  # No speaker classification in universal model


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for speaker conditioning."""
    
    def __init__(self, feature_dim, cond_dim):
        super(FiLM, self).__init__()
        self.gamma_fc = nn.Linear(cond_dim, feature_dim)
        self.beta_fc = nn.Linear(cond_dim, feature_dim)

        # Debug counters
        self._forward_count = 0
        self._log_interval = 50

    def forward(self, cond, x):
        """
        Apply speaker conditioning to features.
        
        Args:
            cond: [B, cond_dim] speaker embedding
            x: [B, C, T, F] feature tensor
        Returns:
            conditioned features: [B, C, T, F]
        """
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        if not hasattr(self, "printed") and self._forward_count % self._log_interval == 0:
            logging.info("🎭 UNIVERSAL FiLM CONDITIONING:")
            logging.info(f"   Gamma (scale): mean={gamma.abs().mean():.2e}, max={gamma.abs().max():.2e}")
            logging.info(f"   Beta (bias): mean={beta.abs().mean():.2e}, max={beta.abs().max():.2e}")
            self.printed = True
            
        self._forward_count += 1
        return gamma * x + beta


class EnUnetModule(nn.Module):
    """Encoder-Decoder U-Net module for multi-scale feature extraction."""
    
    def __init__(self, cin: int, cout: int, k1: tuple, k2: tuple, scale: int):
        super(EnUnetModule, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.cin = cin
        self.cout = cout
        self.scale = scale

        self.in_conv = nn.Sequential(
            GateConv2d(cin, cout, k1, (1, 2)), nn.BatchNorm2d(cout), nn.PReLU(cout)
        )
        self.encoder = nn.ModuleList([Conv2dUnit(k2, cout) for _ in range(scale)])
        self.decoder = nn.ModuleList([Deconv2dUnit(k2, cout, 1)])
        for i in range(1, scale):
            self.decoder.append(Deconv2dUnit(k2, cout, 2))
        self.out_pool = nn.AvgPool2d((3, 1), ceil_mode=True)

    @staticmethod
    def _match_spatial(a: torch.Tensor, b: torch.Tensor):
        # a, b: [B, C, T, F] -> crop both to min(T,F)
        T = min(a.size(-2), b.size(-2))
        F = min(a.size(-1), b.size(-1))
        if a.size(-2) != T or a.size(-1) != F:
            a = a[..., :T, :F]
        if b.size(-2) != T or b.size(-1) != F:
            b = b[..., :T, :F]
        return a, b

    def forward(self, x: torch.Tensor):
        x_resi = self.in_conv(x)
        x = x_resi
        x_list = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            x_list.append(x)

        # deepest level
        x = self.decoder[0](x)

        # up path with skip connections (crop before concat)
        for i in range(1, len(self.decoder)):
            skip = x_list[-(i + 1)]
            x_c, skip_c = self._match_spatial(x, skip)
            x = self.decoder[i](torch.cat([x_c, skip_c], dim=1))

        x_resi, x = self._match_spatial(x_resi, x)
        x_resi = x_resi + x
        return self.out_pool(x_resi)


class GateConv2d(nn.Module):
    def __init__(self, cin: int, cout: int, k: tuple, s: tuple):
        super(GateConv2d, self).__init__()
        self.cin = cin
        self.cout = cout
        self.k = k
        self.s = s

        self.conv = nn.Sequential(
            nn.ConstantPad2d((0, 0, k[0] - 1, 0), value=0.0),
            nn.Conv2d(in_channels=cin, out_channels=cout * 2, kernel_size=k, stride=s),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)

        return outputs * gate.sigmoid()


class Conv2dUnit(nn.Module):
    def __init__(self, k: tuple, c: int):
        super(Conv2dUnit, self).__init__()
        self.k = k
        self.c = c
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, k, (1, 2)), nn.BatchNorm2d(c), nn.PReLU(c)
        )

    def forward(self, x):
        return self.conv(x)


class Deconv2dUnit(nn.Module):
    def __init__(self, k: tuple, c: int, expend_scale: int):
        super(Deconv2dUnit, self).__init__()
        self.k = k
        self.c = c
        self.expend_scale = expend_scale
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c * expend_scale, c, k, (1, 2)),
            nn.BatchNorm2d(c),
            nn.PReLU(c),
        )

    def forward(self, x):
        return self.deconv(x)


if __name__ == "__main__":
    # Test the universal model with smaller, safer configurations
    print("🚀 Testing Universal GridNet...")
    model = UniversalMCxTFGridNet()
    
    # Test single speaker with smaller dimensions
    print("Testing single speaker...")
    audio_single = torch.rand(1, 1, 65, 100, 2)  # [B, M, F, T, 2] - smaller T
    aux_single = torch.rand(1, 65, 100, 2)       # [B, F, T, 2]
    aux_lens_single = torch.tensor([100])
    
    try:
        out_single = model(audio_single, aux_single, aux_lens_single)
        print(f"✅ Single speaker output shape: {out_single.shape}")
    except Exception as e:
        print(f"❌ Single speaker test failed: {e}")
        exit(1)
    
    # Test multiple speakers (K=2) with smaller dimensions
    print("Testing 2 speakers...")
    audio_multi = torch.rand(1, 1, 65, 100, 2)   # [B, M, F, T, 2]
    aux_multi = torch.rand(1, 2, 65, 100, 2)     # [B, K=2, F, T, 2]
    aux_lens_multi = torch.tensor([[100, 100]])
    
    try:
        out_multi = model(audio_multi, aux_multi, aux_lens_multi)
        print(f"✅ Multi-speaker output shape: {out_multi.shape}")
    except Exception as e:
        print(f"❌ Multi-speaker test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("✅ Universal GridNet successfully handles variable speaker counts!")