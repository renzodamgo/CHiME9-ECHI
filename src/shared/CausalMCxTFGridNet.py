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


class MCxTFGridNet(nn.Module):
    """Online TFGridNetV3.

    Adapted from:
        https://github.com/HaoFengyuan/X-TF-GridNet/blob/main/nnet/pTFGridNet.py
        https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tfgridnetv3_separator.py

    Reference:
    [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
    in TASLP, 2023.
    [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural
    Speaker Separation", in ICASSP, 2023.
    [3] Fengyuan Hao, Xiaodong Li, Chengshi Zheng,
    "X-TF-GridNet: A time–frequency domain target speaker extraction network with
    adaptive speaker embedding fusion", in Information Fusion, 2024.

    Args:
        n_srcs: number of output sources/speakers.
        n_fft: stft window size.
        stride: stft stride.
        window: stft window type choose between 'hamming', 'hanning' or None.
        n_imics: number of microphones channels (only fixed-array geometry supported).
        n_layers: number of TFGridNetV3 blocks.
        lstm_hidden_units: number of hidden units in LSTM.
        attn_n_head: number of heads in self-attention
        attn_attn_qk_output_channel: output channels of point-wise conv2d for getting
            key and query
        emb_dim: embedding dimension
        emb_ks: kernel size for unfolding and deconv1D
        emb_hs: hop size for unfolding and deconv1D
        activation: activation function to use in the whole TFGridNetV3 model,
            you can use any torch supported activation e.g. 'relu' or 'elu'.
        eps: small epsilon for normalization layers.
        use_builtin_complex: whether to use builtin complex type or not.
    """

    def __init__(
        self,
        n_srcs=2,
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
        self.n_srcs = n_srcs
        print(f"n_srcs: {n_srcs} (number of output sources/speakers.)")
        self.n_layers = n_layers
        self.n_imics = n_imics

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)

        # Speaker-conditional mixture encoder instead of shared encoder
        self.speaker_conditional_conv = SpeakerConditionalConv2d(
            in_channels=2 * n_imics,
            out_channels=emb_dim,
            kernel_size=ks,
            padding=padding,
            conditioning_dim=emb_dim,
            eps=eps
        )

        self.spk_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2, out_channels=emb_dim, kernel_size=(3, 3), padding=(1, 1)
            ),
            LayerNormalization(emb_dim, eps=eps),
        )

        self.aux_enc = AuxEncoder(emb_dim, n_srcs)

        # Speaker-specific processing layers
        self.speaker_fusions = nn.ModuleList([])
        self.speaker_gridnets = nn.ModuleList([])
        self.speaker_output_heads = nn.ModuleList([])

        for _ in range(n_srcs):
            # Each speaker gets their own processing chain
            layer_fusions = nn.ModuleList([])
            layer_gridnets = nn.ModuleList([])

            for _ in range(n_layers):
                layer_fusions.append(FiLM(emb_dim, emb_dim))
                layer_gridnets.append(
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

            self.speaker_fusions.append(layer_fusions)
            self.speaker_gridnets.append(layer_gridnets)

            # Individual output head per speaker (2 channels: real/imag)
            self.speaker_output_heads.append(
                nn.ConvTranspose2d(emb_dim, 2, ks, padding=padding)
            )
        # log model dict
        logging.info("SPEAKER-SPECIFIC MODEL INITIALIZED")
        logging.info(f"n_srcs (processing chains): {n_srcs}")
        logging.info(f"model dict keys: {list(self.state_dict().keys())[:10]}...")  # Show first 10 keys
        logging.info(
            f"spk_conv.0.weight: {self.state_dict()['spk_conv.0.weight'].shape}"
        )
        logging.info(f"spk_conv.0.bias: {self.state_dict()['spk_conv.0.bias'].shape}")
        logging.info(f"speaker_conditional_conv parameters: {sum(p.numel() for p in self.speaker_conditional_conv.parameters())}")
        logging.info(f"total speaker-specific chains: {len(self.speaker_output_heads)}")

        # Debug counters
        self._forward_count = 0
        self._gradient_log_interval = 50  # Log every N forwards

    def _log_deconv_gradients(self, module, grad_input, grad_output):
        """Log gradient statistics for deconv layer debugging"""
        if self._forward_count % self._gradient_log_interval == 0 and grad_output[0] is not None:
            grad = grad_output[0]  # [B, n_srcs*2, T, F]
            B, channels, T, F = grad.shape

            logging.info("🔍 DECONV GRADIENT ANALYSIS:")

            # Per-speaker channel gradient analysis
            for spk in range(self.n_srcs):
                ch_start = spk * 2
                ch_end = ch_start + 2
                spk_grad = grad[:, ch_start:ch_end, :, :]  # [B, 2, T, F]

                grad_mean = spk_grad.mean().item()
                grad_std = spk_grad.std().item()
                grad_max = spk_grad.abs().max().item()
                grad_norm = spk_grad.norm().item()

                logging.info(f"   Speaker {spk} (ch {ch_start}:{ch_end}): "
                           f"mean={grad_mean:.2e}, std={grad_std:.2e}, "
                           f"max_abs={grad_max:.2e}, norm={grad_norm:.2e}")

            # Check for vanishing/exploding gradients
            total_grad_norm = grad.norm().item()
            if total_grad_norm < 1e-8:
                logging.warning("⚠️  VANISHING GRADIENTS detected in deconv layer!")
            elif total_grad_norm > 100:
                logging.warning("⚠️  EXPLODING GRADIENTS detected in deconv layer!")

    def forward(self, spec: torch.Tensor, spk: torch.Tensor, spk_lens: torch.Tensor):
        """
        spec: [B, M, T, F, 2]  mixture (M = n_imics)
        spk : [B, T, F, 2] OR [B, K, T, F, 2] enrollment(s)
        spk_lens: [B] or [B, K] STFT frame lengths for enroll(s)

        Returns:
        if spk.ndim == 4 (single spk): [B, 1, n_srcs, T, F] complex  (backward compatible)
        if spk.ndim == 5 (K spk):      [B, K,        T, F] complex   (n_srcs is treated as 1 here)
        """
        assert spec.size(-1) == 2, spec.shape
        B, M, D2, D3, RI = spec.shape
        assert RI == 2

        # Log STFT preprocessing info for debugging
        if self._forward_count % self._gradient_log_interval == 0:
            logging.info("🔍 STFT PREPROCESSING ANALYSIS:")
            logging.info(f"   Mixture spec shape: [B={B}, M={M}, D2={D2}, D3={D3}, RI={RI}]")

            # Analyze mixture STFT statistics
            spec_real = spec[..., 0]  # [B, M, D2, D3]
            spec_imag = spec[..., 1]  # [B, M, D2, D3]
            spec_mag = torch.sqrt(spec_real**2 + spec_imag**2)  # [B, M, D2, D3]

            logging.info(f"   Mixture magnitude: mean={spec_mag.mean().item():.4f}, "
                        f"max={spec_mag.max().item():.4f}, std={spec_mag.std().item():.4f}")

            # Analyze enrollment STFT
            if spk.ndim == 4:  # Single enrollment
                spk_real = spk[..., 0]  # [B, D2, D3]
                spk_imag = spk[..., 1]  # [B, D2, D3]
                spk_mag = torch.sqrt(spk_real**2 + spk_imag**2)
                logging.info(f"   Single enrollment shape: [B={B}, D2={D2}, D3={D3}]")
                logging.info(f"   Enrollment magnitude: mean={spk_mag.mean().item():.4f}, "
                            f"max={spk_mag.max().item():.4f}, std={spk_mag.std().item():.4f}")
            elif spk.ndim == 5:  # Multi-enrollment
                K = spk.shape[1]
                logging.info(f"   Multi-enrollment shape: [B={B}, K={K}, D2={D2}, D3={D3}]")
                for k in range(K):
                    spk_k = spk[:, k, :, :, :]  # [B, D2, D3, 2]
                    spk_k_real = spk_k[..., 0]
                    spk_k_imag = spk_k[..., 1]
                    spk_k_mag = torch.sqrt(spk_k_real**2 + spk_k_imag**2)
                    logging.info(f"     Speaker {k} magnitude: mean={spk_k_mag.mean().item():.4f}, "
                                f"max={spk_k_mag.max().item():.4f}, std={spk_k_mag.std().item():.4f}")

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

        # --- Store mixture features for speaker-specific processing ---
        # feat: [B, 2*M, T, F] - mixture spectrogram
        mixture_features = feat  # [B, 2*M, T, F]
        self._forward_count += 1

        # --- Handle enrollments: single or K ---
        if spk.ndim == 4:
            # Single speaker case - use first speaker's processing chain
            # [B, T, F, 2] -> [B, 2, T, F] -> encode -> embedding [B, C]
            spk_feat = spk.permute(0, 3, 1, 2)  # [B, 2, T, F]
            spk_feat = self.spk_conv(spk_feat)  # [B, C, T, F]
            e, _ = self.aux_enc(spk_feat, spk_lens)  # [B, C]

            # Log speaker embedding quality for debugging
            if self._forward_count % self._gradient_log_interval == 0:
                logging.info("🎤 SPEAKER EMBEDDING ANALYSIS (Single):")
                logging.info(f"   Embedding shape: {e.shape}")
                logging.info(f"   Embedding mean: {e.mean().item():.4f}, std: {e.std().item():.4f}")
                logging.info(f"   Embedding norm: {e.norm(dim=1).mean().item():.4f}")

                # Check for embedding collapse
                if e.std() < 0.01:
                    logging.warning("⚠️  SPEAKER EMBEDDING collapse detected! Low variation across features.")

            # Use first speaker's processing chain for single speaker
            # Speaker-conditional mixture processing
            z = self.speaker_conditional_conv(mixture_features, e)  # [B, C, T, F]

            # Process through first speaker's layers
            for i in range(self.n_layers):
                z = self.speaker_fusions[0][i](e, z)
                z = self.speaker_gridnets[0][i](z)

            # Use first speaker's output head
            out_ri = self.speaker_output_heads[0](z)  # [B, 2, T, F]

            # For backward compatibility, expand to [B, n_srcs, 2, T, F] format
            out_expanded = out_ri.unsqueeze(1).expand(B, self.n_srcs, 2, n_frames, n_freqs)

            # Cast re/im to fp32, then pack into complex64
            re = out_expanded[:, :, 0].to(torch.float32)
            im = out_expanded[:, :, 1].to(torch.float32)
            out = torch.complex(re, im)  # [B, n_srcs, T, F] complex64
            return out.unsqueeze(1)  # [B, 1, n_srcs, T, F]

        elif spk.ndim == 5:
            # Multi-speaker case: Each speaker gets dedicated processing chain
            # spk: [B, K, T, F, 2], spec: [B, M, Tm, Fm, 2]
            B, K, T, F, _ = spk.shape

            # --- Encode all speaker enrollments ---
            spk_feat = spk.permute(0, 1, 4, 2, 3).reshape(
                B * K, 2, T, F
            )  # [BK, 2, T, F]
            spk_feat = self.spk_conv(spk_feat)  # [BK, C, T, F]

            # Handle speaker lengths
            if spk_lens.ndim == 1:
                spk_lens = spk_lens.unsqueeze(1).expand(B, K).reshape(B * K)
            else:
                spk_lens = spk_lens.reshape(B * K)

            speaker_embeddings, _ = self.aux_enc(spk_feat, spk_lens)  # [BK, C]
            speaker_embeddings = speaker_embeddings.view(B, K, -1)  # [B, K, C]

            # --- Process each speaker independently with their own chain ---
            speaker_outputs = []

            for k in range(K):
                # Get this speaker's embedding
                spk_emb = speaker_embeddings[:, k]  # [B, C]

                # Use speaker k's processing chain (or first chain if k >= n_srcs)
                chain_idx = min(k, self.n_srcs - 1)

                # Speaker-conditional mixture processing
                z_k = self.speaker_conditional_conv(mixture_features, spk_emb)  # [B, C, T, F]

                if not hasattr(self, '_logged_speaker_chain_assignment'):
                    logging.info(f"🎯 Speaker {k} using processing chain {chain_idx}")

                # Process through speaker-specific layers
                for i in range(self.n_layers):
                    z_k = self.speaker_fusions[chain_idx][i](spk_emb, z_k)
                    z_k = self.speaker_gridnets[chain_idx][i](z_k)

                # Speaker-specific output head
                out_k = self.speaker_output_heads[chain_idx](z_k)  # [B, 2, T, F]
                speaker_outputs.append(out_k)

            # Mark logging
            if not hasattr(self, '_logged_speaker_chain_assignment'):
                self._logged_speaker_chain_assignment = True
                logging.info(f"🔧 Speaker-specific processing chains assigned for {K} speakers")

            # Stack all speaker outputs
            out_ri = torch.stack(speaker_outputs, dim=1)  # [B, K, 2, T, F]

            # Convert to complex format
            re = out_ri[:, :, 0].to(torch.float32)
            im = out_ri[:, :, 1].to(torch.float32)
            out = torch.complex(re, im)  # [B, K, T, F] (complex64)
            return out

        else:
            raise ValueError(f"spk must be 4D or 5D, got {spk.ndim}")

    @property
    def num_spk(self):
        return self.n_srcs


class GridNetV3Block(nn.Module):
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
        """GridNetV2Block Forward.

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
    def __init__(self, emb_dim, num_spks):
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
        self.speaker = nn.Linear(emb_dim, num_spks)

    def forward(
        self, auxs: torch.Tensor, aux_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        aux_lengths = (((aux_lengths // 3) // 3) // 3) // 3
        auxs = auxs.transpose(2, 3)
        for i in range(len(self.aux_enc)):
            auxs = self.aux_enc[i](auxs)  # [B, C, T, F]

        auxs = torch.stack(
            [
                torch.mean(aux[:, :aux_length, :], dim=(1, 2))
                for aux, aux_length in zip(auxs, aux_lengths)
            ],
            dim=0,
        )  # [B, C]
        auxs = self.out_conv(auxs)
        return auxs, self.speaker(auxs)


class FiLM(nn.Module):
    def __init__(self, feature_dim, cond_dim):
        super(FiLM, self).__init__()
        self.gamma_fc = nn.Linear(cond_dim, feature_dim)
        self.beta_fc = nn.Linear(cond_dim, feature_dim)

        # Debug counters for FiLM analysis
        self._forward_count = 0
        self._log_interval = 50

        # Register gradient hooks for FiLM layers
        self.gamma_fc.register_backward_hook(self._log_gamma_gradients)
        self.beta_fc.register_backward_hook(self._log_beta_gradients)

    def _log_gamma_gradients(self, module, grad_input, grad_output):
        """Log gamma (scale) gradient statistics for speaker conditioning debugging"""
        if self._forward_count % self._log_interval == 0 and grad_output[0] is not None:
            grad = grad_output[0]  # Gradient w.r.t. gamma_fc output
            logging.info(f"🎭 FILM GAMMA GRADIENTS: mean={grad.mean().item():.2e}, "
                        f"std={grad.std().item():.2e}, norm={grad.norm().item():.2e}")

    def _log_beta_gradients(self, module, grad_input, grad_output):
        """Log beta (bias) gradient statistics for speaker conditioning debugging"""
        if self._forward_count % self._log_interval == 0 and grad_output[0] is not None:
            grad = grad_output[0]  # Gradient w.r.t. beta_fc output
            logging.info(f"🎭 FILM BETA GRADIENTS: mean={grad.mean().item():.2e}, "
                        f"std={grad.std().item():.2e}, norm={grad.norm().item():.2e}")

    def forward(self, cond, x):
        """
        x:    [B, C, T, F] or [B, C]
        cond: [B, cond_dim]
        """
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        if not hasattr(self, "printed"):
            logging.info("--- Inspecting FiLM Layer ---")
            logging.info(
                f"Gamma (scale for audio): mean abs = {gamma.abs().mean():.2e}, max abs = {gamma.abs().max():.2e}"
            )
            logging.info(
                f"Beta (bias from spk):  mean abs = {beta.abs().mean():.2e}, max abs = {beta.abs().max():.2e}"
            )
            logging.info("-----------------------------")
            self.printed = True
        # ===========================
        return gamma * x + beta


class FusionModule(nn.Module):
    def __init__(self, emb_dim, nhead=4, dropout=0.1):
        super(FusionModule, self).__init__()
        self.nhead = nhead
        self.dropout = dropout
        param_size = [1, 1, emb_dim]

        self.attn = nn.MultiheadAttention(
            emb_dim, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.fusion = nn.Conv2d(emb_dim * 2, emb_dim, kernel_size=1)
        self.alpha = Parameter(torch.Tensor(*param_size).to(torch.float32))

        nn.init.zeros_(self.alpha)

    def forward(self, aux: torch.Tensor, esti: torch.Tensor) -> torch.Tensor:
        B, C, F, T = esti.shape

        aux = aux.unsqueeze(1)  # [B, 1, C]
        flatten_esti = esti.flatten(start_dim=2).transpose(1, 2)  # [B, T*F, C]
        # flatten_esti = esti

        aux_adapt = self.attn(aux, flatten_esti, flatten_esti, need_weights=False)[0]
        aux = aux + self.alpha * aux_adapt  # [B, 1, C]

        aux = aux.unsqueeze(-1).transpose(1, 2).expand_as(esti)
        esti = self.fusion(torch.cat((esti, aux), dim=1))  # [B, C, T, F]

        return esti


class EnUnetModule(nn.Module):
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


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        self.eps = eps

        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))

        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))

        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / (std_ + self.eps)) * self.gamma + self.beta

        return x_hat


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        self.eps = eps

        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))

        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))

        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,1]
        x_hat = ((x - mu_) / (std_ + self.eps)) * self.gamma + self.beta

        return x_hat


if __name__ == "__main__":
    model = MCxTFGridNet()
    audio = torch.rand(1, 65, 1000, 2)
    aux = torch.rand(1, 65, 1000, 2)
    aux_lens = torch.tensor([1000])
    out = model(audio, aux, aux_lens)
