# Step 1 — Enable “multi-joint” mode and build batches

### What the code does

In the training script, we **switch datasets** depending on the split. If the split (e.g., `"train"`) is listed in `dataloading.joint_for`, we use the multi-speaker dataset and its collate; otherwise we fall back to the single-speaker one.

```py
# train_script.py
def get_dataset(split, data_cfg, debug):
    joint_for  = set(getattr(data_cfg, "joint_for", []))  # e.g., ["train"]
    use_joint  = split in joint_for

    if use_joint:
        data = ECHIJoint(
            split, data_cfg.device, data_cfg.noisy_signal,
            data_cfg.ref_signal, data_cfg.rainbow_signal,
            data_cfg.sessions_file, data_cfg.segments_file, debug,
        )
        chosen_collate = collate_fn_joint
    else:
        data = ECHI(...)
        chosen_collate = collate_fn

    loader = DataLoader(data, **data_cfg.loader[split],
                        collate_fn=chosen_collate)
    return loader, samples
```

When **multi-joint** is on, the dataset yields **all three conversational targets** for the scene **plus** their enrollment (“Rainbow Passage”) audios:

```py
# echi.py (ECHIJoint → collate_fn_joint)
"""
Pads to:
  noisy         -> [B, C, Tw],    noisy_lens: [B]
  target_all    -> [B, K, Tw],    target_lens_all: [B, K]
  spkid_all     -> [B, K, Tr],    spkid_lens_all:  [B, K]
"""
MAX_TRAIN_SECS   = 4.0
MAX_ENROLL_SECS  = 6.0
max_samples = int(MAX_TRAIN_SECS  * fs)   # cap train segment length
max_enroll  = int(MAX_ENROLL_SECS * fs)   # cap enrollment length
```

- `noisy`: the device mixture (`C` channels).
    
- `target_all`: **K** clean references (one per talker in the scene; typically K=3).
    
- `spkid_all`: the **K** enrollment clips (Rainbow passages) for conditioning.
    

### Analogy

Think of each batch item as a **soup pot** (`noisy`) cooked from **K ingredients** (`target_all[k]`). You also get a **passport photo** for each ingredient (`spkid_all[k]`) so the model learns _which_ voice to bring forward and _how much_ of it should be in the pot.

### Why this matters

This step guarantees the model always sees:

- the **mixture**,
- **all voices** present (as targets), and
- short **identity snippets** for each voice (as conditioning).    

# Step 2 — STFT packing: shapes, lengths, and model-ready tensors

### What happens

We convert waveforms → complex STFTs, then **pack real/imag into channels** so the backbone sees `[B, 2×mics, T, F]`. We do this for:

- the **mixture** (multi-mic),
    
- **all K clean targets** (for losses),
    
- **all K enrollment clips** (for speaker conditioning).
    

### Code highlights (minimal + shape-safe)

```py
# joint_multi.py (prep step)
stft = STFT(n_fft=fft, hop=hop, win=win)          # wrapper you use everywhere

# 1) Mixture → STFT → pack RI into channels
X_c   = stft.forward_complex(noisy)               # [B, C, T, F] complex
X_ri  = torch.view_as_real(X_c)                   # [B, C, T, F, 2]
X     = X_ri.permute(0, 1, 4, 2, 3)               # [B, C, 2, T, F]
X     = X.reshape(B, C*2, T, F)                   # [B, 2*C, T, F]  ← model input

# 2) Targets (K speakers)
Y_ref_c = []
for k in range(K):
    yk_c = stft.forward_complex(target_all[:, k]) # [B, 1, T, F] or [B, C, T, F]
    Y_ref_c.append(yk_c)
Y_ref_c = torch.stack(Y_ref_c, dim=1)             # [B, K, C, T, F] complex

# 3) Enrollments ("Rainbow") for conditioning
E_c = []
for k in range(K):
    ek_c = stft.forward_complex(spkid_all[:, k])  # [B, 1, TrF, Fr]
    E_c.append(ek_c)
E_c = torch.stack(E_c, dim=1)                     # [B, K, 1, TrF, Fr] complex

# 4) Frame lengths (needed for masks/iSTFT-safe losses)
mix_len_w  = noisy_lens                           # [B] (samples)
tgt_len_w  = target_lens_all                      # [B, K] (samples)
to_frames  = lambda L: (L - stft.win)//stft.hop + 1
mix_len_f  = to_frames(mix_len_w)                 # [B]
tgt_len_f  = to_frames(tgt_len_w)                 # [B, K]

# 5) Speaker features tensor (channel-match trick)
#   Many backbones expect in_channels=2*C. We tile/align enroll feats to match.
spk_feat = torch.view_as_real(E_c)                # [B, K, 1, TrF, Fr, 2]
spk_feat = spk_feat.permute(0,1,2,5,3,4)          # [B, K, 1, 2, TrF, Fr]
spk_feat = spk_feat.reshape(B, K, 2, TrF, Fr)     # [B, K, 2, TrF, Fr]
spk_feat = spk_feat.repeat(1, 1, C, 1, 1)         # [B, K, 2*C, TrF, Fr]
```

> If your backbone takes a **single** conditioning stream per batch (not per-k), you’ll embed each `spk_feat[:, k]` first (Step 3), then fuse with FiLM inside the network.

### Why this is needed

- **Consistent channels**: packing real/imag → the front-end conv expects `in_channels = 2 × n_mics`.
    
- **Exact truncation**: frame lengths let losses/iSTFT ignore padded tails.
    
- **Clean alignment**: we keep **X** (mixture), **Y_ref_c** (targets), and **E_c / spk_feat** (conditioning) all time-freq aligned.
    

### Analogy

Think of STFT as turning raw audio into a **Lego wall** (time × frequency tiles).  
Packing real/imag is like snapping **two thin plates** into **one sturdy brick per mic** so the network can stack them cleanly.

—

# Step 3 — Speaker embeddings → FiLM conditioning

### What happens

Each enrollment clip (per speaker k) is turned into a **fixed-size embedding**. The separator then uses FiLM to **modulate** its feature maps with that embedding so the network “locks on” to that voice.

### Code highlights (essentials)

```py
# losses.py / joint_multi.py (conceptual)
# 1) Per-speaker enrollment → embedding
spk_embeds, spk_logits = [], []
for k in range(K):
    # spk_feat[:, k]: [B, 2*C, TrF, Fr]  (from Step 2)
    z_k, logits_k = aux_encoder(spk_feat[:, k])  # z_k: [B, emb_dim]
    spk_embeds.append(z_k)
    spk_logits.append(logits_k)  # optional CE loss if labels exist

# 2) Run the separator conditioned on each speaker embedding
S_hat_list = []
for k in range(K):
    S_hat_k = separator(X, cond=spk_embeds[k])    # mask or complex est for speaker k
    S_hat_list.append(S_hat_k)

S_hat = torch.stack(S_hat_list, dim=1)            # [B, K, T, F] (or [B,K,2*C,T,F])
```

Inside the model (CausalMCxTFGridNet), FiLM injects the embedding at **every layer**:

```py
# CausalMCxTFGridNet.__init__
self.fusions = nn.ModuleList([FiLM(emb_dim, emb_dim) for _ in range(n_layers)])

# CausalMCxTFGridNet.forward (sketch)
x = self.front_end(X)                              # [B, emb_dim, T, F]
for i in range(self.n_layers):
    x = self.fusions[i](cond_embed, x)             # FiLM: γ(cond) * x + β(cond)
    x = self.gridnets[i](x)                        # TF-GridNet block
out = self.head(x)                                 # → mask/estimate for 1 speaker
```

A minimal FiLM:

```py
class FiLM(nn.Module):
    def __init__(self, in_dim, ch):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 2*ch), nn.ReLU(), nn.Linear(2*ch, 2*ch))
    def forward(self, z, x):                        # z: [B,in_dim], x: [B,ch,T,F]
        gamma, beta = self.mlp(z).chunk(2, dim=-1)  # [B,ch], [B,ch]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)   # [B,ch,1,1]
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta
```

### Why this is needed

- The embedding is a **voice fingerprint**; FiLM turns it into per-channel **gain & bias** so layers emphasize features of _that_ voice and suppress others.
    

### Analogy

Think of FiLM as a **DJ’s EQ panel** that’s auto-tuned by the speaker embedding: it boosts the bands that match the target voice and dips the rest—at every stage of the mix.

—  
# Step 4 — Separation head: turn “who” into masks/estimates (one per speaker)

### What happens

For each target speaker kk, the separator returns either a **complex mask** (cRM) to filter the mixture, or a **direct complex estimate**. We then reshape back to complex tensors per mic.

### Code highlights (masking path + safe reshapes)

```py
# Helpers
def ri2c(x_ri, C):  # x_ri: [B, 2*C, T, F]
    B, _, T, F = x_ri.shape
    x_ri = x_ri.view(B, C, 2, T, F).permute(0,1,3,4,2)   # [B,C,T,F,2]
    return torch.view_as_complex(x_ri.contiguous())      # [B,C,T,F] complex

# Mixture in RI-channels from Step 2
# X: [B, 2*C, T, F]

S_hat_c_list = []
for k in range(K):
    # Model head outputs one speaker at a time conditioned by z_k
    M_k = separator(X, cond=spk_embeds[k])        # [B, 2*C, T, F]
    M_k = torch.tanh(M_k)                         # keep mask in [-1,1] (cRM)

    S_k_ri = M_k * X                              # masked RI-channels
    S_k_c  = ri2c(S_k_ri, C)                      # [B, C, T, F] complex
    S_hat_c_list.append(S_k_c)

S_hat_c = torch.stack(S_hat_c_list, dim=1)        # [B, K, C, T, F] complex
```

> If your head is **direct-estimate**, swap the three lines in the middle for:
> 
> ```py
> S_k_ri = separator(X, cond=spk_embeds[k])       # [B, 2*C, T, F]
> S_k_c  = ri2c(S_k_ri, C)                        # [B, C, T, F] complex
> ```

We keep everything **time–freq aligned** and **mic-preserving** so the loss can compare [B,K,C,T,F][B,K,C,T,F] to the reference.

### Why this is needed

- Producing **one stream per speaker** lets us compute per-speaker losses and combine them **jointly** (next step).
    
- Using cRMs is stable: you nudge the mixture toward the target rather than hallucinating it from scratch.
    

### Analogy

Think of the mask like a **stencil** laid over the mixture’s spectrogram: the stencil for speaker kk reveals _only_ the paint (energy) that belongs to kk.


# Step 5 — Joint losses (STFT + SI-SDR + optional speaker CE + mix-consistency)

### What happens

We train **all K speakers together** by combining:

- a **spectrogram loss** (shape-safe, complex/STFT side),
    
- a **time-domain SI-SDR** (after iSTFT, using exact per-speaker lengths),
    
- (optional) **speaker-ID cross-entropy**, and
    
- (optional) **mixture-consistency** (sum of estimates ≈ mixture).
    

### Code highlights (concise, length-safe)

```py
# losses.py
def l1_complex(A, B):                      # A,B: complex
    return (A.real - B.real).abs().mean() + (A.imag - B.imag).abs().mean()

def si_sdr_batch(s_hat, s, lens):          # s_hat,s: [B,T], lens: [B]
    # per-utterance crop → true length, then SI-SDR; mean over B
    eps = 1e-8
    out = []
    for b in range(s.shape[0]):
        L = int(lens[b].item())
        sh, gt = s_hat[b, :L], s[b, :L]
        alpha = (sh @ gt) / (gt @ gt + eps)
        e_true = alpha * gt
        e_nois = sh - e_true
        out.append(10.0 * torch.log10((e_true.pow(2).sum() + eps) /
                                      (e_nois.pow(2).sum() + eps)))
    return -torch.stack(out).mean()        # negative → minimize

def joint_loss(S_hat_c, Y_ref_c, X_c, batch, stft,
               w=(1.0, 1.0, 0.0, 0.0), ref_ch=0):
    """
    S_hat_c: [B,K,C,T,F] complex  (estimates)
    Y_ref_c: [B,K,C,T,F] complex  (targets)
    X_c    : [B,C,T,F]   complex  (mixture)
    batch keys: target_lens_all[B,K], speaker_labels[B,K] (optional)
    w = (w_sep, w_time, w_spk, w_cons)
    """

    B, K, C = S_hat_c.shape[:3]
    tgt_lens = batch["target_lens_all"]         # [B,K] (samples)

    # 1) STFT-domain separation loss (average over K)
    L_sep = 0.0
    for k in range(K):
        L_sep = L_sep + l1_complex(S_hat_c[:, k], Y_ref_c[:, k])
    L_sep = L_sep / K

    # 2) Time-domain SI-SDR (exact per-(B,K) lengths via iSTFT)
    L_time = 0.0
    for k in range(K):
        # iSTFT back to wave; use a fixed reference mic (C>1 → pick ref_ch)
        s_hat_wav = stft.inverse(S_hat_c[:, k])[:, ref_ch]   # [B,Tw]
        s_ref_wav = stft.inverse(Y_ref_c[:, k])[:, ref_ch]   # [B,Tw]
        L_time = L_time + si_sdr_batch(s_hat_wav, s_ref_wav, tgt_lens[:, k])
    L_time = L_time / K

    # 3) Optional speaker classification CE
    L_spk = torch.tensor(0.0, device=S_hat_c.device)
    if "spk_logits_all" in batch and "speaker_labels" in batch:
        # spk_logits_all: list/stack of K tensors, each [B,num_speakers]
        ce = torch.nn.CrossEntropyLoss()
        for k in range(K):
            L_spk = L_spk + ce(batch["spk_logits_all"][k], batch["speaker_labels"][:, k])
        L_spk = L_spk / K

    # 4) Optional mixture-consistency (sum_k estimate ≈ mixture)
    L_cons = torch.tensor(0.0, device=S_hat_c.device)
    if w[3] != 0.0:
        sum_est = S_hat_c.sum(dim=1)                # [B,C,T,F]
        L_cons = l1_complex(sum_est, X_c)

    # Total
    w_sep, w_time, w_spk, w_cons = w
    L_total = w_sep*L_sep + w_time*L_time + w_spk*L_spk + w_cons*L_cons
    return {"loss": L_total, "L_sep": L_sep, "L_time": L_time, "L_spk": L_spk, "L_cons": L_cons}
```

### Why this is needed

- **Jointness**: all speakers contribute every step, so the model learns balanced separation, not just “the easiest voice.”
    
- **Length-safety**: SI-SDR uses **true per-speaker wave lengths**, avoiding padded-tail penalties.
    
- **Stability**: STFT + SI-SDR blend gives both **spectral fidelity** and **waveform realism**; consistency ties estimates back to the observed mixture.
    

### Analogy

It’s like grading a choir: you score each singer’s **sheet accuracy** (STFT), their **live performance** (SI-SDR), optional **name-check** (CE), and whether their **sum sounds like the original chorus** (consistency).

—  



