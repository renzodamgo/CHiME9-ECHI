# Debugging Logs Added for Speaker Imbalance Investigation

## Overview
Added comprehensive debugging logs to investigate why Speakers 1 & 2 are collapsing to silence while Speaker 0 performs well.

## 🔧 Debug Logs Added

### 1. Gradient Flow Analysis (`CausalMCxTFGridNet.py`)
**Location**: `_log_deconv_gradients()` method
**Trigger**: Every 50 forward passes
**Purpose**: Detect vanishing/exploding gradients per speaker channel

**Logs**:
```
🔍 DECONV GRADIENT ANALYSIS:
   Speaker 0 (ch 0:2): mean=X.XXe-XX, std=X.XXe-XX, max_abs=X.XXe-XX, norm=X.XXe-XX
   Speaker 1 (ch 2:4): mean=X.XXe-XX, std=X.XXe-XX, max_abs=X.XXe-XX, norm=X.XXe-XX  
   Speaker 2 (ch 4:6): mean=X.XXe-XX, std=X.XXe-XX, max_abs=X.XXe-XX, norm=X.XXe-XX
```

**Warnings**:
- `⚠️  VANISHING GRADIENTS detected in deconv layer!` (norm < 1e-8)
- `⚠️  EXPLODING GRADIENTS detected in deconv layer!` (norm > 100)

### 2. Output Magnitude Analysis (`CausalMCxTFGridNet.py`)
**Location**: Forward pass after `self.deconv(z)`
**Trigger**: Every 50 forward passes
**Purpose**: Track per-speaker channel output magnitudes

**Single Speaker Mode**:
```
🔍 DECONV OUTPUT ANALYSIS:
   Speaker 0 (ch 0:2): rms=X.XXXX, max_abs=X.XXXX, mean=X.XXXX
   Speaker 1 (ch 2:4): rms=X.XXXX, max_abs=X.XXXX, mean=X.XXXX
   Speaker 2 (ch 4:6): rms=X.XXXX, max_abs=X.XXXX, mean=X.XXXX
```

**Multi-Speaker Mode**:
```
🔍 MULTI-SPEAKER DECONV OUTPUT ANALYSIS:
   Target Speaker 0:
     Channel 0: rms=X.XXXX, max_abs=X.XXXX, mean=X.XXXX
     Channel 1: rms=X.XXXX, max_abs=X.XXXX, mean=X.XXXX
     ...
```

### 3. Speaker Data Balance Analysis (`joint_multi.py`)
**Location**: `joint_loss()` function
**Trigger**: Every 25 loss computations
**Purpose**: Detect data imbalance between speakers

**Logs**:
```
🔍 SPEAKER DATA BALANCE ANALYSIS:
   Speaker 0: ref_rms=X.XXXX, hat_rms=X.XXXX, ratio=X.XXX
   Speaker 1: ref_rms=X.XXXX, hat_rms=X.XXXX, ratio=X.XXX
   Speaker 2: ref_rms=X.XXXX, hat_rms=X.XXXX, ratio=X.XXX
```

**Status Messages**:
- `✅ Speaker data balance OK. Ratio: X.XX` (imbalance < 2.0)
- `⚠️  SPEAKER DATA IMBALANCE detected! Ratio: X.XX` (imbalance > 2.0)

### 4. STFT Preprocessing Analysis (`CausalMCxTFGridNet.py`) 
**Location**: Beginning of `forward()` method
**Trigger**: Every 50 forward passes
**Purpose**: Analyze input STFT statistics per speaker

**Logs**:
```
🔍 STFT PREPROCESSING ANALYSIS:
   Mixture spec shape: [B=X, M=X, D2=X, D3=X, RI=2]
   Mixture magnitude: mean=X.XXXX, max=X.XXXX, std=X.XXXX
   
   Multi-enrollment shape: [B=X, K=X, D2=X, D3=X]
     Speaker 0 magnitude: mean=X.XXXX, max=X.XXXX, std=X.XXXX
     Speaker 1 magnitude: mean=X.XXXX, max=X.XXXX, std=X.XXXX
     Speaker 2 magnitude: mean=X.XXXX, max=X.XXXX, std=X.XXXX
```

### 5. FiLM Layer Conditioning Analysis (`CausalMCxTFGridNet.py`)
**Location**: `FiLM.forward()` method  
**Trigger**: Every 50 forward passes
**Purpose**: Analyze per-speaker conditioning and gradient flow in FiLM layers

**Logs**:
```
🎭 FiLM CONDITIONING ANALYSIS:
   Conditioning embedding: mean=X.XXXX, std=X.XXXX, range=[X.XXXX, X.XXXX]
     Sample 0: cond_norm=X.XXXX, gamma_mean=X.XXXX, beta_mean=X.XXXX
     Sample 1: cond_norm=X.XXXX, gamma_mean=X.XXXX, beta_mean=X.XXXX
     Sample 2: cond_norm=X.XXXX, gamma_mean=X.XXXX, beta_mean=X.XXXX
   Gamma (scale): mean_abs=X.XXe-XX, max_abs=X.XXe-XX, std=X.XXe-XX
   Beta (bias):   mean_abs=X.XXe-XX, max_abs=X.XXe-XX, std=X.XXe-XX
   Input X: mean=X.XXXX, std=X.XXXX, max_abs=X.XXXX
   FiLM Output: mean=X.XXXX, std=X.XXXX, max_abs=X.XXXX

🎭 FILM GAMMA GRADIENTS: mean=X.XXe-XX, std=X.XXe-XX, norm=X.XXe-XX
🎭 FILM BETA GRADIENTS: mean=X.XXe-XX, std=X.XXe-XX, norm=X.XXe-XX
```

**Warnings**:
- `⚠️  FiLM GAMMA collapse for sample X! std=X.XXXX`
- `⚠️  FiLM BETA collapse for sample X! std=X.XXXX`
- `⚠️  FiLM GAMMA has low variation! range=X.XXXX`
- `⚠️  FiLM BETA has low variation! range=X.XXXX`

### 6. Speaker Embedding Quality Analysis (`CausalMCxTFGridNet.py`)
**Location**: After `self.aux_enc()` in both single and multi-speaker paths
**Trigger**: Every 50 forward passes  
**Purpose**: Analyze speaker embedding quality and distinguish ability

**Single Speaker**:
```
🎤 SPEAKER EMBEDDING ANALYSIS (Single):
   Embedding shape: [B, C]
   Embedding mean: X.XXXX, std: X.XXXX
   Embedding norm: X.XXXX
```

**Multi-Speaker**:
```
🎤 SPEAKER EMBEDDING ANALYSIS (Multi):
   Embedding shape: [BK, C] (BK=X, C=X)
     Speaker 0: mean=X.XXXX, std=X.XXXX, norm=X.XXXX
     Speaker 1: mean=X.XXXX, std=X.XXXX, norm=X.XXXX
     Speaker 2: mean=X.XXXX, std=X.XXXX, norm=X.XXXX
   Speaker 0-1 cosine similarity: X.XXXX
```

**Warnings**:
- `⚠️  SPEAKER EMBEDDING collapse detected! Low variation across features.`
- `⚠️  Speaker X EMBEDDING collapse! std=X.XXXX`
- `⚠️  SPEAKERS TOO SIMILAR! similarity=X.XXXX`
- `⚠️  SPEAKERS ANTI-CORRELATED! similarity=X.XXXX`

## 🎯 What to Look For

### 1. Gradient Flow Issues
- **Vanishing gradients**: Speakers 1 & 2 getting no gradient signal in deconv layer
- **FiLM gradient collapse**: Gamma/Beta gradients weak for certain speakers
- **Gradient imbalance**: Speaker 0 getting much stronger gradients than others

### 2. Output Magnitude Problems
- **Channel collapse**: Speakers 1 & 2 channels producing near-zero outputs
- **Magnitude ratios**: Large differences between speaker channel outputs

### 3. Data Imbalance
- **Reference RMS imbalance**: Some speakers much louder in training data
- **Model prediction ratios**: Poor hat_rms/ref_rms ratios for certain speakers

### 4. STFT Preprocessing Issues
- **Enrollment differences**: Some speakers having weaker STFT representations
- **Magnitude imbalances**: Preprocessing favoring certain speakers

### 5. FiLM Conditioning Problems
- **Conditioning collapse**: FiLM gamma/beta values too similar across speakers
- **Embedding similarity**: Speaker embeddings not sufficiently discriminative
- **FiLM effectiveness**: Low gamma/beta variation means weak speaker-specific conditioning

### 6. Speaker Embedding Issues
- **Embedding collapse**: All speakers getting similar embeddings
- **Low embedding variation**: Embeddings lack sufficient information content
- **Poor speaker discrimination**: High cosine similarity between different speakers

## 🚀 Usage

The debug logs will automatically appear during training. Key patterns to watch:

1. **Healthy training**: All speakers should have similar gradient norms and output magnitudes
2. **Hierarchy collapse**: Speaker 0 dominates while others vanish
3. **Data issues**: Consistent imbalances in reference RMS or STFT magnitudes

## 📊 Expected Impact

With these logs, you can now pinpoint the exact cause of the speaker imbalance:
- **Architecture**: Channel assignment working correctly?
- **Gradients**: Are all speakers getting training signal?
- **Data**: Is the training data balanced?
- **Preprocessing**: Are all speakers equally represented in STFT domain?

This comprehensive logging will reveal which of the suspected issues is the actual root cause.