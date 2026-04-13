# DeepLOB: Limit Order Book Price Direction Prediction

> Reproducing and extending the DeepLOB architecture (Zhang et al., 2019) with a 
> Transformer encoder variant — achieving parameter-efficient parity on the FI-2010 benchmark.

---

## Problem Statement

Limit Order Books (LOBs) capture the full supply-demand picture of a financial market 
at the tick level. Each snapshot contains bid/ask prices and volumes at 10 price levels, 
updated thousands of times per second. The task: given 100 consecutive LOB snapshots, 
predict whether the mid-price will move **up**, **down**, or remain **stationary** 
over the next k timesteps (k=10 in this study).

This is a hard problem because:
- The signal is extremely noisy at the tick level
- Temporal and spatial structure both matter (price levels interact, sequences matter)
- Class imbalance: stationary events dominate real markets

---

## Dataset

**FI-2010** — the standard academic LOB benchmark (Ntakaris et al., 2018)  
- 5 Finnish stocks, 10 trading days
- 40 features: 10 bid/ask price + volume levels (normalized)
- 254,750 training samples | 139,487 test samples
- Labels: 3-class (Down / Stationary / Up) at 5 prediction horizons
- This study uses horizon k=10

---
## Architecture Overview

Four models trained and compared end-to-end:

### 1. MLP Baseline
Flattens the 100×40 window into a 4,000-dim vector through dense layers.
Lower bound — deliberately ignores sequential and spatial structure.

### 2. DeepLOB (reproduction)
Faithfully reproduces Zhang et al. (2019):
- **CNN Blocks**: Conv2D layers extract spatial patterns across price levels
- **Inception Block**: Parallel 1×1, 3×1, 5×1 convolutions capture multi-scale temporal features
- **LSTM**: 64 units capture long-range temporal dependencies across the 100-step sequence

### 3. TransformerLOB (extension)
Identical CNN + Inception front-end. Replaces LSTM with a Transformer encoder:
- Linear projection (960 → 128 dims)
- Multi-Head Self-Attention (4 heads, key_dim=32)
- Feed-Forward Network with residual connections + Layer Normalization
- Global Average Pooling (vs. LSTM's final hidden state)

### 4. Ensemble (best result)
Averages the probability outputs of DeepLOB and TransformerLOB at inference time —
no retraining required. The two models make complementary errors, making their
combination more accurate than either alone (67.2% vs 65.8% / 66.7% individually).

**Hypothesis**: Self-attention can learn which timesteps in the LOB sequence are most 
informative for price direction, without the sequential bottleneck of LSTMs. The ensemble will pick strenghts of the LSTM and add it on top of the Transformer variant strenght and as a result provide better accuracy.
 
---

## Results

## Results

| Model | Test Accuracy | Weighted F1 | Parameters |
|---|---|---|---|
| MLP Baseline | 0.454 | 0.417 | 1,067,139 |
| DeepLOB | 0.658 | 0.660 | 291,363 |
| TransformerLOB | 0.667 | 0.660 | 251,747 |
| **Ensemble (DeepLOB + Transformer)** | **0.672** | **0.669** | **543,110** |

### Per-class F1 — Ensemble

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Down | 0.56 | 0.52 | 0.53 |
| Stationary | 0.74 | 0.80 | 0.79 |
| Up | 0.58 | 0.50 | 0.52 |

---
## Key Findings

**1. Architecture is decisive — flattening destroys the signal.**
The MLP baseline (45.4% accuracy, F1=0.42) treats the 100×40 window as a flat
list, losing both spatial and temporal structure entirely. Its near-zero performance
on directional classes proves that naive approaches cannot capture the patterns
embedded in LOB data — the gap between 45% and 67% is purely architectural.

**2. TransformerLOB outperforms DeepLOB as a standalone model.**
TransformerLOB (66.7% accuracy) edges above DeepLOB (65.8%) with 14% fewer
parameters — 251k vs 291k. Self-attention's ability to directly connect any two
timesteps in the sequence, without the memory decay of sequential LSTM processing,
gives it a consistent advantage on this dataset. This finding held across multiple
independent training runs, suggesting it is structural rather than a random seed artefact.

**3. DeepLOB and TransformerLOB have identical weighted F1 — but different error patterns.**
Both models achieve weighted F1 of 0.660, yet their confusion matrices differ
meaningfully. The LSTM (DeepLOB) tends toward stronger Down recall while the
Transformer achieves slightly better Stationary precision. Same aggregate score,
different failure modes — exactly the condition that makes ensembling effective.

**4. The ensemble is the headline result — 67.2% accuracy, F1=0.669.**
Averaging the probability outputs of DeepLOB and TransformerLOB outperforms either
model alone: +1.4% over DeepLOB, +0.5% over TransformerLOB. The Stationary class
reaches F1=0.79 — higher than either model achieves individually. Directional
classes (Down F1=0.53, Up F1=0.52) are also the best across all four models.
This confirms that LSTM and self-attention capture genuinely complementary signals
in LOB data rather than redundant ones.

**5. Results are stable across training runs.**
This project was trained three times across different Kaggle sessions. TransformerLOB
consistently matched or outperformed DeepLOB, and the ensemble consistently produced
the best result. Variance across runs was under 1.5% on all metrics — indicating the
findings are robust to random initialisation, not an artefact of a lucky seed.

## Visualizations
<img width="1176" height="416" alt="image" src="https://github.com/user-attachments/assets/9b76f048-8e2d-418c-bb4c-f7feab25d46e" />

### Confusion Matrices
<img width="453" height="377" alt="image" src="https://github.com/user-attachments/assets/45297f37-8c6a-4306-b7ef-1d5c7f6b055e" />

