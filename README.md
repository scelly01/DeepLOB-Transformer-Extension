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

- **Architecture matters:** MLP baseline fails (45% accuracy), while sequence-aware models reach ~67%, showing spatial + temporal structure is critical.

- **Transformer > LSTM (slightly, with fewer params):** TransformerLOB achieves 66.7% vs DeepLOB’s 65.8% with ~14% fewer parameters, highlighting efficiency of self-attention.

- **Complementary strengths enable ensembling:** DeepLOB and TransformerLOB have similar F1 (0.66) but different error patterns, making them effective for ensembling.

- **Ensemble performs best:** Combining both models achieves **67.2% accuracy (F1=0.669)**, outperforming individual models.

- **Stable results:** Performance consistent across runs (<1.5% variance), indicating robustness.


## Visualizations
<img width="1176" height="416" alt="image" src="https://github.com/user-attachments/assets/9b76f048-8e2d-418c-bb4c-f7feab25d46e" />

### Confusion Matrix
<img width="303" height="277" alt="image" src="https://github.com/user-attachments/assets/45297f37-8c6a-4306-b7ef-1d5c7f6b055e" />

