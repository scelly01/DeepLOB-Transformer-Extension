# DeepLOB-Transformer-Extension

"""# DeepLOB: Limit Order Book Price Direction Prediction

Reproducing and extending the DeepLOB architecture (Zhang et al., 2019) with a Transformer encoder variant — achieving parameter-efficient parity on the FI-2010 benchmark.

Problem Statement
Limit Order Books (LOBs) capture the full supply-demand picture of a financial market at the tick level. Each snapshot contains bid/ask prices and volumes at 10 price levels, updated thousands of times per second. The task: given 100 consecutive LOB snapshots, predict whether the mid-price will move up, down, or remain stationary over the next k timesteps (k=10 in this study).

This is a hard problem because:

The signal is extremely noisy at the tick level
Temporal and spatial structure both matter (price levels interact, sequences matter)
Class imbalance: stationary events dominate real markets
Dataset
FI-2010 — the standard academic LOB benchmark (Ntakaris et al., 2018)

5 Finnish stocks, 10 trading days
40 features: 10 bid/ask price + volume levels (normalized)
254,750 training samples | 139,487 test samples
Labels: 3-class (Down / Stationary / Up) at 5 prediction horizons
This study uses horizon k=10
Architecture Overview
Three models trained and compared end-to-end:

1. MLP Baseline
Flattens the 100×40 window into a 4,000-dim vector and passes through dense layers. Serves as the lower bound — deliberately ignores sequential and spatial structure.

2. DeepLOB (reproduction)
Faithfully reproduces Zhang et al. (2019):

CNN Block 1 & 2: Conv2D layers extract spatial patterns across price levels
Inception Block: Parallel 1×1, 3×1, 5×1 convolutions capture multi-scale temporal features
LSTM: 64 units capture long-range temporal dependencies across the 100-step sequence
3. TransformerLOB (extension)
Identical CNN + Inception front-end as DeepLOB.
Replaces the LSTM with a Transformer encoder block:

Linear projection (960 → 128 dims)
Multi-Head Self-Attention (4 heads, key_dim=32)
Feed-Forward Network with residual connections + Layer Normalization
Global Average Pooling (vs. LSTM's final hidden state)
Hypothesis: Self-attention can learn which timesteps in the LOB sequence are most informative for price direction, without the sequential bottleneck of LSTMs.

Results
Model	Test Accuracy	Weighted F1	Parameters	Train Time
MLP Baseline	0.371	0.350	1,067,139	1.5 min
DeepLOB	0.672	0.668	291,363	15.3 min
TransformerLOB	0.674	0.665	251,747	13.1 min
Per-class F1 scores
Model	Down F1	Stationary F1	Up F1
MLP Baseline	0.32	0.49	0.01
DeepLOB	0.53	0.78	0.53
TransformerLOB	0.51	0.79	0.51
Key Findings
1. Architecture is decisive.
MLP at 37.1% accuracy (F1=0.35) barely exceeds random chance (33.3%). Critically, it achieves near-zero F1 on the Up class — failing entirely at directional prediction. This confirms that flattening destroys the temporal and spatial structure essential for LOB modeling.

2. TransformerLOB achieves parameter-efficient parity.
The Transformer variant matches DeepLOB's accuracy (+0.42%) with 13.6% fewer parameters and 2.2 minutes faster training. This suggests self-attention is at least as capable as LSTMs for LOB sequence modeling, with better computational efficiency.

3. Directional prediction is the hard problem.
Both DeepLOB and TransformerLOB achieve F1 ~0.53 on Down and ~0.51 on Up — significantly above baseline but with clear room for improvement. Stationary events (F1 ~0.79) are predicted reliably by both models. The Transformer shows slightly higher stationary recall (0.83 vs 0.80), trading off some directional sensitivity for overall stability.

4. LSTM vs Attention tradeoff.
DeepLOB shows higher recall on Down events (0.49 vs 0.45), suggesting the LSTM's sequential processing captures some directional momentum signals the attention mechanism handles differently. This points to a potential hybrid architecture (CNN + Attention + LSTM) as future work.

Visualizations
Confusion Matrices
