# Hybrid GNN for Supply Chain Demand Forecasting

> A novel Graph Neural Network architecture combining local (GATv2) and global (TransformerConv) attention for next-day sales prediction in FMCG supply chains.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.3%2B-red)](https://pyg.org)
[![Dataset](https://img.shields.io/badge/Dataset-SCG%20(Wasi%20et%20al.%2C%202025)-green)](https://github.com/CIOL-SUST/SCG)

---

## Overview

This project extends the [SCG benchmark dataset](https://github.com/CIOL-SUST/SCG) (Wasi et al., 2025) with a custom hybrid GNN architecture and a temporally-aware data pipeline for supply chain demand forecasting. The model predicts next-day sales orders for 28 active FMCG products simultaneously using a graph of product relationships.

**Key result:** R² = 0.68, NRMSE = 0.071 — outperforming all baselines including LSTM, GAT-only, and Transformer-only models on the SCG dataset.

---

## Architecture

The core contribution is a **Hybrid GNN** that runs two parallel attention paths and fuses them:

```
Input (28 features per node)
        │
  [Input Projection]  Linear(28 → 64)
        │
   ┌────┴────┐
   │         │
[GATv2]  [TransformerConv]       ← 2 layers each, 4 heads
Local     Global attention
   │         │
   └────┬────┘
   [Concatenate]  → (128,)
   [Fusion Layer] LayerNorm + GELU + Dropout
   [Output Head]  Linear → 1
        │
   Predicted sales (next day)
```

- **GATv2 path** — captures local neighbourhood effects (products sharing the same storage location)
- **TransformerConv path** — captures global patterns across the entire product graph
- **Fusion** — simple concatenation + linear projection learns which path matters more per context

---

## Key Findings

### 1. Weekly temporal cycles dominate daily signals

Analysis of the SCG dataset revealed:

```
Lag-1 autocorrelation (today → tomorrow):     0.133  (weak)
Lag-7 autocorrelation (same day last week):   0.389  (3× stronger)
```

This motivated a **7-day sliding window** input that improved R² from 0.50 → 0.68 (+36%).

### 2. Storage Location edges outperform Plant edges

```
Plant edges (paper default):      R² = 0.50
Storage Location edges (ours):    R² = 0.68
```

Products sharing storage locations have stronger demand correlations than products sharing manufacturing plants.

### 3. 12 of 40 products are near-zero and distort metrics

Products with >80% zero sales were identified and excluded from evaluation. Including them artificially inflates R² because predicting zero for always-zero products looks like a good prediction.

### 4. Graph structure consistently outperforms no-graph baseline

```
LSTM (no graph):      R² = 0.511
GAT Only:             R² = 0.573
Transformer Only:     R² = 0.581
Hybrid GNN (ours):    R² = 0.676  ← best
```

---

## Results

| Model | R² | RMSE (scaled) | NRMSE |
|---|---|---|---|
| **Hybrid GNN (ours)** | **0.676** | **0.373** | **0.071** |
| Transformer Only | 0.581 | 0.424 | 0.081 |
| GAT Only | 0.573 | 0.429 | 0.082 |
| LSTM (no graph) | 0.511 | 0.458 | 0.088 |
| Product Mean Baseline | 0.472 | — | — |
| Persistence (today=tomorrow) | 0.128 | — | — |

---

## Dataset

This project uses the **SCG dataset** from:

> Wasi, A.T., Islam, M.S., Akib, A.R., & Bappy, M.M. (2025). *Graph Neural Networks in Supply Chain Analytics and Optimization: Concepts, Perspectives, Dataset and Benchmarks.* arXiv:2411.08550

- **Source:** Leading FMCG company in Bangladesh
- **Period:** January 1, 2023 – August 9, 2023 (221 days)
- **Products:** 40 SKUs (28 active after filtering)
- **Features:** Production, Sales Orders, Delivery to Distributors, Factory Issues
- **Dataset repo:** https://github.com/CIOL-SUST/SCG

---

## Project Structure

```
supply-chain-gnn/
├── data/                          # Place SCG dataset files here
│   ├── NodesIndex.csv
│   ├── Edges (Storage Location).csv
│   ├── Edges (Plant).csv
│   ├── Edges (Product Group).csv
│   ├── Sales Order.csv
│   ├── Production .csv
│   ├── Factory Issue.csv
│   └── Delivery To distributor.csv
│
├── supply_chain_gnn.ipynb         # Main notebook (run top to bottom)
│
├── baselines.py                   # Standalone baseline training script
│
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/supply-chain-gnn.git
cd supply-chain-gnn

# 2. Install dependencies
pip install torch torchvision torchaudio
pip install torch-geometric
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

**Tested with:**
- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- CUDA 11.8 (optional, CPU also works)

---

## Usage

### Option A — Jupyter Notebook (recommended)

```bash
jupyter notebook supply_chain_gnn.ipynb
```

Update `DATA_DIR` in Cell 2 to point to your data folder:
```python
DATA_DIR = r'path/to/your/data'
```

Then run all cells top to bottom. Cell order matters:
```
Cell 1  → Imports
Cell 2  → Data pipeline (edges, features, 7-day window)
Cell 3  → Scaling (train-only fit, no leakage)
Cell 4  → Build graph snapshots
Cell 5  → Model definitions
Cell 6  → Training helpers
Cell 7  → Train all 4 models
Cell 8  → Results table
Cell 9  → Visualizations
Cell 10 → Save predictions
```

### Option B — Baselines script

```bash
python baselines.py
```

---

## Data Pipeline

```
221 days × 40 products × 4 features
        ↓
Remove 12 dead products (>80% zero sales)
        ↓
221 days × 28 products × 4 features
        ↓
7-day sliding window
        ↓
213 samples × 28 products × 28 features  (7 days × 4 features)
        ↓
80/20 temporal split
        ↓
Train: 170 snapshots  |  Test: 43 snapshots
```

**Important:** The train/test split is strictly temporal — the model is always trained on earlier data and tested on later data. No shuffling.

---

## Graph Construction

Nodes are products. Edges connect products sharing the same **storage location** (3,046 edges across 28 active nodes). Self-loops are added so each node aggregates its own features. Edge weights are degree-normalized.

```python
# Storage Location edges — best performing edge type
edges_df = pd.read_csv('Edges (Storage Location).csv')

# Self-loops added for all nodes
from torch_geometric.utils import add_self_loops
ei_tensor, ew_tensor = add_self_loops(ei_tensor, ew_tensor,
                                       fill_value=1.0, num_nodes=N_NODES)
```

Edge type comparison:

| Edge Type | Edges | Isolated Nodes | R² |
|---|---|---|---|
| Storage Location | 3,046 | 0 | **0.68** |
| Plant | 1,647 | 5 | 0.50 |
| Product Group | 188 | 0 | not tested |

---

## Model Configuration

```python
# Hybrid GNN — best model
HybridGraphModel(
    in_channels    = 28,    # 7 days × 4 features
    hidden_channels= 64,
    num_layers     = 2,
    dropout        = 0.4
)

# Training
optimizer = AdamW(lr=3e-4, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(patience=10)
max_epochs = 200
patience   = 40   # early stopping
```

---

## Comparison with Original Paper

The SCG paper (Wasi et al., 2025) benchmarks standard temporal GNNs with default hyperparameters and Plant edges. This work extends it with:

| | SCG Paper | This Work |
|---|---|---|
| Architecture | DCRNN, TGCN, GConvGRU (off-the-shelf) | Custom Hybrid GNN (novel) |
| Input | Single-day snapshot (4 features) | 7-day window (28 features) |
| Edge type | Plant edges only | Storage Location (empirically chosen) |
| Dead product analysis | Not addressed | 12 products identified and excluded |
| GNN improvement over LSTM | ~9% RMSE reduction | ~18% RMSE reduction |

---

## Citation

If you use this code, please cite the original dataset paper:

```bibtex
@article{wasi2025scg,
  title={Graph Neural Networks in Supply Chain Analytics and Optimization:
         Concepts, Perspectives, Dataset and Benchmarks},
  author={Wasi, Azmine Toushik and Islam, MD Shafikul and
          Akib, Adipto Raihan and Bappy, Mahathir Mohammad},
  journal={arXiv preprint arXiv:2411.08550},
  year={2025}
}
```

---

## License

This project is released under the MIT License. The SCG dataset is available under CC BY 4.0 at [DOI: 10.5281/zenodo.13652826](https://doi.org/10.5281/zenodo.13652826).

---

## Acknowledgements

- SCG dataset by Wasi et al. (SUST & LSU)
- [PyTorch Geometric](https://pyg.org) for GNN implementations
- GATv2: Brody et al. (2022), TransformerConv: Shi et al. (2021)