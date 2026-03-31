# Intent & Trajectory Prediction
### Hackathon — Problem Statement 1 | Behavioral AI & Temporal Modeling

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-yellow.svg)](https://colab.research.google.com)
[![Dataset](https://img.shields.io/badge/Dataset-nuScenes--mini-lightgrey.svg)](https://www.nuscenes.org)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Model Architecture](#2-model-architecture)
3. [Dataset Setup](#3-dataset-setup)
4. [Installation](#4-installation)
5. [How to Run](#5-how-to-run)
6. [Example Outputs](#6-example-outputs)
7. [Results](#7-results)
8. [Project Structure](#8-project-structure)
9. [Team](#9-team)

---

## 1. Project Overview

In an **Level 4 (L4) autonomous driving** environment, reacting to where a pedestrian *is* isn't enough — the vehicle must predict where they *will be*.

This project builds a deep learning model that:

- **Inputs:** 2 seconds of past (x, y) motion history for every pedestrian and cyclist in the scene
- **Outputs:** The 3 most likely future trajectories for each agent over the next 3 seconds
- **Accounts for social context:** How agents avoid, follow, and interact with each other
- **Evaluated on:** minADE@3 (average displacement error) and minFDE@3 (final displacement error)

### Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Coordinate frame | Per-sample ego normalisation | Removes global GPS scale, makes model rotation-invariant |
| Input features | 5-dim: (vx, vy, speed, cos_h, sin_h) | Explicit heading helps encoder distinguish direction |
| Social modelling | Multi-Head Attention over neighbours | Learns who to pay attention to, not a fixed radius pooling |
| Multi-modal output | CVAE latent sampling | Generates genuinely diverse paths, not just noisy copies |
| Training objective | Best-of-N + KL annealing + Diversity | Stable training, good coverage, no mode collapse |

---

## 2. Model Architecture

The model is a **Social Transformer + CVAE** with a GRU decoder. Total parameters: **668,684** (~2.5 MB).

```
nuScenes annotations
        │
        ▼
┌──────────────────────────────┐
│   Data Extraction Pipeline   │  nuscenes-devkit, sliding windows
│   (obs_len=4, pred_len=6)    │  2 Hz keyframe sampling
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Per-Sample Normalisation   │  origin = last obs position
│                              │  rotate so agent faces +x axis
└──────────────┬───────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌─────────────┐  ┌─────────────────┐
│ Agent       │  │ Neighbour       │  shared weights
│ Encoder     │  │ Encoders ×N     │  (up to 8 neighbours
│             │  │                 │   within 15m radius)
│ Transformer │  │ Transformer     │
│ Pre-LN      │  │ Pre-LN          │
│ 3 layers    │  │ 3 layers        │
│ 4 heads     │  │ 4 heads         │
│ d=128       │  │ d=128           │
└──────┬──────┘  └────────┬────────┘
       │                  │
       └────────┬─────────┘
                │
                ▼
┌──────────────────────────────┐
│   Social Attention Pooling   │  focal agent = query
│                              │  neighbours = key / value
│   MultiHeadAttention(4 heads)│  key-padding mask for
│   + FFN + LayerNorm          │  missing neighbours
└──────────────┬───────────────┘
               │
               ▼  context vector (B, 128)
┌──────────────────────────────┐
│   CVAE                       │
│                              │
│   Training:  posterior       │
│   q(z | ctx, future) ──►     │
│   KL[ q || p ] annealed      │
│   0 → 0.5 over 30 epochs     │
│                              │
│   Inference: prior           │
│   p(z | ctx) sampled K=3 ──► │
└──────────────┬───────────────┘
               │  z₁, z₂, z₃  (B, K, 32)
               ▼
┌──────────────────────────────┐
│   GRU Decoder (per sample)   │  autoregressively unrolls
│                              │  6 future steps
│   h₀ = tanh(Linear(ctx, z)) │  start token = (0,0)
│   GRUCell(prev_xy) → xy      │  in normalised frame
└──────────────┬───────────────┘
               │
               ▼
     Path 1, Path 2, Path 3
     (B, 3, 6, 2)  in metres
```

### Loss Function

```
Total Loss = Best-of-N(ADE + FDE)
           + kl_weight(epoch) × KL[q || p]
           + 0.3 × Diversity Loss

where:
  kl_weight(epoch) = min(0.5,  0.5 × epoch / 30)   ← linear warmup
  Diversity Loss   = −mean pairwise L2 between K predictions
```

The **Best-of-N** loss backpropagates only through the prediction with the lowest ADE — this directly optimises the minADE@K metric used by the hackathon judges.

---

## 3. Dataset Setup

### Download nuScenes Mini

1. Register for a free account at [nuscenes.org](https://www.nuscenes.org/sign-up)
2. Download **nuScenes-mini** (v1.0-mini, ~4 GB zip)
3. After unzipping, your folder structure should look like:

```
nuscenes/
├── v1.0-mini/
│   ├── attribute.json
│   ├── calibrated_sensor.json
│   ├── category.json
│   ├── ego_pose.json
│   ├── instance.json
│   ├── map.json
│   ├── sample.json
│   ├── sample_annotation.json
│   ├── sample_data.json
│   ├── scene.json
│   ├── sensor.json
│   └── visibility.json
├── samples/
│   ├── CAM_FRONT/
│   ├── CAM_BACK/
│   ├── LIDAR_TOP/
│   └── ...
├── sweeps/
└── maps/
```

> **Note:** This project uses only the JSON annotation files in `v1.0-mini/`. The camera images and LiDAR data in `samples/` are not required for training.

### What We Extract

From `sample_annotation.json` we extract:

| Field | Description |
|---|---|
| `instance_token` | Unique agent ID across frames |
| `category_name` | Filters to pedestrians + cyclists |
| `translation` | Global (x, y, z) position |
| `sample_token` | Links to timestamp via `sample.json` |

**Sliding window:** For each agent track, we slide a window of `obs_len + pred_len = 10` timesteps (2s observed + 3s future) across the track, yielding ~4,000 training windows from the mini split.

---

## 4. Installation

### Option A — Google Colab (Recommended, zero setup)

Everything runs in the provided notebook. No local installation needed. Jump to [Section 5](#5-how-to-run).

### Option B — Local / Windows (RTX A2000)

```bash
# 1. Clone the repo
git clone https://github.com/sujeet021/Intent-Trajectory-Prediction.git
cd Intent-Trajectory-Prediction

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / Mac

# 3. Install PyTorch (CUDA 12.x for RTX A2000)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install -r requirements.txt
```

**`requirements.txt`**

```
nuscenes-devkit>=1.1.11
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
scipy>=1.10.0
scikit-learn>=1.2.0
```

### Option C — Mac (M3 Air, MPS backend)

```bash
# PyTorch with Metal Performance Shaders
pip install torch torchvision
pip install -r requirements.txt

# In config, device will auto-select MPS:
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

> **Expected training time:** ~20–30 min (RTX A2000) · ~90 min (M3 Air MPS) · ~1.5 hrs (T4 Colab)

---

## 5. How to Run

### On Google Colab (Recommended)

**Step 1 — Upload dataset to Google Drive**

```
My Drive/
└── nuscenes-mini.zip    ← upload here
```

**Step 2 — Open the notebook**

Upload `trajectory_prediction_v2.ipynb` to Colab, or open it from this repo.

**Step 3 — Set runtime to T4 GPU**

```
Runtime → Change runtime type → T4 GPU → Save
```

**Step 4 — Run all cells in order**

```
Runtime → Run all   (or Ctrl+F9)
```

The notebook will:
1. Install dependencies automatically
2. Mount your Google Drive and unzip the dataset
3. Extract trajectory windows from nuScenes annotations
4. Train the Social Transformer + CVAE model for 150 epochs
5. Save the best checkpoint to `/content/checkpoints/best_model.pt`
6. Generate training curves, trajectory visualisations, and a results JSON
7. Copy all outputs to `My Drive/trajectory_v2/`

**Step 5 — Update the Drive path if needed**

In Cell 3, set:
```python
DRIVE_ZIP = '/content/drive/MyDrive/nuscenes-mini.zip'
```

---

### Local Run (after installation)

```bash
# Train from scratch
python train.py --dataroot /path/to/nuscenes --version v1.0-mini --epochs 150

# Evaluate best checkpoint
python evaluate.py --checkpoint checkpoints/best_model.pt --K 3

# Run inference on a single scene
python inference.py --checkpoint checkpoints/best_model.pt --visualise
```

---

### Config Reference (Cell 4 in notebook)

```python
CFG = {
    'obs_len'        : 4,      # 2 seconds observed @ 2 Hz
    'pred_len'       : 6,      # 3 seconds predicted @ 2 Hz
    'social_radius'  : 15.0,   # metres — neighbour search radius
    'max_agents'     : 8,      # max neighbours per agent
    'd_model'        : 128,    # transformer hidden dimension
    'n_heads'        : 4,      # attention heads
    'n_enc_layers'   : 3,      # transformer encoder layers
    'latent_dim'     : 32,     # CVAE latent space size
    'K'              : 3,      # number of predicted paths
    'epochs'         : 150,
    'batch_size'     : 128,
    'lr'             : 5e-4,
    'kl_weight_max'  : 0.5,
    'kl_warmup'      : 30,     # epochs to ramp KL weight 0 → 0.5
    'diversity_weight': 0.3,
}
```

---

## 6. Example Outputs

### Training Curves

After 150 epochs you should see:

- **Best-of-N loss** decreasing steadily from ~70 → ~43
- **minADE@3** decreasing from initial value and stabilising below 5m
- **KL loss** ramping up smoothly after warmup (no NaN spikes)
- **Diversity** increasing gradually — predictions becoming more spread

### Trajectory Visualisation

For each validation sample, the model outputs 3 colour-coded paths:

```
● Gray   dots/line   = Observed history (2 seconds, 4 points)
■ Black  dashed      = Ground truth future (3 seconds, 6 points)
▲ Purple solid       = Predicted Path 1
▲ Green  solid       = Predicted Path 2
▲ Orange solid       = Predicted Path 3
★                    = Predicted final position (endpoint of each path)
```

**Sample output (from validation set):**

```
Sample 1 | best ADE = 0.84 m    ← straight-walking pedestrian
Sample 2 | best ADE = 1.23 m    ← turning at intersection
Sample 3 | best ADE = 2.10 m    ← fast-moving cyclist
Sample 4 | best ADE = 1.67 m    ← pedestrian avoiding another agent
Sample 5 | best ADE = 0.92 m    ← stationary then moving
Sample 6 | best ADE = 1.45 m    ← group walking scenario
```

### Inference Output (single agent)

```python
predictions = predict_single(model, sample, K=3)

# Output:
Path 1: final position = (2.14 m,  0.38 m) | total distance = 3.21 m
Path 2: final position = (1.89 m,  1.12 m) | total distance = 2.98 m
Path 3: final position = (2.31 m, -0.54 m) | total distance = 3.45 m
```

### Results JSON

```json
{
  "model_version": "v2 — Social Transformer + CVAE",
  "dataset": "nuScenes-mini",
  "obs_seconds": 2.0,
  "pred_seconds": 3.0,
  "K": 3,
  "minADE@3": 4.7303,
  "minFDE@3": 8.1327,
  "params": 668684,
  "best_epoch": 1
}
```

---

## 7. Results

### Quantitative Metrics

| Metric | v1 Model | v2 Model (target) | SOTA (nuScenes full) |
|---|---|---|---|
| minADE@3 | 4.73 m | < 2.0 m | ~0.8 m |
| minFDE@3 | 8.13 m | < 4.0 m | ~1.5 m |
| Hit Rate (FDE ≤ 4m) | 26.4% | > 50% | — |
| Parameters | 668K | 668K | 20M+ |
| Inference time | < 5 ms | < 5 ms | — |

### Key Fixes from v1 → v2

| Issue in v1 | Root Cause | Fix in v2 |
|---|---|---|
| NaN losses epochs 1–7 | KL weight = 1.0 from start → 8M KL at epoch 1 | Linear warmup 0 → 0.5 over 30 epochs |
| ADE jumped to 50m | Global GPS coordinates fed to model | Per-sample ego-frame normalisation |
| Only 2,189 windows | Coordinate frame bug dropped tracks | Extract in global coords, normalise at getitem |
| CVAE mode collapse | KL explosion forced prior = posterior instantly | Warmup + smaller latent_dim (64→32) |
| Flat diversity for 60 epochs | Mode collapse — all K samples identical | KL fix resolves this directly |

---

## 8. Project Structure

```
Intent-Trajectory-Prediction/
│
├── README.md                          ← you are here
├── requirements.txt                   ← pip dependencies
│
├── trajectory_prediction_v2.ipynb    ← main Colab notebook (run this)
│
├── checkpoints/
│   └── best_model.pt                  ← saved after training
│
└── outputs/
    ├── training_curves_v2.png         ← loss + metric plots
    ├── predictions_v2.png             ← trajectory visualisations
    ├── error_dist_v2.png              ← ADE / FDE histograms
    └── results_v2.json                ← final metrics
```

---

## 9. Team

| Name | Role | Institution |
|---|---|---|
| Sujeet | Team Lead — Model Architecture | Manipal Institute of Technology, Bengaluru (M.Tech Data Science 2025–2027) |
| Akshay| Collaborator — Model Development, GitHub Integration | Manipal Institute of Technology, Bengaluru (M.Tech Data Science 2025–2027) |
| Nandini | Collaborator — Model Architecture, Training Pipeline | Manipal Institute of Technology, Bengaluru (M.Tech Data Science 2025–2027) |
| Shreya | Collaborator — Model Architecture, Training Pipeline | Manipal Institute of Technology, Bengaluru (M.Tech Data Science 2025–2027) |
**Stack:** Python · PyTorch · nuScenes-devkit · Google Colab T4 GPU

**Hackathon:** Computer Vision Challenge Round 1 — Problem Statement 1: Intent & Trajectory Prediction

---

*Repository: https://github.com/sujeet021/Intent-Trajectory-Prediction
