<div align="center">

<br/>

```
██╗      █████╗ ████████╗███████╗███╗   ██╗████████╗     ██████╗ ██████╗ ███████╗
██║     ██╔══██╗╚══██╔══╝██╔════╝████╗  ██║╚══██╔══╝    ██╔═══██╗██╔══██╗██╔════╝
██║     ███████║   ██║   █████╗  ██╔██╗ ██║   ██║       ██║   ██║██║  ██║█████╗  
██║     ██╔══██║   ██║   ██╔══╝  ██║╚██╗██║   ██║       ██║   ██║██║  ██║██╔══╝  
███████╗██║  ██║   ██║   ███████╗██║ ╚████║   ██║       ╚██████╔╝██████╔╝███████╗
╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═══╝   ╚═╝        ╚═════╝ ╚═════╝ ╚══════╝
```

### *Predicting the Future of Diabetic Retinopathy through Neural ODEs in Latent Space*

<br/>

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=flat-square&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-7c6af7?style=flat-square)

<br/>

</div>

---

## ✦ Overview

This project combines a **deep convolutional autoencoder** with a **Neural Ordinary Differential Equation (Neural ODE)** solver to model the continuous-time progression of diabetic retinopathy from fundus images.

Rather than predicting a discrete next state, the model learns a **smooth trajectory through latent space** — allowing it to extrapolate how retinal pathology evolves over time by integrating an ODE in the compressed representation of the image.

```
Input Image  ──►  Encoder  ──►  z₀ (latent)  ──►  ODE Solver  ──►  z(t)  ──►  Decoder  ──►  Future State
                                                   dz/dt = f(z,t)
```

<br/>

## ✦ Key Features

| Feature | Description |
|---|---|
| 🔬 **Residual Autoencoder** | 4-stage encoder/decoder with residual blocks and BatchNorm for stable training |
| 🌊 **Latent Neural ODE** | Continuous dynamics modelled by a convolutional ODE function, solved with `dopri5` (adaptive step-size RK45) |
| 🖼️ **CLAHE Preprocessing** | Contrast-limited adaptive histogram equalization applied per-channel before encoding |
| 📊 **Latent Analysis** | PCA visualization of the spatial latent space + per-channel activation heatmaps |
| ⏱️ **Progression Timeline** | Decoded trajectory visualized at every ODE time step, not just the final prediction |
| 🌐 **Flask Web App** | Clean tabbed UI — upload a fundus image, get reconstruction, progression strip, and latent analysis |

<br/>

## ✦ Architecture

### Autoencoder

```
Input [3×512×512]
    │
    ▼
Conv2d(3→64) + ResBlock ──► MaxPool   [64×256×256]
Conv2d(64→128) + ResBlock ──► MaxPool  [128×128×128]
Conv2d(128→256) + ResBlock ──► MaxPool [256×64×64]
Conv2d(256→512) + ResBlock ──► MaxPool [512×32×32]
    │
    ▼
to_latent: Conv2d(512→16, k=1)   ← z₀  [16×32×32]
    │
    ▼  [ODE integration happens here]
    │
from_latent: Conv2d(16→512, k=1)
    │
    ▼
ConvTranspose2d ×4 + ResBlocks
    │
    ▼
Output [3×512×512]  ∈ [0, 1]
```

### Neural ODE Function

The ODE function `f(z, t)` is a lightweight convolutional network that operates entirely in latent space:

```
z [C×H×W]  ──►  Conv(C→128) + Tanh  ──►  Conv(128→128) + Tanh  ──►  Conv(128→C)  ──►  dz/dt
```

Integration is performed using `torchdiffeq.odeint` with the `dopri5` solver over `t ∈ [0.0, 0.5]` at 5 steps.

<br/>

## ✦ Project Structure

```
latent-ode-retinopathy/
│
├── app.py                   # Flask application & inference pipeline
│
├── models/
│   ├── __init__.py
│   ├── network.py           # Autoencoder (encoder, decoder, residual blocks)
│   ├── NODEs.py             # LatentODEModel wrapper around torchdiffeq
│   ├── small_NODEs.py       # ConvLatentODEFunc — the ODE dynamics function
│   ├── dataset.py           # LongitudinalFundusDataset with CLAHE preprocessing
│   └── best_model.pth       # Trained autoencoder weights
│
├── templates/
│   └── index.html           # Tabbed web UI
│
└── images/                  # Uploaded images (auto-created at runtime)
```

<br/>

## ✦ Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/latent-ode-retinopathy.git
cd latent-ode-retinopathy

# Install dependencies
pip install flask torch torchvision pillow opencv-python scikit-learn matplotlib torchdiffeq
```

### Run the App

```bash
python app.py
```

Then open your browser at **`http://localhost:3000`**

<br/>

## ✦ Web Interface

The app provides a **3-tab analysis dashboard** after each inference:

```
┌─────────────────────────────────────────────────────┐
│  [ Reconstruction ]  [ ODE Progression ]  [ Latent ] │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Tab 1 → Processed input vs. predicted future state  │
│  Tab 2 → Decoded frames at each ODE time step        │
│  Tab 3 → PCA scatter of z₀ + channel heatmaps        │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**Upload** any retinal fundus image → **Generate** → explore the three panels.

<br/>

## ✦ Dataset Format

The training dataset (`LongitudinalFundusDataset`) expects **longitudinal patient pairs**:

```
data/
├── patient_001/
│   ├── visit_1.png
│   └── visit_2.png
├── patient_002/
│   ├── visit_1.png
│   └── visit_2.png
...
```

Each visit image is preprocessed with **CLAHE** (clipLimit=2.0, tileGridSize=8×8) applied independently per RGB channel before tensor conversion.

<br/>

## ✦ Latent Space Analysis

After encoding an image to `z₀ ∈ ℝ^{16×32×32}`, the app provides two views:

**PCA Projection** — Each of the 32×32 = 1024 spatial locations is treated as a 16-dimensional vector (one value per channel). PCA reduces this to 2D, colored by activation magnitude. This reveals how the model distributes spatial information across the latent manifold.

**Channel Heatmaps** — The first 8 of 16 latent channels rendered as spatial activation maps, showing which image regions each latent feature responds to most strongly.

<br/>

## ✦ Dependencies

| Package | Purpose |
|---|---|
| `torch` / `torchvision` | Model definition, training, inference |
| `torchdiffeq` | Neural ODE integration (`dopri5` solver) |
| `flask` | Web application server |
| `opencv-python` | Image loading and CLAHE preprocessing |
| `scikit-learn` | PCA for latent space visualization |
| `matplotlib` | Analysis plots (PCA scatter, heatmaps) |
| `pillow` | Image format conversion and base64 encoding |

<br/>

## ✦ How It Works — Step by Step

1. **Upload** a fundus image via the web interface
2. **Preprocess** — resize to 512×512, apply CLAHE per channel, normalize to [0, 1]
3. **Encode** — pass through the convolutional encoder to get `z₀ ∈ ℝ^{16×32×32}`
4. **Integrate** — solve `dz/dt = f(z, t)` from `t=0` to `t=0.5` using `dopri5`, collecting `z(t)` at 5 steps
5. **Decode** — pass `z(t_final)` through `from_latent` → decoder to reconstruct the predicted future image
6. **Analyze** — visualize the full ODE trajectory and the latent structure of `z₀`

<br/>

## ✦ License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

<br/>

---

<div align="center">

*Built with PyTorch · torchdiffeq · Flask*

</div>
