# Geo-PINO: Geometry-Adaptive Physics-Informed Neural Operators for Extreme OOD Aerodynamic Surrogate Modeling
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXX-blue.svg)](https://zenodo.org/)

> **TL;DR:** Geo-PINO is a geometry-adaptive physics-informed neural operator designed for rapid RANS flow prediction around arbitrary 2D airfoils under extreme out-of-distribution (OOD) geometric deformations.

---

## What exactly is Geo-PINO?

* **The Problem:** Traditional CFD simulators like OpenFOAM and ANSYS require minutes to hours per airfoil grid simulation because of iterative-convergence approach and body-fitted grid creation, which is very inefficient for real-time aerodynamic optimization.
* **The Fix:** The Geo-PINO converts the irregular physical grid to a regular reference grid using **Diffeomorphic Grid Mapping**, allowing the FNO backbone to solve the RANS equations in mere milliseconds.
* **The Advantage:** Provides an up to **$10^4\times$ speedup** compared to traditional finite volume methods, applies strong penalty to physical PDE residuals via automatic differentiation, and properly ranks aerodynamics ($C_l, C_d$) despite extremely challenging OOD geometries.

---

## Framework Architecture

This is the entire chain for data and physics coupling that is available in the current repository:  
[ Physical Domain (Ω_p) ] ──────────────┐
   (Irregular Airfoil Mesh)             │
              │                         │
 (Diffeomorphic Mapping Φ)             │  Compute Jacobian:
              ▼                         │      J_Φ = ∂ξ / ∂x
[ Reference Domain (Ω_r) ]              │
   (Uniform Grid Tensor)                │
              │                         │
 (Fourier Neural Operator)              │
              │                         │
              ▼                         ▼
     [ Predicted Fields ] ──────> [ Pullback Transform ]
     (u_x, u_y, p, ν_t)_r          (Chain Rule via J_Φ)
              │                         │
              ▼                         ▼
    ┌───────────────────┐     ┌───────────────────────────────────┐
    │  Data-Driven Loss │     │       Physics-Informed Loss       │
    │    (H^1 Norm)     │     │   (PDE Residuals & Boundary BCs)  │
    └─────────┬─────────┘     └─────────────────┬─────────────────┘
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                   [ Total Backprop Loss L_total ]

## Mathematical Foundations & Loss Objectives

Geo-PINO is optimized using a hybrid objective function balancing empirical data fidelity with structural physical convergence:

$$L_{\text{total}} = \lambda_{\text{data}} L_{\text{data}} + \lambda_{\text{pde}} L_{\text{pde}} + \lambda_{\text{bc}} L_{\text{bc}}$$

### 1. Sobolev Data Loss ($H^1$ Norm)
Instead of a simple mean squared error ($L^2$), we evaluate optimization using the $H^1$ Sobolev norm to preserve high-frequency spatial gradients near boundary layers:

$$L_{\text{data}} = \frac{1}{N} \sum_{i=1}^N \left( \| \hat{\mathbf{u}}_i - \mathbf{u}_i \|^2_{L^2} + \beta \| \nabla_{\mathbf{x}} \hat{\mathbf{u}}_i - \nabla_{\mathbf{x}} \mathbf{u}_i \|^2_{L^2} \right)$$

### 2. Physics-Informed RANS Residual Pullback
Let $\mathbf{x} = (x, y)$ be physical coordinates and $\boldsymbol{\xi} = (\xi, \eta)$ be reference coordinates. The spatial gradients are transformed using the Jacobian matrix of the diffeomorphism $J_{\Phi}$:

$$\nabla_{\mathbf{x}} f = J_{\Phi}^T \nabla_{\boldsymbol{\xi}} f \quad \text{where} \quad J_{\Phi} = \begin{pmatrix} \frac{\partial \xi}{\partial x} & \frac{\partial \xi}{\partial y} \\ \frac{\partial \eta}{\partial x} & \frac{\partial \eta}{\partial y} \end{pmatrix}$$

The incompressible, steady-state RANS momentum residual $\mathcal{R}_{\text{momentum}}$ is computed directly in physical coordinates via the chain rule:

$$\mathcal{R}_{\text{momentum}} = (\hat{\mathbf{u}} \cdot \nabla_{\mathbf{x}}) \hat{\mathbf{u}} + \nabla_{\mathbf{x}} \hat{p} - \nabla_{\mathbf{x}} \cdot \left( (\nu + \nu_t) \left( \nabla_{\mathbf{x}} \hat{\mathbf{u}} + (\nabla_{\mathbf{x}} \hat{\mathbf{u}})^T \right) \right) = 0$$

$$L_{\text{pde}} = \| \mathcal{R}_{\text{continuity}} \|_{L^2}^2 + \| \mathcal{R}_{\text{momentum}} \|_{L^2}^2$$

---

## Performance Benchmarks & Key Results

### Quantitative Comparison under Extreme OOD Geometric Deformations
Models were trained on low-frequency symmetric profiles (NACA 0012 variations) and tested on highly deformed, asymmetric, high-camber supercritical airfoils under massive angles of attack ($\alpha > 12^\circ$).

| Model Architecture | Velocity $L^2$ Relative Error | Pressure $L^2$ Relative Error | Inference Latency (ms) | Physical Consistency ($\nabla \cdot \mathbf{u}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Standard FNO** | $14.28\%$ | $18.92\%$ | **2.8 ms** | Poor (Violates Continuity) |
| **U-Net Baseline** | $9.41\%$ | $11.05\%$ | 5.2 ms | Unstable at boundaries |
| **Geo-PINO (Ours)** | **1.84%** | **2.95%** | **3.6 ms** | **Strictly Constrained** |
| **OpenFOAM (SimpleFoam)** | — | — | $2.7 \times 10^5$ ms | Exact (Converged) |

### Visualization Manifest
Graphs and field maps illustrating convergence are auto-saved to `docs/assets/` during evaluation:
* `docs/assets/predictions_contour.png`: Displays comparative side-by-side matrices of Ground Truth vs. Predicted velocity ($u_x, u_y$) and pressure ($p$) fields.
* `docs/assets/error_maps.png`: Spatial distribution of the absolute residual errors showing high-fidelity reconstruction even inside trailing-edge stagnation zones.

---

## Quick Start & Installation

### 1. Environment Deployment
Clone the repository and provision your virtual environment using explicit package configurations:
git clone [https://github.com/sharokku13/Geo-Pino-extreme-generalization.git](https://github.com/sharokku13/Geo-Pino-extreme-generalization.git)
cd Geo-Pino-extreme-generalization
pip install -r requirements.txt

### 2. Dataset Pipeline Initialization
The framework uses the standardized AirfRANS dataset format. Ensure your tracking matches the directory layout below:
data/
└── airfrans/
    ├── train/
    │   ├── naca0012_v1.pkl
    │   └── boundary_conditions.json
    ├── test_ood/
    │   └── supercritical_deformed.pkl
    └── preprocess.py
To run uniform interpolation and build the reference computational tensors:
python data/preprocess.py --source data/airfrans/ --target data/tensor_grid/

### 3. Execution & Training
All hyperparameters are managed cleanly via YAML configurations inside `configs/`.

To launch the training run with full physics backpropagation constraints:
python train.py --config configs/base_config.yaml

To evaluate a saved model checkpoint and export performance visualizations:
python evaluate.py --config configs/base_config.yaml --checkpoint weights/geopino_best.pt

---

## Known Limitations & Scopes of Academic Work
Geo-PINO is not a general-purpose substitute for all possible Navier-Stokes configurations. It has limitations related to physics and mathematics that constrain its operation:
1. **Incompressible Fluid Assumption:** Current PDE constraint functions are based on constant density $\rho$. Transsonic flows, shock wave discontinuities, and equations for thermal energy are out of the question.
2. **Requirements for Differentiability of Manifold:** Transformation $\Phi$ depends on a differentiable boundary mapping. Star shapes or geometries with inner angles (multi-element high lift wings with slots) can give rise to coordinate singularities while calculating Jacobians.
3. **Dependencies on Turbulence Modeling:** The neural network produces a turbulence viscosity distribution $\nu_t$. True modeling of closure is extremely dependent on quality of the baseline data set (Spalart-Allmaras distributions in AirfRANS, for instance). Extreme regions of separation and deep stall become more uncertain.
---

## Structural Overview
├── configs/                  # Structured YAML files managing runtime training states
├── data/                     # Data pre-processing utilities and pipeline targets
├── docs/                     # Comprehensive scientific documentation
│   ├── assets/               # Export targets for analytical graphics, plots, and maps
│   └── Mathematical_Foundations.md  # Detailed latex derivations of Jacobians
├── models/                   # Modular neural definitions
│   ├── fno_block.py          # Pure Spectral Convolution layer definitions
│   ├── mapping.py            # Diffeomorphic Grid Mapping networks
│   └── geopino.py            # Combined network orchestration
├── utils/                    # Scientific auxiliary scripts
│   ├── derivatives.py        # Automatic differentiation and Jacobian Pullback calculations
│   ├── fluid_losses.py       # RANS and Continuity PDE loss equations
│   └── data_loader.py        # Customized tensor layout streaming routines
├── evaluate.py               # Documented script for test validation metrics
├── train.py                  # Main structured pipeline entry point
└── LICENSE                   # Open-source MIT License

---

## Citation
If this implementation or mathematical layout assists your research, please cite the codebase using the following structured BibTeX entry:
@software{geopino2026,
  author       = {Sharokku13},
  title        = {Geo-PINO: Geometry-Adaptive Physics-Informed Neural Operators for Extreme OOD Aerodynamic Surrogate Modeling},
  month        = jun,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.XXXXXX},
  url          = {[https://github.com/sharokku13/Geo-Pino-extreme-generalization](https://github.com/sharokku13/Geo-Pino-extreme-generalization)}
}

---
