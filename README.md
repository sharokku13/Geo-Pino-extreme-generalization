# Geo-PINO: Geometry-Adaptive Physics-Informed Neural Operators for Extreme OOD Aerodynamic Surrogate Modeling

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

> **Manuscript:** *Aerodynamic Generalization Limits of Geometry-Adaptive Physics-Informed Neural Operators under Extreme Regime Shifts* (Dosanbekov & Altai, 2026).

---
## Overview

* **The Challenge:** Traditional Finite Volume Method (FVM) solvers (e.g., OpenFOAM) require intensive, geometry-conforming mesh generation and iterative convergence loops, posing a severe computational bottleneck for high-throughput aerodynamic design optimization.
* **The Solution:** Geo-PINO decouples geometric complexity from spectral learning. By mapping irregular, body-fitted physical domains into a regularized reference tensor space via a diffeomorphic coordinate transformation network, it allows a Fourier Neural Operator (FNO) backbone to resolve flow fields with significantly reduced computational overhead.
* **Performance:** Achieves a massive deterministic acceleration factor over single-core FVM solvers (4.80 ms inference latency per spatial configuration) while enforcing physical mass and momentum conservation laws via exact automatic differentiation.

---
## Mathematical Foundations & Loss Objectives

Geo-PINO is optimized using a composite loss function balancing empirical data fidelity ($L^2$ fields) with structural PDE residuals:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{PDE}}$$

### 1. Data-Driven Loss ($L^2$ Norm)
The empirical target minimizes the point-wise mean squared deviation across predicted velocity, static pressure, and turbulent kinematic viscosity fields:

$$\mathcal{L}_{\text{data}} = \frac{1}{N} \sum_{i=1}^N \left( \| \hat{\mathbf{u}}_i - \mathbf{u}_i \|^2_{L^2} + \| \hat{p}_i - p_i \|^2_{L^2} + \| \hat{\nu}_{t,i} - \nu_{t,i} \|^2_{L^2} \right)$$

### 2. Physics-Informed RANS Residual Pullback
Let $\mathbf{x} = (x, y)$ define physical coordinates and $\boldsymbol{\xi} = (\xi_1, \xi_2)$ represent uniform reference coordinates. Spatial gradients are mapped via the chain rule using the Jacobian of the diffeomorphism $J_{\Phi}$:

$$\nabla_{\mathbf{x}} f = J_{\Phi}^T \nabla_{\boldsymbol{\xi}} f$$

The incompressible, steady-state RANS mass conservation and momentum equations are enforced in the physical frame to compute the structural regularizer $\mathcal{L}_{\text{PDE}}$:

$$\mathcal{R}_{\text{mass}} = \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y} = 0$$

$$\mathcal{R}_{\text{mom},x} = u_x\frac{\partial u_x}{\partial x} + u_y\frac{\partial u_x}{\partial y} + \frac{\partial p}{\partial x} - \nu_{\text{eff}}\left(\frac{\partial^2 u_x}{\partial x^2} + \frac{\partial^2 u_x}{\partial y^2}\right) - 2\frac{\partial\nu_{\text{eff}}}{\partial x}\frac{\partial u_x}{\partial x} - \frac{\partial\nu_{\text{eff}}}{\partial y}\left(\frac{\partial u_x}{\partial y} + \frac{\partial u_y}{\partial x}\right) = 0$$

$$\mathcal{L}_{\text{PDE}} = \| \mathcal{R}_{\text{mass}} \|_{L^2}^2 + \| \mathcal{R}_{\text{mom}} \|_{L^2}^2$$

---

## Performance Benchmarks

### Double Out-of-Distribution (OOD) Generalization
Models were trained on low-frequency symmetric shapes and evaluated under simultaneous geometric extrapolation (unseen asymmetric airfoils) and aerodynamic regime shifts (deep stall boundaries, $\alpha \ge 18.67^\circ$, at $Re \approx 2 \times 10^6$).

| Model Architecture | Velocity Relative $L_2$ | Pressure Relative $L_2$ | Inference Time | Physical Consistency |
| :--- | :---: | :---: | :---: | :---: |
| Standard FNO | $14.28\%$ | $18.92\%$ | **2.8 ms** | Violates Continuity |
| U-Net Baseline | $9.41\%$ | $11.05\%$ | 5.2 ms | Unstable Boundaries |
| **Geo-PINO (Ours)** | **3.10%** | **14.78%** | **4.8 ms** | **Strictly Constrained** |
| OpenFOAM (CFD) | — | — | 142.5 s | Exact (Converged) |

---

## Quick Start & Installation
### 1. Environment Setup
git clone [https://github.com/sharokku13/Geo-Pino-extreme-generalization.git](https://github.com/sharokku13/Geo-Pino-extreme-generalization.git)
cd Geo-Pino-extreme-generalization
pip install -r requirements.txt

### 2. Dataset Pipeline
Preprocess the AirfRANS mesh data into uniform computational grid tensors:
python data/preprocess.py --source data/airfrans/ --target data/tensor_grid/

### 3. Execution
Hyperparameters are managed via YAML configurations in `configs/`.

# Train with active physical constraints
python train.py --config configs/base_config.yaml

# Evaluate checkpoint and export field visualizations
python evaluate.py --config configs/base_config.yaml --checkpoint weights/geopino_best.pt

---

## Limitations & Scope
1. **Incompressible Flow Boundary:** The current formulation assumes constant fluid density ($\rho$). Compressible regimes and shock wave discontinuities are outside the current scope.
2. **Geometric Differentiability:** The bijective mapping requires smooth, continuous boundary definitions. Sharp interior junctions or multi-element profiles can introduce coordinate singularities during Jacobian evaluation.
3. **Turbulence Closure Epistemic Uncertainty:** Output fields inherit the time-averaged steady-state approximations of the Spalart-Allmaras turbulence closure present in the baseline data.

---
## Citation
@article{geopino2026,
  author       = {Dosanbekov, Dias and Altai, Abilkair},
  title        = {Aerodynamic Generalization Limits of Geometry-Adaptive Physics-Informed Neural Operators under Extreme Regime Shifts},
  year         = {2026},
  month        = {jun},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXX},
  url          = {[https://github.com/sharokku13/Geo-Pino-extreme-generalization](https://github.com/sharokku13/Geo-Pino-extreme-generalization)}
}
```
