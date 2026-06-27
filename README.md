# Geo-PINO: Geometry-Adaptive Physics-Informed Neural Operators for Extreme OOD Aerodynamic Surrogate Modeling

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXX-blue.svg)](https://zenodo.org/)

> **Paper:** Aerodynamic Generalization Limits of Geometry-Adaptive Physics-Informed Neural Operators under Extreme Regime Shifts (Dosanbekov & Altai, 2026).

---

## Abstract

Despite the empirical success of continuous neural operators in interpolative continuum mechanics tasks, their mathematical generalizability boundaries under concurrent geometric extrapolation and extreme out-of-distribution (OOD) aerodynamic regime shifts remain systematically understudied. This repository provides a rigorous quantitative framework for a geometry-adaptive physics-informed neural operator (Geo-PINO) subjected to a double-OOD verification: geometric extrapolation utilizing unseen airfoil topologies, and aerodynamic regime shifts forcing extrapolation into deep stall boundaries ($\alpha \ge 18.67^\circ$) utilizing the high-fidelity AirfRANS dataset ($Re \approx 2 \times 10^6$). 

By mapping complex physical domains into a regularized reference space via diffeomorphic mapping, the framework enables the Fourier Neural Operator (FNO) backbone to resolve Reynolds-Averaged Navier-Stokes (RANS) equations with significantly reduced computational overhead while preserving high-fidelity gradient resolution within flow separation zones.

---

## Methodology & Objective Formulation

The architecture implements a coordinate transformation $\Phi$ mapping the irregular, body-fitted physical mesh onto a uniform reference tensor grid. The fluid operator is evaluated in this invariant latent domain, while physical conservation laws are enforced in the physical space via exact Jacobian automatic differentiation.

### Loss Function Formulation

The framework is optimized using a hybrid objective function balancing empirical data fidelity ($L^2$ norm) with structural PDE residuals:

$$\mathcal{L}_{Total} = \mathcal{L}_{Data} + \lambda \mathcal{L}_{PDE}$$

**1. Data-Driven Loss ($L^2$ Norm):**
The empirical loss evaluates the deviation between the network predictions $(\hat{u}, \hat{p}, \hat{\nu}_t)$ and the ground truth numerical targets:

$$\mathcal{L}_{Data} = \frac{1}{N} \sum_{i=1}^N \left( \| \hat{u}_i - u_i \|^2_{L^2} + \| \hat{p}_i - p_i \|^2_{L^2} + \| \hat{\nu}_{t,i} - \nu_{t,i} \|^2_{L^2} \right)$$

**2. Physics-Informed RANS Residuals:**
The steady-state, incompressible RANS residuals are incorporated to enforce mass and momentum conservation. The spatial gradient of the total effective kinematic viscosity is explicitly accounted for:

$$\mathcal{L}_{PDE} = \| \nabla \cdot \hat{u} \|^2_{L^2} + \left\| (\hat{u} \cdot \nabla)\hat{u} + \nabla \hat{p} - \nabla \cdot ((\nu + \hat{\nu}_t)\nabla \hat{u}) \right\|^2_{L^2}$$

*Note: Spatial gradients $\nabla$ are computed with respect to the physical coordinates via the chain rule using the Jacobian of the diffeomorphism $J_{\Phi}$.*

---

## Quantitative Evaluation

Models were evaluated under severe OOD conditions, specifically focusing on unseen supercritical airfoil geometries operating at high angles of attack ($\alpha \ge 18.67^\circ$).

**Table 1: Generalization Performance on Unseen OOD Airfoils**

| Model Architecture | RMSE Velocity | RMSE Pressure | $R^2$ Drag | Inference Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| U-Net Baseline | 0.142 | 0.089 | 0.61 | 0.015 |
| Standard FNO | 0.115 | 0.062 | 0.74 | 0.012 |
| **Geo-PINO (Ours)** | **0.038** | **0.019** | **0.92** | **0.014** |
| OpenFOAM (CFD) | — | — | 1.00 | 142.50 |

*Geo-PINO demonstrates a substantial reduction in computational time compared to standard finite-volume CFD solvers (0.014s vs 142.5s) while maintaining strong predictive accuracy for aerodynamic coefficients within deep stall boundaries.*

---

## Quick Start & Installation

### 1. Environment Setup
Clone the repository and install the required dependencies:
git clone [https://github.com/sharokku13/Geo-Pino-extreme-generalization.git](https://github.com/sharokku13/Geo-Pino-extreme-generalization.git)
cd Geo-Pino-extreme-generalization
pip install -r requirements.txt

### 2. Dataset Pipeline
The framework utilizes the AirfRANS dataset format. Ensure the directory structure aligns with the following layout:
data/
└── airfrans/
    ├── train/
    ├── test_ood/
    └── preprocess.py

Execute the pre-processing script to construct the reference computational tensors:
python data/preprocess.py --source data/airfrans/ --target data/tensor_grid/

### 3. Execution
Hyperparameters are managed via YAML configurations in the configs/ directory.
To initiate training with PDE backpropagation:
python train.py --config configs/base_config.yaml

To evaluate a trained checkpoint and generate error mappings:
python evaluate.py --config configs/base_config.yaml --checkpoint weights/geopino_best.pt

### Scope and Limitations
The current incarnation of the Geo-PINO framework is constrained to work within the following limitations:Incompressible Flow Assumption: The PDE constraint expressions rely on a constant density ($\rho$). Transonic flows, shock waves formation, and compressible energy equations are currently out of scope.Differentiability of the Geometry: The diffeomorphic mapping $\Phi$ needs a continuously differentiable boundary. Geometries that induce coordinate singularities (e.g., multi-element airfoils with sharp slot angles) need a grid-stitching approach which is currently out of scope.Turbulence Modeling Dependencies: The ability to accurately predict the eddy viscosity field ($\nu_t$) is fundamentally tied to the quality of the Spalart-Allmaras turbulence model contained in the AirfRANS ground-truth dataset.CitationIf you use this framework in your work, please cite:
@article{dosanbekov2026geopino,
  author       = {Dosanbekov, Dias and Altai, Abilkair},
  title        = {Aerodynamic Generalization Limits of Geometry-Adaptive Physics-Informed Neural Operators under Extreme Regime Shifts},
  year         = {2026},
  month        = {jun},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXX},
  url          = {[https://github.com/sharokku13/Geo-Pino-extreme-generalization](https://github.com/sharokku13/Geo-Pino-extreme-generalization)}
}
