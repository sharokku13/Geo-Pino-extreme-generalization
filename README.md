# Aerodynamic Generalization Limits of Geometry-Adaptive Physics-Informed Neural Operators under Extreme Regime Shifts (Geo-PINO-v7)

Official implementation of the Geo-PINO-v7 framework. This repository contains an end-to-end Science-Guided Machine Learning (SciML) pipeline designed for high-fidelity aerodynamic flow prediction and aerodynamic coefficient extrapolation under extreme Out-of-Distribution (OOD) conditions.

---

## Overview

This project solves steady incompressible Reynolds-Averaged Navier-Stokes (RANS) equations over arbitrary 2D airfoil geometries. By embedding a learnable coordinate mapping network (I_Phi) into a 2D Fourier Neural Operator (FNO) backbone, the model successfully handles complex boundary surfaces and generalizes to unseen high angles of attack, where standard deep learning architectures fail due to severe flow separation and stall dynamics.

### Key Enhancements in v7:
* Stabilized Boundary Gradients: Expanded the smooth Signed Distance Field (SDF) boundary mollifier bandwidth parameter to epsilon = 0.5 for better numerical gradient propagation near sharp airfoil edges.
* Physics Loss Weighting: Integrated Wang's adaptive Exponential Moving Average (EMA) gradient-norm balancing to dynamically weight data-driven and physical PDE residuals.

---

## Repository Structure and Execution Order

The framework is structured into sequential Python components and Jupyter Notebooks. To reproduce the evaluation metrics, files should be processed in the following order:

1. dataset.ipynb
   * Extracts raw unstructured mesh data using PyVista.
   * Computes exact Signed Distance Fields (SDF) from airfoil geometries.
   * Performs Delaunay barycentric interpolation onto uniform Cartesian grids (241 x 241).

2. models.ipynb
   * Constructs the Diffeomorphic Coordinate Warping network (I_Phi).
   * Implements the Spectral Convolution Layers (FNO2d) using 2D Real FFT.
   * Integrates Monte Carlo Dropout for epistemic uncertainty quantification.

3. train.ipynb
   * Implements Two-Phase Curriculum learning (Empirical Warmup followed by Physics-Informed regularization).
   * Enforces exact no-slip boundary conditions and balances Navier-Stokes residuals.

4. evaluation.ipynb
   * Executes geometry-grouped cross-validation via GroupShuffleSplit to eliminate structural data shape leakage.
   * Isolates the Out-of-Distribution (OOD) test set based on an Angle of Attack (AoA) percentile threshold.
   * Computes differentiable lift (Cl) and drag (Cd) forces via the SDF mollifier.
   * Generates absolute error fields for Velocity (Ux) and Pressure (P), alongside Spearman rank correlation profiles.

For detailed mathematical proofs, fluid mechanics formulations, and formal scaling laws, refer to THEORY.md.

---

## Out-of-Distribution (OOD) Benchmarking

The model is evaluated strictly on extreme flight regimes (high angles of attack representing heavy stall states) excluded from the training dataset:

* Drag Coefficient (Cd) Extrapolation: R² = 0.8904
* Aerodynamic Trend Realism: Spearman Rank Correlation rho = 0.9775

All loss logs, prediction fields, and validation plots (aero_coeffs.png, error_maps.png, pred_vs_gt.png) are automatically compiled into the output directory upon running the evaluation pipeline.

---

## Maintainer
Developed and maintained by @sharokku.
