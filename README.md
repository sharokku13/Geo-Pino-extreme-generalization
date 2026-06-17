# Aerodynamic Generalization Limits of Geometry-Adaptive Physics-Informed Neural Operators under Extreme Regime Shifts

Official implementation of the Geometry-Adaptive Physics-Informed Neural Operator (Geo-PINO) framework. This repository contains the complete Science-Guided Machine Learning (SciML) pipeline designed to evaluate neural operator generalization boundaries and aerodynamic coefficient extrapolation under extreme Out-of-Distribution (OOD) regime shifts.

---

## Architecture Overview

The core framework bypasses the limitations of standard Fourier Neural Operators (FNO) on unstructured grids by introducing a geometry-adaptive multi-stage mapping pipeline:

1. Diffeomorphic Coordinate Mapping: A learnable grid deformation network (I_Phi) that transforms irregular boundary layer grids from physical space into a uniform, regularized canonical latent space.
2. Fourier Neural Operator Backbone: A 2D FNO block executing spectral convolutions via Fast Fourier Transforms (FFT) on the regularized grid to model global fluid dynamics.
3. Physics-Informed Constraints: Direct regularization during training using exact Automated Differentiation to compute incompressible Reynolds-Averaged Navier-Stokes (RANS) residuals.
4. Epistemic Uncertainty Estimation: Monte Carlo Dropout layers integrated within the spectral blocks to quantify model confidence during extrapolation.

---

## Repository Structure and Execution Order

The codebase is strictly modularized into sequential Jupyter Notebooks. To reproduce the model pipeline and evaluation benchmarks, process the files in the following order:

### 1. dataset.ipynb (Data Processing Pipeline)
* Parses unstructured VTU mesh files from the fluid dynamics dataset using PyVista.
* Generates exact Signed Distance Fields (SDF) to parameterize airfoil geometry boundaries.
* Executes Delaunay barycentric interpolation to map scattered physical fields onto a uniform 241 x 241 Cartesian grid.

### 2. models.ipynb (Network Architectures)
* Implements the learnable coordinate warping layers for grid regularization.
* Implements the FNO2d backbone with truncated low-frequency spectral weights.
* Implements MC-Dropout layers for uncertainty mapping.

### 3. train.ipynb (Physics-Informed Training Loop)
* Implements a Two-Phase Curriculum training sequence: empirical data warmup followed by physical PDE regularization.
* Formulates and backpropagates continuity and momentum RANS equations.
* Integrates adaptive gradient-norm balancing via Exponential Moving Average (EMA) to prevent loss cancellation.

### 4. evaluation.ipynb (OOD Benchmarking and Analysis)
* Performs geometry-grouped validation splits using GroupShuffleSplit to enforce strict structural isolation.
* Filters the test set to isolate extreme Out-of-Distribution flight regimes (high angles of attack).
* Computes differentiable lift (Cl) and drag (Cd) forces using an SDF boundary mollifier.
* Outputs absolute error fields for velocity and pressure components alongside Spearman rank correlation plots.

For the mathematical proofs, formal scaling derivations, and operational O-notation complexity charts, refer to THEORY.md.

---

## Metrics and Deliverables

Running the complete pipeline yields the following validation components saved directly to the output directory:
* Aerodynamic coefficient predictions comparing predicted vs. ground-truth lift and drag profiles.
* Field-wise absolute error visualizations for velocity (Ux, Uy) and pressure (P) fields.
* Epistemic uncertainty variance maps matching high-error flow separation zones.

---

## Maintainer
Developed and maintained by @sharokku and my friend Abilkair
