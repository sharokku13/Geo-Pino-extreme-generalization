# Aerodynamic Generalization Limits of Geometry-Adaptive Physics-Informed Neural Operators under Extreme Regime Shifts

Official Geo-PINO Framework Implementation. This GitHub repository includes the entire Science-Guided Machine Learning (SciML) pipeline that is implemented for determining the limits of neural operator generalizability and the extrapolation of aerodynamic coefficients in a highly OOD environment.

---

## Architecture Overview

The core framework bypasses the limitations of standard Fourier Neural Operators (FNO) on unstructured grids by introducing a geometry-adaptive multi-stage mapping pipeline:

1. Diffeomorphic Grid Mapping: Learnable network (I_ϕ) that warps the irregular grids of the boundary layer in the physical domain to be transformed into canonical latent regularized space.
2. Neural Fourier Operator backbone: 2D FNO architecture that applies convolutional operations through Fast Fourier Transformations on the grid space for global fluid flow modeling.
3. Physically-consistent constraints: Directly regularize model training through the application of exact Automatic Differentiation on incompressible RANS residuals.
4. Uncertainty quantification: Incorporation of Monte-Carlo Dropout layers in the convolutional blocks for uncertainty quantification during inference phase.

---

## Repository Structure and Execution Order

The codebase is strictly modularized into sequential Jupyter Notebooks. To reproduce the model pipeline and evaluation benchmarks, process the files in the following order:

### 1. dataset.ipynb (Data Preprocessing Pipeline)
* Reads raw VTU mesh files from the fluid dynamics dataset and processes them using PyVista.
* Constructs signed distance fields (SDFs) from exact coordinates defining airfoil shape boundaries.
* Applies Delaunay barycentric interpolation for scattering of scattered physical fields to 241 x 241 regular Cartesian grid.

### 2. models.ipynb (Model Architectures)
* Designs learnable coordinate warping layers for grid regularization.
* Designs the backbone for FNO2d with truncated low-frequency Fourier coefficients for weights.
* Designs MC-Dropout layers for modeling uncertainties.

### 3. train.ipynb (PDE Informed Training Routine)
* Designs a Two-Phase Curriculum training regime: empirical data pretraining followed by physics PDE-based regularization.
* Constructs equations for continuity and momentum RANS equations.
* Designs gradient normalization using Exponential Moving Average (EMA) to avoid loss cancellation problem.

### 4. evaluation.ipynb (OOD Evaluation Protocol)
* Uses GroupShuffleSplit to split data into groups corresponding to geometrical structures to ensure structural isolation.
* Filter OOD dataset for high AoA flight conditions.
* Differentiates lift (Cl) and drag (Cd) forces using signed distance field (SDF) boundary mollifier.
* Outputs absolute error fields for velocity and pressure and Spearman rank correlation plots.

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
