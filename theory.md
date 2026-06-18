# Theoretical Analysis, Algorithmic Framework, and Computational Complexity of Geo-PINO

This document provides a rigorous mathematical and algorithmic analysis of the Geometric Physics-Informed Neural Operator (Geo-PINO). It summarizes the model pipeline, the training strategy, the computational complexity, and an interpretation of the validation results.

## 1. Problem Formulation and Architectural Pipeline

The model is designed to predict the steady-state velocity field $\mathbf{u} = [U_x, U_y]^T$ and pressure field $P$ of an incompressible fluid governed by the Reynolds-Averaged Navier-Stokes (RANS) equations.

The architecture processes data through three sequential spatial transformations:

1. **Physical Domain $\Omega$**: The input tensor $X \in \mathbb{R}^{N \times d}$ contains $d$ features, including spatial coordinates, geometry representations, angle of attack $\alpha$, and Reynolds number $Re$, sampled across $N$ nodes of an irregular mesh.

2. **Diffeomorphic Mapping $\phi^{-1}$**: A coordinate-mapping multilayer perceptron (MLP) projects the irregular physical nodes into a structured, uniform latent space parameterized as a two-dimensional torus $\mathbb{T}^2$ with resolution $H \times W$. This operation yields the latent tensor $V_0 \in \mathbb{R}^{H \times W \times d_v}$, where $d_v$ denotes the latent dimensionality.

3. **Spectral Operator $\mathcal{G}_\theta$**: A Fourier Neural Operator (FNO) layer applies a 2D Fast Fourier Transform (FFT) to the latent tensor, truncates higher-frequency modes to retain only the lowest modes $|k| \le k_{\text{max}}$, multiplies them by a parameterized complex weight tensor $R_\phi$, and maps them back via an inverse 2D FFT (iFFT). In parallel, a local linear path ($1 \times 1$ convolution) preserves local features. The outputs are summed and passed through a nonlinear activation function (GELU):

$$
V_{\text{out}} = \sigma\left(V_{\text{spec}} + V_{\text{lin}} + b\right)
$$

4. **Inverse Physical Mapping**: Using the inverse transformation, the predicted latent fields are interpolated back onto the original $N$ nodes of the irregular physical mesh, producing the continuous fields $U_x$, $U_y$, and $P$, along with the integrated aerodynamic lift coefficient $C_L$ and drag coefficient $C_D$.

## 2. Learning Algorithm and Dynamic Gradient Balancing

The optimization schedule transitions from a data-driven regime to a physics-informed regime to support stable convergence.

**Phase 1: Data-Driven Pre-training.** For $\text{epoch} \le E_{\text{trans}}$ (typically the first 100 epochs), the network minimizes the empirical regression loss $\mathcal{L}_{\text{data}}(\theta)$ against ground-truth CFD data. This anchors the diffeomorphic coordinate transformation.

**Phase 2: Physics-Informed Optimization.** For $\text{epoch} > E_{\text{trans}}$, the automatic differentiation engine `torch.autograd` is activated to compute spatial Jacobians $\nabla \hat{u}$, $\nabla \hat{p}$, and $\nabla \hat{\nu}_{\text{eff}}$ directly within the physical domain $\Omega$. These derivatives formulate the RANS momentum residuals $\mathcal{L}_{\text{pde}}(\theta)$ and the wall-boundary no-slip conditions $\mathcal{L}_{\text{bc}}(\theta)$.

### Dynamic Weight Balancing

To prevent the gradients of the differential equations from diminishing or dominating the empirical data gradients, the loss weights $\lambda_{\text{pde}}$ and $\lambda_{\text{bc}}$ are dynamically rescaled at each optimization step based on their relative gradient norms:

$$
\lambda_{\text{pde}}, \lambda_{\text{bc}} \propto
\frac{\|\nabla_\theta \mathcal{L}_{\text{data}}\|}{\|\nabla_\theta \mathcal{L}_{\text{pde}}\|}
$$

This self-stabilizing mechanism ensures that the physical laws act as a regularizer without destabilizing the learned topology of the flow fields.

## 3. Computational Complexity Analysis

To justify the computational efficiency of the proposed geometry-adaptive operator against classical iterative numerical solvers, we evaluate its asymptotic spatiotemporal complexity profile.

Let $B$ denote the batch size, $L$ the number of sequential Fourier Neural Operator (FNO) layers, and $S_1$, $S_2$ the spatial discretization resolutions along the axes of the transformed latent uniform grid $\tilde{\Omega} = \mathbb{T}^2$. Let $k_1$ and $k_2$ denote the maximum truncated wavenumber modes retained in the spectral domain, while $C_{\text{in}}$ and $C_{\text{out}}$ denote the input and output channel dimensionalities per layer.

The forward inference pipeline is decomposed into sequential, non-nested algorithmic segments.

### a) Neural Coordinate Deformation $(\Phi^{-1})$

The coordinate deformation network maps non-conforming spatial topologies pointwise, scaling as:

$$
\mathcal{O}\left(B \cdot S_1 \cdot S_2 \cdot C_{\text{in}}\right)
$$

### b) Discrete Forward and Inverse Fast Fourier Transforms $(\mathcal{F}, \mathcal{F}^{-1})$

The projection of the input tensor fields into the frequency domain via 2D FFT and inverse FFT scales as:

$$
\mathcal{T}_{\text{FFT}} = \mathcal{O}\left(B \cdot L \cdot C_{\text{in}} \cdot S_1 \cdot S_2 \log(S_1 S_2)\right)
$$

### c) Spectral Linear Parameterization $(R_\phi)$

Within the frequency domain, the complex weight multiplication scales with the retained low-frequency modes $k_1 \times k_2$:

$$
\mathcal{T}_{\text{weights}} = \mathcal{O}\left(B \cdot L \cdot k_1 \cdot k_2 \cdot C_{\text{in}} \cdot C_{\text{out}}\right)
$$

### Asymptotic Total Complexity

Because these stages are executed sequentially rather than within nested spatial loops, their asymptotic costs are additive. The global total computational complexity is:

$$
\mathcal{T}_{\text{total}} =
\mathcal{O}\Big(
B \cdot L \cdot
\big[
C_{\text{in}} \cdot S_1 \cdot S_2 \log(S_1 S_2)
+
k_1 \cdot k_2 \cdot C_{\text{in}} \cdot C_{\text{out}}
\big]
\Big)
$$

This additive formulation shows the advantage of continuous neural operators over classical Finite Volume Method (FVM) solvers. When refining the grid for high-fidelity boundary-layer tracking, the spectral weight multiplication remains independent of the global grid density. The computational footprint grows mainly through the FFT term, while classical iterative solvers often require superlinear sparse matrix operations to reduce residuals below convergence thresholds.

## 4. Physical Interpretation of Validation Results

### Integral Aerodynamic Forces

Integrating the predicted pressure and shear-stress fields over the airfoil boundary $\partial \Omega$ yields macroscopic force coefficients.

- **Lift Coefficient ($C_L$):** The model achieves $R^2 = 0.8973$ and a mean absolute error of $\text{MAE} = 0.2173$. It captures the linear regime of lift across varying angles of attack.
- **Drag Coefficient ($C_D$):** The drag coefficient has a lower correlation profile ($R^2 = 0.4821$, $\text{MAE} = 0.0974$). The model captures the overall trend at low angles of attack but underpredicts drag under near-stall conditions where separated turbulent wakes dominate.

### Analysis of the $L_2$ Error in Turbulent Viscosity $(\nu_t)$

Volumetric evaluation of the effective turbulent kinematic viscosity $(\nu_t)$ yields a relatively high error metric ($L_2 = 25.05\%$).

**Spatial Shift Interpretation.** This error does not necessarily imply a breakdown of the flow physics. It can arise when the predicted boundary-layer profile is shifted slightly in space relative to the reference profile. If the true profile is $f(x)$ and the prediction is $f(x + \Delta x)$, then the continuous error norm is

$$
\epsilon_{L_2} = \frac{\|f(x + \Delta x) - f(x)\|_2}{\|f(x)\|_2}
$$

For functions with very steep spatial gradients, even a small shift $\Delta x$ can produce a large pointwise difference and inflate the relative $L_2$ error. In that case, the error reflects a minor spatial misalignment rather than a failure to capture the qualitative boundary-layer structure.
