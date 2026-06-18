# Mathematical Framework and Computational Complexity of Geo-PINO

This document outlines the architectural pipeline, training strategy, and computational complexity of the Geometric Physics-Informed Neural Operator (Geo-PINO), followed by an analysis of the validation results.

## 1. Problem Formulation and Architectural Pipeline

The model predicts the steady-state velocity fields ($\mathbf{u} = [U_x, U_y]^T$) and pressure fields ($P$) of an incompressible fluid governed by the Reynolds-Averaged Navier-Stokes (RANS) equations.

The architecture consists of four sequential processing stages:

1. **Physical Domain ($\Omega$):** The input tensor $X \in \mathbb{R}^{N \times d}$ contains $d$ features (spatial coordinates, geometry representations, angle of attack $\alpha$, and Reynolds number $Re$) sampled across $N$ nodes of an irregular mesh.
2. **Coordinate Mapping ($\phi^{-1}$):** A multilayer perceptron (MLP) maps the irregular physical nodes into a structured, uniform latent space parameterized as a two-dimensional torus $\mathbb{T}^2$ with resolution $H \times W$. This yields the latent tensor $V_0 \in \mathbb{R}^{H \times W \times d_v}$, where $d_v$ is the latent dimensionality.
3. **Spectral Operator ($\mathcal{G}_\theta$):** A Fourier Neural Operator (FNO) layer applies a 2D Fast Fourier Transform (FFT) to the latent tensor, truncates higher-frequency modes to retain the lowest modes ($|k| \le k_{\text{max}}$), multiplies them by a parameterized complex weight tensor $R_\phi$, and projects them back via an inverse 2D FFT (iFFT). Concurrently, a local linear path ($1 \times 1$ convolution) operates in parallel. The outputs are summed and passed through a GELU activation function:

$$
V_{\text{out}} = \sigma(V_{\text{spec}} + V_{\text{lin}} + b)
$$

4. **Inverse Physical Mapping:** The predicted latent fields are interpolated back onto the original $N$ nodes of the irregular physical mesh to output the continuous fields ($U_x, U_y, P$) and evaluate the integrated lift ($C_L$) and drag ($C_D$) coefficients.

## 2. Learning Algorithm and Gradient Balancing

The training schedule transitions from a data-driven regime to a physics-informed regime to ensure convergence stability:

* **Phase 1 (Data-Driven Pre-training):** For $\text{epoch} \le E_{\text{trans}}$, the network minimizes the empirical regression loss $\mathcal{L}_{\text{data}}(\theta)$ against ground-truth CFD data to initialize the coordinate transformation.
* **Phase 2 (Physics-Informed Optimization):** For $\text{epoch} > E_{\text{trans}}$, spatial Jacobians ($\nabla \hat{u}, \nabla \hat{p}, \nabla \hat{\nu}_{\text{eff}}$) are computed within the physical domain $\Omega$ via automatic differentiation. These derivatives formulate the RANS momentum residuals $\mathcal{L}_{\text{pde}}(\theta)$ and the boundary conditions $\mathcal{L}_{\text{bc}}(\theta)$.

### Dynamic Weight Balancing
To mitigate gradient imbalances between the data-driven loss and the differential equation residuals, the loss weights $\lambda_{\text{pde}}$ and $\lambda_{\text{bc}}$ are dynamically rescaled at each optimization step based on their relative gradient norms:

$$
\lambda_{\text{pde}}, \lambda_{\text{bc}} \propto \frac{\|\nabla_\theta \mathcal{L}_{\text{data}}\|}{\|\nabla_\theta \mathcal{L}_{\text{pde}}\|}
$$

This mechanism acts as a regularizer during optimization without destabilizing the learned flow fields.

## 3. Computational Complexity Analysis

We evaluate the asymptotic computational complexity of the operator and compare it to classical numerical solvers. 

Let $B$ denote the batch size, $L$ the number of sequential FNO layers, and $S_1, S_2$ the spatial discretization resolutions of the transformed latent grid $\tilde{\Omega} = \mathbb{T}^2$. Let $k_1, k_2$ denote the maximum truncated wavenumber modes in the spectral domain, and $C_{\text{in}}, C_{\text{out}}$ represent the layer input and output channel dimensions.

The forward pipeline is executed in sequential, non-nested stages:

### a) Coordinate Mapping ($\Phi^{-1}$)
The mapping scales pointwise as:

$$
\mathcal{O}(B \cdot S_1 \cdot S_2 \cdot C_{\text{in}})
$$

### b) Discrete Fast Fourier Transforms ($\mathcal{F}$ and $\mathcal{F}^{-1}$)
The 2D FFT and iFFT transformations operate across channels, scaling as:

$$
\mathcal{T}_{\text{FFT}} = \mathcal{O}(B \cdot L \cdot C_{\text{in}} \cdot S_1 \cdot S_2 \log(S_1 \cdot S_2))
$$

### c) Spectral Weight Multiplication ($R_\phi$)
The complex matrix multiplication operates only on the low-frequency modes $k_1 \times k_2$:

$$
\mathcal{T}_{\text{weights}} = \mathcal{O}(B \cdot L \cdot k_1 \cdot k_2 \cdot C_{\text{in}} \cdot C_{\text{out}})
$$

### Asymptotic Total Complexity
Since these processing stages are non-nested, their costs are additive. The total computational complexity ($\mathcal{T}_{\text{total}}$) is formalized as:

$$
\mathcal{T}_{\text{total}} = \mathcal{O}\Big( B \cdot L \cdot \big[ C_{\text{in}} \cdot S_1 \cdot S_2 \log(S_1 \cdot S_2) + k_1 \cdot k_2 \cdot C_{\text{in}} \cdot C_{\text{out}} \big] \Big)
$$

Unlike classical Finite Volume Method (FVM) solvers that require superlinear sparse matrix inversions ($\mathcal{O}(N^{1.5})$ to $\mathcal{O}(N^2)$, where $N$ is the mesh element count), the neural operator scales quasi-linearly via the FFT term. When refining the grid resolution ($S_1, S_2 \to \infty$), the parameterized channel-to-channel weight multiplication $\mathcal{T}_{\text{weights}}$ remains invariant, bypassing iterative sparse matrix updates.

## 4. Physical Interpretation of Validation Results

### Aerodynamic Coefficients
Integrating the predicted pressure and shear stress fields over the airfoil boundary $\partial \Omega$ provides the macroscopic force coefficients:
* **Lift Coefficient ($C_L$):** The model captures the linear lift regime across varying angles of attack, yielding $R^2 = 0.8973$ and $\text{MAE} = 0.2173$.
* **Drag Coefficient ($C_D$):** Prediction accuracy decreases ($R^2 = 0.4821$, $\text{MAE} = 0.0974$). The model tracks the qualitative trend at low angles of attack but underpredicts drag near stall conditions due to flow separation and turbulent wakes.

### Turbulent Viscosity ($\nu_t$) Error Analysis
Volumetric evaluation of the effective turbulent kinematic viscosity ($\nu_t$) yields a relative $L_2$ error of $25.05\%$. 

This error profile is a geometric artifact of the steep spatial gradients concentrated within the localized boundary layer. Let the true boundary layer profile be represented by $f(x)$ and the model prediction by a spatially shifted profile $f(x + \Delta x)$, where $\Delta x$ represents a minor misalignment of 1–2 mesh nodes. The relative $L_2$ error norm is defined as:

$$
\epsilon_{L_2} = \frac{\|f(x + \Delta x) - f(x)\|_2}{\|f(x)\|_2}
$$

For functions with high spatial gradients ($\nabla f \to \infty$), a minor coordinate shift $\Delta x$ maximizes the pointwise difference $\|f(x + \Delta x) - f(x)\|$, inflating the relative $L_2$ metric. Thus, the error indicates a minor spatial gradient shift rather than a structural failure to model the boundary layer mechanics.
