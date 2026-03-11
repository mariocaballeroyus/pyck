# Single-Variable Timoshenko Beam: Isogeometric Formulation and Constraints

This document details the theoretical foundation and exact matrix structures required to implement a 1-parameter, rotation-free Timoshenko beam using Isogeometric Analysis (IGA). 

## 1. Variational Formulation

[cite_start]The rotation-free Timoshenko beam theory reduces the problem to a single unknown variable: the bending displacement $w_b$[cite: 137]. [cite_start]All other physical quantities are derived from this variable and its derivatives[cite: 137]. 

The total physical displacement $w$ and the cross-sectional rotation $\varphi$ are defined as:
[cite_start]$$w = w_b - \frac{K_b}{K_s} w_b''$$ [cite: 138]
[cite_start]$$\varphi = -w_b'$$ [cite: 139]

The continuous variational form (virtual work) for the beam is:
$$\int_{\Omega}{ \delta w_b'' K_b w_b'' \, d\Omega } + \frac{K_b^2}{K_s^2} \int_{\Omega}{ \delta w_b''' K_s w_b''' \, d\Omega } = \int_{\Omega}{ \delta \left(w_b - \frac{K_b}{K_s} w_b'' \right) p \, d\Omega }$$

## 2. Discretization Strategy

### The Strain-Displacement Matrix ($\mathbf{B}$)
To avoid unphysical cross-terms (e.g., $- 2 \frac{K_b^2}{K_s} N'' N'''$) that arise from squaring a single-row matrix, the formulation utilizes a **two-row generalized strain matrix**. This separates the bending and shear energy contributions into a decoupled block structure.

$$\mathbf{B} = \begin{bmatrix} \mathbf{B}_{bend} \\ \mathbf{B}_{shear} \end{bmatrix} = \begin{bmatrix} N_1'' & N_2'' & \dots & N_n'' \\ -\frac{K_b}{K_s} N_1''' & -\frac{K_b}{K_s} N_2''' & \dots & -\frac{K_b}{K_s} N_n''' \end{bmatrix}$$

### The Constitutive Matrix ($\mathbf{D}$)
The energy terms are linked via a diagonal constitutive matrix. This ensures the stiffness matrix assembly $\mathbf{K} = \int \mathbf{B}^T \mathbf{D} \mathbf{B} \, d\Omega$ preserves the independent energy states:

$$\mathbf{D} = \begin{bmatrix} K_b & 0 \\ 0 & K_s \end{bmatrix}$$

**Stiffness Result:**
$$\mathbf{B}^T \mathbf{D} \mathbf{B} = K_b (N'')^2 + K_s \left(-\frac{K_b}{K_s} N'''\right)^2 = K_b (N'')^2 + \frac{K_b^2}{K_s} (N''')^2$$
This precisely recovers the terms in the original variational energy functional.

### The Shape Matrix ($\tilde{\mathbf{N}}$) for Load Vector
The external work must be performed on the **total physical displacement** $w$. Therefore, the modified shape function matrix $\tilde{\mathbf{N}}$ used for calculating the load vector $\mathbf{F} = \int \tilde{\mathbf{N}}^T p \, d\Omega$ incorporates the shear correction:

$$\tilde{\mathbf{N}} = \begin{bmatrix} N_1 - \frac{K_b}{K_s}N_1'' & \dots & N_n - \frac{K_b}{K_s}N_n'' \end{bmatrix}$$

## 3. Boundary Condition Imposition (Algebraic Constraints)

[cite_start]Because the bending displacement $w_b$ is discretized instead of the total displacement $w$ and rotation $\varphi$, boundary conditions cannot be imposed by directly assigning values to boundary degrees of freedom[cite: 335]. [cite_start]Instead, boundary conditions lead to linear constraints involving the degrees of freedom near the boundary[cite: 336]. 

Assuming an open knot vector—where only specific basis functions and their derivatives are non-zero at the boundary (e.g., $x=0$)—these physical boundary conditions yield the following algebraic constraints that must be enforced during matrix assembly:

* **Zero Displacement ($w(0)=0$):**
  [cite_start]This yields a linear constraint among the first three degrees of freedom[cite: 348]:
  [cite_start]$$\left(N_1''(0) - \frac{K_s}{K_b}\right)\hat{w}_{b,1} + N_2''(0)\hat{w}_{b,2} + N_3''(0)\hat{w}_{b,3} = 0$$ [cite: 349]

* **Zero Rotation ($\varphi(0)=0$):**
  [cite_start]This simply yields a constraint between the first two degrees of freedom[cite: 352]:
  [cite_start]$$\hat{w}_{b,1} = \hat{w}_{b,2}$$ [cite: 353]

* **Clamped Support ($w(0)=\varphi(0)=0$):**
  [cite_start]In the case of a clamped support, the rotation constraint can be substituted into the displacement constraint[cite: 354], which then reduces to:
  [cite_start]$$\left(N_1''(0) + N_2''(0) - \frac{K_s}{K_b}\right)\hat{w}_{b,1} + N_3''(0)\hat{w}_{b,3} = 0$$ [cite: 355]

These linear constraints can be enforced algebraically using penalty methods or master-slave elimination on the global stiffness matrix $\mathbf{K}$ and load vector $\mathbf{F}$.

## 4. Implementation Requirements
* **Continuity:** Since the $\mathbf{B}$ matrix utilizes $N'''$, the chosen basis functions must be at least **$C^2$ continuous** (e.g., Cubic B-Splines or higher degree polynomials).
* **Integration:** Standard Gauss Quadrature is applied. The order of integration must be sufficient to exactly capture the $N''' \cdot N'''$ product.