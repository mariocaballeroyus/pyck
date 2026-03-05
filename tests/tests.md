# pyCk tests

## Basis

### Knot Vector

1. **Monotonicity Test**

   Verify that $\xi_i \le \xi_{i+1}$ for all $i$.

2. **Knot Span Test**

   Successfully find the index $i$ such that $\xi_i \le u < \xi_{i+1}$.

   Edge Case: Ensure $u = \xi_{max}$ returns the last valid span index $n$ (the $n$ such that $N_{n,p}$ is the last basis function).

3. **Clamped Property Test**

   Verify that the first $p+1$ and last $p+1$ knots are equal (e.g., $0.0$ and $1.0$ for a normalized vector).

4. **Unique Spans Test**

   Verify that the number of non-zero length elements (where $\xi_{i+1} - \xi_i > 0$) matches the expected number of physical "elements" in the mesh ($n - p$).

### BSpline

1. **Partition of Unity**

   For any $u \in [\xi_p, \xi_{n+1}]$, the sum of all basis functions must be exactly 1:

   $$\sum_{i=0}^n N_{i,p}(u) = 1.0$$

   Tested for degrees 1 through 3.

2. **Non-negativity**

   Verify $N_{i,p}(u) \ge 0$ for all $i, p, u$.

3. **Local Support**

   Verify $N_{i,p}(u) = 0$ if $u < \xi_i$ or $u > \xi_{i+p+1}$.

4. **Derivative Sum to Zero**

   The sum of derivatives of order $k \ge 1$ must be zero:

   $$\sum_{i=0}^n \frac{d^k}{du^k} N_{i,p}(u) = 0$$

   Tested for $k = 1, 2$.

5. **C-continuity**

   Verify that at a simple internal knot $\xi_i$ (multiplicity $k=1$), the basis is $C^{p-k}$ continuous. Tested via comparing left and right limits of derivatives up to order $p-1$ using close evaluation points.

6. **Analytical Values**

   Verify basis function values against closed-form Bernstein polynomial expressions for degrees 1 and 2.

7. **Analytical Derivatives**

   Verify derivatives of degrees 1 and 2 against their closed-form expressions (first and second derivatives). Also verify that orders above the degree are identically zero.

8. **Finite-Difference Derivative Check**

   Verify analytical first derivatives against central finite differences for general knot vectors.

### Tensor Product

1. **Multivariate Partition of Unity**

   For a 2D basis $R_{i,j}(u, v) = N_{i,p}(u)M_{j,q}(v)$:

   $$\sum_{i,j} R_{i,j}(u, v) = 1.0$$

2. **Lexicographical Ordering**

   Verify that `eval_on_span` returns matrices in the expected order: index $= i_u \cdot (k_v + 1) + i_v$, i.e. $v$-derivatives vary fastest. Verified against manually computed Kronecker products for all four entries of order $(1,1)$.

3. **Linear Consistency (Bilinear Patch)**

   If 1D bases are linear, the tensor product must be able to represent a bilinear function $f(u,v) = 3u + 2v + 5uv + 1$ exactly.

4. **Derivative Kronecker Check**

   Verify mixed partial derivatives factor as $\partial^{a+b}R / (\partial u^a \partial v^b) = (d^a N_u / du^a) \otimes (d^b N_v / dv^b)$.

5. **Finite-Difference Derivative Check**

   Verify $\partial R / \partial u$ and $\partial R / \partial v$ against central finite differences.

## Geometry

### Surface

1. **Corner Interpolation**

   For a clamped surface, verify the mapping of parameter corners to control points:

   $$\mathbf{S}(u_{min}, v_{min}) = \mathbf{P}_{0,0}$$

   $$\mathbf{S}(u_{max}, v_{max}) = \mathbf{P}_{n,m}$$

2. **Rigid Body Transformation (Translation Invariance)**

   Translate all control points by $\vec{T}$. Verify $\mathbf{S}(u,v)_{new} = \mathbf{S}(u,v)_{old} + \vec{T}$.

3. **Affine Covariance (Rotation)**

   Rotating the control points by a rotation matrix $R$ must produce $\mathbf{S}_{new}(u,v) = R \cdot \mathbf{S}_{old}(u,v)$. Tested with a 90° rotation about the z-axis.

4. **Isoparametric Curve Test**

   Fix $v = v^*$. Verify $\mathbf{S}(u, v^*)$ equals a 1D B-spline curve with effective control points $Q_i = \sum_j M_{j,q}(v^*) \cdot P_{i,j}$.

5. **Normal Vector Perpendicularity**

   Verify that $\mathbf{n} = \frac{\partial \mathbf{S}}{\partial u} \times \frac{\partial \mathbf{S}}{\partial v}$ is perpendicular to both tangents ($\mathbf{n} \cdot \partial\mathbf{S}/\partial u \approx 0$). Tested on the parabolic surface.

6. **Parabolic Surface: Analytical Derivatives**

   For $\mathbf{S}(u,v) = (u, v, u^2 + v^2)$, verify position and all derivatives up to second order against closed-form expressions.

7. **Jacobian (Parabolic Surface)**

   Verify the Jacobian matrix $J = [\partial\mathbf{S}/\partial u \mid \partial\mathbf{S}/\partial v]$ and its determinant $|\!J\!| = \sqrt{1 + 4u^2 + 4v^2}$ analytically.

8. **Finite-Difference Derivative Check**

   Verify $\partial\mathbf{S}/\partial u$ and $\partial\mathbf{S}/\partial v$ against central finite differences for a general surface with non-trivial control points.