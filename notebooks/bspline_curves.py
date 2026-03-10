# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: venv (3.12.3)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # **B-Spline Curve Visualization**
#
# This notebook demonstrates how to create and visualize B-spline curves using
# the curve plotting utilities in `pyck.io`. Curve evaluation uses **De Boor's
# algorithm** for efficient, numerically stable geometric point evaluation.
#
# ### **Index**
# 1. [Line Segment](#Line-Segment)
# 2. [Planar Curve](#Planar-Curve)
# 3. [Effect of Degree on Approximation](#Degree-Effect)
# 4. [Knot Locations on Curves](#Knot-Locations)
# 5. [Local Support of B-Splines](#Local-Support)
# 6. [Convex Hull Property](#Convex-Hull)
# 7. [3D Helix Curve](#3D-Helix)
# 8. [Control Point Manipulation](#CP-Manipulation)

# %%
import pyck as ck
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## ***<a name="Line-Segment"></a>Line Segment***
#
# The simplest curve: a straight line along the $x$-axis created with
# `create_line_segment`. Control points are placed at Greville abscissae,
# so the linear mapping is exact regardless of polynomial degree.

# %%
kv = ck.create_clamped_uniform(3, 8)
basis = ck.create_bspline(3, kv)
line = ck.create_line_segment(basis, 10.0, name="line segment")

fig, ax = plt.subplots(figsize=(10, 3))
ck.io.curve.plot_curve(line, ax=ax)
ax.set_title("Straight Line Segment (p = 3)")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
plt.tight_layout()

# %% [markdown]
# ## ***<a name="Planar-Curve"></a>Planar Curve***
#
# A quarter-circle approximation built by placing control points on a
# circular arc. The B-spline curve smoothly interpolates through the
# control polygon.

# %%
n = basis.num_basis
cpts = np.zeros((n, 3))
t = np.linspace(0, np.pi / 2, n)
cpts[:, 0] = np.cos(t)
cpts[:, 1] = np.sin(t)

quarter = ck.create_curve_patch(basis, cpts, name="quarter circle")

fig, ax = plt.subplots(figsize=(6, 6))
ck.io.curve.plot_curve(quarter, ax=ax)
ax.set_title("Quarter-Circle Approximation (p = 3)")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
plt.tight_layout()

# %% [markdown]
# ## ***<a name="Degree-Effect"></a>Effect of Degree on Approximation***
#
# Increasing the polynomial degree $p$ generally produces a smoother curve
# that follows the control polygon less tightly. For degree $p=1$ the
# curve exactly traces the control polygon. As $p$ grows the curve
# increasingly "cuts corners", staying inside the convex hull of the
# control points.

# %%
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

for i, p in enumerate([1, 2, 3, 5]):
    n_basis = max(p + 1, 8)
    kv_p = ck.create_clamped_uniform(p, n_basis)
    bs_p = ck.create_bspline(p, kv_p)

    n_p = bs_p.num_basis
    cp = np.zeros((n_p, 3))
    angles = np.linspace(0, np.pi / 2, n_p)
    cp[:, 0] = np.cos(angles)
    cp[:, 1] = np.sin(angles)

    curve_p = ck.create_curve_patch(bs_p, cp, name=f"p = {p}")
    ck.io.curve.plot_curve(curve_p, ax=axes[i])
    axes[i].set_title(f"Degree p = {p}, n = {n_p}")
    axes[i].set_xlabel("$x$")
    axes[i].set_ylabel("$y$")

plt.suptitle("Effect of Polynomial Degree", fontsize=14, y=1.02)
plt.tight_layout()

# %% [markdown]
# ## ***<a name="Knot-Locations"></a>Knot Locations on Curves***
#
# The `show_knots` option marks where each unique knot value maps to on the
# physical curve via `plot_knot_marks`. This is useful for understanding
# element boundaries in an isogeometric context.

# %%
kv_fine = ck.create_clamped_uniform(3, 12)
bs_fine = ck.create_bspline(3, kv_fine)

n_f = bs_fine.num_basis
cp_f = np.zeros((n_f, 3))
angles_f = np.linspace(0, np.pi, n_f)
cp_f[:, 0] = np.cos(angles_f)
cp_f[:, 1] = np.sin(angles_f)

arc = ck.create_curve_patch(bs_fine, cp_f, name="semicircle")

fig, ax = plt.subplots(figsize=(8, 5))
ck.io.curve.plot_curve(arc, ax=ax, show_knots=True)
ax.set_title("Semicircular Arc with Knot Locations")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
plt.tight_layout()

# %% [markdown]
# ## ***<a name="Local-Support"></a>Local Support of B-Splines***
#
# A key property of B-spline bases is **local support**: each basis
# function $N_{i,p}$ is non-zero only on a limited interval of the knot
# vector. Consequently, moving a single control point only affects the
# curve geometry locally.
#
# Below, we start from a semicircular arc and perturb a single interior
# control point. The resulting deformation is confined to the support of
# the basis functions associated with that control point.

# %%
kv_ls = ck.create_clamped_uniform(3, 12)
bs_ls = ck.create_bspline(3, kv_ls)

n_ls = bs_ls.num_basis
cp_ls = np.zeros((n_ls, 3))
angles_ls = np.linspace(0, np.pi, n_ls)
cp_ls[:, 0] = np.cos(angles_ls)
cp_ls[:, 1] = np.sin(angles_ls)

original = ck.create_curve_patch(bs_ls, cp_ls, name="original")

# Perturb a single control point
perturbed_idx = n_ls // 2
cp_perturbed = cp_ls.copy()
cp_perturbed[perturbed_idx, 1] += 0.5  # push one CP outward

perturbed = ck.create_curve_patch(bs_ls, cp_perturbed, name="perturbed")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: show original plus the perturbed CP
ck.io.curve.plot_curve(original, ax=axes[0])
axes[0].scatter(
    [cp_perturbed[perturbed_idx, 0]],
    [cp_perturbed[perturbed_idx, 1]],
    color="orange", s=100, zorder=10, marker="^", label=f"Moved CP {perturbed_idx}",
)
axes[0].set_title("Original Curve + Target CP")
axes[0].set_xlabel("$x$")
axes[0].set_ylabel("$y$")
axes[0].legend()

# Right: overlay both curves — deformation is local
coords_orig = original.evaluate(300)
coords_pert = perturbed.evaluate(300)

axes[1].plot(coords_orig[:, 0], coords_orig[:, 1], "--", color="gray", linewidth=1, label="original")
ck.io.curve.plot_curve(perturbed, ax=axes[1])
ck.io.curve.plot_knot_marks(perturbed, ax=axes[1])
axes[1].set_title("Local Effect of Moving a Single Control Point")
axes[1].set_xlabel("$x$")
axes[1].set_ylabel("$y$")

plt.tight_layout()

# %% [markdown]
# To confirm that only a local region is affected, we compare the original
# and perturbed curves. The difference is non-zero only on the support of
# $N_{i,p}$ centred on the moved control point $i$.

# %%
u_dense = np.linspace(0.0, 1.0, 500)
diff = np.linalg.norm(original.eval_geometry(u_dense) - perturbed.eval_geometry(u_dense), axis=1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(u_dense, diff, color="#2176ae", linewidth=2)
ck.io.basis.plot_knots(bs_ls, ymin=0.0, ymax=diff.max() * 1.1, ax=ax)
ax.axvline(bs_ls.knots[perturbed_idx], color="orange", linestyle="-",
           linewidth=1.5, label=f"Moved CP span (i={perturbed_idx})")
ax.legend()
ax.set_title("Deviation between Original and Perturbed Curve")
ax.set_xlabel("$u$")
ax.set_ylabel("$||C(u) - C'(u)||$")
ax.grid(True, alpha=0.2)
plt.tight_layout()

# %% [markdown]
# ## ***<a name="Convex-Hull"></a>Convex Hull Property***
#
# A B-spline curve lies within the convex hull of its control points.
# Over each knot span $[u_i, u_{i+1})$, the curve is contained in the
# convex hull of the $p+1$ active control points.  This is a consequence
# of the non-negativity and partition-of-unity properties of B-spline
# basis functions:
#
# $$\sum_{i} N_{i,p}(u) = 1 \quad \text{and} \quad N_{i,p}(u) \ge 0$$
#
# We illustrate this by evaluating a curve and overlaying the convex hull
# of its control polygon.

# %%
kv_ch = ck.create_clamped_uniform(3, 8)
bs_ch = ck.create_bspline(3, kv_ch)

# S-shaped control polygon
n_ch = bs_ch.num_basis
cp_ch = np.zeros((n_ch, 3))
cp_ch[:, 0] = np.linspace(0, 4, n_ch)
cp_ch[:, 1] = np.array([0, 1.5, 2, 0.5, -0.5, -2, -1.5, 0])[:n_ch]

s_curve = ck.create_curve_patch(bs_ch, cp_ch, name="S-curve")

fig, ax = plt.subplots(figsize=(8, 5))

# Simple 2D convex hull via Graham scan (avoid scipy dependency)
pts_2d = cp_ch[:, :2]
centroid = pts_2d.mean(axis=0)
angles = np.arctan2(pts_2d[:, 1] - centroid[1], pts_2d[:, 0] - centroid[0])
hull_order = np.argsort(angles)
hull_pts = pts_2d[np.append(hull_order, hull_order[0])]

ax.fill(hull_pts[:, 0], hull_pts[:, 1], alpha=0.1, color="#57a773", label="Convex hull")
ax.plot(hull_pts[:, 0], hull_pts[:, 1], "-", color="#57a773", alpha=0.4, linewidth=1)

ck.io.curve.plot_curve(s_curve, ax=ax, show_knots=True)
ax.set_title("Convex Hull Property")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
plt.tight_layout()

# %% [markdown]
# ## ***<a name="3D-Helix"></a>3D Helix Curve***
#
# B-spline curves live in 3D space. Here we construct a helical curve
# and visualize it with `plot_curve_3d`.

# %%
kv_helix = ck.create_clamped_uniform(3, 20)
bs_helix = ck.create_bspline(3, kv_helix)

n_h = bs_helix.num_basis
cp_h = np.zeros((n_h, 3))
theta = np.linspace(0, 4 * np.pi, n_h)
cp_h[:, 0] = np.cos(theta)
cp_h[:, 1] = np.sin(theta)
cp_h[:, 2] = np.linspace(0, 3, n_h)

helix = ck.create_curve_patch(bs_helix, cp_h, name="helix")

ax3d = ck.io.curve.plot_curve_3d(helix)
ax3d.set_title("3D Helix (p = 3, n = 20)")
ax3d.set_xlabel("$x$")
ax3d.set_ylabel("$y$")
ax3d.set_zlabel("$z$")
plt.tight_layout()

# %% [markdown]
# ## ***<a name="CP-Manipulation"></a>Control Point Manipulation***
#
# Moving control points directly changes the geometry. Here we start
# from a line segment and perturb interior control points to create a
# wave-like shape, demonstrating the local support of B-splines.

# %%
kv_wave = ck.create_clamped_uniform(3, 12)
bs_wave = ck.create_bspline(3, kv_wave)
wave = ck.create_line_segment(bs_wave, 10.0, name="wave")

# Perturb interior y-coordinates
cpts_wave = wave.control_points.copy()
n_w = cpts_wave.shape[0]
for i in range(1, n_w - 1):
    cpts_wave[i, 1] = 0.5 * np.sin(2 * np.pi * i / (n_w - 1))

wave_modified = ck.create_curve_patch(bs_wave, cpts_wave, name="wave")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ck.io.curve.plot_curve(wave, ax=axes[0])
axes[0].set_title("Original Line Segment")
axes[0].set_xlabel("$x$")
axes[0].set_ylabel("$y$")

ck.io.curve.plot_curve(wave_modified, ax=axes[1])
axes[1].set_title("After Perturbing Interior Control Points")
axes[1].set_xlabel("$x$")
axes[1].set_ylabel("$y$")

plt.tight_layout()
