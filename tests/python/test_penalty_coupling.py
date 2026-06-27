"""Patch tests for multipatch penalty coupling (``PenaltyCouplingCondition``).

A rectangular strip is split into two patches along its mid-line. The far-left
edge is clamped and the far-right edge is pulled in axial tension, so the only
load path to the clamp runs *through* the coupled interface. With displacement
coupling the assembly stretches as one strip (reproducing the closed-form
uniaxial-tension field) and the displacement is continuous across the seam — even
when the two sides are non-conforming (different element counts / degree along
the seam), because the seam is straight and the A↔B map is affine.

The load is membrane (axial tension): displacement-only coupling is C0 across the
interface, so a membrane case reproduces the monolithic field exactly while a
cross-seam bending case would not (that needs rotation coupling, a follow-on).
"""

import numpy as np
import pytest

import pyck as ck

Lx, Ly, E, nu, h, fx = 4.0, 1.0, 1.0e7, 0.3, 0.1, 200.0
ND = 3


def _displacement(u, el, patch, params):
    return np.asarray(
        ck.Function(u, el, patch, ck.FieldType.DISPLACEMENT)(params)
    ).reshape(-1, 3)


def _solve_coupled_strip(left, right):
    """Clamp the far-left edge, couple the seam, pull the far-right edge in axial
    tension; solve and return per-side displacement samples + the closed-form value.

    Returns ``(clamped, loaded, left_iface, right_iface, exact)``, each displacement
    array sampled at the same parametric v-stations (which map to identical physical
    y on both rectangles regardless of their knot counts)."""
    el = ck.ShellKirchhoffLove3p(ck.PlaneStress2d(E, nu, h))
    prob = ck.LinearElasticProblem([left, right], el, ck.GaussLegendre.from_patch(left))
    nL, nR = left.num_control_pts, right.num_control_pts

    # Suppress transverse bending DOFs on both patches (flat membrane problem).
    # Global DOFs: left occupies block [0, nL*ND), right [nL*ND, (nL+nR)*ND).
    transverse = (
        [cp * ND + 2 for cp in range(nL)]
        + [nL * ND + cp * ND + 2 for cp in range(nR)]
    )
    prob.add_constraint(ck.DirectConstraint(transverse, value=0.0))

    # Clamp the far-left edge in-plane (pins translation + rotation -> well-posed).
    clamp = ck.PenaltyBoundaryCondition(left.boundary(0, True), ck.GaussLegendre(4, dim=1))
    clamp.add(ck.Field.U_X, 1.0e12).add(ck.Field.U_Y, 1.0e12)
    prob.add_condition(clamp, patch="left")

    # Couple the shared interface (left u=1  <->  right u=0).
    coupling = ck.PenaltyCouplingCondition(
        left.boundary(0, False), right.boundary(0, True), ck.GaussLegendre(5, dim=1)
    )
    coupling.couple_displacement(1.0e12)
    prob.add_condition(coupling, patch="left")

    # Axial tension on the far-right edge.
    load = ck.LoadBoundaryCondition(right.boundary(0, False), ck.GaussLegendre(4, dim=1))
    load.add(ck.Field.U_X, fx)
    prob.add_condition(load, patch="right")

    K, f = prob.assemble()
    u = ck.solve(K, f, physical_dofs=prob.num_physical_dofs)
    u_left, u_right = u[: nL * ND], u[nL * ND:]

    vs = np.linspace(0.2, 0.8, 5)
    exact = fx * Lx / (E * h)  # closed-form u_x at the far-right edge (x = Lx)
    clamped = _displacement(u_left, el, left, np.array([[0.0, v] for v in vs]))
    loaded = _displacement(u_right, el, right, np.array([[1.0, v] for v in vs]))
    left_iface = _displacement(u_left, el, left, np.array([[1.0, v] for v in vs]))
    right_iface = _displacement(u_right, el, right, np.array([[0.0, v] for v in vs]))
    return clamped, loaded, left_iface, right_iface, exact


def _assert_reproduces_monolith(clamped, loaded, left_iface, right_iface, exact):
    # Load transferred through the seam: clamp held, far edge at the closed-form stretch.
    assert np.abs(clamped).max() < 1e-6 * exact
    assert abs(loaded[:, 0].mean() - exact) / exact < 1e-2
    # Displacement continuous across the interface (penalty drives the jump to ~0).
    assert np.abs(left_iface - right_iface).max() < 1e-4 * exact
    # And the seam sits at the midpoint stretch (x = Lx/2).
    assert abs(left_iface[:, 0].mean() - 0.5 * exact) / exact < 1e-2


def _half(nu, nv, deg, cx, name):
    return ck.SurfacePatch.rectangle(Lx / 2, Ly, nu=nu, nv=nv, deg=deg, cx=cx, name=name)


def test_conforming_strip():
    """Matching discretisation on the seam (the degenerate / conforming case)."""
    left = _half(nu=6, nv=6, deg=3, cx=Lx / 4, name="left")
    right = _half(nu=6, nv=6, deg=3, cx=3 * Lx / 4, name="right")
    _assert_reproduces_monolith(*_solve_coupled_strip(left, right))


def test_nonconforming_different_nv():
    """Different element counts ALONG the seam (left nv=6, right nv=9): the common
    refinement integrates the coupling exactly and still reproduces the strip."""
    left = _half(nu=6, nv=6, deg=3, cx=Lx / 4, name="left")
    right = _half(nu=8, nv=9, deg=3, cx=3 * Lx / 4, name="right")
    _assert_reproduces_monolith(*_solve_coupled_strip(left, right))


def test_nonconforming_different_degree():
    """Different degree along the seam (left deg=3, right deg=2)."""
    left = _half(nu=6, nv=6, deg=3, cx=Lx / 4, name="left")
    right = _half(nu=6, nv=6, deg=2, cx=3 * Lx / 4, name="right")
    _assert_reproduces_monolith(*_solve_coupled_strip(left, right))


def test_geometry_mismatch_raises():
    """Boundaries that are NOT the same physical curve are rejected at construction.

    The right patch is offset in y so its u=0 edge no longer coincides with the
    left patch's u=1 edge — the endpoint/coincidence check throws."""
    left = _half(nu=6, nv=6, deg=3, cx=Lx / 4, name="left")
    right = ck.SurfacePatch.rectangle(
        Lx / 2, Ly, nu=6, nv=6, deg=3, cx=3 * Lx / 4, cy=Ly, name="right"
    )
    coupling = ck.PenaltyCouplingCondition(
        left.boundary(0, False), right.boundary(0, True), ck.GaussLegendre(4, dim=1)
    )
    coupling.couple_displacement(1.0e12)

    el = ck.ShellKirchhoffLove3p(ck.PlaneStress2d(E, nu, h))
    prob = ck.LinearElasticProblem([left, right], el, ck.GaussLegendre.from_patch(left))
    with pytest.raises(ValueError):
        prob.add_condition(coupling, patch="left")  # bind() runs the geometry check


def test_add_rejects_unvalidated_fields():
    """Only U_X/Y/Z and the bending rotation ROT_N are supported; others rejected."""
    left = _half(nu=6, nv=6, deg=3, cx=Lx / 4, name="left")
    right = _half(nu=6, nv=6, deg=3, cx=3 * Lx / 4, name="right")
    coupling = ck.PenaltyCouplingCondition(
        left.boundary(0, False), right.boundary(0, True), ck.GaussLegendre(4, dim=1)
    )
    coupling.add(ck.Field.ROT_S, 1.0e12)  # twist; stored lazily, rejected at bind()
    el = ck.ShellKirchhoffLove3p(ck.PlaneStress2d(E, nu, h))
    prob = ck.LinearElasticProblem([left, right], el, ck.GaussLegendre.from_patch(left))
    with pytest.raises(ValueError):
        prob.add_condition(coupling, patch="left")


# === Bending (out-of-plane) coupling ==========================================
# A transverse-loaded cantilever plate bends; the seam must transfer the bending
# moment, which needs ROT_N (normal/bending rotation) continuity on top of
# displacement. Without it the partner patch is a mechanism (free to rotate about
# the seam); with couple_kinematics the two-patch plate reproduces the monolith.

Lb, Wb, tb, fz = 10.0, 2.0, 0.1, 1.0       # plate length, width, thickness, tip load
AD, AR = 1.0e12, 1.0e10                     # displacement / rotation penalties


def _clamp_root(prob, patch, name):
    """Fully clamp the patch's u=0 edge: U_X/Y/Z = 0 and the slope ROT_N = 0."""
    c = ck.PenaltyBoundaryCondition(patch.boundary(0, True), ck.GaussLegendre(4, dim=1))
    c.add(ck.Field.U_X, AD).add(ck.Field.U_Y, AD).add(ck.Field.U_Z, AD).add(ck.Field.ROT_N, AR)
    prob.add_condition(c, patch=name)


def _tip_w(u, el, patch):
    return float(_displacement(u, el, patch, [[1.0, 0.5]])[0, 2])


def _bending_monolith():
    el = ck.ShellKirchhoffLove3p(ck.PlaneStress2d(E, nu, tb))
    plate = ck.SurfacePatch.rectangle(Lb, Wb, nu=12, nv=4, deg=3, cx=Lb / 2, name="m")
    prob = ck.LinearElasticProblem([plate], el, ck.GaussLegendre.from_patch(plate))
    _clamp_root(prob, plate, "m")
    lc = ck.LoadBoundaryCondition(plate.boundary(0, False), ck.GaussLegendre(4, dim=1))
    lc.add(ck.Field.U_Z, fz)
    prob.add_condition(lc, patch="m")
    return _tip_w(ck.solve(prob), el, plate)


def _bending_two_patch(left_nv, right_nv):
    el = ck.ShellKirchhoffLove3p(ck.PlaneStress2d(E, nu, tb))
    left = ck.SurfacePatch.rectangle(Lb / 2, Wb, nu=7, nv=left_nv, deg=3, cx=Lb / 4, name="left")
    right = ck.SurfacePatch.rectangle(Lb / 2, Wb, nu=7, nv=right_nv, deg=3, cx=3 * Lb / 4, name="right")
    prob = ck.LinearElasticProblem([left, right], el, ck.GaussLegendre.from_patch(left))
    _clamp_root(prob, left, "left")
    seam = ck.PenaltyCouplingCondition(
        left.boundary(0, False), right.boundary(0, True), ck.GaussLegendre(5, dim=1))
    seam.couple_kinematics(AD, AR)
    prob.add_condition(seam, patch="left")
    lc = ck.LoadBoundaryCondition(right.boundary(0, False), ck.GaussLegendre(4, dim=1))
    lc.add(ck.Field.U_Z, fz)
    prob.add_condition(lc, patch="right")
    u = ck.solve(prob)
    return _tip_w(u[left.num_control_pts * ND:], el, right)


def test_bending_conforming_reproduces_monolith():
    """Conforming seam + ROT_N coupling: two-patch bending == monolithic plate."""
    w_mono = _bending_monolith()
    w_two = _bending_two_patch(left_nv=4, right_nv=4)
    assert abs(w_two - w_mono) / abs(w_mono) < 1e-2


def _seam_normal_curvature(pen_c2):
    """Non-conforming cantilever strip, transverse tip load. Returns the recovered normal
    curvature κ_nn = (n¹)² κ_11 at the seam on each side (via the consistent bending
    strain), with the C2 ``couple_curvature_continuity`` tie toggled by ``pen_c2``."""
    el = ck.ShellKirchhoffLove3p(ck.PlaneStress2d(E, nu, tb))
    left = ck.SurfacePatch.rectangle(Lb / 2, Wb, nu=5, nv=4, deg=3, cx=Lb / 4, name="left")
    right = ck.SurfacePatch.rectangle(Lb / 2, Wb, nu=13, nv=8, deg=3, cx=3 * Lb / 4, name="right")
    prob = ck.LinearElasticProblem([left, right], el, ck.GaussLegendre.from_patch(left))
    _clamp_root(prob, left, "left")
    seam = ck.PenaltyCouplingCondition(
        left.boundary(0, False), right.boundary(0, True), ck.GaussLegendre(7, dim=1))
    seam.couple_displacement(AD).couple_director_continuity(AR)
    if pen_c2:
        seam.couple_curvature_continuity(pen_c2)
    prob.add_condition(seam, patch="left")
    lc = ck.LoadBoundaryCondition(right.boundary(0, False), ck.GaussLegendre(5, dim=1))
    lc.add(ck.Field.U_Z, fz)
    prob.add_condition(lc, patch="right")
    u = ck.solve(prob)
    u_left, u_right = u[: left.num_control_pts * ND], u[left.num_control_pts * ND:]

    def kappa_nn(uvec, patch, edge):
        # n¹ = A¹·x̂ = 1/(Lb/2): each half maps u∈[0,1] to a span of Lb/2 along x.
        n1 = 1.0 / (Lb / 2)
        k11 = np.asarray(ck.Function(uvec, el, patch, ck.FieldType.STRAIN)(
            [[edge, 0.5]])).reshape(-1)[3]   # Voigt bending row κ_11
        return n1 * n1 * k11

    return kappa_nn(u_left, left, 1.0), kappa_nn(u_right, right, 0.0)


def test_curvature_continuity_reduces_jump():
    """couple_curvature_continuity ties the cross-seam normal curvature κ_nn = n^α n^β κ_αβ
    (C2 of the bending surface): on a non-conforming seam the κ_nn jump that C0+C1 leaves
    is driven to ~0, completing C2."""
    knn_l, knn_r = _seam_normal_curvature(pen_c2=AR)
    scale = max(abs(knn_l), abs(knn_r), 1e-30)
    assert abs(knn_l - knn_r) / scale < 1e-4


def test_curvature_jump_without_c2_tie():
    """Companion: with only C0+C1 the non-conforming seam carries a real κ_nn jump, so the
    C2 tie above is doing genuine work (it is not an already-continuous field)."""
    knn_l, knn_r = _seam_normal_curvature(pen_c2=0.0)
    scale = max(abs(knn_l), abs(knn_r))
    assert abs(knn_l - knn_r) / scale > 0.02


def test_cartesian_rotation_is_seam_continuous():
    """The covariant rotation components θ_α = θ·A_α are basis dependent, so on a
    non-conforming seam (the two halves parametrise the crossing direction at different
    speeds) the u-covariant component θ_1 jumps even though the physical rotation is
    continuous; θ_2 survives only because A_2 (the seam tangent) is parametrised
    identically. ``as_cartesian_vector`` maps θ_α → physical Cartesian (θ_x,θ_y,θ_z),
    which is continuous across the seam — the form to export for visualization."""
    el = ck.ShellKirchhoffLove3p(ck.PlaneStress2d(E, nu, tb))
    # Unequal split: the left half is a narrow u-strip, the right a wide one.
    left = ck.SurfacePatch.rectangle(Lb * 0.25, Wb, nu=6, nv=4, deg=3, cx=Lb * 0.125, name="left")
    right = ck.SurfacePatch.rectangle(Lb * 0.75, Wb, nu=10, nv=7, deg=3, cx=Lb * 0.625, name="right")
    prob = ck.LinearElasticProblem([left, right], el, ck.GaussLegendre.from_patch(left))
    _clamp_root(prob, left, "left")
    seam = ck.PenaltyCouplingCondition(
        left.boundary(0, False), right.boundary(0, True), ck.GaussLegendre(6, dim=1))
    seam.couple_kinematics(AD, AR)
    prob.add_condition(seam, patch="left")
    lc = ck.LoadBoundaryCondition(right.boundary(0, False), ck.GaussLegendre(5, dim=1))
    lc.add(ck.Field.U_Z, fz)
    prob.add_condition(lc, patch="right")
    u = ck.solve(prob)
    u_left, u_right = u[: left.num_control_pts * ND], u[left.num_control_pts * ND:]

    f_l = ck.Function(u_left, el, left, ck.FieldType.ROTATION)
    f_r = ck.Function(u_right, el, right, ck.FieldType.ROTATION)
    vs = np.linspace(0.2, 0.8, 5)
    p_l = np.array([[1.0, v] for v in vs])
    p_r = np.array([[0.0, v] for v in vs])

    cov_l, cov_r = np.asarray(f_l(p_l)), np.asarray(f_r(p_r))
    # Raw covariant: θ_1 (u-tangent) jumps; θ_2 (seam tangent) is continuous.
    assert np.abs(cov_l[:, 0] - cov_r[:, 0]).max() > 1e-3
    assert np.abs(cov_l[:, 1] - cov_r[:, 1]).max() < 1e-8

    car_l = ck.as_cartesian_vector(f_l, left)(p_l)
    car_r = ck.as_cartesian_vector(f_r, right)(p_r)
    assert car_l.shape == (len(vs), 3)
    scale = max(np.abs(np.r_[car_l, car_r]).max(), 1e-30)
    # Physical Cartesian rotation is continuous across the seam.
    assert np.abs(car_l - car_r).max() < 1e-6 * scale


def test_bending_nonconforming_reproduces_monolith():
    """Non-conforming seam (different nv) + ROT_N coupling still reproduces it."""
    w_mono = _bending_monolith()
    w_two = _bending_two_patch(left_nv=4, right_nv=7)
    assert abs(w_two - w_mono) / abs(w_mono) < 2e-2


def test_bending_rm4p_reproduces_monolith():
    """Element-agnostic: the same displacement+ROT_N tie reproduces the monolith for
    ShellReissnerMindlin4p (4 dofs/cp, psi pinned). RM-4p needs C2 basis *within* a
    patch, but inter-patch continuity is still C0 displacement + G1 rotation."""
    nd = 4  # RM-4p: slot 3 is the psi twist potential

    def pin_psi(prob, patches):
        idx, off = [], 0
        for p in patches:
            n = p.num_control_pts
            idx += [off + cp * nd + 3 for cp in range(n)]
            off += n * nd
        prob.add_constraint(ck.DirectConstraint(idx, value=0.0))

    def clamp_and_load(prob, root, tip, names):
        c = ck.PenaltyBoundaryCondition(root.boundary(0, True), ck.GaussLegendre(5, dim=1))
        c.add(ck.Field.U_X, AD).add(ck.Field.U_Y, AD).add(ck.Field.U_Z, AD).add(ck.Field.ROT_N, AR)
        prob.add_condition(c, patch=names[0])
        lc = ck.LoadBoundaryCondition(tip.boundary(0, False), ck.GaussLegendre(5, dim=1))
        lc.add(ck.Field.U_Z, fz)
        prob.add_condition(lc, patch=names[-1])

    el = ck.ShellReissnerMindlin4p(ck.PlaneStress2d(E, nu, tb))

    mono = ck.SurfacePatch.rectangle(Lb, Wb, nu=14, nv=4, deg=3, cx=Lb / 2, name="m")
    pm = ck.LinearElasticProblem([mono], el, ck.GaussLegendre.from_patch(mono))
    pin_psi(pm, [mono]); clamp_and_load(pm, mono, mono, ["m"])
    w_mono = float(_displacement(ck.solve(pm), el, mono, [[1.0, 0.5]])[0, 2])

    left = ck.SurfacePatch.rectangle(Lb / 2, Wb, nu=8, nv=4, deg=3, cx=Lb / 4, name="left")
    right = ck.SurfacePatch.rectangle(Lb / 2, Wb, nu=8, nv=7, deg=3, cx=3 * Lb / 4, name="right")
    p = ck.LinearElasticProblem([left, right], el, ck.GaussLegendre.from_patch(left))
    pin_psi(p, [left, right])
    seam = ck.PenaltyCouplingCondition(
        left.boundary(0, False), right.boundary(0, True), ck.GaussLegendre(6, dim=1))
    seam.couple_kinematics(AD, AR)
    p.add_condition(seam, patch="left")
    clamp_and_load(p, left, right, ["left", "right"])
    w_two = float(_displacement(ck.solve(p)[left.num_control_pts * nd:], el, right, [[1.0, 0.5]])[0, 2])

    assert abs(w_two - w_mono) / abs(w_mono) < 2e-2


def _hier4p_seam_psi(couple_psi):
    """Cantilever Hier4p plate split at mid-span, transverse tip load. Returns the
    recovered hierarchic shear potential ψ (PRIMAL slot 3) sampled at the seam from
    both sides. ``couple_psi`` toggles the Field.PSI tie on top of U + ROT_N."""
    nd = 4  # Hier4p: slot 3 is the psi shear potential
    el = ck.ShellReissnerMindlinHier4p(ck.PlaneStress2d(E, nu, tb))
    left = ck.SurfacePatch.rectangle(Lb / 2, Wb, nu=8, nv=4, deg=3, cx=Lb / 4, name="left")
    right = ck.SurfacePatch.rectangle(Lb / 2, Wb, nu=8, nv=7, deg=3, cx=3 * Lb / 4, name="right")
    p = ck.LinearElasticProblem([left, right], el, ck.GaussLegendre.from_patch(left))
    p.add_constraint(ck.DirectConstraint([3], value=0.0))  # pin one constant-psi mode

    seam = ck.PenaltyCouplingCondition(
        left.boundary(0, False), right.boundary(0, True), ck.GaussLegendre(6, dim=1))
    seam.couple_kinematics(AD, AR)
    if couple_psi:
        seam.add(ck.Field.PSI, AR)
    p.add_condition(seam, patch="left")

    c = ck.PenaltyBoundaryCondition(left.boundary(0, True), ck.GaussLegendre(5, dim=1))
    c.add(ck.Field.U_X, AD).add(ck.Field.U_Y, AD).add(ck.Field.U_Z, AD).add(ck.Field.ROT_N, AR)
    p.add_condition(c, patch="left")
    lc = ck.LoadBoundaryCondition(right.boundary(0, False), ck.GaussLegendre(5, dim=1))
    lc.add(ck.Field.U_Z, fz)
    p.add_condition(lc, patch="right")

    u = ck.solve(p)
    nL = left.num_control_pts
    u_left, u_right = u[: nL * nd], u[nL * nd:]

    def psi(uvec, patch, edge):
        params = np.array([[edge, v] for v in np.linspace(0.15, 0.85, 5)])
        primal = np.asarray(ck.Function(uvec, el, patch, ck.FieldType.PRIMAL)(params)).reshape(-1, nd)
        return primal[:, 3]

    return psi(u_left, left, 1.0), psi(u_right, right, 0.0)


def test_hier4p_psi_coupling_enforces_continuity():
    """Field.PSI ties the hierarchic shear potential of ShellReissnerMindlinHier4p
    across the seam: the recovered ψ matches on both sides (penalty drives the jump
    to ~0), where without the tie each patch carries its own ψ and they disagree."""
    psi_l, psi_r = _hier4p_seam_psi(couple_psi=True)
    scale = max(np.abs(np.r_[psi_l, psi_r]).max(), 1e-30)
    assert np.abs(psi_l - psi_r).max() < 1e-4 * scale


def test_hier4p_psi_discontinuous_without_coupling():
    """Companion: without the Field.PSI tie the seam ψ is discontinuous, so the tie
    above is doing real work (not merely confirming an already-continuous field)."""
    psi_l, psi_r = _hier4p_seam_psi(couple_psi=False)
    scale = np.abs(np.r_[psi_l, psi_r]).max()
    assert np.abs(psi_l - psi_r).max() > 0.1 * scale


def _hier4p_seam_psi_slope(c1):
    """Cantilever Hier4p plate split at mid-span, transverse tip load, with the seam
    coupled in displacement + director + ψ. Returns the value jump and the relative
    normal-slope (ψ_,n) jump of the recovered ψ at the seam. ``c1`` toggles the PSI_N
    (normal-slope) tie on top of the PSI (C0) tie."""
    nd = 4
    el = ck.ShellReissnerMindlinHier4p(ck.PlaneStress2d(E, nu, tb))
    left = ck.SurfacePatch.rectangle(Lb / 2, Wb, nu=8, nv=4, deg=3, cx=Lb / 4, name="left")
    right = ck.SurfacePatch.rectangle(Lb / 2, Wb, nu=8, nv=7, deg=3, cx=3 * Lb / 4, name="right")
    p = ck.LinearElasticProblem([left, right], el, ck.GaussLegendre.from_patch(left))
    p.add_constraint(ck.DirectConstraint([3], value=0.0))
    seam = ck.PenaltyCouplingCondition(
        left.boundary(0, False), right.boundary(0, True), ck.GaussLegendre(6, dim=1))
    seam.couple_displacement(AD).couple_director_continuity(AR)
    if c1:
        seam.couple_psi_continuity(AR, AR)
    else:
        seam.add(ck.Field.PSI, AR)
    p.add_condition(seam, patch="left")
    c = ck.PenaltyBoundaryCondition(left.boundary(0, True), ck.GaussLegendre(5, dim=1))
    c.add(ck.Field.U_X, AD).add(ck.Field.U_Y, AD).add(ck.Field.U_Z, AD).add(ck.Field.ROT_N, AR)
    p.add_condition(c, patch="left")
    lc = ck.LoadBoundaryCondition(right.boundary(0, False), ck.GaussLegendre(5, dim=1))
    lc.add(ck.Field.U_Z, fz)
    p.add_condition(lc, patch="right")
    u = ck.solve(p)
    nL = left.num_control_pts
    u_left, u_right = u[: nL * nd], u[nL * nd:]

    def psi(uvec, patch, params):
        return np.asarray(ck.Function(uvec, el, patch, ck.FieldType.PRIMAL)(params)).reshape(-1, nd)[:, 3]

    # Physical normal derivative ∂ψ/∂x via a one-sided difference into each half
    # (outward x at the left u=1 edge, into +x from the right u=0 edge); dx/du = Lb/2.
    dxdu, eps = Lb / 2, 1e-3
    vs = np.linspace(0.2, 0.8, 4)
    pL1 = psi(u_left, left, np.array([[1.0, v] for v in vs]))
    pLe = psi(u_left, left, np.array([[1 - eps, v] for v in vs]))
    pR0 = psi(u_right, right, np.array([[0.0, v] for v in vs]))
    pRe = psi(u_right, right, np.array([[eps, v] for v in vs]))
    slope_l = (pL1 - pLe) / (eps * dxdu)
    slope_r = (pRe - pR0) / (eps * dxdu)
    val_jump = np.abs(pL1 - pR0).max()
    slope_scale = max(np.abs(np.r_[slope_l, slope_r]).max(), 1e-30)
    return val_jump, np.abs(slope_l - slope_r).max() / slope_scale


def test_hier4p_psi_c1_continuity():
    """couple_psi_continuity adds the PSI_N (normal-slope) tie on top of PSI (C0), giving
    C1 of ψ: the recovered ψ is continuous AND its boundary-normal derivative matches
    across the seam (slope jump driven to a small fraction)."""
    val_jump, slope_rel = _hier4p_seam_psi_slope(c1=True)
    assert slope_rel < 0.1


def test_hier4p_psi_c0_only_leaves_slope_jump():
    """Companion: with PSI (C0) only, ψ is continuous but its normal slope is not — the
    derivative jumps by order unity, so PSI_N is what supplies the C1 part."""
    _, slope_rel = _hier4p_seam_psi_slope(c1=False)
    assert slope_rel > 0.5


def _bending_two_patch_director(left_nv, right_nv):
    """Like `_bending_two_patch` but ties the slope with the director-form G1 coupling
    (`couple_director_continuity`) instead of `ROT_N`."""
    el = ck.ShellKirchhoffLove3p(ck.PlaneStress2d(E, nu, tb))
    left = ck.SurfacePatch.rectangle(Lb / 2, Wb, nu=7, nv=left_nv, deg=3, cx=Lb / 4, name="left")
    right = ck.SurfacePatch.rectangle(Lb / 2, Wb, nu=7, nv=right_nv, deg=3, cx=3 * Lb / 4, name="right")
    prob = ck.LinearElasticProblem([left, right], el, ck.GaussLegendre.from_patch(left))
    _clamp_root(prob, left, "left")
    seam = ck.PenaltyCouplingCondition(
        left.boundary(0, False), right.boundary(0, True), ck.GaussLegendre(5, dim=1))
    seam.couple_displacement(AD).couple_director_continuity(AR)
    prob.add_condition(seam, patch="left")
    lc = ck.LoadBoundaryCondition(right.boundary(0, False), ck.GaussLegendre(4, dim=1))
    lc.add(ck.Field.U_Z, fz)
    prob.add_condition(lc, patch="right")
    return _tip_w(ck.solve(prob)[left.num_control_pts * ND:], el, right)


def test_director_continuity_reproduces_monolith():
    """couple_displacement + couple_director_continuity reproduces the monolithic plate:
    the director-form slope tie transfers bending moment across the seam (no hinge)."""
    w_mono = _bending_monolith()
    assert abs(_bending_two_patch_director(4, 4) - w_mono) / abs(w_mono) < 1e-2
    assert abs(_bending_two_patch_director(4, 7) - w_mono) / abs(w_mono) < 2e-2


def test_director_continuity_matches_rotn():
    """The director form (a_n^A·a_3^B − ref)² is ROT_N written in director vectors: on a
    G1 seam it gives the same solution as the couple_kinematics ROT_N tie."""
    for left_nv, right_nv in ((4, 4), (4, 7)):
        w_rotn = _bending_two_patch(left_nv, right_nv)
        w_dir = _bending_two_patch_director(left_nv, right_nv)
        assert abs(w_dir - w_rotn) / abs(w_rotn) < 1e-4


def test_director_continuity_hier4p_reproduces_monolith():
    """director_variation is implemented for the hierarchical shells too (the KL part of
    the tilt, from the bending Cartesian slots). For ShellReissnerMindlinHier4p the
    director-form slope tie (+ψ continuity) reproduces the monolithic plate. Here the
    surface-normal tie is genuinely distinct from ROT_N — it ties the bending surface,
    not the full director field — but on a thin plate the two agree."""
    nd = 4
    el = ck.ShellReissnerMindlinHier4p(ck.PlaneStress2d(E, nu, tb))

    def clamp_and_load(prob, root, tip, names):
        c = ck.PenaltyBoundaryCondition(root.boundary(0, True), ck.GaussLegendre(5, dim=1))
        c.add(ck.Field.U_X, AD).add(ck.Field.U_Y, AD).add(ck.Field.U_Z, AD).add(ck.Field.ROT_N, AR)
        prob.add_condition(c, patch=names[0])
        lc = ck.LoadBoundaryCondition(tip.boundary(0, False), ck.GaussLegendre(5, dim=1))
        lc.add(ck.Field.U_Z, fz)
        prob.add_condition(lc, patch=names[-1])

    mono = ck.SurfacePatch.rectangle(Lb, Wb, nu=14, nv=4, deg=3, cx=Lb / 2, name="m")
    pm = ck.LinearElasticProblem([mono], el, ck.GaussLegendre.from_patch(mono))
    pm.add_constraint(ck.DirectConstraint([3], value=0.0))
    clamp_and_load(pm, mono, mono, ["m"])
    w_mono = float(_displacement(ck.solve(pm), el, mono, [[1.0, 0.5]])[0, 2])

    left = ck.SurfacePatch.rectangle(Lb / 2, Wb, nu=8, nv=4, deg=3, cx=Lb / 4, name="left")
    right = ck.SurfacePatch.rectangle(Lb / 2, Wb, nu=8, nv=7, deg=3, cx=3 * Lb / 4, name="right")
    p = ck.LinearElasticProblem([left, right], el, ck.GaussLegendre.from_patch(left))
    p.add_constraint(ck.DirectConstraint([3], value=0.0))
    seam = ck.PenaltyCouplingCondition(
        left.boundary(0, False), right.boundary(0, True), ck.GaussLegendre(6, dim=1))
    seam.couple_displacement(AD).add(ck.Field.PSI, AR).couple_director_continuity(AR)
    p.add_condition(seam, patch="left")
    clamp_and_load(p, left, right, ["left", "right"])
    w_two = float(_displacement(ck.solve(p)[left.num_control_pts * nd:], el, right, [[1.0, 0.5]])[0, 2])

    assert abs(w_two - w_mono) / abs(w_mono) < 2e-2


def test_director_continuity_hier5p_transfers_bending():
    """director_variation works for the difference-vector hierarchical 5p shell: the
    director-form slope tie transfers bending across the seam (finite, sensible tip)."""
    nd = 5
    el = ck.ShellReissnerMindlinHier5p(ck.PlaneStress2d(E, nu, tb))
    left = ck.SurfacePatch.rectangle(Lb / 2, Wb, nu=8, nv=4, deg=3, cx=Lb / 4, name="left")
    right = ck.SurfacePatch.rectangle(Lb / 2, Wb, nu=8, nv=4, deg=3, cx=3 * Lb / 4, name="right")
    p = ck.LinearElasticProblem([left, right], el, ck.GaussLegendre.from_patch(left))
    seam = ck.PenaltyCouplingCondition(
        left.boundary(0, False), right.boundary(0, True), ck.GaussLegendre(6, dim=1))
    seam.couple_displacement(AD).couple_director_continuity(AR)
    p.add_condition(seam, patch="left")
    c = ck.PenaltyBoundaryCondition(left.boundary(0, True), ck.GaussLegendre(5, dim=1))
    c.add(ck.Field.U_X, AD).add(ck.Field.U_Y, AD).add(ck.Field.U_Z, AD).add(ck.Field.ROT_N, AR)
    p.add_condition(c, patch="left")
    lc = ck.LoadBoundaryCondition(right.boundary(0, False), ck.GaussLegendre(5, dim=1))
    lc.add(ck.Field.U_Z, fz)
    p.add_condition(lc, patch="right")
    w_two = float(_displacement(ck.solve(p)[left.num_control_pts * nd:], el, right, [[1.0, 0.5]])[0, 2])
    assert abs(w_two - _bending_monolith()) / abs(_bending_monolith()) < 5e-2


def test_bending_needs_rotation_coupling():
    """Without ROT_N the partner patch is under-constrained (a seam hinge/mechanism),
    so displacement-only coupling does NOT reproduce the monolith."""
    el = ck.ShellKirchhoffLove3p(ck.PlaneStress2d(E, nu, tb))
    left = ck.SurfacePatch.rectangle(Lb / 2, Wb, nu=7, nv=4, deg=3, cx=Lb / 4, name="left")
    right = ck.SurfacePatch.rectangle(Lb / 2, Wb, nu=7, nv=4, deg=3, cx=3 * Lb / 4, name="right")
    prob = ck.LinearElasticProblem([left, right], el, ck.GaussLegendre.from_patch(left))
    _clamp_root(prob, left, "left")
    seam = ck.PenaltyCouplingCondition(
        left.boundary(0, False), right.boundary(0, True), ck.GaussLegendre(5, dim=1))
    seam.couple_displacement(AD)   # displacement only -> seam hinges
    prob.add_condition(seam, patch="left")
    lc = ck.LoadBoundaryCondition(right.boundary(0, False), ck.GaussLegendre(4, dim=1))
    lc.add(ck.Field.U_Z, fz)
    prob.add_condition(lc, patch="right")
    w_two = _tip_w(ck.solve(prob)[left.num_control_pts * ND:], el, right)
    assert abs(w_two - _bending_monolith()) / abs(_bending_monolith()) > 0.1
