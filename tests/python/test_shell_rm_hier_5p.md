# `ShellReissnerMindlinHier5p`

Validation of the hierarchic 5-parameter shell against the numerical benchmarks
in Echter et al. [[1]](#ref-1)[[2]](#ref-2).

## Case 1 — Simply supported (Navier) square plate

**Setup.** A flat square plate of side `L_x = L_y = L = 10.0` (`E = 1000.0`,
`ν = 0.3`), simply supported on all four edges, under a uniform transverse load
`q_z = 1.0 · t³` scaled by the cube of the thickness so that the centre deflection
is `t`-independent in the thin limit.

**Discretization.** A single patch of 10×10 biquadratic (`p = 2`) elements, no
membrane locking treatment.

**Table 2 — center deflection `w_z,max` vs slenderness `L/t`:**

| Shell formulation | 5 | 10 | 100 | 1000 | 10000 |
|-------------------|--------|--------|--------|--------|--------|
| 3p                | 0.4423 | 0.4423 | 0.4423 | 0.4423 | 0.4423 |
| 5p-stand.         | 0.5845 | 0.4947 | 0.4367 | 0.3905 | 0.3878 |
| **5p-hier.**      | 0.5839 | 0.4938 | 0.4431 | 0.4423 | 0.4423 |
| 7p-hier.          | 0.5837 | 0.4936 | 0.4429 | 0.4421 | 0.4420 |

---

## Case 2 — Cylindrical shell strip (§5.2)

**Setup.** A cylindrical segment of radius `R = 10.0` and width `L_y = 1.0`
(`E = 1000.0`, `ν = 0.0`), clamped along `x = 0` and free at the opposite edge,
loaded by a constant radial line load `q_x = 0.1 · t³` on the free edge.

**Discretization.** A single patch of 10 × 1 biquadratic (`p = 2`) elements, no 
membrane locking treatment.

**Table 3 / Table 4 — tip radial displacement `u_x` vs slenderness `R/t`:**

| Shell formulation | 5 | 10 | 100 | 1000 | 10000 |
|-------------------|--------|--------|--------|--------|--------|
| 3p               | 0.9238 | 0.9326 | 0.6635 | 0.0225 | 0.0002 |
| 5p-stand.        | 0.9320 | 0.9357 | 0.6447 | 0.0206 | 0.0002 |
| **5p-hier.**     | 0.9302 | 0.9342 | 0.6635 | 0.0225 | 0.0002 |

---

## Case 3 — Scordelis–Lo roof (dissertation [[2]](#ref-2) §7.1)

**Setup.** The classic Scordelis–Lo roof: a cylindrical shell segment with
`R = 25`, length `L = 50`, thickness `t = 0.25` and total arc `80°`
(`E = 4.32·10⁸`, `ν = 0.0`, slenderness `R/t = 100`), supported by rigid
diaphragms at the two spanwise ends (`u_x = u_z = 0`, the two longitudinal edges
free) and loaded by a self-weight dead load of `90.0` per unit area.

**Discretization.** The full roof is a single biquadratic (`p = 2`) patch — no
symmetry, matching the dissertation — refined by control points per edge
`CP ∈ {7, 11, 19, 35}`, no membrane locking treatment.

**Table 7.1 — vertical displacement `w_z,A` vs CP per edge:**

| Shell formulation | 7 | 11 | 19 | 35 |
|-------------------|--------|--------|--------|--------|
| 3p               | 0.1151 | 0.2584 | 0.2967 | 0.3003 |
| 5p-stand.        | 0.1101 | 0.2480 | 0.2956 | 0.3008 |
| **5p-hier.**     | 0.1151 | 0.2585 | 0.2970 | 0.3008 |
| 7p-hier.         | 0.1151 | 0.2585 | 0.2970 | 0.3008 |

---

## References

<a id="ref-1"></a>[1] R. Echter, B. Oesterle, M. Bischoff, *"A hierarchic family
of isogeometric shell finite elements"*, Comput. Methods Appl. Mech. Engrg.
**254** (2013) 170–180.
[doi:10.1016/j.cma.2012.10.018](https://doi.org/10.1016/j.cma.2012.10.018)

<a id="ref-2"></a>[2] R. Echter, *"Isogeometric analysis of shells"*, PhD
dissertation, Bericht Nr. 59, Institut für Baustatik und Baudynamik, Universität
Stuttgart, 2013.
[doi:10.18419/opus-433](https://doi.org/10.18419/opus-433)
