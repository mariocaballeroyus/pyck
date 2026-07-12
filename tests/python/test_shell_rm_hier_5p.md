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

## Case 3 — Scordelis–Lo roof (§5.3)

**Setup.** The classic Scordelis–Lo roof: a cylindrical shell segment with
`R = 25`, length `L = 50`, thickness `t = 0.25` and arc `φ = 40°`
(`E = 4.32·10⁸`, `ν = 0.0`, slenderness `R/t = 100`), supported by rigid
diaphragms at `y = 0, L` (`u_x = w_z = 0`, other edges free) and loaded by a
self-weight dead load of `90.0` per unit area.

**Discretization.** Biquadratic patches refined by control points per side
`CP ∈ {3, 5, 7, 9, 11, 13}`, no membrane locking treatment.

**Table 5 — vertical displacement `w_z,A` vs CP per side:**

| Shell formulation | 3 | 5 | 7 | 9 | 11 | 13 |
|-------------------|--------|--------|--------|--------|--------|--------|
| **5p-hier.-HS**   | 0.7680 | 0.2517 | 0.2998 | 0.3001 | 0.3005 | 0.3006 |
| 3p-HS (ref)       | 0.7679 | 0.2516 | 0.2996 | 0.3000 | 0.3003 | 0.3005 |
| 7p-hier.-HS       | 0.7681 | 0.2517 | 0.2999 | 0.3001 | 0.3005 | 0.3007 |

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
