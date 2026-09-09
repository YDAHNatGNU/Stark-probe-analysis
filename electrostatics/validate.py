# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Validation of the boundary-element exterior field against the analytical sphere
solution, together with a mesh convergence scan and a sensitivity test on the
background permittivity."""
import numpy as np
from ferroelectric_bem import (build_geometry, solve_sigma, field_at,
                               sphere_analytic, EPS0)

Ps, eps_b, eps_m, R = 0.26, 6.0, 1.0, 1.0

def eval_points(rfac, ntheta=25):
    th = np.linspace(0.01, np.pi - 0.01, ntheta)
    x = rfac * R * np.sin(th)
    z = rfac * R * np.cos(th)
    y = np.zeros_like(th)
    return np.column_stack([x, y, z]), th

print("=== Exterior field validation (eps_b = 6, eps_m = 1) ===")
print(f"{'subdiv':>6} {'panels':>7} {'r=1.5R':>10} {'r=2R':>10} {'r=3R':>10}")
for subdiv in (2, 3, 4):
    g = build_geometry("sphere", subdiv=subdiv, R=R)
    sig, _ = solve_sigma(g, Ps, eps_b, eps_m)
    row = []
    for rfac in (1.5, 2.0, 3.0):
        pts, th = eval_points(rfac)
        _, E = field_at(pts, g, sig)
        Emag = np.linalg.norm(E, axis=1)
        _, Emag_a, _, _ = sphere_analytic(pts, Ps, eps_b, eps_m, R)
        rel = np.max(np.abs(Emag - Emag_a) / Emag_a)
        row.append(rel)
    print(f"{subdiv:>6} {len(sig):>7} {row[0]:>9.2%} {row[1]:>9.2%} {row[2]:>9.2%}")

g = build_geometry("sphere", subdiv=4, R=R)
sig, _ = solve_sigma(g, Ps, eps_b, eps_m)
for rfac in (1.05, 1.5, 2.0):
    p = np.array([[0, 0, rfac * R]])
    _, E = field_at(p, g, sig)
    _, Ea, _, _ = sphere_analytic(p, Ps, eps_b, eps_m, R)
    print(f"pole r={rfac}R: BEM |E|={np.linalg.norm(E):.3e}  analytic={Ea[0]:.3e} V/m")

print("\n=== Sensitivity to eps_b (pole, r = 1.05R) ===")
p = np.array([[0, 0, 1.05 * R]])
for eb in (6.0, 50.0, 300.0, 2000.0):
    _, Ea, _, _ = sphere_analytic(p, Ps, eb, eps_m, R)
    print(f"eps_b={eb:>7.0f}:  |E| = {Ea[0]:.3e} V/m")
