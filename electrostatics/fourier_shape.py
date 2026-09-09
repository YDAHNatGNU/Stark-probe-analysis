# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Fourier decomposition of the contour radius profile, used to (1) identify [001]
projections and (2) estimate the rounding exponent independently of the direct
fit. A calibration curve is built from synthetic superellipses."""
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def radial_profile(xy, cx, cy, nphi=512):
    """Contour points to r(phi) on a uniform angular grid."""
    x = xy[:, 0] - cx
    y = xy[:, 1] - cy
    phi = np.arctan2(y, x)
    r = np.hypot(x, y)
    o = np.argsort(phi)
    phi, r = phi[o], r[o]
    phi_ext = np.concatenate([phi - 2*np.pi, phi, phi + 2*np.pi])
    r_ext = np.concatenate([r, r, r])
    g = np.linspace(-np.pi, np.pi, nphi, endpoint=False)
    return g, interp1d(phi_ext, r_ext)(g)


def fourier_amps(rprof, mmax=8):
    """A_m / A_0 for m = 1..mmax; A_0 is the mean radius."""
    n = len(rprof)
    F = np.fft.rfft(rprof) / n
    A0 = np.abs(F[0])
    return np.array([2*np.abs(F[m]) / A0 for m in range(1, mmax+1)]), A0


def synth_superellipse_profile(p, nphi=512, theta=0.0):
    g = np.linspace(-np.pi, np.pi, nphi, endpoint=False)
    ph = g - theta
    c, s = np.abs(np.cos(ph)), np.abs(np.sin(ph))
    return g, (c**p + s**p) ** (-1.0/p)


_P_GRID = np.linspace(2.0, 20.0, 200)
_A4_GRID = np.array([fourier_amps(synth_superellipse_profile(p)[1])[0][3]
                     for p in _P_GRID])


def p_from_A4(a4):
    """A4/A0 to p, by inverting the calibration table."""
    if a4 <= _A4_GRID[0]:
        return 2.0
    if a4 >= _A4_GRID[-1]:
        return 20.0
    f = interp1d(_A4_GRID, _P_GRID)
    return float(f(a4))


if __name__ == "__main__":
    print("Calibration check: synthetic superellipse -> A4/A0 -> recovered p")
    print(f"{'p_true':>7} {'A4/A0':>9} {'p_rec':>7} {'A2':>8} {'A6':>8}")
    for p in (2, 3, 4, 5, 6, 8, 10, 14):
        _, prof = synth_superellipse_profile(float(p), theta=0.3)
        A, A0 = fourier_amps(prof)
        print(f"{p:>7} {A[3]:>9.4f} {p_from_A4(A[3]):>7.2f} "
              f"{A[1]:>8.4f} {A[5]:>8.4f}")

    print("\nA4 for control shapes (not [001]):")
    g = np.linspace(-np.pi, np.pi, 512, endpoint=False)
    print(f"  circle        A4={fourier_amps(np.ones_like(g))[0][3]:.4f}")
    hexa = (np.abs(np.cos(g))**6 + np.abs(np.sin(g))**6)**(-1/6)
    r6 = 1.0 / np.cos(((g % (np.pi/3)) - np.pi/6))
    A6, _ = fourier_amps(r6)
    print(f"  hexagon   A4={A6[3]:.4f}  A6={A6[5]:.4f}")
    a, b = 1.3, 1.0
    rell = a*b/np.sqrt((b*np.cos(g))**2 + (a*np.sin(g))**2)
    Ae, _ = fourier_amps(rell)
    print(f"  ellipse AR=1.3       A4={Ae[3]:.4f}  A2={Ae[1]:.4f}")
