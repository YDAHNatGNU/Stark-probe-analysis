# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Quantitative TEM analysis of tBTO nanoparticles: watershed separation of
aggregated particles and superellipse fitting of the radial contour, with strict
quality control. The exponent is dimensionless and therefore measurable without a
scale bar."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import opening, disk, remove_small_objects
from skimage.measure import label, regionprops, find_contours
from skimage.segmentation import watershed, clear_border
from skimage.feature import peak_local_max


def rc_over_L(p):
    """Superellipse exponent to corner radius of curvature divided by edge length."""
    return 2.0 ** (0.5 - 1.0 / p) / (2.0 * (p - 1.0))


def p_from_rc_over_L(target):
    from scipy.optimize import brentq
    return brentq(lambda p: rc_over_L(p) - target, 2.0001, 40.0)


# ------------------------------------------------------------------
def segment(path, sigma=2.0, min_area=3000, split=True, thr_scale=1.08):
    img = np.array(Image.open(path).convert("L")).astype(float)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    sm = gaussian(img, sigma=sigma)
    thr = threshold_otsu(sm) * thr_scale
    bw = sm < thr
    bw = opening(bw, disk(3))
    bw = remove_small_objects(bw, min_area)
    bw = ndi.binary_fill_holes(bw)

    if split and bw.any():
        dist = ndi.distance_transform_edt(bw)
        rad = max(8, int(0.6 * dist.max()))
        coords = peak_local_max(dist, min_distance=rad, labels=bw,
                                exclude_border=False)
        markers = np.zeros(bw.shape, dtype=int)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i
        if markers.max() >= 1:
            lab = watershed(-dist, markers, mask=bw)
        else:
            lab = label(bw)
    else:
        lab = label(bw)
    return lab, img


# ------------------------------------------------------------------
def _resid(params, xy):
    cx, cy, a, b, th, p = params
    p = np.clip(p, 2.0, 40.0)
    x = xy[:, 0] - cx
    y = xy[:, 1] - cy
    r = np.hypot(x, y)
    ph = np.arctan2(y, x) - th
    c, s = np.abs(np.cos(ph)), np.abs(np.sin(ph))
    r_se = ((c / a) ** p + (s / b) ** p) ** (-1.0 / p)
    return r - r_se


def fit_superellipse(contour):
    """Fit a contour (N,2) in [row, col] order, with the centre as a free parameter."""
    xy = np.column_stack([contour[:, 1], contour[:, 0]])
    cx0, cy0 = xy.mean(axis=0)
    r0 = np.hypot(xy[:, 0] - cx0, xy[:, 1] - cy0)
    a0 = np.median(r0)
    x0 = [cx0, cy0, a0, a0, 0.0, 4.0]
    lo = [cx0 - 0.3*a0, cy0 - 0.3*a0, a0*0.4, a0*0.4, -np.pi/4, 2.0]
    hi = [cx0 + 0.3*a0, cy0 + 0.3*a0, a0*2.0, a0*2.0,  np.pi/4, 30.0]
    res = least_squares(_resid, x0, args=(xy,), bounds=(lo, hi),
                        loss="soft_l1", f_scale=2.0, max_nfev=8000)
    cx, cy, a, b, th, p = res.x
    resid = _resid(res.x, xy)
    r_mean = float(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy).mean())
    return dict(cx=cx, cy=cy, a=a, b=b, theta=th, p=p,
                rms=float(np.sqrt(np.mean(resid**2))),
                med_abs=float(np.median(np.abs(resid))),
                maxerr=float(np.max(np.abs(resid))),
                r_mean=r_mean)


# ------------------------------------------------------------------
def analyze_image(path, min_area=3000, ar_max=1.20,
                  solidity_min=0.94, rms_frac_max=0.03):
    """Returns a list of accepted particles (those passing quality control)."""
    lab, img = segment(path, min_area=min_area)
    lab = clear_border(lab)
    out = []
    for rp in regionprops(lab):
        if rp.area < min_area or rp.solidity < solidity_min:
            continue
        mask = (lab == rp.label)
        cs = find_contours(mask.astype(float), 0.5)
        if not cs:
            continue
        c = max(cs, key=len)
        if len(c) < 120:
            continue
        f = fit_superellipse(c)
        f["ar"] = max(f["a"], f["b"]) / min(f["a"], f["b"])
        f["rms_frac"] = f["med_abs"] / f["r_mean"]
        if f["ar"] > ar_max:
            continue
        if f["rms_frac"] > rms_frac_max:
            continue
        f["area_px"] = int(rp.area)
        f["L_px"] = f["a"] + f["b"]
        f["solidity"] = float(rp.solidity)
        f["contour"] = c
        f["file"] = path
        out.append(f)
    return out, img, lab
