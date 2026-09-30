# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Population analysis of the TEM images: particle segmentation, contour extraction,
Fourier shape discrimination and measurement of the rounding exponent, with
quality control on boundary contact, elongation (A2), hexagonal character (A6)
and fit residual. Produces overlay images for visual inspection."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import regionprops, find_contours
from skimage.segmentation import clear_border
from tem_fit2 import segment, fit_superellipse, rc_over_L
from fourier_shape import radial_profile, fourier_amps, p_from_A4

MEDIA = "tem/unpacked/word/media"

TBTO = {}
if os.path.exists("tem/pairs.tsv"):
    # full image set with its caption index (population value p = 5.5)
    for line in open("tem/pairs.tsv"):
        img, cap = line.strip().split("\t")
        if cap.startswith("tBTO"):
            mag = cap.split("__")[1]
            TBTO[img] = mag
else:
    # representative micrographs deposited with the capsule (data/TEM)
    _data = os.environ.get("DATA_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    # micrographs may sit in data/TEM or directly in data (TEM_*.tif)
    MEDIA = os.path.join(_data, "TEM")
    if not os.path.isdir(MEDIA):
        MEDIA = _data
    for img in sorted(os.listdir(MEDIA)):
        if img.upper().startswith("TEM") and img.lower().endswith((".tif", ".tiff", ".jpg", ".png")):
            TBTO[img] = "-"
    print(f"Representative micrographs in {MEDIA}: {len(TBTO)} images. "
          "The population value p = 5.5 (IQR 4.6-6.3) comes from the full image set.")

A2_MAX = 0.085
A6_REL = 0.60
A4_MIN = 0.030
MED_FRAC_MAX = 0.030
SOLID_MIN = 0.94
MIN_AREA = 4000


def analyze(path):
    lab, img = segment(path, min_area=MIN_AREA)
    lab = clear_border(lab)
    res = []
    for rp in regionprops(lab):
        if rp.area < MIN_AREA or rp.solidity < SOLID_MIN:
            continue
        m = (lab == rp.label)
        cs = find_contours(m.astype(float), 0.5)
        if not cs:
            continue
        c = max(cs, key=len)
        if len(c) < 150:
            continue
        f = fit_superellipse(c)
        xy = np.column_stack([c[:, 1], c[:, 0]])
        g, prof = radial_profile(xy, f["cx"], f["cy"])
        A, A0 = fourier_amps(prof)
        rec = dict(area=int(rp.area), solidity=float(rp.solidity),
                   p_fit=float(f["p"]), a=float(f["a"]), b=float(f["b"]),
                   med_frac=float(f["med_abs"] / f["r_mean"]),
                   A2=float(A[1]), A3=float(A[2]), A4=float(A[3]),
                   A6=float(A[5]), A0=float(A0),
                   p_fourier=float(p_from_A4(A[3])),
                   cx=float(f["cx"]), cy=float(f["cy"]),
                   theta=float(f["theta"]), contour=c)
        ok = (rec["A2"] <= A2_MAX and rec["A4"] >= A4_MIN
              and rec["A6"] <= A6_REL * rec["A4"]
              and rec["med_frac"] <= MED_FRAC_MAX)
        rec["pass"] = bool(ok)
        res.append(rec)
    return res, img, lab


all_rec = []
def _order(kv):
    digits = "".join(ch for ch in kv[0] if ch.isdigit())
    return int(digits) if digits else 0


for img_name, mag in sorted(TBTO.items(), key=_order):
    path = os.path.join(MEDIA, img_name)
    try:
        recs, img, lab = analyze(path)
    except Exception as e:
        print(f"  {img_name}: ERROR {e}")
        continue
    for r in recs:
        r["file"] = img_name
        r["mag"] = mag
        all_rec.append(r)

passed = [r for r in all_rec if r["pass"]]
print(f"Detected particles {len(all_rec)}, passing QC: {len(passed)}")
print(f"\n{'file':>10} {'mag':>7} {'A2':>7} {'A4':>7} {'A6':>7} "
      f"{'p_fit':>6} {'p_A4':>6} {'medf':>6}")
for r in all_rec:
    flag = "OK " if r["pass"] else "rej"
    print(f"{r['file']:>10} {r['mag']:>7} {r['A2']:>7.3f} {r['A4']:>7.3f} "
          f"{r['A6']:>7.3f} {r['p_fit']:>6.2f} {r['p_fourier']:>6.2f} "
          f"{r['med_frac']:>6.3f}  {flag}")

if passed:
    pf = np.array([r["p_fourier"] for r in passed])
    pl = np.array([r["p_fit"] for r in passed])
    rc = np.array([rc_over_L(p) for p in pf])
    print(f"\n=== Population passing QC (n={len(passed)}) ===")
    print(f"  p (Fourier) : median {np.median(pf):.2f}  "
          f"mean {pf.mean():.2f} ± {pf.std(ddof=1):.2f}  "
          f"range {pf.min():.2f}-{pf.max():.2f}")
    print(f"  p (LSQ fit) : median {np.median(pl):.2f}  "
          f"mean {pl.mean():.2f} ± {pl.std(ddof=1):.2f}")
    print(f"  r_c/L       : median {np.median(rc):.3f}  "
          f"IQR {np.percentile(rc,25):.3f}-{np.percentile(rc,75):.3f}")
    json.dump([{k: v for k, v in r.items() if k != "contour"}
               for r in all_rec], open("tem_results.json", "w"), indent=1)

show = [r for r in all_rec if r["pass"]][:6]
if not show:
    show = all_rec[:6]
if show:
    n = len(show)
    fig, axes = plt.subplots(2, (n+1)//2, figsize=(4.2*((n+1)//2), 8.6))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, show):
        from PIL import Image as PILImage
        im = np.array(PILImage.open(os.path.join(MEDIA, r["file"]))
                      .convert("L"))
        ax.imshow(im, cmap="gray")
        c = r["contour"]
        ax.plot(c[:, 1], c[:, 0], "-", color="lime", lw=1.2,
                label="segmented")
        th = np.linspace(-np.pi, np.pi, 400)
        ph = th - r["theta"]
        cc, ss = np.abs(np.cos(ph)), np.abs(np.sin(ph))
        rr = ((cc/r["a"])**r["p_fit"] + (ss/r["b"])**r["p_fit"])**(-1/r["p_fit"])
        ax.plot(r["cx"] + rr*np.cos(th), r["cy"] + rr*np.sin(th),
                "--", color="red", lw=1.4, label="superellipse fit")
        ax.set_title(f"{r['file']} ({r['mag']})\n"
                     f"p={r['p_fourier']:.1f}, $r_c/L$={rc_over_L(r['p_fourier']):.3f}",
                     fontsize=10)
        ax.axis("off")
        ax.legend(fontsize=7, loc="lower right")
    for ax in axes[len(show):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("fig4_tem_overlay.png", dpi=115, bbox_inches="tight")
    print("\nsaved: fig4_tem_overlay.png")
