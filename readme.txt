============================================================
SUPPLEMENTARY SOFTWARE

  "Solvent-Reconstructed Electric Fields at Solid-Liquid Interfaces"
  Cho et al.
============================================================

All analysis is written in Python and uses only open-source libraries; no
commercial solver is required. The package is organised into four folders that
correspond to the four analyses reported in the paper.


------------------------------------------------------------
1. CONTENTS
------------------------------------------------------------

electrostatics/   Boundary-element electrostatics (Supplementary Note 3)
------------------------------------------------------------
Solves the electrostatics of a uniformly polarised ferroelectric nanoparticle
in a dielectric medium, with no adjustable parameters.

  ferroelectric_bem.py   icosphere mesh, superellipsoid map, geometric matrix M,
                         single-interface solve, analytical sphere reference
  accurate_bem.py        higher-accuracy solve and field evaluation
  core_shell_bem.py      mesh offsetting, multi-surface assembly,
                         concentric-sphere reference
  multilayer_bem.py      arbitrary layer stack solve
  asm_cache.py           caches M across permittivity updates
  surf_rms.py            area-weighted surface RMS on the probe shell
                         (writes surf_rms.json)
  analyze_tem.py         TEM segmentation, superellipse fit, Fourier exponent
                         and quality control (runs on data/TEM)
  tem_fit2.py            shape fitting utilities
  fourier_shape.py       Fourier exponent from the contour
  validate.py            baseline check with centroid collocation (eps_m = 1)
  validation_data.py     Table S3.1 (sphere, block V1) and Table S4.2
                         (core-shell with Richardson extrapolation, block V2)
  rebuild_data.py        EF_Local(d) from the Source Data linewidth table
                         (checks identity with data/ef_data.npy)
  booth_model.py         Kirkwood-Frohlich inversion, characteristic field E_c,
                         saturation curves, half-decay distances by Eq. S5.8
                         (writes booth_results.npy)
  selfcons.py            self-consistent graded permittivity, sphere
                         (writes selfcons_scan.json)
  run_shapes.py          multilayer self-consistent scan on the measured shape
                         (writes shape_selfcons.json) and shape factors
  table_s62.py           near-surface enhancement, Table S6.2
  corr_model.py          local-ansatz exclusion test (writes corr_model.json)
  r1a_recalc.py          unscreened ceiling on a single consistent basis
  r3a_recalc.py          one-variable-at-a-time decomposition of the shell
                         effect
  calc_table.py          field and potential tables for all media
  rms100.py              extended distance range

lineshape/        Differential cumulant analysis (Supplementary Note 1)
------------------------------------------------------------
  cumulant_analysis.py   cumulant estimator, bootstrap over spectra and the
                         window scan; stand-alone

kpfm/             KPFM line-profile analysis
------------------------------------------------------------
  kpfm_v1.py             parses raw line profiles, extracts step-height contrast
  kpfm_z0.py             z = 0 analysis over all regions: robust estimation,
                         hierarchical bootstrap, definition sensitivity
  kpfm_paper_values.py   values reported in the paper, from the regions
                         analysed there
  kpfm_both.py           one tBTO and one cBTO region across lift heights
                         (descriptive)
  make_values_xlsx.py    exports the extracted values to spreadsheet form

data/             Input data (held inside the code folder so that the
                  whole package is self-contained)
------------------------------------------------------------
  Fluorescence spectra (25 tBTO at d = 0, 25 SiO2 control), KPFM line
  profiles (five, four and five regions for tBTO, cBTO and SiO2, each at
  five lift heights), three intermediate inputs and representative TEM
  micrographs. data/README_DATA.txt describes each file and the script
  that reads it. Solvent properties are tabulated inside
  electrostatics/booth_model.py.

Top level
------------------------------------------------------------
  run.sh                 runs the complete analysis (see reproduction.txt)
  reproduction.txt       step-by-step reproduction and expected output
  LICENSE                MIT License


This package contains the code that produces the numerical results reported in
the paper. Figures were redrawn from these outputs for publication, so the
plotting scripts are not included.


------------------------------------------------------------
2. REQUIREMENTS
------------------------------------------------------------

Tested with:

  Python        3.12.3
  numpy         2.4.4
  scipy         1.17.1
  matplotlib    3.10.8
  openpyxl      3.1.5
  scikit-image  0.26.0

On Code Ocean these versions are pinned in the capsule environment (pip).

No compilation step is required. Cost is dominated by assembly of the geometric
matrix M, which scales as O(N^2) in panel count; the converged single-interface
calculations (subdivision level 3-4, 1280-5120 panels) run in minutes on a
desktop workstation.


------------------------------------------------------------
3. LICENSE
------------------------------------------------------------

MIT License. See LICENSE.


------------------------------------------------------------
4. CONTACT
------------------------------------------------------------

Correspondence regarding the code:
  D.S.    daehaseo@postech.ac.kr
  Y.A.    ydahn@gnu.ac.kr
  J.H.P.  yakte@gnu.ac.kr


============================================================

Reproduction steps, expected outputs and instructions for running the
package on Code Ocean are given in reproduction.txt.
