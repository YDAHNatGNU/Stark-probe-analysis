================================================================================
SUPPLEMENTARY SOFTWARE

  "Solvent-Reconstructed Electric Fields at Solid-Liquid Interfaces"
  Cho et al.
================================================================================

All analysis is written in Python and uses only open-source libraries; no
commercial solver is required. The package is organised into four folders that
correspond to the four analyses reported in the paper.


--------------------------------------------------------------------------------
1. CONTENTS
--------------------------------------------------------------------------------

electrostatics/   Boundary-element electrostatics (Supplementary Note 3)
------------------------------------------------------------------------
Solves the electrostatics of a uniformly polarised ferroelectric nanoparticle
in a dielectric medium, with no adjustable parameters.

  ferroelectric_bem.py   icosphere mesh, superellipsoid map, geometric matrix M, single-interface solve, analytical sphere reference
  accurate_bem.py        higher-accuracy solve and field evaluation
  core_shell_bem.py      mesh offsetting, multi-surface assembly, concentric-sphere reference
  multilayer_bem.py      arbitrary layer stack solve
  asm_cache.py           caches M across permittivity updates
  surf_rms.py            area-weighted surface RMS on the probe shell -> surf_rms.json
  analyze_tem.py         TEM segmentation and superellipse fit
  tem_fit2.py            shape fitting utilities
  fourier_shape.py       Fourier exponent from the contour
  validate.py            convergence scan against analytical references
  validation_data.py      Richardson extrapolation
  booth_model.py         Kirkwood-Frohlich inversion, characteristic field E_c, saturation curves -> booth_results.npy
  selfcons.py            self-consistent graded permittivity (sphere) -> selfcons_scan.json
  run_shapes.py          multilayer self-consistent scan on the measured shape -> shape_selfcons.json
  corr_model.py          local-ansatz exclusion test -> corr_model.json
  r1a_recalc.py          unscreened ceiling on a single consistent basis
  r3a_recalc.py          one-variable-at-a-time decomposition of the shell effect
  calc_table.py          field and potential tables for all media
  rms100.py              extended distance range

lineshape/        Differential cumulant analysis (Supplementary Note 1)
------------------------------------------------------------------------
  cumulant_analysis.py   cumulant estimator, bootstrap over spectra, and the window scan; stand-alone

kpfm/             KPFM line-profile analysis
------------------------------------------------------------------------
  kpfm_v1.py             parses raw line profiles, extracts step-height contrast
  kpfm_z0.py             z = 0 analysis, robust estimation, hierarchical
                         bootstrap, definition sensitivity
  kpfm_both.py           sample-to-sample comparison
  make_values_xlsx.py    exports the extracted values to spreadsheet form

data/             Input data
------------------------------------------------------------------------
  Spectrum_Rawdata_tBTO_SiO2_in_EtOH.xlsx
      50 single-particle spectra (25 tBTO at d = 0, 25 SiO2 control).
      Used by lineshape/cumulant_analysis.py
  KPFM_raw.xlsx
      Line profiles: three samples x three regions x five lift heights.
      Used by kpfm/kpfm_v1.py
  solvent_properties.csv
      eps_r, refractive index, density and molar mass at 298.15 K.
      Used by electrostatics/booth_model.py
  TEM/
      TEM micrographs used for particle shape metrology.
      Used by electrostatics/analyze_tem.py


This package contains the code that produces the numerical results reported in
the paper. Figures were redrawn from these outputs for publication, so the
plotting scripts are not included.

--------------------------------------------------------------------------------
2. REQUIREMENTS
--------------------------------------------------------------------------------

Tested with:

  Python        3.12.3
  numpy         2.4.4
  scipy         1.17.1
  matplotlib    3.10.8
  openpyxl      3.1.5
  scikit-image  0.26.0

Install with:

  pip install numpy scipy matplotlib openpyxl scikit-image

No compilation step is required. Cost is dominated by assembly of the geometric
matrix M, which scales as O(N^2) in panel count; the converged single-interface
calculations (subdivision level 3-4, 1280-5120 panels) run in minutes on a
desktop workstation.


--------------------------------------------------------------------------------
3. REPRODUCTION
--------------------------------------------------------------------------------

Run from inside each folder. Intermediate .json / .npy files are written to the
working directory and consumed by later steps, so the order matters within a
folder.

electrostatics/
------------------------------------------------------------------------
  python analyze_tem.py       -> p = 5.5 (IQR 4.6-6.3), r_c/L = 0.139
  python validate.py          -> Table S3.1  convergence 5.7 / 2.2 / 1.0 %
  python validation_data.py    -> Table S3.3  Richardson, k = 1.17
  python surf_rms.py          -> surf_rms.json
  python booth_model.py       -> booth_results.npy  (E_c = 4.1 MV/cm)
  python selfcons.py          -> selfcons_scan.json
  python run_shapes.py        -> shape_selfcons.json
  python corr_model.py        -> corr_model.json   (exclusion test)
  python r1a_recalc.py        -> ceiling 134.8 -> 50.6 -> 45.5 MV/cm
  python r3a_recalc.py        -> geometric x0.667, dielectric x0.953
  python calc_table.py        -> sim_table.json

lineshape/
------------------------------------------------------------------------
  python cumulant_analysis.py

  Expected output at the primary window W = 80 meV:

    tBTO 25 spectra, SiO2 25 spectra

    --- W = 80 meV (primary) ---
    dkappa2 =   310.4 +-  50.8 meV^2   (t =  6.11)
    dkappa3 =  1666.9 +- 824.4 meV^3   (t =  2.02)
    gamma1  = 0.307   95% CI [0.000, 1.004]
    |m|     = 0.152   95% upper bound 0.387

  The script then repeats the analysis for windows of 60-150 meV.

kpfm/
------------------------------------------------------------------------
  python kpfm_v1.py           -> kpfm_v1.json  step-height contrast per region
  python kpfm_z0.py           -> z = 0 statistics, definition sensitivity

  Expected contrast at z = 0 (mean +- s.e.m. of three regions):

    tBTO   34.2 +- 2.5 mV
    cBTO   25.4 +- 1.6 mV
    SiO2   12.3 +- 0.6 mV


--------------------------------------------------------------------------------
4. NOTES ON THE NUMERICAL METHOD
--------------------------------------------------------------------------------

The electrostatics is solved as a Fredholm integral equation of the second kind
for the bound surface charge (the boundary-charge, or induced-surface-charge,
formulation). Because the dielectric contrast |lambda| < 1 at every interface,
the system matrix is diagonally dominant and well conditioned - a property
relied on given the near cancellation between interfaces documented in
Supplementary Note 3.

The geometric matrix M depends only on the mesh, so it is assembled once and
cached (asm_cache.py) across the permittivity updates of each self-consistent
solve.


--------------------------------------------------------------------------------
5. LICENSE
--------------------------------------------------------------------------------

MIT License. See LICENSE.


--------------------------------------------------------------------------------
6. CONTACT
--------------------------------------------------------------------------------

Correspondence regarding the code: D.S.: daehaseo@dgist.ac.kr; Y.A.: ydahn@gnu.ac.kr; J.H.P.: yakte@gnu.ac.kr

================================================================================
