Place the following input files in this folder before running the analysis.

  Spectrum_Rawdata_tBTO_SiO2_in_EtOH.xlsx
      50 single-particle fluorescence spectra (25 tBTO at d = 0, 25 SiO2 control),
      first column photon energy (eV), remaining columns intensity.
      Used by:  lineshape/cumulant_analysis.py

  KPFM_raw.xlsx
      KPFM line profiles. Three sheets (tBTO, cBTO, SiO2); within each sheet,
      one block per region with a position column (um) and five potential
      columns (mV) for lift heights 0, 5, 10, 15 and 20 nm.
      Used by:  kpfm/kpfm_v1.py

  solvent_properties.csv
      Tabulated solvent properties at 298.15 K: name, eps_r, refractive index n,
      density rho (g/cm3), molar mass M (g/mol).
      Used by:  electrostatics/booth_model.py

  TEM/
      TEM micrographs used for particle shape metrology.
      Used by:  electrostatics/analyze_tem.py

[NOTE TO AUTHORS] The spectrum and KPFM files are present in the submission;
solvent_properties.csv and the TEM images must be added before upload.

  tBTO_KPFM_Raw.xlsx / cBTO_KPFM_Raw.xlsx
      Single-region raw exports, retained for the two-sample comparison script.
      Used by:  kpfm/kpfm_both.py
