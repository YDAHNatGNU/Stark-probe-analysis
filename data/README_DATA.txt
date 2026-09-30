Input data read by the analysis scripts.

  Spectrum_Rawdata_tBTO_SiO2_in_EtOH.xlsx
      50 single-particle fluorescence spectra (25 tBTO at d = 0, 25 SiO2
      control); first column photon energy (eV), remaining columns intensity.
      Used by:  lineshape/cumulant_analysis.py

  KPFM_raw.xlsx
      KPFM line profiles. Three sheets (tBTO, cBTO, SiO2); within each sheet,
      one block per region with a position column (um) and five potential
      columns (mV) for lift heights of 0, 5, 10, 15 and 20 nm.
      Used by:  kpfm/kpfm_v1.py, kpfm/kpfm_z0.py, kpfm/make_values_xlsx.py

  tBTO_KPFM_Raw.xlsx, cBTO_KPFM_Raw.xlsx
      Single-region raw exports used for the two-sample comparison.
      Used by:  kpfm/kpfm_both.py

  ef_data.npy
      Stark fields EF_Local(d) extracted from the fluorescence linewidths.
      electrostatics/rebuild_data.py regenerates it from the linewidth table of
      Source Data and checks that the result is identical.
      Used by:  electrostatics/booth_model.py, electrostatics/selfcons.py,
                electrostatics/corr_model.py, electrostatics/table_s62.py

  budget4.json
      Enhancement budget used by the exclusion test.
      Used by:  electrostatics/corr_model.py

  halfdecay_mc.json
      Measured half-decay distances listed in Table S5.1 (Monte Carlo means and
      standard deviations). The script that generated this file was not
      preserved; booth_model.py prints these values next to the half-decay
      distances it recomputes from ef_data.npy by Eq. S5.8.
      Used by:  electrostatics/booth_model.py

  run_b.json
      Bare-particle reference used by the shell decomposition.
      Used by:  electrostatics/r3a_recalc.py

  TEM_1.tif, TEM_2.tif, TEM_3.tif (in data/ or data/TEM/)
      Representative TEM micrographs used for particle shape metrology.
      Used by:  electrostatics/analyze_tem.py

Solvent properties (eps_r, refractive index, density and molar mass at
298.15 K) are tabulated inside electrostatics/booth_model.py and are not held
in a separate file.
