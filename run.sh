#!/usr/bin/env bash
# =============================================================================
# Entry point for the Code Ocean capsule
#
#   "Solvent-Reconstructed Electric Fields at Solid-Liquid Interfaces"
#   Cho et al.
#
# All locations are resolved at run time relative to this script, so the same
# script runs on Code Ocean and on a local machine. Input data are read from
# the data folder beside this script, or from the capsule's own data folder if
# there is none; outputs go to the results folder. Intermediate files produced
# by one step are consumed by later steps, so the order matters.
#
# Approximate run time: 15-25 min (2 cores, 8 GB).
# =============================================================================
set -e

# One BLAS/OpenMP thread per process. On shared cloud machines the default
# (one thread per visible core) can oversubscribe the CPU allocation and slow
# the many small linear solves of steps 6-8 considerably.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(dirname "$ROOT")"
OUT="${PARENT%/}"/"results"
# Input data may sit beside this script or in the capsule's own data folder.
if [ -d "$ROOT"/"data" ]; then
    DATA_DIR="$ROOT"/"data"
else
    DATA_DIR="${PARENT%/}"/"data"
fi
export DATA_DIR
export PYTHONPATH="$ROOT"/"electrostatics":"$ROOT"/"lineshape":"$ROOT"/"kpfm"
mkdir -p "$OUT"
cd "$OUT"

# Deposited intermediate files are linked into the working folder so that the
# scripts find them alongside the files they generate themselves.
for f in ef_data.npy budget4.json run_b.json halfdecay_mc.json; do
    if [ -f "$DATA_DIR"/"$f" ]; then ln -sf "$DATA_DIR"/"$f" "$f"; fi
done

run() { local s="$1"; shift; python "$ROOT"/"$s" "$@"; }
step() { echo; echo "=================================================================="; \
         echo " $1"; echo "=================================================================="; }

step "1. Differential cumulant analysis of the fluorescence line shape"
run lineshape/cumulant_analysis.py

step "2. KPFM line-profile analysis"
run kpfm/kpfm_v1.py
run kpfm/kpfm_z0.py
run kpfm/kpfm_paper_values.py
run kpfm/kpfm_both.py
run kpfm/make_values_xlsx.py

step "3. Boundary-element electrostatics: solver validation"
run electrostatics/validate.py
run electrostatics/validation_data.py

step "4. Particle shape metrology"
run electrostatics/fourier_shape.py
run electrostatics/analyze_tem.py

step "5. Linear-continuum predictions and the unscreened ceiling"
run electrostatics/surf_rms.py
run electrostatics/r1a_recalc.py
run electrostatics/r3a_recalc.py
run electrostatics/rms100.py

step "6. Dielectric saturation: characteristic field and self-consistency"
run electrostatics/rebuild_data.py
run electrostatics/booth_model.py
run electrostatics/selfcons.py

step "7. Multilayer assembly and the shape correction"
run electrostatics/asm_cache.py sphere
run electrostatics/asm_cache.py cube
run electrostatics/run_shapes.py
run electrostatics/table_s62.py

step "8. Exclusion test for the local correlation ansatz"
run electrostatics/corr_model.py

step "9. Field and potential tables over the extended distance range"
run electrostatics/calc_table.py

echo
echo "Done. All outputs are in the results folder."
echo
echo "Note: in step 4, analyze_tem.py runs the full shape pipeline (segmentation,"
echo "superellipse fit, Fourier exponent, quality control) on the three"
echo "representative micrographs in data/TEM. The population value p = 5.5"
echo "(IQR 4.6-6.3) used throughout was obtained with the same script on the"
echo "full TEM image set."
