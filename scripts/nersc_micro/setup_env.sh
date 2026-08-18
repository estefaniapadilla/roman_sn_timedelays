#!/bin/bash
# One-time environment setup on Perlmutter (run on a login node).
# Creates a venv in $SCRATCH/envs/slsim_micro with the latest slsim,
# as requested by the dataset provider.
set -euo pipefail

module load python

ENVDIR="${SCRATCH}/envs/slsim_micro"
SRCDIR="${SCRATCH}/src"

python -m venv "${ENVDIR}"
source "${ENVDIR}/bin/activate"
pip install --upgrade pip

mkdir -p "${SRCDIR}"
cd "${SRCDIR}"
if [[ ! -d slsim ]]; then
    git clone https://github.com/LSST-strong-lensing/slsim.git
fi
cd slsim
git pull
pip install -e .

# Light-curve generation extras (no GPU libs needed: magmaps are cached).
# skypy: imported unconditionally by slsim.Pipelines but not declared as a
# hard dependency by slsim's own install.
pip install sncosmo speclite tqdm skypy

python - <<'EOF'
import slsim, sncosmo, speclite
from slsim.Pipelines import roman_speclite
roman_speclite.configure_roman_filters()
print("slsim OK:", slsim.__file__)
EOF

echo "Environment ready: source ${ENVDIR}/bin/activate"
