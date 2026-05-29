#!/usr/bin/env bash
# One-command replication of all numerical results in the paper.
#
#   bash reproduce.sh
#
# Creates a self-contained virtualenv at ./.venv, installs pinned
# dependencies from requirements.txt, runs the four replication
# scripts in code/, and writes outputs to ./results/.
#
# Wall-clock: roughly 10-15 minutes on a recent laptop.  Python 3.10+.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

VENV="$HERE/.venv"

echo
echo "================================================================"
echo " Replication packet for Khunjua & Malik (2026)"
echo " Working directory: $HERE"
echo "================================================================"

if [ ! -d "$VENV" ]; then
  echo
  echo ">>> Creating fresh virtualenv at $VENV"
  python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo
echo ">>> Installing pinned dependencies (requirements.txt) ..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

mkdir -p results

echo
echo "================================================================"
echo " [1/4] Table 1 -- four validation scenarios (S1 through S4)"
echo "================================================================"
python code/02_table_1_scenarios.py 2>&1 | tee results/table_1_scenarios.txt

echo
echo "================================================================"
echo " [2/4] Paired bootstrap test -- S4 vs S2 on common 2021-22 cohort"
echo "================================================================"
python code/03_paired_bootstrap.py 2>&1 | tee results/paired_bootstrap.txt

echo
echo "================================================================"
echo " [3/4] Table A.4 -- Phase I anchored replication"
echo "================================================================"
python code/04_phase1_table.py 2>&1 | tee results/table_a4_phase1.txt

echo
echo "================================================================"
echo " [4/4] Figure 1 -- validation-scenarios bar chart"
echo "================================================================"
python code/05_figure_scenarios.py 2>&1 | tee results/figure_1.log

echo
echo "================================================================"
echo " Reproduction complete.  Outputs in $HERE/results/:"
echo "================================================================"
ls -la "$HERE/results/"
