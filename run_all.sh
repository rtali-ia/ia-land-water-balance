#!/usr/bin/env bash
# Full pipeline. Pass site IDs to steps 01-04 to limit them; 05-06 use all sites.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-.venv/bin/python}
$PY 00_select_sites.py
$PY 01_download_data.py "$@"
$PY 02_prep_data.py "$@"
$PY 03_basin_attributes.py
$PY 04_calibrate.py "$@"
$PY 07_annual_update.py "$@"
$PY 05_regionalize.py
$PY 06_plot_results.py
