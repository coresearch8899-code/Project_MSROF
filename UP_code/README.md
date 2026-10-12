# Export remuneration, utility margins and institutional solar PV (Utilities Policy version)

Code for the article "Export remuneration, utility margins and the solar capacity of public institutions: evidence from 15-minute campus metering in Thailand". MSROF is the working acronym of the project during development.

## Requirements
Python 3.13 or later; `pip install -r requirements.txt` (NumPy 2.5, SciPy 1.18, pandas 3.0, Matplotlib 3.11, Pillow 12.3, lxml).

## Run sequence
| Step | Command | Output |
|---|---|---|
| 1 | `python 00_prepare_measured_data.py <data folder>` (reads `Data_load/*.xls`, `Data_PV/*`) | `prepared/measured_2025.npz`, `prepared/prep_summary.json` |
| 2 | `python 00b_interval_validation.py <data folder>` | meter screening (Table S2), 30-min PV validation (Table S4), `prepared/interval_validation.json`, `prepared/pv_interval_2025.npz` |
| 3 | `python 01_landuse_maps.py <land-use map> <PV-zone map>` | `outputs/landuse_raster_check.json`, `figures/fig2_maps.png`, `figures/figS1_landuse_shares.png` |
| 4 | `python 02_msrof_analysis.py` | PV, dispatch, billing and economic model; `outputs/results.json`, `outputs/profiles.npz` (about 5 min) |
| 5 | `python 06_utility_policy.py` | `outputs/up_*`: sweeps, block decisions and stopping rule, 50 kWp step test, compensation schemes (uncompensated export, curtailment, counterfactual net billing and net metering) with itemized utility accounts and worked example, joint view, cost-ordered sensitivity, rule tests on 250 and 50 kWp grids, PV-profile substitution and short-term PV-error simulation, Ft sensitivity, Monte Carlo at the fixed design (with correlated variant), synthetic years (independent months and 3-month blocks), observed 12-month windows and a typical year (about 8 min) |
| 6 | `python 09_up_mc_reoptimization.py` | capacity re-optimized in each of the 10,000 Monte Carlo draws, at 2.20 THB/kWh up to the full surface potential: `outputs/up_mc_reopt.*` (about 15 min on two cores) |
| 7 | `python 07_up_figures.py` | Figs. 1 and 3 to 8, Figs. S2 and S3 in `figures_up/` (900 dpi); copies Fig. 2 and Fig. S1 |
| 8 | `python 08_up_graphical_abstract.py` | graphical abstract |

All tariff, cost and policy inputs (Table S5) are embedded in the scripts. Seeds: Monte Carlo `default_rng(20261007)`, correlated variant `default_rng(20261008)`, synthetic weather years `default_rng(7)`, block bootstrap `default_rng(8)`, PV-error simulation `default_rng(9)`. Re-running steps 5 to 8 reproduces every number and figure exactly.

## Data access
1. **Open (this archive):** all scripts, `outputs/` (the derived results behind every table and figure, including the Monte Carlo draws and results, sweeps, block decisions, scheme curves and weather-year results) and the summary statistics in `prepared/prep_summary.json` and `prepared/interval_validation.json`. With these, every table can be checked and steps 7 and 8 regenerate all figures except Figs. 2, 3, S1 and S2.
2. **On request, subject to the owners' permission:** the raw 15-minute meter exports (Chiang Rai Rajabhat University), the raw 30-minute monitoring logs of the CRH Solar Rooftop installation (CRH operator), the prepared 15-minute series (`prepared/measured_2025.npz`, `prepared/pv_interval_2025.npz`, `outputs/profiles.npz`) and the two 500-dpi map images. These are needed to run steps 1 to 6 and to draw Figs. 2, 3, S1 and S2.
3. **Verifiable only:** without the restricted data, results can be checked against the archived outputs but not recomputed from raw data.

## License
MIT (code). Repository: https://github.com/coresearch8899-code/Project_MSROF
