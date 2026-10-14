# COUSINE

COUSINE (COUpled Soil Inorganic-organic carbon model for eNhanced wEathering) is a mechanistic model of soil
inorganic carbon formation and carbon dioxide removal (CDR) under enhanced rock weathering (ERW). This repository
contains the model code, the scripts and the figures of Tao & Houlton, *Optimizing carbon dioxide removal in
agricultural soils: A new mechanistic model of enhanced rock weathering*.

## The model

COUSINE couples two modules:

* a soil organic carbon module, the seven-pool soil carbon scheme of the Community Land Model
  version 5 (CLM5), which supplies soil CO2 from organic matter decomposition;
* a soil inorganic carbon module that tracks 20 chemical species: Al3+, Al(OH)2+, Al(OH)4-, H4SiO4, H3SiO4-,
  monovalent and divalent base cations (M+, D2+), OH-, H+, CO2(aq), CO2(g), HCO3-, CO3--, H2O, the DCO3 ion pair,
  Al(OH)3, cations held on clay exchange sites (PiD, Pi3Al2), and organic acids with their conjugate bases (SOA2D, SOAH).

The inorganic module is driven by soil CO2 supply, parent-rock weathering, the weathering of applied rock dust
(olivine, basalt or lime), cation exchange on clay phases and organic acids, net nitrogen acidity from fertiliser
use, and leaching. Mineral dissolution follows transition-state-theory kinetics with dependencies on surface area,
proton activity, temperature and saturation state. Fast equilibrium reactions and the conservation of mass and
charge are written as one matrix equation. For each driving reaction and time step, the model solves this system
implicitly (backward Euler), to convergence, by a damped Newton iteration in the logarithm of the species amounts,
so mass, charge and the equilibria hold at every step.

The model is written in PyTorch and is differentiable. Site parameters are optimised with a
biogeochemistry-informed neural network (BINN) that maps environmental covariates to model parameters and is
trained against observational benchmarks (soil pH, exchangeable cations, CEC, base saturation, soil CO2 and
parent-rock CDR) under a steady-state constraint. The BINN code is available from the source given in Xu et al.
(2026, *Geosci. Model Dev.*, 19, 6777-6795, https://doi.org/10.5194/gmd-19-6777-2026):
https://doi.org/10.5281/zenodo.20405976. The BINN-optimised parameters of the three sites are provided with the data
(see Data).

## Repository contents

| folder | content |
|---|---|
| `src_cousine/` | the model (`fun_carbon_dynamics.py`; `fun_carbon_dynamics_beerling.py` for the field-trial basalt), the simulation drivers `simulator_main_*.py`, and the site-input loader (`fun_site_inputs.py`) |
| `src_figure/` | scripts of the figures in the paper |
| `figures/` | the figures of the paper (main Figs. 2-8, Supplementary Figs. 1-12) |

| driver | experiment |
|---|---|
| `simulator_main_manage_scenario.py` | olivine, basalt and lime applications across soils, climates, dust amounts and nitrate leaching rates |
| `simulator_main_sensitivity_water.py` | CO2 turnover time and rainfall-leaching sensitivity |
| `simulator_main_sensitivity_kinetics.py` | activation energy and reaction order of dust dissolution |
| `simulator_main_sensitivity_others.py` | solute leaching, organic-acid share of CEC and exchange ratios |
| `simulator_main_beerling.py` | the basalt field trial of Beerling et al. (2024) |

The simulations cover three sites: an Alfisol under a temperate climate (site 7294), an Ultisol under a
subtropical climate (7909) and an Oxisol under a tropical climate (12224).

## Running the model

Requirements: Python 3.8 or later with PyTorch, NumPy, SciPy, pandas, netCDF4, mat73, tifffile, pygad, psutil and matplotlib.

The code reads its inputs from, and writes its outputs to, a data folder whose location is given by the
environment variable `COUSINE_DATA_PATH`:

```bash
export COUSINE_DATA_PATH=/path/to/cousine_site/
cd src_cousine
python simulator_main_manage_scenario.py
cd ../src_figure
python src_figure_matrix_simu.py
```

Each driver runs its experiment in parallel over a process pool (`main()`); the settings of an experiment, such as
the dust type, dust amounts and parameter scalars, are set at the top of the driver and in its `worker_function`.
Simulation outputs are written to `cousine_output/simu/prior_v0_pc/annual_mean/` inside the data folder, and the
figure scripts write the figures to `figures/` of this repository. Each run reports the number of Newton
block-steps and iterations.

## Data

The data folder holds the inputs of the three sites, in the directory layout of the global datasets they were
taken from:

| path | content |
|---|---|
| `Litho_Map/glim_wgs84_0point5deg_sites.csv` | lithology class (GLiM) |
| `GRUN_Runoff/parent_cdr_sites.csv` | parent-rock CDR from GRUN runoff and lithology (t CO2 ha-1 yr-1) |
| `GSDE_Data/GSDE5min_sites.csv` | soil chemistry (GSDE): bulk density, SOC, exchangeable Al and Ca+Mg, CEC, base saturation, pH, exchangeable-to-soluble ratio, maximal cation sink |
| `SIC_storage/Huang_SIC_0_30cm_sites.csv` | soil inorganic carbon (g kg-1) |
| `Soil_Clay_Minerals/clay_weight_gibbsite_sites.csv` | gibbsite (kg m-2) |
| `NO3_Leaching/rfcnleach_sites.csv` | net nitrate leaching (kg N ha-1 yr-1) |
| `AET/GLEAM_MONTH/E_1980-1999_GLEAM_v3.6a_MO_sites.csv` | monthly actual evapotranspiration 1980-1999 (GLEAM v3.6a) |
| `clm5_ouput_ncar/summary_land_mask_sites.csv` | CLM5 grid cell of each site |
| `clm5_ouput_ncar/grid_hist_input/cesm2_hist_input_*_grid_{7572,6200,2107}.mat` | CESM2/CLM5 forcing of the three sites |
| `proda_opt/grid_para_result_sites.csv` | soil organic carbon parameters (PRODA) |
| `global_grid_info/world_grid_envinfo_present_sites.csv` | normalised environmental covariates (BINN input) |
| `cousine_output/binn_opt/prior_v0_pc/version_20260315/binn_para_history_site_<site>.csv` | BINN-optimised parameters |
| `cousine_output/simu/prior_v0_pc/soc_warmup_<site>.pth` | soil organic carbon at steady state |

These site inputs were derived from the global datasets (gap filling, unit conversion and clipping at the global
99th percentile, then the values of each site).

## Figures

| figure | file | script |
|---|---|---|
| Fig. 2 | `weathering_progress_and_cdr_summary_10.0` | `src_figure_system_dynamics.py` |
| Fig. 3 | `proton_attribution_10.0` | `src_figure_proton_contribution.py` |
| Fig. 4 | `cdr_time_series_local_forcing_10.0` | `src_figure_cdr_efficiency.py` |
| Fig. 5 | `climate_gradient_10.0` | `src_figure_climate_gradient.py` |
| Fig. 6, Supp. Fig. 11 | `cdr_accu_dust_scenarios_matrix`, `cdr_efficiency_dust_scenarios_matrix` | `src_figure_matrix_simu.py` |
| Fig. 7 | `cdr_weathering_partial_dependence` | `src_figure_sensitivity_partial_dependence.py` |
| Fig. 8, Supp. Fig. 9 | `cdr_weathering_beerling_best_fit_comparison`, `cdr_weathering_beerling_nse` | `src_figure_beerling_nse.py` |
| Supp. Fig. 1 | `site_locations_map` | `src_figure_site_loc.py` |
| Supp. Fig. 2 | `climate_forcing_time_series` | `src_figure_site_forcing.py` |
| Supp. Fig. 3 | `logistic_sensitivity` | made from the global soil moisture and rainfall distributions |
| Supp. Fig. 4 | `dust_gradient_10.0` | `src_figure_dust_types.py` |
| Supp. Fig. 5 | `no3_leaching_histogram` | made from the global nitrate leaching dataset |
| Supp. Figs. 6-8 | `species_dynamics_inter_annual_variation_*` | `src_figure_species_dynamics.py` |
| Supp. Fig. 10 | `cdr_accu_n_dependence` | `src_figure_nitrogen_gradient.py` |
| Supp. Fig. 12 | `cdr_weathering_para_sensitivity` | `src_figure_sensitivity_water.py` |

## License

COUSINE is copyright © 2026 Cornell University (Cornell docket D-11800). It may be used for educational, research
and non-profit purposes without fee under the terms in [`LICENSE`](LICENSE). Commercial use requires a license from
the Center for Technology Licensing at Cornell University (ctl-connect@cornell.edu). The concepts used in COUSINE are
patent pending.
