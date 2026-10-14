# COUSINE (COUpled Soil Inorganic-organic carbon model for eNhanced wEathering)
# Copyright (c) 2026 Cornell University. All Rights Reserved. Cornell docket D-11800.
# Permission to use for educational, research and non-profit purposes only; see LICENSE for the terms,
# the commercial-licensing contact and the disclaimers. The concepts used in this work are patent pending.

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src_cousine'))
from fun_site_inputs import cousine_data_path
DATAHUB = cousine_data_path() # site-only data root (COUSINE_DATA_PATH, NTU server folder, or Mac folder)
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures') + '/' # the repository's figures folder
os.makedirs(fig_dir, exist_ok = True)
data_path = DATAHUB
cesm2_case_name_hat = 'phxtao_f09_g17_hist_simu'
forcing_year_num = 20
soil_decom_num = 1

# Load land mask
from pandas import read_csv
site_grid = read_csv(data_path + 'clm5_ouput_ncar/summary_land_mask_sites.csv') # 1-degree CLM5 forcing grid of each site

site_info = {
    'Temperate': {'lon': -90, 'lat': 41},
    'Subtropical': {'lon': -85, 'lat': 31},
    'Tropical':  {'lon': -63.5, 'lat': -10},
}

color_scheme = ['#377eb8', '#4daf4a', '#e41a1c']
months = np.arange(1, forcing_year_num * 12 + 1)

fig, axes = plt.subplots(2, 1, figsize=(16, 12), layout='tight')

for i, (soil_name, info) in enumerate(site_info.items()):
    # Find location for 1 deg forcing
    forcing_site_loc_1deg = int(site_grid.loc[(site_grid.lon == info['lon']) & (site_grid.lat == info['lat']), 'loc_1deg'].iloc[0])

    soil_forcing = loadmat(data_path + 'clm5_ouput_ncar/grid_hist_input/cesm2_hist_input_'
                           + cesm2_case_name_hat + '_1901_2014_grid_' + str(forcing_site_loc_1deg) + '.mat')
    soil_forcing = soil_forcing['grid_hist_input']

    # Temperature: surface layer, all months, first 20 years -> Celsius
    temp = soil_forcing['cesm2_simu_soil_temperature_origin'][0][0][0, :, 0:forcing_year_num]  # (12, 20)
    temp_series = temp.flatten(order='F') - 273.15  # K to C

    # Rainfall: mm/s -> mm/day
    rain = soil_forcing['cesm2_simu_rain'][0][0][:, 0:forcing_year_num]  # (12, 20)
    rain_series = rain.flatten(order='F') * 24 * 3600

    # Panel (a): Temperature
    axes[0].plot(months, temp_series, linewidth=4, color=color_scheme[i], alpha=0.9, label=soil_name)

    # Panel (b): Rainfall
    axes[1].plot(months, rain_series, linewidth=4, color=color_scheme[i], alpha=0.9, label=soil_name)

# --- Panel (a): Temperature ---
ax = axes[0]
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['bottom', 'left']].set_linewidth(2.5)
ax.set_ylabel('Soil temperature (°C)', fontsize=30, fontweight='normal')
ax.tick_params(labelsize=25, width=2.5, length=7)
ax.legend(fontsize=25, frameon=False, loc='lower right', ncols=3)
ax.set_xlim([1, forcing_year_num * 12])
year_ticks = np.arange(0, forcing_year_num * 12 + 1, 12)
ax.set_xticks(year_ticks)
ax.set_xticklabels([])
ax.text(-0.08, 1.05, '(a)', transform=ax.transAxes, fontsize=28, fontweight='normal', va='top', ha='left')

# --- Panel (b): Rainfall ---
ax = axes[1]
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['bottom', 'left']].set_linewidth(2.5)
ax.set_xlabel('Year', fontsize=30, fontweight='normal')
ax.set_ylabel('Rainfall ($mm \ day^{-1}$)', fontsize=30, fontweight='normal')
ax.tick_params(labelsize=25, width=2.5, length=7)
ax.set_xlim([1, forcing_year_num * 12])
ax.set_xticks(year_ticks)
ax.set_xticklabels([f'{y}' for y in range(0, forcing_year_num + 1)], rotation=0)
ax.text(-0.08, 1.05, '(b)', transform=ax.transAxes, fontsize=28, fontweight='normal', va='top', ha='left')

plt.savefig(fig_dir + 'climate_forcing_time_series.jpeg', dpi=300, bbox_inches='tight')
plt.savefig(fig_dir + 'climate_forcing_time_series.pdf', bbox_inches='tight')
plt.close()