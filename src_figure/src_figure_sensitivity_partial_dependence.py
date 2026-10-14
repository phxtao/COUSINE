# COUSINE (COUpled Soil Inorganic-organic carbon model for eNhanced wEathering)
# Copyright (c) 2026 Cornell University. All Rights Reserved. Cornell docket D-11800.
# Permission to use for educational, research and non-profit purposes only; see LICENSE for the terms,
# the commercial-licensing contact and the disclaimers. The concepts used in this work are patent pending.

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src_cousine'))
from fun_site_inputs import cousine_data_path
DATAHUB = cousine_data_path() # site-only data root (COUSINE_DATA_PATH, NTU server folder, or Mac folder)
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
import torch

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures') + '/' # the repository's figures folder
os.makedirs(fig_dir, exist_ok = True)
data_dir = DATAHUB + 'cousine_output/simu/prior_v0_pc/annual_mean/'

forcing_cycle_num = 5
year_num = 10
month_num = 12
dust_start_year = 40
post_dust_year_num = 40

dust_type = 'olivine'
forcing_loc_list = ['temperate', 'subtropical', 'tropical']
soil_type = ['alfisol', 'ultisol', 'oxisol']

time_idx_year = post_dust_year_num + year_num - 1  # 10 years after dust application

# one-at-a-time (partial dependence) sweeps: every entry varies a single
# parameter across `values` while every other parameter is held at its
# baseline (scalar = 1); `fixed` gives the baseline scalar of the co-varied
# parameters needed to build the file name of each simulation
param_list = [
    {'name': 'active_energy', 'group': 'kinetics', 'values': [0.5, 0.8, 1, 1.2, 1.5],
     'label': r'Scalar to $E_{a}$',
     'fname': lambda soil, loc, v: f'{data_dir}transit_formal_soil_{soil}_{loc}_nitrogen_0.0_{dust_type}_10.0_energy_{v}_order_1.pth'},
    {'name': 'reaction_order', 'group': 'kinetics', 'values': [0.5, 0.8, 1, 1.2, 1.5],
     'label': r'Scalar to $n$',
     'fname': lambda soil, loc, v: f'{data_dir}transit_formal_soil_{soil}_{loc}_nitrogen_0.0_{dust_type}_10.0_energy_1_order_{v}.pth'},
    {'name': 'pco2', 'group': 'water', 'values': [0.5, 0.8, 1, 1.5, 2],
     'label': r'Scalar to $\tau_{CO_{2}}$',
     'fname': lambda soil, loc, v: f'{data_dir}transit_formal_soil_{soil}_{loc}_nitrogen_0.0_{dust_type}_10.0_pco2_{v}_leach_1_langmuir_1.pth',
     'parent_fname': lambda soil, loc, v: f'{data_dir}transit_formal_soil_{soil}_{loc}_nitrogen_0.0_{dust_type}_0.0_pco2_{v}_leach_1_langmuir_1.pth'},
    {'name': 'leach', 'group': 'water', 'values': [0.2, 0.4, 1, 1.5, 2],
     'label': r'Scalar to $Rain_{0}$',
     'fname': lambda soil, loc, v: f'{data_dir}transit_formal_soil_{soil}_{loc}_nitrogen_0.0_{dust_type}_10.0_pco2_1_leach_{v}_langmuir_1.pth',
     'parent_fname': lambda soil, loc, v: f'{data_dir}transit_formal_soil_{soil}_{loc}_nitrogen_0.0_{dust_type}_0.0_pco2_1_leach_{v}_langmuir_1.pth'},
    {'name': 'solute_leach', 'group': 'others', 'values': [0.5, 0.8, 1, 1.2, 1.5],
     'label': r'Scalar to $\tau_{leach}$',
     'fname': lambda soil, loc, v: f'{data_dir}transit_formal_soil_{soil}_{loc}_nitrogen_0.0_{dust_type}_10.0_solute_leach_{v}_soa_ratio_1_ex_ratio_1.pth',
     'parent_fname': lambda soil, loc, v: f'{data_dir}transit_formal_soil_{soil}_{loc}_nitrogen_0.0_{dust_type}_0.0_solute_leach_{v}_soa_ratio_1_ex_ratio_1.pth'},
    {'name': 'soa_ratio', 'group': 'others', 'values': [0.5, 0.8, 1, 1.2, 1.5],
     'label': 'Scalar to SOA ratio',
     'fname': lambda soil, loc, v: f'{data_dir}transit_formal_soil_{soil}_{loc}_nitrogen_0.0_{dust_type}_10.0_solute_leach_1_soa_ratio_{v}_ex_ratio_1.pth',
     'parent_fname': lambda soil, loc, v: f'{data_dir}transit_formal_soil_{soil}_{loc}_nitrogen_0.0_{dust_type}_0.0_solute_leach_1_soa_ratio_{v}_ex_ratio_1.pth'},
    {'name': 'ex_ratio', 'group': 'others', 'values': [0.5, 0.8, 1, 1.2, 1.5],
     'label': 'Scalar to exchange ratio',
     'fname': lambda soil, loc, v: f'{data_dir}transit_formal_soil_{soil}_{loc}_nitrogen_0.0_{dust_type}_10.0_solute_leach_1_soa_ratio_1_ex_ratio_{v}.pth',
     'parent_fname': lambda soil, loc, v: f'{data_dir}transit_formal_soil_{soil}_{loc}_nitrogen_0.0_{dust_type}_0.0_solute_leach_1_soa_ratio_1_ex_ratio_{v}.pth'},
]

# Paul Tol bright/vibrant qualitative palette: 7 distinct, colorblind-friendly
# hues (one per parameter). Chosen over Okabe-Ito to avoid its pale yellow
# (#F0E442, low contrast on white) and to keep grey free for the baseline line
# color_scheme = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#AA3377', '#66CCEE', '#EE7733']
color_scheme = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628']

def fun_cdr_series(transit_formal_parent, transit_formal):
    stoichio_element_parent = transit_formal_parent['stoichio_element_parent'].detach().numpy()
    stoichio_element_dust = transit_formal['stoichio_element_dust'].detach().numpy()

    molar_weight_element = torch.tensor([26.98, 28.09, 16.00, 1.0079, 12.01, 40.08, 24.31, 39.10, 22.99])

    stoichio_carbon = stoichio_element_dust[4]
    stoichio_m_1cat = stoichio_element_dust[[7, 8]].sum()
    stoichio_m_2cat = stoichio_element_dust[[5, 6]].sum()
    stoichio_al_3cat = stoichio_element_dust[0]
    stoichio_silicon = stoichio_element_dust[1]
    stoichio_hydrogen = stoichio_element_dust[3]
    stoichio_element_dust[2] = (stoichio_m_1cat + 1 * stoichio_hydrogen + 2 * stoichio_m_2cat + 3 * stoichio_al_3cat + 4 * stoichio_silicon + 4 * stoichio_carbon) / 2

    stoichio_carbon_parent = stoichio_element_parent[4]
    stoichio_m_1cat_parent = stoichio_element_parent[[7, 8]].sum()
    stoichio_m_2cat_parent = stoichio_element_parent[[5, 6]].sum()
    stoichio_al_3cat_parent = stoichio_element_parent[0]
    stoichio_silicon_parent = stoichio_element_parent[1]
    stoichio_hydrogen_parent = stoichio_element_parent[3]
    stoichio_element_parent[2] = (stoichio_m_1cat_parent + 1 * stoichio_hydrogen_parent + 2 * stoichio_m_2cat_parent + 3 * stoichio_al_3cat_parent + 4 * stoichio_silicon_parent + 4 * stoichio_carbon_parent) / 2

    dust_molar_weight = (molar_weight_element * stoichio_element_dust).sum().detach().numpy()

    delta_species_stock_parent = transit_formal_parent['delta_species_stock'].detach().numpy()
    leached_ion_stock_parent = transit_formal_parent['leached_ion_stock'].detach().numpy()
    dust_consumed_base_parent = transit_formal_parent['dust_consumed_parent'].detach().numpy()

    delta_species_stock = transit_formal['delta_species_stock'].detach().numpy()
    leached_ion_stock = transit_formal['leached_ion_stock'].detach().numpy()
    dust_consumed = transit_formal['dust_consumed'].detach().numpy()
    dust_consumed_base_dust = transit_formal['dust_consumed_parent'].detach().numpy()

    carbonate_accu_parent = (-delta_species_stock_parent)[0:6, (11, 12, 14), :, :].sum(axis=(0, 1, 2)).cumsum() \
                            + (leached_ion_stock_parent)[(11, 12, 14), :, :].sum(axis=(0, 1)).cumsum() \
                            - np.nansum(dust_consumed_base_parent, axis=(0, 1)).cumsum(axis=0) * stoichio_carbon_parent

    carbonate_accu_tot = (-delta_species_stock)[0:6, (11, 12, 14), :, :].sum(axis=(0, 1, 2)).cumsum() \
                         + (leached_ion_stock)[(11, 12, 14), :, :].sum(axis=(0, 1)).cumsum() \
                         - np.nansum(dust_consumed, axis=(0, 1)).cumsum(axis=0) * stoichio_carbon \
                         - np.nansum(dust_consumed_base_dust, axis=(0, 1)).cumsum(axis=0) * stoichio_carbon_parent

    net_carbonate_accu = carbonate_accu_tot - carbonate_accu_parent

    cdr_accu_net = net_carbonate_accu * 44 * 10 ** 4 / 10 ** 6  # t CO2 ha-1
    accu_dust_release = np.nansum(dust_consumed, axis=(0, 1)).cumsum() * dust_molar_weight / 10 ** 6 * 10 ** 4  # t ha-1
    cdr_efficiency = net_carbonate_accu / (stoichio_m_2cat * accu_dust_release)

    return accu_dust_release, cdr_accu_net, cdr_efficiency


################################################
# load simulations and extract year-(dust_start+time_idx) values
################################################
metrics = ['weathering', 'cdr', 'efficiency']
# results[param_name][metric][isoil] -> list aligned with param_list[i]['values']
results = {
    p['name']: {m: [[] for _ in soil_type] for m in metrics}
    for p in param_list
}

for isoil in range(len(soil_type)):
    soil = soil_type[isoil]
    forcing_loc = forcing_loc_list[isoil]

    # generic reference (default scalars = 1): used for params without a matched dust=0 reference
    transit_formal_parent_generic = torch.load(
        f'{data_dir}transit_formal_soil_{soil}_{forcing_loc}_nitrogen_{0}_{dust_type}_{0.0}.pth')

    for param in param_list:
        for value in param['values']:
            if 'parent_fname' in param:
                # matched reference: same soil/forcing/scalar combo, but dust_amount = 0
                transit_formal_parent = torch.load(param['parent_fname'](soil, forcing_loc, value))
            else:
                transit_formal_parent = transit_formal_parent_generic

            transit_formal = torch.load(param['fname'](soil, forcing_loc, value))

            accu_dust_release, cdr_accu_net, cdr_efficiency = fun_cdr_series(transit_formal_parent, transit_formal)

            results[param['name']]['weathering'][isoil].append(accu_dust_release[time_idx_year])
            results[param['name']]['cdr'][isoil].append(cdr_accu_net[time_idx_year])
            results[param['name']]['efficiency'][isoil].append(cdr_efficiency[time_idx_year])

################################################
# figure: rows = metric, columns = soil type; color = parameter
################################################
panel_letters = [['(a)', '(b)', '(c)'],
                  ['(d)', '(e)', '(f)'],
                  ['(g)', '(h)', '(i)']]
row_labels = ['Weathering', 'Net CDR', 'CDR Efficiency']
y_labels = ['Cumulative weathered dust\n($t \\ ha^{-1}$)', 'Cumulative net CDR\n($t \\ CO_{2} \\ ha^{-1}$)', 'CDR efficiency']
fig, axes_all = plt.subplots(3, 3, figsize=(24, 20), layout='tight')

# shared y-axis range per metric (row), computed across all soils and parameters
row_ylim = {}
for metric in metrics:
    all_values = [v for param in param_list for isoil in range(len(soil_type))
                  for v in results[param['name']][metric][isoil]]
    y_min, y_max = min(all_values), max(all_values)
    margin = 0.05 * (y_max - y_min)
    row_ylim[metric] = (y_min - margin, y_max + margin)

for irow, metric in enumerate(metrics):
    for isoil in range(len(soil_type)):
        axes = axes_all[irow, isoil]
        axes.spines[['top', 'bottom', 'left', 'right']].set_linewidth(1.5)
        axes.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axes.set_ylim(row_ylim[metric])

        for iparam, param in enumerate(param_list):
            axes.plot(param['values'], results[param['name']][metric][isoil],
                      marker='o', markersize=10, linewidth=4,
                      color=color_scheme[iparam], alpha=0.85)

        # X label only on bottom row
        if irow == 2 and isoil == 1:
            axes.set_xlabel('Parameter scalar (relative to baseline)', fontsize=30, labelpad=15)
        # Y label only on left column
        if isoil == 0:
            axes.set_ylabel(y_labels[irow], fontsize=30, labelpad=15)

        axes.spines[['top', 'right']].set_visible(False)
        axes.spines[['bottom', 'left']].set_linewidth(2.0)
        axes.tick_params(labelsize=26, width=3.0, length=12)

        # Title only on top row
        if irow == 0:
            axes.set_title(str.capitalize(soil_type[isoil]), fontsize=32, fontweight='bold', pad=15)

        # Row label on the left
        if isoil == 0:
            axes.annotate(row_labels[irow], xy=(-0.35, 0.5), xycoords='axes fraction',
                          fontsize=28, fontweight='bold', ha='center', va='center', rotation=90)

        axes.text(-0.15, 1.12, panel_letters[irow][isoil], transform=axes.transAxes,
                  fontsize=30, fontweight='bold', va='top', ha='left')

# shared legend: color -> parameter
param_legend_handles = [Line2D([0], [0], color=color_scheme[i], linewidth=4, label=param_list[i]['label'])
                         for i in range(len(param_list))]
fig.legend(handles=param_legend_handles, title='', fontsize=25, title_fontsize=30,
           frameon=False, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=4)

plt.savefig(fig_dir + 'cdr_weathering_partial_dependence.jpeg', bbox_inches='tight')
plt.savefig(fig_dir + 'cdr_weathering_partial_dependence.pdf', bbox_inches='tight')
plt.close()

print('end of programme')
