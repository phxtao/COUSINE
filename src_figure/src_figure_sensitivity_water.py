# COUSINE (COUpled Soil Inorganic-organic carbon model for eNhanced wEathering)
# Copyright (c) 2026 Cornell University. All Rights Reserved. Cornell docket D-11800.
# Permission to use for educational, research and non-profit purposes only; see LICENSE for the terms,
# the commercial-licensing contact and the disclaimers. The concepts used in this work are patent pending.

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src_cousine'))
from fun_site_inputs import cousine_data_path
DATAHUB = cousine_data_path() # site-only data root (COUSINE_DATA_PATH, NTU server folder, or Mac folder)
import numpy as np
from scipy.interpolate import pchip_interpolate, griddata

from scipy.io import loadmat, savemat

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures') + '/' # the repository's figures folder
os.makedirs(fig_dir, exist_ok = True)

# leach_list = [15.0]
forcing_cycle_num = 5
year_num = 20
month_num = 12
dust_start_year = 40

dust_type = 'olivine'
pco2_list = [0.5, 0.8, 1, 1.5, 2]  
leach_list = [0.2, 0.4, 1, 1.5, 2]
langmuir_list = [1]
forcing_loc_list = ['temperate', 'subtropical', 'tropical']
soil_type = ['alfisol', 'ultisol', 'oxisol']

species_list = ['Al+++', 'Al(OH)++', 'Al(OH)4-', 'H4SiO4', 'H3SiO4-', 'M+', 'M++', 'OH-', 'H+', 'CO2(aq)', 'CO2(g)', 'HCO3-', 'CO3--', 'H2O', 'MCO3', 'Al(OH)3', 'SM', 'SH2', 'SOA-', 'SOAH']
species_label = ['$Al^{3+}$', '$Al(OH)^{2+}$', '$Al(OH)_{4}^{-}$', '$H_{4}SiO_{4}$', '$H_{3}SiO_{4}^{-}$', '$M^{+}$', '$M^{2+}$', '$OH^{-}$', '$H^{+}$', '$CO_{2}(aq)$', '$CO_{2}(g)$', '$HCO_{3}^{-}$', '$CO_{3}^{2-}$', '$H_{2}O$', '$MCO_{3}$', '$Al(OH)_{3}$', '$SM$', '$SH_{2}$', '$SOA^{-}$', 'SOAH']

color_scheme = ['#e41a1c','#377eb8','#4daf4a']

cdr_efficiency_summary = np.zeros((forcing_cycle_num*year_num, len(soil_type), len(leach_list), len(langmuir_list), len(pco2_list)))*np.nan
cdr_accu_net_summary = np.zeros((forcing_cycle_num*year_num, len(soil_type), len(leach_list), len(langmuir_list), len(pco2_list)))*np.nan
dust_weather_summary = np.zeros((forcing_cycle_num*year_num, len(soil_type), len(leach_list), len(langmuir_list), len(pco2_list)))*np.nan

data_dir = DATAHUB + 'cousine_output/simu/prior_v0_pc/annual_mean/'

for isoil in range(len(soil_type)):
    iforcing = isoil
    forcing_loc = forcing_loc_list[iforcing]
    for ileach in np.arange(len(leach_list)):
        leach_scalar = leach_list[ileach] # 10.0
        for ilangmuir in range(len(langmuir_list)):
            for ipco2 in range(len(pco2_list)):
                # matched reference: same soil/forcing/scalar combo, but dust_amount = 0
                transit_formal_parent = torch.load(f'{data_dir}transit_formal_soil_{soil_type[isoil]}_{forcing_loc}_nitrogen_{0.0}_{dust_type}_{0.0}_pco2_{pco2_list[ipco2]}_leach_{leach_list[ileach]}_langmuir_{langmuir_list[ilangmuir]}.pth')
                transit_formal = torch.load(f'{data_dir}transit_formal_soil_{soil_type[isoil]}_{forcing_loc}_nitrogen_{0.0}_{dust_type}_{10.0}_pco2_{pco2_list[ipco2]}_leach_{leach_list[ileach]}_langmuir_{langmuir_list[ilangmuir]}.pth')
                
                stoichio_element_parent = transit_formal['stoichio_element_parent'].detach().numpy() 
                stoichio_element_dust = transit_formal['stoichio_element_dust'].detach().numpy()

                molar_weight_element    =     torch.tensor([26.98, 28.09, 16.00,    1.0079, 12.01, 40.08, 24.31, 39.10, 22.99])
                # stoichio_element_dust        =     torch.tensor([0.01,  0.01,  torch.nan, 0.0,    0.0,   1.00,  0.0,   0.01,  0.0  ]) # external application
                stoichio_element_olivine =    torch.tensor([0.00,  1.00,  4.00,     0.0,    0.0,   0.0,   2.00,  0.00,  0.0  ]) # external application
                olivine_molar_weight = (molar_weight_element*stoichio_element_olivine).sum() # g mol-1

                # stoichiometry of dust (M+)_x (M++)_y Al_z Si_w O_...
                stoichio_carbon = stoichio_element_dust[4]
                stoichio_m_1cat = stoichio_element_dust[[7, 8]].sum() # x
                stoichio_m_2cat = stoichio_element_dust[[5, 6]].sum() # y
                stoichio_al_3cat = stoichio_element_dust[0] # z
                stoichio_silicon = stoichio_element_dust[1] # w
                stoichio_hydrogen = stoichio_element_dust[3] 
                stoichio_element_dust[2] = (stoichio_m_1cat + 1*stoichio_hydrogen + 2*stoichio_m_2cat + 3*stoichio_al_3cat + 4*stoichio_silicon + 4*stoichio_carbon)/2 # 
                stoichio_oxygen = stoichio_element_dust[2]
                # (stoichio_m_1cat + 2*stoichio_m_2cat + 3*stoichio_al_3cat + 4*stoichio_silicon)/2 - stoichio_oxygen
                    
                # stoichiometry of parent (M+)_x (M++)_y Al_z Si_w O_...
                stoichio_carbon_parent = stoichio_element_parent[4]
                stoichio_m_1cat_parent = stoichio_element_parent[[7, 8]].sum() # x
                stoichio_m_2cat_parent = stoichio_element_parent[[5, 6]].sum() # y
                stoichio_al_3cat_parent = stoichio_element_parent[0] # z
                stoichio_silicon_parent = stoichio_element_parent[1] # w
                stoichio_hydrogen_parent = stoichio_element_parent[3] 
                stoichio_element_parent[2] = (stoichio_m_1cat_parent + 1*stoichio_hydrogen_parent + 2*stoichio_m_2cat_parent + 3*stoichio_al_3cat_parent + 4*stoichio_silicon_parent + 4*stoichio_carbon_parent)/2 # 
                stoichio_oxygen_parent = stoichio_element_parent[2]

                # parent molar weight
                parent_molar_weight = (molar_weight_element*stoichio_element_parent).sum().detach().numpy() # g mol-1
                # dust molar weight
                dust_molar_weight = (molar_weight_element*stoichio_element_dust).sum().detach().numpy() # g mol-1

                # parent simu
                species_stock_parent = transit_formal_parent['species_stock'].detach().numpy()
                leached_ion_stock_parent = transit_formal_parent['leached_ion_stock'].detach().numpy()
                leached_species_stock_parent = transit_formal_parent['leached_species_stock'].detach().numpy()
                parent_species_stock_parent = transit_formal_parent['parent_species_stock'].detach().numpy()
                species_stock_initial_parent = transit_formal_parent['species_stock_initial'].detach().numpy()
                co2_limitation_parent = transit_formal_parent['co2_limitation'].detach().numpy()
                dust_consumed_base_parent = transit_formal_parent['dust_consumed_parent'].detach().numpy() 
                dust_remained_parent = transit_formal_parent['dust_remained'].detach().numpy()
                dust_concentration_initial_parent = transit_formal_parent['dust_concentration_initial'].detach().numpy()
                soc_stock_parent = transit_formal_parent['soc_stock'].detach().numpy()
                delta_species_stock_parent = transit_formal_parent['delta_species_stock'].detach().numpy()
                leached_ion_stock_parent = transit_formal_parent['leached_ion_stock'].detach().numpy()

                # dust simu onetime
                species_stock = transit_formal['species_stock'].detach().numpy()
                leached_ion_stock = transit_formal['leached_ion_stock'].detach().numpy()
                leached_species_stock = transit_formal['leached_species_stock'].detach().numpy()
                parent_species_stock = transit_formal['parent_species_stock'].detach().numpy()
                species_stock_initial = transit_formal['species_stock_initial'].detach().numpy()
                co2_limitation = transit_formal['co2_limitation'].detach().numpy()
                dust_consumed = transit_formal['dust_consumed'].detach().numpy()
                dust_remained = transit_formal['dust_remained'].detach().numpy()
                dust_consumed_base_dust = transit_formal['dust_consumed_parent'].detach().numpy() 
                dust_concentration_initial = transit_formal['dust_concentration_initial'].detach().numpy()
                soc_stock = transit_formal['soc_stock'].detach().numpy()
                delta_species_stock =  transit_formal['delta_species_stock'].detach().numpy()
                leached_ion_stock = transit_formal['leached_ion_stock'].detach().numpy()
                
                parent_weathering_depress_scalar = np.nansum(dust_consumed_base_dust, axis = (0, 1)).cumsum(axis = 0)/np.nansum(dust_consumed_base_parent, axis = (0, 1)).cumsum(axis = 0)

                # cdr calculation
                carbonate_accu_parent = (-delta_species_stock_parent)[0:6, (11, 12, 14), :, :].sum(axis = (0, 1, 2)).cumsum() \
                                        + (leached_ion_stock_parent)[(11, 12, 14), :, :].sum(axis = (0, 1)).cumsum() \
                                        - np.nansum(dust_consumed_base_parent, axis = (0, 1)).cumsum(axis = 0)*stoichio_carbon_parent 	  # minus carbon introduced by parent and dust

                carbonate_accu_tot = (-delta_species_stock)[0:6, (11, 12, 14), :, :].sum(axis = (0, 1, 2)).cumsum() + \
                                        +  (leached_ion_stock)[(11, 12, 14), :, :].sum(axis = (0, 1)).cumsum() \
                                        - np.nansum(dust_consumed, axis = (0, 1)).cumsum(axis = 0)*stoichio_carbon \
                                        - np.nansum(dust_consumed_base_dust, axis = (0, 1)).cumsum(axis = 0)*stoichio_carbon_parent 	  # minus carbon introduced by parent and dust

                net_carbonate_accu = carbonate_accu_tot - carbonate_accu_parent

                cdr_accu_site_parent = carbonate_accu_parent*44*10**4/10**6 # from mol HCO3 m-2 yr-1 to t CO2 ha-1 yr-1
                cdr_accu_site_net = net_carbonate_accu*44*10**4/10**6
                
                accu_dust_release = np.nansum(dust_consumed, axis = (0, 1)).cumsum()*dust_molar_weight/10**6*10**4 # mol m-2 to t ha-1

                net_carbonate_accu = carbonate_accu_tot - carbonate_accu_parent
                cdr_accu_site_net = net_carbonate_accu*44*10**4/10**6
                
                cdr_accu_net_summary[:, isoil, ileach, ilangmuir, ipco2] = cdr_accu_site_net
                cdr_efficiency_summary[:, isoil, ileach, ilangmuir, ipco2] = (net_carbonate_accu/(stoichio_m_2cat*accu_dust_release))

                dust_weather_summary[:, isoil, ileach, ilangmuir, ipco2] = accu_dust_release
            
post_dust_year_num = 40
time_slice = [20, 40, 60]

################################################
# matrix simulation figures
################################################
panel_letters = [['(a)', '(b)', '(c)'],
['(d)', '(e)', '(f)'],
['(g)', '(h)', '(i)']]

forcing_label = ['Temperate', 'Subtropical', 'Tropical']

from mpl_toolkits.axes_grid1 import make_axes_locatable

cmap_list = ['PuBu', 'BuGn', 'OrRd']
time_point = 10

fig, axes_all = plt.subplots(3, 3, figsize=(20, 16), layout='tight')

data_sources = [dust_weather_summary, cdr_accu_net_summary, cdr_efficiency_summary]
cbar_labels = ['Cumulative weathered dust ($t \ ha^{-1}$)', 'Cumulative CDR ($t \ CO_{2} \ ha^{-1}$)', 'CDR efficiency']
row_labels = ['Weathering', 'Net CDR', 'CDR Efficiency']

leach_scalar = np.asarray(leach_list)
pco2_scalar = np.asarray(pco2_list)

for irow in range(3):
    for isoil in range(len(soil_type)):
        axes = axes_all[irow, isoil]
        axes.spines[['top', 'bottom', 'left', 'right']].set_linewidth(1.5)
        axes.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        current_data = data_sources[irow][(post_dust_year_num + time_point - 1), isoil, :, 0, :]
        Y_origin, X_origin = np.meshgrid(pco2_scalar, leach_scalar)
        X_interp, Y_interp = np.meshgrid(np.linspace(0.2, 2, 200), np.linspace(0.5, 1.5, 200))
        Z_interp = griddata((X_origin.flatten(), Y_origin.flatten()), current_data.flatten(),
                             (X_interp, Y_interp), method='cubic')
        Z_interp[Z_interp < 0] = 0
        Z_interp = np.nan_to_num(Z_interp, nan=0.0)
        Z_min = Z_interp.min()
        Z_max = Z_interp.max()
        contour = axes.contourf(X_interp, Y_interp, Z_interp, cmap=cmap_list[isoil], vmin=Z_min, vmax=Z_max)

        # X label only on bottom row
        if irow == 2 and isoil == 1:
            axes.set_xlabel('Scalar to $Rain_{0}$', fontsize=25, labelpad=15)
        # Y label only on left column
        if isoil == 0:
            axes.set_ylabel(r'Scalar to $\tau_{CO_{2}}$', fontsize=25, labelpad=15)

        # Colorbar for each panel
        divider = make_axes_locatable(axes)
        cbar_ax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(contour, cax=cbar_ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=20)
        if isoil == 2:
            cbar.set_label(cbar_labels[irow], fontsize=20)

        axes.tick_params(labelsize=20, width=1.5, length=7)

        # Title only on top row
        if irow == 0:
            axes.set_title(str.capitalize(soil_type[isoil]), fontsize=35, fontweight='bold')

        # Row label on the left
        if isoil == 0:
            axes.annotate(row_labels[irow], xy=(-0.35, 0.5), xycoords='axes fraction',
                          fontsize=30, fontweight='bold', ha='center', va='center', rotation=90)

        axes.text(-0.15, 1.1, panel_letters[irow][isoil], transform=axes.transAxes,
                  fontsize=30, fontweight='bold', va='top', ha='left')

plt.savefig(fig_dir + 'cdr_weathering_para_sensitivity.jpeg')
plt.savefig(fig_dir + 'cdr_weathering_para_sensitivity.pdf')
plt.close()


print('end of programme')



