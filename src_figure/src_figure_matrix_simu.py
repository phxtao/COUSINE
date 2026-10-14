# COUSINE (COUpled Soil Inorganic-organic carbon model for eNhanced wEathering)
# Copyright (c) 2026 Cornell University. All Rights Reserved. Cornell docket D-11800.
# Permission to use for educational, research and non-profit purposes only; see LICENSE for the terms,
# the commercial-licensing contact and the disclaimers. The concepts used in this work are patent pending.

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src_cousine'))
from fun_site_inputs import cousine_data_path
DATAHUB = cousine_data_path() # site-only data root (COUSINE_DATA_PATH, NTU server folder, or Mac folder)
import sys                                                                                                            
import time
import warnings

from datetime import datetime
from pandas import DataFrame as df
import numpy as np
from scipy.interpolate import pchip_interpolate, griddata

import os

import scipy
from scipy.io import loadmat, savemat
import netCDF4 as ncread
import mat73

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

import torch

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures') + '/' # the repository's figures folder
os.makedirs(fig_dir, exist_ok = True)

# dust_amount_list = [15.0]
forcing_cycle_num = 5
year_num = 20
month_num = 12
dust_start_year = 40

dust_type = 'olivine'
dust_amount_list = [2.5, 5.0, 10.0, 15.0]
nfertilizer_list = [0, 4, 8, 20]
forcing_loc_list = ['temperate', 'subtropical', 'tropical']
soil_type = ['alfisol', 'ultisol', 'oxisol']

species_list = ['Al+++', 'Al(OH)++', 'Al(OH)4-', 'H4SiO4', 'H3SiO4-', 'M+', 'M++', 'OH-', 'H+', 'CO2(aq)', 'CO2(g)', 'HCO3-', 'CO3--', 'H2O', 'MCO3', 'Al(OH)3', 'SM', 'SH2', 'SOA-', 'SOAH']
species_label = ['$Al^{3+}$', '$Al(OH)^{2+}$', '$Al(OH)_{4}^{-}$', '$H_{4}SiO_{4}$', '$H_{3}SiO_{4}^{-}$', '$M^{+}$', '$M^{2+}$', '$OH^{-}$', '$H^{+}$', '$CO_{2}(aq)$', '$CO_{2}(g)$', '$HCO_{3}^{-}$', '$CO_{3}^{2-}$', '$H_{2}O$', '$MCO_{3}$', '$Al(OH)_{3}$', '$SM$', '$SH_{2}$', '$SOA^{-}$', 'SOAH']

color_scheme = ['#e41a1c','#377eb8','#4daf4a']

cdr_efficiency_summary = np.zeros((forcing_cycle_num*year_num, len(soil_type), len(forcing_loc_list), len(dust_amount_list), len(nfertilizer_list)))*np.nan
cdr_accu_net_summary = np.zeros((forcing_cycle_num*year_num, len(soil_type), len(forcing_loc_list), len(dust_amount_list), len(nfertilizer_list)))*np.nan
dust_weather_summary = np.zeros((forcing_cycle_num*year_num, len(soil_type), len(forcing_loc_list), len(dust_amount_list), len(nfertilizer_list)))*np.nan

for idust_amount in np.arange(len(dust_amount_list)):
    dust_amount = dust_amount_list[idust_amount] # 10.0

    for isoil in range(len(soil_type)):
        for iforcing in range(len(forcing_loc_list)):
            forcing_loc = forcing_loc_list[iforcing]
            data_dir = DATAHUB + 'cousine_output/simu/prior_v0_pc/annual_mean/'
            for initrogen in range(len(nfertilizer_list)):
                transit_formal_parent = torch.load(f'{data_dir}transit_formal_soil_{soil_type[isoil]}_{forcing_loc}_nitrogen_{0}_{dust_type}_{0.0}.pth')
                transit_formal = torch.load(f'{data_dir}transit_formal_soil_{soil_type[isoil]}_{forcing_loc}_nitrogen_{nfertilizer_list[initrogen]}_{dust_type}_{dust_amount}.pth')
                
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
                
                accu_dust_release = np.nansum(dust_consumed, axis = (0, 1)).cumsum()

                net_carbonate_accu = carbonate_accu_tot - carbonate_accu_parent
                cdr_accu_site_net = net_carbonate_accu*44*10**4/10**6
               
                cdr_accu_net_summary[:, isoil, iforcing, idust_amount, initrogen] = cdr_accu_site_net
                cdr_efficiency_summary[:, isoil, iforcing, idust_amount, initrogen] = (net_carbonate_accu/(stoichio_m_2cat*accu_dust_release))

                dust_weather_summary[:, isoil, iforcing, idust_amount, initrogen] = accu_dust_release
            
post_dust_year_num = 40
time_slice = [20, 40, 60]

################################################
# matrix simulation figures
################################################
panel_letters = [['(a)', '(b)', '(c)'],
['(d)', '(e)', '(f)'],
['(g)', '(h)', '(i)']]

forcing_label = ['Temperate', 'Subtropical', 'Tropical']

#-----------------------------------------------
# CDR efficiency matrix
#-----------------------------------------------

fig = plt.figure(figsize = (40, 35))
# norm = Normalize(vmin=0, vmax=2)
for iforcing in range(len(forcing_loc_list)):
    forcing_loc = forcing_loc_list[iforcing]
    fig_row = iforcing
    for isoil in range(len(soil_type)):
        # fig = plt.subplot(1, 3, (isoil+1))
        axes = fig.add_subplot(3, 3, fig_row*len(soil_type)+(isoil+1), projection='3d')
        axes.spines[['top','bottom','left','right']].set_linewidth(3)
        axes.patch.set_alpha(0) 
        dust_application = np.asarray(dust_amount_list)
        nitrogen_application = np.asarray(nfertilizer_list)

        for itime in range(len(time_slice)):
            current_data = cdr_efficiency_summary[(post_dust_year_num+time_slice[itime]-1), isoil, iforcing, :, :]        
            X_origin, Y_origin = np.meshgrid(nitrogen_application, dust_application)
            X_interp, Y_interp = np.meshgrid(np.linspace(0, 20, 200), np.linspace(2.5, 15, 200))
            # Interpolate data
            Z_interp = griddata((X_origin.flatten(), Y_origin.flatten()), current_data.flatten(), (X_interp, Y_interp), method = 'cubic')
            Z_interp[Z_interp < 0] = 0
            # axes.plot_surface(X_interp, Y_interp, Z_interp, rstride = 15, cstride = 15, facecolor = color_scheme[isoil], edgecolor = color_scheme[isoil], lw = 1, alpha = 0.1)
            contour = axes.contourf(X_interp, Y_interp, Z_interp, zdir = 'z', offset = time_slice[itime], cmap = 'coolwarm', vmin = 0, vmax = 1.4)
            axes.set_zlim(20, 60)
            axes.view_init(elev = 30, azim = 45)     
        # contour.set_clim(0, 2)
        if isoil == 2:
            axes.set_xlabel('Net nitrate leaching ($kg \ N \ ha^{-1} \ yr^{-1})$', fontsize = 35, labelpad = 20)
        if isoil == 0:
            axes.set_ylabel('Dust amount ($t \ ha^{-1})$', fontsize = 35, labelpad = 20)
            axes.set_zlabel('Year', fontsize = 35, labelpad = 30, rotation = 90)
        if isoil == 1 and iforcing == 2:
            cbar_ax = fig.add_axes([0.3, 0.08, 0.4, 0.02])  # [left, bottom, width, height]
            cbar = fig.colorbar(contour, cax = cbar_ax, orientation='horizontal')
            cbar.ax.tick_params(labelsize = 35)
            cbar.set_label('CDR efficiency', fontsize = 45)

        axes.tick_params(labelsize = 25, width = 1.5, length = 7)
        axes.tick_params(axis = 'z', pad = 15)
        if fig_row == 0:
            plt.title(str.capitalize(soil_type[isoil]), fontsize = 60, fontweight='bold')
        if isoil == 2:
            axes.text(-7, 10, 30, forcing_label[iforcing], fontsize = 60, fontweight='bold')
        axes.text(10, 5, 350, panel_letters[fig_row][isoil], transform = axes.transAxes, fontsize = 50, fontweight='bold', va='top', ha='left')
    plt.subplots_adjust(wspace = 0, hspace = 0.05)
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.savefig(fig_dir + 'cdr_efficiency_dust_scenarios_matrix.jpeg')
plt.savefig(fig_dir + 'cdr_efficiency_dust_scenarios_matrix.pdf')
plt.close()

#-----------------------------------------------
# CDR accumulation matrix
#-----------------------------------------------
fig = plt.figure(figsize = (40, 35))
# norm_accu = Normalize(vmin=0, vmax=15)

for iforcing in range(len(forcing_loc_list)):
    forcing_loc = forcing_loc_list[iforcing]
    fig_row = iforcing
    for isoil in range(len(soil_type)):
        # fig = plt.subplot(1, 3, (isoil+1))
        axes = fig.add_subplot(3, 3, fig_row*len(soil_type)+(isoil+1), projection='3d')
        axes.spines[['top','bottom','left','right']].set_linewidth(3)
        axes.patch.set_alpha(0) 
        dust_application = np.asarray(dust_amount_list)
        nitrogen_application = np.asarray(nfertilizer_list)

        for itime in range(len(time_slice)):
            current_data = cdr_accu_net_summary[(post_dust_year_num+time_slice[itime]-1), isoil, iforcing, :, :]        
            X_origin, Y_origin = np.meshgrid(nitrogen_application, dust_application)
            X_interp, Y_interp = np.meshgrid(np.linspace(0, 20, 200), np.linspace(2.5, 15, 200))
            # Interpolate data
            Z_interp = griddata((X_origin.flatten(), Y_origin.flatten()), current_data.flatten(), (X_interp, Y_interp), method = 'cubic')
            Z_interp[Z_interp < 0] = 0

            contour = axes.contourf(X_interp, Y_interp, Z_interp, zdir = 'z', offset = time_slice[itime], cmap = 'coolwarm', vmin = 0, vmax = 12)
            axes.set_zlim(20, 60)
            axes.view_init(elev = 30, azim = 45)     
        # contour.set_clim(0, 2)
        if isoil == 2:
            axes.set_xlabel('Net nitrate leaching ($kg \ N \ ha^{-1} \ yr^{-1})$', fontsize = 35, labelpad = 20)
        if isoil == 0:
            axes.set_ylabel('Dust amount ($t \ ha^{-1})$', fontsize = 35, labelpad = 20)
            axes.set_zlabel('Year', fontsize = 35, labelpad = 30, rotation = 90)
        if isoil == 1 and iforcing == 2:
            cbar_ax = fig.add_axes([0.3, 0.08, 0.4, 0.02])  # [left, bottom, width, height]
            cbar = fig.colorbar(contour, cax = cbar_ax, orientation='horizontal')
            cbar.ax.tick_params(labelsize = 35)
            cbar.set_label('Cumulative CDR ($t \ CO_{2} \ ha^{-1}$)', fontsize = 45)
        axes.tick_params(labelsize = 25, width = 1.5, length = 7)
        axes.tick_params(axis = 'z', pad = 15)
        if fig_row == 0:
            plt.title(str.capitalize(soil_type[isoil]), fontsize = 60, fontweight='bold')
        if isoil == 2:
            axes.text(-7, 10, 30, forcing_label[iforcing], fontsize = 60, fontweight='bold')
        axes.text(10, 5, 350, panel_letters[fig_row][isoil], transform = axes.transAxes, fontsize = 50, fontweight='bold', va='top', ha='left')
    plt.subplots_adjust(wspace = 0, hspace = 0.05)
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.savefig(fig_dir + 'cdr_accu_dust_scenarios_matrix.jpeg')
plt.savefig(fig_dir + 'cdr_accu_dust_scenarios_matrix.pdf')
plt.close()

print('end of programme')



