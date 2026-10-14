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
import os
import psutil
import random
import subprocess
import traceback

from pandas import DataFrame as df

from datetime import datetime
from pandas import read_csv # import DataFrame as df
import numpy as np
from scipy.interpolate import pchip_interpolate

import torch
from torch import nn
from torch.utils.data import random_split, DataLoader

import scipy
from scipy.io import loadmat, savemat
import netCDF4 as ncread
import mat73
import pygad
import multiprocessing
# from torch.utils.checkpoint import checkpoint

from multiprocessing import cpu_count
import tifffile
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures') + '/' # the repository's figures folder
os.makedirs(fig_dir, exist_ok = True)

# dust_amount_list = [15.0]
forcing_cycle_num = 5
year_num = 20
month_num = 12
dust_start_year = 40
transit_year_size = forcing_cycle_num*year_num

dust_amount_list = [0.0, 2.5, 5.0, 10.0, 15.0]
forcing_list = ['temperate', 'subtropical', 'tropical']
soil_type = ['alfisol', 'ultisol', 'oxisol']
dust_type_list = ['olivine', 'basalt', 'lime'] #  'olivine'

species_list = ['Al+++', 'Al(OH)++', 'Al(OH)4-', 'H4SiO4', 'H3SiO4-', 'M+', 'M++', 'OH-', 'H+', 'CO2(aq)', 'CO2(g)', 'HCO3-', 'CO3--', 'H2O', 'MCO3', 'Al(OH)3', 'SM', 'SH2', 'SOA-', 'SOAH']
species_label = ['$Al^{3+}$', '$Al(OH)^{2+}$', '$Al(OH)_{4}^{-}$', '$H_{4}SiO_{4}$', '$H_{3}SiO_{4}^{-}$', '$M^{+}$', '$M^{2+}$', '$OH^{-}$', '$H^{+}$', '$CO_{2}(aq)$', '$CO_{2}(g)$', '$HCO_{3}^{-}$', '$CO_{3}^{2-}$', '$H_{2}O$', '$MCO_{3}$', '$Al(OH)_{3}$', '$SM$', '$SH_{2}$', '$SOA^{-}$', 'SOAH']

color_scheme = ['#377eb8','#4daf4a','#e41a1c']

cdr_efficiency_summary = np.zeros((forcing_cycle_num*year_num, len(soil_type), len(dust_type_list), len(dust_amount_list)))*np.nan
cdr_accu_net_summary = np.zeros((forcing_cycle_num*year_num, len(soil_type), len(dust_type_list), len(dust_amount_list)))*np.nan
dust_weather_summary = np.zeros((forcing_cycle_num*year_num, len(soil_type), len(dust_type_list), len(dust_amount_list)))*np.nan
ph_summary = np.zeros((forcing_cycle_num*year_num, len(soil_type), len(dust_type_list), len(dust_amount_list)))*np.nan
dust_type_molar_weight = np.zeros(len(dust_type_list))

idust_amount = 3
dust_amount = dust_amount_list[idust_amount] # 10.0

for isoil in range(len(soil_type)):
    species_stock_parent = [None]*len(dust_type_list)
    leached_species_stock_parent = [None]*len(dust_type_list)
    leached_ion_stock_parent = [None]*len(dust_type_list)
    parent_species_stock_parent = [None]*len(dust_type_list)
    species_stock_initial_parent = [None]*len(dust_type_list)
    co2_limitation_parent = [None]*len(dust_type_list)
    dust_consumed_parent = [None]*len(dust_type_list)
    dust_remained_parent = [None]*len(dust_type_list)
    dust_concentration_initial_parent = [None]*len(dust_type_list)
    soc_stock_parent = [None]*len(dust_type_list)

    species_stock = [None]*len(dust_type_list)
    leached_species_stock = [None]*len(dust_type_list)
    leached_ion_stock = [None]*len(dust_type_list)
    parent_species_stock = [None]*len(dust_type_list)
    species_stock_initial = [None]*len(dust_type_list)
    co2_limitation = [None]*len(dust_type_list)
    dust_consumed = [None]*len(dust_type_list)
    dust_remained = [None]*len(dust_type_list)
    dust_concentration_initial = [None]*len(dust_type_list)
    soc_stock = [None]*len(dust_type_list)

    # mol m-2 yr-1 to Pg C yr-1
    co2_series = [None]*len(dust_type_list)
    co2_change_net = [None]*len(dust_type_list)

    accu_dust_release = [None]*len(dust_type_list)
    net_carbonate_accu = [None]*len(dust_type_list)
    cdr_accu_site_net = [None]*len(dust_type_list)


    for idust in range(len(dust_type_list)):
        forcing_loc = forcing_list[isoil]
        data_dir = DATAHUB + 'cousine_output/simu/prior_v0_pc/annual_mean/'
        transit_formal_parent = torch.load(f'{data_dir}transit_formal_soil_{soil_type[isoil]}_{forcing_loc}_nitrogen_{0}_olivine_{0.0}.pth')
        transit_formal = torch.load(f'{data_dir}transit_formal_soil_{soil_type[isoil]}_{forcing_loc}_nitrogen_{0}_{dust_type_list[idust]}_{dust_amount}.pth')
        
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

        cdr_accu_net_summary[:, isoil, idust, idust_amount] = cdr_accu_site_net
        cdr_efficiency_summary[:, isoil, idust, idust_amount] = (net_carbonate_accu/(stoichio_m_2cat*accu_dust_release))

        dust_weather_summary[:, isoil, idust, idust_amount] = np.nansum(dust_consumed, axis = (0, 1))
        
        proton_stock =  np.transpose(species_stock[8, :, :]).mean(axis = 1) # mol m-2
        water_stock = np.transpose(species_stock[13, :, :]).mean(axis = 1) # mol m-2
        ph_series = -np.log10(proton_stock/(water_stock*18/1000)) # mol L-1
        initial_ph = -np.log10(species_stock_initial[8]/(species_stock_initial[13]*18/1000))
        ph_summary[:, isoil, idust, idust_amount] = np.concatenate([[initial_ph], ph_series[0:transit_year_size-1]])
        dust_type_molar_weight[idust] = dust_molar_weight

################################################
# figures
################################################
panel_letters = [['(a)', '(b)', '(c)'],
        ['(d)', '(e)', '(f)'],
        ['(g)', '(h)', '(i)']]

color_scheme = ['#377eb8','#4daf4a', '#e41a1c']
dust_label = ['Olivine', 'Basalt', 'Lime']
#-----------------------------------------------
# CDR with local conditions
#-----------------------------------------------
post_dust_year_num = 60
limits_rate = [-0.05, 1.2]
limits_accu = [-2.5, 12]

plt.figure(figsize = (20, 15), layout = 'tight')

for isoil in range(len(soil_type)):
    #-----------------------------------------------
    # weather rate
    fig_row = 0
    limits_weather = [-0.1, 9]
    axes = plt.subplot(3, 3, fig_row*len(soil_type)+(isoil+1))
    axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
    for idust in range(len(dust_type_list)):
        weathered_rate = dust_weather_summary[np.arange((dust_start_year), (dust_start_year+10)), isoil, idust, idust_amount]*dust_type_molar_weight[idust]/10**6*10**4 # tonnes dust per year per ha
        plt.plot(np.arange(1, (10+1)), weathered_rate, \
            marker = 'o', markersize = 8, linestyle = 'solid', linewidth = 3, color = color_scheme[idust], label = dust_label[idust])
    plt.xlabel(xlabel = '', fontsize = 18)
    plt.ylim(limits_weather)
    plt.tick_params(labelsize = 14, width = 1.5, length = 7)
    if isoil == 0:
        plt.legend(loc='upper right', bbox_to_anchor=(1, 0.85), fontsize = 20, markerscale = 2, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 1)
        plt.ylabel(ylabel = 'Weathering rate (t dust $ha^{-1} \ yr^{-1}$)', fontsize = 18)
    plt.title(str.capitalize(soil_type[isoil]), loc = 'left', x = 0.7, y = 0.87, fontsize = 30, fontweight='bold')
    axes.text(0.03, 0.97, panel_letters[fig_row][isoil], transform = axes.transAxes, fontsize = 30, fontweight='bold', va='top', ha='left')

    #-----------------------------------------------
    # pH
    fig_row = 1
    limits_ph = [4, 8]
    axes = plt.subplot(3, 3, fig_row*len(soil_type)+(isoil+1))
    axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
    for idust in range(len(dust_type_list)):
        soil_ph = ph_summary[np.arange(dust_start_year, (dust_start_year+post_dust_year_num)), isoil, idust, idust_amount]
        plt.plot(np.arange(0, (post_dust_year_num)), soil_ph, \
            marker = 'o', markersize = 8, linestyle = 'solid', linewidth = 3, color = color_scheme[idust], label = dust_label[idust])
    plt.xlabel(xlabel = '', fontsize = 18)
    plt.ylim(limits_ph)
    plt.tick_params(labelsize = 14, width = 1.5, length = 7)
    if isoil == 0:
        plt.ylabel(ylabel = 'pH', fontsize = 18)
    axes.text(0.03, 0.97, panel_letters[fig_row][isoil], transform = axes.transAxes, fontsize = 30, fontweight='bold', va='top', ha='left')

    #-----------------------------------------------
    # accumulative cdr  
    fig_row = 2
    axes = plt.subplot(3, 3, fig_row*len(soil_type)+(isoil+1))
    axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
    plt.axhline(y=0, color='grey', linestyle='--', linewidth=5)
    for idust in range(len(dust_type_list)):
        forcing_loc = dust_type_list[idust]
        cdr_accu = cdr_accu_net_summary[np.arange(dust_start_year, (dust_start_year+post_dust_year_num)), isoil, idust, idust_amount]
        plt.plot(np.arange(0, (post_dust_year_num)), cdr_accu, \
            marker = 'o', markersize = 8, linestyle = 'solid', linewidth = 3, color = color_scheme[idust], label = dust_label[idust])
        plt.xlabel(xlabel = 'Year from amendment', fontsize = 18)
        plt.ylim(limits_accu)
        plt.ylabel(ylabel = 'Cumulative CDR ($t \ CO_{2} \ ha^{-1}$)', fontsize = 18)
        plt.tick_params(labelsize = 14, width = 1.5, length = 7)
    if isoil == 0:
        plt.ylabel(ylabel = 'Cumulative CDR ($t \ CO_{2} \ ha^{-1}$)', fontsize = 18)
    axes.text(0.03, 0.97, panel_letters[fig_row][isoil], transform = axes.transAxes, fontsize = 30, fontweight='bold', va='top', ha='left')

plt.savefig(fig_dir + 'dust_gradient_' + str(dust_amount_list[idust_amount]) + '.jpeg')
plt.savefig(fig_dir + 'dust_gradient_' + str(dust_amount_list[idust_amount]) + '.pdf')
plt.close()

print('end of program')