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

import numpy as np
from scipy.interpolate import pchip_interpolate, griddata

import os

from scipy.io import loadmat, savemat
import netCDF4 as ncread
import torch 

from matplotlib import pyplot as plt

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures') + '/' # the repository's figures folder
os.makedirs(fig_dir, exist_ok = True)

# dust_amount_list = [15.0]
forcing_cycle_num = 5
year_num = 20
month_num = 12
dust_start_year = 40
dust_type = 'olivine'

dust_amount_list = [15.0, 10.0, 5.0, 2.5]
forcing_loc_list = ['temperate', 'subtropical', 'tropical']
soil_type = ['alfisol', 'ultisol', 'oxisol']
species_list = ['Al+++', 'Al(OH)++', 'Al(OH)4-', 'H4SiO4', 'H3SiO4-', 'M+', 'M++', 'OH-', 'H+', 'CO2(aq)', 'CO2(g)', 'HCO3-', 'CO3--', 'H2O', 'MCO3', 'Al(OH)3', 'SM', 'SH2', 'SOA-', 'SOAH']
species_label = ['$Al^{3+}$', '$Al(OH)^{2+}$', '$Al(OH)_{4}^{-}$', '$H_{4}SiO_{4}$', '$H_{3}SiO_{4}^{-}$', '$M^{+}$', '$M^{2+}$', '$OH^{-}$', '$H^{+}$', '$CO_{2}(aq)$', '$CO_{2}(g)$', '$HCO_{3}^{-}$', '$CO_{3}^{2-}$', '$H_{2}O$', '$MCO_{3}$', '$Al(OH)_{3}$', '$SM$', '$SH_{2}$', '$SOA^{-}$', 'SOAH']

color_scheme = ['#e41a1c','#377eb8','#4daf4a']

panel_letters = [['(a)', '(b)', '(c)'],
                 ['(d)', '(e)', '(f)'],
                 ['(g)', '(h)', '(i)'],
                 ['(j)', '(k)', '(l)']]

for idust_amount in [1]: # np.arange(len(dust_amount_list)):
    dust_amount = dust_amount_list[idust_amount] # 10.0

    post_dust_year_num = 60
    plt.figure(figsize = (30, 24), layout = 'tight')

    for isoil in range(len(soil_type)):
        forcing_loc = forcing_loc_list[isoil]

        data_dir = DATAHUB + 'cousine_output/simu/prior_v0_pc/annual_mean/'
        transit_formal_parent = torch.load(f'{data_dir}transit_formal_soil_{soil_type[isoil]}_{forcing_loc}_nitrogen_{0}_{dust_type}_{0.0}.pth')
        transit_formal = torch.load(f'{data_dir}transit_formal_soil_{soil_type[isoil]}_{forcing_loc}_nitrogen_{0}_{dust_type}_{dust_amount}.pth')
        
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

        
        proton_stock =  np.transpose(species_stock[8, :, :]).mean(axis = 1) # mol m-2
        water_stock = np.transpose(species_stock[13, :, :]).mean(axis = 1) # mol m-2
        ph_series = -np.log10(proton_stock/(water_stock*18/1000)) # mol L-1
        initial_ph = -np.log10(species_stock_initial[8]/(species_stock_initial[13]*18/1000))

        transit_year_size = forcing_cycle_num*year_num
        
        #-----------------------------------------------
        # weather rate
        #-----------------------------------------------
        fig_row = 0
        # limits_accu = [-1, 30]
        limits_weather = [-0.5, 7]
        axes = plt.subplot(4, 3, fig_row*len(soil_type)+(isoil+1))
        axes.spines[['top','bottom','left','right']].set_linewidth(2.0)
        weathered_rate = dust_consumed.sum(axis = 1)*dust_molar_weight/10**6*10**4 # tonnes dust per year per ha
        weathered_rate[np.isnan(weathered_rate) == True] = 0.0
        plt.plot(np.arange(0, (transit_year_size)), weathered_rate[0, :], \
            marker = 'o', markersize = 8, linestyle = 'solid', linewidth = 3, color = 'black')

        plt.xlabel(xlabel = '', fontsize = 25)
        plt.ylim(limits_weather)
        if isoil == 0:
            plt.ylabel(ylabel = 'Weathering rate\n(t $ha^{-1} \ yr^{-1}$)', fontsize = 36)
        plt.tick_params(labelsize = 28, width = 3, length = 12)
        plt.title(str.capitalize(soil_type[isoil]), loc = 'left', x = 0.7, y = 0.87, fontsize = 40,  fontweight='bold')
        axes.text(0.03, 0.97, panel_letters[fig_row][isoil], transform = axes.transAxes, fontsize = 40, fontweight='bold', va='top', ha='left')

        #-----------------------------------------------
        # pH changes
        #-----------------------------------------------
        fig_row = 1
        limits_ph = [4, 8]

        axes = plt.subplot(4, 3, fig_row*len(soil_type)+(isoil+1))
        axes.spines[['top','bottom','left','right']].set_linewidth(2.0)
    
        # proton_stock =  np.concatenate((species_stock_initial[0, 8].reshape(1, 1), np.transpose(species_stock[8, :, :]).reshape(((transit_year_size*month_num), 1)))) # mol m-2
        # water_stock = np.concatenate((species_stock_initial[0, 13].reshape(1, 1), np.transpose(species_stock[13, :, :]).reshape(((transit_year_size*month_num), 1)))) # mol m-2
        # ph_series = -np.log10(proton_stock/(water_stock*18/1000)) # mol L-1

        proton_stock =  species_stock[8, :, :].mean(axis = 0) # mol m-2
        water_stock = species_stock[13, :, :].mean(axis = 0) # mol m-2
        ph_series = -np.log10(proton_stock/(water_stock*18/1000)) # mol L-1

        plt.plot(np.arange(0, transit_year_size), ph_series, \
                marker = '', markersize = 0, linestyle = 'solid', linewidth = 6, color = 'black')

        plt.xlabel(xlabel = '', fontsize = 25)
        plt.ylim(limits_ph)
        if isoil == 0:
            plt.ylabel(ylabel = 'pH\n(pore water)', fontsize = 36)
        plt.tick_params(labelsize = 28, width = 3, length = 12)
        axes.text(0.03, 0.97, panel_letters[fig_row][isoil], transform = axes.transAxes, fontsize = 40, fontweight='bold', va='top', ha='left')
        
        #-----------------------------------------------
        # carbon related species absolute change
        #-----------------------------------------------
        fig_row = 2
        limits_carbon = [10**(-10), 1]
        axes = plt.subplot(4, 3, fig_row*len(soil_type)+(isoil+1))
        axes.spines[['top','bottom','left','right']].set_linewidth(2.0)

        # hco3_stock = np.transpose(species_stock[11, :, :]).reshape(((transit_year_size*month_num), 1))
        # co3_stock = np.transpose(species_stock[12, :, :]).reshape(((transit_year_size*month_num), 1))
        # mco3_stock = np.transpose(species_stock[14, :, :]).reshape(((transit_year_size*month_num), 1))

        hco3_stock = species_stock[11, :, :].mean(axis = 0)
        co3_stock = species_stock[12, :, :].mean(axis = 0)
        mco3_stock = species_stock[14, :, :].mean(axis = 0)

        plt.plot(np.arange(0, transit_year_size), hco3_stock, \
            marker = '', markersize = 0, linestyle = 'solid', linewidth = 6, color = '#56B4E9', label = species_label[11])
        plt.plot(np.arange(0, transit_year_size), co3_stock, \
            marker = '', markersize = 0, linestyle = 'solid', linewidth = 6, color = '#009E73', label = species_label[12])
        plt.plot(np.arange(0, transit_year_size), mco3_stock, \
            marker = '', markersize = 0, linestyle = 'solid', linewidth = 6, color = '#D55E00', label = species_label[14])
        
        plt.yscale('log')         
        if isoil == 2:
            plt.legend(loc = 'best', fontsize = 30, markerscale = 6, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 1)
        plt.xlabel(xlabel = '', fontsize = 25)
        plt.ylim(limits_carbon)
        if isoil == 0:
            plt.ylabel(ylabel = 'Species stock\n($mol \ m^{-2}$)', fontsize = 36)
        plt.tick_params(labelsize = 28, width = 3, length = 12)
        axes.text(0.03, 0.97, panel_letters[fig_row][isoil], transform = axes.transAxes, fontsize = 40, fontweight='bold', va='top', ha='left')

        #-----------------------------------------------
        # cdr rate
        #-----------------------------------------------
        fig_row = 3
        limits_rate = [-0.05, 2.0]
        
        axes = plt.subplot(4, 3, fig_row*len(soil_type)+(isoil+1))
        axes.spines[['top','bottom','left','right']].set_linewidth(2.0)
    
        cdr_rate_net = np.diff(cdr_accu_site_net)
        cdr_rate_parent = np.diff(cdr_accu_site_parent)
        cdr_rate_tot = cdr_rate_net + cdr_rate_parent

        plt.plot(np.arange(1, (transit_year_size)), cdr_rate_parent, color = '#FFC20A', linestyle = 'solid', linewidth = 3)
        plt.fill_between(np.arange(1, (transit_year_size)), 0, cdr_rate_parent, label = 'Parent CDR', color = '#FFC20A', alpha=0.5)
        plt.plot(np.arange(1, (transit_year_size))[(dust_start_year-2):], cdr_rate_tot[(dust_start_year-2):], color = '#0C7BDC', linestyle = 'solid', linewidth = 3)
        plt.fill_between(np.arange(1, (transit_year_size)), cdr_rate_parent, cdr_rate_tot, label='ERW induced CDR', color = '#0C7BDC', alpha=0.5)

        # x label only under panel (k): bottom row (CDR rate), middle soil column
        if isoil == 1:
            plt.xlabel(xlabel = 'Simulation Year', fontsize = 36)
        else:
            plt.xlabel(xlabel = '', fontsize = 36)
        plt.ylim(limits_rate)
        if isoil == 0:
            plt.ylabel(ylabel = 'CDR rate\n($t \ CO_{2} \ ha^{-1} \ yr^{-1}$)', fontsize = 36)
            plt.legend(loc = 'best', fontsize = 30, markerscale = 3, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 1)
        plt.tick_params(labelsize = 28, width = 3, length = 12)
        axes.text(0.03, 0.97, panel_letters[fig_row][isoil], transform = axes.transAxes, fontsize = 40, fontweight='bold', va='top', ha='left')

        # #-----------------------------------------------
        # # cec related species absolute change
        # #-----------------------------------------------
        # fig_row = 4
        # limits_cec = [0, 8]
        # axes = plt.subplot(5, 3, fig_row*len(soil_type)+(isoil+1))
        # axes.spines[['top','bottom','left','right']].set_linewidth(2.0)

        # sm_stock = np.transpose(species_stock[16, :, :]).reshape(((transit_year_size*month_num), 1))
        # sh2_stock = np.transpose(species_stock[17, :, :]).reshape(((transit_year_size*month_num), 1))
        # plt.plot(np.arange(0, (transit_year_size*month_num))/12, sm_stock, \
        #     marker = '', markersize = 0, linestyle = 'solid', linewidth = 3, color = '#FFC20A', label = species_label[16])
        # plt.plot(np.arange(0, (transit_year_size*month_num))/12, sh2_stock, \
        #     marker = '', markersize = 0, linestyle = 'solid', linewidth = 3, color = '#0C7BDC', label = species_label[17])
        # if isoil == 2:
        #     plt.legend(loc = 'best', fontsize = 30, markerscale = 3, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 1)
        # plt.xlabel(xlabel = 'Simulation year', fontsize = 25)
        # plt.ylim(limits_cec)
        # if isoil == 0:
        #     plt.ylabel(ylabel = 'Species stock ($mol \ m^{-2}$)', fontsize = 25)
        # plt.tick_params(labelsize = 25, width = 2, length = 7)

    plt.savefig(fig_dir + 'weathering_progress_and_cdr_summary' + '_' + str(dust_amount) + '.pdf')
    plt.savefig(fig_dir + 'weathering_progress_and_cdr_summary' + '_' + str(dust_amount) + '.jpeg')
    plt.close()


print('end of programme')


