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

import torch

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

forcing_cycle_num = 5
year_num = 20
month_num = 12
dust_start_year = 40
tot_simu_year_num = year_num*forcing_cycle_num

species_list =            ['Al+++', 'Al(OH)++', 'Al(OH)4-', 'H4SiO4', 'H3SiO4-', 'M+', 'D++', 'OH-', 'H+', 'CO2(aq)', 'CO2(g)', 'HCO3-', 'CO3--', 'H2O', 'DCO3', 'Al(OH)3', 'PiD', 'Pi3Al2',         'SOA2D', 'SOAH']
species_charge = np.array([3,        2,          -1,         0,        -1,        1,    2,     -1,    1,    0,         0,        -1,      -2,     0,     0,      0,         0,     0,                0,       0])
species_label = ['$Al^{3+}$', '$Al(OH)^{2+}$', '$Al(OH)_{4}^{-}$', '$H_{4}SiO_{4}$', '$H_{3}SiO_{4}^{-}$', '$M^{+}$', '$M^{2+}$', '$OH^{-}$', '$H^{+}$', '$CO_{2}(aq)$', '$CO_{2}(g)$', '$HCO_{3}^{-}$', '$CO_{3}^{2-}$', '$H_{2}O$', '$MCO_{3}$', '$Al(OH)_{3}$', '$SM$', '$SH_{2}$', '$SOA^{-}$', 'SOAH']


dust_type = 'olivine'

dust_amount = 10.0 # [10.0] # [2.5, 5.0, 10.0, 15.0]
nitrogen_list = [0, 8]
forcing_list = ['temperate', 'subtropical', 'tropical']
soil_list = ['alfisol', 'ultisol', 'oxisol']
label_list = [['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)']]

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures') + '/' # the repository's figures folder
os.makedirs(fig_dir, exist_ok = True)
fig = plt.figure(figsize = (25, 13), layout = 'tight')

for icase in np.arange(len(nitrogen_list)):
	case_name = nitrogen_list[icase]
	for isoil in np.arange(len(soil_list)):
		forcing_loc = forcing_list[isoil] # 'ny'
		soil_type = soil_list[isoil] # 'oxisol'
		data_dir = DATAHUB + 'cousine_output/simu/prior_v0_pc/annual_mean/'
		transit_formal_parent = torch.load(f'{data_dir}transit_formal_soil_{soil_type}_{forcing_loc}_nitrogen_{0}_{dust_type}_{0.0}.pth')
		transit_formal = torch.load(f'{data_dir}transit_formal_soil_{soil_type}_{forcing_loc}_nitrogen_{case_name}_{dust_type}_{10.0}.pth')
		
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
		parent_molar_weight = (molar_weight_element*stoichio_element_parent).sum() # g mol-1
		# dust molar weight
		dust_molar_weight = (molar_weight_element*stoichio_element_dust).sum() # g mol-1

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
			
		transit_year_size = forcing_cycle_num*year_num
		
		#-----------------------------------------------
		# weather rate
		#-----------------------------------------------
		weathered_rate = np.nansum(dust_consumed, axis = (0, 1))*stoichio_m_2cat/stoichio_element_olivine[[5, 6]].sum()*olivine_molar_weight/10**6*10**4 # mol m-1 --> tonnes olivine eq per year per ha

		weathered_cumsum_amount = weathered_rate.cumsum(dim = 0)

		#------------------------------------------------
		# hydrogen mass balance
		#------------------------------------------------
		proton_stoichio = stoichio_m_1cat + 2*stoichio_m_2cat + 3*stoichio_al_3cat
		proton_stoichio_parent = stoichio_m_1cat_parent + 2*stoichio_m_2cat_parent + 3*stoichio_al_3cat_parent
		
		tot_required_proton_tot = proton_stoichio*np.transpose(np.nansum(dust_consumed, axis = 0)).sum(axis = 1).cumsum() + proton_stoichio_parent*np.transpose(np.nansum(dust_consumed_base_dust, axis = 0)).sum(axis = 1).cumsum()
		tot_required_proton_parent = proton_stoichio_parent*np.transpose(np.nansum(dust_consumed_base_parent, axis = 0)).sum(axis = 1).cumsum()

		net_h_by_h2o = (delta_species_stock)[0:6, 13, :, :].sum(axis = (0, 1, )).cumsum()
		net_h_by_soah = (delta_species_stock)[0:6, 19, :, :].sum(axis = (0, 1, )).cumsum()
		net_h_by_nitrogen = -(leached_ion_stock[8, :, :]).sum(axis = 0).cumsum()
		
		net_h_by_h2o_parent = (delta_species_stock_parent)[0:6, 13, :, :].sum(axis = (0, 1, )).cumsum()
		net_h_by_soah_parent = (delta_species_stock_parent)[0:6, 19, :, :].sum(axis = (0, 1, )).cumsum()
		net_h_by_nitrogen_parent = -(leached_ion_stock_parent[8, :, :]).sum(axis = 0).cumsum()

		#------------------------------------------------
		# proton attribution
		#------------------------------------------------
		proton_contribution_carbonate_tot = 1*(-delta_species_stock)[0:6, 11, :, :].sum(axis = (0, 1, )).cumsum() + \
										2*(-delta_species_stock)[0:6, 12, :, :].sum(axis = (0, 1, )).cumsum() + \
										2*(-delta_species_stock)[0:6, 14, :, :].sum(axis = (0, 1, )).cumsum() + \
										(1*leached_ion_stock[11, :, :].sum(axis = 0).cumsum() + \
											2*leached_ion_stock[12, :, :].sum(axis = 0).cumsum() + \
											2*leached_ion_stock[14, :, :].sum(axis = 0).cumsum())
		proton_contribution_carbonate_parent = 1*(-delta_species_stock_parent)[0:6, 11, :, :].sum(axis = (0, 1, )).cumsum() + \
										2*(-delta_species_stock_parent)[0:6, 12, :, :].sum(axis = (0, 1, )).cumsum() + \
										2*(-delta_species_stock_parent)[0:6, 14, :, :].sum(axis = (0, 1, )).cumsum() + \
										(leached_ion_stock_parent[11, :, :].sum(axis = 0).cumsum() + \
											2*leached_ion_stock_parent[12, :, :].sum(axis = 0).cumsum() + \
											2*leached_ion_stock_parent[14, :, :].sum(axis = 0).cumsum())

		proton_contribution_si_tot = 1*(-delta_species_stock)[0:6, 4, :, :].sum(axis = (0, 1, )).cumsum() 
		proton_contribution_si_parent = 1*(-delta_species_stock_parent)[0:6, 4, :, :].sum(axis = (0, 1, )).cumsum() 


		proton_contribution_soa_tot = -1*(-delta_species_stock)[0:6, 19, :, :].sum(axis = (0, 1, )).cumsum() 
		proton_contribution_soa_parent = -1*(-delta_species_stock_parent)[0:6, 19, :, :].sum(axis = (0, 1, )).cumsum() 

		proton_contribution_al_tot = 1*(-delta_species_stock)[0:6, 1, :, :].sum(axis = (0, 1, )).cumsum() + \
								4*(-delta_species_stock)[0:6, 2, :, :].sum(axis = (0, 1, )).cumsum() + \
								3*(-delta_species_stock)[0:6, 15, :, :].sum(axis = (0, 1, )).cumsum() + \
								1*(leached_ion_stock[1, :, :].sum(axis = 0)).cumsum() + \
								4*(leached_ion_stock[2, :, :].sum(axis = 0).cumsum()) + \
								3*(leached_ion_stock[15, :, :].sum(axis = 0).cumsum()) 
		
		proton_contribution_al_parent = 1*(-delta_species_stock_parent)[0:6, 1, :, :].sum(axis = (0, 1, )).cumsum() + \
								4*(-delta_species_stock_parent)[0:6, 2, :, :].sum(axis = (0, 1, )).cumsum() + \
								3*(-delta_species_stock_parent)[0:6, 15, :, :].sum(axis = (0, 1, )).cumsum() + \
								1*(leached_ion_stock_parent[1, :, :].sum(axis = 0)).cumsum() + \
								4*(leached_ion_stock_parent[2, :, :].sum(axis = 0).cumsum()) + \
								3*(leached_ion_stock_parent[15, :, :].sum(axis = 0).cumsum()) 

		proton_contribution_nitrate_tot = -(leached_ion_stock[8, :, :].sum(axis = 0).cumsum())
		proton_contribution_nitrate_parent = -(leached_ion_stock_parent[8, :, :].sum(axis = 0).cumsum())


		tot_required_proton = tot_required_proton_tot - tot_required_proton_parent*parent_weathering_depress_scalar

		proton_contribution_soa = proton_contribution_soa_tot - proton_contribution_soa_parent*parent_weathering_depress_scalar
		proton_contribution_al = proton_contribution_al_tot - proton_contribution_al_parent*parent_weathering_depress_scalar
		proton_contribution_si = proton_contribution_si_tot - proton_contribution_si_parent*parent_weathering_depress_scalar
		proton_contribution_carbonate = proton_contribution_carbonate_tot - proton_contribution_carbonate_parent*parent_weathering_depress_scalar
		proton_contribution_nitrate = proton_contribution_nitrate_tot - proton_contribution_nitrate_parent*parent_weathering_depress_scalar

		cation_sink_withhold_gross = np.concatenate((proton_contribution_soa_tot.reshape((tot_simu_year_num, 1)), 
									proton_contribution_al_tot.reshape((tot_simu_year_num, 1)), 
									proton_contribution_si_tot.reshape((tot_simu_year_num, 1)),
									proton_contribution_carbonate_tot.reshape((tot_simu_year_num, 1)),
									proton_contribution_nitrate_tot.reshape((tot_simu_year_num, 1))
									), axis = 1)

		cation_sink_withhold_net = np.concatenate((proton_contribution_soa.reshape((tot_simu_year_num, 1)), 
									proton_contribution_al.reshape((tot_simu_year_num, 1)), 
									proton_contribution_si.reshape((tot_simu_year_num, 1)),
									proton_contribution_carbonate.reshape((tot_simu_year_num, 1)),
									proton_contribution_nitrate.reshape((tot_simu_year_num, 1))
									), axis = 1)
		
		#########################################################
		# figures
		#########################################################
		axes = plt.subplot(2, 3, (icase)*len(soil_list)+(isoil+1))
		axes.spines[['top','bottom','left','right']].set_linewidth(1.5)

		color_list = ['#E69F00','#56B4E9','#009E73', '#CC79A7','#D55E00', '#0072B2']

		process_list = ['Organic acids', 'Clay phases ($Al^{3+}$)', 'Silicic acid', 'Carbonic acid', 'Net nitrogen acidity']

		axes.plot(np.arange(0, tot_simu_year_num-dust_start_year), tot_required_proton[dust_start_year:tot_simu_year_num], linestyle = 'solid', color = 'black', linewidth = 5, label = 'Required $H^{+}$ for ERW')
		for iprocess in np.arange(cation_sink_withhold_net.shape[1]):
			# plt.plot(np.arange(0, 80), current_data[:, 0:(iprocess+1)], linestyle = 'solid', linewidth = 3, color = color_list[iprocess])
			if iprocess == 0:
				axes.fill_between(np.arange(0, tot_simu_year_num-dust_start_year), 0, cation_sink_withhold_net[dust_start_year:tot_simu_year_num, 0:(iprocess+1)].sum(axis = 1), linestyle = 'solid', linewidth = 3, color = color_list[iprocess], label = process_list[iprocess], alpha = 0.5)
			else:
				axes.fill_between(np.arange(0, tot_simu_year_num-dust_start_year), cation_sink_withhold_net[dust_start_year:tot_simu_year_num, 0:(iprocess)].sum(axis = 1), cation_sink_withhold_net[dust_start_year:tot_simu_year_num, 0:(iprocess+1)].sum(axis = 1), linestyle = 'solid', linewidth = 3, label = process_list[iprocess], color = color_list[iprocess], alpha=0.5)
		plt.tick_params(labelsize = 25, width = 2, length = 7)
		plt.ylim([None, 35])
		if icase == 0:
			plt.title(soil_list[isoil].capitalize(), fontsize = 40)
		plt.text(0.03, 0.97, label_list[icase][isoil], transform = axes.transAxes, fontsize = 30, fontweight='bold', va='top', ha='left')
		if icase == 0 and isoil == 2:
			plt.legend(loc = 'center', bbox_to_anchor=(1.5, 0.5), mode = None, fontsize = 25, shadow = False, fancybox = False, framealpha = 0, ncols = 1)


# single figure-level labels: one y label shared across both rows, one x label
# at the bottom of the whole figure
fig.supylabel('Contributed $H^{+}$ ($mol \ m^{-2}$)', fontsize = 35, x = 0.01, y = 0.5)
fig.supxlabel('Year from amendment', fontsize = 35, x = 0.42, y = 0.0)

plt.savefig(fig_dir + 'proton_attribution_' + str(dust_amount) + '.pdf', bbox_inches='tight')
plt.savefig(fig_dir + 'proton_attribution_' + str(dust_amount) + '.jpeg', bbox_inches='tight')
plt.close()

print('end of programme')
