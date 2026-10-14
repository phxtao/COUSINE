# COUSINE (COUpled Soil Inorganic-organic carbon model for eNhanced wEathering)
# Copyright (c) 2026 Cornell University. All Rights Reserved. Cornell docket D-11800.
# Permission to use for educational, research and non-profit purposes only; see LICENSE for the terms,
# the commercial-licensing contact and the disclaimers. The concepts used in this work are patent pending.

import time
import os
import multiprocessing

from datetime import datetime
from pandas import read_csv
import numpy as np

import torch

from scipy.io import loadmat
import netCDF4 as ncread
import tifffile
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from multiprocessing import cpu_count

from fun_carbon_dynamics_beerling import * # import fun_matrix_soc_ss, fun_matrix_transit
from fun_site_inputs import cousine_data_path, load_site_inputs, site_loc_1deg_of, load_site_env_info, site_of, SITES # site-only DATAHUB

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)

print(datetime.now(), '------------all packages loaded------------')

dev = 'cpu'
device = torch.device(dev)
print(datetime.now(), '------------device: ', dev, '------------')

is_plotting = 0

cropland_only = True
is_use_solid_mineral = False

plot_chem_map = False

para_prior_version = 0
opt_version = 'version_20260315'
is_nn_training = 0

################################################
# data path
################################################
cesm2_case_name_hat = 'phxtao_f09_g17_hist_simu' # 'hist_f05_g16_checked'
start_year = 1901
end_year = 1920 

year_num = np.arange(start_year, (end_year+1)).shape[0]

# mac
data_path = cousine_data_path() # site-only data root, see fun_site_inputs.cousine_data_path()

###############################################
# global grid area
###############################################
earth_radius_km = 6371.0 # km
grid_resolution = 0.5

lon_range_list = torch.arange(-180 + grid_resolution/2, 180 - grid_resolution/2, grid_resolution)
lat_range_list = torch.arange(90 - grid_resolution/2, -90 + grid_resolution/2, -grid_resolution)

global_soil_map_lon = (torch.arange(-180 + grid_resolution/2, 180 + grid_resolution/2, grid_resolution)).reshape([1, int(360/grid_resolution)]).repeat(int(180/grid_resolution), 1)
global_soil_map_lat = (torch.arange(90 - grid_resolution/2, -90 - grid_resolution/2, -grid_resolution)).reshape([int(180/grid_resolution), 1]).repeat(1, int(360/grid_resolution))

# 2. Convert to radians
lat_rad_north = np.radians(global_soil_map_lat + grid_resolution/2)
lat_rad_south = np.radians(global_soil_map_lat - grid_resolution/2)
dlon_rad = np.radians(grid_resolution)

# Calculate area formula: R^2 * (sin(lat_n) - sin(lat_s)) * dlon
# We calculate the "slice" area for each latitude band
# The term (sin(lat_north) - sin(lat_south)) handles the meridian convergence
lat_term = np.sin(lat_rad_north) - np.sin(lat_rad_south)

# Area of a cell in this latitude band (same for all longitudes in the band)
global_soil_map_area = (earth_radius_km ** 2) * lat_term * dlon_rad * 10**2  # unit ha

###############################################
# soil lithology map
###############################################
# "OBJECTID" "xx"
#  1;        "su"
#  2;        "vb"
#  3;        "ss"
#  4;        "pb"
#  5;        "sm"
#  6;        "sc"
#  7;        "va"
#  8;        "mt"
#  9;        "pa"
#  10;       "vi"
#  11;       "wb"
#  12;       "py"
#  13;       "pi"
#  14;       "ev"
#  15;       "nd"
#  16;       "ig"

# global soil, lithology, runoff and nitrate maps: site values are read in worker_function (fun_site_inputs.py)

atm_pressure = 101325 # Pa
gas_const = 8.3145 # unit J (mol)-1 K-1
month_num = 12
soil_cpool_num = 7
soil_decom_num = 1
one_layer_compensation = 0.280000000

days_per_month = torch.tensor([31, 30, 31, 28, 31, 30, 31, 31, 30, 31, 30, 31])

#---------------------------CLM5 constants-------------------------------
soc_para_names = ['diffus', 'cryo', 'q10', 'efolding', 'taucwd', 'taul1', 'taul2', 'tau4s1', 'tau4s2', 'tau4s3', 'fl1s1', 'fl2s1', 'fl3s2', 'fs1s2', 'fs1s3', 'fs2s1', 'fs2s3', 'fs3s1', 'fcwdl2', 'w-scaling', 'beta']
# soil depths info
# width between two interfaces
dz = torch.tensor([2.000000000000000E-002, 4.000000000000000E-002, 6.000000000000000E-002, \
8.000000000000000E-002, 0.120000000000000, 0.160000000000000, \
0.200000000000000, 0.240000000000000, 0.280000000000000, \
0.320000000000000, 0.360000000000000, 0.400000000000000, \
0.440000000000000, 0.540000000000000, 0.640000000000000, \
0.740000000000000, 0.840000000000000, 0.940000000000000, \
1.04000000000000, 1.14000000000000, 2.39000000000000, \
4.67553390593274, 7.63519052838329, 11.1400000000000, \
15.1154248593737])

# depth of the interface
zisoi = torch.tensor([2.000000000000000E-002, 6.000000000000000E-002, \
0.120000000000000, 0.200000000000000, 0.320000000000000, \
0.480000000000000, 0.680000000000000, 0.920000000000000, \
1.20000000000000, 1.52000000000000, 1.88000000000000, \
2.28000000000000, 2.72000000000000, 3.26000000000000, \
3.90000000000000, 4.64000000000000, 5.48000000000000, \
6.42000000000000, 7.46000000000000, 8.60000000000000, \
10.9900000000000, 15.6655339059327, 23.3007244343160, \
34.4407244343160, 49.5561492936897])

zisoi_0 = torch.tensor([0])

# when using one layer, need to do:
dz = dz + one_layer_compensation
zisoi = zisoi + one_layer_compensation

# depth of the node
zsoi = (zisoi + torch.concat((zisoi_0, zisoi[:-1])))/2
# depth between two node
dz_node = zsoi - np.append(np.array([0]), zsoi[:-1], axis = 0)

# land mask

###############################################
# site simulation
###############################################
#---------------------------------------------------------
# model simulation
#---------------------------------------------------------
fig_dir = data_path + 'cousine_output/binn_opt/prior_v' + str(para_prior_version) + '_pc/' #  + region_of_interest + '/'
data_dir = data_path + 'cousine_output/binn_opt/prior_v' + str(para_prior_version) + '_pc/' #  + region_of_interest + '/'

simu_fig_dir = data_path + 'cousine_output/simu/prior_v' + str(para_prior_version) + '_pc/' #  + region_of_interest + '/'
simu_data_dir = data_path + 'cousine_output/simu/prior_v' + str(para_prior_version) + '_pc/' #  + region_of_interest + '/'

os.makedirs(fig_dir, exist_ok = True)
os.makedirs(data_dir, exist_ok = True)

os.makedirs(simu_fig_dir, exist_ok = True)
os.makedirs(simu_fig_dir + 'annual_mean/', exist_ok = True)
os.makedirs(simu_fig_dir + 'interannual_var/', exist_ok = True)
os.makedirs(simu_fig_dir + 'climate_change/', exist_ok = True)
	
os.makedirs(simu_data_dir, exist_ok = True)
os.makedirs(simu_data_dir + 'annual_mean/', exist_ok = True)
os.makedirs(simu_data_dir + 'annual_mean/sensitivity_beerling_soil/', exist_ok = True)
os.makedirs(simu_data_dir + 'interannual_var/', exist_ok = True)
os.makedirs(simu_data_dir + 'climate_change/', exist_ok = True)

#---------------------------------------------------
# var of interest input                      
#---------------------------------------------------	
if is_use_solid_mineral == True:	
	species_list =                  ['Al+++', 'Al(OH)++', 'Al(OH)4-', 'H4SiO4', 'H3SiO4-', 'M+', 'D++', 'OH-', 'H+', 'CO2(aq)', 'CO2(g)', 'HCO3-', 'CO3--', 'H2O', 'DCO3', 'Al(OH)3', 'PiD', 'Pi3Al2',       'SOA2D', 'SOAH',  'Carbonate-DCO3(s)',  'Gibbsite-Al(OH)3(s)']
	species_charge =   torch.tensor([3,        2,          -1,         0,        -1,        1,    2,     -1,    1,    0,         0,        -1,      -2,     0,     0,      0,         0,     0,                0,       0,       0,                    0                    ]).to(device)
	element_mass_matrix = torch.tensor([
									[1,        1,           1,         0,         0,        0,    0,      0,    0,    0,         0,         0,       0,     0,     0,      1,         0,     2,                0,       0,       0,                    1                    ], # Al
									[0,        0,           0,         0,         0,        1,    0,      0,    0,    0,         0,         0,       0,     0,     0,      0,         0,     0,                0,       0,       0,                    0                    ], # M
									[0,        0,           0,         0,         0,        0,    1,      0,    0,    0,         0,         0,       0,     0,     1,      0,         1,     0,                1,       0,       1,                    0                    ], # D
									[0,        0,           0,         1,         1,        0,    0,      0,    0,    0,         0,         0,       0,     0,     0,      0,         0,     0,                0,       0,       0,                    0                    ], # Si
									[0,        1,           4,         4,         4,        0,    0,      1,    0,    2,         2,         3,       3,     1,     3,      3,         0,     0,                0,       0,       3,                    3                    ], # O
									[0,        1,           4,         4,         3,        0,    0,      1,    1,    0,         0,         1,       0,     2,     0,      3,         0,     0,                0,       1,       0,                    3                    ], # H
									[0,        0,           0,         0,         0,        0,    0,      0,    0,    1,         1,         1,       1,     0,     1,      0,         0,     0,                0,       0,       1,                    0                    ], # C
									[0,        0,           0,         0,         0,        0,    0,      0,    0,    0,         0,         0,       0,     0,     0,      0,         1,     3,                0,       0,       0,                    0                    ], # Pi
									[0,        0,           0,         0,         0,        0,    0,      0,    0,    0,         0,         0,       0,     0,     0,      0,         0,     0,                2,       1,       0,                    0                    ]  # SOA
								]).to(device)
else:
	species_list =                  ['Al+++', 'Al(OH)++', 'Al(OH)4-', 'H4SiO4', 'H3SiO4-', 'M+', 'D++', 'OH-', 'H+', 'CO2(aq)', 'CO2(g)', 'HCO3-', 'CO3--', 'H2O', 'DCO3', 'Al(OH)3', 'PiD', 'Pi3Al2',       'SOA2D', 'SOAH']
	species_charge =   torch.tensor([3,        2,          -1,         0,        -1,        1,    2,     -1,    1,    0,         0,        -1,      -2,     0,     0,      0,         0,     0,                0,       0                   ]).to(device)
	element_mass_matrix = torch.tensor([
									[1,        1,           1,         0,         0,        0,    0,      0,    0,    0,         0,         0,       0,     0,     0,      1,         0,     2,                0,       0                    ], # Al
									[0,        0,           0,         0,         0,        1,    0,      0,    0,    0,         0,         0,       0,     0,     0,      0,         0,     0,                0,       0                    ], # M
									[0,        0,           0,         0,         0,        0,    1,      0,    0,    0,         0,         0,       0,     0,     1,      0,         1,     0,                1,       0                    ], # D
									[0,        0,           0,         1,         1,        0,    0,      0,    0,    0,         0,         0,       0,     0,     0,      0,         0,     0,                0,       0                    ], # Si
									[0,        1,           4,         4,         4,        0,    0,      1,    0,    2,         2,         3,       3,     1,     3,      3,         0,     0,                0,       0                    ], # O
									[0,        1,           4,         4,         3,        0,    0,      1,    1,    0,         0,         1,       0,     2,     0,      3,         0,     0,                0,       1                    ], # H
									[0,        0,           0,         0,         0,        0,    0,      0,    0,    1,         1,         1,       1,     0,     1,      0,         0,     0,                0,       0                    ], # C
									[0,        0,           0,         0,         0,        0,    0,      0,    0,    0,         0,         0,       0,     0,     0,      0,         1,     3,                0,       0                    ], # Pi
									[0,        0,           0,         0,         0,        0,    0,      0,    0,    0,         0,         0,       0,     0,     0,      0,         0,     0,                2,       1                    ]  # SOA
								]).to(device)

element_list = ['Al', 'D2', 'H', 'O', 'C', 'Si']
species_num = len(species_list)
element_num = len(element_list)


###############################################
# management input
###############################################
dust_amount = 0 # 10.0
is_repeat_app = False

site_list = [
            [- 90,       41,      7294],  # # example temperate   ID: 7294
            [- 85.0,    31.0,     7909],  # # example subtropical ID: 7909
            [- 63.5,   -10.0,     12224]  # # example tropical    ID: 12224
			]

dust_amount_list = [0.0, 2.5, 5.0, 10.0, 15.0] # [0.0, 2.5, 5.0, 10.0, 15.0] # 
soil_type_list = ['alfisol', 'ultisol', 'oxisol'] # ['alfisol'] # ['alfisol', 'ultisol', 'oxisol']
forcing_loc_list = ['temperate', 'subtropical', 'tropical'] #['ga'] # ['ny', 'ga', 'br']

kinetic_scalar_list = [1, 3, 6, 8, 10, 12]
leach_scalar_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5] 
pco2_scalar_list = [0.7, 1, 1.3, 1.5, 1.8, 2]

###############################################
# management input
###############################################
opt_simu = False

def worker_function(ikinetic, ileach, ipco2):
	time.sleep(2)  # Simulate a time-consuming task

	isoil_type = 0
	iforcing_loc = isoil_type

	soil_type = soil_type_list[isoil_type] # 'ultisol' # alfisol, ultisol, oxisol
	forcing_loc = forcing_loc_list[iforcing_loc] # 'ga' # 'ny', 'ga', 'br'

	dust_amount = 50.0*0.6542 # dust_amount_list[idust_amount]# 10.0
		
	nitrogen_fertilizer_input = 20 # kg N ha-1 yr-1
	dust_type = 'basalt' # ['olivine', 'lime', 'basalt']

	adjust_scalar_kinetic = kinetic_scalar_list[ikinetic]
	adjust_scalar_leach = leach_scalar_list[ileach]
	adjust_scalar_pco2 = pco2_scalar_list[ipco2]

	# skip this parameter combination when its output file already exists, so a
	# re-run only fills in the missing simulations instead of recomputing every
	# combination (and skips the expensive forcing/parameter loading below)
	sensitivity_result_path = (f'{simu_data_dir}annual_mean/sensitivity_beerling_soil/sensitivity_beerling_soil_{soil_type}_{forcing_loc}'
		f'_nitrogen_{nitrogen_fertilizer_input}_{dust_type}_{dust_amount}'
		f'_kinetic_{adjust_scalar_kinetic}_leach_{adjust_scalar_leach}_pco2_{adjust_scalar_pco2}.csv')
	# if os.path.exists(sensitivity_result_path) == True:
	# 	print(f'skipping existing simulation: scalars for kinetics {adjust_scalar_kinetic} leach {adjust_scalar_leach} pco2 {adjust_scalar_pco2}')
	# 	return

	# # # example temperate
	if soil_type == 'alfisol':
		site_lon = -90
		site_lat = 41
		site_loc = 7294

	# # # example subtropical 		
	if soil_type == 'ultisol':
		site_lon = -85
		site_lat = 31
		site_loc = 7909

	# # # example tropical 
	if soil_type == 'oxisol':
		site_lon = -63.5
		site_lat = -10.0
		site_loc = 12224

	# try:
	# print(f'processing site {site_loc} at lon: {site_lon} and lat {site_lat}')

	# site litho
	site_inputs = load_site_inputs(data_path, site_loc) # site values extracted from the global datasets
	site_litho_type = site_inputs['site_litho_type']
	site_parent_cdr = site_inputs['site_parent_cdr'] # t CO2 ha-1 yr-1
	site_ex_sol_ratio_cation = site_inputs['site_ex_sol_ratio_cation']
	site_max_cation_sink = site_inputs['site_max_cation_sink']
	site_soil_chem_exal = site_inputs['site_soil_chem_exal']
	site_soil_chem_cec = site_inputs['site_soil_chem_cec']
	site_soil_chem_bs = site_inputs['site_soil_chem_bs']
	site_soil_chem_ph = site_inputs['site_soil_chem_ph']
	site_soil_chem_exd2 = site_inputs['site_soil_chem_exd2']
	site_soil_chem_bd = site_inputs['site_soil_chem_bd']
	site_soil_chem_soc = site_inputs['site_soil_chem_soc']
	site_soil_chem_carbonate = site_inputs['site_soil_chem_carbonate'] # mol m-3
	site_soil_chem_gibbsite = site_inputs['site_soil_chem_gibbsite'] # mol m-3
	site_soil_chem_no3_leach = site_inputs['site_soil_chem_no3_leach'] # kg N ha-1 yr-1
	site_loc_1deg = site_inputs['site_loc_1deg']
	para_soc = site_inputs['para_soc'].to(device)

	#---------------------------------------------------------
	# sic para
	#---------------------------------------------------------
	forcing_cycle_num = 1
	dust_start_year = 9999
	diss_rate_ref_dust = 3*10**(-10.0) # unit mol m-2 s-1 

	para_sic_fix = {
		'site_lon': site_lon,
		'site_lat': site_lat,
		'site_loc': site_loc,
		'cec_proton': 2,
		'soa_proton': 1, 
		'charge_neutral': False, 
		'parent_weathering': True,
		'radius_mean_parent': 50, # unit mu m
		'forcing_cycle_num': forcing_cycle_num,
		'dust_start_year': dust_start_year,
		'amendment_rate': dust_amount, # unit tons ha-1 per application
		'is_repeat_app': is_repeat_app,
		'diss_rate_ref_dust': diss_rate_ref_dust, # unit mol m-2 s-1 
		'diss_order_dust': 0.4,
		'active_energy_dust': 53,
		'bio_cation_turnover_time': 10**20, # 365*60, # days, default 10**5 for oxisol, 10**4 for alfisol, 3*10**4 for ultisol
		'co2_dilute_factor': 20,
		'time_downscale': 1,
		'para_prior_version': para_prior_version, # prior range 
		'ppn_limit': 1, # limitation from chemcial inhibiiton from precipitation
		'om_association': 0, # whether there is organo-mineral association
		'is_use_solid_mineral': is_use_solid_mineral,
		'climate_forcing_type': 0, # 0 for annual mean, 1 for pre-industry variation, 2 for climate change
		'dust_end_year': 9999,
		'dust_type': dust_type, # silicate
		'nitrate_ratio': torch.tensor(0, dtype = torch.float64), # , requires_grad = True), # fraction of nitrification out of total N fertilizer 

		# 'co2_available_scalar': torch.tensor(1, dtype = torch.float64) # , requires_grad = True) # an adjusting scalar to compensate coarse time resoltion in CO2 dynamics
		'water_limit_steepness': torch.tensor(10, dtype = torch.float64), # steepness in water limitation function
		'water_limit_midpoint': torch.tensor(0.335, dtype = torch.float64), # midpoint in water limitation function. when 0.335, the mean limitation over 20 years is 0.5
		'is_use_moisture_scalar': 0, # 0 (as in the submitted results): no moisture effect on weathering progress; 1: soil-moisture scalar (Eq. 24) multiplies the parent/dust weathering progress (Eq. 25)
		'rain_limit_steepness': torch.tensor(1, dtype = torch.float64), # steepness in preci limitation function
		'rain_limit_midpoint': torch.tensor(3, dtype = torch.float64), # midpoint in preci limitation function. unit mm day-1
		'active_energy_parent': torch.tensor(0, dtype = torch.float64), # ref Brantley et al Science, theoretical values for soils: 22 kJ (mol)-1 at global scale, the value 38 correslponds to a Q10 ~ 1.6

		'adjust_scalar_pco2': torch.tensor(1, dtype = torch.float64),
		'adjust_scalar_leach': torch.tensor(1, dtype = torch.float64),
		'adjust_scalar_langmuir': torch.tensor(1, dtype = torch.float64),
		'valid_beerling': False,
	}

	para_sic_map = {
		'al_3cat_ex': site_soil_chem_exal, 
		'ph_initial_nominal': site_soil_chem_ph, # 4.5 for oxisol, 6.0 for alfisol, 5.2 for ultisol
		'cec_total': site_soil_chem_cec, # cec_total, # total CEC 1.9 for oxisol, 10 for alfisol, 5.0 for ultisol
		'base_saturation': site_soil_chem_bs,
		'mineral_2cat_ex': site_soil_chem_exd2,
		'is_nitrogen_fertilizer': False,
		'site_soil_chem_no3_leach': site_soil_chem_no3_leach,
		'ex_sol_ratio_cation': site_ex_sol_ratio_cation,
		'parent_litho_type': site_litho_type,
		'parent_cdr': site_parent_cdr,
		'carbonate_storage': site_soil_chem_carbonate,
		'gibbsite_storage': site_soil_chem_gibbsite
	}

	para_sic_var = {
		'diss_rate_ref_parent': torch.tensor(3*10**(-12), dtype = torch.float64), # , requires_grad = True), # 9.0*10**(-12), # unit mol m-2 s-1 
		'co2_turnover_time': torch.tensor(2, dtype = torch.float64), # , requires_grad = True), # days4.7
		'm2_turnover_time': torch.tensor(270000, dtype = torch.float64), # , requires_grad = True), # fraction of biocarbonate leaching rate compared to cation leaching
		'leachate_turnover_time': torch.tensor(180, dtype = torch.float64), # , requires_grad = True), # 720, # days, default 30 for oxisol, 360 for alfisol, 60 for ultisol
		'al_turnover_time': torch.tensor(2100, dtype = torch.float64), # , requires_grad = True), # days, default 30 for oxisol, 360 for alfisol, 60 for ultisol
		'silica_turnover_time': torch.tensor(30, dtype = torch.float64), # , requires_grad = True) # days, default 30 for oxisol, 360 for alfisol, 60 for ultisol
		'soa_fraction': torch.tensor(0.5, dtype = torch.float64), # set as the same with base saturation
		'ex_sol_ratio_al': torch.tensor(1.25, dtype = torch.float64), 
		'ex_sol_ratio_d': torch.tensor(0, dtype = torch.float64),
		'diss_order_parent': torch.tensor(0, dtype = torch.float64), # 0.1
	}

	para_sic = {**para_sic_fix, **para_sic_var, **para_sic_map}

	#-----------------------------------------------
	# organic carbon analytical solution
	#-----------------------------------------------
	soc_warmup = torch.load(simu_data_dir + '/soc_warmup_' + str(site_loc) + '.pth')
	
	#-----------------------------------------------
	# optimized para values from BINN
	#-----------------------------------------------
	# binn_para_history = np.loadtxt(data_dir + opt_version + '/binn_para_history_site_' + str(site_loc) + '_force_' + str(isoil_type+1) + '.csv', delimiter = ',')
	binn_para_history = np.loadtxt(data_dir + opt_version + '/binn_para_history_site_' + str(site_loc) + '.csv', delimiter = ',')
	final_loss_loc = np.where(np.isnan(binn_para_history[:, 0]) == True)[0][0] - 1
	para_opt = binn_para_history[final_loss_loc, :]
	
	#-----------------------------------------------
	# formal simulation
	#-----------------------------------------------
	# # # example temperate 
	if forcing_loc == 'temperate':
		forcing_site_lon = -90
		forcing_site_lat = 41
		forcing_site_loc = 7294

	# find location for 1 deg forcing 
	forcing_site_loc_1deg = site_loc_1deg_of(data_path, forcing_site_lon, forcing_site_lat)

	forcing_data_soc_add, forcing_data_sic_add = fun_load_forcing(para_sic['climate_forcing_type'], data_path, cesm2_case_name_hat, forcing_site_loc_1deg, forcing_site_lon, forcing_site_lat, device)

	print(f'processing beerling site: soil {soil_type} forcing {forcing_loc} dust {dust_amount} nitrogen {nitrogen_fertilizer_input}')

	try:
		para_sic['dust_start_year'] = 5
		para_sic['amendment_rate'] = dust_amount
		para_sic['is_nitrogen_fertilizer'] = True
		para_sic['site_soil_chem_no3_leach'] = nitrogen_fertilizer_input
		para_sic['is_repeat_app'] = True
		para_sic['valid_beerling'] = True

		para_sic['active_energy_dust'] = 62
		para_sic['diss_order_dust'] = 0.48
		para_sic['ph_initial_nominal'] = 6.36

		# below are the tuned parameter
		para_sic['diss_rate_ref_dust'] =  diss_rate_ref_dust*adjust_scalar_kinetic
		para_sic['adjust_scalar_leach'] = adjust_scalar_leach
		para_sic['adjust_scalar_pco2'] = adjust_scalar_pco2

		transit_formal = fun_main(para_opt, para_sic, para_soc, \
					fig_dir, soc_warmup, \
					is_plotting, \
					is_nn_training, \
					forcing_data_soc_add, forcing_data_sic_add)

		#-----------------------------------------------
		# pH and potential CDR
		#-----------------------------------------------
		proton_stock = transit_formal['species_stock'][8, :, :].mean(axis = 0) # mol m-2
		water_stock = transit_formal['species_stock'][13, :, :].mean(axis = 0) # mol m-2
		ph_series = -torch.log10(proton_stock/(water_stock*18/1000)) # mol L-1

		stoichio_m_2cat = transit_formal['stoichio_element_dust'][[5, 6]].sum() # Ca + Mg
		cdr_potential_accu = torch.nansum(transit_formal['dust_consumed'], axis = (0, 1)).cumsum(dim = 0)*stoichio_m_2cat*2*44*10**4/10**6 # cumulative potential CDR, t CO2 ha-1

		sensitivity_result = np.stack((ph_series.detach().numpy(), cdr_potential_accu.detach().numpy()), axis = 1)
		np.savetxt(sensitivity_result_path, \
					sensitivity_result, delimiter = ',', header = 'ph_series,cdr_potential_accu', comments = '')

		if opt_simu == True:
			torch.save(transit_formal, f'{simu_data_dir}annual_mean/sensitivity_beerling_soil/transit_formal_beerling_soil_{soil_type}_{forcing_loc}_nitrogen_{nitrogen_fertilizer_input}_{dust_type}_{dust_amount}_kinetic_{adjust_scalar_kinetic}_leach_{adjust_scalar_leach}_pco2_{adjust_scalar_pco2}.pth')

		# beerling_pnas_cdr_pot_mean = np.array([0, 3.821523585, 6.752516667, 11.48819547, 10.5085952])

		# beerling_pnas_ph = np.array([[6.05, 6.40, 6.55, 7.05, 6.85],
		# 							[6.15, 6.65, 7.00, 7.20, 7.25],
		# 							[6.50, 6.80, 6.65, 6.65, 7.00],
		# 							[6.55, 6.60, 6.70, 7.15, 6.95],
		# 							[6.55, 6.65, 6.60, 7.05, 6.85]])
		# obs_cdr = beerling_pnas_cdr_pot_mean
		# obs_ph = np.mean(beerling_pnas_ph, axis=0)  # median across the 5 replicate sites, one value per year

		# nse_cdr = 1 - np.sum((obs_cdr - cdr_potential_accu.detach().numpy()[4:9])**2)/np.sum((obs_cdr - obs_cdr.mean())**2)
		# nse_ph = 1 - np.sum((obs_ph - ph_series.detach().numpy()[4:9])**2)/np.sum((obs_ph - obs_ph.mean())**2)

		# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
		# # Panel 1: CDR
		# ax1.plot(obs_cdr, label='observed')
		# ax1.plot(cdr_potential_accu.detach().numpy()[4:9], label='modeled')
		# ax1.set_ylabel('CDR')
		# ax1.set_xlabel('time step')
		# ax1.set_title(f'CDR (NSE = {nse_cdr:.3f})')
		# ax1.legend()

		# # Panel 2: pH
		# ax2.plot(obs_ph, label='observed')
		# ax2.plot(ph_series.detach().numpy()[4:9], label='modeled')
		# ax2.set_ylabel('pH')
		# ax2.set_xlabel('time step')
		# ax2.set_title(f'pH (NSE = {nse_ph:.3f})')
		# ax2.legend()

	except Exception as e:
			print(f'Error occurred at scalars for kinetics {adjust_scalar_kinetic} leach {adjust_scalar_leach} pco2 {adjust_scalar_pco2}:', e)
	# end is_default == 1:
	print(f'simulations finished: scalars for kinetics {adjust_scalar_kinetic} leach {adjust_scalar_leach} pco2 {adjust_scalar_pco2}')

#------------------------------------------------------------------------
# parallel computing
#------------------------------------------------------------------------
def worker(args):
	"""Wrapper function to unpack arguments and call worker_function"""
	return worker_function(*args)

def main():
	number_pairs = [
		(ikinetic, ileach, ipco2)
		for ikinetic in range(len(kinetic_scalar_list))
		for ileach in range(len(leach_scalar_list))
		for ipco2 in range(len(pco2_scalar_list))
	]

	with multiprocessing.Pool(processes=6) as pool:
		pool.map(worker, number_pairs)

if __name__ == '__main__':
	main()

#--------------------------------------
print('end of programme')
