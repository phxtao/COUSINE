# COUSINE (COUpled Soil Inorganic-organic carbon model for eNhanced wEathering)
# Copyright (c) 2026 Cornell University. All Rights Reserved. Cornell docket D-11800.
# Permission to use for educational, research and non-profit purposes only; see LICENSE for the terms,
# the commercial-licensing contact and the disclaimers. The concepts used in this work are patent pending.

import numpy as np
import torch
import matplotlib
# matplotlib.use('Agg')  # Use non-GUI backend

from matplotlib import pyplot as plt
from scipy.io import loadmat
from fun_site_inputs import load_site_aet # site-only DATAHUB
import mat73
import time

torch.set_default_dtype(torch.float64)
# torch.autograd.set_detect_anomaly(True)

#######################################################
# main function
#######################################################
def fun_main(para_proposal, para_sic, para_soc, \
			 fig_dir, soc_warmup, \
			 is_plotting, \
			 is_nn_training, \
			 forcing_data_soc, forcing_data_sic):

	para_prior_version = para_sic['para_prior_version']

	if para_prior_version == 0:
		para_sic['diss_rate_ref_parent']   = 10**(para_proposal[0]*(-11 - (-16)) + (-16))  # diss_rate_ref_parent, # unit mol m-2 s-1  # (para_proposal[0]*(-11 - (-15)) + (-15)) 
		para_sic['co2_turnover_time']      =     (para_proposal[1]*(5 - (0.1)) + (0.1)) # co2_turnover_time days
		para_sic['m2_turnover_time']       = 10**(para_proposal[2]*(6.5 - 2) + 2)  # m2_turnover_time
		para_sic['leachate_turnover_time'] = 10**(para_proposal[3]*(2.5 - 0.1) + 0.1) # leachate_turnover_time
		para_sic['al_turnover_time']       = 10**(para_proposal[4]*(6.5 - 2) + 2) # al_turnover_time
		para_sic['silica_turnover_time']   =      para_proposal[5]*(100 - 10) + 10 # al_turnover_time

		para_sic['soa_fraction']           =  para_proposal[6]*(1 - 0) + 0
		para_sic['ex_sol_ratio_al'] 	   =  para_proposal[7]*(3 - (-2)) + (-2) #      para_proposal[13]*(5 - (-2)) + (-2)
		para_sic['ex_sol_ratio_d']         =  para_proposal[8]*(4 - (-2)) + (-2)
		para_sic['diss_order_parent']      =  para_proposal[9]*(1 - (0)) + (0)

		# para_sic['diss_rate_ref_parent']   = 10**(-12)*(para_proposal[0]*(6 - 0) + 0)  # diss_rate_ref_parent, # unit mol m-2 s-1  # (para_proposal[0]*(-11 - (-15)) + (-15)) 
		# para_sic['co2_turnover_time']      =            para_proposal[1]*(5 - 0.0001) + 0.0001 # co2_turnover_time days
		# # para_sic['nitrate_ratio']          =      para_proposal[2]*(0.2 - (0.0)) + (0.0) # nitrate_ratio, fraction of nitrification out of total N fertilizer 
		# para_sic['m2_turnover_time']       = 10**3    *(para_proposal[2]*(300 - 3) + 3)  # 10**(para_proposal[2]*(6.5 - 3.5) + 3.5)  # m2_turnover_time
		# para_sic['leachate_turnover_time'] =            para_proposal[3]*(200 - 5) + 5 # leachate_turnover_time
		# para_sic['al_turnover_time']       = 10**3    *(para_proposal[4]*(5 - 0.5) + 0.5) # al_turnover_time

		# para_sic['soa_fraction']           =  para_proposal[5]*(1 - 0) + 0
		# para_sic['ex_sol_ratio_al'] 	   =  para_proposal[6]*(5 - (-2)) + (-2) #      para_proposal[13]*(5 - (-2)) + (-2)

		# para_sic['diss_order_parent']      =  para_proposal[7]*(1 - (0)) + (0) 
		# para_sic['active_energy_parent']   =  para_proposal[8]*(90 - 0) + 0 # ref Brantley et al Science, theoretical values for soils: 22 kJ (mol)-1 at global scale, the value 38 correslponds to a Q10 ~ 1.6

		# para_sic['silica_turnover_time']   = 30 # para_proposal[5]*(100 - 20) + 20 # silica_turnover_time
		
		# para_sic['water_limit_steepness']  =  10 # para_proposal[6]*(15 - 5) + 5 # steepness in water limitation function
		# para_sic['water_limit_midpoint']   =  0.335 # para_proposal[7]*(0.8 - 0.2) + 0.2  # midpoint in water limitation function. when 0.335, the mean limitation over 20 years is 0.5
		# para_sic['rain_limit_steepness']   =  1 # para_proposal[8]*(1.3 - 0.2) + 0.2 # steepness in preci limitation function
		# para_sic['rain_limit_midpoint']    =  3 # para_proposal[9]*(10 - 0) + 0 # midpoint in preci limitation function. unit mm day-1
		
		# para_sic['diss_order_parent']      =  0 # para_proposal[10]*(1 - (0)) + (0) 
		# para_sic['active_energy_parent']   =  0 # para_proposal[11]*(100 - 30) + 30 # ref Brantley et al Science, theoretical values for soils: 22 kJ (mol)-1 at global scale, the value 38 correslponds to a Q10 ~ 1.6
		# para_sic['soa_fraction']           =  0.5 # para_proposal[12]*(1 - 0) + 0
		# para_sic['ex_sol_ratio_al'] 	   =  1.25 # para_proposal[13]*(5 - (-2)) + (-2) #      para_proposal[13]*(5 - (-2)) + (-2)
		# para_sic['ex_sol_ratio_d']         =  0 # para_proposal[14]*(3 - (-1)) + (-1) #      para_proposal[14]*(4 - (-0.5)) + (-0.5)
		# para_sic['co2_adjust_scalar']      =  1 # para_proposal[15]*(2 - 0) + (0)

	if para_prior_version == 18:
		para_sic['diss_rate_ref_parent']   = 10**(para_proposal[0]*(-11 - (-16)) + (-16))  # diss_rate_ref_parent, # unit mol m-2 s-1  # (para_proposal[0]*(-11 - (-15)) + (-15)) 
		para_sic['co2_turnover_time']      = 10**(para_proposal[1]*(2 - (-2.5)) + (-2.5)) # co2_turnover_time days
		# para_sic['nitrate_ratio']          =      para_proposal[2]*(0.2 - (0.0)) + (0.0) # nitrate_ratio, fraction of nitrification out of total N fertilizer 
		para_sic['m2_turnover_time']       = 10**(para_proposal[2]*(6.5 - 2) + 2)  # m2_turnover_time
		para_sic['leachate_turnover_time'] = 10**(para_proposal[3]*(4.5 - 1) + 1) # leachate_turnover_time
		para_sic['al_turnover_time']       = 10**(para_proposal[4]*(6.5 - 2) + 2) # al_turnover_time
		para_sic['silica_turnover_time']   = 10**(para_proposal[5]*(5.0 - 1) + 1) # silica_turnover_time
		
		para_sic['water_limit_steepness']  =      para_proposal[6]*(15 - 5) + 5 # steepness in water limitation function
		para_sic['water_limit_midpoint']   =      para_proposal[7]*(0.8 - 0.2) + 0.2  # midpoint in water limitation function. when 0.335, the mean limitation over 20 years is 0.5
		para_sic['rain_limit_steepness']   =      para_proposal[8]*(1.3 - 0.2) + 0.2 # steepness in preci limitation function
		para_sic['rain_limit_midpoint']    =      para_proposal[9]*(10 - 0) + 0 # midpoint in preci limitation function. unit mm day-1
		para_sic['diss_order_parent']      =      para_proposal[10]*(1 - (-1)) + (-1) # 0.1
		para_sic['active_energy_parent']   =      para_proposal[11]*(85 - 5) + 5 # ref Brantley et al Science, theoretical values for soils: 22 kJ (mol)-1 at global scale, the value 38 correslponds to a Q10 ~ 1.6
		para_sic['soa_fraction']           =      para_proposal[12]*(1 - 0) + 0
		para_sic['ex_sol_ratio_al'] 	   =      para_proposal[13]*(4 - (-4)) + (-4)
		para_sic['ex_sol_ratio_d']         =      para_proposal[14]*(4 - (-0.5)) + (-0.5)
		para_sic['co2_adjust_scalar']      =      para_proposal[15]*(2 - 0) + (0)

	is_warmup = 0
	is_weathering = 1
	
	# parent simulation as baseline
	transit_formal_parent = fun_matrix_transit(para_prior_version, fig_dir, para_sic, para_soc, soc_warmup['cpool'], soc_warmup['soc_co2_layer'], \
		is_warmup, is_weathering, \
		is_plotting, \
		is_nn_training, \
		forcing_data_soc, forcing_data_sic)
		
	return transit_formal_parent 

#######################################################
# steady state simulation for clm5
#######################################################
def fun_matrix_soc_ss(para_soc, forcing_data_soc):
	#---------------------------------------------------
	# load forcing
	#---------------------------------------------------
	cesm2_simu_input_sum_cwd = forcing_data_soc['cesm2_simu_input_sum_cwd']
	cesm2_simu_input_sum_litter1 = forcing_data_soc['cesm2_simu_input_sum_litter1']
	cesm2_simu_input_sum_litter2 = forcing_data_soc['cesm2_simu_input_sum_litter2']
	cesm2_simu_input_sum_litter3 = forcing_data_soc['cesm2_simu_input_sum_litter3']
	cesm2_simu_altmax_last_year = forcing_data_soc['cesm2_simu_altmax_last_year']
	cesm2_simu_altmax = forcing_data_soc['cesm2_simu_altmax']
	cesm2_simu_nbedrock = forcing_data_soc['cesm2_simu_nbedrock']
	cesm2_simu_o_scalar = forcing_data_soc['cesm2_simu_o_scalar']
	cesm2_simu_n_scalar = forcing_data_soc['cesm2_simu_n_scalar']
	cesm2_simu_cellsand = forcing_data_soc['cesm2_simu_cellsand']
	cesm2_simu_soil_temperature = forcing_data_soc['cesm2_simu_soil_temperature']
	cesm2_simu_w_scalar = forcing_data_soc['cesm2_simu_w_scalar']
	
	device = para_soc.device
	
	#---------------------------------------------------
	# offical starting simulation
	#---------------------------------------------------
	use_beta = 1
	normalize_q10_to_century_tfunc = False
	global month_num, season_num, month_num, \
		kelvin_to_celsius, use_vertsoilc, npool, npool_vr, n_soil_layer, \
		days_per_year, secspday, days_per_month, days_per_season, days_per_month
		
	season_num = 4
	month_num = 12
	kelvin_to_celsius = 273.15
	use_vertsoilc = 1
	npool = 7
	n_soil_layer = 1
	one_layer_compensation = 0.2800000000
	npool_vr = npool*n_soil_layer 
	days_per_year = 365
	secspday = 24*60*60
	days_per_month = torch.tensor([[31, 30, 31, 28, 31, 30, 31, 31, 30, 31, 30, 31]]).transpose(0, 1).to(device)
	days_per_season = torch.tensor([[92, 89, 92, 92]]).transpose(0, 1).to(device)
	
	global max_altdepth_cryoturbation, max_depth_cryoturb
	max_altdepth_cryoturbation = 2
	max_depth_cryoturb = 3

	global dz, zisoi, zisoi_0, zsoi, dz_node
	# width between two interfaces
	dz = torch.tensor([2.000000000000000E-002, 4.000000000000000E-002, 6.000000000000000E-002, \
		8.000000000000000E-002, 0.120000000000000, 0.160000000000000, \
		0.200000000000000, 0.240000000000000, 0.280000000000000, \
		0.320000000000000, 0.360000000000000, 0.400000000000000, \
		0.440000000000000, 0.540000000000000, 0.640000000000000, \
		0.740000000000000, 0.840000000000000, 0.940000000000000, \
		1.04000000000000, 1.14000000000000, 2.39000000000000, \
		4.67553390593274, 7.63519052838329, 11.1400000000000, \
		15.1154248593737]).to(device)
	
	# depth of the interface
	zisoi = torch.tensor([2.000000000000000E-002, 6.000000000000000E-002, \
		0.120000000000000, 0.200000000000000, 0.320000000000000, \
		0.480000000000000, 0.680000000000000, 0.920000000000000, \
		1.20000000000000, 1.52000000000000, 1.88000000000000, \
		2.28000000000000, 2.72000000000000, 3.26000000000000, \
		3.90000000000000, 4.64000000000000, 5.48000000000000, \
		6.42000000000000, 7.46000000000000, 8.60000000000000, \
		10.9900000000000, 15.6655339059327, 23.3007244343160, \
		34.4407244343160, 49.5561492936897]).to(device)

	zisoi_0 = torch.tensor([0])

	# when using one layer, need to do:
	dz = dz + one_layer_compensation
	zisoi = zisoi + one_layer_compensation

	# depth of the node
	zsoi = (zisoi + torch.concat((zisoi_0, zisoi[:-1])))/2
	# depth between two node
	dz_node = zsoi - np.append(np.array([0]), zsoi[:-1], axis = 0)

	#---------------------------------------------------
	# define parameters to be optimised
	#---------------------------------------------------
	# diffusion (bioturbation) 10^(-4) (m2/yr)
	bio = para_soc[0]*(5*1e-4 - 3*1e-5) + 3*1e-5
	# cryoturbation 5*10^(-4) (m2/yr)
	cryo = para_soc[1]*(16*1e-4 - 3*1e-5) + 3*1e-5
	
	#  Q10 (unitless) 1.5
	q10 = para_soc[2]*(3 - 1.2) + 1.2
	# Q10 when forzen (unitless) 1.5
	fq10 = q10
	# parameters used in vertical discretization of carbon inputs 10 (metre)
	efolding = para_soc[3]*(1 - 0.0001) + 0.0001
	# turnover time of CWD (yr) 3.3333
	tau4cwd = para_soc[4]*(6 - 1) + 1
	# tau for metabolic litter (yr) 0.0541
	tau4l1 = para_soc[5]*(0.11 - 0.0001) + 0.0001
	# tau for cellulose litter (yr) 0.2041
	tau4l2 = para_soc[6]*(0.3 - 0.1) + 0.1
	# tau for lignin litter (yr)
	tau4l3 = tau4l2
	# tau for fast SOC (yr) 0.1370
	tau4s1 = para_soc[7]*(0.5 - 0.0001) + 0.0001
	# tau for slow SOC (yr) 5
	tau4s2 = para_soc[8]*(10 - 1) + 1
	# tau for passive SOC (yr) 222.222
	tau4s3 = para_soc[9]*(400 - 20) + 20
	# fraction from l1 to s2, 0.45
	fl1s1 = para_soc[10]*(0.8 - 0.1) + 0.1
	# fraction from l2 to s1, 0.5
	fl2s1 = para_soc[11]*(0.8 - 0.2) + 0.2
	# fraction from l3 to s2, 0.5
	fl3s2 = para_soc[12]*(0.8 - 0.2) + 0.2
	# fraction from s1 to s2, sand dependeted
	fs1s2 = para_soc[13]*(0.4 - 0.0001) + 0.0001
	# fraction from s1 to s3, sand dependeted
	fs1s3 = para_soc[14]*(0.1 - 0.0001) + 0.0001
	# fraction from s2 to s1, 0.42
	fs2s1 = para_soc[15]*(0.74 - 0.1) + 0.1
	# fraction from s2 to s3, 0.03
	fs2s3 = para_soc[16]*(0.1 - 0.0001) + 0.0001
	# fraction from s3 to s1, 0.45
	fs3s1 = para_soc[17]*(0.9 - 0.0001) + 0.0001
	# fraction from cwd to l2, 0.76
	fcwdl2 = para_soc[18]*(1 - 0.5) + 0.5
	
	# water scaling factor
	w_scaling = para_soc[19]*(5 - 0.0001) + 0.0001
	# beta to describe the shape of vertical profile
	# beta = 0.95
	beta = para_soc[20]*(0.9999 - 0.5) + 0.5
	
	#####################################################
	# steady state solutions
	#####################################################
	#---------------------------------------------------
	# steady state forcing
	#---------------------------------------------------
	input_vector_cwd_steady_state = cesm2_simu_input_sum_cwd.mean(axis = 1)
	input_vector_litter1_steady_state = cesm2_simu_input_sum_litter1.mean(axis = 1)
	input_vector_litter2_steady_state = cesm2_simu_input_sum_litter2.mean(axis = 1)
	input_vector_litter3_steady_state = cesm2_simu_input_sum_litter3.mean(axis = 1)
	altmax_lastyear_profile_steady_state = cesm2_simu_altmax_last_year.mean(axis = 1)
	altmax_current_profile_steady_state = cesm2_simu_altmax.mean(axis = 1)
	nbedrock_steady_state = cesm2_simu_nbedrock.mean(axis = 1).int()
	
	xio_steady_state = cesm2_simu_o_scalar.mean(axis = 2)
	xin_steady_state = cesm2_simu_n_scalar.mean(axis = 2)
	sand_vector_steady_state = cesm2_simu_cellsand.mean(axis = 2)
	soil_temp_profile_steady_state = cesm2_simu_soil_temperature.mean(axis = 2)
	soil_water_profile_steady_state = cesm2_simu_w_scalar.mean(axis = 2)
	
	#---------------------------------------------------
	# steady state env scalar
	#---------------------------------------------------
	xit = (torch.ones(n_soil_layer, month_num)*np.nan).to(device)
	xiw = (torch.ones(n_soil_layer, month_num)*np.nan).to(device)
	xio = xio_steady_state
	xin = xin_steady_state
	
	catanf_30 = catanf(torch.tensor(30.0).to(device))
	normalization_tref = torch.tensor(15).to(device)
	
	for imonth in range(month_num):
		# temperature related function xit
		# calculate rate constant scalar for soil temperature
		# assuming that the base rate constants are assigned for non-moisture
		# limiting conditions at 25 C.
		for ilayer in range(n_soil_layer):
			if soil_temp_profile_steady_state[ilayer, imonth] >= (0 + kelvin_to_celsius):
				xit[ilayer, imonth] = q10**((soil_temp_profile_steady_state[ilayer, imonth] - (kelvin_to_celsius + 25))/10)
			else:
				xit[ilayer, imonth] = q10**((273.15 - 298.15)/10)*(fq10**((soil_temp_profile_steady_state[ilayer, imonth] - (0 + kelvin_to_celsius))/10))
			# end if soil_temp_profile[ilayer, imonth] >= 0 + kelvin_to_celsius:
		# end for layer
		
		if normalize_q10_to_century_tfunc == True:
			# scale all decomposition rates by a constant to compensate for offset between original CENTURY temp func and Q10
			normalization_factor = (catanf(normalization_tref)/catanf_30) / (q10**((normalization_tref-25)/10))
			xit[:, imonth] = xit[:, imonth]*normalization_factor
		# end if normalize_q10_to_century_tfunc == True:
	# end for imonth
	xiw = soil_water_profile_steady_state*w_scaling
	xiw[xiw > 1] = 1

	#---------------------------------------------------
	# steady state tridiagnal matrix, A matrix, K matrix, fire matrix
	#---------------------------------------------------
	sand_vector = torch.mean(sand_vector_steady_state, axis = 1)
	# allocation matrix
	a_ma = a_matrix(fl1s1, fl2s1, fl3s2, fs1s2, fs1s3, fs2s1, fs2s3, fs3s1, fcwdl2, sand_vector)
	
	kk_ma_middle = (torch.zeros([npool_vr, npool_vr, month_num])*np.nan).to(device) 
	tri_ma_middle = (torch.zeros([npool_vr, npool_vr, month_num])*np.nan).to(device) 
	
	for imonth in range(month_num):
		# decomposition matrix
		monthly_xit = xit[:, imonth]
		monthly_xiw = xiw[:, imonth]
		monthly_xio = xio[:, imonth]
		monthly_xin = xin[:, imonth]

		# K matrix
		kk_ma_middle[:, :, imonth] = kk_matrix(monthly_xit, monthly_xiw, monthly_xio, monthly_xin, efolding, tau4cwd, tau4l1, tau4l2, tau4l3, tau4s1, tau4s2, tau4s3)
		# tri matrix
		monthly_nbedrock = nbedrock_steady_state[imonth]
		monthly_altmax_current_profile = altmax_current_profile_steady_state[imonth]
		monthly_altmax_lastyear_profile = altmax_lastyear_profile_steady_state[imonth]
		
		if n_soil_layer == 1:
			tri_ma_middle[:, :, imonth] = 0
	# end for imonth
	tri_ma = torch.mean(tri_ma_middle, axis = 2)
	kk_ma = torch.mean(kk_ma_middle, axis = 2)
	
	#---------------------------------------------------
	# steady state vertical profile, input allocation
	#---------------------------------------------------
	vertical_input = 1.0
	#---------------------------------------------------
	# steady state analytical solution of soc
	#---------------------------------------------------
	matrix_in = (torch.ones([npool_vr, 1])*np.nan).to(device) 
	# total input amount
	input_tot_cwd = torch.nansum(input_vector_cwd_steady_state)/days_per_year # (gc/m2/day)
	input_tot_litter1 = torch.nansum(input_vector_litter1_steady_state)/days_per_year # (gc/m2/day)
	input_tot_litter2 = torch.nansum(input_vector_litter2_steady_state)/days_per_year # (gc/m2/day)
	input_tot_litter3 = torch.nansum(input_vector_litter3_steady_state)/days_per_year # (gc/m2/day)
	
	# redistribution by beta
	matrix_in[0*n_soil_layer:1*n_soil_layer, 0] = input_tot_cwd*vertical_input/dz[0:n_soil_layer] # litter input gc/m3/day
	matrix_in[1*n_soil_layer:2*n_soil_layer, 0] = input_tot_litter1*vertical_input/dz[0:n_soil_layer]
	matrix_in[2*n_soil_layer:3*n_soil_layer, 0] = input_tot_litter2*vertical_input/dz[0:n_soil_layer]
	matrix_in[3*n_soil_layer:4*n_soil_layer, 0] = input_tot_litter3*vertical_input/dz[0:n_soil_layer]
	matrix_in[4*n_soil_layer:npool*n_soil_layer, 0] = 0
	
	# analytical solution of soc pools
	try:
		# torch 1.7
		# cpool_steady_state = torch.solve((-matrix_in), (torch.matmul(a_ma, kk_ma)-tri_ma)).solution
		# torch 1.11
		cpool_steady_state = torch.linalg.solve((torch.matmul(a_ma, kk_ma)-tri_ma), (-matrix_in))
	except:
		print("Solving failed")
		
		# cpool_steady_state = torch.linalg.lstsq((torch.matmul(a_ma, kk_ma)-tri_ma), (-matrix_in)).solution 
		cpool_steady_state = (torch.ones([npool*n_soil_layer, 1])*(-1.0)).to(device)*torch.sum(para_soc)/torch.sum(para_soc)
	# end try
	# cpool_steady_state = (torch.ones([npool*n_soil_layer, 1])*(-9999.)).to(device)*torch.sum(para_soc)/torch.sum(para_soc) 
	# end try
	cpool_steady_state[cpool_steady_state < 0] = 0
	# total organic carbon stock at each layer
	soc_layer = cpool_steady_state.reshape(npool, n_soil_layer).t().sum(axis = 1) # unit gC/m3
	
	# soil respiration
	r_vector = -torch.sum(a_ma, axis = 0).reshape(npool*n_soil_layer, 1)
	r_vector[0:n_soil_layer] = 0

	respired_cpool_steady_state = r_vector*torch.diag(kk_ma).reshape(npool*n_soil_layer, 1)*cpool_steady_state # unit: gc m-3 day-1 
	# if soc_layer[-1] > soc_layer[0]:
	# 	soc_layer =  (torch.ones(n_soil_layer)*(-9999.*3))
	# # end if soc_layer[-1] > soc_layer[0]:
	#---------------------------------------------------
	# outcome
	#---------------------------------------------------
	outcome = {'cpool': cpool_steady_state, 'resp': respired_cpool_steady_state}
	return outcome

# end def fun_matrix_soc_ss

#####################################################
# transient simulations
#####################################################
def fun_matrix_transit(para_prior_version, fig_dir, para_sic, para_soc, initial_cpool, initial_soc_co2, \
	is_warmup, is_weathering, \
	is_plotting, \
	is_nn_training, \
	forcing_data_soc, forcing_data_sic):

	device = para_soc.device

	#---------------------------------------------------
	# structure configuration
	#---------------------------------------------------
	use_beta = 1
	normalize_q10_to_century_tfunc = False
	global month_num, season_num, month_num, \
		kelvin_to_celsius, use_vertsoilc, npool, npool_vr, n_soil_layer, \
		days_per_year, secspday, days_per_month, days_per_season, days_per_month
		
	season_num = 4
	month_num = 12
	kelvin_to_celsius = 273.15
	use_vertsoilc = 1
	npool = 7
	n_soil_layer = 1
	one_layer_compensation = 0.2800000000
	npool_vr = npool*n_soil_layer
	days_per_year = 365
	secspday = 24*60*60
	days_per_month = torch.tensor([[31, 30, 31, 28, 31, 30, 31, 31, 30, 31, 30, 31]]).transpose(0, 1).to(device)
	days_per_season = torch.tensor([[92, 89, 92, 92]]).transpose(0, 1).to(device)
		
	global max_altdepth_cryoturbation, max_depth_cryoturb
	max_altdepth_cryoturbation = 2
	max_depth_cryoturb = 3

	global dz, zisoi, zisoi_0, zsoi, dz_node
	# width between two interfaces
	dz = torch.tensor([2.000000000000000E-002, 4.000000000000000E-002, 6.000000000000000E-002, \
		8.000000000000000E-002, 0.120000000000000, 0.160000000000000, \
		0.200000000000000, 0.240000000000000, 0.280000000000000, \
		0.320000000000000, 0.360000000000000, 0.400000000000000, \
		0.440000000000000, 0.540000000000000, 0.640000000000000, \
		0.740000000000000, 0.840000000000000, 0.940000000000000, \
		1.04000000000000, 1.14000000000000, 2.39000000000000, \
		4.67553390593274, 7.63519052838329, 11.1400000000000, \
		15.1154248593737]).to(device)
	
	# depth of the interface
	zisoi = torch.tensor([2.000000000000000E-002, 6.000000000000000E-002, \
		0.120000000000000, 0.200000000000000, 0.320000000000000, \
		0.480000000000000, 0.680000000000000, 0.920000000000000, \
		1.20000000000000, 1.52000000000000, 1.88000000000000, \
		2.28000000000000, 2.72000000000000, 3.26000000000000, \
		3.90000000000000, 4.64000000000000, 5.48000000000000, \
		6.42000000000000, 7.46000000000000, 8.60000000000000, \
		10.9900000000000, 15.6655339059327, 23.3007244343160, \
		34.4407244343160, 49.5561492936897]).to(device)

	zisoi_0 = torch.tensor([0])

	# when using one layer, need to do:
	dz = dz + one_layer_compensation
	zisoi = zisoi + one_layer_compensation

	# depth of the node
	zsoi = (zisoi + torch.concat((zisoi_0, zisoi[:-1])))/2
	# depth between two node
	dz_node = zsoi - np.append(np.array([0]), zsoi[:-1], axis = 0)
	
	# soil volume at different depth
	soil_vol_z = dz[0:n_soil_layer]*1 # unit m^3 

	#---------------------------------------------------
	# load forcing
	#---------------------------------------------------
	cesm2_simu_input_sum_cwd = forcing_data_soc['cesm2_simu_input_sum_cwd']
	cesm2_simu_input_sum_litter1 = forcing_data_soc['cesm2_simu_input_sum_litter1']
	cesm2_simu_input_sum_litter2 = forcing_data_soc['cesm2_simu_input_sum_litter2']
	cesm2_simu_input_sum_litter3 = forcing_data_soc['cesm2_simu_input_sum_litter3']
	cesm2_simu_altmax_last_year = forcing_data_soc['cesm2_simu_altmax_last_year']
	cesm2_simu_altmax = forcing_data_soc['cesm2_simu_altmax']
	cesm2_simu_nbedrock = forcing_data_soc['cesm2_simu_nbedrock'].int()
	cesm2_simu_o_scalar = forcing_data_soc['cesm2_simu_o_scalar']
	cesm2_simu_n_scalar = forcing_data_soc['cesm2_simu_n_scalar']
	cesm2_simu_cellsand = forcing_data_soc['cesm2_simu_cellsand']
	cesm2_simu_soil_temperature = forcing_data_soc['cesm2_simu_soil_temperature']
	cesm2_simu_w_scalar = forcing_data_soc['cesm2_simu_w_scalar']
	cesm2_simu_gpp = forcing_data_soc['cesm2_simu_gpp']

	cesm2_simu_eff_porosity = forcing_data_sic['cesm2_simu_eff_porosity']
	cesm2_simu_dsl = forcing_data_sic['cesm2_simu_dsl']
	cesm2_simu_h2osoi = forcing_data_sic['cesm2_simu_h2osoi']
	# cesm2_simu_prec10 = forcing_data_sic['cesm2_simu_prec10'] # unit mm/day
	cesm2_simu_rain = forcing_data_sic['cesm2_simu_rain']*3600*24 # unit mm/s --> mm/day
	cesm2_simu_watfc = forcing_data_sic['cesm2_simu_watfc']
	cesm2_simu_watsat = forcing_data_sic['cesm2_simu_watsat']
	cesm2_simu_aet = forcing_data_sic['cesm2_simu_aet'] # unit mm/month

	# construct water driver, unit: mm/day
	# cesm_simu_water_driver = cesm2_simu_prec10 - cesm2_simu_aet/days_per_month.reshape(month_num, 1).repeat(1, cesm2_simu_aet.shape[1])
	# NOTE to be comfirmed
	# cesm2_simu_water_frac = cesm2_simu_watsat - cesm2_simu_eff_porosity # water volumn fraction in the soil
	# water_frac_min = cesm2_simu_water_frac[cesm2_simu_water_frac > 0.0].min()
	# cesm2_simu_water_frac[cesm2_simu_water_frac <= water_frac_min] = water_frac_min 	
	
	#---------------------------------------------------
	# sic parameter to be optimised
	#---------------------------------------------------
	is_nitrogen_fertilizer = para_sic['is_nitrogen_fertilizer']
	site_soil_chem_no3_leach = para_sic['site_soil_chem_no3_leach'] # unit: kg N ha-1 yr-1
	co2_dilute_factor = para_sic['co2_dilute_factor']
	time_downscale = para_sic['time_downscale']# 24 # 4 for time resolution at 6hr
	cec_proton = para_sic['cec_proton'] # 0.0
	soa_proton = para_sic['soa_proton'] # 0.0
	is_use_solid_mineral = para_sic['is_use_solid_mineral']
	
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
	# species_list =               ['Al+++', 'Al(OH)++', 'Al(OH)4-', 'H4SiO4', 'H3SiO4-', 'M+', 'M++', 'OH-', 'H+', 'CO2(aq)', 'CO2(g)', 'HCO3-', 'CO3--', 'H2O', 'MCO3', 'Al(OH)3', 'SM', 'SH-',              'SOAM0.5', 'SOAH']
	# species_charge = torch.tensor([3,        2,          -1,         0,        -1,        1,    2,     -1,    1,    0,         0,        -1,      -2,     0,     0,      0,         0,    -(2 - cec_proton), 0,     -(1 - soa_proton)])
	# species_list =               ['Al+++', 'Al(OH)++', 'Al(OH)4-', 'H4SiO4', 'H3SiO4-', 'M+', 'M++', 'OH-', 'H+', 'CO2(aq)', 'CO2(g)', 'HCO3-', 'CO3--', 'H2O', 'MCO3', 'Al(OH)3', 'SM', 'SH-']
	# species_charge = torch.tensor([3,        2,          -1,         0,        -1,        1,    2,     -1,    1,    0,         0,        -1,      -2,     0,     0,      0,         0,    -(2 - cec_proton)])

	species_num = len(species_list)
	gas_const = 8.3145 # unit J (mol)-1 K-1
	water_density = 10**6 # unit g/m3 1m3 water weights 10^6g, temperature has minor influence
	avogadro_const = 6.02214076*10**(23) # unit (mol)-1
	henry_const_ref = 3.4*10**(-2)*1000/101325 # unit: mol L-1 atm-1 --> mol m-3 Pa-1
	temp_ref = 25+273.15 # unit K
	atm_pressure = 101325 # unit Pa
	
	trivial_con = 10**(-20)

	water_limit_steepness = para_sic['water_limit_steepness']
	water_limit_midpoint = para_sic['water_limit_midpoint']
	is_use_moisture_scalar = para_sic.get('is_use_moisture_scalar', 0) # 0 (default, as in the submitted results): no moisture effect on weathering progress; 1: soil_moisture_scalar (Eq. 24) multiplies the parent and dust weathering progress (Eq. 25)

	rain_limit_steepness = para_sic['rain_limit_steepness']
	rain_limit_midpoint = para_sic['rain_limit_midpoint']
	co2_available_scalar = 1 # para_sic['co2_available_scalar']
	#---------------------------------------------------
	#Pco2 constriant 
	# from A world model of soil carbon dioxide
	# George A. Brook, Michael E. Folkoff, Elgene O. Box
	# First published: January/February 1983 https://doi.org/10.1002/esp.3290080108
	#---------------------------------------------------
	site_lat = para_sic['site_lat']
	annual_mean_aet = cesm2_simu_aet.sum(axis = 0).mean() # unit mm yr-1
	annual_mean_h2osoi = cesm2_simu_h2osoi.sum(axis = 0).mean() # volumetric water content unit v/v
	annual_surface_temp = cesm2_simu_soil_temperature[:, :, 0].mean() # unit K
	pco2_mujalli = 10**((np.exp(-3.0*annual_mean_h2osoi + -0.25/annual_mean_h2osoi))/(0.09 + np.exp(-0.34*(annual_surface_temp - 273.15))) + (-3.4))*atm_pressure # unit Pa
	# annual_surface_temp = np.arange(0+273.15, 20+273.15, 0.1)
	# plt.plot(annual_surface_temp, 10**((np.exp(-3.0*0.3 + -0.25/0.3))/(0.09 + np.exp(-0.34*(annual_surface_temp - 273.15))) + (-3.4))*atm_pressure)
	# annual_mean_h2osoi = np.arange(0.1, 0.4, 0.01)
	# plt.plot(annual_mean_h2osoi, 10**((np.exp(-3.0*annual_mean_h2osoi + -0.25/annual_mean_h2osoi))/(0.09 + np.exp(-0.34*(288.15 - 273.15))) + (-3.4))*atm_pressure)

	pco2_brook = 10**(-3.47 + 2.09*(1 - np.exp(-0.00172*annual_mean_aet)))*atm_pressure # unit Pa
	pco2_constraint = np.max((400/10**6*atm_pressure, pco2_brook)) # unit Pa (convert 400 ppm to Pa)

	# soil_co2_gas_constraint = co2_available_scalar*(pco2_constraint/gas_const/cesm2_simu_soil_temperature[:, :, 0:5].mean()*cesm2_simu_eff_porosity[:, :, 0:5].mean())*0.3 # unit mol m-2, 0.3 is for soil depth (m) 

	# # assume summer time to representing growing season
	if site_lat > 0:
		soil_temp_grow_season = cesm2_simu_soil_temperature[:, 5:8, :].mean()
		soil_porosity_grow_season = cesm2_simu_eff_porosity[:, 5:8, :].mean() # fraction
	else:
		soil_temp_grow_season = cesm2_simu_soil_temperature[:, (0, 1, 11), :].mean()
		soil_porosity_grow_season = cesm2_simu_eff_porosity[:, (0, 1, 11), :].mean() # fraction
	soil_co2_gas_constraint = co2_available_scalar*(pco2_constraint/gas_const/soil_temp_grow_season*soil_porosity_grow_season)*0.3 # unit mol m-2, 0.3 is for soil depth (m) 

	#---------------------------------------------------
	# soc parameters to be optimised
	#---------------------------------------------------
	ratio_ra_to_rh = 0.64

	# diffusion (bioturbation) 10^(-4) (m2/yr)
	bio = para_soc[0]*(5*1e-4 - 3*1e-5) + 3*1e-5
	# cryoturbation 5*10^(-4) (m2/yr)
	cryo = para_soc[1]*(16*1e-4 - 3*1e-5) + 3*1e-5
	#  Q10 (unitless) 1.5
	q10 = para_soc[2]*(3 - 1.2) + 1.2
	# Q10 when forzen (unitless) 1.5
	fq10 = q10
	# parameters used in vertical discretization of carbon inputs 10 (metre)
	efolding = 9999 # para_soc[3]*(1 - 0.0001) + 0.0001
	# turnover time of CWD (yr) 3.3333
	tau4cwd = para_soc[4]*(6 - 1) + 1
	# tau for metabolic litter (yr) 0.0541
	tau4l1 = para_soc[5]*(0.11 - 0.0001) + 0.0001
	# tau for cellulose litter (yr) 0.2041
	tau4l2 = para_soc[6]*(0.3 - 0.1) + 0.1
	# tau for lignin litter (yr)
	tau4l3 = tau4l2
	# tau for fast SOC (yr) 0.1370
	tau4s1 = para_soc[7]*(0.5 - 0.0001) + 0.0001
	# tau for slow SOC (yr) 5
	tau4s2 = para_soc[8]*(10 - 1) + 1
	# tau for passive SOC (yr) 222.222
	tau4s3 = para_soc[9]*(400 - 20) + 20
	# fraction from l1 to s2, 0.45
	fl1s1 = para_soc[10]*(0.8 - 0.1) + 0.1
	# fraction from l2 to s1, 0.5
	fl2s1 = para_soc[11]*(0.8 - 0.2) + 0.2
	# fraction from l3 to s2, 0.5
	fl3s2 = para_soc[12]*(0.8 - 0.2) + 0.2
	# fraction from s1 to s2, sand dependeted
	fs1s2 = para_soc[13]*(0.4 - 0.0001) + 0.0001
	# fraction from s1 to s3, sand dependeted
	fs1s3 = para_soc[14]*(0.1 - 0.0001) + 0.0001
	# fraction from s2 to s1, 0.42
	fs2s1 = para_soc[15]*(0.74 - 0.1) + 0.1
	# fraction from s2 to s3, 0.03
	fs2s3 = para_soc[16]*(0.1 - 0.0001) + 0.0001
	# fraction from s3 to s1, 0.45
	fs3s1 = para_soc[17]*(0.9 - 0.0001) + 0.0001
	# fraction from cwd to l2, 0.76
	fcwdl2 = para_soc[18]*(1 - 0.5) + 0.5
	
	# water scaling factor
	w_scaling = para_soc[19]*(5 - 0.0001) + 0.0001
	# beta to describe the shape of vertical profile
	# beta = 0.95
	beta = para_soc[20]*(0.9999 - 0.5) + 0.5

	catanf_30 = catanf(torch.tensor(30.0).to(device))
	normalization_tref = torch.tensor(15).to(device)
		
	#####################################################
	# transient simulation
	#####################################################
	forcing_year_num = cesm2_simu_input_sum_cwd.shape[1]
	forcing_cycle_num = para_sic['forcing_cycle_num']
	dust_start_year = para_sic['dust_start_year']
	dust_end_year = para_sic['dust_end_year']
	is_repeat_app = para_sic['is_repeat_app']
	nitrate_ratio =  para_sic['nitrate_ratio']
	om_association_index = torch.tensor(para_sic['om_association']).to(torch.float)
	co2_turnover_time = 1*(1 + para_sic['co2_turnover_time'])
	adjust_scalar_langmuir = para_sic['adjust_scalar_langmuir']
	leachate_turnover_time = para_sic['leachate_turnover_time']
	m2_turnover_time = para_sic['m2_turnover_time']
	al_turnover_time = para_sic['al_turnover_time']
	silica_turnover_time = para_sic['silica_turnover_time']
	bio_cation_turnover_time = para_sic['bio_cation_turnover_time']
	if is_warmup == 1:
		transit_year_size = 2000
	else:
		transit_year_size = forcing_cycle_num*forcing_year_num
	# end if is_warmup == 1

	# initial_cpool = cpool_steady_state
	# total organic carbon stock at each layer
	soc_layer = initial_cpool.reshape(npool, n_soil_layer).t().sum(axis = 1) # unit gC/m3
	total_c_stock_initial = torch.sum(soc_layer*dz[0:n_soil_layer]) # unit gc/m2
	ss_threshold = 1 # unit gc/m2/yr
	total_c_stock_record = total_c_stock_initial
	
	soc_stock_record = torch.ones(transit_year_size, month_num)*torch.nan 
	soc_resp_tot_record = torch.ones(transit_year_size, month_num)*torch.nan  
	co2_mix_record = torch.ones(transit_year_size, month_num, 2)*torch.nan  
	soc_co2_tot_record = torch.ones(transit_year_size, month_num)*torch.nan
	soc_co2_layer_record = torch.ones(n_soil_layer, month_num, transit_year_size)*torch.nan  
	
	species_stock = torch.nan*torch.ones(species_num, month_num, transit_year_size)
	leached_species_stock = torch.nan*torch.ones(species_num, month_num, transit_year_size)
	delta_species_stock = torch.nan*torch.ones(10, species_num, month_num, transit_year_size)
	leached_ion_stock = torch.zeros(species_num, month_num, transit_year_size)
	parent_species_stock = torch.nan*torch.ones(species_num, month_num, transit_year_size)
	co2_limitation = torch.nan*torch.ones(month_num, transit_year_size)
	dust_remained = torch.nan*torch.ones(n_soil_layer, month_num, transit_year_size)
	dust_consumed = torch.nan*torch.ones(n_soil_layer, month_num, transit_year_size)
	dust_consumed_parent = torch.nan*torch.ones(n_soil_layer, month_num, transit_year_size)
	total_c_change_rate = torch.tensor(99999)
	leached_species_accu = torch.zeros(species_num)
	parent_species_accu = torch.zeros(species_num)
	for iyear in torch.arange(0, transit_year_size):
		# if is_nn_training == 0:
		if is_plotting == 1:
			print('processing year ', str(iyear.numpy()))
		year_count = torch.remainder(iyear, forcing_year_num)
		
		if is_warmup == 1 and year_count == 0 and iyear != 0:
			total_c_change_rate = torch.abs((total_c_stock - total_c_stock_record)/forcing_year_num)
			if total_c_change_rate < ss_threshold: # unit 1gc/m2/year
				break
			else:
				total_c_stock_record = total_c_stock
			# end if total_c_change_rate < ss_threshold: 
		# end if year_count == 0 && iyear ~= 0:
		
		# env. scalar
		xit = torch.ones(n_soil_layer)*torch.nan
		xiw = torch.ones(n_soil_layer)*torch.nan
		
		for imonth in torch.arange(0, month_num):
			if iyear == 0 and imonth == 0:
				month_water_frac_history = cesm2_simu_h2osoi[:, imonth, year_count]
			else:
				month_water_frac_history = month_water_frac
			# load sic related forcing
			month_porosity = cesm2_simu_eff_porosity[:, imonth, year_count]
			month_water_frac = cesm2_simu_h2osoi[:, imonth, year_count]
			month_soil_temp = cesm2_simu_soil_temperature[:, imonth, year_count]
			month_rain = cesm2_simu_rain[imonth, year_count]
			
			# find if this the start of simulation
			###########################################################################
			# initial consitions
			###########################################################################
			#------------------------------------------
			# weahtering initial state setting
			# NOTE all concentration is with regard soil volume
			#------------------------------------------
			concentration_converter = 10**6
			if imonth == 0 and iyear == 0:
				current_cpool = initial_cpool
				
				charge_neutral = para_sic['charge_neutral'] # True
				parent_weathering = para_sic['parent_weathering'] # True
				radius_mean_parent = para_sic['radius_mean_parent'] # # unit mu m
				parent_litho_type = para_sic['parent_litho_type'] 
				parent_cdr = para_sic['parent_cdr'] 
				ppn_limit_index = torch.tensor(para_sic['ppn_limit']).to(torch.float)
				dust_type = para_sic['dust_type'] 
				# gamma_leach_max = para_sic['gamma_leach'] 
				# print('charge neutral:', str(charge_neutral))
				#---------------------------------------------------
				# temperature and soil shielding effrct for parent CDR
				#----------------------------------------------------
				scalar_soil_shield = 0.1
				# Hartmann et al: Global chemical weathering and associated P-release— The role of lithology, temperature and soil properties
				litho_Ea_vector = {
					#  1;        "su" Unconsolidated sediments
					1: 60, # (AD+SU)/2 in Hartmann 2009
					#  2;        "vb" Basic volcanic rocks
					2: 50, # refer to VB (Basic and intermediate volcanic) in Hartmann 2009
					#  3;        "ss" Siliciclastic sedimentary rocks
					3: 60, # refer to SS (Siliciclastic sedimentary rocks) in Hartmann 2009
					#  4;        "pb" Basic plutonic rocks
					4: 50, # refer to VB (Basic and intermediate volcanic) in Hartmann 2009, source is Hartmann et al 2009
					#  5;        "sm" Mixed sedimentary rocks
					5: 60, # refer to SM in Hartmann 2009, *0.5 due to carbonate rock 
					#  6;        "sc" Carbonate sedimentary rocks
					6: 0, # refer to SC in Hartmann et al 2009, *0.5 due to carbonate rock 
					#  7;        "va" Acid volcanic rocks
					7: 50, # VA in Hartmann 2009
					#  8;        "mt" Metamorphics
					8: 60, # MT in hartmann 2009
					#  9;        "pa" Acid plutonic rocks
					9: 50, # PA in Hartmann 2009
					#  10;       "vi" Intermediate volcanic rocks
					10: 50, # VB in Hartmann 2009
					#  11;       "wb" Water Bodies
					11: 0,
					#  12;       "py" Pyroclastics
					12: 46, # PY in Hartmann 2009
					#  13;       "pi" Intermediate plutonic rocks
					13: 50, # PI = (PB + PA)/2
					#  14;       "ev" Evaporites
					14: 0,
					#  15;       "nd" No Data
					15: 0,
					#  16;       "ig" Ice and Glaciers
					16: 0
				}

				parent_cdr_corrected = parent_cdr*np.exp(-litho_Ea_vector[int(parent_litho_type)]*1000/8.314*(1/annual_surface_temp - 1/284.15))*scalar_soil_shield # 284.15 is the mean temp in Japan, where the flux data from Hartmann et al 2009
				
				#-----------------------------------------------------
				# mineral/dust dissolution parameters
				#------------------------------------------------
				# oxides composition: SiO2 + Al2O3 + CaO + MgO + CaCO3 + MgCO3
				molar_weight_oxide = torch.tensor([60.09, 101.96, 56.08, 40.31, 100.09, 84.32])

				# Hartmann et al 2012: The geochemical composition of the terrestrial surface (without soils) and comparison with the upper continental crust
				oxide_composition_mass_list = {
					1: [52.30, 11.37, 10.60,  2.79, 0,  0 ], # 1 Unconsolidated sediment (su) 
					2: [51.26, 16.02,  8.78,  5.87, 0,  0 ], # 2 Basic volcanics (vb) 
					3: [60.32, 12.98,  4.73,  2.51, 0,  0 ], # 3 Siliciclastic sedimentary rocks (ss)
					4: [46.39,  9.89,  7.34, 19.49, 0,  0 ], # 4 Basic plutonics (pb)
					5: [52.30, 11.37, 10.60,  2.79, 0,  0 ], # 5 Mixed sedimentary rocks (sm)
					6: [ 7.52,  1.94, 43.86,  4.32, 0,  0 ], # 6 Carbonate sedimentary rocks (sc) 
					7: [72.70, 13.25,  1.14,  0.39, 0,  0 ], # 7 Acid Volcanic Rocks (VA) 
					8: [65.91, 15.30,  3.79,  1.93, 0,  0 ], # 8 Metamorphics (mt)
					9: [67.78, 15.25,  3.17,  1.40, 0,  0 ], # 9 Acid plutonics (pa)
					10:[51.26, 16.02,  8.78,  5.87, 0,  0 ], # 10 Intermediate volcanics (vi) REFER to vb 
					11:[ 0,     0,     0,     0,    0,  0 ], # 11 Water bodies (wb) 
					12:[51.26, 16.02,  8.78,  5.87, 0,  0 ], # 12 Pyroclastics (py): REFER to vb
					13:[57.09, 12.57,  5.26, 10.45, 0,  0 ], # 13 Intermediate plutonics (pi) REFER to (PA + PB)/2
					14:[0,      0,     0,     0,    0,  0 ], # 14 Evaporite (ev)
					15:[0,      0,     0,     0,    0,  0 ], # 15 Non-defined (nd)
					16:[0,      0,     0,     0,    0,  0 ], # 16 Ice and glaciers (ig)
				}
				
				co2_composition_mass_list = {
					1:  8.82, # 1 Unconsolidated sediment (su) refer Durr SU/AD
					2:  0.09, # 2 Basic volcanics (vb) 
					3:  3.90, # 3 Siliciclastic sedimentary rocks (ss)
					4:  0.27, # 4 Basic plutonics (pb)
					5:  8.82, # 5 Mixed sedimentary rocks (sm)
					6: 36.74, # 6 Carbonate sedimentary rocks (sc) 
					7:  0.08, # 7 Acid Volcanic Rocks (VA) 
					8:  0.07, # 8 Metamorphics (mt)
					9:  0.07, # 9 Acid plutonics (pa)
					10: 0.09, # 10 Intermediate volcanics (vi)
					11: 0,    # 11 Water bodies (wb) 
					12: 0.09, # 12 Pyroclastics (py): 
					13: 0.17, # 13 Intermediate plutonics (pi) 
					14: 0, # 14 Evaporite (ev)
					15: 0, # 15 Non-defined (nd)
					16: 0, # 16 Ice and glaciers (ig)
				}
				if parent_litho_type != 11 and parent_litho_type < 14:
					molar_weight_caco3 = 100.0869
					molar_weight_mgco3 = 84.3139
					molar_weight_cao   = 56.0774
					molar_weight_mgo   = 40.3044
					molar_weight_co2   = 44.009
					
					parent_oxide_composition_mass = torch.tensor(oxide_composition_mass_list[parent_litho_type])
					parent_co2_mass = torch.tensor(co2_composition_mass_list[parent_litho_type])
					ca_mg_molar_ratio = (parent_oxide_composition_mass[2]/molar_weight_cao)/(parent_oxide_composition_mass[3]/molar_weight_mgo)

					parent_oxide_composition_mass[4] = parent_co2_mass*(ca_mg_molar_ratio/(ca_mg_molar_ratio + 1))/molar_weight_co2*molar_weight_caco3
					parent_oxide_composition_mass[5] = parent_co2_mass*(1/(ca_mg_molar_ratio + 1))                /molar_weight_co2*molar_weight_mgco3

					parent_oxide_composition_mass[2] = parent_oxide_composition_mass[2] - parent_co2_mass*(ca_mg_molar_ratio/(ca_mg_molar_ratio + 1))/molar_weight_co2*molar_weight_cao
					parent_oxide_composition_mass[3] = parent_oxide_composition_mass[3] - parent_co2_mass*(1/(ca_mg_molar_ratio + 1))                /molar_weight_co2*molar_weight_mgo

					parent_oxide_composition_mass = parent_oxide_composition_mass/parent_oxide_composition_mass.sum()*100 # corrected to the molar weight of 100g mol-1
					parent_oxide_composition_molar = parent_oxide_composition_mass/molar_weight_oxide
					
					# oxides composition: SiO2 + Al2O3 + CaO + MgO + CaCO3 + MgCO3
					stoichio_element_parent = torch.tensor([
						parent_oxide_composition_molar[1]*2, # Al
						parent_oxide_composition_molar[0]*1, # Si
						parent_oxide_composition_molar[0]*2 + parent_oxide_composition_molar[1]*3 + parent_oxide_composition_molar[2]*1 + 
						parent_oxide_composition_molar[3]*1 + parent_oxide_composition_molar[4]*3 + parent_oxide_composition_molar[5]*3, # O
						0, # H
						parent_oxide_composition_molar[4]*1 + parent_oxide_composition_molar[5]*1, # C
						parent_oxide_composition_molar[2]*1 + parent_oxide_composition_molar[4]*1, # Ca
						parent_oxide_composition_molar[3]*1 + parent_oxide_composition_molar[5]*1, # Mg
						0, # K
						0 # Na
						])
				else:
					#                                           Al     Si     O          H       C      Ca     Mg     K      Na 
					stoichio_element_parent =     torch.tensor([0.40,  0.83,  torch.nan, 0.0,    0.0,   0.33,  0.33,  0.00,  0.0  ]) # parent material as baseline
				
				# molar weitgh of elements gram/mol 
				#                            	              Al     Si     O          H       C      Ca     Mg     K      Na 
				molar_weight_element    =     	torch.tensor([26.98, 28.09, 16.00,    1.0079, 12.01, 40.08, 24.31, 39.10, 22.99])
				stoichio_element_olivine =    	torch.tensor([0.00,  1.00,  4.00,     0.0,    0.0,   0.0,   2.00,  0.00,  0.0  ]) # external application
				olivine_molar_weight = (molar_weight_element*stoichio_element_olivine).sum() # g mol-1
				
				if dust_type == 'olivine':
					stoichio_element_dust    =  torch.tensor([0.00,  1.00,  torch.nan, 0.0,    0.0,   0.0,   2.00,  0.0,  0.0  ]) # external application
					rock_log_k = 27.8626 # Mg2SiO4 + 4 H+ = SiO2 + 2 H2O + 2 Mg+2
					rock_delta_h_ref = -205.614 # kJ/mol # Calculated enthalpy of reaction Forsterite
				if dust_type == 'basalt':
					# Labradorite Ca0.6Na0.4Al1.6Si2.4O8  # assuming Anorthite 60% / Albite 40% 
					stoichio_element_dust    =  torch.tensor([1.60,  2.40,  torch.nan, 0.0,    0.0,   0.6,   0.00,  0.00,  0.40  ]) # external application					
					# Labradorite Ca0.6Na0.4Al1.6Si2.4O8 does not have Ksp
					# use Anorthite to approximate CaAl2(SiO4)2 + 8 H+ = Ca+2 + 2 Al+3 + 2 SiO2 + 4 H2O
					rock_log_k = 26.578 
					rock_delta_h_ref = -303.039 # kJ/mol
				if dust_type == 'lime':
					stoichio_element_dust    =  torch.tensor([0.00,  0.00,  torch.nan, 0.0,    1.0,   0.0,   1.00,  0.00,  0.00  ]) # external application
					rock_log_k = 1.8487 # CaCO3 + H+ = Ca+2 + HCO3-
					rock_delta_h_ref = -25.7149 # kJ/mol # Calculated enthalpy of reaction Calcite

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

				# (stoichio_m_1cat + 2*stoichio_m_2cat + 3*stoichio_al_3cat + 4*stoichio_silicon)/2 - stoichio_oxygen

				# parent molar weight
				parent_molar_weight = (molar_weight_element*stoichio_element_parent).sum() # g mol-1
				# dust molar weight
				dust_molar_weight = (molar_weight_element*stoichio_element_dust).sum() # g mol-1

				# all the t dust per area is eq to olivine (in terms of cations)
				tot_cation_stoichio_olivine = stoichio_element_olivine[[0]].sum()*3 + stoichio_element_olivine[[5, 6]].sum()*2 + stoichio_element_olivine[[7, 8]].sum()*1
				tot_cation_stoichio_dust = stoichio_element_dust[[0]].sum()*3 + stoichio_element_dust[[5, 6]].sum()*2 + stoichio_element_dust[[7, 8]].sum()*1
				olivine_cation_ratio = dust_molar_weight/olivine_molar_weight*tot_cation_stoichio_olivine/tot_cation_stoichio_dust

				# amendment rate: 20 t ha-1 time-1
				amendment_rate = olivine_cation_ratio*para_sic['amendment_rate']*10**6/10**4 # 40*10**6/10**4 # unit g m-2 time-1 (1 ha = 10000 m2)

				# from wikipidia, Basalt density is 2.9 g cm-3, consider using dense dust with porosity, we use 2.9 g cm-3 in this case
				dust_density = 2.8*10**6 # convert unit from g cm-3 to g m-3
				amendment_vol_frac = amendment_rate/dust_density # unit m3 dust per m2 soil per time
				# mineral fraction
				vol_frac_mineral = 1.0 # assuming 90% of the volume of dust is the assumed minerals to be dissoluted
				
				# application vertical distribution
				dust_max_layer = 5

				# kinetics constants
				# NOTE ad hoc setting
				diss_rate_ref_dust = para_sic['diss_rate_ref_dust']*3600 # 5*10**(-13)*3600 # now the unit is mol m-2 hr-1 10**(-10.06)*24*3600 # unit mol m-2 day-1
				
				diss_rate_ref_parent = para_sic['diss_rate_ref_parent']*3600 # 5*10**(-13)*3600 # now the unit is mol m-2 hr-1 10**(-10.06)*24*3600 # unit mol m-2 day-1
				diss_order_dust = para_sic['diss_order_dust'] # 0.8
				diss_order_parent = para_sic['diss_order_parent']
				ex_sol_ratio_cation = para_sic['ex_sol_ratio_cation']
				active_energy_parent = litho_Ea_vector[int(parent_litho_type)]*10**3 # para_sic['active_energy_parent']*10**3  # 38*10**3 # ref Brantley et al Science, theoretical values for soils: 22 kJ (mol)-1 at global scale, the value 38 correslponds to a Q10 ~ 1.6
				diss_proton_stochio = stoichio_m_1cat + 2*stoichio_m_2cat + 3*stoichio_al_3cat
				diss_proton_stochio_parent = stoichio_m_1cat_parent + 2*stoichio_m_2cat_parent + 3*stoichio_al_3cat_parent
				imblance_order_dust = 0.5
				active_energy_dust = para_sic['active_energy_dust']*10**3 # 53*10**3 # ref Brantley et al Science, theoretical values for soils: 43 - 98 kJ (mol)-1, the value 53 correslponds to a Q10 ~ 2
				# active_energy_parent = 38*10**3 # ref Brantley et al Science, theoretical values for soils: 22 kJ (mol)-1 at global scale, the value 38 correslponds to a Q10 ~ 1.6
				enthalpy_dust = (-26.31 + (-208.59))*10**3 # unit J (mol)-1 Kanzaki et al.
				eq_const_dust = 10**(29.41 + 0.23)*torch.exp(-torch.tensor(enthalpy_dust)/gas_const*(1/temp_ref - 1/(273.15+15))) # Kanzaki et al. unit: (mol L-1)^xx

				# NOTE this is one parameter that can be changed assume an initial pH when at standard condition
				ph_initial_nominal = para_sic['ph_initial_nominal'] # 5 # torch.arange(5, 7, 2/n_soil_layer)

				# diss_rate_ref_gibbsite = 10**(-16.65)*24*3600 # unit mol m-2 day-1
				# diss_order_gibbsite = 0.992
				# active_energy_gibbsite = 47.5*10**3 # unit J mol-1
				# eq_const_gibbsite =  10**(8.11) # Kanzaki et al. unit: (mol L-1)^xx
				#
				# diss_rate_ref_calcite = 10**(-0.3)*24*3600 # unit mol m-2 day-1
				# diss_order_calcite = 1.0
				# active_energy_calcite = 14.4*10**3 # unit J mol-1
				# eq_const_calcite = 10**(-8.43) # Kanzaki et al. unit: (mol L-1)^xx

				#---------------------------------------------------------
				# concentration of species at equilibrium unit: mol/m3 in the solution
				#---------------------------------------------------------
				# reaction constants (unit (mol L-1)^(power_coef)) and enthalpy at standard condition (25 deg C, unit J/mol)
				# for convenience, we need to convert all species concentration to mol m-3
				# thus K_new = K_stansard*(10^3)^(power_coef)
				k_const_ref = torch.ones(12)*torch.nan
				delta_h_ref = torch.ones(12)*torch.nan
				# reaction 0: H2O --> H+(aq) + OH-(aq)
				k_const_ref[0] = 10**(-14)*(10**3)**(2)
				delta_h_ref[0] = 58.74*10**(3)
				# reaction 1: CO2(g) --> CO2(aq)
				k_const_ref[1] = henry_const_ref # unit already converted
				delta_h_ref[1] = -2400*gas_const # -21.33*10**(3) # delta_h_henry: enthalpy of CO3 dissolution unit: J (mol)-1 cf Wikipedia
				# reaction 2: CO2(aq) + H2O --> H+(aq) + HCO3-(aq)
				k_const_ref[2] = 10**(-6.35)*(10**3)**(1) # 10**(-3.36)*(10**3)**(1) # 10**(-6.37)*(10**3)**(1)
				delta_h_ref[2] = 11.94*10**(3) # -9.7*10**(3) # 11.94*10**(3)
				# reaction 3: HCO3-(aq) --> H+(aq) + CO3--(aq)
				k_const_ref[3] = 10**(-10.43)*(10**3)**(1) # this number has been checked
				delta_h_ref[3] = 17*10**(3) # -3.5*10**(3)
				# reaction 5: MgCO3(aq) --> Mg++(aq) + CO3--(aq)
				k_const_ref[5] = 10**(-3)*(10**3)
				delta_h_ref[5] = -11.35*10**(3)
				# reaction 6: Al+++(aq) + H2O --> Al(OH)++(aq) + H+ (aq)
				k_const_ref[6] = 10**(-5)*(10**3)**(1)
				delta_h_ref[6] = 48.07*10**(3)
				# reaction 7: Al+++(aq) + 4H2O --> Al(OH)4-(aq) + 4H+ (aq)
				k_const_ref[7] = 10**(-22.7)*(10**3)**(4)
				delta_h_ref[7] = 176.98*10**(3)
				# reaction 8: Al+++(aq) + 3H2O --> Al(OH)3(aq) + 3H+ (aq) 
				k_const_ref[8] = 10**(-16.9)*(10**3)**(3)
				delta_h_ref[8] = 117.48*10**(3)
				# reaction 9: H4SiO4(aq) --> H+(aq) + H3SiO4-(aq)
				k_const_ref[9] = 10**(-9.83)*(10**3)**(1) # consider quartz dissolution: 10**(-14)*(10**3)**(1) # 10**(-9.83)*(10**3)**(1)
				delta_h_ref[9] = 25.61*10**(3) # for H4SiO4(aq) 25.61*10**(3)
				# reaction 11: SOAH(aq) --> H+(aq) + SOA--(aq)
				# if ph_initial_nominal >= 6.0:
				# 	k_const_ref[11] = 10**(-2.5)*(10**3)**(1) # 5 for alfisol, 
				# elif ph_initial_nominal < 6.0 and ph_initial_nominal >= 5:
				# 	k_const_ref[11] = 10**(-5.0)*(10**3)**(1)# 10**(-5.5)*(10**3)**(1) # 5 for alfisol,
				# else:
				# 	k_const_ref[11] = 10**(-6.0)*(10**3)**(1) # 5 for alfisol,
				
				# delta_h_ref[11] = 7*10**(3)

				molar_h2o = month_water_frac*soil_vol_z*10**6/18 # unit mol
				vol_h2o = month_water_frac*soil_vol_z # unit m3
				
				#---------------------------------------------------------
				# conversion coefficients for reactions from water concentration to soil system concentration
				#----------------------------------------------------------
				eq_convert_coef = torch.ones(n_soil_layer, 12)
				eq_convert_coef[:, 0]  =  (vol_h2o/soil_vol_z)**(1)*(molar_h2o/vol_h2o)**(-1)
				eq_convert_coef[:, 1]  =  (vol_h2o/soil_vol_z)**(1)
				eq_convert_coef[:, 2]  =  (vol_h2o/soil_vol_z)**(0)*(molar_h2o/vol_h2o)**(-1)
				eq_convert_coef[:, 3]  =  (vol_h2o/soil_vol_z)**(1)
				eq_convert_coef[:, 5]  =  (vol_h2o/soil_vol_z)**(1)
				eq_convert_coef[:, 6]  =  (vol_h2o/soil_vol_z)**(0)*(molar_h2o/vol_h2o)**(-1)
				eq_convert_coef[:, 7]  =  (vol_h2o/soil_vol_z)**(0)*(molar_h2o/vol_h2o)**(-4)
				eq_convert_coef[:, 8]  =  (vol_h2o/soil_vol_z)**(0)*(molar_h2o/vol_h2o)**(-3)
				eq_convert_coef[:, 9]  =  (vol_h2o/soil_vol_z)**(1)
				# eq_convert_coef[:, 11] =  (vol_h2o/soil_vol_z)**(soa_proton)
				
				eq_convert_coef = eq_convert_coef.log10()
				# -------------------dust amendment------------------------
				# how the applied dust distributed in soils
				allocation_ratio = torch.ones(n_soil_layer)  # torch.ones(n_soil_layer) # 1/dz[0:n_soil_layer]
				allocation_ratio[dust_max_layer:] = 0 # allocation only happens in the first 30cm (5 layers)
				amendment_vector = allocation_ratio/sum(allocation_ratio)
				
				dust_concentration_initial = 0*vol_frac_mineral*amendment_vector/dust_molar_weight/dz[0:n_soil_layer] # unit: mol m-3
				dust_concentration = dust_concentration_initial*concentration_converter 
				molar_volume = dust_molar_weight/dust_density/concentration_converter # unit m3 mol-1
				dust_vol_frac_initial = dust_concentration*molar_volume # unit m3 dust mineral per m3 soil
				dust_vol_frac = dust_vol_frac_initial
				#----------------calculate corresponding reaction constant at different temperature by van’t Hoff Equation
				# K(T) = K_ref*exp(-delta_h_ref/R(1/T - 1/T_ref))
				eq_const_real = torch.ones(n_soil_layer, k_const_ref.shape[0])*torch.nan
				for ireaction in range(k_const_ref.shape[0]):
					# eq_const_real[:, ireaction] = k_const_ref[ireaction]*torch.exp(-delta_h_ref[ireaction]/gas_const*(1/month_soil_temp - 1/temp_ref))
					eq_const_real[:, ireaction] = k_const_ref[ireaction]*torch.exp(-delta_h_ref[ireaction]/gas_const*(1/month_soil_temp - 1/temp_ref))
				# end for ireaction in range(10)
				eq_const_real = eq_const_real.log10()
				
				#---------------------water
				h2o_initial = np.log10(month_water_frac*water_density/18) # unit: mol m-3 consider 1m3 water weights 10^6g, temperature has minor influence
				
				#---------------------proton and hydroxide (R0) H2O --> H+(aq) + OH-(aq)
				proton_initial = torch.ones(n_soil_layer)*np.log10(10**(-ph_initial_nominal)*10**(3)*(month_water_frac)) # convert concentration from mol L-1 to mol m-3
				# eq. assumption constraint from reaction 0
				hydroxide_initial = eq_convert_coef[:, 0] + eq_const_real[:, 0] + h2o_initial - proton_initial
				# print('equilibrium of water ionization has been set')
				
				#---------------------CO2 partial pressure and dissolution (R1) CO2(g) --> CO2(aq)
				if is_warmup == 1:
					initial_soc_co2_layer = soil_co2_gas_constraint*12.01/0.3  # unit mol m-2 --> gC/m3
				else:
					initial_soc_co2_layer = initial_soc_co2 # unit gC/m3
				# set initial co2, co2 in g c m-3 was first converted to mol m-3 WITH considering porosity, and then to pressure
				p_co2_initial = (initial_soc_co2_layer/12.01/month_porosity)*gas_const*month_soil_temp + 339*10**(-6)*atm_pressure # unit J m-3 == Pa
				co2_g_initial = (p_co2_initial/gas_const/month_soil_temp*month_porosity).log10() # unit mol m-3 
				
				soc_co2_layer = 10**(co2_g_initial)*12.01 # unit gC m-3
				p_co2_initial = p_co2_initial.log10()

				# henry constant unit: mol m-3 Pa-1, have converted concentration to soil system via eq_convert_coef
				co2_aq_initial = eq_convert_coef[:, 1] + eq_const_real[:, 1] + p_co2_initial# unit mol m-3
				
				# print('equilibrium of co2 dissolution has been set')
				#-------------------CO2 hydration (acid, R2) CO2(aq) + H2O --> H+(aq) + HCO3-(aq)
				bicarbo_initial = eq_convert_coef[:, 2] + eq_const_real[:, 2] + co2_aq_initial + h2o_initial - proton_initial
				# print('equilibrium of co2 hydration (acid) has been set')

				#-------------------CO2 hydration (alkaline, R3) HCO3-(aq) --> H+(aq) + CO3--(aq)
				carbo_initial = eq_convert_coef[:, 3] + eq_const_real[:, 3] + bicarbo_initial - proton_initial
				# print('equilibrium of co2 hydration (alkaline) has been set')

				#-----------------------------CEC partition, clay
				# two parts contribute CEC. one from clay, one from organic acid
				cec_total = torch.tensor([para_sic['cec_total']]) # unit mol m-3
				soa_fraction = para_sic['soa_fraction'] 
				base_saturation = torch.tensor([para_sic['base_saturation']])

				if False:
					sh_an_initial = (cec_total*(1-base_saturation)/2/3).log10() # mol m-3, /2 because of two negaitve chrages, /3 because there are three Pi in each Pi3Al2
					# the base saturated one is PiD
					sm_initial = (cec_total*base_saturation/2).log10() # mol m-3, /2 because of two negaitve chrages,

					soa_saturation = base_saturation # torch.tensor([para_sic['soa_saturation']])
					organic_acid_layer = soa_fraction*current_cpool[4] # unit gC/m3 assume active SOC is acting as soil organic acid
					soah_initial = (organic_acid_layer/(12)*(1-soa_saturation)).log10() # (organic_acid_layer/(6*12)).log10() # (0.000001*torch.ones(n_soil_layer)).log10() # (doc_layer/(10*12)).log10() # assume the avergae carbon chain has 10 carbon. unit mol m-3
					# over 2 because now it is SOA2D
					soa_an_initial = (organic_acid_layer/(12)*(soa_saturation)/2).log10()# eq_convert_coef[:, 11] + eq_const_real[:, 11] - soa_proton*proton_initial

				if True:
					total_cec_clay = cec_total*(1 - soa_fraction) # mol m-3 
					# now the unsaturated ones are Pi3Al2, assuming Pi with two negative charges
					sh_an_initial = (total_cec_clay*(1-base_saturation)/2/3).log10() # mol m-3, /2 because of two negaitve chrages, /3 because there are three Pi in each Pi3Al2
					# the base saturated one is PiD
					sm_initial = (total_cec_clay*base_saturation/2).log10() # mol m-3, /2 because of two negaitve chrages,
					
					##-----------------------------CEC partition, organic acid				
					total_cec_soa = cec_total*soa_fraction # mol m-3
					# SOAH
					soah_initial = (total_cec_soa*(1-base_saturation)).log10() # unit mol m-3
					# SOA2D
					soa_an_initial = (total_cec_soa*(base_saturation)/2).log10()# unit mol m-3,  /2 because of two SOA- in SOA2D

				# M++ initial	
				# mco3_initial = (para_sic['mco3_initial']*torch.ones(n_soil_layer)).log10()
				# mineral_2cat_initial = eq_convert_coef[:, 5] + eq_const_real[:, 5] + mco3_initial - carbo_initial 
				if False:
					ex_sol_ratio_d = ex_sol_ratio_cation
					mineral_2cat_initial = (10**(sm_initial)/ex_sol_ratio_d*torch.ones(n_soil_layer)).log10()

				if True:
					ex_sol_ratio_d = para_sic['ex_sol_ratio_d']
					mineral_2cat_initial = (10**(sm_initial)/10**(ex_sol_ratio_d)*torch.ones(n_soil_layer)).log10()
				
				mco3_initial = mineral_2cat_initial + carbo_initial - eq_convert_coef[:, 5] - eq_const_real[:, 5]
				# M+ initial
				mineral_1cat_initial = mineral_2cat_initial # mineral_2cat_initial # (10**(-3)*torch.ones(n_soil_layer)).log10()
				# print('equilibrium of calcite has been set')

				# print(10**(mineral_2cat_initial))

				#--------------------Aluminum related				
				ex_sol_ratio_al	= para_sic['ex_sol_ratio_al']

				# if ph_initial_nominal <= 6: # Al in the data product is mainly Al3+
				# 	al_3cat_initial_middle = (para_sic['al_3cat_ex']/10**(ex_sol_ratio_al)*torch.ones(n_soil_layer)).log10() # exchangable Al in soil: 50 - 150 mg kg-1 --> 4mol m-3 when taking 100mg kg-1
				# 	aloh3_initial_middle = eq_convert_coef[:, 8] + eq_const_real[:, 8] + al_3cat_initial_middle + 3*h2o_initial - 3*proton_initial
				# 	# reaction 6: Al+++(aq) + H2O --> Al(OH)++(aq) + H+ (aq)
				# 	aloh_2cat_initial_middle = eq_convert_coef[:, 6] + eq_const_real[:, 6] + al_3cat_initial_middle + h2o_initial - proton_initial
				# 	# reaction 7: Al+++(aq) + 4H2O --> Al(OH)4-(aq) + 4H+ (aq)
				# 	aloh4_1an_initial_middle = eq_convert_coef[:, 7] + eq_const_real[:, 7] + al_3cat_initial_middle  + 4*h2o_initial - 4*proton_initial
				# elif ph_initial_nominal > 6 and ph_initial_nominal < 7: # Al in the data product is mainly Al(OH)3
				# 	aloh3_initial_middle = (para_sic['al_3cat_ex']/10**(ex_sol_ratio_al)*torch.ones(n_soil_layer)).log10() 
				# 	al_3cat_initial_middle = aloh3_initial_middle - eq_convert_coef[:, 8] - eq_const_real[:, 8] - 3*h2o_initial + 3*proton_initial
				# 	# reaction 6: Al+++(aq) + H2O --> Al(OH)++(aq) + H+ (aq)
				# 	aloh_2cat_initial_middle = eq_convert_coef[:, 6] + eq_const_real[:, 6] + al_3cat_initial_middle + h2o_initial - proton_initial
				# 	# reaction 7: Al+++(aq) + 4H2O --> Al(OH)4-(aq) + 4H+ (aq)
				# 	aloh4_1an_initial_middle = eq_convert_coef[:, 7] + eq_const_real[:, 7] + al_3cat_initial_middle  + 4*h2o_initial - 4*proton_initial
				# else:
				# 	aloh4_1an_initial_middle = (para_sic['al_3cat_ex']/10**(ex_sol_ratio_al)*torch.ones(n_soil_layer)).log10() # Al in the data product is mainly Al(OH)4-
				# 	al_3cat_initial_middle = aloh4_1an_initial_middle - eq_convert_coef[:, 7] - eq_const_real[:, 7] - 4*h2o_initial + 4*proton_initial
				# 	aloh3_initial_middle = eq_convert_coef[:, 8] + eq_const_real[:, 8] + al_3cat_initial_middle + 3*h2o_initial - 3*proton_initial
				# 	aloh_2cat_initial_middle = eq_convert_coef[:, 6] + eq_const_real[:, 6] + al_3cat_initial_middle + h2o_initial - proton_initial
				
				# al_species_tot_middle = 10**(al_3cat_initial_middle) + 10**(aloh_2cat_initial_middle) + 10**(aloh3_initial_middle) + 10**(aloh4_1an_initial_middle)
				# # redistribution according to their relative fractions
				# al_3cat_initial = ((10**(al_3cat_initial_middle)/al_species_tot_middle)*para_sic['al_3cat_ex']/10**(ex_sol_ratio_al)*torch.ones(n_soil_layer)).log10() 
				# aloh_2cat_initial = ((10**(aloh_2cat_initial_middle)/al_species_tot_middle)*para_sic['al_3cat_ex']/10**(ex_sol_ratio_al)*torch.ones(n_soil_layer)).log10() 
				# aloh3_initial = ((10**(aloh3_initial_middle)/al_species_tot_middle)*para_sic['al_3cat_ex']/10**(ex_sol_ratio_al)*torch.ones(n_soil_layer)).log10() 
				# aloh4_1an_initial = ((10**(aloh4_1an_initial_middle)/al_species_tot_middle)*para_sic['al_3cat_ex']/10**(ex_sol_ratio_al)*torch.ones(n_soil_layer)).log10() 

				if para_prior_version == 0:
					al_3cat_initial = 1/2*(3*mineral_2cat_initial + 2*((1 - base_saturation).log10()) - ex_sol_ratio_al - 3*(base_saturation.log10()))
					# reaction 6: Al+++(aq) + H2O --> Al(OH)++(aq) + H+ (aq)
					aloh_2cat_initial = eq_convert_coef[:, 6] + eq_const_real[:, 6] + al_3cat_initial + h2o_initial - proton_initial
					# reaction 8: Al+++(aq) + 3H2O --> Al(OH)3 (aq) + 3H+ (aq)
					aloh3_initial = eq_convert_coef[:, 8] + eq_const_real[:, 8] + al_3cat_initial + 3*h2o_initial - 3*proton_initial
					# reaction 7: Al+++(aq) + 4H2O --> Al(OH)4-(aq) + 4H+ (aq)
					aloh4_1an_initial = eq_convert_coef[:, 7] + eq_const_real[:, 7] + al_3cat_initial  + 4*h2o_initial - 4*proton_initial
				# print('equilibrium of aluminum has been set')

				# print(10**(al_3cat_initial) + 10**(aloh3_initial) + 10**(aloh_2cat_initial) + 10**(aloh4_1an_initial))

				#--------------------silica related reaction 9: H4SiO4(aq) --> H+(aq) + H3SiO4-(aq)
				h4sio4_initial = (0.1*torch.ones(n_soil_layer)).log10() # <-- ~ 0.1 H4SiO4 (aq) from Epstein 1993 PNAS, quartz (SiO2) in soil is 6.2 - 81.7 wt% --> 5 mol m-3 when taking 20 wt%
				h3sio4_1an_initial = eq_convert_coef[:, 9] + eq_const_real[:, 9] + h4sio4_initial - proton_initial
				# print('equilibrium of silica has been set')
				
				#------------------------------ calcite and gibbsite initial
				carbonate_storage = torch.tensor([para_sic['carbonate_storage']]) # mol m-3
				gibbsite_storage = torch.tensor([para_sic['gibbsite_storage']]) # mol m-3
				carbonate_initial = carbonate_storage.log10()
				gibbsite_initial = gibbsite_storage.log10()

				
				# print('chemical equilibriums have been set')
				# convert all unit (mol m-3) to be measured by soil system volumes
				species_vector = torch.ones((species_num*n_soil_layer), 1)*torch.nan
				species_vector[0*n_soil_layer:1*n_soil_layer, 0] = al_3cat_initial
				species_vector[1*n_soil_layer:2*n_soil_layer, 0] = aloh_2cat_initial
				species_vector[2*n_soil_layer:3*n_soil_layer, 0] = aloh4_1an_initial
				species_vector[3*n_soil_layer:4*n_soil_layer, 0] = h4sio4_initial
				species_vector[4*n_soil_layer:5*n_soil_layer, 0] = h3sio4_1an_initial
				species_vector[5*n_soil_layer:6*n_soil_layer, 0] = mineral_1cat_initial
				species_vector[6*n_soil_layer:7*n_soil_layer, 0] = mineral_2cat_initial
				species_vector[7*n_soil_layer:8*n_soil_layer, 0] = hydroxide_initial
				species_vector[8*n_soil_layer:9*n_soil_layer, 0] = proton_initial
				species_vector[9*n_soil_layer:10*n_soil_layer, 0] = co2_aq_initial
				species_vector[10*n_soil_layer:11*n_soil_layer, 0] = co2_g_initial
				species_vector[11*n_soil_layer:12*n_soil_layer, 0] = bicarbo_initial
				species_vector[12*n_soil_layer:13*n_soil_layer, 0] = carbo_initial
				species_vector[13*n_soil_layer:14*n_soil_layer, 0] = h2o_initial
				species_vector[14*n_soil_layer:15*n_soil_layer, 0] = mco3_initial
				species_vector[15*n_soil_layer:16*n_soil_layer, 0] = aloh3_initial
				species_vector[16*n_soil_layer:17*n_soil_layer, 0] = sm_initial
				species_vector[17*n_soil_layer:18*n_soil_layer, 0] = sh_an_initial
				species_vector[18*n_soil_layer:19*n_soil_layer, 0] = soa_an_initial
				species_vector[19*n_soil_layer:20*n_soil_layer, 0] = soah_initial
				if is_use_solid_mineral == True:
					species_vector[20*n_soil_layer:21*n_soil_layer, 0] = carbonate_initial
					species_vector[21*n_soil_layer:22*n_soil_layer, 0] = gibbsite_initial
				
				# plt.plot(species_vector)

				# check charge balance at initial point
				species_matrix = 10**(species_vector.reshape(species_num, n_soil_layer).t())  # unit mol m-3
				neg_charge_tot_initial = (species_matrix[:, species_charge.sign() < 0]*species_charge[species_charge.sign() < 0].reshape(1, species_charge[species_charge.sign() < 0].shape[0]).repeat((n_soil_layer, 1))).sum(axis = 1)
				pos_charge_tot_initial = (species_matrix[:, species_charge.sign() > 0]*species_charge[species_charge.sign() > 0].reshape(1, 5).repeat((n_soil_layer, 1))).sum(axis = 1)
				initial_charge_conservation = pos_charge_tot_initial + neg_charge_tot_initial
				# print('initial charge balance check (pos + neg, unit: mol m-3) I = ' + str(initial_charge_conservation.numpy()))

				# balance the charges
				if charge_neutral == True:
					for ilayer in range(n_soil_layer):
						# more positive charges, add negative charges to SH-
						if initial_charge_conservation[ilayer] > 0:
							balanced_sh_an = (10**(sh_an_initial[ilayer]) + abs(initial_charge_conservation[ilayer])/(2 - cec_proton)).log10()
							species_vector[(17*n_soil_layer+ilayer), 0] = balanced_sh_an
						# more negative charges, add positive charges to M+
						else:
							balanced_mineral_1cat = (10**(mineral_1cat_initial[ilayer]) + abs(initial_charge_conservation[ilayer])).log10()
							species_vector[(5*n_soil_layer+ilayer), 0] = balanced_mineral_1cat
					
					species_matrix = 10**(species_vector.reshape(species_num, n_soil_layer).t())  # unit mol m-3
					neg_charge_tot_initial = (species_matrix[:, species_charge.sign() < 0]*species_charge[species_charge.sign() < 0].reshape(1, species_charge[species_charge.sign() < 0].shape[0]).repeat((n_soil_layer, 1))).sum(axis = 1)
					pos_charge_tot_initial = (species_matrix[:, species_charge.sign() > 0]*species_charge[species_charge.sign() > 0].reshape(1, 5).repeat((n_soil_layer, 1))).sum(axis = 1)
					initial_charge_conservation = pos_charge_tot_initial + neg_charge_tot_initial
					# print('initial charge balance check (pos + neg, unit: mol m-3) II = ' + str(initial_charge_conservation.numpy()))
				
				species_stock_initial = (species_matrix*dz[0:n_soil_layer].reshape(n_soil_layer, 1).repeat(1, species_num)).sum(axis = 0) # unit mol m-2

				# figure for initial concentration
				if is_plotting == 1:
					color_scheme = ['#a6cee3','#1f78b4','#b2df8a','#33a02c', \
					'#fb9a99','#e31a1c','#fdbf6f','#ff7f00', \
					'#cab2d6','#6a3d9a','#ffff99','#b15928', \
					'#003c30','#543005','#40004b','#1a1a1a', \
					'#7bccc4', '#a6bddb', '#67001f', '#878787', '#FFC20A', '#0C7BDC']

					plt.figure(figsize = (6, 8), layout = 'tight')
					axes = plt.gca()
					axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
					plt.barh(torch.arange(species_num), species_matrix[0, :].detach().numpy(), edgecolor = color_scheme, color = color_scheme, linewidth = 2, tick_label = species_list)
					plt.axvline(x = 0, color = 'grey', linewidth = 2, linestyle = 'solid')
					axes.set_xlim([species_matrix[0, :].detach().numpy().min()/10, species_matrix[0, :].detach().numpy().max()*10])
					#end for ispecies in range(len(species_list)):
					plt.xscale('log')
					# plt.legend(loc = (1.1, 0.1), fontsize = 14, markerscale = 1.5, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 1)
					plt.xlabel(xlabel = 'Concentration ($mol \ m^{-3}$)', fontsize = 18)
					plt.tick_params(labelsize = 14, width = 1.5, length = 7)
					plt.savefig(fig_dir + 'species_initial_concentration.jpeg', dpi = 300)
					plt.close()

				# plt.plot(species_vector)
				species_vector = 10**(species_vector)*concentration_converter
				species_vector_initial = species_vector
			# end if imonth == 0 and iyear == 0 and is_weathering == 1:

			###########################################################################
			# start: organic carbon iterations
			###########################################################################
			#----------------------------------------
			# Environmental Scalar (Xi)
			#----------------------------------------
			# temperature related function xit
			# calculate rate constant scalar for soil temperature
			# assuming that the base rate constants are assigned for non-moisture
			# limiting conditions at 25 C.
			for ilayer in torch.arange(0, n_soil_layer):
				if cesm2_simu_soil_temperature[ilayer, imonth, year_count] >= (0 + kelvin_to_celsius):
					xit[ilayer] = q10**((cesm2_simu_soil_temperature[ilayer, imonth, year_count] - (kelvin_to_celsius + 25))/10)
				else:
					xit[ilayer] = q10**((273.15 - 298.15)/10)*(fq10**((cesm2_simu_soil_temperature[ilayer, imonth, year_count] - (0 + kelvin_to_celsius))/10))
				# end if soil_temp_profile[ilayer, imonth] >= 0 + kelvin_to_celsius:
			# end for layer
			
			if normalize_q10_to_century_tfunc == True:
				# scale all decomposition rates by a constant to compensate for offset between original CENTURY temp func and Q10
				normalization_factor = (catanf(normalization_tref)/catanf_30) / (q10**((normalization_tref-25)/10))
				xit = xit*normalization_factor
			# end if normalize_q10_to_century_tfunc == True:
			
			# water related function xiw
			xiw = cesm2_simu_w_scalar[:, imonth, year_count]*w_scaling
			xiw[xiw > 1] = 1
			
			xin = cesm2_simu_n_scalar[:, imonth, year_count]
			xio = cesm2_simu_o_scalar[:, imonth, year_count]
			#----------------------------------------
			# Triangle Matrix, A Matrix and K Matrix
			#----------------------------------------
			# A matrix
			a_ma = a_matrix(fl1s1, fl2s1, fl3s2, fs1s2, fs1s3, fs2s1, fs2s3, fs3s1, fcwdl2, cesm2_simu_cellsand[:, imonth, year_count])

			#------------organo-mineral association
			if is_weathering == 0:
				om_modify_scalar = 1
			if is_weathering == 1 and om_association_index == 1:
				if imonth == 0 and iyear == 0:
					om_modify_scalar = 1
				else:
					om_modify_scalar = torch.nanmean(om_association_scalar_record)
			else:
				om_modify_scalar = 1
			
			tau4s2_modify = tau4s2*om_modify_scalar
			tau4s3_modify = tau4s3*om_modify_scalar
			
			# K matrix
			kk_ma = kk_matrix(xit, xiw, xio, xin, efolding, tau4cwd, tau4l1, tau4l2, tau4l3, tau4s1, tau4s2_modify, tau4s3_modify)
			# tri matrix
			if n_soil_layer == 1:
				tri_ma = torch.zeros(npool*n_soil_layer, npool*n_soil_layer)
			
			#---------------------------------------------------
			# input allocation
			#---------------------------------------------------
			vertical_input = 1.0
			#---------------------------------------------------
			# input vector
			#---------------------------------------------------
			matrix_in = torch.ones(npool*n_soil_layer, 1)*torch.nan
			# construct vector unit gc/m3/day
			matrix_in[0*n_soil_layer:1*n_soil_layer, 0] = cesm2_simu_input_sum_cwd[imonth, year_count]*vertical_input/(dz[0:n_soil_layer]*days_per_month[imonth])
			matrix_in[1*n_soil_layer:2*n_soil_layer, 0] = cesm2_simu_input_sum_litter1[imonth, year_count]*vertical_input/(dz[0:n_soil_layer]*days_per_month[imonth])
			matrix_in[2*n_soil_layer:3*n_soil_layer, 0] = cesm2_simu_input_sum_litter2[imonth, year_count]*vertical_input/(dz[0:n_soil_layer]*days_per_month[imonth])
			matrix_in[3*n_soil_layer:4*n_soil_layer, 0] = cesm2_simu_input_sum_litter3[imonth, year_count]*vertical_input/(dz[0:n_soil_layer]*days_per_month[imonth])
			matrix_in[4*n_soil_layer:npool*n_soil_layer, 0] = 0
			
			#---------------------------------------------------
			# stepwise iteration
			#---------------------------------------------------
			next_cpool = (matrix_in + torch.matmul((torch.matmul(a_ma, kk_ma) - tri_ma), current_cpool))*days_per_month[imonth] + current_cpool # unit: g c/m3
			next_cpool[next_cpool < 0] = 0
			
			next_soc_layer = next_cpool.reshape(npool, n_soil_layer).t().sum(axis = 1)*dz[0:n_soil_layer] # unit gC/m2
			
			#---------------------------------------------------
			# respired flux
			#---------------------------------------------------
			r_vector = -torch.sum(a_ma, axis = 0).reshape(npool*n_soil_layer, 1) # fraction of respiration in decomposition for each cpool at each depth
			r_vector[0:n_soil_layer] = 0
			respired_cpool = r_vector*torch.diag(kk_ma).reshape(npool*n_soil_layer, 1)* current_cpool # unit: gc m-3 day-1 # *days_per_month[imonth] # unit: gc m-3 month-1
		
			# respiration flux
			respired_layer = respired_cpool.reshape(npool, n_soil_layer).t().sum(axis = 1) # unit gC/m3/day
			total_resp = (respired_layer*dz[0:n_soil_layer]).sum() # # unit gC/m2/day
			
			#---------------------------------------------------
			# carbon pool update
			#---------------------------------------------------
			current_cpool = next_cpool
			total_c_stock = next_soc_layer.sum() 
			###########################################################################
			# end: organic carbon iterations
			###########################################################################
			
			if is_weathering == 1 and imonth == 0 and iyear == dust_start_year: # forcing_year_num: # (iyear+1)%300 == 0:
				rain_limit_midpoint = rain_limit_midpoint*para_sic['adjust_scalar_leach']
				co2_turnover_time = co2_turnover_time*para_sic['adjust_scalar_pco2']

				species_vector[species_list.index('SOA2D')] = species_vector[species_list.index('SOA2D')]*para_sic['adjust_soa_ratio']

				species_vector[species_list.index('Pi3Al2')] = species_vector[species_list.index('Pi3Al2')]*para_sic['adjust_ex_ratio']
				species_vector[species_list.index('PiD')] = species_vector[species_list.index('PiD')]*para_sic['adjust_ex_ratio']

				leachate_turnover_time = leachate_turnover_time*para_sic['adjust_solute_tau']
				m2_turnover_time = m2_turnover_time*para_sic['adjust_solute_tau']
				al_turnover_time = al_turnover_time*para_sic['adjust_solute_tau']
				silica_turnover_time = silica_turnover_time*para_sic['adjust_solute_tau']

				dust_concentration_initial = amendment_rate*vol_frac_mineral*amendment_vector/dust_molar_weight/dz[0:n_soil_layer] # unit: mol m-3
				dust_concentration = dust_concentration_initial*concentration_converter 
				molar_volume = dust_molar_weight/dust_density/concentration_converter # unit m3 mol-1
				dust_vol_frac_initial = dust_concentration*molar_volume # unit m3 dust mineral per m3 soil
				dust_vol_frac = dust_vol_frac_initial
 			
			if is_repeat_app == True and is_weathering == 1 and imonth == 0 and iyear > dust_start_year and (iyear - dust_start_year) % 2 == 0 and iyear < dust_end_year: # repeat application every other year, counted from the first application
				dust_concentration_initial = dust_concentration_initial + amendment_rate*vol_frac_mineral*amendment_vector/dust_molar_weight/dz[0:n_soil_layer] # unit: mol m-3
				remaining_dust_tot = (dust_concentration*dz[0:n_soil_layer]).sum()/concentration_converter*dust_molar_weight # unit: g m-2
				new_dust_tot = amendment_rate # amendment_rate/20 # eqv to 1/20 of initial application
				dust_concentration_recurrent = (new_dust_tot+remaining_dust_tot)*vol_frac_mineral*amendment_vector/dust_molar_weight/dz[0:n_soil_layer] # unit: mol m-3 # torch.tensor([0.0]) 
				dust_concentration = dust_concentration_recurrent*concentration_converter
				dust_vol_frac = dust_concentration*molar_volume # unit m3 dust mineral per m3 soil   

			###########################################################################
			# start: weathering --- interations
			###########################################################################
			soil_moisture_scalar = fun_h2o_limitation(month_water_frac, water_limit_steepness, water_limit_midpoint, time_downscale) # soil-moisture factor zeta_theta, Eq. 24; multiplies the parent and dust weathering progress if is_use_moisture_scalar == 1 (Eq. 25)
			leach_frac_base = fun_h2o_limitation(month_rain, rain_limit_steepness, rain_limit_midpoint, time_downscale)

			# tri_ma_co2 = tri_matrix_alternative(npool, nbedrock_gas, slope_co2, intercept_co2, float_ratio_co2, leach_ratio, rate_to_atmos_co2, rate_to_rock)
			co2_turnover_rate = 1/co2_turnover_time
			tri_ma_co2 = torch.diag(co2_turnover_rate*torch.ones(npool*n_soil_layer))

			# NOTE here is the control of time resolution 
			resp_co2_timestep = concentration_converter*respired_cpool.reshape(npool, n_soil_layer).t().sum(axis = 1)/time_downscale/co2_dilute_factor # unit concentration_converter*gC/m3/timstep 
			soc_co2_layer = soc_co2_layer*concentration_converter # concentration_converter*unit gc m-3
			plant_synthesis = cesm2_simu_gpp[imonth, year_count]*vertical_input/dz[0:n_soil_layer] # unit gc m-3 s-1
			plant_synthesis = plant_synthesis/12*3600*24*concentration_converter/time_downscale # unit gc m-2 s-1 --> concentration_converter*mol m-2 timestep-1

			co2_g_record = torch.zeros(time_downscale*days_per_month[imonth], n_soil_layer)*torch.nan
			consumed_co2_record = torch.zeros(n_soil_layer, 1)
			consumed_h2o_record = torch.zeros(time_downscale*days_per_month[imonth], 6)*torch.nan # for leaching, co2 dynamics, h2o dynamics, dust weathering, biosynthesis, and parent weathering
			delta_co2_record = torch.zeros(time_downscale*days_per_month[imonth], species_num)
			leached_ion_record = torch.zeros(time_downscale*days_per_month[imonth], species_num)
			process_list = ['leached total', 'CO2', 'H2O', 'ERW', 'nitrification', 'parent weathering', 'C leach', 'D2 leach', 'Al leach', 'Si leach']
			delta_species_record = torch.zeros(time_downscale*days_per_month[imonth], 10, species_num)*torch.nan
			species_record = torch.zeros(time_downscale*days_per_month[imonth], species_num)*torch.nan
			weathering_rate_record = torch.zeros(time_downscale*days_per_month[imonth], n_soil_layer)*torch.nan
			weathering_rate_record_parent = torch.zeros(time_downscale*days_per_month[imonth], n_soil_layer)*torch.nan

			om_association_scalar_record = torch.ones(time_downscale*days_per_month[imonth], n_soil_layer)
			
			# iterations at the finest time scale 
			for itimestep in range(time_downscale*days_per_month[imonth]):
				#--------------------------------------------------
				# gas mixing
				#--------------------------------------------------				
				soc_co2_layer_laststep = soc_co2_layer # unit concentration_converter*g c m-3
				# respiration flux update
				soc_co2_layer = soc_co2_layer + resp_co2_timestep*(1+ratio_ra_to_rh) # unit concentration_converter*gc m-3
				
				# vertical mixing update
				delta_co2 = torch.matmul(-tri_ma_co2[0:n_soil_layer, 0:n_soil_layer], soc_co2_layer.reshape(n_soil_layer, 1)) # unit concentration_converter*gc m-3
				soc_co2_layer = soc_co2_layer + delta_co2[:, 0] # unit concentration_converter*gc m-3
				
				delta_co2_record[itimestep, 0] = delta_co2/concentration_converter
				delta_co2_record[itimestep, 1] = resp_co2_timestep/concentration_converter
				# print((soc_co2_layer*dz[0:n_soil_layer]).sum())
				if (torch.isnan(soc_co2_layer[0]) or soc_co2_layer[0] < 0) == True:
					raise ValueError('nan value in species vector happened in timestep ' + str(itimestep) + ' month ' + str(imonth) + ' year ' + str(iyear))
				
				if is_weathering == 1:
					#------------------------------------------------
					# co2 driven alteration
					#------------------------------------------------
					
					# calculate the progress of co2(g) change
					progress_co2_g = 1*(soc_co2_layer - soc_co2_layer_laststep)/12.01 # torch.zeros(5) # -1*(soc_co2_layer - soc_co2_layer_laststep)/12.01 # unit mol m-3
					#--------------------alteration matrix and directional vector
					direct_vector_co2 = torch.ones(n_soil_layer*species_num, 1)*0

					direct_vector_co2[4*n_soil_layer:5*n_soil_layer, 0] = 2
					direct_vector_co2[6*n_soil_layer:7*n_soil_layer, 0] = 1

					alter_matrix_co2 = alteration_matrix(species_vector, species_num, cec_proton, soa_proton, is_use_solid_mineral)

					# species_relative_change = torch.linalg.solve(alter_matrix, direct_vector)

					max_progress = progress_co2_g.reshape(n_soil_layer, 1).repeat(species_num, 1)
					
					# converged block-step: no per-species allowance and no floor; the full progress is applied
					apparent_progress = max_progress

					# update we only use apparent progress to calculate changes of species other than co2. for co2, we still use its real progress
					delta_species_vector = newton_block(species_vector, alter_matrix_co2, direct_vector_co2, apparent_progress[0, 0], block_label = 'co2 supply, year ' + str(int(iyear)) + ' month ' + str(int(imonth)) + ' step ' + str(itimestep)) # converged block-step (damped Newton in ln n)
					
					delta_species_matrix = delta_species_vector.reshape(species_num, n_soil_layer).t() # unit mol m-3
					delta_species_matrix = delta_species_matrix[0:n_soil_layer, :]*dz[0:n_soil_layer].reshape(n_soil_layer, 1).repeat(1, species_num) # unit mol m-2
					delta_species_record[itimestep, 1, :] = delta_species_matrix.sum(axis = 0)

					net_charge_change = (delta_species_vector[:, 0]*species_charge).sum()
					if abs(net_charge_change) > torch.tensor(10**(0)): # torch.tensor(10**(-4)):
						raise ValueError('charge balance not correct in CO2 dynamics timestep ' + str(itimestep) + ' month ' + str(imonth.numpy()) + ' year ' + str(iyear.numpy()) + ' change: ' + str(net_charge_change.numpy()))

					net_mass_change = (element_mass_matrix*delta_species_vector.reshape(1, species_num).repeat([9, 1])).sum(axis = 1)
					net_mass_change[4] = net_mass_change[4] - 2*apparent_progress[0] # added mass O
					net_mass_change[6] = net_mass_change[6] - 1*apparent_progress[0] # added mass C

					if max(abs(net_mass_change)) > torch.tensor(10**(0)): # torch.tensor(10**(-4))::
						raise ValueError('mass balance not correct in CO2 dynamics timestep ' + str(itimestep) + ' month ' + str(imonth.numpy()) + ' year ' + str(iyear.numpy()) + ' change: ' + str(net_mass_change.numpy()))

					consumed_h2o_record[itimestep, 1] = -1*(delta_species_vector[13*n_soil_layer:14*n_soil_layer, 0]*dz[0:n_soil_layer]).sum()
					
					species_vector_current = species_vector + delta_species_vector
					# species_vector_current[species_vector_current <= +0.0] = trivial_con					
					if torch.where(species_vector_current < 0)[0].shape[0] > 0 \
						or torch.isnan(species_vector_current.sum()) == True:
						raise ValueError('invalid species vector updates (co2) happened in timestep ' + str(itimestep) + ' month ' + str(imonth) + ' year ' + str(iyear))
					#end if
					
					species_vector = species_vector_current

					if is_plotting == 1 and is_weathering == 1 and iyear == 0 and imonth == 0 and itimestep == 0:

						plt.figure(figsize = (6, 12), layout = 'tight')
						axes = plt.gca()
						axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
						plt.barh(torch.arange(species_num), delta_species_vector[:, 0].detach().numpy(), edgecolor = color_scheme, color = color_scheme, linewidth = 2, tick_label = species_list)
						plt.axvline(x = 0, color = 'grey', linewidth = 2, linestyle = 'solid')
						#end for ispecies in range(len(species_list)):
						# plt.yscale('symlog')
						# plt.legend(loc = (1.1, 0.1), fontsize = 14, markerscale = 1.5, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 1)
						plt.xlabel(xlabel = 'Absolute progress', fontsize = 18)
						plt.tick_params(labelsize = 14, width = 1.5, length = 7)
						plt.savefig(fig_dir + 'species_abs_change_co2_driven.jpeg', dpi = 300)
						plt.close()

					#------------------------------------------------
					# leaching HCO3-
					#------------------------------------------------
										
					# if (species_vector/species_vector_initial)[6*n_soil_layer:7*n_soil_layer, 0] < 10**(-4):
					# 	rate_leach = 0
					# else:
					# 	rate_leach = (1/leachate_turnover_time)*leach_frac_base
					rate_leach = (1/leachate_turnover_time)*leach_frac_base

					## leach HCO3- and CO3--. take DCO3 (aq) as immobile
					leach_c_control = torch.tensor([1, 1, 1])
					progress_leach = 1*(rate_leach*(species_vector[11*n_soil_layer:12*n_soil_layer]*leach_c_control[0] + 
													species_vector[12*n_soil_layer:13*n_soil_layer]*leach_c_control[1] + 
													species_vector[14*n_soil_layer:15*n_soil_layer]*leach_c_control[2]
													))
					carbon_aq_sepcies_fraction  = (species_vector[[11, 12, 14], 0]/species_vector[[11, 12, 14]].sum())*leach_c_control
					carbon_aq_sepcies_fraction = carbon_aq_sepcies_fraction/carbon_aq_sepcies_fraction.sum()

					#--------------------alteration matrix and directional vector
					direct_vector_leach = torch.ones(n_soil_layer*species_num, 1)*0

					# direct_vector_leach[4*n_soil_layer:5*n_soil_layer, 0] = -3 # -3 # O
					# direct_vector_leach[5*n_soil_layer:6*n_soil_layer, 0] = -1 # -1 # H
					# direct_vector_leach[6*n_soil_layer:7*n_soil_layer, 0] = -1 # -1	# C

					direct_vector_leach[2*n_soil_layer:3*n_soil_layer, 0] = -1*carbon_aq_sepcies_fraction[2] 	# D only DCO3 has 1
					direct_vector_leach[4*n_soil_layer:5*n_soil_layer, 0] = -3 								# O HCO3-, CO3--, and DCO3 all have 3
					direct_vector_leach[5*n_soil_layer:6*n_soil_layer, 0] = -1*carbon_aq_sepcies_fraction[0] 	# H only HCO3 has 1
					direct_vector_leach[6*n_soil_layer:7*n_soil_layer, 0] = -1 								# C HCO3-, CO3--, and DCO3 all have 1

					alter_matrix_leach = alteration_matrix(species_vector, species_num, cec_proton, soa_proton, is_use_solid_mineral)

					# species_relative_change = torch.linalg.solve(alter_matrix, direct_vector)

					max_progress = progress_leach.reshape(n_soil_layer, 1).repeat(species_num, 1)
					
					# converged block-step: no per-species allowance and no floor; the full progress is applied
					apparent_progress = max_progress
					
					# max_stock_change_constraint = -(species_vector*max_comsume_frac) # under most extreme conditions, there will sitll be 0.1% species remained after reactions
					# constraint_loc = torch.where((species_relative_change*max_progress) < 0)[0]

					# constraint_stock_change = torch.concat((abs(max_stock_change_constraint[constraint_loc]), abs(max_progress*species_relative_change)[constraint_loc]), axis = 0)
					# constraint_stock_change_final, constraint_stock_change_loc = torch.min(constraint_stock_change, dim = 0)
						
					# if (constraint_stock_change_loc + 1) > constraint_loc.shape[0]:
					# 	constraint_progress = max_progress[0].sign()*abs(constraint_stock_change_final/species_relative_change)[constraint_loc][constraint_stock_change_loc - constraint_loc.shape[0]]
					# else:
					# 	constraint_progress = max_progress[0].sign()*abs(constraint_stock_change_final/species_relative_change)[constraint_loc][constraint_stock_change_loc]
					
					# apparent_progress = constraint_progress.reshape(n_soil_layer, 1).repeat(species_num, 1)

					# update we only use apparent progress to calculate changes of species other than co2. for co2, we still use its real progress
					delta_species_vector = newton_block(species_vector, alter_matrix_leach, direct_vector_leach, apparent_progress[0, 0], block_label = 'leaching carbonate, year ' + str(int(iyear)) + ' month ' + str(int(imonth)) + ' step ' + str(itimestep)) # converged block-step (damped Newton in ln n)
					
					delta_species_matrix = delta_species_vector.reshape(species_num, n_soil_layer).t() # unit mol m-3
					delta_species_matrix = delta_species_matrix[0:n_soil_layer, :]*dz[0:n_soil_layer].reshape(n_soil_layer, 1).repeat(1, species_num) # unit mol m-2
					delta_species_record[itimestep, 0, :] = delta_species_matrix.sum(axis = 0)
					delta_species_record[itimestep, 6, :] = delta_species_matrix.sum(axis = 0)

					leached_ion_record[itimestep, 11] = carbon_aq_sepcies_fraction[0]*apparent_progress[0]
					leached_ion_record[itimestep, 12] = carbon_aq_sepcies_fraction[1]*apparent_progress[0]
					leached_ion_record[itimestep, 14] = carbon_aq_sepcies_fraction[2]*apparent_progress[0]

					net_charge_change = (delta_species_vector[:, 0]*species_charge).sum() + -1*carbon_aq_sepcies_fraction[0]*apparent_progress[0] + (-2)*carbon_aq_sepcies_fraction[1]*apparent_progress[0]
					if abs(net_charge_change) > torch.tensor(10**(0)): # torch.tensor(10**(-4)):
						raise ValueError('charge balance not correct in HCO3- leaching timestep ' + str(itimestep) + ' month ' + str(imonth.numpy()) + ' year ' + str(iyear.numpy()) + ' change: ' + str(net_charge_change.numpy()))

					net_mass_change = (element_mass_matrix*delta_species_vector.reshape(1, species_num).repeat([9, 1])).sum(axis = 1)
					net_mass_change[2] = net_mass_change[2] + 1*carbon_aq_sepcies_fraction[2]*apparent_progress[0] 		# leached mass D
					net_mass_change[4] = net_mass_change[4] + 3*apparent_progress[0] 									# leached mass O
					net_mass_change[5] = net_mass_change[5] + 1*carbon_aq_sepcies_fraction[0]*apparent_progress[0] 		# leached mass H
					net_mass_change[6] = net_mass_change[6] + 1*apparent_progress[0] 									# leached mass C

					if max(abs(net_mass_change)) > torch.tensor(10**(0)): # torch.tensor(10**(-4))::
						raise ValueError('mass balance not correct in HCO3- leaching timestep ' + str(itimestep) + ' month ' + str(imonth.numpy()) + ' year ' + str(iyear.numpy()) + ' change: ' + str(net_mass_change.numpy()))

					consumed_h2o_record[itimestep, 0] = -1*(delta_species_vector[13*n_soil_layer:14*n_soil_layer, 0]*dz[0:n_soil_layer]).sum()
					
					species_vector_current = species_vector + delta_species_vector
					# species_vector_current[species_vector_current <= +0.0] = trivial_con					
					if torch.where(species_vector_current < 0)[0].shape[0] > 0 \
						or torch.isnan(species_vector_current.sum()) == True:
						raise ValueError('invalid species vector updates (leaching M2+) happened in timestep ' + str(itimestep) + ' month ' + str(imonth) + ' year ' + str(iyear))
					#end if
					
					species_vector = species_vector_current

					##--------------------plot figure 
					if is_plotting == 1 and is_weathering == 1 and iyear == 0 and imonth == 0 and itimestep == 0:

						plt.figure(figsize = (6, 12), layout = 'tight')
						axes = plt.gca()
						axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
						plt.barh(torch.arange(species_num), delta_species_vector[:, 0].detach().numpy(), edgecolor = color_scheme, color = color_scheme, linewidth = 2, tick_label = species_list)
						plt.axvline(x = 0, color = 'grey', linewidth = 2, linestyle = 'solid')
						#end for ispecies in range(len(species_list)):
						# plt.yscale('symlog')
						# plt.legend(loc = (1.1, 0.1), fontsize = 14, markerscale = 1.5, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 1)
						plt.xlabel(xlabel = 'Absolute progress', fontsize = 18)
						plt.tick_params(labelsize = 14, width = 1.5, length = 7)
						plt.savefig(fig_dir + 'species_abs_change_leach_driven.jpeg', dpi = 300)
						plt.close()

					#------------------------------------------------
					# leaching M2+
					#------------------------------------------------	
						
					rate_leach = (1/m2_turnover_time)*leach_frac_base
					# use the leaching of M2 as the progress
					progress_leach = 1*(rate_leach*species_vector[6*n_soil_layer:7*n_soil_layer])						
					#--------------------alteration matrix and directional vector
					direct_vector_leach = torch.ones(n_soil_layer*species_num, 1)*0
					# M2+ leaches with HCO3-. emperically set HCO3- balances 1/8 of the leached M2+  
					direct_vector_leach[2*n_soil_layer:3*n_soil_layer, 0] = -1 # -1/2*8 # M2+

					alter_matrix_leach = alteration_matrix(species_vector, species_num, cec_proton, soa_proton, is_use_solid_mineral)

					# species_relative_change = torch.linalg.solve(alter_matrix, direct_vector)

					max_progress = progress_leach.reshape(n_soil_layer, 1).repeat(species_num, 1)

					# converged block-step: no per-species allowance and no floor; the full progress is applied
					apparent_progress = max_progress

					# max_stock_change_constraint = -(species_vector*max_comsume_frac) # under most extreme conditions, there will sitll be 0.1% species remained after reactions
					# constraint_loc = torch.where((species_relative_change*max_progress) < 0)[0]

					# constraint_stock_change = torch.concat((abs(max_stock_change_constraint[constraint_loc]), abs(max_progress*species_relative_change)[constraint_loc]), axis = 0)
					# constraint_stock_change_final, constraint_stock_change_loc = torch.min(constraint_stock_change, dim = 0)
						
					# if (constraint_stock_change_loc + 1) > constraint_loc.shape[0]:
					# 	constraint_progress = max_progress[0].sign()*abs(constraint_stock_change_final/species_relative_change)[constraint_loc][constraint_stock_change_loc - constraint_loc.shape[0]]
					# else:
					# 	constraint_progress = max_progress[0].sign()*abs(constraint_stock_change_final/species_relative_change)[constraint_loc][constraint_stock_change_loc]
					
					# apparent_progress = constraint_progress.reshape(n_soil_layer, 1).repeat(species_num, 1)

					# update we only use apparent progress to calculate changes of species other than co2. for co2, we still use its real progress
					delta_species_vector = newton_block(species_vector, alter_matrix_leach, direct_vector_leach, apparent_progress[0, 0], block_label = 'leaching D2+, year ' + str(int(iyear)) + ' month ' + str(int(imonth)) + ' step ' + str(itimestep)) # converged block-step (damped Newton in ln n)
					
					delta_species_matrix = delta_species_vector.reshape(species_num, n_soil_layer).t() # unit mol m-3
					delta_species_matrix = delta_species_matrix[0:n_soil_layer, :]*dz[0:n_soil_layer].reshape(n_soil_layer, 1).repeat(1, species_num) # unit mol m-2
					delta_species_record[itimestep, 0, :] = delta_species_record[itimestep, 0, :] + delta_species_matrix.sum(axis = 0)
					delta_species_record[itimestep, 7, :] = delta_species_matrix.sum(axis = 0)
					leached_ion_record[itimestep, 6] = apparent_progress[0]

					net_charge_change = (delta_species_vector[:, 0]*species_charge).sum() + 2*apparent_progress[0]
					if abs(net_charge_change) > torch.tensor(10**(0)): # torch.tensor(10**(-4)):
						raise ValueError('charge balance not correct in D2+ leaching timestep ' + str(itimestep) + ' month ' + str(imonth.numpy()) + ' year ' + str(iyear.numpy()) + ' change: ' + str(net_charge_change.numpy()))

					net_mass_change = (element_mass_matrix*delta_species_vector.reshape(1, species_num).repeat([9, 1])).sum(axis = 1)
					net_mass_change[2] = net_mass_change[2] + 1*apparent_progress[0] # leached mass D

					if max(abs(net_mass_change)) > torch.tensor(10**(0)): # torch.tensor(10**(-4)):
						raise ValueError('mass balance not correct in D2+ leaching timestep ' + str(itimestep) + ' month ' + str(imonth.numpy()) + ' year ' + str(iyear.numpy()) + ' change: ' + str(net_mass_change.numpy()))

					consumed_h2o_record[itimestep, 0] = consumed_h2o_record[itimestep, 0] + (-1*(delta_species_vector[13*n_soil_layer:14*n_soil_layer, 0]*dz[0:n_soil_layer]).sum())
					
					species_vector_current = species_vector + delta_species_vector
					# species_vector_current[species_vector_current <= +0.0] = trivial_con					
					if torch.where(species_vector_current < 0)[0].shape[0] > 0 \
						or torch.isnan(species_vector_current.sum()) == True:
						raise ValueError('invalid species vector updates (leaching M2+) happened in timestep ' + str(itimestep) + ' month ' + str(imonth) + ' year ' + str(iyear))
					#end if
					species_vector = species_vector_current

					#------------------------------------------------
					# leaching Al3+
					#------------------------------------------------	
										
					current_proton_stock = species_vector[8*n_soil_layer:9*n_soil_layer, 0]*dz[0:n_soil_layer]/concentration_converter # mol m-2 
					current_water_stock =  species_vector[13*n_soil_layer:14*n_soil_layer, 0]*dz[0:n_soil_layer]/concentration_converter  # mol m-2
					current_ph = -torch.log10(current_proton_stock/(current_water_stock*18/1000)) # mol L-1

					rate_leach = (1/al_turnover_time)*leach_frac_base
					# leach Al3+, Al(OH)++, and Al(OH)4-, take Al(OH)3 as immobile

					leach_al_control = torch.tensor([1, 1, 1, 1])
					
					progress_leach = 1*(rate_leach*(species_vector[0*n_soil_layer:1*n_soil_layer]*leach_al_control[0] + 
													species_vector[1*n_soil_layer:2*n_soil_layer]*leach_al_control[1] + 
													species_vector[2*n_soil_layer:3*n_soil_layer]*leach_al_control[2] +
													species_vector[15*n_soil_layer:16*n_soil_layer]*leach_al_control[3]
													))
					al_aq_sepcies_fraction  = (species_vector[[0, 1, 2, 15], 0]/species_vector[[0, 1, 2, 15]].sum())*leach_al_control
					al_aq_sepcies_fraction = al_aq_sepcies_fraction/al_aq_sepcies_fraction.sum()

					#--------------------alteration matrix and directional vector
					direct_vector_leach = torch.ones(n_soil_layer*species_num, 1)*0
					# M2+ leaches with HCO3-. emperically set HCO3- balances 1/8 of the leached M2+  
					direct_vector_leach[0*n_soil_layer:1*n_soil_layer, 0] = -1 # Al3+
					direct_vector_leach[4*n_soil_layer:5*n_soil_layer, 0] = -1*al_aq_sepcies_fraction[1] + -4*al_aq_sepcies_fraction[2] + -3*al_aq_sepcies_fraction[3] # -gamma_leach*3 # -3 # O
					direct_vector_leach[5*n_soil_layer:6*n_soil_layer, 0] = -1*al_aq_sepcies_fraction[1] + -4*al_aq_sepcies_fraction[2] + -3*al_aq_sepcies_fraction[3] # -gamma_leach*1 # -1 # H

					# progress_leach = 1*(rate_leach*(species_vector[0*n_soil_layer:1*n_soil_layer]))
					# #--------------------alteration matrix and directional vector
					# direct_vector_leach = torch.ones(n_soil_layer*species_num, 1)*0
					# # M2+ leaches with HCO3-. emperically set HCO3- balances 1/8 of the leached M2+  
					# direct_vector_leach[0*n_soil_layer:1*n_soil_layer, 0] = -1 # Al3+

					alter_matrix_leach = alteration_matrix(species_vector, species_num, cec_proton, soa_proton, is_use_solid_mineral)

					# species_relative_change = torch.linalg.solve(alter_matrix, direct_vector)

					max_progress = progress_leach.reshape(n_soil_layer, 1).repeat(species_num, 1)

					# converged block-step: no per-species allowance and no floor; the full progress is applied
					apparent_progress = max_progress

					# max_stock_change_constraint = -(species_vector*max_comsume_frac) # under most extreme conditions, there will sitll be 0.1% species remained after reactions
					# constraint_loc = torch.where((species_relative_change*max_progress) < 0)[0]

					# constraint_stock_change = torch.concat((abs(max_stock_change_constraint[constraint_loc]), abs(max_progress*species_relative_change)[constraint_loc]), axis = 0)
					# constraint_stock_change_final, constraint_stock_change_loc = torch.min(constraint_stock_change, dim = 0)
						
					# if (constraint_stock_change_loc + 1) > constraint_loc.shape[0]:
					# 	constraint_progress = max_progress[0].sign()*abs(constraint_stock_change_final/species_relative_change)[constraint_loc][constraint_stock_change_loc - constraint_loc.shape[0]]
					# else:
					# 	constraint_progress = max_progress[0].sign()*abs(constraint_stock_change_final/species_relative_change)[constraint_loc][constraint_stock_change_loc]
					
					# apparent_progress = constraint_progress.reshape(n_soil_layer, 1).repeat(species_num, 1)

					# update we only use apparent progress to calculate changes of species other than co2. for co2, we still use its real progress
					delta_species_vector = newton_block(species_vector, alter_matrix_leach, direct_vector_leach, apparent_progress[0, 0], block_label = 'leaching Al, year ' + str(int(iyear)) + ' month ' + str(int(imonth)) + ' step ' + str(itimestep)) # converged block-step (damped Newton in ln n)
					
					delta_species_matrix = delta_species_vector.reshape(species_num, n_soil_layer).t() # unit mol m-3
					delta_species_matrix = delta_species_matrix[0:n_soil_layer, :]*dz[0:n_soil_layer].reshape(n_soil_layer, 1).repeat(1, species_num) # unit mol m-2
					delta_species_record[itimestep, 0, :] = delta_species_record[itimestep, 0, :] + delta_species_matrix.sum(axis = 0)
					delta_species_record[itimestep, 8, :] = delta_species_matrix.sum(axis = 0)
					leached_ion_record[itimestep, 0] = al_aq_sepcies_fraction[0]*apparent_progress[0] 
					leached_ion_record[itimestep, 1] = al_aq_sepcies_fraction[1]*apparent_progress[0]
					leached_ion_record[itimestep, 2] = al_aq_sepcies_fraction[2]*apparent_progress[0]
					leached_ion_record[itimestep, 15] = al_aq_sepcies_fraction[3]*apparent_progress[0]

					net_charge_change = (delta_species_vector[:, 0]*species_charge).sum() + 3*al_aq_sepcies_fraction[0]*apparent_progress[0] + 2*al_aq_sepcies_fraction[1]*apparent_progress[0] + (-1)*al_aq_sepcies_fraction[2]*apparent_progress[0]
					if abs(net_charge_change) > torch.tensor(10**(0)): # torch.tensor(10**(-4)):
						raise ValueError('charge balance not correct in Al leaching timestep ' + str(itimestep) + ' month ' + str(imonth.numpy()) + ' year ' + str(iyear.numpy()) + ' change: ' + str(net_charge_change.numpy()))

					net_mass_change = (element_mass_matrix*delta_species_vector.reshape(1, species_num).repeat([9, 1])).sum(axis = 1)
					net_mass_change[0] = net_mass_change[0] + 1*apparent_progress[0] # leached mass Al
					net_mass_change[4] = net_mass_change[4] + 1*al_aq_sepcies_fraction[1]*apparent_progress[0] + 4*al_aq_sepcies_fraction[2]*apparent_progress[0] + 3*al_aq_sepcies_fraction[3]*apparent_progress[0] # leached mass O
					net_mass_change[5] = net_mass_change[5] + 1*al_aq_sepcies_fraction[1]*apparent_progress[0] + 4*al_aq_sepcies_fraction[2]*apparent_progress[0] + 3*al_aq_sepcies_fraction[3]*apparent_progress[0] # leached mass H

					if max(abs(net_mass_change)) > torch.tensor(10**(0)): # torch.tensor(10**(-4))::
						raise ValueError('mass balance not correct in Al leaching timestep ' + str(itimestep) + ' month ' + str(imonth.numpy()) + ' year ' + str(iyear.numpy()) + ' change: ' + str(net_mass_change.numpy()))

					consumed_h2o_record[itimestep, 0] = consumed_h2o_record[itimestep, 0] + (-1*(delta_species_vector[13*n_soil_layer:14*n_soil_layer, 0]*dz[0:n_soil_layer]).sum())
					
					species_vector_current = species_vector + delta_species_vector
					# species_vector_current[species_vector_current <= +0.0] = trivial_con					
					if torch.where(species_vector_current < 0)[0].shape[0] > 0 \
						or torch.isnan(species_vector_current.sum()) == True:
						raise ValueError('invalid species vector updates (leaching Al) happened in timestep ' + str(itimestep) + ' month ' + str(imonth) + ' year ' + str(iyear))
					#end if
					species_vector = species_vector_current

					#------------------------------------------------
					# leaching H4SiO4
					#------------------------------------------------
					
					# if (species_vector/species_vector_initial)[3*n_soil_layer:4*n_soil_layer, 0] < 10**(-4):
					# 	rate_leach = 0
					# else:
					# 	rate_leach = (1/silica_turnover_time)*leach_frac_base
					rate_leach = (1/silica_turnover_time)*leach_frac_base
					# use the leaching of M2 as the progress
					progress_leach = 1*(rate_leach*species_vector[3*n_soil_layer:4*n_soil_layer])
					#--------------------alteration matrix and directional vector
					direct_vector_leach = torch.ones(n_soil_layer*species_num, 1)*0
					# M2+ leaches with HCO3-. emperically set HCO3- balances 1/8 of the leached M2+  
					direct_vector_leach[3*n_soil_layer:4*n_soil_layer, 0] = -1 # -gamma_leach*1 # -1 # Si
					direct_vector_leach[4*n_soil_layer:5*n_soil_layer, 0] = -4 # -gamma_leach*3 # -3 # O
					direct_vector_leach[5*n_soil_layer:6*n_soil_layer, 0] = -4 # -gamma_leach*1 # -1 # H

					alter_matrix_leach = alteration_matrix(species_vector, species_num, cec_proton, soa_proton, is_use_solid_mineral)

					# species_relative_change = torch.linalg.solve(alter_matrix, direct_vector)

					max_progress = progress_leach.reshape(n_soil_layer, 1).repeat(species_num, 1)

					# converged block-step: no per-species allowance and no floor; the full progress is applied
					apparent_progress = max_progress

					# max_stock_change_constraint = -(species_vector*max_comsume_frac) # under most extreme conditions, there will sitll be 0.1% species remained after reactions
					# constraint_loc = torch.where((species_relative_change*max_progress) < 0)[0]

					# constraint_stock_change = torch.concat((abs(max_stock_change_constraint[constraint_loc]), abs(max_progress*species_relative_change)[constraint_loc]), axis = 0)
					# constraint_stock_change_final, constraint_stock_change_loc = torch.min(constraint_stock_change, dim = 0)
						
					# if (constraint_stock_change_loc + 1) > constraint_loc.shape[0]:
					# 	constraint_progress = max_progress[0].sign()*abs(constraint_stock_change_final/species_relative_change)[constraint_loc][constraint_stock_change_loc - constraint_loc.shape[0]]
					# else:
					# 	constraint_progress = max_progress[0].sign()*abs(constraint_stock_change_final/species_relative_change)[constraint_loc][constraint_stock_change_loc]
					
					# apparent_progress = constraint_progress.reshape(n_soil_layer, 1).repeat(species_num, 1)

					# update we only use apparent progress to calculate changes of species other than co2. for co2, we still use its real progress
					delta_species_vector = newton_block(species_vector, alter_matrix_leach, direct_vector_leach, apparent_progress[0, 0], block_label = 'leaching Si, year ' + str(int(iyear)) + ' month ' + str(int(imonth)) + ' step ' + str(itimestep)) # converged block-step (damped Newton in ln n)
					
					delta_species_matrix = delta_species_vector.reshape(species_num, n_soil_layer).t() # unit mol m-3
					delta_species_matrix = delta_species_matrix[0:n_soil_layer, :]*dz[0:n_soil_layer].reshape(n_soil_layer, 1).repeat(1, species_num) # unit mol m-2
					delta_species_record[itimestep, 0, :] = delta_species_record[itimestep, 0, :] + delta_species_matrix.sum(axis = 0)
					delta_species_record[itimestep, 9, :] = delta_species_matrix.sum(axis = 0)

					net_charge_change = (delta_species_vector[:, 0]*species_charge).sum()
					if abs(net_charge_change) > torch.tensor(10**(0)): # torch.tensor(10**(-4)):
						raise ValueError('charge balance not correct in H4SiO4 leaching timestep ' + str(itimestep) + ' month ' + str(imonth.numpy()) + ' year ' + str(iyear.numpy()) + ' change: ' + str(net_charge_change.numpy()))

					net_mass_change = (element_mass_matrix*delta_species_vector.reshape(1, species_num).repeat([9, 1])).sum(axis = 1)
					net_mass_change[3] = net_mass_change[3] + 1*apparent_progress[0] # leached mass Si
					net_mass_change[4] = net_mass_change[4] + 4*apparent_progress[0] # leached mass O
					net_mass_change[5] = net_mass_change[5] + 4*apparent_progress[0] # leached mass H

					if max(abs(net_mass_change)) > torch.tensor(10**(0)): # torch.tensor(10**(-4))::
						raise ValueError('mass balance not correct in H4SiO4 leaching timestep ' + str(itimestep) + ' month ' + str(imonth.numpy()) + ' year ' + str(iyear.numpy()) + ' change: ' + str(net_mass_change.numpy()))

					consumed_h2o_record[itimestep, 0] = consumed_h2o_record[itimestep, 0] + (-1*(delta_species_vector[13*n_soil_layer:14*n_soil_layer, 0]*dz[0:n_soil_layer]).sum())
					
					species_vector_current = species_vector + delta_species_vector
					# species_vector_current[species_vector_current <= +0.0] = trivial_con					
					if torch.where(species_vector_current < 0)[0].shape[0] > 0 \
						or torch.isnan(species_vector_current.sum()) == True:
						raise ValueError('invalid species vector updates (leaching H4SiO4) happened in timestep ' + str(itimestep) + ' month ' + str(imonth) + ' year ' + str(iyear))
					#end if
					species_vector = species_vector_current

					#------------------------------------------------
					# nitrification
					#------------------------------------------------
					
					nitrification_rate = 6*nitrate_ratio*1000*2/days_per_year/14*10**(-4) #assume 6 kg N ha-1 yr-1 (Yu and Zhuang, BG, 2020) but different soils have different ratio for intrification to form H+ and nitrate
					
					if iyear > dust_start_year and is_nitrogen_fertilizer == True:
						# nitrification_rate = nitrification_rate + nitrogen_fertilizer_input*0.15*1000*2/days_per_year/14*10**(-4) # assume 125 kg N ha-1 yr-1 under management
						nitrification_rate = nitrification_rate + site_soil_chem_no3_leach*1000*2/days_per_year/14*10**(-4) # unit: mol H+ m-2 day-1 from nitrification, assuming each NO3- corresponds to 2 H+ generation
					
					progress_nitrate = nitrification_rate/0.3*concentration_converter # 7.3*10^(-4) mol H+ m-2 d-1 for 0 - 30cm
					
					#--------------------alteration matrix and directional vector
					direct_vector_nitrate = torch.ones(n_soil_layer*species_num, 1)*0
					direct_vector_nitrate[5*n_soil_layer:6*n_soil_layer, 0] = 1

					alter_matrix_nitrate = alteration_matrix(species_vector, species_num, cec_proton, soa_proton, is_use_solid_mineral)

					# species_relative_change = torch.linalg.solve(alter_matrix, direct_vector)

					max_progress = progress_nitrate.reshape(n_soil_layer, 1).repeat(species_num, 1)
					
					# converged block-step: no per-species allowance and no floor; the full progress is applied
					apparent_progress = max_progress

					# max_stock_change_constraint = -(species_vector*max_comsume_frac) # under most extreme conditions, there will sitll be 0.1% species remained after reactions
					# constraint_loc = torch.where((species_relative_change*max_progress) < 0)[0]

					# constraint_stock_change = torch.concat((abs(max_stock_change_constraint[constraint_loc]), abs(max_progress*species_relative_change)[constraint_loc]), axis = 0)
					# constraint_stock_change_final, constraint_stock_change_loc = torch.min(constraint_stock_change, dim = 0)
						
					# if (constraint_stock_change_loc + 1) > constraint_loc.shape[0]:
					# 	constraint_progress = max_progress[0].sign()*abs(constraint_stock_change_final/species_relative_change)[constraint_loc][constraint_stock_change_loc - constraint_loc.shape[0]]
					# else:
					# 	constraint_progress = max_progress[0].sign()*abs(constraint_stock_change_final/species_relative_change)[constraint_loc][constraint_stock_change_loc]
					
					# apparent_progress = constraint_progress.reshape(n_soil_layer, 1).repeat(species_num, 1)

					# update we only use apparent progress to calculate changes of species other than co2. for co2, we still use its real progress
					delta_species_vector = newton_block(species_vector, alter_matrix_nitrate, direct_vector_nitrate, apparent_progress[0, 0], block_label = 'nitrification, year ' + str(int(iyear)) + ' month ' + str(int(imonth)) + ' step ' + str(itimestep)) # converged block-step (damped Newton in ln n)
					
					delta_species_matrix = delta_species_vector.reshape(species_num, n_soil_layer).t() # unit mol m-3
					delta_species_matrix = delta_species_matrix[0:n_soil_layer, :]*dz[0:n_soil_layer].reshape(n_soil_layer, 1).repeat(1, species_num) # unit mol m-2
					delta_species_record[itimestep, 4, :] = delta_species_matrix.sum(axis = 0)
					leached_ion_record[itimestep, 8] = -apparent_progress[0]

					net_charge_change = (delta_species_vector[:, 0]*species_charge).sum() - apparent_progress[0]
					if abs(net_charge_change) > torch.tensor(10**(0)): # torch.tensor(10**(-4)):
						raise ValueError('charge balance not correct in nitrification timestep ' + str(itimestep) + ' month ' + str(imonth.numpy()) + ' year ' + str(iyear.numpy()) + ' change: ' + str(net_charge_change.numpy()))

					net_mass_change = (element_mass_matrix*delta_species_vector.reshape(1, species_num).repeat([9, 1])).sum(axis = 1)
					net_mass_change[5] = net_mass_change[5] - 1*apparent_progress[0] # added mass H

					if max(abs(net_mass_change)) > torch.tensor(10**(0)): # torch.tensor(10**(-4))::
						raise ValueError('mass balance not correct in nitrification timestep ' + str(itimestep) + ' month ' + str(imonth.numpy()) + ' year ' + str(iyear.numpy()) + ' change: ' + str(net_mass_change.numpy()))

					consumed_h2o_record[itimestep, 4] = -1*(delta_species_vector[13*n_soil_layer:14*n_soil_layer, 0]*dz[0:n_soil_layer]).sum()
					
					species_vector_current = species_vector + delta_species_vector
					# species_vector_current[species_vector_current <= +0.0] = trivial_con					
					if torch.where(species_vector_current < 0)[0].shape[0] > 0 \
						or torch.isnan(species_vector_current.sum()) == True:
						raise ValueError('invalid species vector updates (nitrification) happened in timestep ' + str(itimestep) + ' month ' + str(imonth) + ' year ' + str(iyear))
					#end if
					
					species_vector = species_vector_current

					if is_plotting == 1 and is_weathering == 1 and iyear == 0 and imonth == 0 and itimestep == 0:

						plt.figure(figsize = (6, 12), layout = 'tight')
						axes = plt.gca()
						axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
						plt.barh(torch.arange(species_num), delta_species_vector[:, 0].detach().numpy(), edgecolor = color_scheme, color = color_scheme, linewidth = 2, tick_label = species_list)
						plt.axvline(x = 0, color = 'grey', linewidth = 2, linestyle = 'solid')
						#end for ispecies in range(len(species_list)):
						# plt.yscale('symlog')
						# plt.legend(loc = (1.1, 0.1), fontsize = 14, markerscale = 1.5, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 1)
						plt.xlabel(xlabel = 'Absolute progress', fontsize = 18)
						plt.tick_params(labelsize = 14, width = 1.5, length = 7)
						plt.savefig(fig_dir + 'species_abs_change_nitrification_driven.jpeg', dpi = 300)
						plt.close()

					#------------------------------------------------
					# H20 updates
					#------------------------------------------------
					species_vector[13*n_soil_layer:14*n_soil_layer, 0] = month_water_frac*water_density/18*concentration_converter # unit: mol m-3 consider 1m3 water weights 10^6g, temperature has minor influence
					delta_species_record[itimestep, 2, :] = 0.0

					#------------------------------------------------
					# parent weathering driven alteration
					#------------------------------------------------					
					
					#--------------------alteration matrix and directional vector
					direct_vector_rock = torch.ones(n_soil_layer*species_num, 1)*0
					direct_vector_rock[0*n_soil_layer:1*n_soil_layer, 0] = stoichio_al_3cat_parent
					direct_vector_rock[1*n_soil_layer:2*n_soil_layer, 0] = stoichio_m_1cat_parent
					direct_vector_rock[2*n_soil_layer:3*n_soil_layer, 0] = stoichio_m_2cat_parent
					direct_vector_rock[3*n_soil_layer:4*n_soil_layer, 0] = stoichio_silicon_parent
					direct_vector_rock[4*n_soil_layer:5*n_soil_layer, 0] = stoichio_oxygen_parent
					direct_vector_rock[5*n_soil_layer:6*n_soil_layer, 0] = stoichio_hydrogen_parent
					direct_vector_rock[6*n_soil_layer:7*n_soil_layer, 0] = stoichio_carbon_parent # if this is none zero, then there is carbonate weathering

					alter_matrix = alteration_matrix(species_vector, species_num, cec_proton, soa_proton, is_use_solid_mineral)
					
					# species_relative_change = torch.linalg.solve(alter_matrix, direct_vector)
					if parent_weathering == True:
						
						#----------------------precipitation limiting scalar---------------
						delta_ph = torch.log10(species_vector_initial[8]/species_vector[8])
						ppn_limitation_scalar = torch.tensor(0).to(torch.float) # fun_ppn_limitation(delta_ph)
						#----------------------dissolution rate-------------------------
						# correction for temperature # unit mol m-2 day-1
						diss_rate_real_parent = diss_rate_ref_parent*(24/time_downscale)*torch.exp(-active_energy_parent/gas_const*(1/month_soil_temp - 1/temp_ref))
						# diss_rate_real_gibbsite = diss_rate_ref_gibbsite*torch.exp(-active_energy_dust/gas_const*(1/month_soil_temp - 1/temp_ref))
						#diss_rate_real_calcite = diss_rate_ref_calcite*torch.exp(-active_energy_dust/gas_const*(1/month_soil_temp - 1/temp_ref))
						# radius_mean_parent = 2000 # unit mu m
						
						# convert unit to (mu m)^2/m3 to m^2/m3
						surface_area_parent = 8.55/(radius_mean_parent/10**6)

						# progress constraint
						# unit: (m2 m-3) * (mol m-2 day-1)  = mol m-3 day-1
						# max_progress = surface_area*diss_proton_stochio*diss_rate_real_dust*((proton_month/(10**(proton_initial)*concentration_converter))**diss_order_dust)*concentration_converter # *(1-imbalance_ratio)**imblance_order_dust
						# max_progress = surface_area*diss_proton_stochio*diss_rate_real_dust*((proton_month/concentration_converter)**diss_order_dust)*concentration_converter # *(1-imbalance_ratio)**imblance_order_dust
						proton_activity_appro = species_vector[8*n_soil_layer:9*n_soil_layer, 0]/species_vector[13*n_soil_layer:14*n_soil_layer, 0]*1000/18 # convert unit from mol H+ / mol H2O to mol H+ / L H2O
						max_progress = surface_area_parent*diss_proton_stochio_parent*diss_rate_real_parent*((proton_activity_appro)**diss_order_parent)*(1 - ppn_limitation_scalar)*concentration_converter
						# *(1-imbalance_ratio)**imblance_order_dust
						max_progress = max_progress.reshape(n_soil_layer, 1).repeat(species_num, 1)

						# converged block-step: no per-species allowance and no floor; the progress is the rate-law progress,
						# times the soil-moisture factor soil_moisture_scalar if is_use_moisture_scalar == 1 (Eq. 25)
						apparent_progress_parent = max_progress*(soil_moisture_scalar if is_use_moisture_scalar == 1 else 1.0)

						# max_stock_change_constraint = -(species_vector*max_comsume_frac) # under most extreme conditions, there will sitll be 0.1% species remained after reactions
						# constraint_loc = torch.where((species_relative_change*max_progress) < 0)[0]

						# constraint_stock_change = torch.concat((abs(max_stock_change_constraint[constraint_loc]), abs(max_progress*species_relative_change)[constraint_loc]), axis = 0)
						# constraint_stock_change_final, constraint_stock_change_loc = torch.min(constraint_stock_change, dim = 0)
						  
						# if (constraint_stock_change_loc + 1) > constraint_loc.shape[0]:
						# 	constraint_progress = max_progress[0].sign()*abs(constraint_stock_change_final/species_relative_change)[constraint_loc][constraint_stock_change_loc - constraint_loc.shape[0]]
						# else:
						# 	constraint_progress = max_progress[0].sign()*abs(constraint_stock_change_final/species_relative_change)[constraint_loc][constraint_stock_change_loc]
						
						# apparent_progress_parent = constraint_progress.reshape(n_soil_layer, 1).repeat(species_num, 1)

						weathering_rate_record_parent[itimestep, :] = apparent_progress_parent[0:n_soil_layer, 0]/concentration_converter # unit mol m-3 day-1
						# print(abs(apparent_progress_dust[:, 0]) - abs(max_progress[:, 0]))
						# update
						delta_species_vector = newton_block(species_vector, alter_matrix, direct_vector_rock, apparent_progress_parent[0, 0], block_label = 'parent weathering, year ' + str(int(iyear)) + ' month ' + str(int(imonth)) + ' step ' + str(itimestep)) # converged block-step (damped Newton in ln n)
						# print(apparent_progress_dust[0:n_soil_layer, 0])

						if is_plotting == 1 and is_weathering == 1 and iyear == 0 and imonth == 0 and itimestep == 0:
							plt.figure(figsize = (6, 12), layout = 'tight')
							axes = plt.gca()
							axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
							plt.barh(torch.arange(species_num), delta_species_vector[:, 0].detach().numpy(), edgecolor = color_scheme, color = color_scheme, linewidth = 2, tick_label = species_list)
							plt.axvline(x = 0, color = 'grey', linewidth = 2, linestyle = 'solid')
							#end for ispecies in range(len(species_list)):
							# plt.yscale('symlog')
							# plt.legend(loc = (1.1, 0.1), fontsize = 14, markerscale = 1.5, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 1)
							plt.xlabel(xlabel = 'Absolute progress', fontsize = 18)
							plt.tick_params(labelsize = 14, width = 1.5, length = 7)
							plt.savefig(fig_dir + 'species_absolute_change_parent_driven.jpeg', dpi = 300)
							plt.close()							
					else:
						delta_species_vector = torch.zeros(species_num*n_soil_layer, 1)
					# end if dust_concentration.sum() > 0:
						
					delta_species_matrix = delta_species_vector.reshape(species_num, n_soil_layer).t() # unit mol m-3
					delta_species_matrix = delta_species_matrix[0:n_soil_layer, :]*dz[0:n_soil_layer].reshape(n_soil_layer, 1).repeat(1, species_num) # unit mol m-2
					delta_species_record[itimestep, 5, :] = delta_species_matrix.sum(axis = 0)
					
					net_charge_change = (delta_species_vector[:, 0]*species_charge).sum()
					if abs(net_charge_change) > torch.tensor(10**(0)): # torch.tensor(10**(-4)):
						raise ValueError('charge balance not correct in parent weathering timestep ' + str(itimestep) + ' month ' + str(imonth.numpy()) + ' year ' + str(iyear.numpy()) + ' change: ' + str(net_charge_change.numpy()))

					net_mass_change = (element_mass_matrix*delta_species_vector.reshape(1, species_num).repeat([9, 1])).sum(axis = 1)
					net_mass_change[0] = net_mass_change[0] - stoichio_al_3cat_parent*apparent_progress_parent[0] # added mass Al
					net_mass_change[1] = net_mass_change[1] - stoichio_m_1cat_parent*apparent_progress_parent[0] # added mass M
					net_mass_change[2] = net_mass_change[2] - stoichio_m_2cat_parent*apparent_progress_parent[0] # added mass D
					net_mass_change[3] = net_mass_change[3] - stoichio_silicon_parent*apparent_progress_parent[0] # added mass Si
					net_mass_change[4] = net_mass_change[4] - stoichio_oxygen_parent*apparent_progress_parent[0] # added mass O
					net_mass_change[5] = net_mass_change[5] - stoichio_hydrogen_parent*apparent_progress_parent[0] # added mass H
					net_mass_change[6] = net_mass_change[6] - stoichio_carbon_parent*apparent_progress_parent[0] # added mass C

					if max(abs(net_mass_change)) > torch.tensor(10**(0)): # torch.tensor(10**(-4))::
						raise ValueError('mass balance not correct in parent weathering timestep ' + str(itimestep) + ' month ' + str(imonth.numpy()) + ' year ' + str(iyear.numpy()) + ' change: ' + str(net_mass_change.numpy()))

					consumed_h2o_record[itimestep, 5] = -1*(delta_species_vector[13*n_soil_layer:14*n_soil_layer, 0]*dz[0:n_soil_layer]).sum()
					
					species_vector_current = species_vector + delta_species_vector
					
					if torch.where(species_vector_current < 0)[0].shape[0] > 0 \
						or torch.isnan(species_vector_current.sum()) == True:
						raise ValueError('invalid species vector updates (parent) happened in timestep ' + str(itimestep) + ' month ' + str(imonth) + ' year ' + str(iyear))
					#end if
		
					# species_vector_current[species_vector_current <= +0.0] = trivial_con
					species_vector = species_vector_current
					
					#------------------------------------------------
					# dust weathering driven alteration
					#------------------------------------------------					
					
					if iyear == 0 and imonth == 0:
						h20_ionization_0 = (species_vector[7*n_soil_layer:8*n_soil_layer, 0])*(species_vector[8*n_soil_layer:9*n_soil_layer, 0])
					# end if iyear == 0 and imonth == 0:
					
					h20_ionization_diff = (species_vector[7*n_soil_layer:8*n_soil_layer, 0])*(species_vector[8*n_soil_layer:9*n_soil_layer, 0])/h20_ionization_0
					
					#--------------------alteration matrix and directional vector
					direct_vector_rock = torch.ones(n_soil_layer*species_num, 1)*0
					direct_vector_rock[0*n_soil_layer:1*n_soil_layer, 0] = stoichio_al_3cat
					direct_vector_rock[1*n_soil_layer:2*n_soil_layer, 0] = stoichio_m_1cat
					direct_vector_rock[2*n_soil_layer:3*n_soil_layer, 0] = stoichio_m_2cat
					direct_vector_rock[3*n_soil_layer:4*n_soil_layer, 0] = stoichio_silicon
					direct_vector_rock[4*n_soil_layer:5*n_soil_layer, 0] = stoichio_oxygen
					direct_vector_rock[5*n_soil_layer:6*n_soil_layer, 0] = stoichio_hydrogen
					direct_vector_rock[6*n_soil_layer:7*n_soil_layer, 0] = stoichio_carbon # if this is none zero, then there is carbonate weathering

					alter_matrix = alteration_matrix(species_vector, species_num, cec_proton, soa_proton, is_use_solid_mineral)
					
					# species_relative_change = torch.linalg.solve(alter_matrix, direct_vector)
					if dust_concentration.sum() > 0.01:
						#----------------------precipitation limiting scalar---------------
						delta_ph = torch.log10(species_vector_initial[8]/species_vector[8])
						soa_concentration = (2*species_vector[18*n_soil_layer:19*n_soil_layer, 0] + species_vector[19*n_soil_layer:20*n_soil_layer, 0])/concentration_converter*dz[0:n_soil_layer]
						ca_concentration = species_vector[14*n_soil_layer:15*n_soil_layer, 0]/concentration_converter*dz[0:n_soil_layer] # unit mol m-2
						cum_weather_frac = 1 - dust_concentration/concentration_converter/dust_concentration_initial
						# print(cum_weather_frac)
						ppn_limitation_scalar = ppn_limit_index*fun_ppn_limitation(ca_concentration, cum_weather_frac, adjust_scalar_langmuir)
						# print(ppn_limitation_scalar)
						om_association_scalar = fun_om_association(ppn_limitation_scalar, soa_concentration)
						# print([ppn_limitation_scalar, om_association_scalar])
						
						om_association_scalar_record[itimestep, :] = om_association_scalar

						#----------------------saturation state scalar---------------
						if dust_type == 'olivine': # Mg2SiO4 + 4 H+ = SiO2 + 2 H2O + 2 Mg+2
							log_ion_active_product = 2*(species_vector[species_list.index('D++')]/(species_vector[species_list.index('H2O')]*18/1000)).log10() \
													+ 1*(species_vector[species_list.index('H4SiO4')]/(species_vector[species_list.index('H2O')]*18/1000)).log10() \
													- 4*(species_vector[species_list.index('H+')]/(species_vector[species_list.index('H2O')]*18/1000)).log10() 

						if dust_type == 'basalt': 
							# Labradorite Ca0.6Na0.4Al1.6Si2.4O8 does not have Ksp
							# use Anorthite to approximate CaAl2(SiO4)2 + 8 H+ = Ca+2 + 2 Al+3 + 2 SiO2 + 4 H2O
							log_ion_active_product = 1.0*(species_vector[species_list.index('D++')]/(species_vector[species_list.index('H2O')]*18/1000)).log10() \
													+ 2.0*(species_vector[species_list.index('Al+++')]/(species_vector[species_list.index('H2O')]*18/1000)).log10() \
													+ 2.0*(species_vector[species_list.index('H4SiO4')]/(species_vector[species_list.index('H2O')]*18/1000)).log10() \
													- 4.0*(species_vector[species_list.index('H+')]/(species_vector[species_list.index('H2O')]*18/1000)).log10() 
						
						if dust_type == 'lime': # CaCO3 + H+ = Ca+2 + HCO3-
							log_ion_active_product = 1.0*(species_vector[species_list.index('D++')]/(species_vector[species_list.index('H2O')]*18/1000)).log10() \
													+ 1.0*(species_vector[species_list.index('HCO3-')]/(species_vector[species_list.index('H2O')]*18/1000)).log10() \
													- 1.0*(species_vector[species_list.index('H+')]/(species_vector[species_list.index('H2O')]*18/1000)).log10() 

						omega_saturation_state = 10**((log_ion_active_product - rock_log_k))
						# print(omega_saturation_state)

						if omega_saturation_state > 1:
							omega_saturation_state = 1
						if omega_saturation_state < 0:
							omega_saturation_state = 0

						#----------------------dissolution rate-------------------------
						# correction for temperature # unit mol m-2 day-1
						diss_rate_real_dust = diss_rate_ref_dust*(24/time_downscale)*torch.exp(-active_energy_dust/gas_const*(1/month_soil_temp - 1/temp_ref))
						# diss_rate_real_gibbsite = diss_rate_ref_gibbsite*torch.exp(-active_energy_dust/gas_const*(1/month_soil_temp - 1/temp_ref))
						#diss_rate_real_calcite = diss_rate_ref_calcite*torch.exp(-active_energy_dust/gas_const*(1/month_soil_temp - 1/temp_ref))

						#------------------------ dissolved radius---------------------------
						# NOTE: we do not consider fractal effect here
						if iyear == dust_start_year and imonth == 0 and itimestep == 0:
							delta_radius = torch.zeros(n_soil_layer)
						else:
							delta_radius = torch.zeros(n_soil_layer) 
							# delta_radius = delta_radius + diss_rate_real_dust*molar_volume*10**6 # mol m-2 day-1 * m3 mol-1 = m day-1, further convert to mu m
							dust_concentration = dust_concentration -  apparent_progress_dust[0:n_soil_layer, 0]  # unit: mol m-3 
							# print(apparent_progress_dust[0:n_soil_layer, 0])
							dust_vol_frac = dust_concentration*molar_volume # unit m3 dust mineral per m3 soil   
						#end if iyear == 0 and imonth == 0:

						#-------------------surface area------------------------
						# distribution of grains radius as a log normal distribtion
						dist_mean = torch.log(torch.tensor(50)) # mean radius of grains in the dust, unit = mu m (10^-6 m)
						dust_max_radius = 3000 # unir mu m

						dist_sigma = 1 # Standard deviation of the underlying normal distribution.
						# creat likelihood function for surface area, assuming distribution of grains are lognormal
						radius_range = torch.arange(0, dust_max_radius, 0.1) # unit mu m
						fractal_scalar = 0.33 # from Beerling et al 2020
						fractal_fun = (10**(10)*(radius_range*10**(-6)))**fractal_scalar
						# probability density function of particle distribution (assumed to be log nomral with radius mean of dist_mean and standard deviation = dist_sigma)
						dist_pdf = 1/(radius_range*dist_sigma*(2*torch.pi)**0.5)*torch.exp(-(torch.log(radius_range)-dist_mean)**2/(2*dist_sigma**2))
						dist_pdf[0] = 0
						# likelihood function for surface area (4*pi*r^2) considering fractal function
						likelihood_fun = fractal_fun*(4*torch.pi*radius_range**2)*dist_pdf
						likelihood_fun[0] = 0

						# num of grains in the dust 4/3*pi*r^3*num_grain/V_dust = vol_frac_mineral, assume V_dust = 1 m3
						# here we calculate the grain number for each amended soil layers
						num_grain = 3*dust_vol_frac/(4*torch.pi*(torch.exp(dist_mean)*10**(-6))**3)
						surface_area = torch.zeros(n_soil_layer)

						for ilayer in range(min(n_soil_layer, dust_max_layer)):
							# set the limits of integration (you may choose appropriate limits) unit mu m
							lower_limit = delta_radius[ilayer]
							upper_limit = dust_max_radius
							# intergal of surface area # unit (mu m)^2
							integral_space = torch.where((radius_range > lower_limit) & (radius_range < upper_limit))
							surface_area[ilayer] = num_grain[ilayer]*torch.trapezoid(likelihood_fun[integral_space], radius_range[integral_space])
						# end for ilayer in range(dust_max_layer):
						
						# convert unit to (mu m)^2/m3 to m^2/m3
						surface_area = surface_area*(10**(-6))**2

						# progress constraint
						# unit: (m2 m-3) * (mol m-2 day-1)  = mol m-3 day-1
						# max_progress = surface_area*diss_proton_stochio*diss_rate_real_dust*((proton_month/(10**(proton_initial)*concentration_converter))**diss_order_dust)*concentration_converter # *(1-imbalance_ratio)**imblance_order_dust
						# max_progress = surface_area*diss_proton_stochio*diss_rate_real_dust*((proton_month/concentration_converter)**diss_order_dust)*concentration_converter # *(1-imbalance_ratio)**imblance_order_dust

						proton_activity_appro = species_vector[8*n_soil_layer:9*n_soil_layer, 0]/species_vector[13*n_soil_layer:14*n_soil_layer, 0]*1000/18 # convert unit from mol H+ / mol H2O to mol H+ / L H2O
						max_progress = surface_area*diss_proton_stochio*diss_rate_real_dust*((proton_activity_appro)**diss_order_dust)*(1 - ppn_limitation_scalar)*(1 - omega_saturation_state)*concentration_converter 
						max_progress = max_progress.reshape(n_soil_layer, 1).repeat(species_num, 1)
						
						# converged block-step: no per-species allowance and no floor; the progress is the rate-law progress,
						# times the soil-moisture factor soil_moisture_scalar if is_use_moisture_scalar == 1 (Eq. 25)
						apparent_progress_dust = max_progress*(soil_moisture_scalar if is_use_moisture_scalar == 1 else 1.0)

						# max_stock_change_constraint = -(species_vector*max_comsume_frac) # under most extreme conditions, there will sitll be 0.1% species remained after reactions
						# constraint_loc = torch.where((species_relative_change*max_progress) < 0)[0]

						# constraint_stock_change = torch.concat((abs(max_stock_change_constraint[constraint_loc]), abs(max_progress*species_relative_change)[constraint_loc]), axis = 0)
						# constraint_stock_change_final, constraint_stock_change_loc = torch.min(constraint_stock_change, dim = 0)
						  
						# if (constraint_stock_change_loc + 1) > constraint_loc.shape[0]:
						# 	constraint_progress = max_progress[0].sign()*abs(constraint_stock_change_final/species_relative_change)[constraint_loc][constraint_stock_change_loc - constraint_loc.shape[0]]
						# else:
						# 	constraint_progress = max_progress[0].sign()*abs(constraint_stock_change_final/species_relative_change)[constraint_loc][constraint_stock_change_loc]
						
						# apparent_progress_dust = constraint_progress.reshape(n_soil_layer, 1).repeat(species_num, 1)

						weathering_rate_record[itimestep, :] = apparent_progress_dust[0:n_soil_layer, 0]/concentration_converter # unit mol m-3 day-1
						# print(abs(apparent_progress_dust[:, 0]) - abs(max_progress[:, 0]))
						# update
						delta_species_vector = newton_block(species_vector, alter_matrix, direct_vector_rock, apparent_progress_dust[0, 0], block_label = 'dust weathering, year ' + str(int(iyear)) + ' month ' + str(int(imonth)) + ' step ' + str(itimestep)) # converged block-step (damped Newton in ln n)
						# print(apparent_progress_dust[0:n_soil_layer, 0])

						if is_plotting == 1 and is_weathering == 1 and iyear == 0 and imonth == 0 and itimestep == 0:
							plt.figure(figsize = (6, 12), layout = 'tight')
							axes = plt.gca()
							axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
							plt.barh(torch.arange(species_num), delta_species_vector[:, 0].detach().numpy(), edgecolor = color_scheme, color = color_scheme, linewidth = 2, tick_label = species_list)
							plt.axvline(x = 0, color = 'grey', linewidth = 2, linestyle = 'solid')
							#end for ispecies in range(len(species_list)):
							# plt.yscale('symlog')
							# plt.legend(loc = (1.1, 0.1), fontsize = 14, markerscale = 1.5, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 1)
							plt.xlabel(xlabel = 'Absolute progress', fontsize = 18)
							plt.tick_params(labelsize = 14, width = 1.5, length = 7)
							plt.savefig(fig_dir + 'species_absolute_change_dust_driven.jpeg', dpi = 300)
							plt.close()							
					else:
						apparent_progress_dust = torch.zeros(species_num*n_soil_layer, 1)
						delta_species_vector = torch.zeros(species_num*n_soil_layer, 1)
					# end if dust_concentration.sum() > 0:
						
					delta_species_matrix = delta_species_vector.reshape(species_num, n_soil_layer).t() # unit mol m-3
					delta_species_matrix = delta_species_matrix[0:n_soil_layer, :]*dz[0:n_soil_layer].reshape(n_soil_layer, 1).repeat(1, species_num) # unit mol m-2
					delta_species_record[itimestep, 3, :] = delta_species_matrix.sum(axis = 0)
					
					net_charge_change = (delta_species_vector[:, 0]*species_charge).sum()
					if abs(net_charge_change) > torch.tensor(10**(0)): # torch.tensor(10**(-4)):
						raise ValueError('charge balance not correct in enhanced weathering timestep ' + str(itimestep) + ' month ' + str(imonth.numpy()) + ' year ' + str(iyear.numpy()) + ' change: ' + str(net_charge_change.numpy()))

					net_mass_change = (element_mass_matrix*delta_species_vector.reshape(1, species_num).repeat([9, 1])).sum(axis = 1)
					net_mass_change[0] = net_mass_change[0] - stoichio_al_3cat*apparent_progress_dust[0] # added mass Al
					net_mass_change[1] = net_mass_change[1] - stoichio_m_1cat*apparent_progress_dust[0] # added mass M
					net_mass_change[2] = net_mass_change[2] - stoichio_m_2cat*apparent_progress_dust[0] # added mass D
					net_mass_change[3] = net_mass_change[3] - stoichio_silicon*apparent_progress_dust[0] # added mass Si
					net_mass_change[4] = net_mass_change[4] - stoichio_oxygen*apparent_progress_dust[0] # added mass O
					net_mass_change[5] = net_mass_change[5] - stoichio_hydrogen*apparent_progress_dust[0] # added mass H
					net_mass_change[6] = net_mass_change[6] - stoichio_carbon*apparent_progress_dust[0] # added mass C

					if max(abs(net_mass_change)) > torch.tensor(10**(0)): # torch.tensor(10**(-4))::
						raise ValueError('mass balance not correct in enhanced weathering timestep ' + str(itimestep) + ' month ' + str(imonth.numpy()) + ' year ' + str(iyear.numpy()) + ' change: ' + str(net_mass_change.numpy()))

					consumed_h2o_record[itimestep, 3] = -1*(delta_species_vector[13*n_soil_layer:14*n_soil_layer, 0]*dz[0:n_soil_layer]).sum()
					
					species_vector_current = species_vector + delta_species_vector
					
					if torch.where(species_vector_current < 0)[0].shape[0] > 0 \
						or torch.isnan(species_vector_current.sum()) == True:
						raise ValueError('invalid species vector updates (rock) happened in timestep ' + str(itimestep) + ' month ' + str(imonth) + ' year ' + str(iyear))
					#end if
		
					# species_vector_current[species_vector_current <= +0.0] = trivial_con
					species_vector = species_vector_current
					
					species_matrix = species_vector.reshape(species_num, n_soil_layer).t() # unit mol m-3
					species_matrix = species_matrix[0:n_soil_layer, :]*dz[0:n_soil_layer].reshape(n_soil_layer, 1).repeat(1, species_num) # unit mol m-2
					species_record[itimestep, :] = species_matrix.sum(axis = 0)
					
					# print(species_vector_current[:, 0])
					if torch.isnan(species_matrix.sum(axis = 0)[0]) == True:
						raise ValueError('nan value in species vector happened in timestep ' + str(itimestep) + ' month ' + str(imonth) + ' year ' + str(iyear))
					# end if torch.isnan(species_matrix.sum(axis = 0)[0]) == True:
					
					# redistributed co2 to all soc pools, so that the inorganic-derived co2 could be updated
					# unit converted from mol CO2 m-3 to g C m-3
					soc_co2_layer = species_vector[10*n_soil_layer:11*n_soil_layer, 0]*12.01 # unit gc/m3
					consumed_co2_record = consumed_co2_record + (-delta_species_vector[10*n_soil_layer:11*n_soil_layer, 0])
				#end if is_weathering == 1:
				
				co2_g_record[itimestep, :] = soc_co2_layer*dz[0:n_soil_layer]/concentration_converter # unit gc/m2
			# end for itimestep in 
			soc_co2_layer = soc_co2_layer/concentration_converter # unit gc/m3
			total_co2 = (soc_co2_layer*dz[0:n_soil_layer]).sum() # # unit gC/m2

			###########################################################################
			# end: weathering --- interations
			###########################################################################
			#---------------------------------------------------
			# monthly record
			#---------------------------------------------------
			soc_stock_record[iyear, imonth] = total_c_stock
			soc_resp_tot_record[iyear, imonth] = total_resp
			soc_co2_tot_record[iyear, imonth] = total_co2
			soc_co2_layer_record[:, imonth, iyear] = soc_co2_layer

			co2_mix_record[iyear, imonth, 0] = delta_co2_record[:, 0].sum() # due to mixing
			co2_mix_record[iyear, imonth, 1] = delta_co2_record[:, 1].sum() # due to resp

			if is_weathering == 1:
				species_stock[:, imonth, iyear] = species_matrix.sum(axis = 0)/concentration_converter  # unit mol m-2
				
				leached_species_accu = leached_species_accu + (-delta_species_record[:, 0, :]).sum(axis = 0)/concentration_converter # unit mol m-2
				leached_species_stock[:, imonth, iyear] = leached_species_accu # unit mol m-2
				delta_species_stock[:, :, imonth, iyear] = (-delta_species_record).sum(axis = 0)/concentration_converter # unit mol m-2
				leached_ion_stock[:, imonth, iyear] = leached_ion_record.sum(axis = 0)*dz[0:n_soil_layer]/concentration_converter # unit mol m-2
				parent_species_accu = parent_species_accu + (delta_species_record[:, 5, :]).sum(axis = 0)/concentration_converter # unit mol m-2
				parent_species_stock[:, imonth, iyear] = parent_species_accu # unit mol m-2	
				
				co2_limitation[imonth, iyear] = (consumed_co2_record[0:n_soil_layer, 0]*dz[0:n_soil_layer]).sum()/concentration_converter #apparent_diss_dust[0:n_soil_layer, 0]/max_diss_dust[0:n_soil_layer, 0]
				dust_remained[:, imonth, iyear] = dust_concentration*dz[0:n_soil_layer]/concentration_converter # unit mol m-2
				dust_consumed[:, imonth, iyear] = weathering_rate_record.sum(axis = 0)*dz[0:n_soil_layer] # unit mol m-2
				dust_consumed_parent[:, imonth, iyear] = weathering_rate_record_parent.sum(axis = 0)*dz[0:n_soil_layer] # unit mol m-2
			#end if isweathering == 1:

			# figure for proton changes
			if is_plotting == 1 and is_weathering == 1 and imonth == 11 and (iyear+1) % 20 == 0 and iyear != (transit_year_size-1):
				plt.figure(figsize = (11, 8), layout = 'tight')
				axes = plt.gca()
				axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
				plot_start_year = 5  
				for iprocess in np.arange(len(process_list)):
					process_driven_proton_change = -delta_species_stock[iprocess, species_list.index('H+'), :, plot_start_year:iyear].sum(axis = 0).detach().numpy() # positive value --> increase' neg value --> decrease
					plt.plot(torch.arange(plot_start_year, iyear), process_driven_proton_change, \
						color = color_scheme[iprocess], linestyle = 'solid', linewidth = 3, label = process_list[iprocess])
				
				net_proton_change = -delta_species_stock[1:, species_list.index('H+'), :, plot_start_year:iyear].sum(axis = (0, 1)).detach().numpy()
				plt.plot(torch.arange(plot_start_year, iyear), net_proton_change, \
						color = 'black', linestyle = 'solid', linewidth = 3, label = 'Net total')
				# plt.yscale('log')
				plt.xlabel(xlabel = 'Year from amendment', fontsize = 18)
				plt.ylabel(ylabel = 'Proton changes', fontsize = 18)
				plt.legend(loc = 'best', fontsize = 14, markerscale = 1.5, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 1)
				plt.tick_params(labelsize = 14, width = 1.5, length = 7)
				# plt.xticks((time_series-1901)[np.arange(0, (iyear+1)*month_num, 12)])

				plt.savefig(fig_dir + 'process_driven_proton_change_year_' + str(iyear.numpy()) + '.jpeg', dpi = 300)
				plt.close()

			# figure for calcite changes
			if is_use_solid_mineral == True and is_plotting == 1 and is_weathering == 1 and imonth == 11 and (iyear+1) % 20 == 0 and iyear != (transit_year_size-1):
				plt.figure(figsize = (11, 8), layout = 'tight')
				axes = plt.gca()
				axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
				plot_start_year = 5
				for iprocess in np.arange(len(process_list)):
					process_driven_proton_change = -delta_species_stock[iprocess, species_list.index('Carbonate-DCO3(s)'), :, plot_start_year:iyear].sum(axis = 0).detach().numpy() # positive value --> increase' neg value --> decrease
					plt.plot(torch.arange(plot_start_year, iyear), process_driven_proton_change, \
						color = color_scheme[iprocess], linestyle = 'solid', linewidth = 3, label = process_list[iprocess])
				
				net_proton_change = -delta_species_stock[1:, species_list.index('Carbonate-DCO3(s)'), :, plot_start_year:iyear].sum(axis = (0, 1)).detach().numpy()
				plt.plot(torch.arange(plot_start_year, iyear), net_proton_change, \
						color = 'black', linestyle = 'solid', linewidth = 3, label = 'Net total')
				# plt.yscale('log')
				plt.xlabel(xlabel = 'Year from amendment', fontsize = 18)
				plt.ylabel(ylabel = 'calcite changes', fontsize = 18)
				plt.legend(loc = 'best', fontsize = 14, markerscale = 1.5, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 1)
				plt.tick_params(labelsize = 14, width = 1.5, length = 7)
				# plt.xticks((time_series-1901)[np.arange(0, (iyear+1)*month_num, 12)])

				plt.savefig(fig_dir + 'process_driven_calcite_change_year_' + str(iyear.numpy()) + '.jpeg', dpi = 300)
				plt.close()

			# figure for CO2 changes
			if is_plotting == 1 and is_weathering == 1 and imonth == 11 and (iyear+1) % 20 == 0 and iyear != (transit_year_size-1):
				plt.figure(figsize = (11, 8), layout = 'tight')
				axes = plt.gca()
				axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
				plot_start_year = 5
				for iprocess in np.arange(len(process_list)):
					process_driven_proton_change = -delta_species_stock[iprocess, species_list.index('CO2(g)'), :, plot_start_year:iyear].sum(axis = 0).detach().numpy() # positive value --> increase' neg value --> decrease
					plt.plot(torch.arange(plot_start_year, iyear), process_driven_proton_change, \
						color = color_scheme[iprocess], linestyle = 'solid', linewidth = 3, label = process_list[iprocess])
				
				net_proton_change = -delta_species_stock[1:, species_list.index('CO2(g)'), :, plot_start_year:iyear].sum(axis = (0, 1)).detach().numpy()
				plt.plot(torch.arange(plot_start_year, iyear), net_proton_change, \
						color = 'black', linestyle = 'solid', linewidth = 3, label = 'Net total')
				# plt.yscale('log')
				plt.xlabel(xlabel = 'Year from amendment', fontsize = 18)
				plt.ylabel(ylabel = 'CO2 changes', fontsize = 18)
				plt.legend(loc = 'best', fontsize = 14, markerscale = 1.5, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 1)
				plt.tick_params(labelsize = 14, width = 1.5, length = 7)
				# plt.xticks((time_series-1901)[np.arange(0, (iyear+1)*month_num, 12)])

				plt.savefig(fig_dir + 'process_driven_co2_change_year_' + str(iyear.numpy()) + '.jpeg', dpi = 300)
				plt.close()

			# figure for relative changes
			if is_plotting == 1 and is_weathering == 1 and imonth == 11 and (iyear+1) % 20 == 0 and iyear != (transit_year_size-1):
				time_series = torch.arange(1901, (1901+transit_year_size), 1/12)
				point_interval = 3

				plt.figure(figsize = (24, 12), layout = 'tight')
				axes = plt.gca()
				axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
				for ispecies in range(species_num):
					plt.subplot(4, 6, (ispecies+1))
					# plt.title(species_list[ispecies])
					plt.plot(torch.arange(0, iyear), species_stock[ispecies, :, 0:iyear].mean(axis = 0).detach().numpy(), \
						color = color_scheme[ispecies], marker = '$'+str(ispecies)+'$', markersize = 6, linestyle = 'solid', linewidth = 1, label = species_list[ispecies])
					plt.legend(loc = 'best', fontsize = 14, markerscale = 1.5, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 8)
					if ispecies == 17:
						plt.xlabel(xlabel = 'Year from amendment', fontsize = 18)
					if ispecies == 5:
						plt.ylabel(ylabel = 'Stock (mol m-2)', fontsize = 18)
					plt.tick_params(labelsize = 14, width = 1.5, length = 7)
				plt.savefig(fig_dir + 'species_dynamics_seasonal_variation_year_' + str(iyear.numpy()) + '.jpeg', dpi = 300)
				plt.close()

				# leachate dynamics
				plt.figure(figsize = (24, 12), layout = 'tight')
				axes = plt.gca()
				axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
				for ispecies in range(species_num):
					plt.subplot(4, 6, (ispecies+1))
					# plt.title(species_list[ispecies])
					plt.plot(torch.arange(0, iyear), leached_ion_stock[ispecies, :, 0:iyear].mean(axis = 0).cumsum(dim = 0).detach().numpy(), \
						color = color_scheme[ispecies], marker = '$'+str(ispecies)+'$', markersize = 6, linestyle = 'solid', linewidth = 1, label = species_list[ispecies])
					plt.legend(loc = 'best', fontsize = 14, markerscale = 1.5, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 8)
					if ispecies == 17:
						plt.xlabel(xlabel = 'Year from amendment', fontsize = 18)
					if ispecies == 5:
						plt.ylabel(ylabel = 'Leachates stock (mol m-2)', fontsize = 18)
					plt.tick_params(labelsize = 14, width = 1.5, length = 7)
				plt.savefig(fig_dir + 'leachates_seasonal_variation_year_' + str(iyear.numpy()) + '.jpeg', dpi = 300)
				plt.close()

				# element dynamics
				plt.figure(figsize = (12, 6), layout = 'tight')
				axes = plt.gca()
				axes.spines[['top','bottom','left','right']].set_linewidth(1.5)

				al_dynamics_summary = torch.cat((species_stock_initial[0].reshape(1), species_stock[0, :, 0:iyear+1].t().reshape((-1, )))) \
										+ torch.cat((species_stock_initial[1].reshape(1), species_stock[1, :, 0:iyear+1].t().reshape((-1, )))) \
										+ torch.cat((species_stock_initial[2].reshape(1), species_stock[2, :, 0:iyear+1].t().reshape((-1, )))) \
										+ torch.cat((species_stock_initial[15].reshape(1), species_stock[15, :, 0:iyear+1].t().reshape((-1, )))) \
										+ 2*torch.cat((species_stock_initial[17].reshape(1), species_stock[17, :, 0:iyear+1].t().reshape((-1, )))) 
				
				m2_dynamics_summary = torch.cat((species_stock_initial[6].reshape(1), species_stock[6, :, 0:iyear+1].t().reshape((-1, )))) \
										+ torch.cat((species_stock_initial[14].reshape(1), species_stock[14, :, 0:iyear+1].t().reshape((-1, )))) \
										+ torch.cat((species_stock_initial[16].reshape(1), species_stock[16, :, 0:iyear+1].t().reshape((-1, )))) \
										+ torch.cat((species_stock_initial[18].reshape(1), species_stock[18, :, 0:iyear+1].t().reshape((-1, )))) 

				plt.subplot(1, 2, 1)
				plt.plot(time_series[torch.arange(0, (12*(iyear+1)+1), 1)]-1901, al_dynamics_summary.detach().numpy(), \
					color = 'black', linestyle = 'solid', linewidth = 1, label = 'Al')
				plt.legend(loc = 'best', fontsize = 14, markerscale = 1.5, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 8)
				if ispecies == 17:
					plt.xlabel(xlabel = 'Year from amendment', fontsize = 18)
				if ispecies == 5:
					plt.ylabel(ylabel = 'Stock (mol m-2)', fontsize = 18)
				plt.tick_params(labelsize = 14, width = 1.5, length = 7)

				plt.subplot(1, 2, 2)
				plt.plot(time_series[torch.arange(0, (12*(iyear+1)+1), 1)]-1901, m2_dynamics_summary.detach().numpy(), \
					color = 'black', linestyle = 'solid', linewidth = 1, label = 'D2+')
				plt.legend(loc = 'best', fontsize = 14, markerscale = 1.5, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 8)
				if ispecies == 17:
					plt.xlabel(xlabel = 'Year from amendment', fontsize = 18)
				if ispecies == 5:
					plt.ylabel(ylabel = 'Stock (mol m-2)', fontsize = 18)
				plt.tick_params(labelsize = 14, width = 1.5, length = 7)
				
				plt.savefig(fig_dir + 'key_element_seasonal_variation_year_' + str(iyear.numpy()) + '.jpeg', dpi = 300)
				plt.close()

			# figure for co2 variation
			if is_plotting == 1 and is_weathering == 1 and imonth == 11 and (iyear+1) % 20 == 0 and iyear != (transit_year_size-1):
				co2_g_series = species_stock[10, :, :].t().reshape((-1, )).detach().numpy() # mol m-2
				co2_g_adjust_series = co2_g_series/(cesm2_simu_eff_porosity*dz[0:n_soil_layer].reshape(n_soil_layer, 1, 1).repeat(1, month_num, cesm2_simu_eff_porosity.shape[2])).sum(axis = 0).t().reshape((-1, )).repeat(forcing_cycle_num) # mol m-3
				co2_p_from_g = co2_g_adjust_series*gas_const*298.15 # unit of gas_const: J (mol)-1 K-1 --> unit of co2_p_from_g: J m-3 = Pa
				
				plt.figure(figsize = (11, 8), layout = 'tight')
				axes = plt.gca()
				axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
				# plt.plot(time_series-1901, co2_p_from_aq/atm_pressure*10**6, \
				# 	color = 'blue', linestyle = 'solid', linewidth = 3, label = '$P_{CO_{2}} \ from \ aq$')
				plt.plot(time_series-1901, co2_p_from_g/atm_pressure*10**6, \
					color = 'black', linestyle = 'solid', linewidth = 3, label = '$P_{CO_{2}} \ from \ g$')
				# plt.yscale('log')
				plt.xlabel(xlabel = 'Year from amendment', fontsize = 18)
				plt.ylabel(ylabel = '$P_{CO_{2}} (ppm)$', fontsize = 18)
				# plt.legend(loc = 'best', fontsize = 14, markerscale = 1.5, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 1)
				plt.tick_params(labelsize = 14, width = 1.5, length = 7)
				# plt.xticks((time_series-1901)[np.arange(0, (iyear+1)*month_num, 12)])
				plt.savefig(fig_dir + 'retrieved_p_co2_year_' + str(iyear.numpy()) + '.jpeg', dpi = 300)
				plt.close()

			# comsumed co2 
			if is_plotting == 1 and is_weathering == 1 and imonth == 11 and (iyear+1) % 20 == 0 and iyear != (transit_year_size-1):
				plt.figure(figsize = (16, 6), layout = 'tight')
				axes = plt.gca()
				axes.spines[['top','bottom','left','right']].set_linewidth(1.5)

				plt.plot(time_series[torch.arange(0, transit_year_size*month_num, 1)]-1901, co2_limitation.t().reshape((-1, ))[torch.arange(0, transit_year_size*month_num, 1)].detach().numpy(), \
							color = 'red', marker = 'o', markersize = 8, linestyle = 'solid', linewidth = 3, label = 'Consumed')
				plt.plot(time_series[torch.arange(0, transit_year_size*month_num, 1)]-1901, species_stock[10, :, :].t().reshape((-1, ))[torch.arange(0, transit_year_size*month_num, 1)].detach().numpy(), \
							color = 'black', marker = 'o', markersize = 8, linestyle = 'solid', linewidth = 3, label = 'Stock')
				# plt.yscale('log')
				plt.legend(loc = 'best', fontsize = 14, markerscale = 1.5, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 1)
				plt.xlabel(xlabel = 'Year from amendment', fontsize = 18)
				plt.ylabel(ylabel = 'CO2 stock (mol m-2)', fontsize = 18)
				plt.tick_params(labelsize = 14, width = 1.5, length = 7)
				plt.savefig(fig_dir + 'co2_limitation_time_year_' + str(iyear.numpy()) + '.jpeg', dpi = 300)
				plt.close()


			# comsumed co2 
			if is_plotting == 1 and is_weathering == 1 and imonth == 11 and (iyear+1) % 20 == 0 and iyear != (transit_year_size-1):
				if is_use_solid_mineral == True:
					carbonate_accu = leached_ion_stock[(11, 12, 14, 20), :, :].sum(dim = (0, 1)).cumsum(dim = 0) + (species_stock[(11, 12, 14, 20), :, :].sum(dim = 0).mean(dim = 0) - species_stock_initial[[11, 12, 14, 20]].sum())
				else:
					carbonate_accu = leached_ion_stock[(11, 12, 14), :, :].sum(dim = (0, 1)).cumsum(dim = 0) + (species_stock[(11, 12, 14), :, :].sum(dim = 0).mean(dim = 0) - species_stock_initial[[11, 12, 14]].sum())
				
				carbonate_accu = carbonate_accu \
								- torch.nansum(dust_consumed, axis = (0, 1)).cumsum(dim = 0)*stoichio_carbon \
								- torch.nansum(dust_consumed_parent, axis = (0, 1)).cumsum(dim = 0)*stoichio_carbon_parent 	  # minus carbon introduced by parent and dust

				cdr_accu = carbonate_accu*44*10**4/10**6

				fig, ax1 = plt.subplots(figsize=(8, 8), layout = 'tight')
				ax2 = ax1.twinx()
				ax1.spines[['top','bottom','left','right']].set_linewidth(1.5)
				ax2.spines[['top','bottom','left','right']].set_linewidth(1.5)
				ax1.plot(np.arange(1, (transit_year_size)), cdr_accu.diff().detach().numpy(), \
					marker = 'o', markersize = 8, linestyle = 'solid', linewidth = 3, color = '#994F00', label = 'CDR')

				ax2.plot(np.arange(0, (transit_year_size)), cdr_accu.detach().numpy(), \
						marker = 'o', markersize = 8, linestyle = 'solid', linewidth = 3, color = '#006CD1', label = 'CDR')

				ax2.legend(loc = (0.1, 0.8), fontsize = 14, markerscale = 2, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 1)
				leg = ax2.get_legend()
				leg.legend_handles[0].set_color('black')
				ax1.set_xlabel(xlabel = 'Year from amendment', fontsize = 18)
				ax1.set_ylabel(ylabel = 'CDR Rate ($t CO2 \ m^{-2} \ yr^{-1}$)', fontsize = 18, color = '#994F00')
				ax1.tick_params(axis = 'y', labelcolor = '#994F00')
				ax1.tick_params(labelsize = 14, width = 1.5, length = 7)
				# ax1.set_ylim(limits_rate)
				# ax1.set_xticks(np.arange(0, (transit_year_size+1), 2))


				ax2.set_ylabel(ylabel = 'Accumulation ($t CO2 \ m^{-2}$)', fontsize = 18, color = '#006CD1')
				ax2.tick_params(axis= 'y', labelcolor = '#006CD1')
				ax2.tick_params(labelsize = 14, width = 1.5, length = 7)
				# ax2.set_ylim(limits_accu)
				
				plt.savefig(fig_dir + 'CDR_retrieved_analysis_year_' + str(iyear.numpy()) + '.jpeg', dpi = 300)
				plt.close()

			# comsumed rock dust 
			if is_plotting == 1 and is_weathering == 1 and imonth == 11 and (iyear+1) % 20 == 0 and iyear != (transit_year_size-1):
				cumsum_rock_dust = torch.nansum(dust_consumed, axis = (0, 1)).cumsum(dim = 0)
				time_series = torch.arange(1901, (1901+transit_year_size), 1/12)

				plt.figure(figsize = (10, 10), layout = 'tight')
				axes = plt.gca()
				axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
				plt.plot(torch.arange(0, transit_year_size), cumsum_rock_dust.detach().numpy(), \
					color = 'black', linestyle = 'solid', marker = '^', markersize = 10, linewidth = 3, label = 'net charges')
				
				# plt.yscale('symlog')
				plt.xlabel(xlabel = 'Year', fontsize = 18)
				plt.ylabel(ylabel = 'Consumed rock dust (mol m-2)', fontsize = 18)
				# plt.ylim([-1*10**(-6), 1*10**(-6)])
				plt.tick_params(labelsize = 14, width = 1.5, length = 7)
				# plt.xticks((time_series-1901)[np.arange(0, (iyear+2)*month_num, 12)])
				plt.savefig(fig_dir + 'consumed_rock_dust_year_' + str(iyear.numpy()) + '.jpeg', dpi = 300)
				plt.close()

			if is_plotting == 1 and is_weathering == 1 and imonth == 11 and (iyear+1) % 20 == 0 and iyear != (transit_year_size-1):
				time_series = torch.arange(1901, (1901+transit_year_size), 1/12)

				tot_delta = (-delta_species_stock[0:6, :, :, :].sum(axis = 0)) + leached_ion_stock
				net_tot_charge_change = (tot_delta*species_charge.reshape(species_num, 1, 1).repeat(1, month_num, transit_year_size)).sum(axis = 0).sum(axis = 0)

				plt.figure(figsize = (10, 10), layout = 'tight')
				axes = plt.gca()
				axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
				plt.plot(torch.arange(0, transit_year_size), net_tot_charge_change.detach().numpy(), \
					color = 'black', linestyle = 'solid', marker = '^', markersize = 10, linewidth = 3, label = 'net charges')
				
				# plt.yscale('symlog')
				plt.xlabel(xlabel = 'Year', fontsize = 18)
				plt.ylabel(ylabel = 'Charge chanegs (mol m-2)', fontsize = 18)
				plt.ylim([-1*10**(-6), 1*10**(-6)])
				plt.tick_params(labelsize = 14, width = 1.5, length = 7)
				# plt.xticks((time_series-1901)[np.arange(0, (iyear+2)*month_num, 12)])
				plt.savefig(fig_dir + 'pos_and_neg_charge_change_seasonal_variation_year_' + str(iyear.numpy()) + '.jpeg', dpi = 300)
				plt.close()

			if is_plotting == 1 and is_weathering == 1 and imonth == 11 and (iyear+1) % 20 == 0 and iyear != (transit_year_size-1):
				# pH time series
				limits_y = [3, 9]
				hydroxide_stock =  torch.concat((species_stock_initial[7].reshape(1, 1), species_stock[7, :, 0:(iyear+1)].t().reshape((month_num*(iyear+1), 1)))) # mol m-2
				proton_stock =     torch.concat((species_stock_initial[8].reshape(1, 1), species_stock[8, :, 0:(iyear+1)].t().reshape((month_num*(iyear+1), 1)))) # mol m-2
				water_stock = torch.concat((species_stock_initial[13].reshape(1, 1), species_stock[13, :, 0:(iyear+1)].t().reshape((month_num*(iyear+1), 1)))) # mol m-2
				ph_series = -torch.log10(proton_stock/(water_stock*18/1000)) # mol L-1
				# ph_series = ((hydroxide_stock/proton_stock).log10()+14)/2

				plt.figure(figsize = (13, 5), layout = 'tight')
				axes = plt.gca()
				axes.spines[['top','bottom','left','right']].set_linewidth(1.5)

				plt.plot(time_series[0:month_num*(iyear+1)]-1901, ph_series.detach().numpy()[0:month_num*(iyear+1)], \
							color = 'black', marker = 'o', markersize = 5, linestyle = 'solid', linewidth = 3)
				# plt.yscale('log')
				plt.xlabel(xlabel = 'Year from amendment', fontsize = 18)
				plt.ylabel(ylabel = 'pH', fontsize = 18)
				# plt.ylim(limits_y)
				# plt.xticks((time_series[0:month_num*(iyear+1)]-1901)[np.arange(0, month_num*(iyear+1), 12)])
				plt.tick_params(labelsize = 14, width = 1.5, length = 7)
				plt.savefig(fig_dir + 'pH_time_series_year_' + str(iyear.numpy()) + '.jpeg', dpi = 300)
				plt.close()

			if is_plotting == 1 and is_weathering == 1 and imonth == 11 and (iyear+1) % 20 == 0 and iyear != (transit_year_size-1):
				# SOC stock
				soc_series = soc_stock_record[0:(iyear+1), :].reshape((month_num*(iyear+1), 1))/1000 # unit kgc/m2

				plt.figure(figsize = (13, 5), layout = 'tight')
				axes = plt.gca()
				axes.spines[['top','bottom','left','right']].set_linewidth(1.5)

				plt.plot(time_series[0:month_num*(iyear+1)]-1901, soc_series.detach().numpy()[0:month_num*(iyear+1)], \
							color = 'black', marker = 'o', markersize = 5, linestyle = 'solid', linewidth = 3)
				# plt.yscale('log')
				plt.xlabel(xlabel = 'Year from amendment', fontsize = 18)
				plt.ylabel(ylabel = 'SOC stock (kg C m-2)', fontsize = 18)
				# plt.ylim(limits_y)
				# plt.xticks((time_series[0:month_num*(iyear+1)]-1901)[np.arange(0, month_num*(iyear+1), 12)])
				plt.tick_params(labelsize = 14, width = 1.5, length = 7)
				plt.savefig(fig_dir + 'soc_time_series_year_' + str(iyear.numpy()) + '.jpeg', dpi = 300)
				plt.close()

			# figure for co2 concentration changes
			if is_plotting == 1 and is_weathering == 0 and iyear == 0:
				co2_pressure_month = co2_g_record/12.01/dz[0:n_soil_layer].reshape(1, n_soil_layer).repeat(days_per_month[imonth]*time_downscale, 1) \
							/month_porosity.reshape(1, n_soil_layer).repeat(days_per_month[imonth]*time_downscale, 1) \
							*gas_const*month_soil_temp.reshape(1, n_soil_layer).repeat(days_per_month[imonth]*time_downscale, 1)/1000 # unit kPa # atm_pressure*10**6 # unit ppm
				
				if imonth == 0:
					co2_pressure_record_intermonth = co2_pressure_month
				else:
					co2_pressure_record_intermonth = torch.cat((co2_pressure_record_intermonth, co2_pressure_month), axis = 0)
			
				# how the co2 changes beforing strating weahtering module
				if imonth == 11:
					plt.figure(figsize = (8, 8), layout = 'tight')
					axes = plt.gca()
					axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
					plt.plot(co2_pressure_record_intermonth.detach().numpy(), linestyle = 'solid', linewidth = 3, label = ['0', '1', '2', '3', '4'])
					plt.xlabel(xlabel = 'DOY', fontsize = 18)
					plt.ylabel(ylabel = 'CO2 concentration (kPa)', fontsize = 18)
					plt.tick_params(labelsize = 14, width = 1.5, length = 7)
					plt.legend(loc = 'best', fontsize = 14, markerscale = 1.5, shadow = False, frameon = False, fancybox = False, framealpha = 0, ncols = 1)
					plt.savefig(fig_dir + 'co2_concentration_one_year_series.jpeg', dpi = 300)
					plt.close()
				# end if imonth == 0 and iyear == 0 and is_weathering == 0

			# figure for species concentration changes
			if is_plotting == 1 and is_weathering == 1 and iyear == 0:
				if imonth == 0:
					delta_species_record_intermonth = delta_species_record[:, 0:6, :].sum(axis = 1)
					species_record_intermonth = species_record
				else:
					delta_species_record_intermonth = torch.cat((delta_species_record_intermonth, delta_species_record[:, 0:6, :].sum(axis = 1)), axis = 0)
					species_record_intermonth = torch.cat((species_record_intermonth, species_record), axis = 0)

				if imonth == 11:
					time_series = torch.arange(0, 365, 1/time_downscale)
					# delta changes
					plt.figure(figsize = (24, 15), layout = 'tight')
					axes = plt.gca()
					axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
					for ispecies in range(species_num):
						plt.subplot(4, 6, (ispecies+1))
						plt.title(species_list[ispecies])
						plt.axhline(y = 0.0, color = 'grey', linestyle = 'dashed')
						plt.plot(time_series[1:], delta_species_record_intermonth[:, ispecies][1:].detach().numpy(), color = 'blue', linestyle = 'solid', label = 'delta')
						plt.plot(time_series[1:], species_record_intermonth[:, ispecies].diff().detach().numpy(), color = 'red', linestyle = 'dashed', label = 'record')
						plt.xlabel(xlabel = 'DOY', fontsize = 13)
						plt.ylabel(ylabel = 'Delta (umol m-2)', fontsize = 13)
						plt.tick_params(labelsize = 10, width = 0.5, length = 3)
						if ispecies == 0:
							plt.legend(loc = 'best')
					
					plt.savefig(fig_dir + 'species_delta_changes_one_year_series.jpeg', dpi = 300)
					plt.close()

					# accumulative changes
					plt.figure(figsize = (24, 15), layout = 'tight')
					axes = plt.gca()
					axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
					for ispecies in range(species_num):
						plt.subplot(4, 6, (ispecies+1))
						plt.title(species_list[ispecies])
						plt.plot(time_series, species_record_intermonth[:, ispecies].log10().detach().numpy(), color = 'black', linestyle = 'solid', label = 'record')
						plt.xlabel(xlabel = 'DOY', fontsize = 13)
						plt.ylabel(ylabel = 'Stock (log10 umol m-2)', fontsize = 13)
						plt.tick_params(labelsize = 10, width = 0.5, length = 3)
						if ispecies == 0:
							plt.legend(loc = 'best')
					
					plt.savefig(fig_dir + 'species_stock_changes_one_year_series.jpeg', dpi = 300)
					plt.close()

					# pH
					limits_y = [3, 9]
					proton_stock =  torch.concat((species_stock_initial[8].reshape(1, 1), species_record_intermonth[:, 8].reshape(species_record_intermonth.shape[0], 1))) # mol m-2
					water_stock = torch.concat((species_stock_initial[13].reshape(1, 1), species_record_intermonth[:, 13].reshape(species_record_intermonth.shape[0], 1))) # mol m-2
					ph_series = -torch.log10(proton_stock/(water_stock*18/1000)) # mol L-1
					
					plt.figure(figsize = (13, 5), layout = 'tight')
					axes = plt.gca()
					axes.spines[['top','bottom','left','right']].set_linewidth(1.5)

					plt.plot(time_series, ph_series[0:(ph_series.shape[0]-1)].detach().numpy(), \
								color = 'black', markersize = 5, linestyle = 'solid', linewidth = 3)
					# plt.yscale('log')
					plt.xlabel(xlabel = 'DOY', fontsize = 18)
					plt.ylabel(ylabel = 'pH', fontsize = 18)
					# plt.ylim(limits_y)
					plt.tick_params(labelsize = 14, width = 1.5, length = 7)
					plt.savefig(fig_dir + 'pH_time_series_1_years.jpeg', dpi = 300)
					plt.close()

			# if is_weathering == 1:
				# if imonth == 0:
				# 	species_record_intermonth = species_record
				# else:
				# 	species_record_intermonth = torch.cat((species_record_intermonth, species_record), axis = 0)

				# for ispecies in range(species_num):
				# 	plt.subplot(3, 5, (ispecies+1))
				# 	plt.title(species_list[ispecies])
				# 	plt.plot(species_record_intermonth[:, ispecies])
				# plt.close()
				# print('Dust remained (%): ' + str(((dust_vol_frac[0:n_soil_layer]/dust_vol_frac_initial[0:n_soil_layer]).numpy()*100).round(2)))
				# print('water content (mol m-2): ' + str(species_matrix.sum(axis = 0)[13].numpy()))
				# print('total water consumed (mol/m2): ' + str(-1*(delta_species_vector_accu[13*n_soil_layer:14*n_soil_layer, 0]*dz[0:n_soil_layer]).sum().numpy()))
				# print('consumed water due to co2, h2o, and dust (mol/2): ' + str(consumed_h2o_record.sum(axis = 0).numpy()))
				# print('apparent h20 progress: ' + str(apparent_progress_h2o[0:5, 0].numpy()))
				# print('h2o fraction: ' + str(month_water_frac.numpy()))
				# print('Consumed CO2 (mol/m3): ' + str(consumed_co2_record[0:5, 0].numpy()))
				# print(surface_area[0:5])
				# print(h20_ionization_diff[0:5])
				# print('imbalance ratio: ' + str(imbalance_ratio.numpy()))
				
			#end if is_weathering == 1:			
		# end for imonth in 1:month_num:

		if is_weathering == 1:
			# annual pH summary
			hydroxide_stock =  torch.concat((species_stock_initial[7].reshape(1, 1), species_stock[7, :, iyear].reshape(month_num, 1)))# mol m-2
			proton_stock =     torch.concat((species_stock_initial[8].reshape(1, 1), species_stock[8, :, iyear].reshape(month_num, 1)))# mol m-2
			water_stock = torch.concat((species_stock_initial[13].reshape(1, 1), species_stock[13, :, iyear].reshape(month_num, 1)))# mol m-2
			ph_series = -torch.log10(proton_stock/(water_stock*18/1000)) # mol L-1
			# ph_series = ((hydroxide_stock/proton_stock).log10()+14)/2
			# annual CDR summary
			if iyear == 0:
				if is_use_solid_mineral == True:
					carbonate_accu = leached_ion_stock[(11, 12, 14, 20), :, iyear].sum() + (species_stock[(11, 12, 14, 20), 11, iyear].sum() - species_stock_initial[[11, 12, 14, 20]].sum())
				else:
					carbonate_accu = leached_ion_stock[(11, 12, 14), :, iyear].sum() + (species_stock[(11, 12, 14), 11, iyear].sum() - species_stock_initial[[11, 12, 14]].sum())
			else:
				if is_use_solid_mineral == True:
					carbonate_accu = leached_ion_stock[(11, 12, 14, 20), :, iyear].sum() + (species_stock[(11, 12, 14, 20), 11, iyear].sum() - species_stock[(11, 12, 14, 20), 11, iyear-1].sum())
				else:
					carbonate_accu = leached_ion_stock[(11, 12, 14), :, iyear].sum() + (species_stock[(11, 12, 14), 11, iyear].sum() - species_stock[(11, 12, 14), 11, iyear-1].sum())
			annual_cdr_rate = carbonate_accu*44*10**4/10**6 # unit: mol m-2 yr-1 to t CO2 ha-1 yr-1 


			# delta_carbonate = leached_ion_stock[11, :, iyear]  # unit mol m-2
			# annual_cdr_rate = delta_carbonate.sum()*44*10**4/10**6 # unit: mol m-2 yr-1 to t CO2 ha-1 yr-1 
			
			# print info if needed
			# print('Dust remained (%): ' + str(((dust_vol_frac[0:n_soil_layer]/dust_vol_frac_initial[0:n_soil_layer]).numpy()*100).round(2)))
			# print('pH changes: ' + str(ph_series[1:, 0].numpy().round(4)))
			# print('Annual CDR (t CO2 ha-1 yr-1): ' + str(annual_cdr_rate.numpy().round(4)))
		# end if
		
		#---------------------------------------------------
		# relative changes
		#---------------------------------------------------
		if is_weathering == 1 and (iyear+1) % 10 == 0:
			species_annual_mean = species_stock.mean(axis = 1)
			species_annual_ratio_change = abs((species_annual_mean[:, iyear] - species_annual_mean[:, 0])/species_annual_mean[:, 0])/((iyear+1))
			if site_lat > 0:
				co2_g_grow_season_simu = torch.nanmean(species_stock[10, 5:8, 4:]) # mol m-2, calculating from 5th year, assuming at this time point, the system has reached to a stead state
			else:
				co2_g_grow_season_simu = torch.nanmean(species_stock[10, (0, 1, 11), 4:]) # mol m-2
			# co2_g_abs_change = abs(co2_g_grow_season_simu - soil_co2_gas_constraint)/soil_co2_gas_constraint			
			co2_g_abs_change = abs(torch.log(co2_g_grow_season_simu/soil_co2_gas_constraint)) # unit mol m-2/mol m-2

			# co2_g_abs_change = abs(torch.log(abs(torch.nanmean(species_stock[10, :, :]) - soil_co2_gas_constraint)/soil_co2_gas_constraint))
			# co2_g_abs_change = abs(torch.log(torch.nanmean(species_stock[10, :, :])/soil_co2_gas_constraint))
			if is_use_solid_mineral == True:
				cdr_cumsum_series = leached_ion_stock[(11, 12, 14, 20), :, :].sum(axis = (0, 1)).cumsum(dim = 0)[0:iyear+1] + \
									(species_stock[(11, 12, 14, 20), 11, :].sum(axis = 0) - species_stock_initial[[11, 12, 14, 20]].sum())[0:iyear+1] \
									- (torch.nansum(dust_consumed_parent, axis = (0, 1)).cumsum(axis = 0)*stoichio_carbon_parent)[0:iyear+1]
			else:
				cdr_cumsum_series = leached_ion_stock[(11, 12, 14), :, :].sum(axis = (0, 1)).cumsum(dim = 0)[0:iyear+1] + \
									(species_stock[(11, 12, 14), 11, :].sum(axis = 0) - species_stock_initial[[11, 12, 14]].sum())[0:iyear+1] \
									- (torch.nansum(dust_consumed_parent, axis = (0, 1)).cumsum(axis = 0)*stoichio_carbon_parent)[0:iyear+1]

			annual_cdr_rate = cdr_cumsum_series.diff()*44*10**4/10**6 # unit: mol m-2 yr-1 to t CO2 ha-1 yr-1 
			# plt.bar(range(0, (iyear+1)), annual_cdr_rate.detach())
			if annual_cdr_rate.nanmean() <= 0:
				parent_cdr_abs_change = 999.9*torch.ones(annual_cdr_rate.shape)
			else:
				parent_cdr_abs_change = abs(torch.log(annual_cdr_rate/parent_cdr))  # abs(annual_cdr_rate - parent_cdr)/parent_cdr
			
			if is_nn_training == 1:
				break
			# if species_annual_ratio_change.max() > 0.001 or co2_g_abs_change/soil_co2_gas_grow_season > 2 or co2_g_abs_change/soil_co2_gas_grow_season < 1/2:
			# 	break

			# # for COUSINE app only
			# if is_nn_training == 0 and is_weathering == 1:
			# 	data_dir = data_path + 'output_data/random_site/'

			# 	intermediate_results = {'species_annual_ratio_change': species_annual_ratio_change, \
			#   			'co2_g_abs_change': co2_g_abs_change, \
			# 			'soc_stock': soc_stock_record, \
			# 			'soc_resp_tot': soc_resp_tot_record, \
			# 			'soc_co2_tot': soc_co2_tot_record, \
			# 			'soc_co2_layer': soc_co2_layer_record, \
			# 			'species_stock': species_stock, \
			# 			'leached_ion_stock': leached_ion_stock, \
			# 			'delta_species_stock': delta_species_stock,\
			# 			'leached_species_stock': leached_species_stock, \
			# 			'parent_species_stock': parent_species_stock, \
			# 			'species_stock_initial': species_stock_initial, \
			# 			'co2_limitation': co2_limitation, \
			# 			'dust_concentration_initial': dust_concentration_initial, \
			# 			'dust_remained': dust_remained, \
			# 			'dust_consumed': dust_consumed, \
			# 			'dust_consumed_parent': dust_consumed_parent
			# 			}
			# 	torch.save(intermediate_results, data_dir + '/site_intermediate_results.pth')

		# end if
	# end for iyear in torch.arange(0, warmup_year_size)
	#####################################################
	# outcome
	#####################################################
	if is_warmup == 1:
		outcome = {'cpool': current_cpool, 'soc_co2_layer': soc_co2_layer, 'imbalance': total_c_change_rate}
	else:
		if is_weathering == 1:
			outcome = {'species_annual_ratio_change': species_annual_ratio_change, \
			  			'co2_g_abs_change': co2_g_abs_change, \
						'parent_cdr_abs_change': parent_cdr_abs_change, \
						'soc_stock': soc_stock_record, \
						'soc_resp_tot': soc_resp_tot_record, \
						'soc_co2_tot': soc_co2_tot_record, \
						'soc_co2_layer': soc_co2_layer_record, \
						'species_stock': species_stock, \
						'leached_ion_stock': leached_ion_stock, \
						'delta_species_stock': delta_species_stock,\
						'leached_species_stock': leached_species_stock, \
						'parent_species_stock': parent_species_stock, \
						'species_stock_initial': species_stock_initial, \
						'co2_limitation': co2_limitation, \
						'dust_concentration_initial': dust_concentration_initial, \
						'dust_remained': dust_remained, \
						'dust_consumed': dust_consumed, \
						'dust_consumed_parent': dust_consumed_parent, \
						'stoichio_element_parent': stoichio_element_parent, \
						'stoichio_element_dust': stoichio_element_dust
						}
		else:
			outcome = {'soc_stock': soc_stock_record, \
						'soc_resp_tot': soc_resp_tot_record, \
						'soc_co2_tot': soc_co2_tot_record, \
						'soc_co2_layer': soc_co2_layer_record}
		#end if is_weathering == 1:
	#end if is_warmup == 1:
	if len(newton_statistics['iterations']) > 0:
		iteration_record = torch.tensor(newton_statistics['iterations'], dtype = torch.float)
		print('converged block-step: ' + str(len(newton_statistics['iterations'])) + ' block-steps, mean iterations ' + str(round(float(iteration_record.mean()), 2)) + ', max ' + str(int(iteration_record.max())) + ', non-converged ' + str(newton_statistics['non_converged']), flush = True)
		newton_statistics['iterations'] = []
		newton_statistics['non_converged'] = 0
	return outcome
	
#end def fun_forward_simu_clm5 

##################################################
# subfunctions in inorganic carbon matrix equation
##################################################
def fun_h2o_limitation(water_content, curve_steepness, mid_point, time_downscale):
	upper_limit =  1/(1*time_downscale**1) # 1
	water_limit_factor = upper_limit/(1+torch.exp(-curve_steepness*(water_content - mid_point)))
	water_limit_factor = min(torch.tensor(0.95), water_limit_factor)
	water_limit_factor = max(torch.tensor(0.05), water_limit_factor)
	return water_limit_factor
# limitation of possible precipitation. here we use delta pH to indicate the progress of dissolution equilibrium

# def fun_ppn_limitation(delta_pH):
	# curve_steepness = 1.5
	# ppn_limit_factor = 2/(1+torch.exp(-curve_steepness*(delta_pH))) - 1
	# if ppn_limit_factor < 0:
	# 	ppn_limit_factor = 0.0
	# # end if
	# return ppn_limit_factor

def fun_ppn_limitation(ca_concentration, cum_weather_frac, adjust_scalar_langmuir):
	k_eq_langmuir = 1*adjust_scalar_langmuir
	ppn_limit_factor = ca_concentration*k_eq_langmuir/(1 + ca_concentration*k_eq_langmuir)
	
	return ppn_limit_factor


def fun_om_association(ppn_limitation_scalar, soa_concentration):
	k_eq_om = 0.01
	om_association_factor = 1 + ppn_limitation_scalar*soa_concentration*k_eq_om/(1 + soa_concentration*k_eq_om)

	return om_association_factor

def alteration_matrix(species_vector, species_num, cec_proton, soa_proton, is_use_solid_mineral):
	soil_decom_num = 1
	#------------------------------------------------
	# alteration matrix
	#------------------------------------------------
	al_3cat_month      = species_vector[0*soil_decom_num:1*soil_decom_num, 0]
	aloh_2cat_month    = species_vector[1*soil_decom_num:2*soil_decom_num, 0]
	aloh4_1an_month    = species_vector[2*soil_decom_num:3*soil_decom_num, 0]
	h4sio4_month       = species_vector[3*soil_decom_num:4*soil_decom_num, 0]
	h3sio4_1an_month   = species_vector[4*soil_decom_num:5*soil_decom_num, 0]
	mineral_1cat_month = species_vector[5*soil_decom_num:6*soil_decom_num, 0]
	mineral_2cat_month = species_vector[6*soil_decom_num:7*soil_decom_num, 0]
	hydroxide_month    = species_vector[7*soil_decom_num:8*soil_decom_num, 0]
	proton_month       = species_vector[8*soil_decom_num:9*soil_decom_num, 0]
	co2_aq_month       = species_vector[9*soil_decom_num:10*soil_decom_num, 0]
	co2_g_month        = species_vector[10*soil_decom_num:11*soil_decom_num, 0]
	bicarbo_month      = species_vector[11*soil_decom_num:12*soil_decom_num, 0]
	carbo_month        = species_vector[12*soil_decom_num:13*soil_decom_num, 0]
	h2o_month          = species_vector[13*soil_decom_num:14*soil_decom_num, 0]
	mco3_month         = species_vector[14*soil_decom_num:15*soil_decom_num, 0]
	aloh3_month        = species_vector[15*soil_decom_num:16*soil_decom_num, 0]
	sm_month           = species_vector[16*soil_decom_num:17*soil_decom_num, 0]
	sh_an_month        = species_vector[17*soil_decom_num:18*soil_decom_num, 0]
	soa_an_month       = species_vector[18*soil_decom_num:19*soil_decom_num, 0]
	soah_month         = species_vector[19*soil_decom_num:20*soil_decom_num, 0]
	if is_use_solid_mineral == True:
		carbonate_s_month  = species_vector[20*soil_decom_num:21*soil_decom_num, 0]
		gibbsite_month     = species_vector[21*soil_decom_num:22*soil_decom_num, 0]
	
	alter_matrix = torch.zeros((species_num*soil_decom_num), (species_num*soil_decom_num))
	
	# Al+++
	alter_matrix[(0*soil_decom_num) : ((0+1)*soil_decom_num) , (0*soil_decom_num) : ((0+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	
	alter_matrix[(11*soil_decom_num) : ((11+1)*soil_decom_num) , (0*soil_decom_num) : ((0+1)*soil_decom_num)] \
		= -torch.diag(1/al_3cat_month)
	alter_matrix[(12*soil_decom_num) : ((12+1)*soil_decom_num) , (0*soil_decom_num) : ((0+1)*soil_decom_num)] \
		= -torch.diag(1/al_3cat_month)
	alter_matrix[(15*soil_decom_num) : ((15+1)*soil_decom_num) , (0*soil_decom_num) : ((0+1)*soil_decom_num)] \
		= -torch.diag(1/al_3cat_month)
	alter_matrix[(16*soil_decom_num) : ((16+1)*soil_decom_num) , (0*soil_decom_num) : ((0+1)*soil_decom_num)] \
		= 2*torch.diag(1/al_3cat_month)
	
	# Al(OH)++
	alter_matrix[(0*soil_decom_num) : ((0+1)*soil_decom_num) , (1*soil_decom_num) : ((1+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(4*soil_decom_num) : ((4+1)*soil_decom_num) , (1*soil_decom_num) : ((1+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(5*soil_decom_num) : ((5+1)*soil_decom_num) , (1*soil_decom_num) : ((1+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(11*soil_decom_num) : ((11+1)*soil_decom_num) , (1*soil_decom_num) : ((1+1)*soil_decom_num)] \
		= torch.diag(1/aloh_2cat_month)
	
	# Al(OH4)-
	alter_matrix[(0*soil_decom_num) : ((0+1)*soil_decom_num) , (2*soil_decom_num) : ((2+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(4*soil_decom_num) : ((4+1)*soil_decom_num) , (2*soil_decom_num) : ((2+1)*soil_decom_num)] \
		= 4*torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(5*soil_decom_num) : ((5+1)*soil_decom_num) , (2*soil_decom_num) : ((2+1)*soil_decom_num)] \
		= 4*torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(12*soil_decom_num) : ((12+1)*soil_decom_num) , (2*soil_decom_num) : ((2+1)*soil_decom_num)] \
		= torch.diag(1/aloh4_1an_month)
	
	# H4SiO4
	alter_matrix[(3*soil_decom_num) : ((3+1)*soil_decom_num) , (3*soil_decom_num) : ((3+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(4*soil_decom_num) : ((4+1)*soil_decom_num) , (3*soil_decom_num) : ((3+1)*soil_decom_num)] \
		= 4*torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(5*soil_decom_num) : ((5+1)*soil_decom_num) , (3*soil_decom_num) : ((3+1)*soil_decom_num)] \
		= 4*torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(13*soil_decom_num) : ((13+1)*soil_decom_num) , (3*soil_decom_num) : ((3+1)*soil_decom_num)] \
		= -torch.diag(1/h4sio4_month)
	
	# H3SiO4
	alter_matrix[(3*soil_decom_num) : ((3+1)*soil_decom_num) , (4*soil_decom_num) : ((4+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(4*soil_decom_num) : ((4+1)*soil_decom_num) , (4*soil_decom_num) : ((4+1)*soil_decom_num)] \
		= 4*torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(5*soil_decom_num) : ((5+1)*soil_decom_num) , (4*soil_decom_num) : ((4+1)*soil_decom_num)] \
		= 3*torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(13*soil_decom_num) : ((13+1)*soil_decom_num) , (4*soil_decom_num) : ((4+1)*soil_decom_num)] \
		= torch.diag(1/h3sio4_1an_month)
	
	# M1
	alter_matrix[(1*soil_decom_num) : ((1+1)*soil_decom_num) , (5*soil_decom_num) : ((5+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	
	# M2
	alter_matrix[(2*soil_decom_num) : ((2+1)*soil_decom_num) , (6*soil_decom_num) : ((6+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(14*soil_decom_num) : ((14+1)*soil_decom_num) , (6*soil_decom_num) : ((6+1)*soil_decom_num)] \
		= torch.diag(1/mineral_2cat_month)
	alter_matrix[(16*soil_decom_num) : ((16+1)*soil_decom_num) , (6*soil_decom_num) : ((6+1)*soil_decom_num)] \
		= -3*torch.diag(1/mineral_2cat_month)
	alter_matrix[(18*soil_decom_num) : ((18+1)*soil_decom_num) , (6*soil_decom_num) : ((6+1)*soil_decom_num)] \
		= -1*torch.diag(1/mineral_2cat_month)
	
	# OH-
	alter_matrix[(4*soil_decom_num) : ((4+1)*soil_decom_num) , (7*soil_decom_num) : ((7+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(5*soil_decom_num) : ((5+1)*soil_decom_num) , (7*soil_decom_num) : ((7+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(7*soil_decom_num) : ((7+1)*soil_decom_num) , (7*soil_decom_num) : ((7+1)*soil_decom_num)] \
		= torch.diag(1/hydroxide_month)
	
	# H+
	alter_matrix[(5*soil_decom_num) : ((5+1)*soil_decom_num) , (8*soil_decom_num) : ((8+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(7*soil_decom_num) : ((7+1)*soil_decom_num) , (8*soil_decom_num) : ((8+1)*soil_decom_num)] \
		= torch.diag(1/proton_month)
	alter_matrix[(9*soil_decom_num) : ((9+1)*soil_decom_num) , (8*soil_decom_num) : ((8+1)*soil_decom_num)] \
		= torch.diag(1/proton_month)
	alter_matrix[(10*soil_decom_num) : ((10+1)*soil_decom_num) , (8*soil_decom_num) : ((8+1)*soil_decom_num)] \
		= torch.diag(1/proton_month)

	alter_matrix[(11*soil_decom_num) : ((11+1)*soil_decom_num) , (8*soil_decom_num) : ((8+1)*soil_decom_num)] \
		= torch.diag(1/proton_month)
	alter_matrix[(12*soil_decom_num) : ((12+1)*soil_decom_num) , (8*soil_decom_num) : ((8+1)*soil_decom_num)] \
		= 4*torch.diag(1/proton_month)

	alter_matrix[(13*soil_decom_num) : ((13+1)*soil_decom_num) , (8*soil_decom_num) : ((8+1)*soil_decom_num)] \
		= torch.diag(1/proton_month)
	
	alter_matrix[(15*soil_decom_num) : ((15+1)*soil_decom_num) , (8*soil_decom_num) : ((8+1)*soil_decom_num)] \
		= 3*torch.diag(1/proton_month)
	
	alter_matrix[(18*soil_decom_num) : ((18+1)*soil_decom_num) , (8*soil_decom_num) : ((8+1)*soil_decom_num)] \
		= 2*torch.diag(1/proton_month)

	# CO2(aq)
	alter_matrix[(4*soil_decom_num) : ((4+1)*soil_decom_num) , (9*soil_decom_num) : ((9+1)*soil_decom_num)] \
		= 2*torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(6*soil_decom_num) : ((6+1)*soil_decom_num) , (9*soil_decom_num) : ((9+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(8*soil_decom_num) : ((8+1)*soil_decom_num) , (9*soil_decom_num) : ((9+1)*soil_decom_num)] \
		= torch.diag(1/co2_aq_month)
	alter_matrix[(9*soil_decom_num) : ((9+1)*soil_decom_num) , (9*soil_decom_num) : ((9+1)*soil_decom_num)] \
		= -1*torch.diag(1/co2_aq_month)
	
	# CO2(g)
	alter_matrix[(4*soil_decom_num) : ((4+1)*soil_decom_num) , (10*soil_decom_num) : ((10+1)*soil_decom_num)] \
		= 2*torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(6*soil_decom_num) : ((6+1)*soil_decom_num) , (10*soil_decom_num) : ((10+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(8*soil_decom_num) : ((8+1)*soil_decom_num) , (10*soil_decom_num) : ((10+1)*soil_decom_num)] \
		= -1*torch.diag(1/co2_g_month)
	
	# HCO3-
	alter_matrix[(4*soil_decom_num) : ((4+1)*soil_decom_num) , (11*soil_decom_num) : ((11+1)*soil_decom_num)] \
		= 3*torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(5*soil_decom_num) : ((5+1)*soil_decom_num) , (11*soil_decom_num) : ((11+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(6*soil_decom_num) : ((6+1)*soil_decom_num) , (11*soil_decom_num) : ((11+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(9*soil_decom_num) : ((9+1)*soil_decom_num) , (11*soil_decom_num) : ((11+1)*soil_decom_num)] \
		= torch.diag(1/bicarbo_month)
	alter_matrix[(10*soil_decom_num) : ((10+1)*soil_decom_num) , (11*soil_decom_num) : ((11+1)*soil_decom_num)] \
		= -1*torch.diag(1/bicarbo_month)

	# CO3--
	alter_matrix[(4*soil_decom_num) : ((4+1)*soil_decom_num) , (12*soil_decom_num) : ((12+1)*soil_decom_num)] \
		= 3*torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(6*soil_decom_num) : ((6+1)*soil_decom_num) , (12*soil_decom_num) : ((12+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	
	alter_matrix[(10*soil_decom_num) : ((10+1)*soil_decom_num) , (12*soil_decom_num) : ((12+1)*soil_decom_num)] \
		= torch.diag(1/carbo_month)
	alter_matrix[(14*soil_decom_num) : ((14+1)*soil_decom_num) , (12*soil_decom_num) : ((12+1)*soil_decom_num)] \
		= torch.diag(1/carbo_month)
	
	# H2O
	alter_matrix[(4*soil_decom_num) : ((4+1)*soil_decom_num) , (13*soil_decom_num) : ((13+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(5*soil_decom_num) : ((5+1)*soil_decom_num) , (13*soil_decom_num) : ((13+1)*soil_decom_num)] \
		= 2*torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(7*soil_decom_num) : ((7+1)*soil_decom_num) , (13*soil_decom_num) : ((13+1)*soil_decom_num)] \
		= -1*torch.diag(1/h2o_month)
	alter_matrix[(9*soil_decom_num) : ((9+1)*soil_decom_num) , (13*soil_decom_num) : ((13+1)*soil_decom_num)] \
		= -1*torch.diag(1/h2o_month)

	alter_matrix[(11*soil_decom_num) : ((11+1)*soil_decom_num) , (13*soil_decom_num) : ((13+1)*soil_decom_num)] \
		= -1*torch.diag(1/h2o_month)
	alter_matrix[(12*soil_decom_num) : ((12+1)*soil_decom_num) , (13*soil_decom_num) : ((13+1)*soil_decom_num)] \
		= -4*torch.diag(1/h2o_month)
	alter_matrix[(15*soil_decom_num) : ((15+1)*soil_decom_num) , (13*soil_decom_num) : ((13+1)*soil_decom_num)] \
		= -3*torch.diag(1/h2o_month)

	# DCO3
	alter_matrix[(2*soil_decom_num) : ((2+1)*soil_decom_num) , (14*soil_decom_num) : ((14+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(4*soil_decom_num) : ((4+1)*soil_decom_num) , (14*soil_decom_num) : ((14+1)*soil_decom_num)] \
		= 3*torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(6*soil_decom_num) : ((6+1)*soil_decom_num) , (14*soil_decom_num) : ((14+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))

	alter_matrix[(14*soil_decom_num) : ((14+1)*soil_decom_num) , (14*soil_decom_num) : ((14+1)*soil_decom_num)] \
		= -torch.diag(1/mco3_month)
	
	if is_use_solid_mineral == True:
		alter_matrix[(20*soil_decom_num) : ((20+1)*soil_decom_num) , (14*soil_decom_num) : ((14+1)*soil_decom_num)] \
			= torch.diag(1/mco3_month)
	
	# Al(OH)3
	alter_matrix[(0*soil_decom_num) : ((0+1)*soil_decom_num) , (15*soil_decom_num) : ((15+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(4*soil_decom_num) : ((4+1)*soil_decom_num) , (15*soil_decom_num) : ((15+1)*soil_decom_num)] \
		= 3*torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(5*soil_decom_num) : ((5+1)*soil_decom_num) , (15*soil_decom_num) : ((15+1)*soil_decom_num)] \
		= 3*torch.diag(torch.ones(soil_decom_num))
	
	alter_matrix[(15*soil_decom_num) : ((15+1)*soil_decom_num) , (15*soil_decom_num) : ((15+1)*soil_decom_num)] \
		= torch.diag(1/aloh3_month)
	
	if is_use_solid_mineral == True:
		alter_matrix[(21*soil_decom_num) : ((21+1)*soil_decom_num) , (15*soil_decom_num) : ((15+1)*soil_decom_num)] \
			= torch.diag(1/aloh3_month)

	# PiM
	alter_matrix[(2*soil_decom_num) : ((2+1)*soil_decom_num) , (16*soil_decom_num) : ((16+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(16*soil_decom_num) : ((16+1)*soil_decom_num) , (16*soil_decom_num) : ((16+1)*soil_decom_num)] \
		= 3*torch.diag(1/sm_month)
	alter_matrix[(17*soil_decom_num) : ((17+1)*soil_decom_num) , (16*soil_decom_num) : ((16+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	
	# Pi3Al2
	alter_matrix[(0*soil_decom_num) : ((0+1)*soil_decom_num) , (17*soil_decom_num) : ((17+1)*soil_decom_num)] \
		= 2*torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(16*soil_decom_num) : ((16+1)*soil_decom_num) , (17*soil_decom_num) : ((17+1)*soil_decom_num)] \
		= -1*torch.diag(1/sh_an_month)
	alter_matrix[(17*soil_decom_num) : ((17+1)*soil_decom_num) , (17*soil_decom_num) : ((17+1)*soil_decom_num)] \
		= 3*torch.diag(torch.ones(soil_decom_num))
	
	# SOA
	alter_matrix[(2*soil_decom_num) : ((2+1)*soil_decom_num) , (18*soil_decom_num) : ((18+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(18*soil_decom_num) : ((18+1)*soil_decom_num) , (18*soil_decom_num) : ((18+1)*soil_decom_num)] \
		= torch.diag(1/soa_an_month)
	alter_matrix[(19*soil_decom_num) : ((19+1)*soil_decom_num) , (18*soil_decom_num) : ((18+1)*soil_decom_num)] \
		= 2*torch.diag(torch.ones(soil_decom_num))
	
	# SOAH
	alter_matrix[(5*soil_decom_num) : ((5+1)*soil_decom_num) , (19*soil_decom_num) : ((19+1)*soil_decom_num)] \
		= soa_proton*torch.diag(torch.ones(soil_decom_num))
	alter_matrix[(18*soil_decom_num) : ((18+1)*soil_decom_num) , (19*soil_decom_num) : ((19+1)*soil_decom_num)] \
		= -2*torch.diag(1/soah_month)
	alter_matrix[(19*soil_decom_num) : ((19+1)*soil_decom_num) , (19*soil_decom_num) : ((19+1)*soil_decom_num)] \
		= torch.diag(torch.ones(soil_decom_num))
	
	if is_use_solid_mineral == True:
		# Calcite-DCO3(s)
		alter_matrix[(2*soil_decom_num) : ((2+1)*soil_decom_num) , (20*soil_decom_num) : ((20+1)*soil_decom_num)] \
			=  1*torch.diag(torch.ones(soil_decom_num))
		alter_matrix[(4*soil_decom_num) : ((4+1)*soil_decom_num) , (20*soil_decom_num) : ((20+1)*soil_decom_num)] \
			=  3*torch.diag(torch.ones(soil_decom_num))
		alter_matrix[(6*soil_decom_num) : ((6+1)*soil_decom_num) , (20*soil_decom_num) : ((20+1)*soil_decom_num)] \
			=  1*torch.diag(torch.ones(soil_decom_num))

		alter_matrix[(20*soil_decom_num) : ((20+1)*soil_decom_num) , (20*soil_decom_num) : ((20+1)*soil_decom_num)] \
			= -1*torch.diag(1/carbonate_s_month)

		# Gibbsite-Al(OH)3(s)
		alter_matrix[(0*soil_decom_num) : ((0+1)*soil_decom_num) , (21*soil_decom_num) : ((21+1)*soil_decom_num)] \
			=  1*torch.diag(torch.ones(soil_decom_num))
		alter_matrix[(4*soil_decom_num) : ((4+1)*soil_decom_num) , (21*soil_decom_num) : ((21+1)*soil_decom_num)] \
			=  3*torch.diag(torch.ones(soil_decom_num))
		alter_matrix[(5*soil_decom_num) : ((5+1)*soil_decom_num) , (21*soil_decom_num) : ((21+1)*soil_decom_num)] \
			=  3*torch.diag(torch.ones(soil_decom_num))

		alter_matrix[(21*soil_decom_num) : ((21+1)*soil_decom_num) , (21*soil_decom_num) : ((21+1)*soil_decom_num)] \
			= -1*torch.diag(1/gibbsite_month)


	return alter_matrix
#end def alteration_matrix

#---------------------------------------------------------------------------------------------
# converged block-step: solve one driving-reaction block to convergence instead of taking a
# single linearized step (manuscript Eqs. 14-16). Unknown: log_species_current = ln n. Target, both measured
# from the start-of-block state n^t and the fixed progress:
#   conservation rows (MC-El): sum_k e_El,k (n_k - n_k^t) - progress*d_El = 0
#   mass-action rows (E0-E11): sum_k nu_rk (ln n_k - ln n_k^t) = 0  (each residual keeps its entry value)
# Jacobian = alteration matrix with column k multiplied by n_k. Damped Newton, no floors, no allowance.
#---------------------------------------------------------------------------------------------
mass_conservation_row_index = [0, 1, 2, 3, 4, 5, 6, 17, 19] # conservation rows (MC-El) of the alteration matrix; the others are mass-action rows (E0-E11)
newton_statistics = {'iterations': [], 'non_converged': 0, 'last_iterations': 0, 'last_converged': True}
def newton_iteration(species_start, log_species_start, alter_matrix_mass_conservation, stoichiometry_mass_const, element_net_change, mass_conservation_base, block_label, max_log_change_per_iteration, tolerance_mass_conservation, tolerance_eq_conservation, max_iteration, is_reporting = True):
	# the Newton loop of one block: returns (ln n, converged flag); ln n is None if the linear system is singular
	log_species_current = log_species_start.clone() # y^(0) = ln n^t
	iteration = 0
	is_converged = False
	while True:
		species_current = torch.exp(log_species_current)
		residual_mass_conservation = alter_matrix_mass_conservation @ (species_current - species_start) - element_net_change # measured from n^t
		residual_eq_conservation = stoichiometry_mass_const @ (log_species_current - log_species_start) # measured from n^t
		if (residual_mass_conservation.abs()/mass_conservation_base).max() < tolerance_mass_conservation and residual_eq_conservation.abs().max() < tolerance_eq_conservation:
			is_converged = True
			break
		if iteration >= max_iteration: # the final iterate has been checked above
			break
		jacobian = torch.cat([alter_matrix_mass_conservation*species_current.reshape(1, -1), stoichiometry_mass_const], 0) # alteration matrix with column k times n_k
		newton_step = torch.linalg.solve(jacobian, -torch.cat([residual_mass_conservation, residual_eq_conservation])) # J * step = -F
		if torch.isnan(newton_step.sum()):
			newton_statistics['last_iterations'] = iteration
			if is_reporting == True:
				newton_statistics['non_converged'] += 1
			return None, False
		damping_scalar = min(1.0, max_log_change_per_iteration/(newton_step.abs().max().item() + 1e-300)) # no species changes by more than a factor e^0.5 per iteration
		log_species_current = torch.clamp(log_species_current + damping_scalar*newton_step, min = -700.0) # keep ln n above -700 (n > 1e-304) so that exp() never underflows to zero; a numerical bound only, 300 orders of magnitude below any real pool, not a chemical floor
		iteration += 1
	newton_statistics['last_iterations'] = iteration
	if is_converged == False and is_reporting == True:
		newton_statistics['non_converged'] += 1
		species_current = torch.exp(log_species_current)
		residual_mass_conservation = alter_matrix_mass_conservation @ (species_current - species_start) - element_net_change
		residual_eq_conservation = stoichiometry_mass_const @ (log_species_current - log_species_start)
		raise ValueError('converged block-step did not converge after ' + str(max_iteration) + ' iterations (' + block_label + '): '
			+ 'max relative mass-conservation residual ' + '{:.2e}'.format(float((residual_mass_conservation.abs()/mass_conservation_base).max())) + ' (tolerance ' + str(tolerance_mass_conservation) + '), '
			+ 'max equilibrium residual ' + '{:.2e}'.format(float(residual_eq_conservation.abs().max())) + ' (tolerance ' + str(tolerance_eq_conservation) + ')')
	if is_reporting == True:
		newton_statistics['iterations'].append(iteration)
	return log_species_current, is_converged
#end def newton_iteration

def newton_block(species_vector, alter_matrix, direct_vector, progress, block_label = '', max_log_change_per_iteration = 0.5, tolerance_mass_conservation = 1e-10, tolerance_eq_conservation = 1e-8, max_iteration = 100, is_implicit_gradient = True, is_reporting = True):
	species_start = species_vector[:, 0] # n^t: amounts at the start of the block, the fixed reference of the whole solve
	species_num = species_start.shape[0]
	mass_conservation_rows = torch.tensor(mass_conservation_row_index)
	eq_conservation_rows = torch.tensor([irow for irow in range(species_num) if irow not in mass_conservation_row_index])
	alter_matrix_mass_conservation = alter_matrix[mass_conservation_rows] # e_El,k: atom counts, constant
	stoichiometry_mass_const = alter_matrix[eq_conservation_rows]*species_start.reshape(1, -1) # nu_rk: (nu_rk / n_k^t) times n_k^t
	element_net_change = direct_vector[mass_conservation_rows, 0]*progress # net change of each element brought by the driving reaction (positive = added, negative = leached), computed once
	log_species_start = torch.log(species_start)
	mass_conservation_base = (alter_matrix_mass_conservation.abs()*species_start.reshape(1, -1)).sum(1) + 1e-300 # total amount of each element at the start of the block: reference for the relative tolerance

	#------------------------------------------------
	# how the gradient for the BINN is obtained
	#   'none':     forward simulation, no gradient needed
	#   'implicit': BINN training (default). The iterations are not recorded by autograd; one recorded Newton step
	#               at the converged state gives the gradient, so the graph holds one solve per block
	#   'unrolled': BINN training with every iteration recorded by autograd (is_implicit_gradient = False)
	#------------------------------------------------
	if torch.is_grad_enabled() == False:
		gradient_mode = 'none'
	elif species_start.requires_grad == False and alter_matrix.requires_grad == False and element_net_change.requires_grad == False:
		gradient_mode = 'none'
	elif is_implicit_gradient == True:
		gradient_mode = 'implicit'
	else:
		gradient_mode = 'unrolled'

	#------------------------------------------------
	# Newton iterations
	#------------------------------------------------
	if gradient_mode == 'unrolled':
		log_species_current, is_converged = newton_iteration(species_start, log_species_start, alter_matrix_mass_conservation, stoichiometry_mass_const, element_net_change, mass_conservation_base, block_label, max_log_change_per_iteration, tolerance_mass_conservation, tolerance_eq_conservation, max_iteration, is_reporting)
	else:
		with torch.no_grad():
			log_species_current, is_converged = newton_iteration(species_start, log_species_start, alter_matrix_mass_conservation, stoichiometry_mass_const, element_net_change, mass_conservation_base, block_label, max_log_change_per_iteration, tolerance_mass_conservation, tolerance_eq_conservation, max_iteration, is_reporting)

	newton_statistics['last_converged'] = is_converged
	if log_species_current is None:
		return torch.zeros(species_vector.shape, dtype = species_vector.dtype) # singular system: no change, as the linearized code did

	#------------------------------------------------
	# implicit gradient: one recorded Newton step from the converged state (a constant here). Its value changes the state by
	# less than the tolerance; its derivative is -J^-1 dF/d(inputs), the exact sensitivity of the converged state to the
	# start state, the matrix and the progress
	#------------------------------------------------
	if gradient_mode == 'implicit':
		species_current = torch.exp(log_species_current)
		residual_mass_conservation = alter_matrix_mass_conservation @ (species_current - species_start) - element_net_change
		residual_eq_conservation = stoichiometry_mass_const @ (log_species_current - log_species_start)
		jacobian = torch.cat([alter_matrix_mass_conservation*species_current.reshape(1, -1), stoichiometry_mass_const], 0).detach()
		log_species_current = log_species_current - torch.linalg.solve(jacobian, torch.cat([residual_mass_conservation, residual_eq_conservation]))

	return (torch.exp(log_species_current) - species_start).reshape(-1, 1) # delta n of the block
#end def newton_block

##################################################
# sub-function in organic carbon matrix equation
##################################################

def a_matrix(fl1s1, fl2s1, fl3s2, fs1s2, fs1s3, fs2s1, fs2s3, fs3s1, fcwdl2, sand_vector):
	device = fl1s1.device
	nlevdecomp = n_soil_layer
	nspools = npool
	nspools_vr = npool_vr
	# a diagnal matrix
	a_ma_vr = torch.diag(-1*torch.ones(nspools_vr)).to(device)

	fcwdl3 = 1 - fcwdl2
	
	transfer_fraction = [fl1s1, fl2s1, fl3s2, fs1s2, fs1s3, fs2s1, fs2s3, fs3s1, fcwdl2, fcwdl3]
	for j in range(nlevdecomp):
		a_ma_vr[(3-1)*nlevdecomp+j,(1-1)*nlevdecomp+j] = transfer_fraction[8]
		a_ma_vr[(4-1)*nlevdecomp+j,(1-1)*nlevdecomp+j] = transfer_fraction[9]
		a_ma_vr[(5-1)*nlevdecomp+j,(2-1)*nlevdecomp+j] = transfer_fraction[0]
		a_ma_vr[(5-1)*nlevdecomp+j,(3-1)*nlevdecomp+j] = transfer_fraction[1]
		a_ma_vr[(5-1)*nlevdecomp+j,(6-1)*nlevdecomp+j] = transfer_fraction[5]
		a_ma_vr[(5-1)*nlevdecomp+j,(7-1)*nlevdecomp+j] = transfer_fraction[7]
		a_ma_vr[(6-1)*nlevdecomp+j,(4-1)*nlevdecomp+j] = transfer_fraction[2]
		a_ma_vr[(6-1)*nlevdecomp+j,(5-1)*nlevdecomp+j] = transfer_fraction[3]
		a_ma_vr[(7-1)*nlevdecomp+j,(5-1)*nlevdecomp+j] = transfer_fraction[4]
		a_ma_vr[(7-1)*nlevdecomp+j,(6-1)*nlevdecomp+j] = transfer_fraction[6]
	# end for ilayer
	return a_ma_vr
	
# end def a_matrix

def kk_matrix(xit, xiw, xio, xin, efolding, tau4cwd, tau4l1, tau4l2, tau4l3, tau4s1, tau4s2, tau4s3):
	device = xit.device
	
	n_soil_layer = 1
	days_per_year = 365
	# env scalars
	n_scalar = xin[:n_soil_layer] # nitrogen 
	t_scalar = xit[:n_soil_layer] # temperature
	w_scalar = xiw[:n_soil_layer] # water
	o_scalar = xio[:n_soil_layer] # oxygen
	
	# decomposition rate
	kl1 = 1/(days_per_year * tau4l1)
	kl2 = 1/(days_per_year  * tau4l2)
	kl3 = 1/(days_per_year  * tau4l3)
	ks1 = 1/(days_per_year  * tau4s1)
	ks2 = 1/(days_per_year  * tau4s2)
	ks3 = 1/(days_per_year  * tau4s3)
	kcwd = 1/(days_per_year  * tau4cwd)

	decomp_depth_efolding = efolding

	depth_scalar = torch.exp(-zsoi/decomp_depth_efolding)
	xi_tw = t_scalar*w_scalar*o_scalar
	
	kk_ma_vr = (torch.zeros([npool_vr, npool_vr])).to(device)
	for j in range(n_soil_layer):
		kk_ma_vr[0*n_soil_layer+j, 0*n_soil_layer+j] = kcwd * xi_tw[j] * depth_scalar[j]
		kk_ma_vr[1*n_soil_layer+j, 1*n_soil_layer+j] = kl1 * xi_tw[j] * depth_scalar[j] * n_scalar[j]
		kk_ma_vr[2*n_soil_layer+j, 2*n_soil_layer+j] = kl2 * xi_tw[j] * depth_scalar[j] * n_scalar[j]
		kk_ma_vr[3*n_soil_layer+j, 3*n_soil_layer+j] = kl3 * xi_tw[j] * depth_scalar[j] * n_scalar[j]
		kk_ma_vr[4*n_soil_layer+j, 4*n_soil_layer+j] = ks1 * xi_tw[j] * depth_scalar[j]
		kk_ma_vr[5*n_soil_layer+j, 5*n_soil_layer+j] = ks2 * xi_tw[j] * depth_scalar[j]
		kk_ma_vr[6*n_soil_layer+j, 6*n_soil_layer+j] = ks3 * xi_tw[j] * depth_scalar[j]
	# end for j
	return kk_ma_vr

# end def kk_matrix

def catanf(t1):
	catanf_results = 11.75 +(29.7 / np.pi) * torch.arctan( torch.tensor(np.pi) * 0.031  * ( t1 - 15.4 ))
	return catanf_results
# end def catanf


def fun_load_forcing(climate_forcing_type, data_path, cesm2_case_name_hat, site_loc, site_lon, site_lat, device):
	import netCDF4 as ncread
	
	if climate_forcing_type == 2:
		forcing_year_num = 114
	else:
		forcing_year_num = 20
	
	soil_decom_num = 1
	one_layer_compensation = 0.2800000000

	days_per_month = torch.tensor([31, 30, 31, 28, 31, 30, 31, 31, 30, 31, 30, 31]).to(device)
	dz = torch.tensor([2.000000000000000E-002, 4.000000000000000E-002, 6.000000000000000E-002, \
	8.000000000000000E-002, 0.120000000000000, 0.160000000000000, \
	0.200000000000000, 0.240000000000000, 0.280000000000000, \
	0.320000000000000, 0.360000000000000, 0.400000000000000, \
	0.440000000000000, 0.540000000000000, 0.640000000000000, \
	0.740000000000000, 0.840000000000000, 0.940000000000000, \
	1.04000000000000, 1.14000000000000, 2.39000000000000, \
	4.67553390593274, 7.63519052838329, 11.1400000000000, \
	15.1154248593737]).to(device)
	dz = dz + one_layer_compensation

	soil_forcing = loadmat(data_path + 'clm5_ouput_ncar/grid_hist_input/cesm2_hist_input_' + cesm2_case_name_hat + '_1901_2014_grid_' + str(site_loc) + '.mat')
	soil_forcing = soil_forcing['grid_hist_input']
		
	#----------------generate forcings for sic---------------
	cesm2_simu_dsl = torch.tensor(soil_forcing['cesm2_simu_dsl'][0][0][:, 0:(forcing_year_num)], dtype = torch.float64) # dry surface area thickness unit: mm
	cesm2_simu_eff_porosity = torch.tensor(soil_forcing['cesm2_simu_eff_porosity'][0][0][0:soil_decom_num, :, 0:(forcing_year_num)], dtype = torch.float64) # effective porosity = porosity - vol_ice unit: proportion
	cesm2_simu_watfc = torch.tensor(soil_forcing['cesm2_simu_watfc'][0][0][0:soil_decom_num, :, 0:(forcing_year_num)], dtype = torch.float64) # water field capacity unit: m^3/m^3
	cesm2_simu_watsat = torch.tensor(soil_forcing['cesm2_simu_watsat'][0][0][0:soil_decom_num, :, 0:(forcing_year_num)], dtype = torch.float64) # water saturated unit m^3/m^3
	cesm2_simu_h2osoi = torch.tensor(soil_forcing['cesm2_simu_soil_water_vol'][0][0][0:soil_decom_num, :, 0:(forcing_year_num)], dtype = torch.float64) #  % volumetric soil water (natural vegetated and crop landunits only) mm3/mm3
	cesm2_simu_et_tot = torch.tensor(soil_forcing['cesm2_simu_et_tot'][0][0][:, 0:(forcing_year_num)], dtype = torch.float64) # qflx_evap_soi + qflx_evap_can + qflx_tran_veg kg m-2 s-1
	cesm2_simu_rain = torch.tensor(soil_forcing['cesm2_simu_rain'][0][0][:, 0:(forcing_year_num)], dtype = torch.float64) # atmospheric rain, after rain/snow repartitioning based on temperature mm/s
	cesm2_simu_rain_from_atm = torch.tensor(soil_forcing['cesm2_simu_rain_from_atm'][0][0][:, 0:(forcing_year_num)], dtype = torch.float64) # atmospheric rain received from atmosphere (pre-repartitioning) mm/s
	
	#----------------generate forcings for soc---------------
	cesm2_simu_nbedrock = torch.tensor(soil_forcing['cesm2_simu_nbedrock'][0][0][:, 0:(forcing_year_num)], dtype = torch.float64)
	cesm2_simu_altmax = torch.tensor(soil_forcing['cesm2_simu_altmax_origin'][0][0][:, 0:(forcing_year_num)], dtype = torch.float64)
	cesm2_simu_altmax_last_year = torch.tensor(soil_forcing['cesm2_simu_altmax_last_year_origin'][0][0][:, 0:(forcing_year_num)], dtype = torch.float64)
	cesm2_simu_cellsand = torch.tensor(soil_forcing['cesm2_simu_cellsand'][0][0][0:soil_decom_num, :, 0:(forcing_year_num)], dtype = torch.float64)
	
	cesm2_simu_soil_water_potnetial = torch.tensor(soil_forcing['cesm2_simu_soil_water_potnetial_origin'][0][0][0:soil_decom_num, :, 0:(forcing_year_num)], dtype = torch.float64)
	cesm2_simu_soil_temperature = torch.tensor(soil_forcing['cesm2_simu_soil_temperature_origin'][0][0][0:soil_decom_num, :, 0:(forcing_year_num)], dtype = torch.float64)
	cesm2_simu_w_scalar = torch.tensor(soil_forcing['cesm2_simu_w_scalar_origin'][0][0][0:soil_decom_num, :, 0:(forcing_year_num)], dtype = torch.float64)
	cesm2_simu_t_scalar = torch.tensor(soil_forcing['cesm2_simu_t_scalar_origin'][0][0][0:soil_decom_num, :, 0:(forcing_year_num)], dtype = torch.float64)
	cesm2_simu_o_scalar = torch.tensor(soil_forcing['cesm2_simu_o_scalar_origin'][0][0][0:soil_decom_num, :, 0:(forcing_year_num)], dtype = torch.float64)
	cesm2_simu_n_scalar = torch.tensor(soil_forcing['cesm2_simu_n_scalar_origin'][0][0][0:soil_decom_num, :, 0:(forcing_year_num)], dtype = torch.float64)
	
	# mean monthly input from 1901 - 1920
	cesm2_simu_input_vector_litter1 = torch.tensor(soil_forcing['cesm2_simu_input_vector_litter1_origin'][0][0][0:soil_decom_num, :, 0:(forcing_year_num)], dtype = torch.float64)
	cesm2_simu_input_vector_litter2 = torch.tensor(soil_forcing['cesm2_simu_input_vector_litter2_origin'][0][0][0:soil_decom_num, :, 0:(forcing_year_num)], dtype = torch.float64)
	cesm2_simu_input_vector_litter3 = torch.tensor(soil_forcing['cesm2_simu_input_vector_litter3_origin'][0][0][0:soil_decom_num, :, 0:(forcing_year_num)], dtype = torch.float64)
	cesm2_simu_input_vector_cwd = torch.tensor(soil_forcing['cesm2_simu_input_vector_cwd_origin'][0][0][0:soil_decom_num, :, 0:(forcing_year_num)], dtype = torch.float64)# convert unit from gc/m3 to gc/m2
	for ilayer in np.arange(0, soil_decom_num):
		cesm2_simu_input_vector_litter1[ilayer, :, :] = cesm2_simu_input_vector_litter1[ilayer, :]*dz[ilayer]
		cesm2_simu_input_vector_litter2[ilayer, :, :] = cesm2_simu_input_vector_litter2[ilayer, :]*dz[ilayer]
		cesm2_simu_input_vector_litter3[ilayer, :, :] = cesm2_simu_input_vector_litter3[ilayer, :]*dz[ilayer]
		cesm2_simu_input_vector_cwd[ilayer, :, :] = cesm2_simu_input_vector_cwd[ilayer, :]*dz[ilayer]
	#end
	
	# mean annual input
	cesm2_simu_input_sum_litter1 = torch.sum(cesm2_simu_input_vector_litter1, axis = 0)
	cesm2_simu_input_sum_litter2 = torch.sum(cesm2_simu_input_vector_litter2, axis = 0)
	cesm2_simu_input_sum_litter3 = torch.sum(cesm2_simu_input_vector_litter3, axis = 0)
	cesm2_simu_input_sum_cwd = torch.sum(cesm2_simu_input_vector_cwd, axis = 0)
	
	cesm2_simu_gpp = torch.tensor(soil_forcing['cesm2_simu_gpp'][0][0][:, 0:(forcing_year_num)], dtype = torch.float64) # unit g C m-2 s-1

	# annual npp
	cesm_simu_npp = torch.tensor(soil_forcing['cesm2_simu_npp'][0][0][:, 0:(forcing_year_num)], dtype = torch.float64)
	annual_npp = torch.sum(cesm_simu_npp*(days_per_month.reshape(12, 1).repeat(1, forcing_year_num))*24*3600, axis = 0)
	annual_input = torch.sum((cesm2_simu_input_sum_litter1 + \
		cesm2_simu_input_sum_litter2 + \
		cesm2_simu_input_sum_litter3 + \
		cesm2_simu_input_sum_cwd), axis = 0)
	
	# print('the difference between annual input and npp is: ' + str((annual_input.mean() - annual_npp.mean())/annual_npp.mean()*100) + '(%)')
	
	# DriveLink: https://drive.google.com/drive/folders/1el0l4ZERobhsBckmxP82MHT-UBZBoixo?usp=sharing
	# actual evaportranspiration (AET) original units: mm.month-1 saptial resolution: 0.25 deg
	cesm2_simu_aet = load_site_aet(data_path, site_lon, site_lat) # GLEAM v3.6a 1980-1999 at the site
	# # precipitation from dataset MSWEP: unit: mm d-1
	# cesm2_simu_prec = np.ones((12, 20))*np.nan
	# for iyear in range(20):
	# 	year_name = 1980+iyear
	# 	nc_data_middle = ncread.Dataset(data_path + '../AET/MSWEP_PREC_MONTH/Precipitation_' + str(year_name) + '.nc')
	# 	lon_coord = np.ma.getdata(nc_data_middle['lon'][:])
	# 	lat_coord = np.ma.getdata(nc_data_middle['lat'][:])
	# 	site_lon_loc = (abs(lon_coord - site_lon)).argmin()
	# 	site_lat_loc = (abs(lat_coord - site_lat)).argmin()
	# 	cesm2_simu_prec[:, iyear] = np.ma.getdata(nc_data_middle['precipitation'][:, site_lat_loc, site_lon_loc])
	# nc_data_middle.close()

	###########################################
	#  convert all input to 20-year mean
	###########################################
	if climate_forcing_type == 0:
		# for soc
		cesm2_simu_input_sum_cwd = cesm2_simu_input_sum_cwd.mean(axis = 1).reshape([12, 1]).repeat([1, forcing_year_num])
		cesm2_simu_input_sum_litter1 = cesm2_simu_input_sum_litter1.mean(axis = 1).reshape([12, 1]).repeat([1, forcing_year_num])
		cesm2_simu_input_sum_litter2 = cesm2_simu_input_sum_litter2.mean(axis = 1).reshape([12, 1]).repeat([1, forcing_year_num])
		cesm2_simu_input_sum_litter3 = cesm2_simu_input_sum_litter3.mean(axis = 1).reshape([12, 1]).repeat([1, forcing_year_num])
		cesm2_simu_altmax_last_year = cesm2_simu_altmax_last_year.mean(axis = 1).reshape([12, 1]).repeat([1, forcing_year_num])
		cesm2_simu_altmax = cesm2_simu_altmax.mean(axis = 1).reshape([12, 1]).repeat([1, forcing_year_num])
		cesm2_simu_nbedrock = cesm2_simu_nbedrock.mean(axis = 1).reshape([12, 1]).repeat([1, forcing_year_num])
		
		cesm2_simu_o_scalar = cesm2_simu_o_scalar.mean(axis = 2).reshape([1, 12, 1]).repeat([1, 1, forcing_year_num])
		cesm2_simu_n_scalar = cesm2_simu_n_scalar.mean(axis = 2).reshape([1, 12, 1]).repeat([1, 1, forcing_year_num])
		cesm2_simu_cellsand = cesm2_simu_cellsand.mean(axis = 2).reshape([1, 12, 1]).repeat([1, 1, forcing_year_num])
		cesm2_simu_soil_temperature = cesm2_simu_soil_temperature.mean(axis = 2).reshape([1, 12, 1]).repeat([1, 1, forcing_year_num])
		cesm2_simu_w_scalar = cesm2_simu_w_scalar.mean(axis = 2).reshape([1, 12, 1]).repeat([1, 1, forcing_year_num])

		cesm2_simu_gpp = cesm2_simu_gpp.mean(axis = 1).reshape([12, 1]).repeat([1, forcing_year_num])


		# for sic
		cesm2_simu_dsl = cesm2_simu_dsl.mean(axis = 1).reshape([12, 1]).repeat([1, forcing_year_num])

		cesm2_simu_eff_porosity = cesm2_simu_eff_porosity.mean(axis = 2).reshape([1, 12, 1]).repeat([1, 1, forcing_year_num])
		cesm2_simu_watfc = cesm2_simu_watfc.mean(axis = 2).reshape([1, 12, 1]).repeat([1, 1, forcing_year_num])
		cesm2_simu_watsat = cesm2_simu_watsat.mean(axis = 2).reshape([1, 12, 1]).repeat([1, 1, forcing_year_num])
		cesm2_simu_h2osoi = cesm2_simu_h2osoi.mean(axis = 2).reshape([1, 12, 1]).repeat([1, 1, forcing_year_num])

		cesm2_simu_et_tot = cesm2_simu_et_tot.mean(axis = 1).reshape([12, 1]).repeat([1, forcing_year_num])
		cesm2_simu_rain = cesm2_simu_rain.mean(axis = 1).reshape([12, 1]).repeat([1, forcing_year_num])
		cesm2_simu_rain_from_atm = cesm2_simu_rain_from_atm.mean(axis = 1).reshape([12, 1]).repeat([1, forcing_year_num])
		cesm2_simu_aet = cesm2_simu_aet.mean(axis = 1).reshape([12, 1]).repeat([1, forcing_year_num])
	

	###########################################
	#  time series shift
	###########################################
	# shift_point = 6
	# # for soc
	# cesm2_simu_input_sum_cwd = fun_shift_series(cesm2_simu_input_sum_cwd, shift_point, forcing_year_num)
	# cesm2_simu_input_sum_litter1 = fun_shift_series(cesm2_simu_input_sum_litter1, shift_point, forcing_year_num)
	# cesm2_simu_input_sum_litter2 = fun_shift_series(cesm2_simu_input_sum_litter2, shift_point, forcing_year_num)
	# cesm2_simu_input_sum_litter3 = fun_shift_series(cesm2_simu_input_sum_litter3, shift_point, forcing_year_num)
	# cesm2_simu_altmax_last_year = fun_shift_series(cesm2_simu_altmax_last_year, shift_point, forcing_year_num)
	# cesm2_simu_altmax = fun_shift_series(cesm2_simu_altmax, shift_point, forcing_year_num)
	# cesm2_simu_nbedrock = fun_shift_series(cesm2_simu_nbedrock, shift_point, forcing_year_num)
	
	# cesm2_simu_o_scalar = fun_shift_series(cesm2_simu_o_scalar, shift_point, forcing_year_num)
	# cesm2_simu_n_scalar = fun_shift_series(cesm2_simu_n_scalar, shift_point, forcing_year_num)
	# cesm2_simu_cellsand = fun_shift_series(cesm2_simu_cellsand, shift_point, forcing_year_num)
	# cesm2_simu_soil_temperature = fun_shift_series(cesm2_simu_soil_temperature, shift_point, forcing_year_num)
	# cesm2_simu_w_scalar = fun_shift_series(cesm2_simu_w_scalar, shift_point, forcing_year_num)
	
	# cesm2_simu_gpp = fun_shift_series(cesm2_simu_gpp, shift_point, forcing_year_num)
		
	# # for sic
	# cesm2_simu_dsl = fun_shift_series(cesm2_simu_dsl, shift_point, forcing_year_num)

	# cesm2_simu_eff_porosity = fun_shift_series(cesm2_simu_eff_porosity, shift_point, forcing_year_num)
	# cesm2_simu_watfc = fun_shift_series(cesm2_simu_watfc, shift_point, forcing_year_num)
	# cesm2_simu_watsat = fun_shift_series(cesm2_simu_watsat, shift_point, forcing_year_num)
	# cesm2_simu_h2osoi = fun_shift_series(cesm2_simu_h2osoi, shift_point, forcing_year_num)

	# cesm2_simu_et_tot = fun_shift_series(cesm2_simu_et_tot, shift_point, forcing_year_num)
	# cesm2_simu_rain = fun_shift_series(cesm2_simu_rain, shift_point, forcing_year_num)
	# cesm2_simu_rain_from_atm = fun_shift_series(cesm2_simu_rain_from_atm, shift_point, forcing_year_num)
	# cesm2_simu_aet = fun_shift_series(cesm2_simu_aet, shift_point, 21)

	forcing_data_soc = {'cesm2_simu_input_sum_cwd': cesm2_simu_input_sum_cwd, \
						'cesm2_simu_input_sum_litter1': cesm2_simu_input_sum_litter1, \
						'cesm2_simu_input_sum_litter2': cesm2_simu_input_sum_litter2, \
						'cesm2_simu_input_sum_litter3': cesm2_simu_input_sum_litter3,\
						'cesm2_simu_altmax_last_year': cesm2_simu_altmax_last_year, \
						'cesm2_simu_altmax': cesm2_simu_altmax, \
						'cesm2_simu_nbedrock': cesm2_simu_nbedrock, \
						'cesm2_simu_o_scalar': cesm2_simu_o_scalar, \
						'cesm2_simu_n_scalar': cesm2_simu_n_scalar, \
						'cesm2_simu_cellsand': cesm2_simu_cellsand, \
						'cesm2_simu_soil_temperature': cesm2_simu_soil_temperature, \
						'cesm2_simu_w_scalar': cesm2_simu_w_scalar, \
						'cesm2_simu_gpp': cesm2_simu_gpp \
						}

	forcing_data_sic = {'cesm2_simu_dsl': cesm2_simu_dsl, \
						'cesm2_simu_eff_porosity': cesm2_simu_eff_porosity, \
						'cesm2_simu_watfc': cesm2_simu_watfc, \
						'cesm2_simu_watsat': cesm2_simu_watsat, \
						'cesm2_simu_h2osoi': cesm2_simu_h2osoi, \
						'cesm2_simu_et_tot': cesm2_simu_et_tot, \
						'cesm2_simu_rain': cesm2_simu_rain, \
						'cesm2_simu_rain_from_atm': cesm2_simu_rain_from_atm, \
						'cesm2_simu_aet': cesm2_simu_aet \
						}
	
	return [forcing_data_soc, forcing_data_sic]
#end def fun_load_forcing():

def fun_shift_series(input_data, shift_point, original_year_size):
	ilayer = 0
	if input_data.dim() == 3:
		flat_data = input_data[ilayer, :, :].t().flatten() # Size: [240]
	else:
		flat_data = input_data.t().flatten() # Size: [240]

	# Trim the edges
	output_data = flat_data[shift_point:(flat_data.shape[0]-shift_point)] # Size: [228]
	# Each column is now a full July-to-June cycle
	if input_data.dim() == 3:
		output_data = output_data.reshape([(original_year_size-1), 12]).t().repeat([1, 1, 1])
	else:
		output_data = output_data.reshape([(original_year_size-1), 12]).t()
	
	return output_data
