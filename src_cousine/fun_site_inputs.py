# COUSINE (COUpled Soil Inorganic-organic carbon model for eNhanced wEathering)
# Copyright (c) 2026 Cornell University. All Rights Reserved. Cornell docket D-11800.
# Permission to use for educational, research and non-profit purposes only; see LICENSE for the terms,
# the commercial-licensing contact and the disclaimers. The concepts used in this work are patent pending.

# Site inputs of the three manuscript sites, read from the site-only data folder (its layout mirrors the global datasets).
# Data root: the environment variable COUSINE_DATA_PATH.
import os
import numpy as np
import torch
from pandas import read_csv

def cousine_data_path():
	data_path = os.environ.get('COUSINE_DATA_PATH')
	if not data_path:
		raise RuntimeError('set the environment variable COUSINE_DATA_PATH to the data folder (cousine_site/)')
	return data_path if data_path.endswith('/') else data_path + '/'

# site_loc (row of the half-degree EnvInfo table): lon, lat, soil order, climate
SITES = {
	7294:  (-90.0,  41.0, 'alfisol', 'temperate'),
	7909:  (-85.0,  31.0, 'ultisol', 'subtropical'),
	12224: (-63.5, -10.0, 'oxisol',  'tropical')}

def site_of(lon, lat):
	match = [site_loc for site_loc, (site_lon, site_lat, _, _) in SITES.items() if site_lon == lon and site_lat == lat]
	if len(match) != 1:
		raise ValueError(f'no site-only input for lon {lon}, lat {lat}; available sites: {SITES}')
	return match[0]

_tables = {}
def _table(data_path, rel):
	if (data_path, rel) not in _tables:
		_tables[(data_path, rel)] = read_csv(data_path + rel, float_precision = 'round_trip').set_index('site_loc') # exact float64 round trip
	return _tables[(data_path, rel)]

def load_site_inputs(data_path, site_loc):
	# the site values the drivers used to compute from the global maps, same names, types and units
	value = lambda rel, col: float(_table(data_path, rel).loc[site_loc, col])
	scalar = lambda x: torch.tensor(x, dtype = torch.float64)
	gsde = lambda col: scalar(value('GSDE_Data/GSDE5min_sites.csv', col))
	site_soil_chem_bd = gsde('bd')
	return {
		'site_litho_type': np.float64(value('Litho_Map/glim_wgs84_0point5deg_sites.csv', 'litho_type')),
		'site_parent_cdr': scalar(value('GRUN_Runoff/parent_cdr_sites.csv', 'parent_cdr_t_co2_ha_yr')), # t CO2 ha-1 yr-1
		'site_ex_sol_ratio_cation': gsde('ex_sol_ratio_cation'),
		'site_max_cation_sink': gsde('max_cation_sink'),
		'site_soil_chem_exal': gsde('exal'),
		'site_soil_chem_cec': gsde('cec'),
		'site_soil_chem_bs': gsde('bs'),
		'site_soil_chem_ph': gsde('ph'),
		'site_soil_chem_exd2': gsde('exd2'),
		'site_soil_chem_bd': site_soil_chem_bd,
		'site_soil_chem_soc': gsde('soc'),
		'site_soil_chem_carbonate': scalar(value('SIC_storage/Huang_SIC_0_30cm_sites.csv', 'sic_g_kg'))*site_soil_chem_bd/100.0869, # g kg-1 --> mol m-3
		'site_soil_chem_gibbsite': scalar(value('Soil_Clay_Minerals/clay_weight_gibbsite_sites.csv', 'gibbsite_kg_m2'))*1000/0.3/78, # kg m-2 --> mol m-3
		'site_soil_chem_no3_leach': scalar(value('NO3_Leaching/rfcnleach_sites.csv', 'no3_leach_kg_n_ha_yr')), # kg N ha-1 yr-1
		'site_loc_1deg': np.int64(value('clm5_ouput_ncar/summary_land_mask_sites.csv', 'loc_1deg')),
		'para_soc': torch.tensor(np.asarray(_table(data_path, 'proda_opt/grid_para_result_sites.csv').loc[site_loc], dtype = np.float64)),
	}

def site_loc_1deg_of(data_path, lon, lat):
	# index of the 1-degree CLM5 forcing grid of the site at (lon, lat)
	return np.int64(_table(data_path, 'clm5_ouput_ncar/summary_land_mask_sites.csv').loc[site_of(lon, lat), 'loc_1deg'])

def load_site_env_info(data_path):
	# normalised environmental covariates of each site (BINN input), {site_loc: tensor}
	table = _table(data_path, 'global_grid_info/world_grid_envinfo_present_sites.csv')
	return {site_loc: torch.tensor(np.asarray(table.loc[site_loc], dtype = np.float64)) for site_loc in table.index}

def load_site_aet(data_path, lon, lat):
	# monthly actual evapotranspiration 1980-1999 at the site (GLEAM v3.6a), mm month-1, shape (12, 20)
	row = _table(data_path, 'AET/GLEAM_MONTH/E_1980-1999_GLEAM_v3.6a_MO_sites.csv').loc[site_of(lon, lat)]
	return torch.tensor(np.asarray(row[[f'month_{i + 1}' for i in range(240)]], dtype = np.float64).reshape(20, 12).transpose(), dtype = torch.float64)
