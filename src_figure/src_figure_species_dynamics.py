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


from scipy.io import loadmat, savemat
import torch

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures') + '/' # the repository's figures folder
os.makedirs(fig_dir, exist_ok = True)

# dust_amount_list = [15.0]
forcing_cycle_num = 5
year_num = 20
month_num = 12
dust_start_year = 40
dust_type = 'olivine'

dust_amount_list = [10.0]
forcing_loc_list = ['temperate', 'subtropical', 'tropical']
soil_type = ['alfisol', 'ultisol', 'oxisol']
species_list = ['Al+++', 'Al(OH)++', 'Al(OH)4-', 'H4SiO4', 'H3SiO4-', 'M+', 'M++', 'OH-', 'H+', 'CO2(aq)', 'CO2(g)', 'HCO3-', 'CO3--', 'H2O', 'MCO3', 'Al(OH)3', 'SM', 'SH2', 'SOA-', 'SOAH']
species_label = ['$Al^{3+}$', '$Al(OH)^{2+}$', '$Al(OH)_{4}^{-}$', '$H_{4}SiO_{4}$', '$H_{3}SiO_{4}^{-}$', '$M^{+}$', '$D^{2+}$', '$OH^{-}$', '$H^{+}$', '$CO_{2}(aq)$', '$CO_{2}(g)$', '$HCO_{3}^{-}$', '$CO_{3}^{2-}$', '$H_{2}O$', '$DCO_{3}$', '$Al(OH)_{3}$', '$\Pi D$', '$\Pi_{3}Al_{2}$', '$SOA_{2}D$', 'SOAH']

color_scheme = ['#e41a1c','#377eb8','#4daf4a']

panel_letters = [['(a)', '(b)', '(c)'],
                 ['(d)', '(e)', '(f)'],
                 ['(g)', '(h)', '(i)'],
                 ['(j)', '(k)', '(l)']]

for idust_amount in np.arange(len(dust_amount_list)):
    dust_amount = dust_amount_list[idust_amount] # 10.0

    post_dust_year_num = 60
    plt.figure(figsize = (30, 24), layout = 'tight')

    for isoil in range(len(soil_type)):
        forcing_loc = forcing_loc_list[isoil]

        data_dir = DATAHUB + 'cousine_output/simu/prior_v0_pc/annual_mean/'
        transit_formal_parent = torch.load(f'{data_dir}transit_formal_soil_{soil_type[isoil]}_{forcing_loc}_nitrogen_{0}_{dust_type}_{0.0}.pth')
        transit_formal = torch.load(f'{data_dir}transit_formal_soil_{soil_type[isoil]}_{forcing_loc}_nitrogen_{0}_{dust_type}_{dust_amount}.pth')
        
        # dust simu onetime
        species_stock = transit_formal['species_stock'].detach().numpy()

  
        transit_year_size = forcing_cycle_num*year_num
        
        #-----------------------------------------------
        # species absolute change
        #-----------------------------------------------
        # color_scheme = ['#a6cee3','#1f78b4','#b2df8a','#33a02c', \
        # '#fb9a99','#e31a1c','#fdbf6f','#ff7f00', \
        # '#cab2d6','#6a3d9a','#ffff99','#b15928', \
        # '#003c30','#543005','#40004b','#1a1a1a', \
        # '#7bccc4', '#a6bddb', '#67001f', '#878787']
        iforcing = isoil
        plt.figure(figsize = (25, 20), layout = 'tight')
        for ispecies in range(len(species_list)):
            plt.subplot(5, 4, (ispecies+1))
            axes = plt.gca()
            axes.spines[['top','bottom','left','right']].set_linewidth(1.5)
            axes.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            plt.plot(np.arange(1, (transit_year_size*month_num+1))/12, np.transpose(species_stock[ispecies, :, :]).reshape(((transit_year_size*month_num), 1)), \
                    color = 'black', marker = '', markersize = 0, linestyle = 'solid', linewidth = 2)
            plt.title(species_label[ispecies], fontsize = 30)
            plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
            axes.yaxis.get_offset_text().set_fontsize(20) 
            if ispecies == 17:
                plt.xlabel(xlabel = 'Simulation year', fontsize = 35)
                axes.xaxis.set_label_coords(1.0, -0.2)
            if ispecies == 8:
                plt.ylabel(ylabel = 'Stock ($mol \ m^{-2}$)', fontsize = 35)
                # axes.yaxis.set_label_coords(-0.12, -0.1)
            plt.tick_params(labelsize = 20, width = 1.5, length = 7)
            # plt.xticks((time_series-1901)[np.arange(0, (transit_year_size*month_num), 60)])
        plt.savefig(fig_dir + 'species_dynamics_inter_annual_variation_' + soil_type[isoil] + '_dust_matrix_' + forcing_loc + '_' + str(dust_amount) + '.pdf')
        plt.savefig(fig_dir + 'species_dynamics_inter_annual_variation_' + soil_type[isoil] + '_dust_matrix_' + forcing_loc + '_' + str(dust_amount) + '.jpeg')
        plt.close()

print('end of programme')


