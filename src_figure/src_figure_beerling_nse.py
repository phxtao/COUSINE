# COUSINE (COUpled Soil Inorganic-organic carbon model for eNhanced wEathering)
# Copyright (c) 2026 Cornell University. All Rights Reserved. Cornell docket D-11800.
# Permission to use for educational, research and non-profit purposes only; see LICENSE for the terms,
# the commercial-licensing contact and the disclaimers. The concepts used in this work are patent pending.

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src_cousine'))
from fun_site_inputs import cousine_data_path
DATAHUB = cousine_data_path() # site-only data root (COUSINE_DATA_PATH, NTU server folder, or Mac folder)
import os

import numpy as np
import matplotlib
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures') + '/' # the repository's figures folder
os.makedirs(fig_dir, exist_ok = True)

para_prior_version = 0
data_path = DATAHUB
simu_data_dir = data_path + 'cousine_output/simu/prior_v' + str(para_prior_version) + '_pc/'

# fixed site/management settings used by simulator_main_beerling.py::worker_function
soil_type = 'alfisol'
forcing_loc = 'temperate'
nitrogen_fertilizer_input = 20  # matches simulator_main_beerling.py's hardcoded nitrogen_fertilizer_input = 20
dust_type = 'basalt'
dust_amount = 50.0 * 0.6542


kinetic_scalar_list = [1, 3, 6, 8, 10, 12] 
leach_scalar_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5] 
pco2_scalar_list = [0.7, 1, 1.3, 1.5, 1.8, 2]

# rows 4:9 (0-indexed) of the per-year output are the 5 years to compare
# against Beerling et al. 2024: row 4 is model year 5 (dust_start_year, i.e.
# the pre-ERW baseline, matching obs index 0 = 0) through row 8 (4 years
# after amendment), matching the 5 post-amendment observation years
sim_year_slice = slice(4, 9)

###############################################
# observations (Beerling et al. 2024, PNAS)
###############################################
beerling_pnas_cdr_pot_mean = np.array([0, 3.821523585, 6.752516667, 11.48819547, 10.5085952])
beerling_pnas_cdr_pot_std = np.array([0, 1.789155663, 2.615106751, 3.185674532, 3.757679717])

beerling_pnas_ph = np.array([[6.05, 6.40, 6.55, 7.05, 6.85],
                              [6.15, 6.65, 7.00, 7.20, 7.25],
                              [6.50, 6.80, 6.65, 6.65, 7.00],
                              [6.55, 6.60, 6.70, 7.15, 6.95],
                              [6.55, 6.65, 6.60, 7.05, 6.85]])

obs_cdr = beerling_pnas_cdr_pot_mean
obs_ph = np.mean(beerling_pnas_ph, axis=0)  # median across the 5 replicate sites, one value per year


def fun_nse(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    if sim.shape != obs.shape or np.any(np.isnan(sim)):
        return np.nan
    nse = 1 - np.sum((obs - sim) ** 2) / (np.sum((obs - obs.mean()) ** 2))
    nnse = 1 / (2 - nse)
    return nse

###############################################
# load simulations and compute NSE for every parameter combination
###############################################
nse_cdr = np.full((len(kinetic_scalar_list), len(leach_scalar_list), len(pco2_scalar_list)), np.nan)
nse_ph = np.full_like(nse_cdr, np.nan)

missing_file_num = 0
for ik, kinetic_val in enumerate(kinetic_scalar_list):
    for il, leach_val in enumerate(leach_scalar_list):
        for ip, pco2_val in enumerate(pco2_scalar_list):
            file_path = (f'{simu_data_dir}annual_mean/sensitivity_beerling_soil/sensitivity_beerling_soil_{soil_type}_{forcing_loc}'
                         f'_nitrogen_{nitrogen_fertilizer_input}_{dust_type}_{dust_amount}'
                         f'_kinetic_{kinetic_val}_leach_{leach_val}_pco2_{pco2_val}.csv')

            if os.path.exists(file_path) == False:
                missing_file_num = missing_file_num + 1
                continue

            sim_data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # columns: ph_series, cdr_potential_accu
            sim_ph = sim_data[sim_year_slice, 0]
            sim_cdr = sim_data[sim_year_slice, 1]

            nse_ph[ik, il, ip] = fun_nse(obs_ph, sim_ph)
            nse_cdr[ik, il, ip] = fun_nse(obs_cdr[1:], sim_cdr[1:])

if missing_file_num > 0:
    print(f'warning: {missing_file_num} out of {len(kinetic_scalar_list)*len(leach_scalar_list)*len(pco2_scalar_list)} sensitivity result files were not found')

nse_combined = np.nanmean(np.stack([nse_cdr, nse_ph]), axis=0)

###############################################
# report best-fit parameter combinations
###############################################
def fun_report_best(nse_array, metric_name):
    if np.all(np.isnan(nse_array)):
        print(f'{metric_name}: no valid simulations found')
        return
    ik, il, ip = np.unravel_index(np.nanargmax(nse_array), nse_array.shape)
    print(f'{metric_name}: best NSE = {nse_array[ik, il, ip]:.3f} at '
          f'kinetic = {kinetic_scalar_list[ik]}, leach = {leach_scalar_list[il]}, pco2 = {pco2_scalar_list[ip]}')

fun_report_best(nse_cdr, 'CDR')
fun_report_best(nse_ph, 'pH')
fun_report_best(nse_combined, 'combined')

###############################################
# figure: 3 panels (CDR / pH / combined), each a 3D contour with axes =
# kinetic / leach / pco2 scalars, drawn as stacked filled-contour slices
# (one continuous pco2-leach contour per kinetic level, interpolated for
# smoothness), colored by NSE
###############################################
# baseline (scalar = 1) physical values that the sensitivity scalars multiply,
# so the axes show absolute values instead of dimensionless scalars (mapping
# confirmed in fun_carbon_dynamics_beerling.py):
#   kinetic scalar -> dust dissolution rate k        (mol m-2 s-1), diss_rate_ref_dust
#   pco2 scalar    -> CO2 turnover time tau_CO2      (days), co2_turnover_time
#   leach scalar   -> SIC leaching turnover tau_SIC  (days), leachate_turnover_time
k_baseline = 10 ** (-9.52)  # mol m-2 s-1
tau_co2_baseline = 2.07     # days
tau_sic_baseline = 80.6     # days

kinetic_arr = np.asarray(kinetic_scalar_list, dtype=float) * k_baseline
leach_arr = np.asarray(leach_scalar_list, dtype=float) * tau_sic_baseline
pco2_arr = np.asarray(pco2_scalar_list, dtype=float) * tau_co2_baseline

# all three axes (k on the stacked z-axis, tau_SIC on the in-plane y-axis,
# tau_CO2 on the in-plane x-axis) are plotted as absolute values on a normal
# (linear) scale, so the arrays above are used directly as plot coordinates
n_fine = 60
pco2_fine = np.linspace(pco2_arr.min(), pco2_arr.max(), n_fine)
leach_fine = np.linspace(leach_arr.min(), leach_arr.max(), n_fine)
pco2_fine_mesh, leach_fine_mesh = np.meshgrid(pco2_fine, leach_fine, indexing='ij')


def plot_nse_contour_3d(ax, nse_array, title, vmin=-1, vmax=1):
    # use draw order (not mplot3d's estimated depth) to decide what's on
    # top: with computed_zorder left at its default, the star marker can
    # get blended under a contour slice whenever their depths are close,
    # which flips unpredictably as the parameter ranges/box geometry change
    ax.computed_zorder = False
    ax.view_init(elev = 20, azim = -30)  # camera angle: elev = up/down tilt, azim = rotation around the z-axis

    cmap = matplotlib.colormaps['RdYlBu']
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 25)

    mappable = None
    for ik, z_pos in enumerate(kinetic_arr):
        # values are left unclipped (contourf's extend='both' below handles
        # out-of-range coloring); clipping to exactly vmin/vmax here would
        # create a flat plateau sitting on the first/last contour level,
        # which is numerically degenerate and fragments into tiny spurious
        # polygons once interpolated
        nse_slice = nse_array[ik, :, :].T  # (pco2, leach)

        if np.any(np.isnan(nse_slice)):
            # too many gaps to interpolate reliably: fall back to the raw grid
            nse_slice_fine = nse_slice
            x_plot, y_plot = pco2_arr, leach_arr
        else:
            interpolator = RegularGridInterpolator((pco2_arr, leach_arr), nse_slice, method='linear')
            nse_slice_fine = interpolator((pco2_fine_mesh, leach_fine_mesh))
            x_plot, y_plot = pco2_fine, leach_fine

        mappable = ax.contourf(x_plot, y_plot, nse_slice_fine.T, levels=levels,
                                zdir='z', offset=z_pos, cmap=cmap, norm=norm, alpha=0.8,
                                extend='both')

    z_margin = 0.2 * (kinetic_arr.max() - kinetic_arr.min())

    valid = np.isnan(nse_array) == False
    if np.any(valid):
        ik_max, il_max, ip_max = np.unravel_index(np.nanargmax(nse_array), nse_array.shape)
        x_best, y_best = pco2_arr[ip_max], leach_arr[il_max]
        z_best = kinetic_arr[ik_max]

        ax.scatter([x_best], [y_best], [z_best], facecolor='gold', edgecolor='black',
                   s=800, marker='*', linewidth=1.2, depthshade=False, zorder=21)

        coord_label = (f'($\\log_{{10}}k$={np.log10(kinetic_arr[ik_max]):.2f}, '
                       f'$\\tau_{{CO_2}}$={pco2_arr[ip_max]:.1f}, '
                       f'$\\tau_{{SIC}}$={leach_arr[il_max]:.0f})')
        x_label_offset = x_best + 0.02 * (pco2_arr.max() - pco2_arr.min())
        y_label_offset = y_best + 0.02 * (leach_arr.max() - leach_arr.min())
        z_label_offset = z_best + 0.02 * (kinetic_arr.max() - kinetic_arr.min())
        ax.text(x_label_offset, y_label_offset, z_label_offset, coord_label, fontsize=18,
                fontweight='bold', color='black', zorder=22, va='bottom', ha='left')

    ax.set_xlim(pco2_arr.min(), pco2_arr.max())
    ax.set_ylim(leach_arr.min(), leach_arr.max())
    ax.set_zlim(kinetic_arr.min() - z_margin, kinetic_arr.max() + z_margin)
    ax.set_xticks(pco2_arr)
    ax.set_xticklabels([f'{v:.1f}' for v in pco2_arr])
    ax.set_yticks(leach_arr)
    ax.set_yticklabels([f'{v:.0f}' for v in leach_arr])
    ax.set_zticks(kinetic_arr)
    # tick positions stay linear in k (normal scale); only the labels show log10(k)
    ax.set_zticklabels([f'{np.log10(v):.2f}' for v in kinetic_arr])

    ax.set_xlabel(r'$\tau_{CO_{2}}$ (days)', fontsize=30, labelpad=20)
    ax.set_ylabel(r'$\tau_{SIC}$ (days)', fontsize=30, labelpad=20)
    ax.set_zlabel(r'$\log_{10} k$ (mol m$^{-2}$ s$^{-1}$)', fontsize=30, labelpad=20)
    ax.set_title(title, fontsize=40, pad=20)
    ax.tick_params(labelsize=25)

    return mappable


fig = plt.figure(figsize=(31, 13))
fig.subplots_adjust(left=0.02, right=0.92, top=0.88, bottom=0.16, wspace=0.1)

panel_specs = [
    (nse_cdr, 'NSE for CDR'),
    (nse_ph, 'NSE for pH'),
    (nse_combined, 'Average NSE'),
]
panel_letters = ['(a)', '(b)', '(c)']

for ipanel, (nse_array, panel_title) in enumerate(panel_specs):
    ax = fig.add_subplot(1, 3, ipanel + 1, projection='3d')
    mappable = plot_nse_contour_3d(ax, nse_array, panel_title)
    ax.text2D(-0.05, 1.06, panel_letters[ipanel], transform=ax.transAxes,
               fontsize=30, fontweight='bold', va='top', ha='left')

# fig.suptitle('Model-observation agreement across parameter scalars', fontsize=20, y=0.98)

cbar_ax = fig.add_axes([0.3, 0.15, 0.4, 0.02])
cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Nash-Sutcliffe efficiency (NSE)', fontsize=30)
cbar.ax.tick_params(labelsize=30)

plt.savefig(fig_dir + 'cdr_weathering_beerling_nse.jpeg')
plt.savefig(fig_dir + 'cdr_weathering_beerling_nse.pdf')
plt.close()

###############################################
# figure: best-fit (combined CDR + pH NSE) simulation vs. Beerling et al.
# 2024 observations, styled after src_figure_obs_constraint.py
###############################################
ik_best, il_best, ip_best = np.unravel_index(np.nanargmax(nse_combined), nse_combined.shape)
kinetic_best = kinetic_scalar_list[ik_best]
leach_best = leach_scalar_list[il_best]
pco2_best = pco2_scalar_list[ip_best]

best_file_path = (f'{simu_data_dir}annual_mean/sensitivity_beerling_soil/sensitivity_beerling_soil_{soil_type}_{forcing_loc}'
                   f'_nitrogen_{nitrogen_fertilizer_input}_{dust_type}_{dust_amount}'
                   f'_kinetic_{kinetic_best}_leach_{leach_best}_pco2_{pco2_best}.csv')
best_sim_data = np.loadtxt(best_file_path, delimiter=',', skiprows=1)
best_sim_ph = best_sim_data[sim_year_slice, 0]
best_sim_cdr = best_sim_data[sim_year_slice, 1]

###############################################
# realised (net) CDR for the best-fit combination, following the
# net-carbonate-accumulation approach in src_figure_cdr_efficiency.py:
# actual carbonate formation (from species-stock changes + leaching),
# minus the carbon already carried by parent-material/dust dissolution,
# compared between the dust-amended run and a dust-free baseline run
# (same kinetic/leach/pco2 scalars, dust_amount = 0) to isolate the CDR
# attributable to the amendment
###############################################
def fun_realised_cdr_accu(transit_dust, transit_parent):
    stoichio_element_parent = transit_parent['stoichio_element_parent'].detach().numpy()
    stoichio_element_dust = transit_dust['stoichio_element_dust'].detach().numpy()
    stoichio_carbon_parent = stoichio_element_parent[4]
    stoichio_carbon = stoichio_element_dust[4]

    delta_species_stock_parent = transit_parent['delta_species_stock'].detach().numpy()
    leached_ion_stock_parent = transit_parent['leached_ion_stock'].detach().numpy()
    dust_consumed_base_parent = transit_parent['dust_consumed_parent'].detach().numpy()

    delta_species_stock = transit_dust['delta_species_stock'].detach().numpy()
    leached_ion_stock = transit_dust['leached_ion_stock'].detach().numpy()
    dust_consumed = transit_dust['dust_consumed'].detach().numpy()
    dust_consumed_base_dust = transit_dust['dust_consumed_parent'].detach().numpy()

    carbonate_accu_parent = (-delta_species_stock_parent)[0:6, (11, 12, 14), :, :].sum(axis=(0, 1, 2)).cumsum() \
        + leached_ion_stock_parent[(11, 12, 14), :, :].sum(axis=(0, 1)).cumsum() \
        - np.nansum(dust_consumed_base_parent, axis=(0, 1)).cumsum(axis=0) * stoichio_carbon_parent

    carbonate_accu_tot = (-delta_species_stock)[0:6, (11, 12, 14), :, :].sum(axis=(0, 1, 2)).cumsum() \
        + leached_ion_stock[(11, 12, 14), :, :].sum(axis=(0, 1)).cumsum() \
        - np.nansum(dust_consumed, axis=(0, 1)).cumsum(axis=0) * stoichio_carbon \
        - np.nansum(dust_consumed_base_dust, axis=(0, 1)).cumsum(axis=0) * stoichio_carbon_parent

    net_carbonate_accu = carbonate_accu_tot - carbonate_accu_parent
    return net_carbonate_accu * 44 * 10 ** 4 / 10 ** 6  # cumulative realised CDR, t CO2 ha-1


best_transit_path = (f'{simu_data_dir}annual_mean/sensitivity_beerling_soil/transit_formal_beerling_soil_{soil_type}_{forcing_loc}'
                      f'_nitrogen_{nitrogen_fertilizer_input}_{dust_type}_{{dust_amount_val}}'
                      f'_kinetic_{kinetic_best}_leach_{leach_best}_pco2_{pco2_best}.pth')
best_transit_dust = torch.load(best_transit_path.format(dust_amount_val=dust_amount))
best_transit_parent = torch.load(best_transit_path.format(dust_amount_val=0))

best_sim_cdr_realised = fun_realised_cdr_accu(best_transit_dust, best_transit_parent)[sim_year_slice]

###############################################
# report best-fit potential vs. realised CDR (cumulative, t CO2 ha-1)
# potential = dust-dissolution CO2 uptake capacity (sensitivity CSV);
# realised  = net carbonate actually retained relative to a dust-free run
###############################################
best_sim_cdr_potential = best_sim_cdr  # potential CDR loaded from the sensitivity CSV
np.set_printoptions(precision=3, suppress=True)
print(f'best-fit combination: kinetic = {kinetic_best}, leach = {leach_best}, pco2 = {pco2_best}')
print(f'potential CDR (t CO2 ha-1), years 0-4: {best_sim_cdr_potential}')
print(f'realised  CDR (t CO2 ha-1), years 0-4: {best_sim_cdr_realised}')
print(f'final-year potential CDR = {best_sim_cdr_potential[-1]:.3f} t CO2 ha-1')
print(f'final-year realised  CDR = {best_sim_cdr_realised[-1]:.3f} t CO2 ha-1')
with np.errstate(divide='ignore', invalid='ignore'):
    cdr_efficiency = best_sim_cdr_realised[-1] / best_sim_cdr_potential[-1]
print(f'realised/potential CDR efficiency (final year) = {cdr_efficiency:.3f}')

years = np.arange(0, 5)
best_fit_label = 'COUSINE'

fig, axes = plt.subplots(1, 2, figsize=(15, 8), layout='tight')

# -----------------------------------------------
# Panel (a): pH comparison
# -----------------------------------------------
ax = axes[0]
ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(1.5)

ph_mean = beerling_pnas_ph.mean(axis=0)
ph_min = beerling_pnas_ph.min(axis=0)
ph_max = beerling_pnas_ph.max(axis=0)

ax.fill_between(years, ph_min, ph_max, color='#377eb8', alpha=0.2, label='Beerling et al. 2024 (full range)')
ax.plot(years, ph_mean, marker='o', markersize=10, linewidth=3, color='#377eb8', label='Beerling et al. 2024 (mean)')
ax.plot(years, best_sim_ph, marker='s', markersize=10, linewidth=3, color='#e41a1c', label=best_fit_label)

ax.set_xlabel('Year', fontsize=23)
ax.set_ylabel('pH', fontsize=23)
ax.set_xticks(years)
ax.tick_params(labelsize=20, width=1.5, length=7)
ax.legend(fontsize=18, frameon=False, loc='upper left')
ax.text(-0.15, 1.05, '(a)', transform=ax.transAxes, fontsize=30, fontweight='bold', va='top', ha='left')

# -----------------------------------------------
# Panel (b): CDR comparison
# -----------------------------------------------
ax = axes[1]
ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(1.5)

ax.fill_between(years, beerling_pnas_cdr_pot_mean - beerling_pnas_cdr_pot_std,
                 beerling_pnas_cdr_pot_mean + beerling_pnas_cdr_pot_std,
                 color='#377eb8', alpha=0.2, label='Beerling et al. 2024 (SEM)')
ax.plot(years, beerling_pnas_cdr_pot_mean, marker='o', markersize=10, linewidth=3, color='#377eb8',
        label='Beerling et al. 2024 (Potential Mean)')
ax.plot(years, best_sim_cdr, marker='s', markersize=10, linewidth=3, color='#e41a1c', label=f'{best_fit_label} (Potential)')
ax.plot(years, best_sim_cdr_realised, marker='D', markersize=10, linewidth=3, color='#e41a1c', linestyle='dashed', label=f'{best_fit_label} (Realized)')

ax.set_xlabel('Year', fontsize=23)
ax.set_ylabel('Cumulative CDR ($t \\ CO_{2} \\ ha^{-1}$)', fontsize=23)
ax.set_xticks(years)
ax.tick_params(labelsize=20, width=1.5, length=7)
ax.legend(fontsize=18, frameon=False, loc='upper left')
ax.text(-0.15, 1.05, '(b)', transform=ax.transAxes, fontsize=30, fontweight='bold', va='top', ha='left')

plt.savefig(fig_dir + 'cdr_weathering_beerling_best_fit_comparison.jpeg', bbox_inches='tight')
plt.savefig(fig_dir + 'cdr_weathering_beerling_best_fit_comparison.pdf', bbox_inches='tight')
plt.close()

print('end of programme')
