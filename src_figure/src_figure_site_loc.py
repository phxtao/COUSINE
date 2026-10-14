# COUSINE (COUpled Soil Inorganic-organic carbon model for eNhanced wEathering)
# Copyright (c) 2026 Cornell University. All Rights Reserved. Cornell docket D-11800.
# Permission to use for educational, research and non-profit purposes only; see LICENSE for the terms,
# the commercial-licensing contact and the disclaimers. The concepts used in this work are patent pending.

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src_cousine'))
from fun_site_inputs import cousine_data_path
DATAHUB = cousine_data_path() # site-only data root (COUSINE_DATA_PATH, NTU server folder, or Mac folder)
import matplotlib.pyplot as plt
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()}, layout='tight')

ax.set_extent([-110, -40, -25, 55], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none')
# ax.add_feature(cfeature.OCEAN, facecolor='#d4e9f7')
ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='#333333')
# ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle='--', edgecolor='#999999')
ax.add_feature(cfeature.LAKES, edgecolor='#333333', linewidth=1.5)
ax.add_feature(cfeature.RIVERS, linewidth=1.5, edgecolor='#6baed6')

# Site coordinates
sites = {
    'Alfisol (Temperate)': (-90, 41),
    'Ultisol (Subtropical)': (-85, 31),
    'Oxisol (Tropical)': (-63.5, -10),
}
colors = ['#377eb8', '#4daf4a', '#e41a1c']
markers = ['o', 'o', 'o']

for i, (label, (lon, lat)) in enumerate(sites.items()):
    ax.plot(lon, lat, marker=markers[i], markersize=18, color=colors[i],
            markeredgecolor='black', markeredgewidth=2, transform=ccrs.PlateCarree(), zorder=5)
    ax.annotate(label, xy=(lon, lat), xytext=(15, 15), textcoords='offset points',
                fontsize=16, fontweight='bold', color=colors[i],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors[i], linewidth=2, alpha=0.9),
                # arrowprops=dict(arrowstyle='->', color=colors[i], linewidth=2),
                transform=ccrs.PlateCarree(), zorder=6)

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='grey', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 16, 'weight': 'normal'}
gl.ylabel_style = {'size': 16, 'weight': 'normal'}

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures') + '/' # the repository's figures folder
os.makedirs(fig_dir, exist_ok = True)

plt.savefig(fig_dir + 'site_locations_map.jpeg', dpi=300, bbox_inches='tight')
plt.savefig(fig_dir + 'site_locations_map.pdf', bbox_inches='tight')
plt.close()