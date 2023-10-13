# BASICS
import xarray as xr
import numpy as np
import pandas as pd
import os
from datetime import datetime

# PLOTS
import matplotlib.pyplot as plt
# from matplotlib.colors import from_levels_and_colors, ListedColormap, LogNorm, Normalize, rgb2hex, SymLogNorm
import matplotlib.ticker as tkr
import cartopy.crs as ccrs

# REGRIDDING
import xesmf as xe

# CUSTOM FUNCTIONS
import utils

import warnings
warnings.filterwarnings('ignore')


# LOAD SIBD
data = {}
data['DFS'] = xr.open_mfdataset('/home/pfarnole/data/sibd/DFS_*')
data['CGRF'] = xr.open_mfdataset('/home/pfarnole/data/sibd/CGRF_*')
data['JRA'] = xr.open_mfdataset('/home/pfarnole/data/sibd/JRA_*')
data['ASMR'] = xr.open_mfdataset('/home/pfarnole/data/sibd/ASMR_*')

# REGRID 
from_dataset,to_dataset = 'DFS','ASMR'
regridder = xe.Regridder(data[from_dataset],data[to_dataset], 'nearest_s2d', periodic=True, ignore_degenerate=True)
data[f'{from_dataset}_regridded'] = regridder(data[from_dataset])

# SAVE / LOAD IF NEEDED
# # SAVE
# data[f'{from_dataset}_regridded'].to_netcdf(f'/home/pfarnole/data/sibd/{from_dataset}_regridded_to_{to_dataset}.nc')

# # LOAD
# data[f'{from_dataset}_regridded'] = xr.open_dataset(f'/home/pfarnole/data/sibd/{from_dataset}_regridded_to_{to_dataset}.nc')    

# VISUALIZE ORIGINAL VS REGRIDDED
y = 2014
v = 'sibd50' # sibd50, sibd85

plot_extent =  [-150,-100,67,85] # [-180,180,60,90] # [-140,-110,67,75] # >> you can change for regional focus
central_longitude = 0.5*(plot_extent[0]+plot_extent[1])
vmax = float(data[from_dataset][v].sel(year=y).max())

fig, axs = plt.subplots(ncols=2,figsize=(10,6),subplot_kw={'projection':ccrs.NorthPolarStereo(central_longitude=central_longitude)})
fig.suptitle(y)
for i,k in enumerate([from_dataset,f'{from_dataset}_regridded']):
    axs[i].coastlines()
    axs[i].set_title(k)
    axs[i].set_extent(plot_extent, ccrs.PlateCarree())
    pl = axs[i].pcolormesh(data[k].lon, data[k].lat, data[k][v].sel(year=y), transform=ccrs.PlateCarree(),vmin=0,vmax=vmax)
    plt.colorbar(pl, ax=axs[i],orientation='horizontal',label=v,shrink=0.7,pad=0.03)
fig.tight_layout()
plt.show()


# VISUALIZE REGRIDDED VS ASMR
y = 2014
z = (data['naa_regridded'] - data['asmr']).sel(year=y)
vmin,vmax = -100,100 # float(z.doySIbreak.min()),float(z.doySIbreak.max())

fig, ax = plt.subplots(figsize=(10,10),subplot_kw={'projection':ccrs.NorthPolarStereo(central_longitude=central_longitude)})
ax.coastlines()
ax.set_title(y)
ax.set_extent(plot_extent, ccrs.PlateCarree())
pl = ax.pcolormesh(z.lon, z.lat, z.sibd50, transform=ccrs.PlateCarree(),cmap='PuOr',vmin=vmin,vmax=vmax) #norm=SymLogNorm(linthresh=10, linscale=0.1, vmin=vmin,vmax=vmax))
plt.colorbar(pl, ax=ax,orientation='horizontal', label='SIBD NAA minus ASMR',shrink=0.7,pad=0.03)
plt.show()

