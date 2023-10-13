# BASICS
import xarray as xr
import numpy as np
import pandas as pd
import os
from datetime import datetime

# PLOTS
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import cartopy.crs as ccrs

# REGRIDDING
import xesmf as xe

# CUSTOM FUNCTIONS
import utils

import warnings
warnings.filterwarnings('ignore')


# LOAD SIBD DATA
data = {}
data['DFS'] = xr.open_mfdataset('/home/pfarnole/data/sibd/DFS_*')
data['CGRF'] = xr.open_mfdataset('/home/pfarnole/data/sibd/CGRF_*')
data['JRA'] = xr.open_mfdataset('/home/pfarnole/data/sibd/JRA_*')
data['ASMR'] = xr.open_mfdataset('/home/pfarnole/data/sibd/ASMR_*')

# PLOT and SHOW or SAVE
save_prefix = None # '/home/pfarnole/data/sibd/images/panarctic_naa_vs_asmr_sibd50-85'
years = list(range(2003,2012))+list(range(2013,2016))
plot_extent = [-180,180,60,90] # [-140,-110,67,75] # [-150,-100,67,85]   # >> you can change for regional focus
central_longitude = 0.5*(plot_extent[0]+plot_extent[1]) #-90 #

datasets = data.keys() #['DFS','CGRF','JRA','ASMR']
ncols = len(datasets)
vmax = 365 # None

for y in years:
    fig, axs = plt.subplots(nrows=2,ncols=ncols,figsize=(15,10),subplot_kw={'projection':ccrs.NorthPolarStereo(central_longitude=central_longitude)})
    fig.suptitle(y)
    for i,k in enumerate(datasets):
        axs[0,i].coastlines()
        axs[0,i].set_title(k)
        axs[0,i].set_extent(plot_extent, ccrs.PlateCarree())
        axs[1,i].coastlines()
        axs[1,i].set_title(k)
        axs[1,i].set_extent(plot_extent, ccrs.PlateCarree())
        pl0 = axs[0,i].pcolormesh(data[k].lon, data[k].lat, data[k].sibd85.sel(year=y), transform=ccrs.PlateCarree(),vmin=0,vmax=vmax)
        plt.colorbar(pl0, ax=axs[0,i],orientation='horizontal', label='Day of SIC<85% (openning)',shrink=0.7,pad=0.03, format=tkr.FormatStrFormatter('%d'))
        pl1 = axs[1,i].pcolormesh(data[k].lon, data[k].lat, data[k].sibd50.sel(year=y), transform=ccrs.PlateCarree(),vmin=0,vmax=vmax)
        plt.colorbar(pl0, ax=axs[1,i],orientation='horizontal', label='Day of SIC<50% (breakup)',shrink=0.7,pad=0.03, format=tkr.FormatStrFormatter('%d'))
    fig.tight_layout()
    if save_prefix:
        plt.savefig(f'{save_prefix}_{y}.png')
    else:
        plt.show()
