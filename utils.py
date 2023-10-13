import xarray as xr
import numpy as np
import pandas as pd
import os

from datetime import datetime, timedelta


def doySIbreak(sic,threshold=0.5,timedim='dayofyear'):
    """ first day after winter max SI concetration is below threshold
        return Nan for cells with max above or min below for 30 day rolling mean

        Note: if your sic dataset is not daily, use .resample(time="1D").interpolate("linear") prior to this function.
        Note: it assumes 365-days years.
        ### Version used in Antoine's paper ###
    """
    if timedim=='dayofyear': doys=sic[timedim]
    else: doys=sic[timedim].dt.dayofyear
   
    # set =0 if SI always below or  above threshold- will return day 0 which is then changed to nan
    # use 30 day rolling mean for this
    smth30_sic = sic.rolling(dim={timedim:30},center=True,min_periods=1).mean()
    sic_dropbelow = xr.where((smth30_sic.max(dim=[timedim]) <threshold) | (smth30_sic.min(dim=[timedim]) >threshold), 0, sic) 
    # for finding day of threshold, smooth daily signal with 14 day rolling mean
    smth_sic = sic_dropbelow.rolling(dim={timedim:14},center=True,min_periods=1).mean()
    
    # winter max ; before day 200 ; need to fill nan because argmax is stupid
    wmax = smth_sic.where(doys<200).fillna(0).argmax(dim=timedim)+1    
    # SIc afetr winter max
    sic_afterwintermax = smth_sic.where(doys>=wmax)
    
    # floor at threshold
    sic_floored = xr.where( sic_afterwintermax<threshold, threshold, sic_afterwintermax)
    # # first min is day threshood is reached
    iday= sic_floored.fillna(1).argmin(dim=timedim)+1
    # iday > 1 to filter NaNs
    res= iday.where(iday>1)
    
    del smth30_sic, sic_dropbelow, smth_sic, wmax, sic_afterwintermax, sic_floored, iday
    return res


def prep_sibd_naa(forcing,year):
    """ Uses doySIbreak        
        Computes day of the year of sea ice breakup for either of the 3 forcings: DFS, CGRF, JRA
        Creates two variables: sibd50 and sibd85 for each threshold (50% and 85% ice concentrations)
        
        ! Will soon also create variables for 15%, 50% and 85% freeze-up.
        
    """
    
    if forcing == 'DFS':
        datafiles = f'/tsanta/ahaddon/data/historical-DFS-G510.00/NAA_1d_{year}0101_{year}1231_biolog.nc'
        ic_var = 'ileadfra2'

    if forcing == 'CGRF':
        datafiles = f'/net/venus/kenes/data/arcticBGC/NAA_IAMIP2/HIST_KN_CGRF_ORAS025_C524.06/NAA_1d_{year}0101_{year}1231_biolog.nc'
        ic_var = 'ileadfrad'

    if forcing == 'JRA':
        datadir1 = '/net/venus/kenes/data/arcticBGC/NAA/IAMIP2-C524.00'
        datadir2 = '/net/venus/kenes/data/arcticBGC/NAA_IAMIP2/HIST_KN_EXTENSION_C524.08'
        datafiles = [os.path.join(datadir1,fp) for fp in os.listdir(datadir1) if '_biolog.nc' in fp and f'NAA_1d_{year}' in fp]
        datafiles += [os.path.join(datadir2,fp) for fp in os.listdir(datadir2) if '_biolog.nc' in fp and f'NAA_1d_{year}' in fp]
        ic_var = 'ileadfrad'

    # LOAD DATA
    data_tmp = xr.open_mfdataset(datafiles, combine='by_coords', parallel=True, chunks={'time_counter':365,'x':100,'y':100})
    data_tmp = data_tmp.where(data_tmp.nav_lat>60).chunk({'x':100,'y':100}).compute()

    # SAVE SIBD
    data = xr.Dataset()
    data['sibd50'] = data_tmp[ic_var].groupby('time_counter.year').map(dgnst.doySIbreak,threshold=0.5,timedim='time_counter').rename('sibd50')
    data['sibd85'] = data_tmp[ic_var].groupby('time_counter.year').map(dgnst.doySIbreak,threshold=0.85,timedim='time_counter').rename('sibd85')

    return data

