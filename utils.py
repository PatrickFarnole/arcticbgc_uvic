import xarray as xr
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from pyproj import Proj, itransform, transform


def find_closest_loc(ds,lat,lon):
    abslat = np.abs(ds.lat-lat)
    abslon = np.abs(ds.lon-lon)
    c = np.maximum(abslat,abslon)
    ([yloc], [xloc]) = np.where(c == np.min(c))
    return xloc,yloc


def load_asmr_sic(year,month='*',xy=None):

    data_amsr = xr.open_mfdataset(f"/tsanta/ahaddon/data/asi-AMSR/*{year}{month}.nc", 
                                  combine='by_coords', chunks={"time": 10,'x':300,'y':300})

    # OLDER SHORTER VERSION
    ## Output coordinates are in WGS 84 longitude and latitude
    projOut = Proj(init='epsg:4326')
    ## projection given in the netCDF file
    projIn = Proj(init='epsg:3411',preserve_units=True)
    xx, yy = np.meshgrid(data_amsr.x.values, data_amsr.y.values)
    lon,lat= transform(projIn, projOut, xx, yy )

    # # NEWER LONGER VERSION
    # # projection 0: polar_stereographic (defined by epsg code 3411)
    # p0 = Proj('epsg:3411', preserve_units=False)
    # # projection 1: WGS84 (defined by epsg code 4326)
    # p1 = Proj('epsg:4326', preserve_units=False)
    # # format points for itransform
    # xx, yy = np.meshgrid(data_amsr.x.values, data_amsr.y.values)
    # nx,ny = xx.shape
    # points = list(zip(xx.flatten(),yy.flatten()))
    # # transform xx,yy to projection 1 coordinates.
    # flat_lonlat = itransform(p0,p1,points,always_xy=True)
    # flat_lon,flat_lat = zip(*flat_lonlat)
    # lon,lat = np.reshape(flat_lon,(nx,ny)),np.reshape(flat_lat,(nx,ny))

    # CREATE COORDS FOR LAT LON
    data_amsr['lon'] = (('y','x'),lon)
    data_amsr['lat'] = (('y','x'),lat)
    coords={'lat': (['y','x'],np.array(data_amsr.lat)),
            'lon': (['y','x'],np.array(data_amsr.lon))}
    data_amsr = data_amsr.assign_coords(coords).drop_vars(['x','y','polar_stereographic'])
    data_amsr = data_amsr.where(data_amsr.lat>60).rename({'z':'sic'})
    data_amsr['sic'] = 0.01*data_amsr.sic #.resample(time="1D").interpolate("linear")
    
    try:
        data_amsr = data_amsr.drop_sel(time=np.datetime64(f'{year}-02-29'))
    except:
        pass
    
    if xy:
        xloc,yloc = xy
        data_amsr = data_amsr.sel(x=xloc,y=yloc)
    
    return data_amsr


def load_naa_sic(forcing,year,xy=None):
    
    if forcing == 'DFS':
        datafiles = f'/tsanta/ahaddon/data/historical-DFS-G510.00/NAA_1d_{year}0101_{year}1231_biolog.nc'
        ic_var = 'ileadfra2'
        factor = 1

    if forcing == 'RCP85':
        datafiles = f'/tsanta/ahaddon/data/RCP85-G510.14-G515.00-01/NAA_1d_{year}0101_{year}1231_biolog.nc'
        ic_var = 'ileadfra2'
        factor = 1

    if forcing == 'CGRF':
        datafiles = f'/net/venus/kenes/data/arcticBGC/NAA_IAMIP2/HIST_KN_CGRF_ORAS025_C524.06/NAA_1d_{year}0101_{year}1231_biolog.nc'
        ic_var = 'ileadfrad'
        factor = 1

    if forcing == 'JRA':
        if year<2018:
            datadir = '/net/venus/kenes/data/arcticBGC/NAA/IAMIP2-C524.00'
        else:
            datadir = '/net/venus/kenes/data/arcticBGC/NAA_IAMIP2/HIST_KN_EXTENSION_C524.08'
        datafiles = [os.path.join(datadir,fp) for fp in os.listdir(datadir) if '_biolog.nc' in fp and f'NAA_1d_{year}' in fp]
        ic_var = 'ileadfrad'
        factor = 1

    # LOAD DATA
    data_tmp = xr.open_mfdataset(datafiles, combine='by_coords', parallel=True, chunks={'time_counter':365,'x':100,'y':100})
    data_tmp = data_tmp.where(data_tmp.nav_lat>60).chunk({'x':100,'y':100})
    data_tmp = data_tmp.rename({'nav_lat':'lat','nav_lon':'lon',ic_var:'sic','time_counter':'time'}) #.compute()
    data_tmp['sic'] = factor*data_tmp['sic']
    
    try:
        data_tmp = data_tmp.drop_sel(time=np.datetime64(f'{year}-02-29T12:00:00.000000000'))
    except:
        pass
    
    if xy:
        xloc,yloc = xy
        data_tmp = data_tmp.sel(x=xloc,y=yloc)

    return data_tmp


def doySI(sic,threshold=0.5,timedim='time',breakup=True):
    """ first day after winter max SI concetration is below threshold
        return Nan for cells with max above or min below for 30 day rolling mean

        Note: if your sic dataset is not daily, use .resample(time="1D").interpolate("linear") prior to this function.
        Note: it assumes 365-days years.
        ### Version used in Antoine's paper ###
    """

    max_breakup_doy,min_freezup_doy = 200,265 # 265,265 would be better ~September 25

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
    
    if breakup:    
        # SIc afetr winter max
        sic_afterwintermax = smth_sic.where(doys>=wmax)
        # floor at threshold
        sic_floored = xr.where( sic_afterwintermax<threshold, threshold, sic_afterwintermax)
        # # first min is day threshood is reached
        iday = sic_floored.fillna(1).argmin(dim=timedim)+1
        # iday > 1 to filter NaNs
        res = iday.where(iday>1)
    else:
        # selecting freeze-up time range
        sic_afterminfreezup = smth_sic.where(doys>=min_freezup_doy)
        # setting minimum sic to threshold
        sic_floored = xr.where(sic_afterminfreezup>threshold, threshold, sic_afterminfreezup)
        # selecting time of the first minimum i.e. time of the first 0.5 sic
        iday1 = sic_floored.fillna(0).argmax(dim=timedim)+1
        res = iday1.where(iday1>0)
    
    # del smth30_sic,sic_dropbelow,smth_sic,wmax,sic_afterwintermax,sic_floored,iday
    
    return res


def prep_sibd(data_sic,year):
    """ Uses doySIbreak        
    """
    data_out = xr.Dataset()
    data_out['sibd85'] = data_sic.sic.groupby('time.year').map(doySI,threshold=0.85,timedim='time',breakup=True).rename('sibd85')
    data_out['sibd50'] = data_sic.sic.groupby('time.year').map(doySI,threshold=0.5,timedim='time',breakup=True).rename('sibd50')
    data_out['sifd50'] = data_sic.sic.groupby('time.year').map(doySI,threshold=0.5,timedim='time',breakup=False).rename('sifd50')
    data_out['sifd85'] = data_sic.sic.groupby('time.year').map(doySI,threshold=0.85,timedim='time',breakup=False).rename('sifd85')
    
    return data_out
