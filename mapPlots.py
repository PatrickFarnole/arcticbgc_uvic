import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import datetime


from glob import glob 


def shiftCMAP(vmin, vmax, midpoint,cmap,nticks=4):
    """ 
        Shift a colormap, useful for plotting positive and negative values with a divergent colormap and different bounds
        eaxmple
        shiftCMAP(-30, 100, 0,'coolwarm') 
        plotvalues from -30 to +100 with blue-white-red (coolwarm cmap), blue for negative, white for 0 and red for positive
    """
    nl=256
    levels = np.linspace(vmin, vmax, nl)
    if isinstance(cmap , str): clmap = plt.colormaps[cmap]
    else: clmap=cmap
    cmap_shifted, norm = colors.from_levels_and_colors(np.linspace(vmin, vmax, nl-1), 
                                         clmap(np.interp(levels, [vmin, midpoint, vmax], [0, 0.5, 1])), extend='both')
    return cmap_shifted, norm, [t for t in np.linspace(vmin,vmax,nticks)] 






import matplotlib.path as mpath

def roundBoundary(ax):
    """
        Set round boundary of axis
    """
    
    # Compute a circle in axes coordinates, which we can use as a boundary for the map. 
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)



def initmaps(rows=1,columns=1,extent=[-180,180, 60,90], rnd=False, central_longitude=0,
             landcolor=cfeature.COLORS['land'],
             **kw): 
    """
        Initate maps
        north polar projection by default
        adds land        
    """
    fig, ax = plt.subplots(rows,columns,**kw, 
                       subplot_kw={'projection':ccrs.NorthPolarStereo(central_longitude=central_longitude)} )    

    for a in fig.axes: 
        a.coastlines(linewidth=0.5,resolution='50m')
        a.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor=None,# don't use this for coast line problem with 180 degree longitude line drawn on north polar stereo
                                        facecolor=landcolor,alpha=0.7) )
        if extent is not None: a.set_extent( extent, ccrs.PlateCarree())
        if rnd: roundBoundary(a)
    
    return fig,ax





def singleMap(var, lon, lat, 
              fig=None, ax=None,
              cbar=True, shrinkcbar=0.4, cbarorient='horizontal', extend='both',
              unitName=None, title=None, 
              mask=xr.DataArray(True), landMask=xr.DataArray(True), 
              hatchMsk=None, noHatchNan=True, hatchlw=0.1, hlbl=None, hatchColor='k',
              **pltkw
             ):

    if fig is None: fig,ax=initmaps(figsize=(7,7))

    try: ax.set_title(var.attrs['standard_name'] if title is None else title)
    except: pass

    pl = ax.pcolormesh(lon, lat,
                       var.where(landMask).where(mask),
                       transform=ccrs.PlateCarree(), 
                       **pltkw
                        )
    
    if hatchMsk is not None:
        plt.rcParams['hatch.linewidth'] = hatchlw
        # add hatching everywhere withn a circle up to ~60N (done in axis coordinates)
        theta = np.linspace(-np.pi/2 *0.68, np.pi/2 *0.45 , 100)
        center, radius = [0.5, 0.5], 0.54
        cx,cyp = np.sin(theta)* radius + center[0], np.cos(theta)* radius + center[1]
        ax.fill_between(cx,cyp,1-cyp, transform=ax.transAxes,
                         hatch='xxx',color="none",edgecolor=hatchColor,lw=0,)#label=hlbl)
        # draw on land (+NE pacific, baltic, etc) to mask off hatching
        # ax.pcolormesh(lon, lat, dadic['landMask'].where(dadic['landMask']==0), 
        #               cmap='binary', transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, zorder=1, facecolor=[1,1,1])#, edgecolor='black')
        # redraw variable over hatching in areas without hatching
        ax.pcolormesh(lon, lat, var.where(np.logical_not(hatchMsk)).where(landMask).where(mask),
                      transform=ccrs.PlateCarree(), **pltkw)
        # redraw white over hatching in areas where variable is nan
        if noHatchNan: 
            ax.pcolormesh(lon, lat, 
                      var.isnull().where(var.isnull()), cmap='binary', transform=ccrs.PlateCarree())

    
    if cbar:
        cl = fig.colorbar(pl, ax=ax, pad=0.05, extend=extend, orientation=cbarorient, shrink=shrinkcbar)
        try: 
            cl.set_label(var.attrs['units'] if unitName is None else unitName) 
        except: pass
        return fig,ax,pl,cl
    else:
        return fig,ax,pl






def pltwithhatch(ax,lon,lat,var,
                 hatchMsk, hatchColor='k', hatchlw=0.1, noHatchNan=True, hatchStyle='xxx',
                 **pltkw):
    ## simpler version of hatching but does not seem to work with NAA
    
    # plot variable 
    pl=ax.pcolormesh(lon, lat, var, transform=ccrs.PlateCarree(), **pltkw)

    # add hatching
    plt.rcParams.update({'hatch.color': hatchColor, 'hatch.linewidth': hatchlw})
    ax.contourf(lon, lat, hatchMsk, 
                levels=[0.5,1], colors='none', hatches=[hatchStyle]*2,
                transform_first=True, transform=ccrs.PlateCarree(),
    )
    return pl




    
   
    