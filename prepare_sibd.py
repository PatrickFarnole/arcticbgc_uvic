import xarray as xr
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

import utils

# GLOBAL VARIABLE
save_dir = '/home/pfarnole/data/sibd/' # your savedir

# COMPUTE SIBD for ASMR dataset
for y in range(2003,2020): # Dataset covers 2003-2019
    print(y)
    data = utils.prep_sibd_asmr(y)
    data.to_netcdf(os.path.join(save_dir,f'ASMR_50-85_{y}.nc'))


# COMPUTE SIBD for NAA runs
runs = {'DFS':'G510.00','CGRF':'C524.06','JRA':'C524.00-08'}
years = range(2003,2016)

for forcing in runs:
    print(forcing)
    for y in years:
        print(y)
        data = utils.prep_sibd_naa(forcing,y)
        data.to_netcdf(os.path.join(save_dir,f'{forcing}_{runs[forcing]}_{y}.nc'))


# COMPUTE SIBD FOR ANHA4 runs
### Coming soon
