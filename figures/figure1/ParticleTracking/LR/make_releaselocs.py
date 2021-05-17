from parcels import (FieldSet, ParticleSet, JITParticle, AdvectionRK4_3D,
                     ErrorCode, ParticleFile, Variable, Field)
from datetime import timedelta as delta
from datetime import  datetime
import numpy as np
import math
from glob import glob
import sys
import pandas as pd

dirread_grid = '/projects/0/acc/cesm/cesm1_0_5/b.EO_38Ma_paleomag_2pic_f19g16_NESSC_control_correct_veg_final/OUTPUT/ocn/hist/yearly/'
bfile = dirread_grid+'b.EO_38Ma_paleomag_2pic_f19g16_NESSC_control_correct_veg_final.pop.h.avg0576.nc'

dirread_grid = '/projects/0/palaeo-parcels/Eocene/grids/LR/'
bfile = dirread_grid+ 'bathymetry.nc'


bvariables = ('B', 'HT')
bdimensions = {'lon': 'ULONG', 'lat': 'ULAT'}

indices = {'lat': range(15,151)}

Bfield = Field.from_netcdf(bfile, bvariables, bdimensions, allow_time_extrapolation=True, interp_method='bgrid_tracer', field_chunksize=False, indices=indices)

for h in range(8):
    print(h)
#    if(h%2==0):
#    lons, lats = np.meshgrid(np.arange(115, 210,2)+0.5, np.arange(-80+6*h,-74+6*h,1)+0.5)
    lons, lats = np.meshgrid(np.arange(0, 360,2)+0.5, np.arange(-80+6*h,-74+6*h,1)+0.5)
#    else:
#        lons, lats = np.meshgrid(np.arange(60, 210,2)+0.5, np.arange(-79+6*h,-75+5*h,2)+0.5)
    lons = lons.flatten()
#    lons[lons>180] -= 360
    lons[lons>340] -= 360
    lats = lats.flatten()

#    idx = np.where(np.logical_and(np.isclose(lons, 292.5, atol=3), np.isclose(lats, -80.5, atol=3)))[0]
#    lons = lons[np.invert(idx)]; lats = lats[np.invert(~idx)];

    print(lons.shape)

    lon = np.zeros(0)
    lat = np.zeros(0)
    for i in range(len(lons)):
        #if(not (np.isclose(lons[i], 292.5, atol=0) and np.isclose(lats[i], -80.5, atol=3))):
        #if(not (lons[i]>= 288.5 and np.isclose(lats[i], -80.5, atol=5))):
        #if(not (np.isclose(lats[i], -80.5, atol=5))):
        try:
            if(Bfield[0,0, lats[i], lons[i]]>0):
                lon = np.append(lon,lons[i])
                lat = np.append(lat,lats[i])
        except:
            print(lons[i], lats[i])

    np.savez('releaselocs/locs%d.npz'%(h), lons=lon, lats=lat)



