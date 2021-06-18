#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:08:09 2019

@author: nooteboom
"""

import numpy as np
from scipy.interpolate import griddata
from netCDF4 import Dataset
import csv
from numba import jit
import time as ti
from scipy.ndimage import gaussian_filter
import datetime

@jit(nopython=True)
def loop_land(array, kmt, l0, l1):
    for i in range(l0):
        for j in range(l1):
            dl = kmt[i,j]
            if(dl<41):
                array[dl:,i,j] = -1
    return array

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def interpolator_3D(depth_highres, lats_highres, lons_highres, depth_lowres, 
                    lats_lowres, lons_lowres, kmt, rf, name = '', pic=2):
    print(name)
    array = np.full((depth_highres.shape[0], lats_highres.shape[0], lons_highres.shape[1]),-1.)
    for d in range(len(depth_highres)):
        dl = min(58,find_nearest_index(depth_lowres, depth_highres[d]))
        array[d] = interpolator_2D(lats_highres, lons_highres, lats_lowres, lons_lowres, 
                    kmt, rf, name = name, dep=dl, dep_hr=d, pic=pic)    
    return array

@jit(nopython=True)
def loop_land2D(arrayy, kmt, l0, l1, d=0):
    result = arrayy.copy()
    for i in range(l0):
        for j in range(l1):
            dl = kmt[i,j]
            if(dl<=d):
                result[i,j] = 0
    return result

@jit(nopython=True)
def loop_land2D_nan(arrayy, kmt, l0, l1, d=0):
    for i in range(l0):
        for j in range(l1):
            dl = kmt[i,j]
            if(dl<=d):
                arrayy[i,j] = np.nan
    return arrayy

def interpolator_2D(lats_highres, lons_highres, lats_lowres, lons_lowres, 
                    kmt, rf, name = '', dep=0, dep_hr=0, pic=2):
    assert (len(rf[name][:].shape)!=2), 'use interp_init_alot.py if you like to interpolate 2d fields'
    assert (len(rf[name][:].shape)in [3,4]), 'interpolate a 3D field'
    if(pic==2):
        ocean = np.where(rf[name][dep,:]!=0)
        points = np.concatenate((np.expand_dims(lats_lowres[ocean].flatten(), axis=0),np.expand_dims(lons_lowres[ocean].flatten(), axis=0)), axis=0)
        assert points.shape[1]==rf[name][dep,:][ocean].flatten().shape[0], 'amount of points must equal input data for interpolation'
        if((rf[name][dep,:][ocean]==0).all()):
            array = np.zeros(lats_highres.shape)
            print('only ocean')
        else:
            array = griddata(np.swapaxes(points,0,1), rf[name][dep,:][ocean].flatten().astype(float), (lats_highres, lons_highres), method='nearest').astype(float)
        arrayt = loop_land2D(array, kmt, array.shape[0], array.shape[1], d=dep_hr)
        array = zonal_averages(array, arrayt, lats_highres, lons_highres, 2)

        array[:300] = gaussian_filter(array, sigma=30)[:300]

        array = gaussian_filter(array, sigma=5)
    elif(pic==4):
        ocean = np.where(rf[name][0,dep,:]<1e10)
        points = np.concatenate((np.expand_dims(lats_lowres[ocean].flatten(), axis=0),np.expand_dims(lons_lowres[ocean].flatten(), axis=0)), axis=0)
        assert points.shape[1]==rf[name][0,dep,:][ocean].flatten().shape[0], 'amount of points must equal input data for interpolation'
        if((rf[name][0,dep,:][ocean]==0).all()):
            array = np.zeros(lats_highres.shape)
            print('only ocean')
        else:    
            array = griddata(np.swapaxes(points,0,1), rf[name][0,dep,:][ocean].flatten().astype(float), (lats_highres, lons_highres), method='nearest').astype(float)
        arrayt = loop_land2D(array, kmt, array.shape[0], array.shape[1], d=dep_hr)
        array = zonal_averages(array, arrayt, lats_highres, lons_highres, 2)
        array[:300] = gaussian_filter(array, sigma=30)[:300]
        array = gaussian_filter(array, sigma=5)
    else:
        assert False,'pic should be 2 or 4'
    return array

def zonal_averages(array, arrayt, lats_highres, lons_highres, lat_int): #arrayt,
    # by indices instead of latitudes
    result = array.copy()
    ids = 240
    idx = np.where(arrayt[:ids]!=0)
    result[:ids] = np.nanmean(arrayt[:ids][idx])
    return result

#%%
# Read the lowres locations
pic = 4 # choose 2 or 4, so 2 or 4 times pre-industrial carbon configuration

rf = Dataset('TSm01_CESM_38Ma_2PIC.nc','r') # Use only the longitudes and latitudes of this file. Same for 4PIC and 2 PIC
lons_lowres = rf['LON'][:]
lons_lowres[lons_lowres>180] -= 360
lats_lowres = rf['LAT'][:]
with open ('Layers.txt', 'r') as f:
    depth_lowres = np.array([float(row[1]) for row in csv.reader(f,delimiter='\t')])

# Read the high resolution kmt file 
read_kmt = '/projects/0/palaeo-parcels/tx0.1_POP_EO38/velden/results/'
kmt = Dataset(read_kmt + 'kmt_tx0.1_POP_EO38.nc')['kmt'][:]

# Read the highres grid file
read_nf = '/home/nooteb/petern/parcels/38MA/interp_kmt/'
nf = Dataset(read_nf + 'grid_coordinates_pop_tx0.1_38ma.nc','r')

lons_highres = nf['T_LON_2D'][:]
lats_highres = nf['T_LAT_2D'][:]
    
print('start interpolation and write file')

# Read the lowres data
if(pic==2):
    read_lr = '/projects/0/palaeo-parcels/tx0.1_POP_EO38/initial_TEMP_SALT/'#'/home/nooteb/petern/initial_TEMP_SALT/'
    rf2 = Dataset(read_lr + 'b.EO_38Ma_paleomag_2pic_f19g16_NESSC_control_correct_veg_final.pop.r.0600-01-01-00000.nc','r')
elif(pic==4):
    read_lr = '/projects/0/palaeo-parcels/tx0.1_POP_EO38/initial_TEMP_SALT/'#'/home/nooteb/petern/initial_TEMP_SALT/'
    rf2 = Dataset(read_lr + 'b.EO_38Ma_paleomag_4pic_f19g16_NESSC_control_correct_veg_final.pop.h.0600-12.nc','r')

#write the spinup file:
dataset = Dataset('spinup_38MA_pop_tx01_%dpic.nc'%(pic), 'w')

#create dimensions
sj = dataset.createDimension('j', 2550)
si = dataset.createDimension('i', 3600)
sk = dataset.createDimension('k', 42)

TEMP_CURs = dataset.createVariable('TEMPERATURE', np.float64, ('k','j','i',)) 
SALT_CURs = dataset.createVariable('SALINITY', np.float64, ('k','j','i',)) 

# copy global attributes all at once via dictionary
dataset.setncatts(rf2.__dict__)
# copy all file data except for the excluded
for name, variable in rf2.variables.items():
    if(name in ['TEMPERATURE', 'SALINITY']):
    # copy variable attributes all at once via dictionary
        dataset[name].setncatts(rf2[name].__dict__)

print('unique lons lowres: ', np.unique(lons_lowres))
print('unique lons highres ',np.unique(lons_highres))

time = ti.time()

# Increase the dimensions of the grid with the depths:
##for the high res: 
depth_highres = nf['depth_t'][:]

print('second the 3D interpolations')
time = ti.time()
if(pic==2):
    TEMP_CURs[:] = interpolator_3D(depth_highres, lats_highres, lons_highres, depth_lowres, 
                        lats_lowres, lons_lowres, kmt, rf2, name = 'TEMP_CUR')
    print('One 3D interpolation time (s):  ', ti.time()-time)

    SALT_CURs[:] = 1000. * interpolator_3D(depth_highres, lats_highres, lons_highres, depth_lowres, 
                        lats_lowres, lons_lowres, kmt, rf2, name = 'SALT_CUR')
elif(pic==4):
    TEMP_CURs[:] = interpolator_3D(depth_highres, lats_highres, lons_highres, depth_lowres, 
                        lats_lowres, lons_lowres, kmt, rf2, name = 'TEMP', pic=4)
    print('One 3D interpolation time (s):  ', ti.time()-time)

    SALT_CURs[:] = interpolator_3D(depth_highres, lats_highres, lons_highres, depth_lowres, 
                        lats_lowres, lons_lowres, kmt, rf2, name = 'SALT', pic=4)

#%% Write file  
dataset.close()
