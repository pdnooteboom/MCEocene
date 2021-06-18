##!/usr/bin/env python2
## -*- coding: utf-8 -*-
#"""
#Created on Mon May  6 16:57:32 2019
#
#@author: nooteboom
#"""
#
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from numba import jit

@jit(nopython=True)
def weigthed_large_matrix(matrix,i1, i2, res):
    sum_matrix = 0.
    for i in range(len(i1)):
        sum_matrix += matrix[i1[i], i2[i]]
    for i in range(len(i1)):
        res[i1[i],i2[i]] =  matrix[i1[i], i2[i]] / sum_matrix
        
    return res, sum_matrix

@jit(nopython=True)
def weigthed_large_matrix2(matrix, matrixarea, matrix_rel_area,i1, i2):
    sum_matrix = 0.
    for i in range(len(i1)):
        sum_matrix += matrix[i1[i], i2[i]] * matrixarea[i1[i], i2[i]]  # kg / s
    for i in range(len(i1)):
        matrix[i1[i], i2[i]] -= (sum_matrix * matrix_rel_area[i1[i], i2[i]]) / matrixarea[i1[i], i2[i]]#  kg/ m^2 / s  = (kg/s  * 1) / m^2
    mean_matrix = sum_matrix / float(len(i1))
    return matrix, sum_matrix, mean_matrix

@jit(nopython=True)
def check_integral(matrix, matrixarea,i1, i2):
    res = 0.
    for i in range(len(i1)):
        res += matrix[i1[i], i2[i]] * matrixarea[i1[i], i2[i]]   # kg / s
    return res

#%% Read nc bathymetry file
# Load the coarse forcing fields
dirr = '/projects/0/palaeo-parcels/tx0.1_POP_EO38/gx1_forcing/'
print(dirr + 'b.EO_38Ma_paleomag_2pic_f19g16_NESSC_control_correct_veg_final.pop.h.climatology.years_450-500.nc')
ffile = Dataset(dirr + 'b.EO_38Ma_paleomag_2pic_f19g16_NESSC_control_correct_veg_final.pop.h.climatology.years_450-500.nc')

# Load the final grid we like to interpolate to 
readgrid = '/projects/0/palaeo-parcels/tx0.1_POP_EO38/velden/'
gridfile = Dataset(readgrid + 'grid_coordinates_pop_tx0.1_38ma.nc')
gridlon = gridfile['T_LON_2D'][:]
gridlat = gridfile['T_LAT_2D'][:]
gridarea = gridfile['TAREA'][:]
lon = ffile['TLONG'][:]; lon[lon>180] -= 360;
lat = ffile['TLAT'][:]
print(np.unique(lon.shape))
print(np.unique(gridlon.shape))

# Convert to a meshgrid
lons, lats = lon, lat

#Add some points in the north
def add_north(lats, lons, ndegs):
    for i in range(ndegs):
        deg = 90 + i/10.        
        lats = np.concatenate((lats,np.full(lons.shape[1], deg)[np.newaxis,:]), axis = 0)
        lons = np.concatenate((lons,lons[-1][np.newaxis,:]), axis = 0)
    return lats, lons
lats, lons = add_north(lats, lons, 40)
#lats = np.concatenate((lats,np.full(lons.shape[1], 91)[np.newaxis,:]), axis = 0)
#lons = np.concatenate((lons,lons[-1][np.newaxis,:]), axis = 0)
#lats = np.concatenate((lats,np.full(lons.shape[1], 92)[np.newaxis,:]), axis = 0)
#lons = np.concatenate((lons,lons[-1][np.newaxis,:]), axis = 0)

#lats = np.concatenate((lats[:,-1][:,np.newaxis],lats,lats[:,0][:,np.newaxis]), axis = 1)
#lons = np.concatenate((lons[:,-1][:,np.newaxis]-360,lons,lons[:,0][:,np.newaxis]+360), axis = 1)

lats = np.concatenate((lats[:],lats,lats), axis = 1)
lons = np.concatenate((lons[:]-360,lons,lons+360), axis = 1)

SHF = ffile['SHF'][:][0,0]
SHF = np.concatenate((SHF, SHF[-1,:][np.newaxis,:]), axis = 0)
SHF = np.concatenate((SHF[:,-1][:,np.newaxis],SHF,SHF[:,0][:,np.newaxis]), axis = 1)

lonsred = lons[np.where(SHF<10000)]
latsred = lats[np.where(SHF<10000)]
# Concatenate the gridlon and gridlat to one array:
points = np.concatenate((lons.flatten()[:,np.newaxis], lats.flatten()[:,np.newaxis]), axis=1)
pointsred = np.concatenate((lonsred.flatten()[:,np.newaxis], latsred.flatten()[:,np.newaxis]), axis=1)

# Will be the interpolation results:
grid_TX = np.full((12, gridlon.shape[0], gridlon.shape[1]),np.nan)
grid_TY = np.full((12, gridlon.shape[0], gridlon.shape[1]),np.nan)

# Load also the ocean mask:
readkmt = '/projects/0/palaeo-parcels/tx0.1_POP_EO38/velden/results/'
kmtf = Dataset(readkmt + 'kmt_tx0.1_POP_EO38.nc')
#oceanmask = kmtf['OCEAN_MASK'][:].astype(int)
oceanmask = (kmtf['kmt'][:]!=0)
assert (np.unique(oceanmask)==np.array([0,1])).all(), 'oceanmask does not consist of only zeros and ones'

#Define the relative grid area in the ocean:
grid_rel_area = np.zeros(gridlon.shape)
total_ocean_area = 0.
idx = np.where(oceanmask)

grid_rel_area, total_ocean_area = weigthed_large_matrix(gridarea,idx[0], idx[1], grid_rel_area)
print('total ocean area: ', total_ocean_area)

def add_north_data(TX, TY, ndeg):
    for i in range(ndeg):
        TX = np.concatenate((TX, TX[-1,:][np.newaxis,:]), axis = 0)
        TY = np.concatenate((TY, TY[-1,:][np.newaxis,:]), axis = 0)
    return TX,TY

for i in range(12):
    print('month: ',i)
    TX = ffile['TAUX'][:][i,0]   #taux shape: [record(12), time(1), nlat(384), nlon(320)]
    TY = ffile['TAUY'][:][i,0]
    
    # Add a '90 degrees North' latitude band, to assure that all interpolated values are within the domain of the starting grid
    TX,TY = add_north_data(TX, TY, 40) 
    
    # Use periodic boundary conditions in the longitude direction for the starting grid

    TX = np.concatenate((TX,TX,TX), axis = 1)
    TY = np.concatenate((TY,TY,TY), axis = 1)

    
    print('minlat, maxlat, minlon, maxlon of starting dataset: ', np.min(lats), np.max(lats), np.min(lons), np.max(lons))
    print('minlat, maxlat, minlon, maxlon of interpolated dataset: ', np.min(gridlat), np.max(gridlat), np.min(gridlon), np.max(gridlon))
    print('amount of NaN values in starting TX grid: ' , np.sum(np.isnan(TX)))
    #%%
    # Perform the interpolation 
    lonsred = lons[np.where(TX<10000)]
    latsred = lats[np.where(TX<10000)]
    pointsred = np.concatenate((lonsred.flatten()[:,np.newaxis], latsred.flatten()[:,np.newaxis]), axis=1)
    singular = (lats>78)#np.logical_and(np.logical_and(lons<14,lons>-38),lats>84)
    
    mind, maxd = np.min(TX),np.max(TX)
#    avgn = np.nanmean(TY[singular])
#    TY[singular] = avgn    
    TX[np.abs(TX)>10000] = 0
    grid_TX[i] = np.ma.getdata(griddata(points, TX.flatten(),(gridlon, gridlat), method='cubic')) 
    singular = (gridlat>87)#np.logical_and(np.logical_and(lons<14,lons>-38),lats>84)
    avgn = np.nanmean(grid_TX[i][singular])
    grid_TX[i][singular] = 0#avgn
    grid_TX[i] = gaussian_filter(grid_TX[i], sigma=7)
    print('avgn TX:', avgn)
    
    mind, maxd = np.min(TY),np.max(TY)
#    avgn = np.nanmean(TY[singular])
#    TY[singular] = avgn    
    TY[np.abs(TY)>10000] = 0
    grid_TY[i] = np.ma.getdata(griddata(points, TY.flatten(),(gridlon, gridlat), method='cubic')) 
    singular = (gridlat>87)#np.logical_and(np.logical_and(lons<14,lons>-38),lats>84)
    avgn = np.nanmean(grid_TY[i][singular])
    grid_TY[i][singular] = 0#avgn
    grid_TY[i] = gaussian_filter(grid_TY[i], sigma=7)
    print('avgn TY:', avgn) 

#    avgn = np.nanmean(TY[singular])
#    TY[singular] = avgn    
#    TY[singular] = gaussian_filter(TY[singular], sigma=15)
#    TY = gaussian_filter(TY, sigma=5)



#    grid_TY[i][np.where(oceanmask==0)] = 9.96921e+36

#%%

# Write forcing fields to netcdf file:
dataset = Dataset('TAU.nc', 'w')

i_indexs = dataset.createDimension('i_index', 3600)
j_indexs = dataset.createDimension('j_index', 2400+150)
depth_ts = dataset.createDimension('depth_t', 42)
w_deps = dataset.createDimension('w_dep', 43)
records = dataset.createDimension('record', 12)

ins = dataset.createVariable('i_index', np.float32,('i_index',))
jns = dataset.createVariable('j_index', np.float32,('j_index',))
depth_tns = dataset.createVariable('depth_t', np.float32,('depth_t',))
w_depns = dataset.createVariable('w_dep', np.float32,('w_dep',))
latitudes = dataset.createVariable('T_LAT_2D', np.float32,('j_index','i_index',))
longitudes = dataset.createVariable('T_LON_2D', np.float32,('j_index','i_index',))
latitudes2 = dataset.createVariable('U_LAT_2D', np.float32,('j_index','i_index',))
longitudes2 = dataset.createVariable('U_LON_2D', np.float32,('j_index','i_index',))

tx = dataset.createVariable('TAUX', np.float32,('record','j_index','i_index',), fill_value=9.96921e+36)
ty = dataset.createVariable('TAUY', np.float32,('record','j_index','i_index',), fill_value=9.96921e+36)

# Write data
ins[:] = gridfile['i_index'][:]
jns[:] = gridfile['j_index'][:]
depth_tns[:] = gridfile['depth_t'][:]
w_depns[:] = gridfile['w_dep'][:]
latitudes[:] = gridfile['T_LAT_2D'][:]
longitudes[:] = gridfile['T_LON_2D'][:]
latitudes2[:] = gridfile['U_LAT_2D'][:]
longitudes2[:] = gridfile['U_LON_2D'][:]

tx[:] = grid_TX
ty[:] = grid_TY

#Attributes:
latitudes.long_name = 'latitude on t-grid'
latitudes.units = 'degrees N'
longitudes.long_name = 'longitude on t-grid'
longitudes.units = 'degrees N'
latitudes2.long_name = 'latitude on u-grid'
latitudes2.units = 'degrees N'
longitudes2.long_name = 'longitude on u-grid'
longitudes2.units = 'degrees N'
w_depns.units = 'meters'
w_depns.long_name = 'T-grid depth'
depth_tns.units = 'meters'
depth_tns.long_name = 'T-grid depth'
ins.long_name = 'i-coordinate index'
jns.long_name = 'j-coordinate index'

tx.long_name = 'Windstress in grid-x direction'
tx.units = 'dyne/centimeter^2'
tx.cell_methods = 'time: mean'
tx.missing_value = 9.96921e+36
ty.long_name = 'Windstress in grid-y direction'
ty.units = 'dyne/centimeter^2'
ty.cell_methods = 'time: mean'
ty.missing_value = 9.96921e+36

dataset.close()
