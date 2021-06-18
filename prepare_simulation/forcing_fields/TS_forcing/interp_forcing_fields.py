##!/usr/bin/env python2
## -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:57:32 2019

This file was used to interpolate the forcing fields of the Shishng forcing of 
POP.

@author: nooteboom
"""
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
ffile = Dataset(dirr + 'b.EO_38Ma_paleomag_2pic_f19g16_NESSC_control_correct_veg_final.pop.h.climatology.years_450-500.nc')

# Load the final grid we like to interpolate to 
readgrid = '/projects/0/palaeo-parcels/tx0.1_POP_EO38/velden/'
gridfile = Dataset(readgrid + 'grid_coordinates_pop_tx0.1_38ma.nc')
gridlon = gridfile['T_LON_2D'][:]
gridlat = gridfile['T_LAT_2D'][:]
gridarea = gridfile['TAREA'][:]
lon = ffile['TLONG'][:]; lon[lon>180] -= 360;
lat = ffile['TLAT'][:]

# Convert to a meshgrid
lons, lats = lon, lat

#Add some points in the north
# This is done to let all grid points of the new grid be within the domain of the old grid
def add_north(lats, lons, ndegs):
    for i in range(ndegs):
        deg = 90 + i/10.        
        lats = np.concatenate((lats,np.full(lons.shape[1], deg)[np.newaxis,:]), axis = 0)
        lons = np.concatenate((lons,lons[-1][np.newaxis,:]), axis = 0)
    return lats, lons
lats, lons = add_north(lats, lons, 40)

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
grid_T = np.full((12, gridlon.shape[0], gridlon.shape[1]),np.nan)
grid_SHF = np.full((12, gridlon.shape[0], gridlon.shape[1]),np.nan)
grid_SFWF = np.full((12, gridlon.shape[0], gridlon.shape[1]),np.nan)

# Load also the ocean mask:
readkmt = '/projects/0/palaeo-parcels/tx0.1_POP_EO38/velden/results/'
kmtf = Dataset(readkmt + 'kmt_tx0.1_POP_EO38_new.bin.swapped_final.nc')
oceanmask = (kmtf['kmt'][:]!=0)
assert (np.unique(oceanmask)==np.array([0,1])).all(), 'oceanmask does not consist of only zeros and ones'

#Define the relative grid area in the ocean:
grid_rel_area = np.zeros(gridlon.shape)
total_ocean_area = 0.
idx = np.where(oceanmask)

grid_rel_area, total_ocean_area = weigthed_large_matrix(gridarea,idx[0], idx[1], grid_rel_area)

def add_north_data(TX, TY, T,SHF,SFWF, ndeg):
    for i in range(ndeg):
        TX = np.concatenate((TX, TX[-1,:][np.newaxis,:]), axis = 0)
        TY = np.concatenate((TY, TY[-1,:][np.newaxis,:]), axis = 0)
        T = np.concatenate((T, T[-1,:][np.newaxis,:]), axis = 0)
        SHF = np.concatenate((SHF, SHF[-1,:][np.newaxis,:]), axis = 0)
        SFWF = np.concatenate((SFWF, SFWF[-1,:][np.newaxis,:]), axis = 0)
    return TX,TY,T,SHF,SFWF

for i in range(12):
    print('month: ',i)
    TX = ffile['TAUX'][:][i,0]   #taux shape: [record(12), time(1), nlat(384), nlon(320)]
    TY = ffile['TAUY'][:][i,0]
    T = ffile['TEMP'][:][i,0,0]   #temp shape : [record(12), time(1), zt(60), nlat(384), nlon(320)]
    SHF = ffile['SHF'][:][i,0]
    SFWF = ffile['SFWF'][:][i,0]

    print(SFWF.shape)
    SFWFtemp = SFWF[315:335, 220:246]
    to0 = (SFWFtemp<1000)
    print(to0.shape)

    SFWF[315:335, 220:246][to0] = 0

    maxvalSFWF = 0.0003
    tomaxval = np.where(np.logical_and(SFWF<1000, SFWF>maxvalSFWF))
    SFWF[tomaxval] = maxvalSFWF    
    # Add a '90 degrees North' latitude band, to assure that all interpolated values are within the domain of the starting grid
    TX,TY,T,SHF,SFWF = add_north_data(TX, TY, T,SHF,SFWF, 40)
    TX = np.concatenate((TX,TX,TX), axis = 1)
    TY = np.concatenate((TY,TY,TY), axis = 1)
    T = np.concatenate((T,T,T), axis = 1)
    SHF = np.concatenate((SHF,SHF,SHF), axis = 1)
    SFWF = np.concatenate((SFWF,SFWF,SFWF), axis = 1)
    #%%
    # Perform the interpolation 
    lonsred = lons[np.where(T<10000)]
    latsred = lats[np.where(T<10000)]
    pointsred = np.concatenate((lonsred.flatten()[:,np.newaxis], latsred.flatten()[:,np.newaxis]), axis=1)
    singular = (lats>84)
    
    
    mind, maxd = np.min(TX),np.max(TX)
    avgn = np.nanmean(TX[singular])
    TX[singular] = avgn
    TX[np.abs(TX)>10000] = 0
    grid_TX[i] = np.ma.getdata(griddata(points, TX.flatten(),(gridlon, gridlat), method='cubic'))
    grid_TX[i][np.where(oceanmask==0)] = 9.96921e+36
    mind, maxd = np.min(TY),np.max(TY)
    avgn = np.nanmean(TY[singular])
    TY[singular] = avgn    
    TY[np.abs(TY)>10000] = 0
    grid_TY[i] = np.ma.getdata(griddata(points, TY.flatten(),(gridlon, gridlat), method='cubic')) 
    grid_TY[i][np.where(oceanmask==0)] = 9.96921e+36
    
    # these three field are not close to zero near the boundary. Therefore first nearest
    # neighbour, then gaussian smoothing, then cubic spline interpolation
   
    nearest_T = griddata(pointsred, T[np.where(T<10000)].flatten(),(lons, lats), method='nearest')
    nearest_T = np.ma.getdata(nearest_T)
    gaus_smooth_T = gaussian_filter(nearest_T, sigma=5)
    mind, maxd = np.min(gaus_smooth_T),np.max(gaus_smooth_T)
    avgn = np.nanmean(gaus_smooth_T[singular])
    gaus_smooth_T[singular] = avgn
    grid_T[i] = np.ma.getdata(griddata(points, gaus_smooth_T.flatten(),(gridlon, gridlat), method='cubic'))
    grid_T[i][np.where(oceanmask==0)] = 9.96921e+36

    nearest_SFWF = griddata(pointsred, SFWF[np.where(SFWF<10000)].flatten(),(lons, lats), method='nearest')
    nearest_SFWF = np.ma.getdata(nearest_SFWF)
    gaus_smooth_SFWF = gaussian_filter(nearest_SFWF, sigma=5)
    mind, maxd = np.min(gaus_smooth_SFWF),np.max(gaus_smooth_SFWF)
    avgn = np.nanmean(gaus_smooth_SFWF[singular])
    gaus_smooth_SFWF[singular] = avgn
    grid_SFWF[i] = np.ma.getdata(griddata(points, gaus_smooth_SFWF.flatten(),(gridlon, gridlat), method='cubic'))
    grid_SFWF[i][np.where(oceanmask==0)] = 9.96921e+36
    
    grid_SFWF[i], total_surface_flux, avg = weigthed_large_matrix2(grid_SFWF[i], gridarea, grid_rel_area, idx[0], idx[1])

    print('Here we add a constant value to the whole field, to assure that the integral of the surface flux equals zero')
    print('average grid flux: ', avg)
    print('total SFWF: ', total_surface_flux)
    print('integrated total flux: ', check_integral(grid_SFWF[i], gridarea, idx[0], idx[1]))


    grid_SFWF[i], total_surface_flux, avg = weigthed_large_matrix2(grid_SFWF[i], gridarea, grid_rel_area, idx[0], idx[1])   
    print('average grid flux2: ', avg)
    print('total SFWF2: ', total_surface_flux)  
    int_flux = check_integral(grid_SFWF[i], gridarea, idx[0], idx[1])
    print('integrated total flux2: ', int_flux)

    
    nearest_SHF = griddata(pointsred, SHF[np.where(SHF<10000)].flatten(),(lons, lats), method='nearest')
    nearest_SHF = np.ma.getdata(nearest_SHF)
    gaus_smooth_SHF = gaussian_filter(nearest_SHF, sigma=5)
    mind, maxd = np.min(gaus_smooth_SHF),np.max(gaus_smooth_SHF)
    avgn = np.nanmean(gaus_smooth_SHF[singular])
    gaus_smooth_SHF[singular] = avg
    grid_SHF[i] = np.ma.getdata(griddata(points, gaus_smooth_SHF.flatten(),(gridlon, gridlat), method='cubic'))
    grid_SHF[i][np.where(oceanmask==0)] = 9.96921e+36

    grid_SHF[i], total_surface_flux, avg = weigthed_large_matrix2(grid_SHF[i], gridarea, grid_rel_area, idx[0], idx[1])
    int_flux = check_integral(grid_SHF[i], gridarea, idx[0], idx[1])
    
    grid_SHF[i], total_surface_flux, avg = weigthed_large_matrix2(grid_SHF[i], gridarea, grid_rel_area, idx[0], idx[1])   
    int_flux = check_integral(grid_SHF[i], gridarea, idx[0], idx[1])
#%%

# Write forcing fields to netcdf file:
dataset = Dataset('forcing_pop_38MA.nc', 'w')

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
sst = dataset.createVariable('SST', np.float32,('record','j_index','i_index',), fill_value=9.96921e+36)
shf = dataset.createVariable('SHF', np.float32,('record','j_index','i_index',), fill_value=9.96921e+36)
sfwf = dataset.createVariable('SFWF', np.float32,('record','j_index','i_index',), fill_value=9.96921e+36)

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
sst[:] = grid_T
shf[:] = grid_SHF
sfwf[:] = grid_SFWF

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

sst.long_name = 'Sea Surface Temperature'
sst.units = 'deg C'
sst.cell_methods = 'time: mean'
sst.missing_value = 9.96921e+36

shf.long_name = 'Total Surface Heat Flux, Including SW'
shf.units = 'watt/m^2'
shf.cell_methods = 'time: mean'
shf.missing_value = 9.96921e+36

sfwf.long_name = 'Virtual Salt Flux in FW Flux formulation'
sfwf.units = 'kg/m^2/s'
sfwf.cell_methods = 'time: mean'
sfwf.missing_value = 9.96921e+36

dataset.close()
