#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:06:46 2019

@author: nooteboom
"""

import numpy as np
from netCDF4 import Dataset
import matplotlib.pylab as plt

# Read the bathymetry and the grid file
topbathfile = Dataset('interp_kmt/TopoBathy38.nc')
gridfile = Dataset('grid_coordinates_pop_tx0.1.nc')

Z = topbathfile['Z'][:]
lon = topbathfile['longitude'][:]
lat = topbathfile['latitude'][:]

gridlon = gridfile['T_LON_2D'][:]
gridlon[gridlon>180] -= 360
gridlat = gridfile['T_LAT_2D'][:]
idi = gridfile['i_index'][:]

gridlontemp = gridfile['U_LON_2D'][:]
gridlattemp = gridfile['U_LAT_2D'][:]

# Read the kmt file 
kmtf = Dataset('interp_kmt/kmt_pbc.p1_tripole.s2.0-og.20060315.no_caspian_or_black.nc')
kmtlon = kmtf['ULON'][:]
kmtlat = kmtf['ULAT'][:]
htn = kmtf['HTN'][:]
hte = kmtf['HTE'][:]
hus = kmtf['HUS'][:]
huw = kmtf['HUW'][:]
angle = kmtf['ANGLE'][:]
#%%
# Compare longitudes and latitudes of kmt and grid files
plt.plot(kmtlon[-1,:])
plt.plot(kmtlon[-2,:])
plt.plot(gridlon[-1,:], '--')
plt.show()
plt.plot(kmtlat[-1,:], label='kmt')
plt.plot(kmtlat[-2,:], label='kmt 2')
plt.plot(gridlattemp[-1,:], '--', label='grid')
plt.legend()
plt.show()

# Plot grids
gridlon = gridlon[::10,::10]
gridlat = gridlat[::10,::10]
plt.figure(figsize=(15,15))
plt.contourf(lon, lat, Z, [-10000,-0.1,0.1,10000], cmap='Spectral')
plt.colorbar()
plt.scatter(gridlon.flatten(), gridlat.flatten(), s=1, c='k')
plt.show()

gridlon = gridlon + 25
gridlon[gridlon>180] -= 360
plt.figure(figsize=(15,15))
plt.contourf(lon, lat, Z, [-10000,-0.1,0.1,10000], cmap='Spectral')
plt.colorbar()
plt.scatter(gridlon.flatten(), gridlat.flatten(), s=1, c='k')
plt.show()

#%%
gridlon = gridfile['T_LON_2D'][:]
gridlon[gridlon>180] -= 360
gridlat = gridfile['T_LAT_2D'][:]
gridlon2 = gridfile['U_LON_2D'][:]
gridlon2[gridlon2>180] -= 360
gridlat2 = gridfile['U_LAT_2D'][:]
idi = gridfile['i_index'][:]
idj = gridfile['j_index'][:]

#Determine 'minlat' the lowest latitude where there is ocean
minlat = 20
for lo in range(len(lon)):
    dep = Z[:,lo]
    bo = True
    la = 0
    while(bo):
        if(dep[la]<0):
            bo = False
            if minlat>lat[la]:
                minlat = lat[la]
        la += 1
        
minlatold = gridlat[0,0]
minlatold2 = gridlat2[0,0]
latstep = 0.04225922

def concatenate_south(newlats, old):
    new = np.zeros((len(newlats),3600));
    for i in range(len(newla)): new[i,:] = old[0,:];
    res = np.concatenate((new,old),axis=0)
    return res

#concatenate latitudes south of the grid
#For the latitudes of the grid (for U-grid and T-grid):
newlats = np.flip(-np.arange(-minlatold+latstep,-minlat+2*latstep, latstep))
newla = np.zeros((len(newlats),3600)); 
for i in range(3600): newla[:,i] = newlats;
gridlat = np.concatenate((newla,gridlat), axis=0)

newlats2 = np.flip(-np.arange(-minlatold2+latstep,-minlat+latstep, latstep))
newla = np.zeros((len(newlats),3600)); 
for i in range(3600): newla[:,i] = newlats2;
gridlat2 = np.concatenate((newla,gridlat2), axis=0)

#For the longitudes of the grid (for U-grid and T-grid):
newlo = np.zeros((len(newlats),3600));
for i in range(3600): newlo[:,i] = np.full(len(newlats), gridlon[0,i]);
gridlon = np.concatenate((newlo,gridlon),axis=0)
newlo = np.zeros((len(newlats),3600));
for i in range(3600): newlo[:,i] = np.full(len(newlats), gridlon2[0,i]);
gridlon2 = np.concatenate((newlo,gridlon2),axis=0)

# Also concatenate for the other grids: (leave the last latitude out)
htn = concatenate_south(newlats, htn[:-1])
hte = concatenate_south(newlats, hte[:-1])
hus = concatenate_south(newlats, hus[:-1])
huw = concatenate_south(newlats, huw[:-1])
angle = concatenate_south(newlats, angle[:-1])

new_j_index = np.append(np.arange(1,len(newlats)+1),idj+149)

gridlon = gridlon + 25
gridlon[gridlon>180] -= 360

gridlonplot = gridlon[::10,::10]
gridlatplot = gridlat[::10,::10]

plt.figure(figsize=(15,15))
plt.contourf(lon, lat, Z, [-10000,-0.1,0.1,10000], cmap='Spectral')
plt.colorbar()
plt.scatter(gridlonplot.flatten(), gridlatplot.flatten(), s=1, c='k')
plt.show()

plt.figure(figsize=(5,5))
plt.contourf(gridlonplot)
plt.colorbar()
plt.title('gridlons')
plt.show()

#%% Calculate the area of every grid cell
tarea = np.full(htn.shape, np.nan)

for j in range(htn.shape[0]):
    if(j%300==0):
        print(j/float(htn.shape[0]))
    for i in range(htn.shape[1]):
        dxt = (htn[j,i] + htn[j,(i+1)%htn.shape[0],]) / 2.
        dyt = (hte[j,i] + hte[j,i-1]) / 2.
        tarea[j,i] = dxt * dyt
        
assert ~np.isnan(tarea).any(), 'some area is NaN'
#%% Write new netcdf file with new grid
dataset = Dataset('grid_coordinates_pop_tx0.1_38ma.nc', 'w')

i_indexs = dataset.createDimension('i_index', 3600)
j_indexs = dataset.createDimension('j_index', 2400+len(newlats))
depth_ts = dataset.createDimension('depth_t', 42)
w_deps = dataset.createDimension('w_dep', 43)

ins = dataset.createVariable('i_index', np.float32,('i_index',))
jns = dataset.createVariable('j_index', np.float32,('j_index',))
depth_tns = dataset.createVariable('depth_t', np.float32,('depth_t',))
w_depns = dataset.createVariable('w_dep', np.float32,('w_dep',))
latitudes = dataset.createVariable('T_LAT_2D', np.float32,('j_index','i_index',))
longitudes = dataset.createVariable('T_LON_2D', np.float32,('j_index','i_index',))
latitudes2 = dataset.createVariable('U_LAT_2D', np.float32,('j_index','i_index',))
longitudes2 = dataset.createVariable('U_LON_2D', np.float32,('j_index','i_index',))

htns = dataset.createVariable('HTN', np.float32,('j_index','i_index',))
htes = dataset.createVariable('HTE', np.float32,('j_index','i_index',))
huss = dataset.createVariable('HUS', np.float32,('j_index','i_index',))
huws = dataset.createVariable('HUW', np.float32,('j_index','i_index',))
angles = dataset.createVariable('ANGLE', np.float32,('j_index','i_index',))
tareas = dataset.createVariable('TAREA', np.float32,('j_index','i_index',))

# Write data
ins[:] = idi
jns[:] = new_j_index
depth_tns[:] = gridfile['depth_t'][:]
w_depns[:] = gridfile['w_dep'][:]
latitudes[:] = gridlat
longitudes[:] = gridlon
latitudes2[:] = gridlat2
longitudes2[:] = gridlon2

htns[:] = htn
htes[:] = hte
huss[:] = hus
huws[:] = huw
angles[:] = angle
tareas[:] = tarea

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


dataset.close()



