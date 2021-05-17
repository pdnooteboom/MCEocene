# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:30:32 2018

This file merges the .nc files with back-tracked particles from different 
domains together.

@author: nooteboom
"""

from __future__ import division
import numpy as np
from netCDF4 import Dataset

#%% Some functions
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]
    
def find_nearest_index(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx    
    
def find_down(array,value):
    if(value>=array[0]):
        array = [n-value for n in array]
        idx = np.array([n for n in array if n<0]).argmax()
    else:
        idx = np.nan
    return idx     

#%% To open a file
adv = True#False
nadv = False#True

config = '4pic'
res = 1
sp = 6.
dd = 10. 
ext = ''
ddeg = 1
dirRead = '/projects/0/palaeo-parcels/Eocene/PT/'
dirWrite2 = ''

#%% Getting the average and variance differences 
 
pltse = 'annual' 
if(pltse=='winter'):
    se = 0
elif(pltse=='spring'):
    se = 1
elif(pltse=='summer'):
    se = 2
elif(pltse=='autumn'):
    se = 3
else:
    se = 4

if(adv):
    lons0 = np.array([])
    lats0 = np.array([])
    lons = np.array([])
    lats = np.array([])
    season = np.array([])
    temp = np.array([])
    salin = np.array([])
    time = np.array([])
    age = np.array([])
    zs = np.array([])

#ncn = Dataset(dirRead + 'surface/surfacegrid' +'_id'+'_dd'+str(int(dd))+"_res"+str(res) + '.nc')
#Arrays for fixed surface locations
for posidx in range(8):
    if(posidx%10==0):
        print(posidx)
    if(adv): # atsf_dd10_sp6_0_2pic.nc
        if(ext=='Tas'):
            nc = Dataset(dirRead + 'sp%d/Tas_dd%d_sp%d_%d_%s.nc'%(sp,dd,sp,posidx,config)) 
        else:
            nc = Dataset(dirRead + 'sp%d/atsf_dd%d_sp%d_%d_%s.nc'%(sp,dd,sp,posidx,config)) 
        if(nc['lon0'][0,0]<0):
            lons0 = np.append(lons0,nc['lon0'][:,0]+360)   
        else:
            lons0 = np.append(lons0,nc['lon0'][:,0])         
#        lons0 = np.concatenate((lons0,nc['lon0'][:,0]),axis=0)
        lats0 = np.concatenate((lats0,nc['lat0'][:,0]), axis=0)
        lons = np.concatenate((lons,nc['lon'][:,0]),axis=0)
        lats = np.concatenate((lats,nc['lat'][:,0]), axis=0) 
        temp = np.concatenate((temp,nc['temp'][:,0]), axis=0) 
        salin = np.concatenate((salin,nc['salin'][:,0]), axis=0)
        time = np.concatenate((time,nc['time'][:,0]), axis=0)
        age = np.concatenate((age,nc['age'][:,0]), axis=0)
        zs = np.concatenate((zs,nc['z'][:,0]), axis=0)    
        
if(adv):
    lons = lons%360
    lons0 = lons0%360

if(nadv):
    fixlon = fixlon%360
    
    fixtemp = np.ma.filled(fixtemp, fill_value=np.nan)
    fixsalin = np.ma.filled(fixsalin, fill_value=np.nan)
    fixtime = np.ma.filled(fixtime, fill_value=np.nan)
    fixlon = np.ma.filled(fixlon, fill_value=np.nan)
    fixlat = np.ma.filled(fixlat, fill_value=np.nan) 

#%% Write two netcdf files where all posidx files are concatenated
#First the advected nc file
if(adv):
    dirWrite = ''  
    if(ext=='Tas'):
        dataset = Dataset(dirWrite + 'concatenatedTas%s_sp%d_dd%d_res%d'%(config,int(sp),int(dd),res)+'.nc','w',format='NETCDF4_CLASSIC')
    else:
        dataset = Dataset(dirWrite + 'concatenated%s_sp%d_dd%d_res%d'%(config,int(sp),int(dd),res)+'.nc','w',format='NETCDF4_CLASSIC')

    traj = dataset.createDimension('traj',lats.shape[0])
    
    times = dataset.createVariable('time', np.float64, ('traj',))
    lat = dataset.createVariable('lat', np.float64, ('traj',))
    lon = dataset.createVariable('lon', np.float64, ('traj',))
    salins = dataset.createVariable('salin', np.float64, ('traj',)) 
    temps = dataset.createVariable('temp', np.float64, ('traj',))
    ages = dataset.createVariable('age', np.float64, ('traj',))
    lat0 = dataset.createVariable('lat0', np.float64, ('traj',))
    lon0 = dataset.createVariable('lon0', np.float64, ('traj',))
    z = dataset.createVariable('z', np.float64, ('traj',))
    
    lat[:] = lats
    lon[:] = lons
    lon0[:] = lons0
    lat0[:] = lats0
    times[:] = time
    salins[:] = salin
    temps[:] = temp
    ages[:] = age
    z[:] = zs
    
    dataset.close()

# Then the fixed surface nc file
if(nadv):
    dataset = Dataset(dirWrite2 + 'concatenatedsurface_dd%d_res%d.nc'%(int(dd),res),'w',format='NETCDF4_CLASSIC')

    fixtemp = fixtemp[:,1:]
    fixlon = fixlon[:,1:]
    fixlat = fixlat[:,1:]
    fixtime = fixtime[:,1:]
    fixsalin = fixsalin[:,1:]
    
    traj = dataset.createDimension('traj',fixlon.shape[0])
    obs = dataset.createDimension('obs',fixlon.shape[1])
    
    fixlons = dataset.createVariable('lon', np.float64, ('traj','obs',))
    fixlats = dataset.createVariable('lat', np.float64, ('traj','obs',))
    fixtimes = dataset.createVariable('time', np.float64, ('traj','obs',))
    fixtemps = dataset.createVariable('temp', np.float64, ('traj','obs',))
    fixsalins = dataset.createVariable('salin', np.float64, ('traj','obs',))
    
    print('any nan in fixtemp:', np.isnan(fixtemp).any())
    fixlons[:] = fixlon[:]; del fixlon;
    fixlats[:] = fixlat[:]; fixlat
    fixtimes[:] = fixtime[:]; del fixtime
    fixtemps[:] = fixtemp[:];
    fixsalins[:] = fixsalin; del fixsalin
    
    dataset.close()
