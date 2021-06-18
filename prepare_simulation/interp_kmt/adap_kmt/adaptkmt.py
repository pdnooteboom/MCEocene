#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 08:20:40 2020

@author: nooteboom
"""

import numpy as np
from netCDF4 import Dataset
from numba import jit
import math
from scipy.ndimage import gaussian_filter

def skewnorm(mu, sigma, lam, x,maxdep=1200):
    h = np.zeros(len(x))
    for xi in range(len(x)):
        xn = (x[xi] - mu)/sigma
        if(xn<-3/lam):
            h[xi] = 0
        elif(xn<-1/lam):
            h[xi] = 1 / (8*np.sqrt(2*math.pi)) * math.e**(-xn**2/2)*(9*lam*xn+3*lam**2*xn**2+1/3*lam**3*xn**3+9)
        elif(xn<1/lam):
            h[xi] = 1 / (4*np.sqrt(2*math.pi)) * math.e**(-xn**2/2)*(3*lam*xn-1/3*lam**3*xn**3+4)
        elif(xn<3/lam):
            h[xi] = 1 / (8*np.sqrt(2*math.pi)) * math.e**(-xn**2/2)*(9*lam*xn-3*lam**2*xn**2+1/3*lam**3*xn**3+7)
        else:
            h[xi] = np.sqrt(2/math.pi)*math.e**(-xn**2/2)
    assert np.abs(np.max(h)) >= 1e-1, 'do not create land'
    mult = maxdep / np.max(h)
    return h*mult

@jit(nopython=True)
def create_bathymetry(kmt, wdep, bath):
    for i in range(bath.shape[0]):
        for j in range(bath.shape[1]):
            bath[i,j] = wdep[kmt[i,j]]
    return bath

def take_min_kmt(kmt, iman, imax, jmin, jmax, kmtmin):
    for i in range(imin, imax):
        for j in range(jmin, jmax):
            kmt[j,i] = min(kmt[j,i], kmtmin)
    return kmt

def create_bump(imin,imax, jmin, jmax, kmt, wdep, mu, sigma, maxdep=1200):
    for i in range(imin,imax):
        deps = skewnorm(mu, sigma, 5, np.array([jmax-ji+2 for ji in range(jmin,jmax)]), 
                        maxdep=maxdep)
        for j in range(jmin,jmax):
            depsk = deps[j-jmin]
#            print(skewnorm(5, 20, 5, np.array([jmax-j])))
            dep = wdep[kmt[j,i]]
            newdep = max(dep, depsk)
#            print(newdep, wdep)
            if(len(np.where(newdep>wdep)[0])>0):
                if(kmt[j,i]!=0): # do not adjust land
                    kmt[j,i] = np.where(newdep>wdep)[0][-1]+1
#            print(kmt[j,i])
    return kmt

typ = ''#'TGclosed'#'TG1-2km'#'DP100'#'DP2km'#'DP100'#'TG1-2km'#''#'TG1-2km'#'DPshallow'#'TGclosed'#

dirr = '/Users/nooteboom/Documents/PhD/Eocene_POP/velden/adapt_kmt/'
gridf = Dataset('/Users/nooteboom/Documents/PhD/Eocene_POP/velden/grid_coordinates_pop_tx0.1_38ma.nc')
wdep = gridf['w_dep'][:]
if(typ=='TGclosed'):

    kmtf = Dataset(dirr + 'kmt_tx0.1_POP_EO38.nc', 'r+')
    bath = kmtf.createVariable('bathymetry', 'f4', ('j_index','i_index',))
#    lats = kmtf['ULAT'][:]

    kmtf.createDimension('w_dep', len(wdep))
    wdeps = kmtf.createVariable('w_dep', 'f4', ('w_dep',))
    wdeps[:] = wdep
    
    imin = 2388
    imax = 2392
    jmin = 500
    jmax = 638
    
    
    kmtf['kmt'][jmin:jmax,imin:imax] = 0
    kmt = kmtf['kmt'][:]#
    kmt[jmin:jmax, imin-40:imax+40] = gaussian_filter(kmt[jmin-20:jmax+20, imin-80:imax+80], 2)[20:-20, 40:-40]
    kmt[jmin:jmax,imin:imax] = 0
    kmt[kmtf['kmt'][:]==0] = 0
    
    kmtf['kmt'][:] = kmt
    kmt = kmtf['kmt'][:].astype(int)
    
    kmtf['bathymetry'][:] = create_bathymetry(kmt, wdep, np.zeros(kmt.shape))
    
    kmtf.close()
elif(typ=='TG1-2km'):
    
    kmtf = Dataset(dirr + 'kmt_tx0.1_POP_EO38.nc', 'r+')
    bath = kmtf.createVariable('bathymetry', 'f4', ('j_index','i_index',))
    
    kmtf.createDimension('w_dep', len(wdep))
    wdeps = kmtf.createVariable('w_dep', 'f4', ('w_dep',))
    wdeps[:] = wdep
    
    imin = 2350
    imax = 2450
    jmin = 500
    jmax = 630
    
    maxdepth = 1200 # meter     
    
    kmt = kmtf['kmt'][:].astype(int)
    for i in range(2130, 2450):
        kmt = create_bump(i,i+1, jmin, jmax, kmt, wdep, -45+((i-2130)*90/320), 30)
        
#    kmt = create_bump(2380,imax, jmin, jmax, kmt, wdep, 25, 30)
#    kmt = create_bump(2350,2380, jmin, jmax, kmt, wdep, 20, 25)
#    kmt = create_bump(2330,2350, jmin, jmax, kmt, wdep, 15, 20)
    
    kmtf['kmt'][:] = kmt#create_bump(imin,imax, jmin, jmax, kmt, wdep)
    kmtf['bathymetry'][:] = create_bathymetry(kmt, wdep, np.zeros(kmt.shape))
    
    kmtf.close()
elif(typ=='DP100'):

    kmtf = Dataset(dirr + 'kmt_tx0.1_POP_EO38.nc', 'r+')
    bath = kmtf.createVariable('bathymetry', 'f4', ('j_index','i_index',))
#    lats = kmtf['ULAT'][:]

    kmtf.createDimension('w_dep', len(wdep))
    wdeps = kmtf.createVariable('w_dep', 'f4', ('w_dep',))
    wdeps[:] = wdep
    
    kmt = kmtf['kmt'][:].astype(int)
    
    imin = 217
    imax = 276
    jmin = 406
    jmax = 409
    kmt = take_min_kmt(kmt, imax, imax, jmin, jmax, 9)
    imin = 275
    imax = 277
    jmin = 407
    jmax = 530
    kmt = take_min_kmt(kmt, imax, imax, jmin, jmax, 9)  
    
    kmtf['kmt'][:] = kmt
    kmtf['bathymetry'][:] = create_bathymetry(kmt, wdep, np.zeros(kmt.shape))
    
    kmtf.close()
elif(typ=='DP2km'):
    
    kmtf = Dataset(dirr + 'kmt_tx0.1_POP_EO38.nc', 'r+')
    bath = kmtf.createVariable('bathymetry', 'f4', ('j_index','i_index',))
    
    kmtf.createDimension('w_dep', len(wdep))
    wdeps = kmtf.createVariable('w_dep', 'f4', ('w_dep',))
    wdeps[:] = wdep
    
    imin = 240
    imax = 310
    jmin = 420
    jmax = 490
    
    maxdepth = 2000 # meter     
    
    kmt = kmtf['kmt'][:].astype(int)
    kmt = create_bump(imin,imax, jmin, jmax, kmt, wdep, 35, 30, maxdep=maxdepth)
    
    kmtf['kmt'][:] = kmt#create_bump(imin,imax, jmin, jmax, kmt, wdep)
    kmtf['bathymetry'][:] = create_bathymetry(kmt, wdep, np.zeros(kmt.shape))
    
    kmtf.close()
else:
    
    kmtf = Dataset(dirr + 'kmt_tx0.1_POP_EO38.nc', 'r+')
    bath = kmtf.createVariable('bathymetry', 'f4', ('j_index','i_index',))
 
    kmtf.createDimension('w_dep', len(wdep))
    wdeps = kmtf.createVariable('w_dep', 'f4', ('w_dep',))
    wdeps[:] = wdep
    
    kmt = kmtf['kmt'][:].astype(int)
    
    kmtf['kmt'][:] = kmt#create_bump(imin,imax, jmin, jmax, kmt, wdep)
    kmtf['bathymetry'][:] = create_bathymetry(kmt, wdep, np.zeros(kmt.shape))
    
    kmtf.close()    
