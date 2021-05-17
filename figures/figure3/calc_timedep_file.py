#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:35:51 2020

@author: nooteboom
"""

from netCDF4 import Dataset
import numpy as np
from glob import glob

lw=2
yl = [-5, 5]
fs=18
years = 34
years2 = 20
coms = [''] # Considers the global horizontal mean

#%% The transports part
#% Load the transport content file
readTR = '/Volumes/HD/Eocene/output/transport_files/'

names = np.loadtxt(readTR + 'tx01_EO38_transport_contents', 
                   skiprows=1, usecols=7, dtype=str)
transports = {}
for name in names: transports[name] = np.array([]);
order = {}
for name in names: order[name] = np.array([]);
timet = np.array([])  # in days


#%load the transport files
files = glob(readTR+'transp.*')
tr_col = 1
for f in files:
    namea = np.loadtxt(f, 
               skiprows=1, usecols=4, dtype=str)
    
    for ni in range(len(names)):
        idx = np.where(names[ni]==namea)
        if(ni==0):
            timet = np.append(timet, np.loadtxt(f, skiprows=1, 
                                              usecols=0)[idx])
        transports[names[ni]] = np.append(transports[names[ni]], 
                  np.loadtxt(f, skiprows=1, usecols=tr_col,
                             converters = {tr_col: lambda s: float(s[:-4])*10**(float(s[-3:]))})[idx])
        order[names[ni]] = np.append(order[names[ni]], 
                  np.loadtxt(f, skiprows=1, usecols=tr_col,
                             converters = {tr_col: lambda s: (float(s[-3:]))})[idx])

# Sort transport by timet 
for ni in range(len(names)):
    transports[names[ni]] = transports[names[ni]][np.argsort(timet)]
timet = np.sort(timet); timet /= 365

#% Also create yearly averages
yearlytransports = {}
for name in names: yearlytransports[name] = np.array([]);
yearstep = 365
yeartimet = np.array([])
for t in range(len(transports['EJ'][::yearstep])-1):
    for ni in range(len(names)):
        if(ni==0):
            yeartimet = np.append(yeartimet, t)
        yearlytransports[names[ni]] = np.append(yearlytransports[names[ni]], 
                        np.mean(transports[names[ni]][t*yearstep:(t+1)*yearstep]))
    
    
#% Load the transport content file 4pic
readTR = '/Volumes/HD/Eocene/output/transport_files/4pic/'
 
transports4 = {}
for name in names: transports4[name] = np.array([]);
order4 = {}
for name in names: order4[name] = np.array([]);
timet4 = np.array([])  # in days


#%load the transport files
files = glob(readTR+'transp.*')
tr_col = 1
for f in files:
    namea = np.loadtxt(f, 
               skiprows=1, usecols=4, dtype=str)
    
    for ni in range(len(names)):
        idx = np.where(names[ni]==namea)
        if(ni==0):
            timet4 = np.append(timet4, np.loadtxt(f, skiprows=1, 
                                              usecols=0)[idx])
        transports4[names[ni]] = np.append(transports4[names[ni]], 
                  np.loadtxt(f, skiprows=1, usecols=tr_col,
                             converters = {tr_col: lambda s: float(s[:-4])*10**(float(s[-3:]))})[idx])
        order4[names[ni]] = np.append(order4[names[ni]], 
                  np.loadtxt(f, skiprows=1, usecols=tr_col,
                             converters = {tr_col: lambda s: (float(s[-3:]))})[idx])

# Sort transport by timet 
for ni in range(len(names)):
    transports4[names[ni]] = transports4[names[ni]][np.argsort(timet4)]
timet4 = np.sort(timet4); timet4 /= 365

#% Also create yearly averages
yearlytransports4 = {}
for name in names: yearlytransports4[name] = np.array([]);
yearstep = 365
yeartimet4 = np.array([])
for t in range(len(transports4['EJ'][::yearstep])-1):
    for ni in range(len(names)):
        if(ni==0):
            yeartimet4 = np.append(yeartimet4, t)
        yearlytransports4[names[ni]] = np.append(yearlytransports4[names[ni]], 
                        np.mean(transports4[names[ni]][t*yearstep:(t+1)*yearstep]))

#% Set the same ylims for 2pic and 4pic
plotnames = ['Drake','Tasman','Agulhas']  # the tranport files that will be plotted
ylims = {}
for pn in range(len(plotnames)):
    ylims[plotnames[pn]] = [np.min(transports[plotnames[pn]][40:])-0.05, 
                             np.max(transports[plotnames[pn]][40:])+0.05]
#%% The hovmuller part
var = 'TEMP'#'TEMP'#'N' # TEMP # PD # SALT # V
unit = {'TEMP':'$^{\circ}$C','N':'s$^{-1}$',
        'PD':'gr cm$^{-3}$','SALT':'psu','V':'cm s$^{-1}$'}
tits = {'':'global','N':'above 30$^{\circ}$N','M':'30$^{\circ}$S-30$^{\circ}$N',
        'S':'below 30$^{\circ}$S','SH':'southern hemisphere',
        'NH':'northern hemisphere','S60':'below 60$^{\circ}$S',
        'N60':'above 60$^{\circ}$N'}
tits2 = {'TEMP':'T','N':'N',
        'PD':'$\rho$','SALT':'S','V':'$v$'}
configs=['2pic','4pic']

dirr = '/Volumes/HD/Eocene/output/spinup_temp_salt/'

time= np.arange((years+7)*12) / 12
time2 = np.arange((years2+7)*12)/12

ncHR2 = Dataset(dirr + 'hov_38Ma_2pic_full.nc')
ncLR2 = Dataset(dirr + 'initial_hormeans_38Ma_2pic.nc')

ncHR4 = Dataset(dirr + 'hov_38Ma_4pic_full.nc')
ncLR4 = Dataset(dirr + 'initial_hormeans_38Ma_4pic.nc')

if(var=='N'):
    depHR = (ncHR2['w_dep'][2:]+ncHR2['w_dep'][1:-1])/2 / 1000
    depLR = (ncLR2['w_dep'][2:]+ncLR2['w_dep'][1:-1])/2 / 1000
else:
    depHR = (ncHR2['w_dep'][1:]+ncHR2['w_dep'][:-1])/2 / 1000
    depLR = (ncLR2['w_dep'][1:]+ncLR2['w_dep'][:-1])/2 / 1000

deps = np.arange(0.00005,5.5,0.01)
HR2 = np.zeros((len(time), len(deps)))
LR2 = np.zeros(len(deps))
HR4 = np.zeros((len(time2), len(deps)))
LR4 = np.zeros(len(deps))

for c in coms:
    if(var=='N'):
        LR2 = 1e3*np.interp(deps, depLR[:-1], ncLR2[var+c][:-1])
    else:
        LR2 = np.interp(deps, depLR[:-1], ncLR2[var+c][:-1])
    fno = 0
    for t in range(len(time)):
        if((t>12*years and fno==0) or (fno>0 and (t-12*years)%(12)==0)):
            ncHR2 = Dataset(dirr + 'hov_38Ma_y%d_2pic_full_daily.nc'%(fno))
            fno += 1
        if(fno==0):
            if(var=='SALT'):
                HR2[t] = 1e3*np.interp(deps, depHR[:-1], ncHR2[var+c][t,:-1])
            else:
                HR2[t] = np.interp(deps, depHR[:-1], ncHR2[var+c][t,:-1])
        else:
            ti = t-12*years - (fno-1)*12
            if(var=='SALT'):
                HR2[t] = 1e3*np.interp(deps, depHR[:-1], np.nanmean(ncHR2[var+c][ti:ti+12,:-1], axis=0))
            else:
                HR2[t] = np.interp(deps, depHR[:-1], np.nanmean(ncHR2[var+c][ti:ti+12,:-1], axis=0))


for c in coms:
    if(var=='N'):
        LR4 = 1e3*np.interp(deps, depLR[:-1], ncLR4[var+c][:-1])
    else:
        LR4 = np.interp(deps, depLR[:-1], ncLR4[var+c][:-1])
    fno = 0
    for t in range(len(time2)):
        if((t>=12*years2 and fno==0) or (fno>0 and (t-12*years2)%(12)==0)):
            ncHR4 = Dataset(dirr + 'hov_38Ma_y%d_4pic_full_daily.nc'%(fno))
            fno += 1
        if(fno==0):
            if(var=='SALT'):
                HR4[t] = 1e3*np.interp(deps, depHR[:-1], ncHR4[var+c][t,:-1])
            else:
                HR4[t] = np.interp(deps, depHR[:-1], ncHR4[var+c][t,:-1])
        else:
            ti = t-12*years2 - (fno-1)*12
            if(var=='SALT'):
                HR4[t] = 1e3*np.interp(deps, depHR[:-1], np.nanmean(ncHR4[var+c][ti:ti+12,:-1], axis=0))
            else:
                HR4[t] = np.interp(deps, depHR[:-1], np.nanmean(ncHR4[var+c][ti:ti+12,:-1], axis=0))
HR4[6] = HR4[4]

deps2, time2 = np.meshgrid(deps,time2)
deps, time = np.meshgrid(deps,time)
#%%

np.savez('Hov_transport_refinit_toplot_com%s_full.npz'%(coms[0]),
         time=time, time2=time2, deps=deps,deps2=deps2,
         depHR=depHR, depLR=depLR,HR4=HR4, LR4=LR4,
         plotnames=plotnames, ylims=ylims, HR2=HR2, LR2=LR2,

         timet=timet, transports=transports,
         timet4=timet4, transports4=transports4,
         yeartimet=yeartimet, yearlytransports=yearlytransports,
         yeartimet4=yeartimet4, yearlytransports4=yearlytransports4,
         )