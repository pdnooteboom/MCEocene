#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:26:33 2021

@author: nooteboom
"""

import numpy as np
import matplotlib.pylab as plt
from netCDF4 import Dataset
import seaborn as sns
from copy import copy
import matplotlib
import functions as fu
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sp = 25
config = '2pic'
sns.set(context='paper',style="white",font="Arial",font_scale=2)
fs = 20
font = {'size'   : fs}
matplotlib.rc('font', **font)

if(sp==6):
    if(config=='2pic'):
        tbL = [12,20, 200]
        tbH = [12,20, 200]
    else:
        tbL = [15,22, 200]
        tbH = [15,22, 200]
elif(sp==25):
    if(config=='2pic'):
        tbL = [12,20, 200]
        tbH = [12,20, 200]
    else:
        tbL = [15,22, 200]
        tbH = [15,22, 200]
assert tbL==tbH
#%%Load the data consisting of sediment samples with dinocyst counts

def check_len(dirr):
    for di, d in enumerate(dirr.keys()):
        if(di==0):
            d0 = copy(d)
        assert len(dirr[d]) == len(dirr[d0]), d
        
# 36-38 Ma
sites2 = {
        'names': np.array(['CIROS', 'DSDP269', 'DSDP274' ,'DSDP277', 'DSDP280' ,
                           'DSDP281' ,'DSDP282',
                           'DSDP283' ,'DSDP511', 'ODP696', 'ODP739' ,'ODP742',
                           'ODP748', 'ODP1090',
                           'ODP1168' ,'ODP1170' ,'ODP1171' ,'ODP1172', 'CB',
                           'BC', 'GB' ,'MB', 'RT' ,'TK',
                           'W&H' ,'ODP1128', 'BR']),
        'plotname': np.array([0,0,0,
                              1,0,
                              0,0,
                              0,0,0,
                              0,0,
                              0,1,
                              0,0,
                              0,0,
                              0,0,0,
                              0,0,1,0,0,0]), 
    
            'plon': np.array([ 157.96467903 , 141.05834928 , 167.79178807 ,-177.78537058 , 156.74265209,
                              156.5904805 ,  148.31607306 , 163.08386265  , -42.38969549  ,-55.05969297,
                              87.01094942 ,  87.25209991  , 87.03443785 ,   8.70770576 , 149.68399384,
                              154.33680136 , 159.21331713,  157.2141268 ,  -41.57215799 , 139.987953,
                              138.31634897 , 142.82926612 , -61.47378574, -177.3460362 ,  -61.84823732,
                              125.61555606,  -60.96896054]),
            'plat': np.array([-71.73353542 ,-55.33788518 ,-63.69916578 ,-55.12936021 ,-62.71487538,
                              -61.72413677 ,-57.16092938 ,-56.4916745 ,-55.01726316 ,-66.67557915,
                              -64.18013212 ,-64.32163566, -55.12720401 ,-51.55493766, -57.22321557,
                              -61.49855082 ,-62.08055248 ,-57.45599826, -46.92805521 ,-50.35622297,
                              -54.25238514 ,-49.47830819, -59.61608928, -45.4592145 , -68.41190526,
                              -51.09432913, -60.7765138 ]),
        'endemic (%)': np.array([90,5,90,0,90,90,10,80,50,90,
                 80,90,18,0,10,90,76,93,5,0,90,0,35,10,80,0,90])
        }
# 41-39 Ma
sites4 = {
        'names': np.array(['DSDP277','DSDP280' ,'DSDP512', 'ODP696', 'ODP748' ,
                           'ODP1090', 'ODP1170',
                           'ODP1171', 'ODP1172' ,'PdE' ,'CB' ,'DF','Dj277' ,
                           'GB' ,'MG' ,'MH' ,'MS' ,'RT',
                           'SanB', 'SB1', 'SB2', 'AGZO1' ,'AGZO2' ,'AGZO3',
                           'W&H' ,'BR',#, 'ODP1182'
                           'Latrobe-1']),
        'plotname': np.array([1,0,0, # if 1, this name is labeled in the plot
                              0,0,0,
                              0,0,0,
                              0,0,0,
                              0,0,0,
                              1,0,0,
                              1,0,0,
                              0,0,0,
                              0,0,0]),#0,
        'plon': np.array([-177.78537058,  156.74265209,  -35.80843518,
                          -55.05969297,   87.30550845,
                          8.70770576,  154.33680136,  159.21331713 ,
                          157.2141268,   -40.82648255,
                          -41.57215799,  -57.31080745,  -64.5589034,
                          138.31634897, -174.73948922,-169.68959959,
                          159.01235171, -61.47378574 , -30.0109409,
                          148.83183226, 150.58774204,150.22815754, 150.7558011,
                          151.9906121, -61.84823732,
                           -60.96896054, 146.90316034]),#125.61555606,
    
        'plat': np.array([-55.12936021, -62.71487538 ,-53.46408118, -66.67557915, -56.11542268,
                          -51.55493766, -61.49855082, -62.08055248, -57.45599826, -44.69962342,
                          -46.92805521, -62.06077594, -68.42448082, -54.25238514, -50.14353342,
                          -49.13014607, -72.82133151, -59.61608928, -42.68988538 ,-55.41232688,
                          -56.81432014, -58.93218079, -59.81388716, -61.15662017, -68.41190526,
                           -60.7765138,  -53.79297381]), #-51.09432913,
    
        'endemic (%)': np.array([0,95,60,95,5,
                 0,90,93,80,35,
                 30,75,90,90,5,
                 15,95,35,10,0,
                 0,60,60,60,95,
                  40, 0])#np.nan,
        }
    
check_len(sites2)
sites2['plon'][sites2['plon']<0] += 360
check_len(sites4)
sites4['plon'][sites4['plon']<0] += 360
        
#%%
sns.set(context='paper',style="darkgrid",font="Arial",font_scale=3)

fig = plt.figure(figsize=(14,17))
widths = [2, 3.5, 1]
heigths = [0.2, 1, 0.1, 0.1, 1, 0.2, 0.05, 0.05, 0.05]
spec = fig.add_gridspec(ncols=3, nrows=9, width_ratios=widths,
                        height_ratios=heigths, hspace=0.05, wspace=-0.15)

#% Create figure with different threshold temperatures
cmap = 'Spectral_r'
Tths = np.zeros(4)

ax14 = fig.add_subplot(spec[4, 0], label='12')
ax13 = fig.add_subplot(spec[1, 0], label='13')
if(config=='2pic'):
    temps, FLR, Tths[2], Fl, esL, vLonstopL, vLatstopL, percL, iminL = fu.loopSSThat(sites2, config=config,
                    res='LR', tb=tbL, sp=sp)
    temps, FHR, Tths[3], Fh, esH, vLonstopH, vLatstopH, percH, iminH = fu.loopSSThat(sites2,
                    config=config, tb=tbH, sp=sp)

    if(True):
        fu.subplot(ax13, temps, sites2,
                     Fh, vLonstopH, vLatstopH, percH, FHR,
                     title='(a)', config='2pic', res='HR', cmap=cmap, fs=fs)
        fu.subplot(ax14, temps, sites2,
                     Fl, vLonstopL, vLatstopL, percL, FLR, 
                     title='(c)', config='2pic', res='LR', cmap=cmap, fs=fs)
if(config=='4pic'):
    temps, FLR, Tths[2], Fl, esL, vLonstopL, vLatstopL, percL, iminL = fu.loopSSThat(sites4,config=config,
                    res='LR', tb=tbL, sp=sp)
    temps, FHR, Tths[3], Fh, esH, vLonstopH, vLatstopH, percH, iminH = fu.loopSSThat(sites4,
                config=config, tb=tbH, sp=sp)
    if(True):
        fu.subplot(ax13, temps, sites4, 
                     Fh, vLonstopH, vLatstopH, percH, FHR,
                     title='(a)',config='4pic', res='HR', cmap=cmap, fs=fs)
        fu.subplot(ax14, temps, sites4,
                     Fl, vLonstopL, vLatstopL, percL, FLR,
                     title='(c)', config='4pic', res='LR', cmap=cmap, fs=fs)

axcb = fig.add_subplot(spec[6, 0], label='21')

norm = matplotlib.colors.Normalize(vmin=tbH[0], vmax=tbH[1])
cmap = matplotlib.cm.Spectral_r
cb1 = matplotlib.colorbar.ColorbarBase(axcb, cmap=cmap,
                                norm=norm,
                                orientation='horizontal', extend='both')
cb1.set_label(r"$\widehat{\rm SST}$ ($^{\circ}$C)", fontsize=fs)
if(config=='2pic'):
    cb1.set_ticks([12,14,16, 18,20])
else:
    cb1.set_ticks([12,14,16, 18,20,22])
#%% And the SO plot
minlat = -90
maxlat = -33
minlon = 1
maxlon = 359#179#
minlat38 = -90
lonticks = [125, 150,175,200]
latticks = [-80, -60,-40, -20]
sns.set(context='paper',style="whitegrid",font="Arial")
font = {'size'   : fs}
matplotlib.rc('font', **font)
projection = ccrs.SouthPolarStereo(25)

#%%
def plot_color_gradients(cmap):
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    # Create figure and adjust figure height to number of colormaps
    nrows = 1
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)

    axs[0].imshow(gradient, aspect='auto', cmap=cmap)


    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()
        
#co = plt.cm.hsv(np.linspace(0, 1, 200))[30:90]
#co2 = plt.cm.hsv(np.linspace(0, 1, 100))[:2]
co = plt.cm.coolwarm(np.linspace(0, 1, 200))[140:-10][::-1]
co2 = plt.cm.YlGnBu(np.linspace(0, 1, 100))[-2:][::-1]

co2 = np.vstack((co2,co))
co2 = mcolors.LinearSegmentedColormap.from_list('my_colormap',co2)
cmap_perc = co2
#plot_color_gradients(co2)
#%% Load a field from the model. To plot the land.

dirReadHR = '/Volumes/HD/Eocene/output/time_mean/'
nc38 = Dataset(dirReadHR + 'pop_38Ma_avg6years.nc')
dirRead = '/Users/nooteboom/Documents/PhD/Eocene_POP/OUPUT/gridf/'
nc38grid = Dataset(dirRead + 'grid_coordinates_pop_tx0.1_38ma.nc')
dz = nc38grid['w_dep'][1:] - nc38grid['w_dep'][:-1]
lats38 = nc38['ULAT'][:]
lons38 = nc38['ULONG'][:]+25
exte38 = [minlon, maxlon, minlat38, maxlat]
idxlat, idxlon = fu.create_locidx(lats38,lons38,minlat,maxlat,minlon,maxlon)
bath38 = (nc38['TEMP'][0])[np.min(idxlat[0]):np.max(idxlat[0]), np.min(idxlon[0]):np.max(idxlon[0])]

lats38 = lats38[np.min(idxlat[0]):np.max(idxlat[0]), np.min(idxlon[0]):np.max(idxlon[0])]
lons38 = lons38[np.min(idxlat[0]):np.max(idxlat[0]), np.min(idxlon[0]):np.max(idxlon[0])]
land = np.full(bath38.shape, np.nan); land[bath38==0] = 1;
land, lons38, lats38 = fu.add_min90_bath(land, lons38, lats38)


#%%

ax21 = fig.add_subplot(spec[:3, 1:], label='12', projection=projection)
ax22 = fig.add_subplot(spec[3:6, 1:], label='13', projection=projection)
axcb23 = fig.add_subplot(spec[7, 1], label='21')
#axcb33 = fig.add_subplot(spec[8, 1], label='22')
fraction=0.01; pad=0.01;

if(config=='2pic'):
    sites = sites2
else:
    sites = sites4

p, p0 = fu.subplotPT_ocm(ax22, lons38, lats38, land, vLonstopL, vLatstopL,
                     percL[iminL],
                     fs=fs, 
                     exte38=exte38,  cmap=cmap_perc, sites=sites,
                     title='(d) LR'+config[0])

p, p0 = fu.subplotPT_ocm(ax21, lons38, lats38, land, vLonstopH, vLatstopH,
                      percH[iminH],
                      fs=fs, 
                      exte38=exte38,  cmap=cmap_perc, sites=sites,
                      title='(b) HR'+config[0])

axins = inset_axes(axcb23,
                width="80%",  
                height="100%")
axcb23.axis("off")
cbar = plt.colorbar(p0,fraction=fraction, pad=pad, 
                    orientation="horizontal", cax=axins)
cbar.ax.tick_params(labelsize=fs)
cbar.ax.set_title('endemism (%)', fontsize=fs)
cbar.set_ticks([0, 20, 40, 60, 80, 100])
cbar.set_ticklabels(['', '', '', '', '', ''])


# axins = inset_axes(axcb33,
#                 width="80%",
#                 height="100%")
# axcb33.axis("off")
cbar = plt.colorbar(p,fraction=fraction, pad=pad, 
                    orientation="horizontal", cax=axins)
cbar.ax.tick_params(labelsize=fs)
#cbar.ax.set_xlabel('endemism (model; %)', fontsize=fs)
    
#%%

if(True):
    plt.savefig('figure1_%s_%d.png'%(config, sp),bbox_inches='tight',
                pad_inches=0)#, dpi=100
plt.show()
