#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:41:33 2019

@author: nooteboom
"""

import numpy as np
import matplotlib.pylab as plt
from netCDF4 import Dataset
import matplotlib
import cartopy.crs as ccrs
import seaborn as sns
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from copy import copy
import merTf as mtf

#%% Define the data with SST reconstructions from Baatsen et al., 2020
def check_lens(dic):
    d0 = list(dic.keys())[0]
    for l in list(dic.keys()):
        assert len(dic[d0])==len(dic[l]), l+' '+str(len(dic[d0]))+' '+str(len(dic[l]))

# 42-38 Ma
proxydata_4238MA = {
        'T': np.array([28,31,32,
                       30,31.5,32,
                       24,10,26,
                       24,35,36.3,
                       26,12.7,13.1,
                       20.3,34,32,
                       23.2
                       ]),
        'sigma': np.array([4.7,4.7,2.5,
                           2.5,2.5,1.2, 
                           2.5,2.7,1.4,
                           0.7,2.0,1.8,
                           4.7, 2.4,2.4,
                           2.5,2.5,0.7,
                           2.6
                           ]),
        'lat': np.array([-55.5,11.8,3.5,
                         -1.8,-5.23,24.2,
                         -57.8,82.5,28.3,
                         34.1,0.1,0.1,
                         -49.3,-68.5,-68.5,
                         -68.5,
                         -16.6,-17.6,43.6
                            ]),
        'lon': np.array([178.1,-147.,-30.9,
                         -31.2, -2.59, -61.6,
                         157.9,-5.7,-74.1,
                         13.1,114.5,114.5,
                         -169.7,-62.6,-62.5,
                         -62.5,40.6,40.6,
                         -2.9
                        ]),
        'ref': np.array(['Hines et al. 2017','Tripati et al. 2003','Liu et al . 2009',
                         'Liu et al . 2009','Cramwinckel et al . 2018', 'Okafor et al.2009',
                         'Bijl et al . 2009',
                         'Evans et al. 2018','Kobashi et al. 2004','Pearson et al. 2001',
                         'Evans et al. 2018','Evans et al. 2018','Hines et al.2017',
                         'Douglas et al. 2014','Douglas et al. 2014','Douglas et al. 2014',
                         'Pearson et al.2007','Pearson et al. 2001','Evans et al. 2018']),
        'method': np.array(['Mg/Ca','Mg/Ca','TEX$_{86}^H$','TEX$_{86}^H$',
                            'TEX$_{86}^H$','Mg/Ca','TEX$_{86}^H$',
                            'UK$_{37}$','$\delta^{18}$O','$\delta^{18}$O','$\Delta_{47}$','$\Delta_{47}$',
                            'Mg/Ca','$\Delta_{47}$','$\Delta_{47}$','TEX$_{86}^H$',
                            'TEX$_{86}^H$','$\delta^{18}$O','$\Delta_{47}$'])
        
        }
    
scatlons4238 = copy(proxydata_4238MA['lon'])
scatlons4238 += (180-70)

# 38-34 Ma
proxydata_3834MA= {
        'T': np.array([26.6,25.6,20,
                       19.5,19.6,12.3,
                       22,27.4,18.3,
                       30,29,28.5,
                       22.1,30,22,
                       22,27,18.9,
                       22,30.9,32,
                       12.2,13,16,
                       32.5,34.9,30.5,
                       29.5,
                       33,29.7,30]),
        'sigma': np.array([2.5,4.7,1.5,2.5,1.5,4.6,1.5,2.5,1.5,2.5,2.5,2.5,2.5,1.2,
                           2.5,2.5,0.7,0.7,1.4,4.7,1.4,2.4,3.0,2.5,2.2,4.7,2.5,
                           0.7,2.5,3.2,0.7]),
        'lat': np.array([-55.5,-55.5,55.7,-58.8,-58.8,-68.3,-68.3,-0.8,66.2,3.5,-1.8,-5.23,
                         14.6,24.2,-51.9,-57.8,27.6,-53.8,28.3,-49.3,0.9,-68.5,
                         -68.5,-68.5,27.2,27.2,27.2,-16.6,-16.6,-16.6,-17.6]),
        'lon': np.array([178.1,178.1,-11,-30.4,-30.4,14.,14.,-169.1,-2.4,-30.9,
                         -31.2,
                         -2.59,
                         -69.4,-61.6,157.9,157.9,-72.4,147.2,-74.1,-169.7,
                         -72.9,-62.5,-62.5,-62.5,-72.2,-72.2,-72.2,40.6,40.6,
                         40.6,40.6]),
        'method': np.array(['TEX$_{86}^H$', 'UK$_{37}$','UK$_{37}$','TEX$_{86}^H$',
                            'UK$_{37}$', '$\Delta_{47}$','$\Delta_{47}$', 'TEX$_{86}^H$',
                            'UK$_{37}$','TEX$_{86}^H$','TEX$_{86}^H$','TEX$_{86}^H$','TEX$_{86}^H$',
                            'Mg/Ca', 'UK$_{37}$', 'TEX$_{86}^H$','$\delta^{18}$O',
                            '$\delta^{18}$O','$\delta^{18}$O','Mg/Ca','$\delta^{18}$O',
                            '$\Delta_{47}$','$\Delta_{47}$','TEX$_{86}^H$','Mg/Ca',
                            'Mg/Ca','TEX$_{86}^H$','$\delta^{18}$O','TEX$_{86}^H$',
                            '$\Delta_{47}$','$\delta^{18}$O']),
        
        }
scatlons3834 = copy(proxydata_3834MA['lon'])
scatlons3834 += (180-70)
check_lens(proxydata_3834MA)
check_lens(proxydata_4238MA)
#%% Set plotting parameters
sns.set(context='paper',style="ticks",font="Arial")
fs = 20
font = {'size'   : fs}
matplotlib.rc('font', **font)

## global:
minlat = -82
maxlat = 85
minlon = 1
maxlon = 359
minlat38 = minlat
xticksl = [-180,-90,0,90,200]

projection = ccrs.PlateCarree(70-180)  
#%% Load 2pic files
# Load LR
var = 'SST' 
exte = [1, 360, -75, -10]
cmap = 'Spectral_r'#
vsbath = [-3,3]

dirReadLR = '/Volumes/HD/Eocene/output/time_mean/final/'
ncPDgrid = Dataset(dirReadLR + 'cesm_38Ma_2pic_avg%dyears.nc'%(5))
latsPD = ncPDgrid['ULAT'][:]
lonsPD = ncPDgrid['ULONG'][:]
lonsPD[lonsPD<0] += 360
if((minlon<180 and maxlon>180)):
    extePD = [minlon-360, maxlon-360, minlat, maxlat]
    extePD = [minlon, maxlon, minlat, maxlat]
else:
    extePD = [minlon, maxlon, minlat, maxlat]

bathPD = (ncPDgrid['TEMP'][0])

fHPD = bathPD

# Load HR
dirReadHR = '/Volumes/HD/Eocene/output/time_mean/final/'
nc38 = Dataset(dirReadHR + 'pop_38Ma_2pic_merT_avgyear23to27.nc')
lats38 = nc38['ULAT'][:]
lons38 = nc38['ULONG'][:]+25
lons38[lons38<0] += 360
if((minlon<180 and maxlon>180)):
    exte38 = [minlon-360, maxlon-360, minlat38, maxlat]
else:
    exte38 = [minlon, maxlon, minlat38, maxlat]

bath38 = (nc38['TEMP'][0])
    
bath38[bath38==0] = np.nan;
#%% Load 4PIC
dirReadLR = '/Volumes/HD/Eocene/output/time_mean/final/'
ncPDgrid = Dataset(dirReadLR + 'cesm_38Ma_4pic_avg%dyears.nc'%(5))

latsPD4 = ncPDgrid['ULAT'][:]
lonsPD4 = ncPDgrid['ULONG'][:]
lonsPD4[lonsPD4>180.1] -= 360
lonsPD4[lonsPD4<0] += 360

bathPD4 = (ncPDgrid['TEMP'][0])

# Load HR 4PIC 
dirReadHR = '/Volumes/HD/Eocene/output/time_mean/final/'
nc38 = Dataset(dirReadHR + 'pop_38Ma_4pic_merT_avgyear23to27.nc')
lats384 = nc38['ULAT'][:]
lons384 = nc38['ULONG'][:]+25
lons384[lons384>180.01] -= 360
lons384[lons384<0] += 360

bath384 = (nc38['TEMP'][0])
bath384[bath384==0] = np.nan; 
#%% Interpolate both on the same grid and take the difference

grid_x, grid_y = np.meshgrid(np.arange(0.1,360,0.1),
                             np.arange(-87,maxlat-0.1,0.1))

points = np.concatenate((lonsPD.flatten()[:,np.newaxis], latsPD.flatten()[:,np.newaxis]), axis=1)
values = bathPD.flatten()
bathPD = griddata(points, values, (grid_x, grid_y), method='nearest')

points = np.concatenate((lons38.flatten()[:,np.newaxis], lats38.flatten()[:,np.newaxis]), axis=1)
values = bath38.flatten()
bath38 = griddata(points, values, (grid_x, grid_y), method='nearest')

zonaltempHR2 = np.nanmean(bath38, axis=1)
zonaltempLR2 = np.nanmean(bathPD, axis=1)
latszm = grid_y[:,0]

bath = bath38 - bathPD


points = np.concatenate((lonsPD4.flatten()[:,np.newaxis], latsPD4.flatten()[:,np.newaxis]), axis=1)
values = bathPD4.flatten()
bathPD4 = griddata(points, values, (grid_x, grid_y), method='nearest')

points = np.concatenate((lons384.flatten()[:,np.newaxis], lats384.flatten()[:,np.newaxis]), axis=1)
values = bath384.flatten()
bath384 = griddata(points, values, (grid_x, grid_y), method='nearest')

zonaltempHR4 = np.nanmean(bath384, axis=1)
zonaltempLR4 = np.nanmean(bathPD4, axis=1)

bath4 = bath384 - bathPD4

bath4[bath4 < -10**10] = np.nan
bath[bath < -10**10] = np.nan
#%%
#plt.pcolormesh(bath4, vmin=-3, vmax=3)
#plt.colorbar()
#plt.show()

#assert False
#%% start figure

print('start figure')
axes = []


fig = plt.figure(figsize=(15,14))


outer_grid = gridspec.GridSpec(nrows=2, ncols=1, wspace=0.05, hspace=0.14, 
                               height_ratios=[4, 1])

# the inner grid which contains the SST difference plots in space
inner_grid = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=3,
    subplot_spec=outer_grid[0,:], wspace=0., hspace=0.42,
    width_ratios=[15.6,1,9])

inner_grid2 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=4,
    subplot_spec=outer_grid[1,:], wspace=0.)


#% subplot (a)
ax0 = fig.add_subplot(inner_grid[0,0], projection=projection)
im2 = mtf.spatial_subplot(ax0,grid_x, grid_y, bath,
                          scat=scatlons3834, prd=proxydata_3834MA,
                          title = r'(a) 2$\times$pre-industrial carbon',
                          exte = extePD)

#% subplot (b)
ax = fig.add_subplot(inner_grid[1,0], projection=projection)
im2 = mtf.spatial_subplot(ax,grid_x, grid_y, bath4,config='4pic',
                          scat=scatlons4238, prd=proxydata_4238MA,
                          title =r'(d) 4$\times$pre-industrial carbon',
                          exte = extePD)

#% colorbar:
cbar_ax = fig.add_axes([0.29, 0.57, 0.19, 0.07])
cbar = fig.colorbar(im2, ax=cbar_ax, orientation = 'horizontal', fraction = 0.8,
                    aspect=18, extend='both')

cbar_ax.set_visible(False)
#    cbar.ax.xaxis.set_label_position('right')
cbar.ax.set_ylabel('$^{\circ}$C', fontsize=fs, rotation='horizontal')
cbar.ax.yaxis.set_label_coords(-0.1, -0.3)
cbar.ax.tick_params(labelsize=fs)

#legend
custm = [Line2D([0], [0], markeredgecolor='k', markerfacecolor='k', 
                lw=0, marker='x', markersize=15),
        Line2D([0], [0], markeredgecolor='k', markerfacecolor='k', 
                lw=0, marker='+', markersize=15)]
#    assert False

ax.legend(custm, ['38-34Ma proxies', '42-38Ma proxies'],
          bbox_to_anchor=(.25, 1.41), loc='upper right',prop={'size':fs-4})

custm = [Line2D([0], [0], color='k',
                lw=5),
        Line2D([0], [0], color='red',
                lw=5),

        Line2D([0], [0], markeredgecolor='cyan', markerfacecolor='cyan', 
                lw=2, marker='o', markersize=10, color='cyan'),
        Line2D([0], [0], markeredgecolor='red', markerfacecolor='red',
                lw=2, marker='^', markersize=10, color='red'),
        Line2D([0], [0], markeredgecolor='green', markerfacecolor='green', 
                lw=2, marker='v', markersize=10, color='green'),
        Line2D([0], [0], markeredgecolor='k', markerfacecolor='k',
                lw=2, marker='s', markersize=10, color='k'),
        Line2D([0], [0], markeredgecolor='grey', markerfacecolor='grey',
                lw=2, marker='X', markersize=10, color='grey')]
ax0.legend(custm, ['HR2, HR4', 'LR2, LR4', '$\Delta_{47}$','$\delta^{18}$O',
                   'Mg/Ca','TEX$^H_{86}$','UK$_{37}$'],
          bbox_to_anchor=(0.79, -0.02), loc='upper left',prop={'size':fs-4},
          ncol=4)
#%
# The meridional temperature gradient part

#%
file = '/Volumes/HD/Eocene/output/time_mean/final/cesm_38Ma_2pic_avg%dyears.nc'%(5)
file2 = '/Volumes/HD/Eocene/output/time_mean/final/pop_38Ma_2pic_merT_avgyear23to27.nc'
file4 = '/Volumes/HD/Eocene/output/time_mean/final/cesm_38Ma_4pic_avg%dyears.nc'%(5)
file24 = '/Volumes/HD/Eocene/output/time_mean/final/pop_38Ma_4pic_merT_avgyear23to27.nc'
#%

panell = {'2pic':['(b)', '(c) 38-34Ma proxy data','(g) HR2','(h) LR2'],
          '4pic':['(e)', '(f) 42-38Ma proxy data','(i) HR4','(j) LR4']}

# Define the inner grids

ax12 = fig.add_subplot(inner_grid[0, 2])
ax122 = fig.add_subplot(inner_grid[0, 1])
ax22 = fig.add_subplot(inner_grid2[0, 0])
ax32 = fig.add_subplot(inner_grid2[0, 1])

ax14 = fig.add_subplot(inner_grid[1, 2])
ax142 = fig.add_subplot(inner_grid[1, 1])
ax24 = fig.add_subplot(inner_grid2[0, 2])
ax34 = fig.add_subplot(inner_grid2[0, 3])

zm = np.nanmean(bath.data, axis=1)[np.logical_and(grid_y[:,0]<extePD[3],
                                                   grid_y[:,0]>extePD[2])]
ax122.pcolormesh(np.concatenate((zm[:,np.newaxis],zm[:,np.newaxis]), axis=1),
             vmin=-3, vmax=3, cmap='Spectral_r')
ax122.set_title('(b)', fontsize=fs)
ax122.set_yticks([])
ax122.set_xticks([])

zm = np.nanmean(bath4.data, axis=1)[np.logical_and(grid_y[:,0]<extePD[3],
                                                   grid_y[:,0]>extePD[2])]
ax142.pcolormesh(np.concatenate((zm[:,np.newaxis],zm[:,np.newaxis]), axis=1),
             vmin=-3, vmax=3, cmap='Spectral_r')
ax142.set_title('(e)', fontsize=fs)
ax142.set_yticks([])
ax142.set_xticks([])
        

#%% plot both the 2pic and 4pic
mtf.plot_merT(file, file2, ax12, ax22, ax32, ax122,
              proxydata_3834MA, panell=panell)
mtf.plot_merT(file4, file24, ax14, ax24, ax34,  ax142,
              proxydata_4238MA, 
              panell=panell, config='4pic')
        
if(True):
    plt.savefig('SSTdiff_merT.png', 
                dpi=300,bbox_inches='tight',pad_inches=0)

plt.show()