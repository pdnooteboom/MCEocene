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
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import math
from numba import jit
import cartopy.mpl.ticker as cticker
from matplotlib.lines import Line2D
import cmocean.cm as cm
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
from copy import copy
import matplotlib.colors as mcolors
#%%
def z_masked_overlap(axe, X, Y, Z, source_projection=None):
    """
    for data in projection axe.projection
    find and mask the overlaps (more 1/2 the axe.projection range)

    X, Y either the coordinates in axe.projection or longitudes latitudes
    Z the data
    operation one of 'pcorlor', 'pcolormesh', 'countour', 'countourf'

    if source_projection is a geodetic CRS data is in geodetic coordinates
    and should first be projected in axe.projection

    X, Y are 2D same dimension as Z for contour and contourf
    same dimension as Z or with an extra row and column for pcolor
    and pcolormesh

    return ptx, pty, Z
    """
    if not hasattr(axe, 'projection'):
        return Z
    if not isinstance(axe.projection, ccrs.Projection):
        return Z

    if len(X.shape) != 2 or len(Y.shape) != 2:
        return Z

    if (source_projection is not None and
            isinstance(source_projection, ccrs.Geodetic)):
        transformed_pts = axe.projection.transform_points(
            source_projection, X, Y)
        ptx, pty = transformed_pts[..., 0], transformed_pts[..., 1]
    else:
        ptx, pty = X, Y


    with np.errstate(invalid='ignore'):
        # diagonals have one less row and one less columns
        diagonal0_lengths = np.hypot(
            ptx[1:, 1:] - ptx[:-1, :-1],
            pty[1:, 1:] - pty[:-1, :-1]
        )
        diagonal1_lengths = np.hypot(
            ptx[1:, :-1] - ptx[:-1, 1:],
            pty[1:, :-1] - pty[:-1, 1:]
        )
        to_mask = (
            (diagonal0_lengths > (
                abs(axe.projection.x_limits[1]
                    - axe.projection.x_limits[0])) / 2) |
            np.isnan(diagonal0_lengths) |
            (diagonal1_lengths > (
                abs(axe.projection.x_limits[1]
                    - axe.projection.x_limits[0])) / 2) |
            np.isnan(diagonal1_lengths)
        )

        # TODO check if we need to do something about surrounding vertices

        # add one extra colum and row for contour and contourf
        if (to_mask.shape[0] == Z.shape[0] - 1 and
                to_mask.shape[1] == Z.shape[1] - 1):
            to_mask_extended = np.zeros(Z.shape, dtype=bool)
            to_mask_extended[:-1, :-1] = to_mask
            to_mask_extended[-1, :] = to_mask_extended[-2, :]
            to_mask_extended[:, -1] = to_mask_extended[:, -2]
            to_mask = to_mask_extended
        if np.any(to_mask):

            Z_mask = getattr(Z, 'mask', None)
            to_mask = to_mask if Z_mask is None else to_mask | Z_mask

            Z = np.ma.masked_where(to_mask, Z)

        return ptx, pty, Z


#%%

sns.set(context='paper',style="whitegrid",font="Arial")
fs = 17
font = {'size'   : fs}
matplotlib.rc('font', **font)

#variables
typ = 'TGclosed'#'TG1-2km'#'DP100'#'TG'#'DP100'#'TG1-2km'#'TGclosed'#''#'DPshallow'#'TGclosed'#
projection = ccrs.PlateCarree(180)
exte = [200, 250, -75, -50]
cmap = 'viridis'#cm.deep

rr = 1 # determines the resolution reduction for the plotting
#%% Load files
contours = [-6e-5,-5e-5,-4e-5,-3e-5,-2e-5,-1e-5,0,1e-5, 2E-5, 3E-5, 5E-5]

#Drake : 
if(typ in ['DP2km','DP100','DP']):
    minlat = -78
    maxlat = -40
    minlat38 = -78
    minlon=270
    maxlon=330
    vsbath = [0,6]
    if(typ=='DP'):
        colors1 = cm.matter(np.linspace(0, 1, 17))
        colors2 = cm.deep(np.linspace(0, 1, 83))
           # colors3 = plt.cm.cool_r(np.linspace(0, 1, 55))[]
        colors = np.vstack((colors1, colors2))
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)#'Spectral_r'#
    elif(typ=='DP2km'):
        colors1 = cm.matter(np.linspace(0, 1, 33))
        colors2 = cm.deep(np.linspace(0, 1, 67))
           # colors3 = plt.cm.cool_r(np.linspace(0, 1, 55))[]
        colors = np.vstack((colors1, colors2))
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)#'Spectral_r'#
    elif(typ=='DP100'):
        colors1 = cm.matter(np.linspace(0, 1, 3))
        colors2 = cm.deep(np.linspace(0, 1, 97))
           # colors3 = plt.cm.cool_r(np.linspace(0, 1, 55))[]
        colors = np.vstack((colors1, colors2))
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)#'Spectral_r'#
else:
# Tasman : 
    minlat = -70
    maxlat = -30
    minlat38 = -70
    minlon=100
    maxlon=170
    vsbath = [0,6]
    if(typ=='TG'):
        colors1 = cm.matter(np.linspace(0, 1, 11))
        colors2 = cm.deep(np.linspace(0, 1, 89))
           # colors3 = plt.cm.cool_r(np.linspace(0, 1, 55))[]
        colors = np.vstack((colors1, colors2))
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)#'Spectral_r'#
    elif(typ=='TGclosed'):
        colors1 = cm.matter(np.linspace(0, 1, 11))
        colors2 = cm.deep(np.linspace(0, 1, 89))
           # colors3 = plt.cm.cool_r(np.linspace(0, 1, 55))[]
        colors = np.vstack((colors1, colors2))
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)#'Spectral_r'#
    elif(typ=='TG1-2km'):
        colors1 = cm.matter(np.linspace(0, 1, 21))
        colors2 = cm.deep(np.linspace(0, 1, 79))
           # colors3 = plt.cm.cool_r(np.linspace(0, 1, 55))[]
        colors = np.vstack((colors1, colors2))
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)#'Spectral_r'#


dirRead = '/Users/nooteboom/Documents/PhD/Eocene_POP/OUPUT/gridf/'
# Load 38MA
if(typ in ['DP','TG']):
    nc38 = Dataset('kmt_tx0.1_POP_EO38.nc')  
elif(typ=='DP2km'):
    nc38 = Dataset('kmt_tx0.1_POP_EO38_DPdeep.nc')  
elif(typ=='DP100'):
    nc38 = Dataset('kmt_tx0.1_POP_EO38_DP100.nc')
elif(typ=='TG1-2km'):
    nc38 = Dataset('kmt_tx0.1_POP_EO38_TGdeep.nc')
elif(typ=='TGclosed'):
    nc38 = Dataset('kmt_tx0.1_POP_EO38_TGclosed.nc')
else:
    assert False, 'change \'typ\''

#nc38grid = Dataset(dirRead + 'grid_coordinates_pop_tx0.1_38ma.nc')
lats38 = nc38['U_LAT_2D'][:]
lons38 = nc38['U_LON_2D'][:] + 25
lons38[lons38>180] -= 360
#minlon = np.min(lons38)
#maxlon = np.max(lons38)
exte38 = [minlon, maxlon, minlat38, maxlat]
idx = np.where(np.logical_and(lats38[:,0]>=minlat38, lats38[:,0]<=maxlat))

bath38 = (nc38['bathymetry'][:] / 1000)[idx[0]]
fH38 = (-2 * 7.2921 * 10**(-5) * np.sin(lats38[idx[0]]*np.pi/180) / bath38)
lats38 = lats38[idx[0]]
lons38 = lons38[idx[0]]
#%% start figure
latticks = [-100,-75,-50,-25, 0, 25, 50, 75, 100]#[-75,-65,-55,-45,-35,-25, 0, 25, 50, 75, 100]
fig = plt.figure(figsize=(8,6))
#% subplot (a)
#%% subplot (a)
ax2 = plt.subplot(111, projection=projection)
plt.title('38Ma bathymetry, %s'%(typ), fontsize=fs)

g = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
g.xlocator = mticker.FixedLocator([-180,-90, -0, 90, 180])
g.xlabels_top = False
g.ylabels_right = False
g.xlabels_bottom = False
g.xlabels_left = True
g.xlabel_style = {'fontsize': fs}
g.ylabel_style = {'fontsize': fs}
g.xformatter = LONGITUDE_FORMATTER
g.yformatter = LATITUDE_FORMATTER
g.ylocator = mticker.FixedLocator(latticks)
ax2.set_extent(exte38, ccrs.PlateCarree())
#ax2.set_xticks([0., 90., 180., 270.], crs=ccrs.PlateCarree())
#ax2.set_xticklabels([0., 90., 180., 270.], fontsize=fs)

im2 = plt.pcolormesh(lons38[::rr,::rr], lats38[::rr,::rr], bath38[::rr,::rr], cmap=cmap, 
                     vmin=vsbath[0], vmax=vsbath[1],
                     transform=ccrs.PlateCarree()
                     )

X, Y, masked_MDT = z_masked_overlap(ax2, lons38[::rr,::rr], lats38[::rr,::rr], 
                                    fH38[::rr,::rr],
                                    source_projection=ccrs.Geodetic())

plt.contour(X-180, Y, masked_MDT, contours, 
            color='k', vmin=vsbath[0], vmax=vsbath[1],
                     transform=ccrs.PlateCarree(),zorder=1
                     )

land = np.full(bath38.shape, np.nan); land[bath38<=0] = 1;
plt.pcolormesh(lons38[::rr,::rr], lats38[::rr,::rr], land[::rr,::rr],  
               vmin=0, vmax=1.6,cmap='binary',
                     transform=ccrs.PlateCarree(),zorder=2
                     )

#%% final
fig.subplots_adjust(right=0.95)
cbar_ax = fig.add_axes([0.95, 0.1, 0.1, 0.8])
cbar_ax.set_visible(False)
cbar = fig.colorbar(im2, ax=cbar_ax, orientation = 'vertical', fraction = 0.8,
                    aspect=18)
cbar.ax.xaxis.set_label_position('bottom')

#fig.subplots_adjust(bottom=0.17)
#cbar_ax = fig.add_axes([0.135, 0., 0.8, 0.07])
#cbar_ax.set_visible(False)
#cbar = fig.colorbar(im2, ax=cbar_ax, orientation = 'horizontal', fraction = 1.2,
#                    aspect=18)
#cbar.ax.xaxis.set_label_position('bottom')
cbar.ax.set_xlabel('km', fontsize=fs)
cbar.ax.tick_params(labelsize=fs)


plt.savefig('FH_bathymetry_%s.png'%(typ), dpi=300,bbox_inches='tight',pad_inches=0)
plt.show()