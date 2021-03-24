#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:38:57 2021

@author: nooteboom
"""


import numpy as np
import matplotlib.pylab as plt
from netCDF4 import Dataset
import cartopy.crs as ccrs
import seaborn as sns
from matplotlib.lines import Line2D
import cartopy.feature as cfeature
from copy import copy
import matplotlib.path as mpath
from matplotlib import gridspec
from scipy.interpolate import griddata
import pandas as pd
from matplotlib.lines import Line2D

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

def create_locidx(LATS,LONS,minlat,maxlat,minlon,maxlon):
    bolat = np.logical_and(LATS[:,0]>=minlat-5,LATS[:,0]<=maxlat+1)
    if(minlon<180 and maxlon>180):
        bolon = np.logical_and(LONS[50,:]<=minlon,
                               LONS[50,:]<=maxlon+5) 
    else:
        bolon = np.logical_and(LONS[50,:]>=minlon,LONS[50,:]<=maxlon+5)    
    return np.where(bolat), np.where(bolon)

def add_min90_bath(bath, lon, lat, val=-90):
    bath = np.concatenate((np.ones((1,bath.shape[1])), bath), axis=0)
    lat = np.concatenate((np.full((1,bath.shape[1]), val), lat), axis=0)
    lon = np.concatenate((lon[0][np.newaxis,:], lon), axis=0)

    return bath, lon, lat

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

def subplot(ax, lons38, lats38, land,  blons, blats, Rloc, lonsPD, latsPD,
            circle=circle, fs=25, exte38=[], sc=300,
            projection=ccrs.PlateCarree(), cmap='viridis', sites={}, title='',
            th = 16):

    ax.set_boundary(circle, transform=ax.transAxes)
    ax.add_feature(cfeature.OCEAN, zorder=0, color='aliceblue')
    ax.add_feature(cfeature.LAND, zorder=0, color='aliceblue')
    ax.set_title(title, fontsize=fs)
    
    ax.set_extent(exte38, ccrs.PlateCarree())
    
    X, Y, masked_MDT = z_masked_overlap(
        ax, lons38, lats38, land,
        source_projection=ccrs.Geodetic())
    
    ax.contourf(X, Y, masked_MDT, #transform=projection, 
                   vmin=0, vmax=1.2,cmap='gist_earth',zorder=2
                         )
                   
    land2 = copy(land)
    land2[np.isnan(land2)] = 0
    X, Y, masked_MDT = z_masked_overlap(
        ax, lons38, lats38, land2,
        source_projection=ccrs.Geodetic())
    
    ax.contour(X, Y, masked_MDT, [0.5], colors='k', zorder=3, linewidth=2)
    
    ax.scatter(blons, blats, c='k', transform=projection, 
                   zorder=1)
    ax.scatter([Rloc[0]], [Rloc[1]], c='k', transform=projection, 
                   zorder=1, marker='P', s=sc)
    
    X, Y, masked_MDT = z_masked_overlap(ax, lonsPD, latsPD, 
                                    bathPD,
                                    source_projection=ccrs.Geodetic())

    np.random.seed(0)
    CS = ax.contour(X, Y, masked_MDT, [th], colors='r', linewidths=5)
    ax.clabel(CS,  inline=1, fmt='%d', fontsize=15)
    CS = ax.contour(X, Y, masked_MDT, [12,14,18,22], colors='k')
    ax.clabel(CS, inline=1, fmt='%d', fontsize=15)
    
    custom_lines = [Line2D([0], [0], color='k', linewidth=2),
                    Line2D([0], [0], color='r', linewidth=5),
                    Line2D([0], [0], markerfacecolor='k',
                           linewidth=0, marker='o',
                           markersize=15),
                    Line2D([0], [0], markerfacecolor='k',
                           linewidth=0, marker='P',
                           markersize=15)]
    
    ax.legend(custom_lines, ['SST ($^{\circ}$C)',
                             'threshold SST ($^{\circ}$C)',
                             'back-tracked origin location',
                             'release location'],
              loc='lower center',bbox_to_anchor=(0.25, -0.3, 0.5, 0.5))

def subplot2(ax, temp, fs=25, sc=150, title='', th=16):
    dics = {'$^{\circ}$C':temp,
            'endemic':np.full(len(temp), 'yes')}
    dics['endemic'][dics['$^{\circ}$C']>th] = 'no'
    
    dics = pd.DataFrame.from_dict(dics)
    print(dics)
    sns.histplot(dics,
                 x='$^{\circ}$C', hue='endemic', ax=ax,
                 binwidth=1, binrange = [13,23])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    
    ax.set_title(title, fontsize=fs)

    side = ax.spines["left"]
    side.set_visible(False)
    side = ax.spines["top"]
    side.set_visible(False)


#%% Load a field from the model. To plot the land.
minlat = -90
maxlat = -40
minlon = 1
maxlon = 359#179#
minlat38 = -90
projection = ccrs.SouthPolarStereo(25)

dirReadHR = '/Volumes/HD/Eocene/output/time_mean/'
nc38 = Dataset(dirReadHR + 'pop_38Ma_avg6years.nc')
dirRead = '/Users/nooteboom/Documents/PhD/Eocene_POP/OUPUT/gridf/'
nc38grid = Dataset(dirRead + 'grid_coordinates_pop_tx0.1_38ma.nc')
dz = nc38grid['w_dep'][1:] - nc38grid['w_dep'][:-1]
lats38 = nc38['ULAT'][:]
lons38 = nc38['ULONG'][:]+25
exte38 = [minlon, maxlon, minlat38, maxlat]
idxlat, idxlon = create_locidx(lats38,lons38,minlat,maxlat,minlon,maxlon)
bath38 = (nc38['TEMP'][0])[np.min(idxlat[0]):np.max(idxlat[0]), np.min(idxlon[0]):np.max(idxlon[0])]

lats38 = lats38[np.min(idxlat[0]):np.max(idxlat[0]), np.min(idxlon[0]):np.max(idxlon[0])]
lons38 = lons38[np.min(idxlat[0]):np.max(idxlat[0]), np.min(idxlon[0]):np.max(idxlon[0])]
land = np.full(bath38.shape, np.nan); land[bath38==0] = 1;
land, lons38, lats38 = add_min90_bath(land, lons38, lats38)

#%% Load SST field
def create_locidx(LATS,LONS,minlat,maxlat,minlon,maxlon):
    bolat = np.logical_and(LATS[:,0]>=minlat-5,LATS[:,0]<=maxlat+1)
    return np.where(bolat)
# Load PD
years = 6
# Load PD
dirReadLR = '/Volumes/HD/Eocene/LR/output/time_mean/'
#ncPD = Dataset(dirRead + 'bathymetry_POP_0.1res.nc')
ncPDgrid = Dataset(dirReadLR + 'cesm_38Ma_avg6years_nobolus.nc')

latsPD = ncPDgrid['ULAT'][:]
lonsPD = ncPDgrid['ULONG'][:]
lonsPD[lonsPD>180.1] -= 360
extePD = [minlon, maxlon, minlat, maxlat]
idxlat = create_locidx(latsPD,lonsPD,minlat,maxlat,minlon,maxlon)

bathPD = (ncPDgrid['TEMP'][0])[idxlat[0][0]:idxlat[0][-1]]

latsPD = latsPD[np.min(idxlat[0]):np.max(idxlat[0])]
lonsPD = lonsPD[np.min(idxlat[0]):np.max(idxlat[0])]
#%%
np.random.seed(0)
sigma = 15
sigma2 = 3

Rloc = [90, -50]
blons = np.random.normal(45,sigma,300)
blats = np.random.normal(-54,sigma2,300)

# interpolate the temperature field on the back-tracked locations
temp = griddata(np.concatenate((lonsPD.flatten()[:,np.newaxis],
                                latsPD.flatten()[:,np.newaxis]), 
                               axis=1),
                bathPD.flatten(), 
                (blons, blats),
                method='nearest')
#%%
sns.set(context='paper',style='white',font_scale=2)
fig = plt.figure(figsize=(16,10))
gs = gridspec.GridSpec(1, 2, figure = fig,
#                       height_ratios=(20, 20, 1),
                       width_ratios=(1, 1),
#                       hspace = 0.0, wspace = 0.03
                       )
ax = plt.subplot(gs[0,0], projection=projection)

#% subplot (a)

subplot(ax, lons38, lats38, land, blons, blats, Rloc, lonsPD, latsPD,
            circle=circle,
            exte38=exte38,  #cmap=cmap_perc,
            title='(a)')

ax = plt.subplot(gs[0,1])

#% subplot (b)

subplot2(ax, temp, title='(b)')   

if(True):
    plt.savefig('Endemism_illustration.png',bbox_inches='tight',
                pad_inches=0)#, dpi=100
plt.show()
    
