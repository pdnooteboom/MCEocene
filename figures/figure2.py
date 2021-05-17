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
import cmocean.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
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

def create_locidx(LATS,LONS,minlat,maxlat,minlon,maxlon):
    bolat = np.logical_and(LATS[:,0]>=minlat-5,LATS[:,0]<=maxlat+1)
    if(minlon<180 and maxlon>180):
        bolon = np.logical_and(LONS[50,:]<=minlon,
                               LONS[50,:]<=maxlon+5) 
    else:
        bolon = np.logical_and(LONS[50,:]>=minlon,LONS[50,:]<=maxlon+5)    
    return np.where(bolat), np.where(bolon)

def around_zerolon(lons, lats, field):
    idn = np.where(lons[50]<0)
    idn2 = np.where(lons[50]>=0)
    lats = np.concatenate((lats[:,idn2[0]], lats[:,idn[0]]), axis=1)
    field = np.concatenate((field[:,idn2[0]], field[:,idn[0]]), axis=1)
    return lons, lats, field    

def one_subplot(field, lons, lats, ax, exte=[], cmap='jet', vmin=0, vmax=1,
                lonticks=[], latticks=[], title='()', region='Global',
                fs = 30):
    ax.set_title(title, fontsize=fs)
    
    ax.set_extent(exte, ccrs.PlateCarree())    
    im2 = plt.pcolormesh(lons, lats, field, cmap=cmap, 
                         vmin=vmin, vmax=vmax,
                         transform=ccrs.PlateCarree())
    
    land = np.full(field.shape, np.nan); 
    land[field>10**20] = 1;land[field==0] = 1;
    plt.pcolormesh(lons, lats, land,  
                   vmin=0, vmax=1.6,cmap='binary',
                         transform=ccrs.PlateCarree(), zorder=2
                         )
    return im2   
#%%
sns.set(context='paper',style="whitegrid",font="Arial")
fs = 24
font = {'size'   : fs}
matplotlib.rc('font', **font)

#variables
config = '2pic'
exte = [1, 360, -75, -10]
cmap = cm.speed
vsbath = [0,0.4]

contours = [0]
lonticks = [-90,0,90,180, 270]
latticks = [-90,-75,-25,25,75, 90]

minlat = -83
maxlat = 88
minlon = 0.1
maxlon = 359.9
minlat38 = -83

projection = ccrs.EqualEarth(180)
#%% Load files
# Load PD
dirReadLR = '/Volumes/HD/Eocene/output/time_mean/final/'
ncPD = Dataset(dirReadLR + 'cesm_merT_avg10years.nc')
dz = ncPD['Zbot'][:] - ncPD['Ztop'][:]
if(config=='2pic'):
    ncPDgrid = Dataset(dirReadLR + 'cesm_38Ma_2pic_avg5years.nc')
else:
    ncPDgrid = Dataset(dirReadLR + 'cesm_38Ma_4pic_avg50years.nc')
latsPD = ncPDgrid['ULAT'][:]
lonsPD = ncPDgrid['ULONG'][:]
#lonsPD[lonsPD>180] -= 360
lonsPD[lonsPD<0] += 360
extePD = [minlon, maxlon, minlat, maxlat]

U = (ncPDgrid['UVEL'][0])
V = (ncPDgrid['VVEL'][0])
bathPD = 0.01*np.sqrt(U.data**2+V.data**2)

fHPD = bathPD

# Load 38MA
dirReadHR = '/Volumes/HD/Eocene/output/time_mean/final/'
if(config=='2pic'):
    nc38 = Dataset(dirReadHR + 'pop_2pic_38Ma_avgyear23to27.nc')
else:
    nc38 = Dataset(dirReadHR + 'pop_4pic_38Ma_avgyear23to27.nc')
dirRead = '/Users/nooteboom/Documents/PhD/Eocene_POP/OUPUT/gridf/'
nc38grid = Dataset(dirRead + 'grid_coordinates_pop_tx0.1_38ma.nc')
dz = nc38grid['w_dep'][1:] - nc38grid['w_dep'][:-1]
lats38 = nc38['ULAT'][:]
lons38 = nc38['ULONG'][:]+25
lons38[lons38<0] += 360
exte38 = [minlon, maxlon, minlat38, maxlat]

U38 = (nc38['UVEL'][0])
V38 = (nc38['VVEL'][0])
bath38 = 0.01*np.sqrt(U38**2+V38**2)
#%% Interpolate both on the same grid
grid_x, grid_y = np.meshgrid(np.arange(0.1,360,0.1),
                             np.arange(-87,maxlat-0.1,0.1))

points = np.concatenate((lonsPD.flatten()[:,np.newaxis], latsPD.flatten()[:,np.newaxis]), axis=1)
values = bathPD.flatten()
bathPD = griddata(points, values, (grid_x, grid_y), method='nearest')

points = np.concatenate((lons38.flatten()[:,np.newaxis], lats38.flatten()[:,np.newaxis]), axis=1)
values = bath38.flatten()
bath38 = griddata(points, values, (grid_x, grid_y), method='nearest')

#%% start figure
# parameters for the labels
tc = 'k'
bb =dict(facecolor='white', alpha=0.6, edgecolor='k')

fig = plt.figure(figsize=(10,10))
#% subplot (a)
ax = plt.subplot(211, projection=projection)#EqualEarth(180))#Robinson(180))#
im2 = one_subplot(bathPD, grid_x, grid_y, ax, exte=exte38, cmap=cmap, 
                  vmin=vsbath[0], vmax=vsbath[1],
                lonticks=lonticks, latticks=latticks, title='(a) 1$^{\circ}$ resolution',
                fs = fs)

ax.annotate('DP', xy=(-70, -63), xytext=(-120,-70), fontsize=15,
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            arrowprops=dict(facecolor='red', arrowstyle='simple',
                            edgecolor='k',
                            alpha=0.95)
            )
ax.annotate('TG', xy=(150, -60), xytext=(90,-73), fontsize=15,
            bbox=bb, zorder=3001,
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            arrowprops=dict(facecolor='red', arrowstyle='simple',
                            edgecolor='k',
                            alpha=0.95)
            )

#% subplot (a)
ax2 = plt.subplot(212, projection=projection)
im2 = one_subplot(bath38, grid_x, grid_y, ax2, exte=exte38, cmap=cmap, 
                  vmin=vsbath[0], vmax=vsbath[1],
                lonticks=lonticks, latticks=latticks, title='(b) 0.1$^{\circ}$ resolution',
                fs = fs)
ax2.annotate('EAC', xy=(160, -37), xytext=(164,-25), fontsize=15,
            bbox=bb, zorder=3001,
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax2),
            arrowprops=dict(facecolor='red', arrowstyle='simple',
                            edgecolor='k',
                            alpha=0.95)
            )
ax2.annotate('Kuroshio', xy=(170, 37), xytext=(200,30), fontsize=15,
            bbox=bb, zorder=3001,
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax2),
            arrowprops=dict(facecolor='red', arrowstyle='simple',
                            edgecolor='k',
                            alpha=0.95)
            )
ax2.annotate('Agulhas', xy=(50, -37), xytext=(70,-30), fontsize=15,
            bbox=bb, zorder=3001,
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax2),
            arrowprops=dict(facecolor='red', arrowstyle='simple',
                            edgecolor='k',
                            alpha=0.95)
            )
ax2.annotate('ACC',  xy=(235, -47), xytext=(250,-35), fontsize=15,
            bbox=bb, zorder=3001,
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax2),
            arrowprops=dict(facecolor='red', arrowstyle='simple',
                            edgecolor='k',
                            alpha=0.95)
            )
            

#%% final (colorbar etc)

fig.subplots_adjust(bottom=0.08)
cbar_ax = fig.add_axes([0.2, 0.01, 0.6, 0.06])
cbar_ax.set_visible(False)
cbar = fig.colorbar(im2, ax=cbar_ax, orientation = 'horizontal', fraction = 0.8,
                    aspect=18, extend='max')
cbar.ax.xaxis.set_label_position('bottom')
cbar.ax.set_xlabel('m/s', fontsize=fs)
cbar.ax.tick_params(labelsize=fs)

plt.savefig('figure2.png', dpi=300,
            bbox_inches='tight',pad_inches=0)
plt.show()