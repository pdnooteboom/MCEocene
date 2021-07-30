#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:20:18 2020

@author: nooteboom
"""

import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from netCDF4 import Dataset
import seaborn as sns
import pandas as pd
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

#%% for cartopy plotting
def fieldplot(field):
    plt.figure(figsize=(10,8))
    plt.imshow(field, cmap='coolwarm', vmin=-100,vmax=100)
    plt.colorbar()
    plt.show()
    
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
def zonal_mean(T,area, lats, latsHR):
    # T must be 2D
    assert len(T.shape==2)
    res = np.zeros(len(lats)-1)
    weights = area
    weights[T.mask] = 0
    for i in range(len(lats)-1):
        idx = np.where(np.logical_and(latsHR>=lats[i], latsHR<lats[i+1]))
        res[i] = np.average(T[0][idx], weights=weights[idx])
    return res
    
def nearest_int(T, latm, lonm, lond, latd, res='HR'):
    # first nearest interpolate land values
    if(res=='LR'):
        lonnan = lonm[T<1e20]
        latnan = latm[T<1e20]
        points = np.concatenate((lonnan.flatten()[:,np.newaxis], latnan.flatten()[:,np.newaxis]), axis=1)
        values = T[T<1e20].flatten()
    else:
        lonnan = lonm[T!=0]
        latnan = latm[T!=0]
        points = np.concatenate((lonnan.flatten()[:,np.newaxis], latnan.flatten()[:,np.newaxis]), axis=1)
        values = T[T!=0].flatten()
    assert lonnan.shape[0]>0
    Tnew = griddata(points, values, (lonm, latm), method='nearest')
        
    points = np.concatenate((lonm.flatten()[:,np.newaxis], latm.flatten()[:,np.newaxis]), axis=1)
    values = Tnew.flatten()
    Td = griddata(points, values, (lond, latd), method='nearest')
    return Td.flatten()
 
def model_uncertainty(T, latm, lonm, lond, latd, deg = 3, res='HR'):
    # deg determines the degree box that is used to calculate the uncertainty
    # in the model due to the paleolocation
    # first nearest interpolate land values
    if(res=='LR'):
        lonnan = lonm[T<1e20]
        latnan = latm[T<1e20]
        points = np.concatenate((lonnan.flatten()[:,np.newaxis], latnan.flatten()[:,np.newaxis]), axis=1)
        values = T[T<1e20].flatten()
    else:
        lonnan = lonm[T!=0]
        latnan = latm[T!=0]
        points = np.concatenate((lonnan.flatten()[:,np.newaxis], latnan.flatten()[:,np.newaxis]), axis=1)
        values = T[T!=0].flatten()
    assert lonnan.shape[0]>0
    Tnew = griddata(points, values, (lonm, latm), method='nearest')

    lonmf = lonm.flatten(); latmf = latm.flatten();
    T = Tnew.flatten()
    distances = np.zeros(T.shape)
    resl = np.zeros(lond.shape)
    resh = np.zeros(lond.shape)
    for i in range(len(resl)):
        distances = (lond[i]-lonmf)**2+(latd[i]-latmf)**2
        arg = np.argmin(distances)
 #       idx = np.where(np.logical_and(np.logical_and(np.logical_and(lonmf>=lonmf[arg]-deg,lonmf<=lonmf[arg]+deg),
#                                                     latmf>=latmf[arg]-deg),
#                latmf<=latmf[arg]+deg))
        dist = np.sqrt((lonmf-lonmf[arg])**2+(latmf-latmf[arg])**2)
        
        idx = np.where(dist<=deg)
        resl[i] = np.min(T[idx])
        resh[i] = np.max(T[idx])
        
    return resl, resh

def cse(ar1, ar2):
    res = 0
    count = 0
    for i in range(len(ar1)):
        count += 1
        res += (ar1[i]-ar2[i])**2
    return np.sqrt(res/count)

def set_axis_label_fontsize(ax, fs):
    # increase the fontsize of all tick labels
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fs) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs)
    ax.xaxis.label.set_fontsize(fs)    
    ax.yaxis.label.set_fontsize(fs)
    
def optimize(T, Tmin, Tmax, Td):
    res = np.zeros(len(T))
    for i in range(len(Td)):
        if(Tmax[i]<Td[i]):
            res[i] = Tmax[i]
        elif(Tmin[i]>Td[i]):
            res[i] = Tmin[i]
        else:
            res[i] = Td[i]
    return res
    
def plot_merT(file, file2, ax1, ax2, ax3, axcb, prd={}, 
              config='2pic', panell=None, fs=22, 
              season='annual', opt=True):
    """
    file points to the low resolution CESM .nc output field
    file2  points to the high resolution POP .nc output field
    og is the outer grid
    prd is a dictionary which contains the proxy SSTs
    config flags which PIC configuration is used
    panell contains the titles of the subplots
    opt determines whether to use the optimized model SST values
    
    This function plots subplots (b), (c), (e), (f) and (g)-(j)
    """
    # Define the gridspec grid
    
    LRlatsres = 1.
    HRlatsres = 0.2
    lats = np.arange(-82,85,LRlatsres)
    pltlats = lats[:-1]+LRlatsres/2
    HRlats = np.arange(-77,88,HRlatsres)
    HRpltlats = HRlats[:-1]+HRlatsres/2
    ncf = Dataset(file)
    
    colors = ['cyan','red','green', 'k','grey','purple']
    markers = ['o','^','v','s','X','p']
    #%% Load the temperature and calculate the mean, minimum 
    # and maximum for every latitude (for the low resolution)
    if(season=='summer'):
        T = ncf['TEMP'][:]
    else:
        T = ncf['TEMP'][:]
    T = np.ma.masked_array(T, mask=(T>1000))
    ncft = Dataset('/Volumes/HD/Eocene/output/time_mean/final/cesm_merT_avg%dyears.nc'%(10))
    weight = ncft['TAREA'][:]
    ncft.close()
    for j in range(T.shape[1]):
        if(not (T[0,j].mask).all()):
            weight[j][T[0,j].mask] = 0
    latsLR = ncf['TLAT'][:]
    MTLR = np.zeros(len(lats)-1)
    MTLRmax = np.zeros(len(lats)-1)
    MTLRmin = np.zeros(len(lats)-1)
    for i in range(len(pltlats)):
        idx = np.where(np.logical_and(latsLR>=lats[i], latsLR<lats[i+1]))
        MTLR[i] = np.average(T[0][idx], weights=weight[idx])
        MTLRmax[i] = np.max(T[0][idx])
        MTLRmin[i] = np.min(T[0][idx])
        
    #%% Load the temperature and calculate the mean, minimum 
    # and maximum for every latitude (for the high resolution)
    ncf2 = Dataset(file2)
    
    T = ncf2['TEMP'][:]
    T = np.ma.masked_array(T, mask=(T==0))
    weight = ncf2['TAREA'][:]
    latsHR = ncf2['TLAT'][:]
    for j in range(T.shape[1]):
        if(not (T[0,j].mask).all()):
            weight[j][T[0,j].mask] = 0
    MTHR = np.zeros(len(HRlats)-1)
    MTHRmax = np.zeros(len(HRlats)-1)
    MTHRmin = np.zeros(len(HRlats)-1)
    for i in range(len(HRpltlats)):
        idx = np.where(np.logical_and(latsHR>=HRlats[i], latsHR<HRlats[i+1]))
        MTHR[i] = np.average(T[0][idx], weights=weight[idx])
        MTHRmax[i] = np.max(T[0][idx])
        MTHRmin[i] = np.min(T[0][idx])
    #%% Scatter model-data comparison
    if(True):
        lond, latd = prd['lon'], prd['lat']
        
        #% First the HR
        ncf2 = Dataset(file2)
        T = ncf2['TEMP'][0] 
        latm = ncf2['ULAT'][:]
        lonm = ncf2['ULONG'][:] + 25
        lonm[lonm>180] -= 360
        TdHR = nearest_int(T, latm, lonm, lond, latd)
        TdHRmin,TdHRmax  = model_uncertainty(T, latm, lonm, lond, latd)
        
        ##% Then the LR
        ncf = Dataset(file)
        if(season=='summer'):
            T = ncf['TEMP'][0]
        else:
            T = ncf['TEMP'][0]#/ counter
        latm = ncf['ULAT'][:]
        lonm = ncf['ULONG'][:]
        # Determine the temperature in the model, near the sites with SST proxies
        TdLR = nearest_int(T, latm, lonm, lond, latd, res='LR')
        
        # Determine the temperature uncertainty in the model, 
        # which is caused by uncertainty in the paleolocation of sites
        TdLRmin,TdLRmax  = model_uncertainty(T, latm, lonm, lond, latd, res='LR')
    
    #%%
    # define the data in dictionary, used by seaborn for plotting
        data1={
              'SST ($^{\circ}$C) model': TdHR,
              'SST ($^{\circ}$C) data': prd['T'],
              }
        df = pd.DataFrame.from_dict(data1)
        
        data2={
              'SST ($^{\circ}$C) model': TdLR,
              'SST ($^{\circ}$C) data': prd['T'],
              }
        df2 = pd.DataFrame.from_dict(data2)
    
    # subplot a
    ax1.set_title('%s'%(panell[config][1]), 
                  fontsize=fs)
    ax1.plot(MTHR,HRpltlats, linewidth=4, 
             label='HR%s'%(config[0]),c='k', zorder=1)
    ax1.fill_betweenx(HRpltlats, MTHRmin, MTHRmax,
                     facecolor="k", # The fill color
                     color='k',       # The outline color
                     alpha=0.15) # make the fill transparent/shaded
    
    ax1.set_yticks([-75,-25,25,75])
    
    ax1.tick_params(direction='in')

    ax1.plot(MTLR, pltlats, linewidth=4, 
             label='LR%s'%(config[0]),c='red', zorder=1)
    ax1.fill_betweenx(pltlats, MTLRmin, MTLRmax,
                     facecolor="red", # The fill color
                     color='red',       # The outline color
                     alpha=0.3)
    for m in range(len(np.unique(prd['method']))):
        idx = (prd['method']==np.unique(prd['method'])[m])
        ax1.errorbar(prd['T'][idx],prd['lat'][idx],
                     fmt=markers[m], xerr=prd['sigma'][idx], 
                     color=colors[m], label=np.unique(prd['method'])[m],
                     linewidth=0, elinewidth=1, barsabove=True, zorder = 2)
    
    ax1.set_yticklabels([])

    ax1.set_xlabel('$^{\circ}$C', fontsize=fs)
    ax1.xaxis.set_label_coords(1.09, -0.035)
    ax1.set_ylim(pltlats.min(),pltlats.max())
    ax1.set_xlim(2,40)
    ax1.xaxis.grid()
    if('2pic'==config):
        ax1.set_xlabel('')
        ax1.set_xticklabels([])
    
    if(True):
        # subplot b
        ax2.plot([5,40],[5,40], '--', color='k', linewidth=2)
        
        if(opt):
            df["SST ($^{\circ}$C) model"] = optimize(df["SST ($^{\circ}$C) model"],
                                                      TdHRmin,
                                                      TdHRmax,
                                                      df["SST ($^{\circ}$C) data"])
            sns.regplot(x="SST ($^{\circ}$C) data", y="SST ($^{\circ}$C) model",
                        data=df,color='k',
                       marker="o", ci=0, ax=ax2, fit_reg=False)
            ax2.errorbar(prd['T'], df["SST ($^{\circ}$C) model"],
                         xerr=prd['sigma'],
                     color='k',
                     linewidth=0, elinewidth=1, barsabove=True)
            seHR = cse(df["SST ($^{\circ}$C) model"], prd['T']) 
            
        else:
            yerr = [(TdHR - TdHRmin),(TdHRmax - TdHR)]
            sns.regplot(x="SST ($^{\circ}$C) data", y="SST ($^{\circ}$C) model",
                        data=df,color='k',
                       marker="o", ci=0, ax=ax2, fit_reg=False)
            ax2.errorbar(prd['T'], df["SST ($^{\circ}$C) model"],
                         xerr=prd['sigma'], yerr=yerr,
                     color='k',
                     linewidth=0, elinewidth=1, barsabove=True)
            seHR = cse(TdHR, prd['T']) 
        ax2.set_title('%s'%(panell[config][2])+', RMSE: %.2f'%seHR, fontsize=fs-6)
        
        ax2.set_xlim(6,39)
        ax2.set_ylim(6,39)
        if(config=='4pic'):
            ax2.set_ylabel('')
            ax2.set_yticklabels(['']*6)
        
        # subplot linear regression c
        if(opt):
            df2["SST ($^{\circ}$C) model"] = optimize(df2["SST ($^{\circ}$C) model"],
                                                      TdLRmin,
                                                      TdLRmax,
                                                      df2["SST ($^{\circ}$C) data"])
            ax3.errorbar(prd['T'], df2["SST ($^{\circ}$C) model"],
                         xerr=prd['sigma'], 
                         color='red',
                         linewidth=0, elinewidth=1, barsabove=True)
            sns.regplot(x="SST ($^{\circ}$C) data", y="SST ($^{\circ}$C) model", 
                        data=df2,color='red',
                       marker="x", ci=0, ax=ax3, fit_reg=False);
            seLR = cse(df2["SST ($^{\circ}$C) model"], prd['T']) 
            
        else:
            yerr = [(TdLR - TdLRmin),(TdLRmax - TdLR)]
            ax3.errorbar(prd['T'], df2["SST ($^{\circ}$C) model"],
                         xerr=prd['sigma'], yerr=yerr, 
                         color='red',
                         linewidth=0, elinewidth=1, barsabove=True)
        
            sns.regplot(x="SST ($^{\circ}$C) data", y="SST ($^{\circ}$C) model", 
                        data=df2,color='red',
                       marker="x", ci=0, ax=ax3, fit_reg=False);
            seLR = cse(TdLR, prd['T']) 
        ax3.set_title('%s'%(panell[config][3])+', RMSE: %.2f'%seLR,
                      fontsize=fs-6)
        ax3.plot([5,40],[5,40], '--', color='k', linewidth=2)
        ax3.set_ylabel('')
        ax3.set_yticklabels([])
        
        ax3.set_xlim(5,39)
        ax3.set_ylim(5,39)
        if('4pic'==config):
            ax3.set_ylabel('')
        
        set_axis_label_fontsize(ax1, fs-3)
        set_axis_label_fontsize(ax2, fs-3)
        set_axis_label_fontsize(ax3, fs-3)
        # print the standard error of the linear regression, both for the high and 
        # low resolution
        print('%s RMS LR, HR: '%(config),seLR, seHR)
    
#%%

def spatial_subplot(ax,grid_x, grid_y, bath, exte = [], title = '',
                    prd={},config='2pic', scat=None, cmap='Spectral_r',
                    fs=20, xticksl = [-180,-90,0,90,200],
                    latticks = [-88,-75,-25,25,75,88], vsbath=[-3,3]):
    # This functions creates the difference plots of (a) and (d)
    ax.set_title(title, fontsize=fs)

    g = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    g.xlocator = mticker.FixedLocator(xticksl)
    g.xlabels_top = False
    if(title[4]=='2'):
        g.xlabels_bottom = False
    g.ylabels_right = False
    g.xlabel_style = {'fontsize': fs-2}
    g.ylabel_style = {'fontsize': fs-2}
    g.xformatter = LONGITUDE_FORMATTER
    g.yformatter = LATITUDE_FORMATTER
    g.ylocator = mticker.FixedLocator(latticks)
    
    ax.set_extent(exte, ccrs.PlateCarree())

    bath = np.ma.masked_where(np.isnan(bath), bath)
    X, Y, masked_MDT = z_masked_overlap(ax, grid_x, grid_y, 
                                        bath,
                                        source_projection=ccrs.Geodetic())
    im2 = plt.pcolor(X, Y, masked_MDT, cmap=cmap, 
                          vmin=vsbath[0], vmax=vsbath[1],
                          )
    #im2 = ax.contourf(X, Y, masked_MDT,
    #                  levels=[-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3],
    #                  cmap=cmap, 
    #                     vmin=vsbath[0], vmax=vsbath[1],
    #                     )
    
    land = np.full(bath.shape, 0); land[bath>10**20] = 1;land[bath==0] = 1;
    land[bath.mask] = 1;
    land[np.isnan(bath)] = 1;
    land[bath<-1e20] = 1;
    
    ax.contour(grid_x, grid_y, land, [1], colors='k',
                         zorder=300, transform=ccrs.PlateCarree(),
                         )
    land = np.full(bath.shape, np.nan); land[bath>10**20] = 1;land[bath==0] = 1;
    land[bath.mask] = 1;
    land[np.isnan(bath)] = 1;
    land[bath<-1e20] = 1;
    
    X, Y, land = z_masked_overlap(ax, grid_x, grid_y, 
                                        land,
                                        source_projection=ccrs.Geodetic())
    
#    plt.pcolormesh(X, Y, land,  
#                   vmin=0, vmax=1.7,cmap='binary',
#                         zorder=2
#                         )
    plt.contourf(X, Y, land,  
                   vmin=0, vmax=1.7,cmap='binary',
                         zorder=2
                         )
    if(config=='2pic'):
        ax.scatter(scat, prd['lat'], color='k',
               marker='x', s=160, zorder=400)
    else:
        ax.scatter(scat, prd['lat'], color='k',
                   marker='+', s=160, zorder=400)
    return im2