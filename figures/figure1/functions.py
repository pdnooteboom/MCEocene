#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:08:51 2021

@author: nooteboom
"""
import numpy as np
import matplotlib.pylab as plt
from netCDF4 import Dataset
from math import sin, cos, sqrt, atan2, radians
from matplotlib.lines import Line2D 
from copy import copy
import cartopy.crs as ccrs
import seaborn as sns
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker
from matplotlib.lines import Line2D
import cmocean.cm as cm
import cartopy.feature as cfeature
import matplotlib
from copy import copy
import matplotlib.path as mpath
from matplotlib import gridspec

def remove_land(lon, lat, perc):
    idx = np.where(perc!=0)
    lon = lon[idx]
    lat = lat[idx]
    perc = perc[idx]
    return lon, lat, perc

def model_uncertainty(T, latm, lonm, lond, latd, deglon = 1.2, deglat=0.6):
    # deg determines the degree box that is used to calculate the uncertainty
    # in the model due to the paleolocation
    # first nearest interpolate land values
    lonmf = lonm.flatten(); latmf = latm.flatten();
    T = T.flatten()

    distances = np.zeros(T.shape)
    resl = np.zeros(lond.shape)
    resh = np.zeros(lond.shape)
    for i in range(len(resl)):
        distances = np.sqrt((lond[i]-lonmf)**2+(latd[i]-latmf)**2)
        
        if(i in [26, 22] and False):
            fig,ax = plt.subplots(1,1,figsize=(4,3))
            im = ax.scatter(lonmf, latmf, vmin=0, vmax=3, c=distances, cmap='Spectral')
            plt.colorbar(im)
            ax.scatter([lond[i]], [latd[i]], c='k')
            ax.set_xlim(lond[i]-10,lond[i]+10)
            ax.set_ylim(latd[i]-10,latd[i]+10)
            plt.show()
        
        arg = np.argmin(distances)
        if(distances[arg]<1):
            deglon = 1.5
            deglat = 1.5
            idx = np.where(np.logical_and(np.logical_and(
                            np.logical_and(lonmf>=lonmf[arg]-deglon,
                                   lonmf<=lond[i]+deglon),
                                   latmf>=latd[i]-deglat),
                                latmf<=latd[i]+deglat))
            resl[i] = np.min(T[idx])
            resh[i] = np.max(T[idx])
        else:
            resl[i] = T[arg]
            resh[i] = T[arg]

    return resl, resh

def opt(Td, Tdmin, Tdmax, site):
    assert (Tdmax >= Tdmin).all()
    for i in range(len(Td)):
        if(Tdmax[i]<site['endemic (%)'][i]):
            Td[i] = Tdmax[i]
        elif(Tdmin[i]>site['endemic (%)'][i]):
            Td[i] = Tdmin[i]
        else:
            Td[i] = site['endemic (%)'][i]
    return Td

#%%
def det_land(sites={}, res='HR', config='2pic', sp=6):
    if(res=='HR'):
        dirRead = '/Volumes/HD/Eocene/PT/HR/'
        ncf = Dataset(dirRead + 'timeseries_per_location_ddeg1_sp%d_dd10_%s.nc'%(sp,config))
        vLons = ncf['vLons'][:];
        vLons += 25
    elif(res=='LR'):
        #% Load tracer output for the LR
        dirRead = '/Volumes/HD/Eocene/PT/LR/correct/'
        ncf = Dataset(dirRead + 'timeseries_per_locationLR_ddeg1_sp%d_dd10_%s.nc'%(sp,config))
        vLons = ncf['vLons'][:]; 

    vLons[vLons<0] += 360
    vLons[vLons>360] -= 360
    temp0 = ncf['temp'][:]
    lon0 = ncf['lon'][:]
    
    mland = np.full(len(vLons), False)
    for i in range(len(vLons)):
        if((lon0[i]==lon0[i,0]).all()):
            mland[i] = True
    
    idxnoland = np.logical_or(~(np.sum(temp0.mask,axis=1)>temp0.shape[1]-10),
                              mland)
    
    return idxnoland

#%%
#@jit(nopython=True)
def cdistance(lon1, lat1, lon2, lat2):
    res = np.zeros(len(lon2))
    
    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    
    for i in range(len(res)):
        lati2 = radians(lat2[i])
        loni2 = radians(lon2[i])
        dlon = loni2 - lon1
        dlat = lati2 - lat1
        a = (sin(dlat/2))**2 + cos(lat1) * cos(lati2) * (sin(dlon/2))**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        res[i] = R*c
    return res

def load_res(wmasks= [[-10, 17]], sites={}, config='2pic', res='HR', sp=6):
    #% Load tracer output for the HR
    if(res=='HR'):
        dirRead = '/Volumes/HD/Eocene/PT/HR/'
        ncf = Dataset(dirRead + 'timeseries_per_location_ddeg1_sp%d_dd10_%s.nc'%(sp,config))
        vLons = ncf['vLons'][:]; vLats = ncf['vLats'][:];
        vLons += 25
    elif(res=='LR'):
        #% Load tracer output for the LR
        dirRead = '/Volumes/HD/Eocene/PT/LR/correct/'
        ncf = Dataset(dirRead + 'timeseries_per_locationLR_ddeg1_sp%d_dd10_%s.nc'%(sp,config))
        vLons = ncf['vLons'][:]; vLats = ncf['vLats'][:];

    vLons[vLons<0] += 360
    vLons[vLons>360] -= 360
    
    # percentages
    temp0 = ncf['temp'][:]
    lon = ncf['lon'][:]
    
    idxnoland = det_land(sites=sites, res=res,
                         config=config, sp=sp) 
    
    temp = temp0[idxnoland]
    lon = lon[idxnoland]
    vLons = vLons[idxnoland]
    vLats = vLats[idxnoland]
    
    perc1 = np.zeros(len(vLons))
    tt = 0
    for i in range(len(vLons)):
        tslen = ((~temp.mask[i]).sum())
        if(tslen>0):
            tt += 1
            perc1[i] = 100*((temp[i]<=wmasks[0][1]).sum()) / tslen    

    memin, memax =  model_uncertainty(perc1, vLats, vLons, sites['plon'],
                                          sites['plat'])

    me = opt(np.zeros(len(sites['plon'])), memin, memax, sites)                                      
    
    pt = 2
    perc1[perc1<pt]= 0
    
    return me, vLons, vLats, perc1

#%% Heatmap Figure
def cse(ar1, ar2): # calculate RMSE between ar1 and ar2
    assert len(ar1)==len(ar2)
    assert (len(ar1.shape)==1), 'array should be one-dimensional'
    assert (len(ar2.shape)==1), 'array should be one-dimensional'
    res = 0
    count = 0
    for i in range(len(ar1)):
        count += 1
        res += (ar1[i]-ar2[i])**2
    return np.sqrt(res/count)

def Confusionm(ed, em):
    thres = 0
    FP = np.sum(np.logical_and(ed>0, em<=0))
    TP = np.sum(np.logical_and(ed>0, em>0))
    FN = np.sum(np.logical_and(ed<=0, em>thres))
    TN = np.sum(np.logical_and(ed<=0, em<=thres))
    Falses = np.full(len(ed), 'True')
    for i in range(len(ed)):
        if(ed[i]>0 and em[i]<=0):
            Falses[i] = 'FN'
        elif(ed[i]<=0 and em[i]>thres):
            Falses[i] = 'FP'
    return [[TP, FP], [FN, TN]], Falses

def add_jitter(ar1, ar2, sig=0.2):
    ar1 += np.random.uniform(-sig, sig, 1)[0]
    ar2 += np.random.uniform(-sig/2, sig/2, 1)[0]
    return ar1, ar2

def loopSSThat(sites, config = '2pic', res='HR', tb = [12,32, 15], sp=6):
    
    temps = np.linspace(tb[0],tb[1],tb[2])
    Fs = np.zeros(len(temps))
    
    ed = sites['endemic (%)']
    Fmin = 20
    
    F = [np.zeros(len(ed))]*len(temps)
    percs = []
    for i in range(len(temps)):
        em,  vLons, vLats, perc1 = load_res(wmasks= [[-10, temps[i]]], sites=sites,
                      config=config, res=res, sp=sp)
        M, F[i] = Confusionm(ed, em)
    
        FP = M[0][1]
        FN = M[1][0]
        Fs[i] = FP + FN
        
        percs.append(perc1)
        if(Fmin>Fs[i]):
            Fmin = Fs[i]
            es = [ed, em]
        
    imin, thf = opt_dist(vLons, vLats, percs, Fs, F, temps, sites=sites)
        
    return temps, Fs, thf, F, es, vLons, vLats, percs, imin

#%%
def opt_dist(vLons, vLats, percs, Fs, F, SSThs, sites={}):

    minT = np.where(Fs==Fs.min())[0]
    itmin = 10**10
    for j in minT:
        idxF = np.where(np.logical_or(np.array(F[j])=='FN',
                              np.array(F[j])=='FP'))
        distances = []
        for i in range(len(F[j])):
            if(F[j][i]=='FP'):
                cd = np.min(cdistance(sites['plon'][i], sites['plat'][i],
                               vLons[percs[j]>0],
                               vLats[percs[j]>0]))
                distances.append(cd)
            elif(F[j][i]=='FN'):
                cd = np.min(cdistance(sites['plon'][i], sites['plat'][i],
                               vLons[percs[j]==0],
                               vLats[percs[j]==0]))
                distances.append(cd)
            else:
                distances.append(0)
        distances = np.array(distances)
        if(cse(distances[idxF], np.zeros(len(distances[idxF]))) < itmin):
            itmin = cse(distances[idxF], np.zeros(len(distances[idxF])))
            argfinal = copy(j)
            thfinal = SSThs[argfinal]

    return argfinal, thfinal

#%%
def gdist(F,vLons, vLats,
             perc, config='2pic', sites={}):
    distsum = []
    for j in range(len(F)):
        dist = []
        for i in range(len(sites['names'])):
            if(F[j][i]=='FP'):
                cd = cdistance(sites['plon'][i], sites['plat'][i],
                               vLons[perc[j]==0], vLats[perc[j]==0])
                dist.append(np.min(cd))
            elif(F[j][i]=='FN'):
                cd = cdistance(sites['plon'][i], sites['plat'][i],
                               vLons[perc[j]>0], vLats[perc[j]>0])
                dist.append(np.min(cd))
        distsum.append(np.sum(dist))
    return np.array(distsum)

def subplot(ax, temps, sites,
             F, vLons, vLats, perc, Fs,
             title='', cmap='Spectral_r',
             config='2pic', res='HR', fs=20):

    distance = gdist(F,vLons, vLats,
                    perc, config=config, sites=sites)
        
    ax.scatter(Fs, distance / 1000, c=temps, s=120, edgecolors='k',
               cmap=cmap, vmin=temps.min(), vmax=temps.max())

    ax.set_yticks([2, 4 , 6, 8])
    if(title!='(a)'):
        ax.set_xlabel('# wrong', fontsize=fs)
    else:
        ax.set_xticklabels([])   
    ax.set_ylabel('distance (10$^{3}$ km)', fontsize=fs)
     
    ax.set_xticks([0,3, 6, 9, 12])
    if(config=='2pic'):
        ax.set_xlim(-0.05,11)
        ax.set_ylim(-0.05,6)
        ax.set_yticks([0,2,4,6])
    else:
        ax.set_ylim(-0.05,5)
        ax.set_xlim(-0.05,10)

    ax.set_title(title+' '+res+config[0], fontsize=fs)

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

def add_min90_bath(bath, lon, lat, val=-90):
    bath = np.concatenate((np.ones((1,bath.shape[1])), bath), axis=0)
    lat = np.concatenate((np.full((1,bath.shape[1]), val), lat), axis=0)
    lon = np.concatenate((lon[0][np.newaxis,:], lon), axis=0)
    return bath, lon, lat

def add_low_end(bath, lon, lat, val=-90):
    minlat = lat.min()
    addlats = np.arange(val,minlat)
    ulon = np.unique(lon)
    for i in range(len(addlats)):
        lon = np.append(lon, ulon)
        bath = np.append(bath, np.full(len(ulon), 100))
        lat = np.append(lat, np.full(len(ulon), addlats[i]))

    return bath, lon, lat


def subplotPT(ax, lons38, lats38, land, vLons, vLats, perc, 
            fs=20, exte38=[], cmapd='Spectral',
            projection=ccrs.PlateCarree(), cmap='viridis', sites={}, title=''):
    
    
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    
    sc = 150 # scatter size of cores

    ax.set_boundary(circle, transform=ax.transAxes)
    ax.add_feature(cfeature.OCEAN, zorder=0, color='k')
  #  ax.add_feature(cfeature.LAND, zorder=0, color='k')
    ax.text(60, -30, title, dict(size=fs) , transform=ccrs.Geodetic())
    
    ax.set_extent(exte38, ccrs.PlateCarree())
    ax.pcolormesh(lons38, lats38, land, transform=projection, 
                  vmin=0, vmax=1.2,cmap='gist_earth',zorder=2
                  )
    
    land2 = copy(land)
    land2[np.isnan(land2)] = 0
    X, Y, masked_MDT = z_masked_overlap(
        ax, lons38, lats38, land2,
        source_projection=ccrs.Geodetic())
    
    ax.contour(X, Y, masked_MDT, [0.5], colors='k', zorder=3, linewidth=2)
    perc, vLons, vLats = add_low_end(perc, vLons, vLats)
    p = ax.scatter(vLons[perc>0], vLats[perc>0], c=perc[perc>0], transform=projection, 
                   cmap=cmap, vmin=0, vmax=100, zorder=1)
    if(title[-3]=='L'):
        tc = 'k'
        idx = np.where(sites['plotname']==1)
        for i in range(len(idx[0])):#sites['names'])):
            if(sites['names'][idx[0][i]] in ['SanB','ODP1090']):
                bb =dict(facecolor='white', alpha=0.75, edgecolor='k')
                ax.text(sites['plon'][idx[0][i]]-5, sites['plat'][idx[0][i]]+2, 
                         sites['names'][idx[0][i]],
                         transform=projection,horizontalalignment='right',
                         color=tc, bbox=bb, zorder=3001, fontsize=15)
            elif(sites['names'][idx[0][i]] in ['TK']):
                bb =dict(facecolor='white', alpha=0.75, edgecolor='k')
                ax.text(sites['plon'][idx[0][i]]-5, sites['plat'][idx[0][i]]+2, 
                         sites['names'][idx[0][i]],
                         transform=projection,horizontalalignment='left',
                         color=tc, bbox=bb, zorder=3001, fontsize=15)
            else:
                bb =dict(facecolor='white', alpha=0.75, edgecolor='k')
                ax.text(sites['plon'][idx[0][i]]+5, sites['plat'][idx[0][i]], 
                         sites['names'][idx[0][i]],
                         transform=projection,horizontalalignment='right',
                         color=tc, bbox=bb, zorder=3001, fontsize=15)
    
    p0= ax.scatter(sites['plon'], sites['plat'], c=sites['endemic (%)'],
                 transform=ccrs.PlateCarree(),
                 cmap=cmapd,
                 s=sc, vmin=0, vmax=100, 
                 zorder=5, edgecolors='k', linewidth=2)
    
    return p, p0

def subplotPT_ocm(ax, lons38, lats38, land, vLons, vLats, perc, 
            fs=20, exte38=[],
            projection=ccrs.PlateCarree(), cmap='viridis', sites={}, title=''):
    
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    
    sc = 150 # scatter size of cores

    ax.set_boundary(circle, transform=ax.transAxes)
    ax.add_feature(cfeature.OCEAN, zorder=0, color=cmap(0))
    ax.add_feature(cfeature.LAND, zorder=0, color=cmap(0))
    ax.text(60, -30, title, dict(size=fs) , transform=ccrs.Geodetic())
    
    ax.set_extent(exte38, ccrs.PlateCarree())
    
    land2 = copy(land)

    # ax.pcolormesh(lons38, lats38, land, transform=projection, 
    #               vmin=0, vmax=1.4,cmap='Greys',zorder=2,
    #   #            extent= exte38
    #                     )
    
    land2[np.isnan(land2)] = 0
    X, Y, masked_MDT = z_masked_overlap(
        ax, lons38, lats38, land2,
        source_projection=ccrs.Geodetic())
    
    ax.contour(X, Y, masked_MDT, [0.5], colors='k', zorder=3, linewidth=2)
    
    land = np.ma.masked_where(np.isnan(land), land)
    
    X, Y, land = z_masked_overlap(
        ax, lons38, lats38, land,
        source_projection=ccrs.Geodetic())
    ax.contourf(X, Y, land, levels=10, #transform=projection, 
                   vmin=0, vmax=1.5,cmap='Greys',zorder=2,
       #            extent= exte38
                         )    
    
    perc, vLons, vLats = add_low_end(perc, vLons, vLats)
    perc[np.logical_and(perc==100, vLats>-50)] = 0
    p = ax.scatter(vLons[perc>0], vLats[perc>0], c=perc[perc>0], transform=projection, 
                   cmap=cmap, vmin=0, vmax=100, zorder=1)
    if(title[-3]=='L'):
        tc = 'k'
        idx = np.where(sites['plotname']==1)
        for i in range(len(idx[0])):#sites['names'])):
            if(sites['names'][idx[0][i]] in ['SanB','ODP1090']):
                bb =dict(facecolor='white', alpha=0.75, edgecolor='w')
                ax.text(sites['plon'][idx[0][i]]-5, sites['plat'][idx[0][i]]+2, 
                         sites['names'][idx[0][i]],
                         transform=projection,horizontalalignment='right',
                         color=tc, bbox=bb, zorder=3001, fontsize=15)
            elif(sites['names'][idx[0][i]] in ['TK']):
                bb =dict(facecolor='white', alpha=0.75, edgecolor='w')
                ax.text(sites['plon'][idx[0][i]]-5, sites['plat'][idx[0][i]]+2, 
                         sites['names'][idx[0][i]],
                         transform=projection,horizontalalignment='left',
                         color=tc, bbox=bb, zorder=3001, fontsize=15)
            else:
                bb =dict(facecolor='white', alpha=0.75, edgecolor='w')
                ax.text(sites['plon'][idx[0][i]]+5, sites['plat'][idx[0][i]], 
                         sites['names'][idx[0][i]],
                         transform=projection,horizontalalignment='right',
                         color=tc, bbox=bb, zorder=3001, fontsize=15)
    
    p0= ax.scatter(sites['plon'], sites['plat'], c=sites['endemic (%)'],
                 transform=ccrs.PlateCarree(),
                 cmap=cmap,
                 s=sc, vmin=0, vmax=100, 
                 zorder=5, edgecolors='w', linewidth=2)
    
    return p, p0