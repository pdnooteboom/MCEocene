#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:42:23 2020

@author: nooteboom
"""

import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import cmocean.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from glob import glob
from netCDF4 import Dataset
from matplotlib.lines import Line2D

def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)

pl = ['(c)','(e)','(g)']
pl4 = ['(d)','(f)','(h)']
# low resolution gateway tranports:
LRtr2 = {'Drake': 27.5,
            'Tasman':46,
            'Agulhas':29.5}
LRtr4 = {'Drake': 29,
            'Tasman':47,
            'Agulhas':32}

sns.set(style='whitegrid')
lw=2
yl = [-5, 5]
fs = 30 # fontsize
ssc= 60 # scatter size
years = 41
years2 = 26
coms = ['']

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

vs = [-1.5,1.5]
cmap = 'coolwarm'
contours = [0]

Hovres = 1
#%% Load the MOC time series
def load_MOCts(Read):
    #Calculate the maximum below: 
    mdepth = 1500 # meters
    #Calculate the maximum at higher latitudes: 
    mlat = 0
    years = np.array([])
    MOC = np.array([])
    NMOC = np.array([])
    SMOC = np.array([])
    for r in Read:
        files = np.sort(glob(r))
        if(len(years)>0):
            years = np.append(years,(np.arange(len(files))+1)/12+years[-1]) 
        else:
            years = np.append(years,(np.arange(len(files))+1)/12)
        for f in range(len(files)):
            nc = Dataset(files[f])
            if(f==0):
                depth = nc['depth_w'][:]
                lats = nc['lat_mht'][:]
                argd = np.where((depth>mdepth))[0]
                argl1 = np.where((lats>mlat))[0]
                argl2 = np.where((lats<-mlat))[0]      
            MOC = np.append(MOC,np.nanmax(nc['TMTG'][argd]))
            NMOC = np.append(NMOC,np.nanmax(nc['TMTG'][argd, argl1]))
            SMOC = np.append(SMOC,np.nanmin(nc['TMTG'][argd, argl2]))        
    return years, MOC, NMOC, SMOC

def load_yearly_MOCts(Read):
    #Calculate the maximum below: 
    mdepth = 1000 # meters
    #Calculate the maximum at higher latitudes: 
    mlat = 0
    MOC = np.array([])
    NMOC = np.array([])
    SMOC = np.array([])
    for r in Read:
        files = np.sort(glob(r))
        for f in range(len(files)):
            nc = Dataset(files[f])
            if(f==0):
                depth = nc['depth_w'][:]
                lats = nc['lat_mht'][:]
                argd = np.where((depth>mdepth))[0]
                argl1 = np.where((lats>mlat))[0]
                argl2 = np.where((lats<-mlat))[0]
                MOCc = np.zeros(nc['TMTG'][:].shape)
            MOCt = nc['TMTG'][:]
            MOCt[MOCt < -10**8] = 0
            MOCc += MOCt
            if(f%12==0):
                MOCc /= 12
                MOC = np.append(MOC,np.nanmax(MOCc[argd]))
                NMOC = np.append(NMOC,np.nanmax(MOCc[argd][:,argl1]))
                SMOC = np.append(SMOC,np.nanmin(MOCc[argd][:,argl2])) 
                MOCc = np.zeros(MOCc.shape)       
    return np.arange(len(MOC))+0.5, MOC, NMOC, SMOC


def yearlyMOC(NMOC, SMOC, monthsMOC):
    NMOCr = np.zeros(len(monthsMOC)//12+1)
    SMOCr = np.zeros(len(monthsMOC)//12+1)
    yearsMOC = []
    for i in range(len(monthsMOC)//12):
        yearsMOC.append(0.5+i)
        NMOCr[i] = np.mean(NMOC[i:i+12])
        SMOCr[i] = np.mean(SMOC[i:i+12])
    yearsMOC.append(0.5+monthsMOC.shape[0]//12)
    NMOCr[-1] = np.mean(NMOC[-monthsMOC.shape[0]%12:])
    SMOCr[-1] = np.mean(SMOC[-monthsMOC.shape[0]%12:])
    assert len(yearsMOC)>0, 'cannot find the MSF files'
    return np.array(yearsMOC), NMOCr, SMOCr

def last5year_avg(NMOC, SMOC, years=5):
    return np.mean(NMOC[-years*12:]), np.mean(SMOC[-years*12:])

# The 2pic MOC:
Read = ['/Volumes/HD/Eocene/output/MOCfiles/M*']
monthsMOC, MOCm, NMOCm, SMOCm = load_yearly_MOCts(Read)
yearsMOC, NMOCr, SMOCr = yearlyMOC(NMOCm, SMOCm, monthsMOC)
# Load the LR MOC 2pic
Read = ['/Volumes/HD/Eocene/output/MOCfiles/LR/M*']
monthsMOCLR, MOCmLR, NMOCmLR, SMOCmLR = load_yearly_MOCts(Read)


# Load the HR MOC 4pic
Read = ['/Volumes/HD/Eocene/output/MOCfiles/4pic/HR/M*']
monthsMOC4, MOCm4, NMOCm4, SMOCm4 = load_yearly_MOCts(Read)
# Load the LR MOC 4pic
Read = ['/Volumes/HD/Eocene/output/MOCfiles/4pic/LR/monthly/M*']
monthsMOCLR4, MOCmLR4, NMOCmLR4, SMOCmLR4 = load_yearly_MOCts(Read)

# Takse the mean of the last 5 years of the low resolution MOC
NMOCmLR4 = np.mean(NMOCmLR4[-5:])
SMOCmLR4 = np.mean(SMOCmLR4[-5:])
NMOCmLR = np.mean(NMOCmLR[-5:])
SMOCmLR = np.mean(SMOCmLR[-5:])

#%% Load the result

ff = np.load('Hov_transport_refinit_toplot_com%s_full.npz'%(coms[0]), allow_pickle=True)
depHR=ff['depHR']; depLR=ff['depLR'];
plotnames=ff['plotnames']; ylims=ff['ylims'][()]; 
HR4=ff['HR4'][:,::Hovres]; LR4=ff['LR4'][::Hovres];
time=ff['time'][:,::Hovres]; time2=ff['time2'][:,::Hovres]; 
deps=ff['deps'][:,::Hovres];deps2=ff['deps2'][:,::Hovres];
HR2=ff['HR2'][:,::Hovres]; LR2=ff['LR2'][::Hovres];HR2[282] = HR2[281];
timet=ff['timet']; transports=ff['transports'][()];
timet4=ff['timet4']; transports4=ff['transports4'][()];
yeartimet=ff['yeartimet']; yearlytransports=ff['yearlytransports'][()];
yeartimet4=ff['yeartimet4']; yearlytransports4=ff['yearlytransports4'][()];
print('loaded')
plotnames = np.array(['Drake', 'Tasman'])
#%%
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.edgecolor"] = 'k'
plt.rcParams["axes.linewidth"] = 4
#%%
fig = plt.figure(figsize=(16,16))
gs0 = gridspec.GridSpec(2, 1, figure=fig, hspace=0.17, 
                        wspace=0.02, height_ratios=[1,2])

gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[0], hspace=0)

ax00 = fig.add_subplot(gs00[0, 0])
ax01 = fig.add_subplot(gs00[0, 1])
c = coms[0]
ax00.invert_yaxis()
ax00.set_ylim(1,0)
ax01.set_ylim(1,0)
ax00.set_ylabel('depth (km)', fontsize=fs)
ax01.set_yticklabels([])
im = ax00.pcolormesh(time, deps,HR2-LR2, label='high resolution - low resolution',
                    cmap=cmap, vmin=vs[0], vmax=vs[1])

CS2 = ax00.contour(time, deps,HR2-LR2, contours, colors='k')

ax01.pcolormesh(time2, deps2,HR4-LR4, label='high resolution - low resolution',
                    cmap=cmap, vmin=vs[0], vmax=vs[1])
ax01.contour(time2, deps2,HR4-LR4, contours, colors='k')


ax00.set_title(r'(a) 2$\times$pre-industrial carbon', fontsize=fs)
ax01.set_title(r'(b) 4$\times$pre-industrial carbon', fontsize=fs)
plt.setp(ax00.get_xticklabels(), fontsize=fs)
plt.setp(ax01.get_xticklabels(), fontsize=fs)
ax01.xaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.setp(ax00.get_yticklabels(), fontsize=fs)

fig.subplots_adjust(right=0.95)
cbar_ax = fig.add_axes([0.95, 0.65, 0.1, 0.23])
cbar_ax.set_visible(False)
cbar = fig.colorbar(im, ax=cbar_ax, orientation = 'vertical', fraction = 0.8,
                    aspect=18, extend='both')
cbar.add_lines(CS2)
cbar.ax.xaxis.set_label_position('bottom')
cbar.ax.set_xlabel(unit[var], fontsize=fs)
cbar.ax.tick_params(labelsize=fs)

#%%
tyticks = [20,30,40] # The y-ticks used for the transport plots

gs01 = gs0[1].subgridspec(len(plotnames)+1, 2, hspace=0.25)

# transports 2pic

for pn in range(len(plotnames)):
    ax = fig.add_subplot(gs01[pn, 0])
    ax.set_ylabel('Sv', fontsize=fs)
    plt.setp(ax.get_yticklabels(), fontsize=fs)
    if(pn==0):
        ax.plot(timet, transports[plotnames[pn]],
           linewidth=lw, label='transport', color='k')
        ax.scatter(yeartimet+0.5, yearlytransports[plotnames[pn]], s=ssc,
           color='tab:red', zorder=10, edgecolor='white', label='yearly average')
        ax.plot([0,timet[-1]], [LRtr2[plotnames[pn]]]*2, '--',
           color='k', linewidth=lw+1, label='$1^{\circ}$ resolution')       
    else:
        ax.plot(timet, transports[plotnames[pn]],# label=plotnames[pn], 
           linewidth=lw, color='k')
        ax.scatter(yeartimet+0.5, yearlytransports[plotnames[pn]],s=ssc,
           color='tab:red', zorder=10, edgecolor='white')
        ax.plot([0,timet[-1]], [LRtr2[plotnames[pn]]]*2, '--',
           color='k', linewidth=lw+1)
    ax.set_xticklabels([])
    ax.set_yticks(tyticks)
    ax.set_title(pl[pn]+' '+plotnames[pn], fontsize=fs)
    if plotnames[pn] in ylims.keys():
        ax.set_ylim(ylims[plotnames[pn]][0], ylims[plotnames[pn]][1])
    else:
        ax.set_ylim(yl[0], yl[1])
    ax.set_xlim(0,years)


# transports 4pic

for pn in range(len(plotnames)):
    ax = fig.add_subplot(gs01[pn, 1])
    if(pn==0):
        ax.plot(timet4, transports4[plotnames[pn]], 
           linewidth=lw, label='transport', color='k')
        ax.scatter(yeartimet4+0.5, yearlytransports4[plotnames[pn]], s=ssc,
           color='tab:red', zorder=10, edgecolor='white', label='yearly average') 
        ax.plot([0,timet4[-1]], [LRtr4[plotnames[pn]]]*2, '--',
           color='k', linewidth=lw+1, label='$1^{\circ}$ resolution')       
    else:
        axf = ax
        ax.plot(timet4, transports4[plotnames[pn]],
           linewidth=lw, color='k')
        ax.scatter(yeartimet4+0.5, yearlytransports4[plotnames[pn]],s=ssc,
           color='tab:red', zorder=10, edgecolor='white')
        ax.plot([0,timet4[-1]], [LRtr4[plotnames[pn]]]*2, '--',
           color='k', linewidth=lw+1)
    ax.set_xticks([0,5,10,15,20,25])
    ax.set_xticklabels([])
    ax.set_yticks(tyticks)
    ax.set_title(pl4[pn]+' '+plotnames[pn], fontsize=fs)
    if plotnames[pn] in ylims.keys():
        ax.set_ylim(ylims[plotnames[pn]][0], ylims[plotnames[pn]][1])
    else:
        ax.set_ylim(yl[0], yl[1])
    ax.set_yticklabels([])
    ax.set_xlim(0,years2)

#%% plot the MOC yearsMOC, MOC, NMOC, SMOC
SMOCc = 'tab:red'
NMOCc = 'tab:blue'
ylims = [-21,15]
    
ax1 = fig.add_subplot(gs01[-1, 0])
ax1.plot(monthsMOC, NMOCm, marker='o', 
   color=NMOCc, zorder=10, label='yearly average') 
ax1.plot(monthsMOC, SMOCm, marker='o', 
   color=SMOCc, zorder=10, label='yearly average')

ax1.plot([0,monthsMOC[-1]], [NMOCmLR]*2, '--',
   color=NMOCc, linewidth=lw+1, label='$1^{\circ}$ resolution')
ax1.plot([0,monthsMOC[-1]], [SMOCmLR]*2, '--',
   color=SMOCc, linewidth=lw+1, label='$1^{\circ}$ resolution')

ax1.set_ylim(ylims[0],ylims[1])
ax1.set_xlim(0,years)
ax1.set_xlabel('year', fontsize=fs)
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(fs) 
ax1.set_ylabel('Sv', fontsize=fs) 
plt.setp(ax1.get_yticklabels(), fontsize=fs)
ax1.set_title('(g) MOC', fontsize=fs)

ax2 = fig.add_subplot(gs01[-1, 1])
ax2.plot(monthsMOC4, NMOCm4, marker='o', 
   color=NMOCc, zorder=10, label='yearly average') 
ax2.plot(monthsMOC4, SMOCm4, marker='o', 
   color=SMOCc, zorder=10, label='yearly average')
ax2.plot([0,monthsMOC4[-1]], [NMOCmLR4]*2, '--',
   color=NMOCc, linewidth=lw+1, label='$1^{\circ}$ resolution')
ax2.plot([0,monthsMOC4[-1]], [SMOCmLR4]*2, '--',
   color=SMOCc, linewidth=lw+1, label='$1^{\circ}$ resolution')

ax2.set_ylim(ylims[0],ylims[1])
ax2.set_xlim(0,years2)
ax2.set_yticklabels([])
ax2.set_xlabel('year', fontsize=fs)
ax2.set_title('(h) MOC', fontsize=fs)
ax2.set_xticks([0,5,10,15,20,25])
ax2.set_xticklabels([0,5,10,15,20,25])
for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(fs)
#%% Add the legends
custom_l1 = [Line2D([0], [0], color='k', lw=lw),
             Line2D([0], [0], marker='o', markerfacecolor='tab:red',
                    markersize=10,
                    markeredgecolor='white', color='tab:red', lw=0)]
custom_l12 = [
             Line2D([0], [0], linestyle='--', color='k', lw=lw)]

custom_l2 = [Line2D([0], [0], marker='o', color=NMOCc, lw=lw),
             Line2D([0], [0], marker='o', color=SMOCc, lw=lw)]

ax01.legend(custom_l1, 
          ['transport', 'yearly mean'],
          facecolor='lightgray',
          bbox_to_anchor=(-1.31, -2.85, 0.5, 0.5), prop={'size': fs})
ax.legend(custom_l12, 
          ['1$^{\circ}$ resolution'],
          facecolor='lightgray',
          bbox_to_anchor=(-0.3, -1.9, 0.5, 0.5), prop={'size': fs})
ax2.legend(custom_l2, 
          ['NMOC', 'SMOC'], facecolor='lightgray',
          bbox_to_anchor=(0.65, -0.76, 0.5, 0.5), prop={'size': fs})
#%% save and show figure
plt.savefig('figure3.png', dpi=300,bbox_inches='tight',pad_inches=0)
plt.show()