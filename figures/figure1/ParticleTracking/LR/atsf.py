# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:31:22 2017

@author: nooteboom
"""

from parcels import (FieldSet, ParticleSet, JITParticle, AdvectionRK4_3D,
                     ErrorCode, ParticleFile, Variable, Field)
from datetime import timedelta as delta
from datetime import  datetime
import numpy as np
import math
from glob import glob
import sys

dirread_grid = '/projects/0/palaeo-parcels/Eocene/grids/'
dirread_U = '/projects/0/acc/pop/parcels/tx0.1/p21a.EO38Ma.tx0.1.2pic_control/tavg/'
dirread_T = '/projects/0/acc/pop/parcels/tx0.1/p21a.EO38Ma.tx0.1.2pic_control/movie/'

config = '2pic'
sp = float(sys.argv[1]) #The sinkspeed m/day
no = float(sys.argv[2])
dd = 10. #The dwelling depth

assert no<8
assert sp in [6,11,25,50,100,250]

dirwrite = '/projects/0/palaeo-parcels/Eocene/PT/sp%d/'%(sp)
locs = np.load('releaselocs/locs%d.npz'%(no))
latsz = locs['lats']
lonsz = locs['lons']
print('amount of releas locations: ', len(latsz))
assert ~(np.isnan(latsz)).any(), 'locations should not contain any NaN values'
dep = dd * np.ones(latsz.shape)

times = np.array([datetime(2009, 12, 30) - delta(days=x) for x in range(0,int(365),1)])
time = np.empty(shape=(0));lons = np.empty(shape=(0));lats = np.empty(shape=(0));
for i in range(len(times)):
    lons = np.append(lons,lonsz)
    lats = np.append(lats, latsz)
    time = np.append(time, np.full(len(lonsz),times[i])) 
#%%
def set_the_fieldset(ufiles, tfiles, bfile):
    filenames = { 'U': {'lon': bfile,
                        'lat': bfile,
                        'depth': bfile,
                        'data':ufiles},
                'V' : {'lon': bfile,
                        'lat': bfile,
                        'depth': bfile,
                        'data':ufiles},
                'W' : {'lon': bfile,
                        'lat': bfile,
                        'depth': bfile,
                        'data':ufiles},  
                'S' : {'lon': bfile,
                        'lat': bfile,
                        'depth': bfile,
                        'data':tfiles},   
                'T' : {'lon': bfile,
                        'lat': bfile,
                        'depth': bfile,
                        'data':tfiles} , 
                }

    variables = {'U': 'UVEL',
                 'V': 'VVEL',
                 'W': 'WVEL',
                 'T': 'TEMP_5m',
                 'S': 'SALT_5m'}

    dimensions = {
                  'U':{'lon': 'U_LON_2D', 'lat': 'U_LAT_2D', 'depth': 'w_dep','time':'time'},#
                  'V':{'lon': 'U_LON_2D', 'lat': 'U_LAT_2D', 'depth': 'w_dep','time':'time'},#
                  'W':{'lon': 'U_LON_2D', 'lat': 'U_LAT_2D', 'depth': 'w_dep','time':'time'},#
                  'T':{'lon': 'U_LON_2D', 'lat': 'U_LAT_2D', 'time':'time'},#
                  'S':{'lon': 'U_LON_2D', 'lat': 'U_LAT_2D', 'time':'time'}}
    if(no==0):
        indices = {'lat':range(700)}  
    elif(no==1):
        indices = {'lat':range(80000)}  
    elif(no==2):
        indices = {'lat':range(900)}  
    elif(no==3):
        indices = {'lat':range(1000)}  
    elif(no==4):
        indices = {'lat':range(250,1100)}  
    elif(no==5):
        indices = {'lat':range(250,1200)}
    elif(no==6):
        indices = {'lat':range(350,1300)}  
    elif(no==7):
        indices = {'lat':range(400,1406)}

    bfiles = {'lon': bfile, 'lat': bfile, 'data': [bfile, ]}
    bvariables = ('B', 'bathymetry')
    bdimensions = {'lon': 'U_LON_2D', 'lat': 'U_LAT_2D'}

    timestamps = np.expand_dims(np.array([np.datetime64('2009-12-31') - np.timedelta64(x,'D') for x in range(len(ufiles))])[::-1], axis=1)
    assert (len(timestamps)==np.array([len(ufiles), len(tfiles)])).all(), 'ts  %d  uf   %d  tf   %d'%(len(timestamps), len(ufiles),len(tfiles))
    fieldset = FieldSet.from_pop(filenames, variables, dimensions, allow_time_extrapolation=False, indices=indices,
                                     timestamps=timestamps)
    Bfield = Field.from_netcdf(bfiles, bvariables, bdimensions, allow_time_extrapolation=True, interp_method='bgrid_tracer', field_chunksize=False)
    fieldset.add_field(Bfield, 'B')
    fieldset.U.vmax = 10
    fieldset.V.vmax = 10
    fieldset.W.vmax = 10   
    return fieldset
        

def periodicBC(particle, fieldSet, time):
    if particle.lon > 180:
        particle.lon -= 360        
    if particle.lon < -180:
        particle.lon += 360   
        
#Sink Kernel if only atsf is saved:
def Sink(particle, fieldset, time):
    if(particle.depth>fieldset.dwellingdepth):
        particle.depth = particle.depth + fieldset.sinkspeed * particle.dt
    elif(particle.depth<=fieldset.dwellingdepth):
        particle.depth = fieldset.surface
        particle.temp = fieldset.T[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        particle.salin = fieldset.S[time+particle.dt, fieldset.surface, particle.lat, particle.lon]
        particle.delete()

def Age(particle, fieldset, time):
    particle.age = particle.age + math.fabs(particle.dt)  

def DeleteParticle(particle, fieldset, time):
    print('particle out of bounds %f %f %f'%(particle.lat,particle.lon,particle.depth))
    particle.delete()

def initials(particle, fieldset, time):
    if(particle.age)==0.:
        particle.lon0 = particle.lon
        particle.lat0 = particle.lat
        particle.depth = fieldset.B[time, fieldset.surface, particle.lat, particle.lon]
#        particle.depth0 = particle.depth

def run_corefootprintparticles(dirwrite,outfile,lonss,latss,dep):
    ufiles = sorted(glob(dirread_U + 't.p21a.EO38Ma.tx0.1.2pic_control.0036????.nc'))#[:150]
    tfiles = sorted(glob(dirread_T + 'm.p21a.EO38Ma.tx0.1.2pic_control.00*.nc'))[-1*len(ufiles):]
    bfile = dirread_grid+'kmt_tx0.1_POP_EO38.nc'
    print('set the field')
    fieldset = set_the_fieldset(ufiles, tfiles, bfile)
    fieldset.add_periodic_halo(zonal=True)       
    fieldset.add_constant('dwellingdepth', np.float(dd))
    fieldset.add_constant('sinkspeed', sp/86400.)
    fieldset.add_constant('maxage', 300000.*86400)
    fieldset.add_constant('surface', 2.5)
    print('field set')
    class DinoParticle(JITParticle):
        temp = Variable('temp', dtype=np.float32, initial=np.nan)
        age = Variable('age', dtype=np.float32, initial=0.)
        salin = Variable('salin', dtype=np.float32, initial=np.nan)
        lon0 = Variable('lon0', dtype=np.float32, initial=0.)
        lat0 = Variable('lat0', dtype=np.float32, initial=0.)
        
    pset = ParticleSet.from_list(fieldset=fieldset, pclass=DinoParticle, lon=lonss.tolist(), lat=latss.tolist(), 
                       time = time)

    pfile = ParticleFile(dirwrite + outfile, pset, 
                         write_ondelete=True)

    kernels = pset.Kernel(initials) + Sink  + pset.Kernel(AdvectionRK4_3D) + Age + periodicBC  

    pset.execute(kernels, runtime=delta(days=365*5), dt=delta(minutes=-20), output_file=pfile, verbose_progress=False, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle,ErrorCode.Delete: DeleteParticle})

    print('Execution finished')

outfile = 'atsf_dd'+str(int(dd)) +'_sp'+str(int(sp)) + '_%d'%(no) + '_%s'%(config)
run_corefootprintparticles(dirwrite,outfile,lons,lats,dep)

