#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:26:25 2019

@author: nooteboom
"""

import numpy as np
import matplotlib.pylab as plt
from netCDF4 import Dataset
from copy import copy

pf = Dataset('bathymetry.nc')

bath = pf['Bathymetry'][:]
lons = pf['U_LON_2D'][:]
lats = pf['U_LAT_2D'][:]

fig = plt.figure(figsize=(7,5))
plt.imshow(np.flip(bath, axis=0), interpolation='none', vmax = 0)
plt.colorbar()
plt.show()
#%% Retrieve an ocean mask without any lakes
def check_ocean(j, i, bath): # Test whether index j,i is ocean
    if(bath[j][i]<0):
        return 1
    else:
        return -1

def check_neighbours(j, i, bath, oceanmask): # Check whether all the neighours of j,i are ocean
    uncheck = []
    for sj in [j-1,j+1]:
        if(sj<bath.shape[0]):
            si = i
            if(oceanmask[sj][si]==0):
                oceanmask[sj][si] = check_ocean(sj, si, bath)
                if(oceanmask[sj][si] == 1):
                    uncheck.append([sj, si])
    for si in [i-1,i+1]:
        sj = j
        if(oceanmask[sj][si]==0):
            oceanmask[sj][si] = check_ocean(sj, si, bath)
            if(oceanmask[sj][si] == 1):
                uncheck.append([sj, si])                
    return oceanmask, uncheck


def det_new_locations(list_diff, t,  oceanmask):
    t+=1
    fordel = []
    uncheck = []    
    for l in range(len(list_diff)):
        oceanmask, uncheckn = check_neighbours(list_diff[l][0], list_diff[l][1], bath, oceanmask)
        for i in range(len(uncheckn)): uncheck.append(uncheckn[i]);
        fordel.append(list_diff[l])
        
    return oceanmask, uncheck, fordel, t

def determine_oceanmask(bath):
    oceanmask = np.zeros(bath.shape)
    oceanmask[300, 500] = 1
    # put here grid cells of which you know that they are in the open ocean:
    list_diff = [[300, 500], [2500,3020]]
    t = 0
   
    while(len(list_diff)!=0):
        assert type(list_diff)==list, 'list_diff should be a list'
        oceanmask, uncheck, fordel, t = det_new_locations(list_diff, t,  oceanmask)
        for loc in uncheck:
            list_diff.append(loc)        
        for i in range(len(fordel)): list_diff.remove(fordel[i])        
        if(t%100==0):
            plt.figure(figsize=(5,5))
            plt.imshow(np.flip(oceanmask, axis=0), interpolation='none')
            plt.show()
        assert (bath[oceanmask==1]<0).all(), 'some ocean mask is incorrectly masked, bathymetry has to be lower than 0 at mask'
    return oceanmask

def test_check_ocean(bath):
    assert check_ocean(0,0,bath) == -1
    assert check_ocean(500,500,bath) == 1
    assert check_ocean(0,500,bath) == -1      
    print('test passed')
    
def test_check_neighbours():
    oceanmask = np.zeros(bath.shape)
    oceanmask[500, 500] = 1
    list_diff = [[500, 500]]
    t = 0 
    oceanmask, uncheck, fordel, t = det_new_locations(list_diff, t,  oceanmask)
    assert t==1
    assert uncheck == [[499, 500], [501, 500], [500, 499], [500, 501]]
    assert fordel == [[500, 500]]
    print('second test passed')
    
test_check_ocean(bath)
test_check_neighbours()

oceanmask = determine_oceanmask(bath)

foceanmask = np.zeros(bath.shape)
foceanmask[np.where(oceanmask==1)] = 1
foceanmask[0] = 1

plt.figure(figsize=(20,20))
plt.imshow(np.flip(foceanmask, axis=0), interpolation='none')
plt.show()

#%% Algorithm that checks in the ocean mask for narrow gateways.
def check_west_flow(j, i, mask):
    flow = False
    if(mask[j-1][i]==1):
        if(mask[j-2][i]==1):
            flow = True 
        elif(mask[j-1][i+1]==1):
            if(mask[j-2][i+1]==1):
                flow=True
            elif(mask[j-1, i+2]==1):
                 if(mask[j-1,i+2]==1):
                     flow = True
        elif(mask[j-1][i-1]==1):
            if(mask[j-2][i-1]==1):
                flow=True
            elif(mask[j-1, i-2]==1):
                 if(mask[j-1,i-2]==1):
                     flow = True                     
    return flow

def check_flow(j, i, mask, direction, ew = True): 
    # this function checks if a flow from one side to the other is possible in 
    # the 5*5 subgrid.
    # direction 1 is east or north, direction -1 is west or south
    # ew means east west direction
    flow = False
    if(ew):
        if(mask[j+direction][i]==1):
            if(mask[j+direction*2][i]==1):
                flow = True 
            elif(mask[j+direction][i+1]==1):
                if(mask[j+direction*2][i+1]==1):
                    flow=True
                elif(mask[j+direction, i+2]==1):
                     if(mask[j+direction*2,i+2]==1):
                         flow = True
            elif(mask[j+direction][i-1]==1):
                if(mask[j+direction*2][i-1]==1):
                    flow=True
                elif(mask[j+direction, i-2]==1):
                     if(mask[j+direction*2,i-2]==1):
                         flow = True  
    else:
        if(mask[j][i+direction]==1):
            if(mask[j,i+direction*2]==1):
                flow = True 
            elif(mask[j+1,i+direction]==1):
                if(mask[j+1,i+direction*2]==1):
                    flow=True
                elif(mask[j+2, i+direction]==1):
                     if(mask[j+2,i+direction*2]==1):
                         flow = True
            elif(mask[j-1,i+direction]==1):
                if(mask[j-1,i+direction*2]==1):
                    flow=True
                elif(mask[j-2, i+direction]==1):
                     if(mask[j+direction*2,i-2]==1):
                         flow = True          
    return flow  

def check_flow_east_west(j, i, mask):
    if(check_flow(j, i, mask, -1) and check_flow(j, i, mask, 1)):
        return True
    else:
        return False         

def check_flow_north_south(j, i, mask):
    if(check_flow(j, i, mask, -1, ew=False) and check_flow(j, i, mask, 1, ew=False)):
        return True
    else:
        return False 

def check_gate(j, i, mask):
    submask = mask[j-2:j+3, i-2:i+3]
    gate = False
    if(check_flow_east_west(2, 2, submask)):# check whether east-west flow is possible
        if((submask[2,3:]==0).any() and (submask[2,:2]==0).any()):
            if(np.sum(submask[2,2:])<3 and np.sum(submask[2,:3]==1)<3 and np.sum(submask[2,1:4])<3):    # check whether the gateway is of size less than 3      
                gate = True
    elif(check_flow_north_south(2, 2, submask)): # check whether north-south flow is possible
        if((submask[3:,2]==0).any() and (submask[:2,2]==0).any()):
            if(np.sum(submask[2:,2])<3 and np.sum(submask[:3,2])<3 and np.sum(submask[1:4,2])<3):    # check whether the gateway is of size less than 3             
                gate = True
    return gate

def test_check_west_flow():
    i = 2; j = 2;
    mask = np.zeros((5,5))
    mask[0,4]= 1
    mask[1,4]= 1
    mask[1,3]=1
    mask[1,2]=1
    mask[2,2]=1
    assert check_west_flow(j, i, mask), 'fail'
    assert check_flow(j, i, mask, -1), 'checkflow fail'
    mask = np.zeros((5,5))
    mask[4,4]= 1
    mask[3,4]= 1
    mask[3,3]= 1
    mask[3,2]= 1
    mask[2,2]= 1
    assert check_flow(j, i, mask, 1), 'checkflow in opposite direction fail'
    mask = np.zeros((5,5))
    mask[3,4]= 1
    mask[3,3]= 1
    mask[2,3]= 1
    mask[2,2]= 1 
    assert check_flow(j, i, mask, 1, ew=False), 'checkflow north-south direction fail'

def test_check_gate():
    i = 2; j = 2;
    mask = np.zeros((5,5))
    mask[3,4]= 1
    mask[3,3]= 1
    mask[2,3]= 1
    mask[2,2]= 1  
    mask[3,0]= 1
    mask[3,1]= 1
    mask[2,1]= 1     
    plt.imshow(mask)
    plt.show()
    assert check_gate(j, i, mask), 'checkgate fails'
    mask = np.zeros((5,5))
    mask[2] = 1  
    plt.imshow(mask)
    plt.show()
    assert check_gate(j, i, mask), 'checkgate 2 fails'
    mask = np.zeros((5,5))
    mask[2] = 1
    mask[3,2]= 1     
    mask[4,2]= 1     
    plt.imshow(mask)
    plt.show()
    assert check_gate(j, i, mask)==False, 'checkgate 3 fails'    

test_check_west_flow()
test_check_gate()

nga = np.zeros(bath.shape) # will mask the grid cells in narrow gateways
mask = foceanmask.astype(bool)

idx = np.where(mask)
print('no periodic boundaries implemented yet')
for l in range(np.sum(mask)):
    j = idx[0][l]
    i = idx[1][l]
    if(i>1 and j>1 and j<2546 and i<3597):
        nga[j, i] = check_gate(j, i, mask)
    if(l%500000==0):
        print(l/np.float(len(idx[0])))

oceanmask = copy(foceanmask)
foceanmask[np.where(nga)] = 3 #set all narrow ocean gateways to 3

plt.figure(figsize=(40,40))
plt.imshow(np.flip(foceanmask,axis=0), interpolation='none')
plt.colorbar()
plt.show()
#%% plot all grid cells in narrow gateways
domain = 40
for i in range(len(np.where(nga)[0])):
    j1 = np.where(nga)[0][i]-domain
    j2 = np.where(nga)[0][i]+domain
    i1 = np.where(nga)[1][i]-domain
    i2 = np.where(nga)[1][i]+domain
    
    print('j: %d   i:   %d'%(np.where(nga)[0][i], np.where(nga)[1][i]))

    plt.figure(figsize=(10,10))
    plt.imshow(np.flip(foceanmask[j1:j2,i1:i2], axis=0), interpolation='none')
    plt.colorbar()
    plt.show()
    
#%% Perform a check on single bulges 
def test_check_single_land_bulge():
    #vertical:
    mask = np.ones((5,5))
    mask[0] = 0
    mask[1] = 0
    mask[2,2] = 0
    plt.imshow(mask, interpolation ='none')
    plt.colorbar()
    plt.show()
    assert check_single_land_bulge(2, 2, mask)
    
def test_check_double_bulge():
    #vertical:
    mask = np.ones((3,3))
    mask[0,1] = 0; mask[1,1]=0;
    plt.imshow(mask, interpolation ='none')
    plt.colorbar()
    plt.show()
    assert check_double_land_bulge(1, 1, mask)
    mask = np.zeros((3,3))
    mask[0,1] = 1; mask[1,1]=1;
    plt.imshow(mask, interpolation ='none')
    plt.colorbar()
    plt.show()
    assert check_double_ocean_bulge(1, 1, mask)
    
def check_single_land_bulge(j, i, mask):
    res = False
    submask = mask[j-2:j+3, i-2:i+3]
    if((submask[:2] == 0).all() and np.sum(submask[2:])==14):
        res = True
    elif((submask[3:] == 0).all() and np.sum(submask[:3])==14):
        res = True
    elif((submask[:,:2] == 0).all() and np.sum(submask[:,2:])==14):
        res = True
    elif((submask[:,3:] == 0).all() and np.sum(submask[:,:3])==14):
        res = True    
    return res

def check_double_land_bulge(j, i, mask):
    res = False
    submask = mask[j-1:j+2, i-1:i+2]
    if(submask[0,1]==0 or submask[1,0]==0 or submask[2,1]==0 or submask[1,2]==0):
        if(np.sum(submask)==7):
            res = True  
    return res

def check_double_ocean_bulge(j, i, mask):
    res = False
    submask = mask[j-1:j+2, i-1:i+2]
    if(submask[0,1]==1 or submask[1,0]==1 or submask[2,1]==1 or submask[1,2]==1):
        if(np.sum(submask)==2):
            res = True  
    return res

def check_single_ocean_bulge(j, i, mask):
    res = False
    submask = mask[j-2:j+3, i-2:i+3]   
    if((submask[:2] == 1).all() and np.sum(submask[2:])==1):
        res = True
    elif((submask[3:] == 1).all() and np.sum(submask[:3])==1):
        res = True
    elif((submask[:,:2] == 1).all() and np.sum(submask[:,2:])==1):
        res = True
    elif((submask[:,3:] == 1).all() and np.sum(submask[:,:3])==1):
        res = True    
    return res

test_check_single_land_bulge()
test_check_double_bulge()
    
nga = copy(foceanmask) # will mask the grid cells which are single bulges
print('no periodic boundaries implemented yet')
for j in range(bath.shape[0]):
    for i in range(bath.shape[1]):
        if(i>1 and j>1 and j<2546 and i<3597):
            if(foceanmask[j,i]==1):
                if(check_single_ocean_bulge(j, i, oceanmask)):
                    nga[j,i] = 4
                elif(check_double_ocean_bulge(j, i, oceanmask)):
                    nga[j,i] = 5
            else:
                if(check_single_land_bulge(j, i, oceanmask)):
                    nga[j,i] = 6        
                elif(check_double_land_bulge(j, i, oceanmask)):
                    nga[j,i] = 7
    if(j%200==0):
        print(j/np.float(bath.shape[0]))

#%% plot all grid cells which are single bulges
domain = 30
print('first plot for the ocean grid cells')
print('#: ',len(np.where(nga==4)[0]))
for i in range(len(np.where(nga==4)[0])):
    j1 = max(0,np.where(nga==4)[0][i]-domain)
    j2 = np.where(nga==4)[0][i]+domain
    i1 = max(0,np.where(nga==4)[1][i]-domain)
    i2 = np.where(nga==4)[1][i]+domain
    
    print('j: %d   i:   %d'%(np.where(nga==4)[0][i], np.where(nga==4)[1][i]))


    plt.figure(figsize=(15,7))
    plt.subplot(121)
    plt.imshow(np.flip(nga[j1:j2,i1:i2],axis=0), interpolation='none', cmap='Dark2')#
    plt.colorbar()
    plt.subplot(122)
    plt.title('oceanmask')
    plt.imshow(np.flip(oceanmask[j1:j2,i1:i2],axis=0), interpolation='none')
    plt.colorbar()    
    plt.show()    
    plt.show()

print('#: ',len(np.where(nga==5)[0]))
for i in range(len(np.where(nga==5)[0])):
    j1 = np.where(nga==5)[0][i]-domain
    j2 = np.where(nga==5)[0][i]+domain
    i1 = np.where(nga==5)[1][i]-domain
    i2 = np.where(nga==5)[1][i]+domain

    print('j: %d   i:   %d'%(np.where(nga==5)[0][i], np.where(nga==5)[1][i]))

    plt.figure(figsize=(15,7))
    plt.subplot(121)
    plt.imshow(np.flip(nga[j1:j2,i1:i2],axis=0), interpolation='none', cmap='Dark2')#)
    plt.colorbar()
    plt.subplot(122)
    plt.title('oceanmask')
    plt.imshow(np.flip(oceanmask[j1:j2,i1:i2],axis=0), interpolation='none')
    plt.colorbar()    
    plt.show()    
    plt.show()   

print('Second for the land grid cells')
print('#: ',len(np.where(nga==6)[0]))
for i in range(len(np.where(nga==6)[0])):
    j1 = np.where(nga==6)[0][i]-domain
    j2 = np.where(nga==6)[0][i]+domain
    i1 = np.where(nga==6)[1][i]-domain
    i2 = np.where(nga==6)[1][i]+domain

    print('j: %d   i:   %d'%(np.where(nga==6)[0][i], np.where(nga==6)[1][i]))

    plt.figure(figsize=(15,7))
    plt.subplot(121)
    plt.imshow(np.flip(nga[j1:j2,i1:i2],axis=0), interpolation='none', cmap='Dark2')#)
    plt.colorbar()
    plt.subplot(122)
    plt.title('oceanmask')
    plt.imshow(np.flip(oceanmask[j1:j2,i1:i2],axis=0), interpolation='none')
    plt.colorbar()    
    plt.show()     
    plt.show()  
    
print('#: ',len(np.where(nga==7)[0]))
for i in range(len(np.where(nga==7)[0])):
    j1 = np.where(nga==7)[0][i]-domain
    j2 = np.where(nga==7)[0][i]+domain
    i1 = np.where(nga==7)[1][i]-domain
    i2 = np.where(nga==7)[1][i]+domain
    
    print('j: %d   i:   %d'%(np.where(nga==7)[0][i], np.where(nga==7)[1][i]))

    plt.figure(figsize=(15,7))
    plt.subplot(121)
    plt.imshow(np.flip(nga[j1:j2,i1:i2],axis=0), interpolation='none', cmap='Dark2')#)
    plt.colorbar()
    plt.subplot(122)
    plt.title('oceanmask')
    plt.imshow(np.flip(oceanmask[j1:j2,i1:i2],axis=0), interpolation='none')
    plt.colorbar()    
    plt.show() 
#%% Define the KMT levels
w_dep = pf['w_dep'][:]

kmt = np.zeros(bath.shape, dtype=int)
part_bot = np.zeros(bath.shape)

for j in range(bath.shape[0]):
    if(j%300==0):
        print(j/float(bath.shape[0]))
    for i in range(bath.shape[1]):
        if(w_dep[-1]<-bath[j,i]): # bathymetry deeper than deepest layer
            kmt[j,i] = 42
            part_bot[j,i] = min(-(w_dep[kmt[j,i]] + bath[j,i]), w_dep[-1]- w_dep[-2])
        elif(bath[j, i]<0): # ocean
            kmt[j,i] = int(np.argwhere(w_dep>-bath[j, i])[0][0] - 1)
            part_bot[j,i] = -(w_dep[kmt[j,i]] + bath[j,i])
        else: # land
            kmt[j,i] = 0
            part_bot[j,i] = 0

for j in range(bath.shape[0]):
    if(j%300==0):
        print(j/float(bath.shape[0]))
    for i in range(bath.shape[1]):
        if(w_dep[-1]<-bath[j,i]): # bathymetry deeper than deepest layer
            kmt[j,i] = 42
            part_bot[j,i] = min(-(w_dep[kmt[j,i]] + bath[j,i]), w_dep[-1]- w_dep[-2])
        elif(bath[j, i]<0): # ocean
            kmt[j,i] = int(np.argwhere(w_dep>-bath[j, i])[0][0] - 1)
            part_bot[j,i] = -(w_dep[kmt[j,i]] + bath[j,i])
        else: # land
            kmt[j,i] = 0
            part_bot[j,i] = 0

# put land where no ocean mask:
kmt[oceanmask == 0] = 0
part_bot[oceanmask ==0] = 0

#%% Read gridfile
pf_grid = Dataset('grid_coordinates_pop_tx0.1_38ma.nc')


# Write kmt file
dataset = Dataset('kmt_tx0.1_POP_EO38.nc', 'w')

longitudes = dataset.createDimension('longitude', 3600)
latitudes = dataset.createDimension('latitude', 2401+150)
i_indexs = dataset.createDimension('i_index', 3600)
j_indexs = dataset.createDimension('j_index', 2400+150)

kmts = dataset.createVariable('kmt', np.float32,('j_index','i_index'))#('latitude','longitude',)) # 
oceanmasks = dataset.createVariable('OCEAN_MASK', np.float32,('j_index','i_index'))#('latitude','longitude',)) # 
part_bots = dataset.createVariable('PART_BOT', np.float32,('j_index','i_index'))#('latitude','longitude',))
latitudes2 = dataset.createVariable('U_LAT_2D', np.float32,('j_index','i_index'))#('latitude','longitude',))
longitudes2 = dataset.createVariable('U_LON_2D', np.float32,('j_index','i_index'))#('latitude','longitude',))

htns = dataset.createVariable('HTN', np.float32,('j_index','i_index'))#('latitude','longitude',))
htes = dataset.createVariable('HTE', np.float32,('j_index','i_index'))#('latitude','longitude',))
huss = dataset.createVariable('HUS', np.float32,('j_index','i_index'))#('latitude','longitude',))
huws = dataset.createVariable('HUW', np.float32,('j_index','i_index'))#('latitude','longitude',))
angles = dataset.createVariable('ANGLE', np.float32,('j_index','i_index'))#('latitude','longitude',))
tarea = dataset.createVariable('TAREA', np.float32,('j_index','i_index'))#('latitude','longitude',))

latitudes2[:] = pf['U_LAT_2D'][:]
longitudes2[:] = pf['U_LON_2D'][:]

kmts[:] = kmt
part_bots[:] = part_bot
oceanmasks[:] = oceanmask.astype(bool)

htns[:] = pf_grid['HTN'][:]
htes[:] = pf_grid['HTE'][:]
huss[:] = pf_grid['HUS'][:]
huws[:] = pf_grid['HUW'][:]
angles[:] = pf_grid['ANGLE'][:]

part_bots.long_name = 'Depth of lowest partial bottom cell'
part_bots.units = 'meters'

dataset.close()
