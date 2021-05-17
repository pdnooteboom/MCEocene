import numpy as np
import matplotlib.pylab as plt

lons = np.zeros(0)
lats = np.zeros(0)

for i in range(8):
    lo = np.load('locs%d.npz'%i)
    lons = np.append(lons,lo['lons'])
    lats = np.append(lats,lo['lats'])
#lons = np.array(lons)
#print(lons)

plt.scatter(lons,lats)
plt.show() 

