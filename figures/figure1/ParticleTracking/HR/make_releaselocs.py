import numpy as np

dirread_grid = '/projects/0/palaeo-parcels/Eocene/grids/'
bfile = dirread_grid+'kmt_tx0.1_POP_EO38.nc'
bvariables = ('B', 'bathymetry')
bdimensions = {'lon': 'U_LON_2D', 'lat': 'U_LAT_2D'}

Bfield = Field.from_netcdf(bfile, bvariables, bdimensions, allow_time_extrapolation=True, interp_method='bgrid_tracer', field_chunksize=False)

for h in range(8):
    print(h)
    lons, lats = np.meshgrid(np.arange(0, 360,2)+0.5, np.arange(-80+6*h,-74+6*h,1)+0.5)
    lons = lons.flatten()
    lons[lons>180] -= 360
    lats = lats.flatten()

    lon = np.zeros(0)
    lat = np.zeros(0)
    for i in range(len(lons)):
        if(Bfield[0,0, lats[i], lons[i]]>0):
            lon = np.append(lon,lons[i])
            lat = np.append(lat,lats[i])

    np.savez('releaselocs/locs%d.npz'%(h), lons=lon, lats=lat)



