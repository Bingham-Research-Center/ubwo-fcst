"""Script to download HRRR archive forecasts

TODO:
* Decide on variables
    - Sfc snow depth
    - 3D temperature, pressure/height, specific humidity, wind vectors
    - 2m temp/dew-point
    - 10m wind vector

* Download each initialisation time every 3h
* Get all forecasts times up to 24h ahead, inclusive
    - This gives option of lagged ensemble
* Check size and time to download - save in xarray/numpy format?
* lat/lons are same each time, so can save separately to conserve disc space

* Get satellite and obs data to verify each day
    - Where is forecasting drop-off in the winter?

* Need naming scheme to save then load data depending on
    - init time
    - forecast time
    - variable and levels
* Also save lats, lons, levels

"""
import datetime
import time

import matplotlib as M
import matplotlib.pyplot as plt
from matplotlib import rc
import xarray as xr
import numpy as np
import cartopy.feature as cfeature
from cartopy import crs
from herbie import  FastHerbie, Herbie

# Time and check size of download
data_source = "aws"
nprocs = 10


### FUNCTIONS
def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def create_regex(vrbls):
    # qstr = "(:CIMIXR:|:CLMR:)"
    qstr = "("
    for i,v in enumerate(vrbls):
        qstr += f":{v}:"
        if i == len(vrbls):
            qstr += "|"
    qstr += ")"
    return qstr


# Test for one init time,
init_dt = datetime.datetime(2023,1,30,12,0,0)
init_dt_str = datetime.datetime.strftime(init_dt, "%Y-%m-%d %H:%M")

fchrs = list(range(0,26,3))
Wlim = -110.7
Elim = -108.7
Nlim = 41.05
Slim = 39.35
# my_extent = [-110.6,-108.7,41.05,39.65]
my_extent = [Wlim, Elim, Nlim, Slim]
# map_extent = [Wlim-1, Elim+1, Nlim+0.5, Slim-0.5]
map_extent = my_extent

# v_regex = "TMP:2 m"
v_regex = "(:SNOD:)"

if v_regex == "TMP:2 m":
    FH = FastHerbie([init_dt_str,], model="hrrr", fxx=fchrs, source=data_source)
    FH.download(v_regex, max_threads=nprocs)
    ds_t2m = FH.xarray(v_regex, remove_grib=True)

    # This only works for lats/lons in 1D - even grid
    ds_t2m_coord = ds_t2m.assign_coords({"y": (("y",), ds_t2m.latitude.values[:, 0]),
                                   "x": (("x",), ds_t2m.longitude.values[0, :])})
    ds_t2m_crop = ds_t2m_coord.sel(y=slice(Slim,Nlim), x=slice(Wlim+360,Elim+360))

elif v_regex == "(:SNOD:)":
    FH = FastHerbie([init_dt_str,], model="hrrr", fxx=fchrs, source=data_source, product="sfc")
    FH.download(v_regex, max_threads=nprocs)
    ds_snod = FH.xarray(v_regex, remove_grib=True)

    ds_snod_coord = ds_snod.assign_coords({"y": (("y",), ds_snod.latitude.values[:, 0]),
                                   "x": (("x",), ds_snod.longitude.values[0, :])})
    ds_snod_crop = ds_snod_coord.sel(y=slice(Slim,Nlim), x=slice(Wlim+360,Elim+360))
    ds_t2m_crop = ds_snod_crop

locations_of_interest = [(-109.5287,40.4555),(-110.4029,40.1633),
                         (-109.6774,40.0891),(-109.07315,39.78549),
                         (-110.03689,39.53777),(-110.3728,40.7764),
                         (-109.9888,40.2994),
                         ]
names_of_interest = ["Vernal","Duchesne","Ouray","Dragon",
                        "Rock Creek Ranch","Kings Peak",
                        "Roosevelt",
                        ]
my_transform = crs.PlateCarree()
nlines = 5
lonlines = trunc(np.linspace(my_extent[0], my_extent[1], nlines), 1)
latlines = trunc(np.linspace(my_extent[2], my_extent[3], nlines), 1)

coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',
                            edgecolor='black', name='coastline')

counties = cfeature.NaturalEarthFeature(category='cultural', scale='10m',
                            edgecolor='black', name='admin_2_counties_lakes',
                                        alpha=0.2)

# plot_data = ds_t2m_crop.isel(step=fchrs.index(6)).t2m - 273.15
plot_data = ds_t2m_crop.isel(step=fchrs.index(6)).sde

fig, ax = plt.subplots(1, figsize=[7, 5], constrained_layout=True, dpi=200,
                        # subplot_kw={'projection': ds_t2m_crop.herbie.crs})
                        subplot_kw={'projection': my_transform})
f = ax.pcolormesh(
                    ds_t2m_crop.x, ds_t2m_crop.y, plot_data,
                    # ds_t2m_crop.longitude, ds_t2m_crop.latitude, plot_data,
                    # vmin=-25, vmax=5,
                    alpha=0.5, cmap=M.cm.Purples, transform=my_transform)



c1 = plt.colorbar(f, fraction=0.046, pad=0.04)
# c1.set_label(label='2m Temperature (Celsius)', size=18, weight='bold')
c1.ax.tick_params(labelsize=18)

# Zoom
ax.set_extent(map_extent, crs=my_transform)

# Gridlines
# gl1 = ax.gridlines(xlocs=lonlines, ylocs=latlines, x_inline=False, rotate_labels=False)
# gl1.xlabels_bottom = True
# gl1.ylabels_left = True
# gl1.xlabel_style = {'size': 18}
# gl1.ylabel_style = {'size': 18}
# ax.set_title(tstr, loc="left")

# Map features
ax.add_feature(cfeature.STATES, facecolor='none', edgecolor='black')
ax.add_feature(cfeature.RIVERS.with_scale("10m"), facecolor='blue', edgecolor='blue')
ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor='cyan', edgecolor='cyan')
ax.add_feature(coast, facecolor='none', edgecolor='black')
ax.add_feature(counties, facecolor='none', edgecolor='black')

# Plot Locations
ax.scatter([loc[0] for loc in locations_of_interest],
           [loc[1] for loc in locations_of_interest], transform=my_transform, marker='o', color='r')

for i in range(len(names_of_interest)):
    ax.text(locations_of_interest[i][0], locations_of_interest[i][1],
            names_of_interest[i], transform=my_transform, size=15)
fig.show()
pass
