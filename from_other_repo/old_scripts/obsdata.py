"""Functions for downloading observation data
"""
import datetime
import io
from urllib.request import urlopen, Request

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

from synoptic.plots import map_metadata
import synoptic.services as ss

def load_observations(stid,vrbls,recent_hrs=None,attime=None,within=None):
    if recent_hrs is not None:
        df = stations_timeseries(stid=stid,vars=vrbls,
                                 recent=datetime.timedelta(hours=recent_hrs))
    elif (attime is not None) and (within is not None):
        df = stations_nearesttime(stid=stid,vars=vrbls,
                                  attime=attime,within=within)
    else:
        # Always a good idea to have tripwires in your code whilst developing research!
        raise Exception
    return df


def image_spoof(self, tile): # this function pretends not to be a Python script
    """Thanks to Joshua Hrisko
    https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy
    """
    url = self._image_url(tile) # get the url of the street map API
    req = Request(url) # start request
    req.add_header('User-agent','Anaconda 3') # add user agent to request
    fh = urlopen(req)
    im_data = io.BytesIO(fh.read()) # get image
    fh.close() # close url
    img = Image.open(im_data) # open image with PIL
    img = img.convert(self.desired_tile_form) # set image format
    return img, self.tileextent(tile), 'lower' # reformat for cartopy

def plot_available_stations(radius, recent, vars="temperature",):
    # Terrain and roads etc
    cimgt.Stamen.get_image = image_spoof  # reformat web request for street map spoofing
    osm_img = cimgt.Stamen('terrain')  # spoofed, downloaded street map
    # ax1 = plt.axes(projection=osm_img.crs)  # project using coordinate reference system (CRS) of street map


    # fig,ax = plt.subplots(1, figsize=(8,6), subplot_kw=dict(projection=ccrs.LambertConformal()))
    # fig,ax = plt.subplots(1,figsize=(8,6),dict(projection=osm_img.crs))

    fig = plt.figure(figsize=(12, 9))  # open matplotlib figure
    ax = plt.axes(projection=osm_img.crs)  # project using coordinate reference system (CRS) of street map
    # f = map_metadata(ax=ax, verbose="HIDE",radius=radius,recent=recent)
    df = ss.stations_metadata(ax=ax, radius=radius, recent=recent)

    lats = df.loc["latitude"]
    lons = df.loc["longitude"]
    stid = df.loc["STID"]

    ax.scatter(lons, lats, transform=ccrs.PlateCarree())#, **scatter_kwargs)
    if True:
        for lon, lat, stn in zip(lons, lats, stid):
            # Colour-code by altitude?
            # ax.text(lon, lat, stn, transform=ccrs.PlateCarree(),)# **text_kwargs)
            pass

    # Add reference towns in RED
    towns = {
        "Vernal":[40.4555,-109.5287],
        "Roosevelt":[40.2994,-109.9888],

    }

    for town,latlon in towns.items():
        ax.scatter(latlon[1], latlon[0], color='red',transform=ccrs.PlateCarree())#, **scatter_kwargs)
        ax.text(latlon[1],latlon[0],town,color='red',transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.STATES.with_scale("10m"))
    ax.add_feature(cfeature.RIVERS.with_scale("10m"))
    # ax.stock_img()
    # ax.add_feature(cfeature.NaturalEarthFeature("physical","",scale="10m"))

    extent = [-110.5,-108.8,39.7,41.03]
    ax.set_extent(extent)  # set extents

    # empirical solve for scale based on zoom
    scale = np.ceil(-np.sqrt(2) * np.log(np.divide((extent[1] - extent[0]) / 2.0, 350.0)))
    scale = (scale < 20) and scale or 19  # scale cannot be larger than 19
    ax.add_image(osm_img, int(scale))  # add OSM with zoom specification

    ax.set_title("Station Locations", loc="left", fontweight="bold")
    ax.set_title(f"Total Stations: {len(df)}", loc="right")
    fig.tight_layout()
    fig.show()
    return

# valid_times = [datetime.datetime(2022,6,27,19,0,0),datetime.datetime(2022,6,27,23,0,0),datetime.datetime(2022,6,28,1,0,0)]
# obs_df = dict()
# for vt in valid_times:
#     obs_df[vt] = load_observations(airports,["air_temp","wind_direction","wind_speed"],attime=vt,within=20,)
#     Get rid of superfluous columns and we have our station names.
    # stids = [o for o in obs_df[vt].columns if "date_time" not in o]