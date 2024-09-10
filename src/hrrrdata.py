"""Class for loading, processing HRRR data.
"""
import datetime
import warnings

import pandas as pd
import pytz
import xarray
import numpy as np
import metpy
from cartopy import crs as ccrs
import matplotlib as M
import matplotlib.pyplot as plt

from herbie import Herbie

from src import utils

# Use for "operations"
# warnings.filterwarnings(action='ignore')

class HRRRData:
    def __init__(self):
        """Download, process, and plot HRRR data.
        """
        pass

    @staticmethod
    def setup_herbie(inittime, fxx=0, product="nat", model="hrrr"):
        # if inittime is not timezone-naive, make it so

        hb_dt = inittime.replace(tzinfo=None)

        H = Herbie(
            hb_dt,
            model=model,
            product=product,
            fxx=fxx,
        )
        return H

    @classmethod
    def determine_latest_hrrr(cls,long_only=False) -> datetime.datetime:
        """Go back hour-by-hour and see if HRRR data exists.
        """
        # ok_lengths = ("long",) if long_only else ("long","short")
        current_time = datetime.datetime.now(pytz.UTC)
        current_time = current_time.replace(minute=0, second=0, microsecond=0)

        while True:
            any_exists = cls.check_hrrr_exists(current_time)
            if not any_exists:
                current_time -= datetime.timedelta(hours=1)
            else:
                if long_only and (cls.get_hrrr_fmt(current_time) == "short"):
                    current_time -= datetime.timedelta(hours=1)
                else:
                    return current_time

    @classmethod
    def check_hrrr_exists(cls,dt):
        # Last forecast time
        last_fx = cls.get_nhr(dt)

        H = cls.setup_herbie(dt, fxx=last_fx)

        try:
            H.inventory()
        except ValueError:
            return False
        else:
            return True

    @classmethod
    def get_nhr(cls,dt):
        return 48 if cls.get_hrrr_fmt(dt) == "long" else 18

    @staticmethod
    def get_hrrr_fmt(dt) -> str:
        hours_set = {0, 6, 12, 18}
        if dt.hour in hours_set:
            return "long"
        return "short"

    @staticmethod
    def get_CONUS(qstr, herbie_inst):
        ds = herbie_inst.xarray(qstr, remove_grib=True)
        variables = [i for i in list(ds) if len(ds[i].dims) > 0]
        ds = ds.metpy.parse_cf(varname=variables).squeeze().metpy.assign_y_x()
        return ds

    @staticmethod
    def get_closest_point(ds, vrbl, lat, lon):
        grid_crs = ds[vrbl].metpy.cartopy_crs
        latlon_crs = ccrs.PlateCarree(globe=ds[vrbl].metpy.cartopy_globe)
        x_t, y_t = grid_crs.transform_point(lon, lat, src_crs=latlon_crs)
        pass
        return ds[vrbl].sel(x=x_t, y=y_t, method="nearest")

    @staticmethod
    def crop_to_UB(ds, ):
        sw_corner = (39.4, -110.9)
        ne_corner = (41.1, -108.5)
        lats = ds.latitude.values
        lons = ds.longitude.values

        if np.max(lons) > 180.0:
            lons -= 360.0

        crop = xarray.DataArray(np.logical_and(np.logical_and(lats > sw_corner[0],
                                                              lats < ne_corner[0]),
                                               np.logical_and(lons > sw_corner[1], lons < ne_corner[1])),
                                # dims=['south_north','west_east'])
                                dims=['y', 'x'])
        return ds.where(crop, drop=True)

    @staticmethod
    def save_xr_todisc(ds,fpath):
        pass
        # Save xarray to disc archive

    @staticmethod
    def do_sfc_plot(ds, vrbl, minmax=None):
        """Quick and dirty to check data is OK
        """
        # TODO: units and conversion elegantly!
        data = ds[vrbl]
        fig, ax = plt.subplots(1)
        if minmax is None:
            im = ax.imshow(data[::-1, :])
        else:
            im = ax.imshow(data[::-1, :], vmin=minmax[0], vmax=minmax[1])
        plt.colorbar(im)
        return fig, ax

    def make_query_string(vrbl):
        """To do
        """
        pass
        # Dictionary for variable names?

    @staticmethod
    def variable_catalogue(key):
        """Dictionary relating all variable names, titles, units, etc.

        vrbls[key] returns

        HRRR key
        xarray key (?)
        title string
        units

        My keys have a xxx_yyy format.

        Can leave query as blank string, as we use ":surface:" to get
        """
        vrbls = {}
        vrbls["accum_snow"] = {
            "hrrr_name":"ACSNO",
            "ds_name":"acsno",
            "title":"Accumulated Snowfall",
            "query":"",
            "units":"m"
        }
        vrbls["snow_depth"] = {
            ":SNOD:"
        }
        vrbls["temp_2m"] = {
            "hrrr_name":"T",
            "ds_name":"t2m",
            "title":"2-m Drybulb Temperature",
            "query": ":TMP:2 m",
            "units":"K",
        }
        return

    @staticmethod
    def sfc_vrbl_lookup():
        vrbls = {
            "Accum snowfall (m)": "acsno",
            "Temperature (K)": "t",
            "Pressure (Pa)":"sp",
            "PBL height (m)":"blh",
            "Gust (m/s)": "gust",
            "Downward SW (W/m2)":"dswrf",
            "Upward SW (W/m2)":"uswrf",
            "Download LW (W/m2)":"dlwrf",
            "Upward LW (W/m2)":"ulwrf",
            "Ground HF (W/m2)":"gflux",
            "Cloud Forcing NSF": "cfnsf",
            "Visible Beam Downward SF":"vbdsf",
            "Visible Diffuse Downward SF":"vddsf",
        }
        return vrbls

    @classmethod
    def generate_timeseries(cls, fxx, inittime, hrrr_regex, ds_key, lat, lon):
        """Need more info on variable names etc

        """
        timeseries = []
        for f in fxx:
            H = cls.setup_herbie(inittime, fxx=f, product="nat", model="hrrr")
            ds = cls.get_CONUS(hrrr_regex, H)
            ds_crop = cls.crop_to_UB(ds)
            val = cls.get_closest_point(ds_crop, ds_key, lat, lon)
            timeseries.append(val.values)
        return timeseries

    @classmethod
    def multiple_station_timeseries(cls,fxx,loc_dict,inittime,hrrr_regex,ds_key):
        loc_names = loc_dict.keys()
        data_dict = {l:[] for l in loc_names}
        lats = [x[0] for x in loc_dict.values()]
        lons = [x[1] for x in loc_dict.values()]
        timestamps = []
        for f in fxx:
            timestamp = inittime + datetime.timedelta(hours=f)
            timestamps.append(timestamp)
            inittime_naive = utils.convert_to_naive_utc(inittime)
            H = cls.setup_herbie(inittime_naive, fxx=f, product="nat", model="hrrr")
            ds = cls.get_CONUS(hrrr_regex, H)
            ds_crop = cls.crop_to_UB(ds)
            for loc_name, lat,lon in zip(loc_names,lats,lons):
                val = cls.get_closest_point(ds_crop, ds_key, lat, lon).values
                data_dict[loc_name].append(val)
        data_df = pd.DataFrame(data_dict, index=timestamps)
        return data_df

    def get_sfc_vrbls(self,fxx,inittime):#,lat,lon):
        qv = ":surface:"
        ds_list = []
        sfc_vrbls = self.sfc_vrbl_lookup()
        for f in fxx:
            H2 = self.setup_herbie(inittime, fxx=f, product="nat", model="hrrr")
            # for qv, timeseries in q_vars.items():
            ds = self.get_CONUS(qv, H2)
            ds_crop = self.crop_to_UB(ds).rename({'unknown': 'acsno'})
            ds_list.append(ds_crop)
        return ds_list
        # validtime = pd.Timestamp(i) + pd.Timedelta(hours=f)
        # for vrbl_title, vkey in sfc_vrbls.items():
        #     val = self.get_closest_point(ds_crop, vkey, lat, lon)
        #     print(val)

    @classmethod
    def get_cropped_data(cls,inittime,fxx,q_str,product="nat"):#,xr_str):
        H = cls.setup_herbie(inittime, fxx=fxx, product=product)
        ds = cls.get_CONUS(q_str, H)
        ds_crop = cls.crop_to_UB(ds)
        return ds_crop

    @classmethod
    def get_profile_df(cls,ds_T,ds_Z,lat,lon,max_height=10E3):
        # Label altitudes
        # can get profile of other things than temp...
        T_prof = cls.get_closest_point(ds_T, "t", lat, lon).values - 273.15  # Celsius
        Z_prof = cls.get_closest_point(ds_Z, "gh", lat, lon).values # m
        df = pd.DataFrame({"height":Z_prof, "temp":T_prof})

        # Now we find where Z_prof < max_height (m)
        return df[df["height"] < max_height]
