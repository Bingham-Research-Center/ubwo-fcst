""" Wrapper to make it easier to extract data from HRRR data
"""
import datetime

import matplotlib as M
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import cartopy.feature as cfeature
from cartopy import crs
from herbie import FastHerbie

import utils

class HRRRData:
    def __init__(self, init_dt, valid_dt=None, fchr=None, crop_extent=None, nprocs=1):
        """
        init_dt (datetime.datetime)     : Initialsation time for HRRR 
        valid_dt (datetime.datetime)    : Valid time for extracted data
        crop_extent (list,tuple)        : my_extent = [Wlim, Elim, Nlim, Slim]
        
        """
        self.init_dt = init_dt
        self.valid_dt = valid_dt
        self.crop_extent = crop_extent

        if fchr is not None:
            self.fchr = fchr
            self.valid_dt = self.init_dt + datetime.timedelta(hours=fchr)
        else:
            assert self.valid_dt is not None
            self.fchr = int((valid_dt - init_dt).total_seconds()/3600)

        self.data_source = "aws"

        # Some settings for plotting
        font = {
                # 'family': 'Arial Nova',
                'family': "helvetica",
                # 'family' : 'sans-serif',
                'weight': 'bold',
                'size': 12}

        rc('font', **font)


    @staticmethod
    def trunc(values, decs=0):
        return np.trunc(values*10**decs)/(10**decs)

    @staticmethod
    def create_regex(vrbls):
        # qstr = "(:CIMIXR:|:CLMR:)"
        qstr = "("
        for i,v in enumerate(vrbls):
            qstr += f":{v}:"
            if i == len(vrbls):
                qstr += "|"
        qstr += ")"
        return qstr

    @staticmethod
    def str_from_dt(dt):
        return datetime.datetime.strftime(dt, "%Y-%m-%d %H:%M")

    def get(vrbl, inittime, validtime, fcsthr, lv):
        """Eventually this will download any subset with one function... 
        """

        pass
        return


    @staticmethod
    def lookup(vrbl, key):
        """Don't put units here - get from metadata
        """
        VRBLS = dict()
        VRBLS["t2m"] = {'long_name': 'Temperature (2m)',
                        'grib_name': "TMP:2 m",
                        'attr_name': 't2m'
                        }
        VRBLS["snod"] = {'long_name': 'Snow depth',
                         'grib_name': "(:SNOD:)",
                         'attr_name': "sde"
                         }
        VRBLS["temp"] = {'long_name': "Temperature (drybulb? Pot?)",
                        'grib_name': "TMP",
                        "attr_name": "t"
                        }
        return VRBLS[vrbl][key]

    def get_3d(self,vrbl,fchr):
        """Get things like 3-D temperature.
        """
        FH = FastHerbie([self.str_from_dt(self.init_dt),], model="hrrr", fxx=range(fchr,fchr+1),
                        source=self.data_source,
                        # product="sfc",
                        product="nat",
                        )
        v_regex = self.lookup(vrbl, "grib_name")
        FH.download(v_regex, max_threads=nprocs)
        ds = FH.xarray(v_regex, remove_grib=True)

        # This only works for lats/lons in 1D - an even grid
        ds_coord = ds.assign_coords({"y": (("y",), ds.latitude.values[:, 0]),
                                     "x": (("x",), ds.longitude.values[0, :])})
        ds_crop = ds_coord.sel(y=slice(Nlim,Slim), x=slice(Wlim + 360, Elim + 360))
        # ds_crop = ds_coord.sel(y=slice(Slim, Nlim), x=slice(Wlim + 360, Elim + 360))
        return ds_crop

    def get_t2m(self,vrbl,fchr):
        """
        I'm sure this works for more than 2-m temp.
    
        t2m or TMP:2 m --> does not need "product"
        snod or snow_depth or SNOD --> 
        """
        FH = FastHerbie([self.str_from_dt(self.init_dt),], model="hrrr", fxx=range(fchr,fchr+1),
                        source=self.data_source)  # ,product="sfc")
        v_regex = self.lookup(vrbl, "grib_name")
        FH.download(v_regex, max_threads=nprocs)
        ds = FH.xarray(v_regex, remove_grib=True)

        pass
        units = ds.variables[v_regex].attrs["units"]
        pass

        # This only works for lats/lons in 1D - even grid
        ds_coord = ds.assign_coords({"y": (("y",), ds.latitude.values[:, 0]),
                                     "x": (("x",), ds.longitude.values[0, :])})
        ds_crop = ds_coord.sel(y=slice(Slim, Nlim), x=slice(Wlim + 360, Elim + 360))
        return ds_crop

    @classmethod
    def plot_data(cls, vrbl, x, y, ds, plot_extent=None, locs_dict=None, do_lat_lon_lines=True,
                    title='', save_fpath=None):
        """TODO:
        * Deal with units somewhere
        """
        v_attr = cls.lookup(vrbl, "attr_name")
        data = ds.variables[v_attr]
        transform_crs = crs.PlateCarree()
        fig, ax = plt.subplots(1, figsize=[8, 6], constrained_layout=True, dpi=200,
                               # subplot_kw={'projection': ds_t2m_crop.herbie.crs})
                               subplot_kw={'projection': transform_crs})
        f = ax.pcolormesh(
            x, y, data,
            # ds_t2m_crop.longitude, ds_t2m_crop.latitude, plot_data,
            # vmin=-25, vmax=5,
            alpha=0.5, cmap=M.cm.Purples, transform=transform_crs)

        c1 = plt.colorbar(f, fraction=0.046, pad=0.04,)
        c1.set_label(label='Variable (Units)', size=14, weight='bold')
        c1.ax.tick_params(labelsize=14)

        # Zoom
        ax.set_extent(plot_extent, crs=transform_crs)



        # Map features
        ax.add_feature(cfeature.STATES, facecolor='none', edgecolor='black')
        ax.add_feature(cfeature.RIVERS.with_scale("10m"), facecolor='blue', edgecolor='blue')
        ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor='cyan', edgecolor='cyan')

        coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',
                                             edgecolor='black', name='coastline')

        counties = cfeature.NaturalEarthFeature(category='cultural', scale='10m',
                                                edgecolor='black', name='admin_2_counties_lakes',
                                                alpha=0.2)

        ax.add_feature(coast, facecolor='none', edgecolor='black')
        ax.add_feature(counties, facecolor='none', edgecolor='black')

        if do_lat_lon_lines:
            nlines = 5
            lonlines = HRRRData.trunc(np.linspace(plot_extent[0], plot_extent[1], nlines), 1)
            latlines = HRRRData.trunc(np.linspace(plot_extent[2], plot_extent[3], nlines), 1)
            gl1 = ax.gridlines(xlocs=lonlines, ylocs=latlines, x_inline=False, rotate_labels=False)
            gl1.xlabels_bottom = True
            gl1.ylabels_left = True
            gl1.xlabel_style = {'size': 10}
            gl1.ylabel_style = {'size': 10}


        if isinstance(locs_dict,dict):
            for name, (lon,lat) in locs_dict.items():
                # Add a small delta to text labels
                delta = 0.02
                #ax.scatter([loc[0] for loc in locs],
                #           [loc[1] for loc in locs], transform=transform_crs, marker='o', color='r')
                ax.scatter(lon, lat, transform=transform_crs, marker='o', color='r')
                ax.text(lon-delta, lat-delta, name, transform=transform_crs, verticalalignment="top",
                        horizontalalignment="right")
                # for i in range(len(names)):
                #     ax.text(locs[i][0], locs[i][1],
                #             names[i], transform=transform_crs, size=15)

        ax.set_title(title, loc="left")
        # fig.tight_layout()

        fig.show()
        if save_fpath:
            fig.save(save_fpath)
        return fig, ax


if __name__ == "__main__":
    test_birdseye = True
    test_profile = False

    ### Settings
    Wlim = 360.0-110.7
    Elim = 360.0-108.7
    Nlim = 41.05
    Slim = 39.35
    crop_extent = [Wlim, Elim, Nlim, Slim]
    plot_extent = crop_extent
    init_dt = datetime.datetime(2023, 1, 30, 0, 0, 0)
    fchr = 12
    valid_dt = datetime.datetime(2023, 1, 30, 12, 0, 0)

    nprocs = 10

    # vrbl_2d = 'snod'
    vrbl_2d = "t2m"

    # Plotting options
    locs_dict = dict()
    locs_dict["Vernal"] = (-109.5287,40.4555)
    locs_dict["Duchesne"] = (-110.4029,40.1633)
    locs_dict["Ouray"] = (-109.6774,40.0891)
    locs_dict["Dragon"] = (-109.07315,39.78549)
    locs_dict["Rock Creek Ranch"] = (-110.03689,39.53777)
    locs_dict["Kings Peak"] = (-110.3728,40.7764)
    locs_dict["Roosevelt"] =  (-109.9888,40.2994)
    locs_dict["Dinosaur"] = (-109.0146,40.2436)

    if test_birdseye:
        hrrr = HRRRData(init_dt,valid_dt,crop_extent=crop_extent)
        ds_crop = hrrr.get_t2m(vrbl_2d,fchr)
        hrrr.plot_data(vrbl_2d, ds_crop.x, ds_crop.y, ds_crop, plot_extent=plot_extent, locs_dict=locs_dict,
                       do_lat_lon_lines=True, title="Test")
        pass


    #########################
    if test_profile:
        vrbl_3d = "temp"
        hrrr = HRRRData(init_dt,valid_dt,crop_extent=crop_extent)
        ds_crop = hrrr.get_3d(vrbl_3d,fchr)


    # Find closest cell to desired point
    lats = ds_crop.latitude.data
    lons = ds_crop.longitude.data

    # lat_array = np.array([[0, 10, 20], [30, 40, 50]])
    # lon_array = np.array([[0, 10, 20], [30, 40, 50]])
    # target_lat = 32
    # target_lon = 42

    target_name = "Roosevelt"
    target_lon, target_lat = locs_dict[target_name]

    closest_lat, closest_lon, closest_idx = utils.find_closest_point(lats, lons, target_lat, target_lon)
    print(f"Closest latitude: {closest_lat}, Closest longitude: {closest_lon}, Index: {closest_idx}")

