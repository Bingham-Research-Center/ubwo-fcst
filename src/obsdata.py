import datetime
import io
from urllib.request import urlopen, Request
import sqlite3
import pickle
import itertools

import metpy.calc
from metpy.units import units
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import pandas as pd

from synoptic.plots import map_metadata
import synoptic.services as ss

import src.utils as utils
from src.lookups import obs_vars

class ObsData:
    def __init__(self,start_date,end_date,recent=12*60*60, radius="UCL21,50",
                    tests=None):
        """Download, process, and archive observation data.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.recent = recent
        self.radius = radius

        self.vrbls = self.return_variable_list()
        self.meta_df = self.create_metadata_df()
        # self.elevs = self.get_elevations()
        pass
        self.stids = list(self.meta_df.columns.unique())

        if (tests is not False) or (tests is not None):
            # Limit number of stations to speed up tests
            self.stids = self.stids[:tests]

        self.df = self.create_df()

    def create_metadata_df(self):
        df_meta = ss.stations_metadata(radius=self.radius, recent=self.recent)
        return df_meta

    def get_elevations(self):
        elevs = []
        for stid in self.stids:
            elevs.append(self.meta_df[stid].loc["ELEVATION"]*0.304)
        return elevs

    def create_df(self):
        # Make dataframe of data for this period
        df_list = []

        for stid in self.stids:
            print("Loading data for station", stid)
            try:
                _df = ss.stations_timeseries(stid=stid, start=self.start_date, end=self.end_date, vars=self.vrbls,
                                                verbose=False,
                                             qc_checks='all',
                                             # qc_checks='synopticlabs',
                                             )
            except AssertionError:
                # raise
                print("Skipping", stid)
                # continue

            # Assign this metadata to the variables df for ease of access
            # TODO - maybe remove?
            stid_lat = self.meta_df[stid].loc["latitude"]  # .values.squeeze()
            stid_lon = self.meta_df[stid].loc["longitude"]  # .values.squeeze()
            elev = self.meta_df[stid].loc["ELEVATION"] * 0.304  # .values.squeeze()*0.304

            ############# Uncomment below for density calculation: ##########
            # if ("pressure" in _df.columns) and ("air_temp" in _df.columns) and ("dew_point_temperature" in _df.columns):
            #     rho = get_density(_df["pressure"],_df["air_temp"]+273.15,_df["dew_point_temperature"]+273.15)
            #     _df = _df.assign(air_density=rho.values)
            _df = _df.assign(stid=stid, elevation=elev, latitude=stid_lat, longitude=stid_lon)

            # Check certain data for QC
            # _df = self.check_constant_temps(_df)

            pd.to_datetime(_df.index.strftime('%Y-%m-%dT%H:%M:%SZ'))
            df_list.append(_df)

        df = pd.concat(df_list, axis=0, ignore_index=False)

        # Reduce memory use
        col64 = [df.columns[i] for i in range(len(list(df.columns))) if (df.dtypes.iloc[i] == np.float64)]
        change_dict = {c: np.float32 for c in col64}
        df = df.astype(change_dict)
        return df

    @classmethod
    def filter_temperature_outliers(cls, df, elev_bins, num_std_dev=2):
        filtered_dfs = []

        for min_elev, max_elev in itertools.zip_longest(elev_bins[:-1], elev_bins[1:]):
            # Select the subset of the DataFrame within the elevation bin
            pass
            sub_df = df[(df["elevation"] > min_elev) & (df["elevation"] <= max_elev)]

            # Calculate mean and standard deviation of air_temp in this bin
            mean_temp = sub_df["drybulb"].mean()
            std_dev_temp = sub_df["drybulb"].std()

            # Define the acceptable range for air_temp
            lower_bound = mean_temp - num_std_dev * std_dev_temp
            upper_bound = mean_temp + num_std_dev * std_dev_temp

            # Filter out rows where air_temp is outside the acceptable range
            filtered_sub_df = sub_df[(sub_df["drybulb"] > lower_bound) & (sub_df["drybulb"] < upper_bound)]

            # Append the filtered sub-dataframe to the list
            filtered_dfs.append(filtered_sub_df)

        # Concatenate all the filtered sub-dataframes
        filtered_df = pd.concat(filtered_dfs)
        pass
        return filtered_df

    @staticmethod
    def return_variable_list():
        # Variables we care about in operations
        # from
        return obs_vars

    @staticmethod
    def get_latest_hour():
        current_time = datetime.datetime.utcnow()
        # If past 20 min, use latest hour
        if current_time.minute > 20:
            latest_hr_dt = current_time.replace(minute=0, second=0, microsecond=0)
        # Else go back 1 hour + minutes
        else:
            latest_hr_dt = current_time.replace(minute=0, second=0, microsecond=0) - datetime.timedelta(hours=1)
        return latest_hr_dt

    def get_profile_df(self, dt:pd.Timestamp, temp_type="drybulb",tolerance=5):
        profile_data = []
        df = self.df

        for stid in self.stids:
            elev = self.meta_df[stid].loc["ELEVATION"]*0.304
            # print(stid)
            # Time window is just for memory efficiency - tolerance is set later
            sub_df = df[
                (df['stid'] == stid) &
                (df.index <= dt + pd.Timedelta(minutes=tolerance)) &
                (df.index >= dt - pd.Timedelta(minutes=tolerance))
                ]

            # print(sub_df["air_temp"])

            # i = sub_df.index.get_indexer([prof_dt,],method='nearest')
            # print(i)
            if len(sub_df) == 0:
                print("no measurement in this range.")
                continue

            pass

            # temp = sub_df["air_temp"].iloc[i]
            temp = utils.get_closest_non_nan(sub_df, "air_temp", dt, pd.Timedelta(f'{tolerance} minutes'))
            # print(temp)

            elev = utils.get_closest_non_nan(sub_df, "elevation", dt, pd.Timedelta(f'{tolerance}minutes'))
            # print(elev)

            if temp_type == "drybulb":
                if (not np.isnan(temp)) and (not np.isnan(elev)):
                    profile_data.append([elev, temp])

            elif temp_type == "theta":
                raise Exception
                # p = sub_df["pressure"].iloc[i]
                p = utils.get_closest_non_nan(sub_df, "pressure", dt, pd.Timedelta('30 minutes'))

                # Can we estimate p from elev?

                if (not np.isnan(temp)) and (not np.isnan(p)):
                    theta = metpy.calc.potential_temperature(p * units("pascal"), temp * units("celsius")).magnitude
                    # print(theta)

                    # print(elev)

                    profile_data.append([elev, theta])
                    # print("Added theta", theta, "at", elev, )
                # print("Skipping due to missing in something:", temp, p)
        t_df = pd.DataFrame(profile_data, columns=["elevation", temp_type])
        elev_bins = [1000,1500,2000,2500,3000,3500,4000]
        t_df = self.filter_temperature_outliers(t_df,elev_bins)
        return t_df

    @classmethod
    def combine_dataframes(cls, df_old, df_new):
        # Make index (datetime) a column (date_time)
        df_old = df_old.reset_index()
        df_new = df_new.reset_index()

        # Combine the dataframes
        combined_df = pd.concat([df_old, df_new])

        # Identify columns that don't have NaNs and aren't dicts for duplicate checking
        hashable_cols = [col for col in combined_df.columns
                         if not combined_df[col].isna().any()
                         and not isinstance(combined_df[col].iloc[0], dict)]

        # Drop duplicates considering only hashable columns
        combined_df = combined_df.drop_duplicates(subset=hashable_cols)

        # Set the 'datetime' column back as the index
        combined_df.set_index('date_time', inplace=True)

        return combined_df