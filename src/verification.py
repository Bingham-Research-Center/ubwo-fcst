"""Verification functions

Need to import my info gain package.

TODO:
* Brier Score
* Info Gain Score
* Visualisations and dashboards
"""
import os
import datetime

import pandas as pd
import numpy as np

from src import utils
from src.hrrrdata import HRRRData

class Verification:
    def __init__(self,obs_df,fcst_df):
        # Only available for HRRR right now 
        self.obs_df = obs_df
        self.fcst_df = fcst_df

    @classmethod
    def extract_obs(cls,obs_df,col,first_n_stids=1):
        # Stations that have this observation
        stids = cls.find_stids_with_vrbl(obs_df,col)
        print(f"About to extract {col} for stids {stids[:first_n_stids]}")

        # print(obs_df)

        stid_ts_all = []
        for stid in stids[:first_n_stids]:
            vrbl_ts = obs_df[obs_df["stid"] == stid][[col]]
            pass
            #.rename(columns={stid:col})
            stid_ts_all.append(vrbl_ts)
        obs_ts_df = pd.concat(stid_ts_all,axis=1)
        return obs_ts_df

    @classmethod
    def find_stids_with_vrbl(cls,obs_df,col):
        return obs_df['stid'][obs_df[col].notnull()].unique()

    @staticmethod
    def get_stid_latlon(obs,stid):
        # obs is instance of obsdata class
        obs_lat = obs.meta_df[stid].latitude
        obs_lon = obs.meta_df[stid].longitude
        return {stid:(obs_lat,obs_lon)}


    @staticmethod
    def extract_nwp(grib_q,ds_q,fxx,loc_lat_lon_dict,init_dt):
        ts_df = HRRRData.multiple_station_timeseries(fxx, loc_lat_lon_dict, init_dt, grib_q, ds_q, )
        return ts_df

    def subsample_timeseries(self,_obs_df, _fcst_df):
        obs_sub_df = utils.select_nearest_neighbours(_obs_df, _fcst_df)
        return obs_sub_df

    @staticmethod
    def calculate_rmse(observed, predicted):
        observed = np.array(observed)
        predicted = np.array(predicted)
        return np.sqrt(((predicted - observed) ** 2).mean())

    @staticmethod
    def calculate_mae(observed, predicted):
        observed = np.array(observed)
        predicted = np.array(predicted)
        return np.abs(predicted - observed).mean()

    @staticmethod
    def calculate_mbe(observed, predicted):
        observed = np.array(observed)
        predicted = np.array(predicted)
        return (predicted - observed).mean()

    @staticmethod
    def assign_custom_daily_period(dt):
        if dt.hour < 7:
            return dt.date() - pd.Timedelta(days=1)
        else:
            return dt.date()

    @classmethod
    def daily_evaluation(cls,df_forecast, df_observed, col, metric):
        metric_funcs = {
                    "RMSE":cls.calculate_rmse,
                    "MAE":cls.calculate_mae,
                    "MBE":cls.calculate_mbe,
                    }

        df_forecast = df_forecast.copy()
        df_observed = df_observed.copy()

        df_forecast['Adjusted_Date'] = df_forecast.index.map(cls.assign_custom_daily_period)
        df_observed['Adjusted_Date'] = df_observed.index.map(cls.assign_custom_daily_period)

        grouped_forecast = df_forecast.groupby('Adjusted_Date')
        grouped_observed = df_observed.groupby('Adjusted_Date')

        results = []
        for adjusted_date, forecast_group in grouped_forecast:
            if adjusted_date in grouped_observed.groups:
                observed_group = grouped_observed.get_group(adjusted_date)
                metric_score = metric_funcs[metric](observed_group[col], forecast_group[col])
                results.append({'Date': adjusted_date, 'Score': metric_score})

        return pd.DataFrame(results)

    @staticmethod
    def concatenate_metrics(dfs, metric_names):
        # Assuming dfs is a list of DataFrames and metric_names is a list of column names
        for df, name in zip(dfs, metric_names):
            df.rename(columns={'Score': name}, inplace=True)

        # Merge the DataFrames on the 'Date' column
        merged_df = pd.concat(dfs, axis=1)
        # Removing duplicate 'Date' columns
        merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]

        return merged_df

    def do_det_metrics(self,fcst_df,obs_df,col):

        rmse = self.daily_evaluation(fcst_df,obs_df,col,"RMSE")

