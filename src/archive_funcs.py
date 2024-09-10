"""Functions to help with loading/saving archive data.

Obs will be in .h5 format of dataframe.
NWP data will be also as long as h5netcdf is installed
"""
import os

from src import utils

def load_obs_df(fpath):
    """T

    This should be a predictable fpath
    """

    pass

def generate_obsh5_fpath(root_dir,df=None,start_date=None,end_date=None):
    if start_date is None:
        start_date = min(df.index)
    if end_date is None:
        end_date = max(df.index)
    fname = f"basin_obs_{start_date:%Y%m%d%H}Z-{end_date:%Y%m%d%H}Z.h5"
    fpath = os.path.join(root_dir,fname)
    return fpath