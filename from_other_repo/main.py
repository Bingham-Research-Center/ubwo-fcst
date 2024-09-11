"""Download archives of weather observations in-situ and satellite
(SynopticPy, goes2go)
Download live forecasts from various NWP models (Herbie)
Extract point time series and other subsets/slices (xarray, MetPy)
Visualise in 2-D to see performance for Basin (Jupyter, MetPy)
Create SQL database; sync between locations (sqlite3, rsync)
Functions for extracting data into Pandas DataFrames (pandas)
Create ML models/forecasts to add to NWP data (pytorch)
Visualisations such as heatmap/scorecard, live time-series plots (matplotlib)
Website to show visualisations (plot.ly, other HTML languages w/ Python)
"""

import torch

from old_scripts import obsdata

# To test pytorch GPU M1/2 acceleration for Macs
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

# Download obs (satellite and in-situ)
# Load obs
obsdata.plot_available_stations(radius="KVEL,50", recent=120, )
# def plot_available_stations(radius, recent, vars="temperature",):

# Download NWP


# Extract data we want


# Visualise some data as test


# Create SQL database - convert from Pandas
# Need to update if doesn't exist so we don't keep calling Synoptic API


# Create or load ML model with PyTorch



# Generate forecast


# Verify with info gain


# Now visualise forecast time series with verification and observations


# Put onto website