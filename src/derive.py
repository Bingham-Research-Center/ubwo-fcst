import math
import datetime

import pandas as pd
import pvlib
import pytz

import numpy as np
from scipy import stats

def calculate_lapse_rate(elevation, temperature, std_dev_threshold=2):
    """
    Calculate the vertical lapse rate using least squares linear regression,
    ignoring outliers beyond a certain standard deviation threshold.

    :param elevation: List or array of elevations (in meters).
    :param temperature: Corresponding list or array of temperatures (in °C).
    :param std_dev_threshold: Number of standard deviations for outlier detection.
    :return: Lapse rate in °C per kilometer.
    """
    # Convert to numpy arrays
    elevation = np.array(elevation)
    temperature = np.array(temperature)

    # Filter out outliers
    mean_temp = np.mean(temperature)
    std_dev_temp = np.std(temperature)
    mask = np.abs(temperature - mean_temp) < std_dev_threshold * std_dev_temp
    filtered_elevation = elevation[mask]
    filtered_temperature = temperature[mask]

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_elevation, filtered_temperature)

    # Convert slope to °C per kilometer
    lapse_rate = slope * 1000  # since elevation is in meters

    # Return negative as that's the way people do it
    return -lapse_rate

def solar_declination(day_of_year):
    """ Calculate solar declination from day of year """
    return 23.45 * math.sin(math.radians((360/365) * (day_of_year - 81)))

def equation_of_time(day_of_year):
    """ Approximate the equation of time for a given day of the year """
    B = (360/365) * (day_of_year - 81)
    EOT = 9.87 * math.sin(math.radians(2*B)) - 7.53 * math.cos(math.radians(B)) - 1.5 * math.sin(math.radians(B))
    return EOT

def solar_elevation(latitude, declination, hour_angle):
    """ Calculate solar elevation angle """
    return math.degrees(math.asin(math.sin(math.radians(latitude)) * math.sin(math.radians(declination)) +
                                  math.cos(math.radians(latitude)) * math.cos(math.radians(declination)) * math.cos(math.radians(hour_angle))))

def hour_angle(utc_time, longitude):
    """ Calculate solar hour angle with adjustments for longitude and equation of time """
    day_of_year = utc_time.timetuple().tm_yday
    EOT = equation_of_time(day_of_year)

    # Convert longitude to time (1 degree = 4 minutes)
    longitude_time = longitude * 4 / 60

    # Local Solar Time
    lst = utc_time + datetime.timedelta(minutes=EOT) + datetime.timedelta(hours=longitude_time)

    solar_noon = datetime.datetime(lst.year, lst.month, lst.day, 12, 0, 0)
    time_difference = (lst - solar_noon).total_seconds() / 3600
    return 15 * time_difference  # 15 degrees per hour

def max_solar_radiation(datetimes, latitude, longitude, elevation):
    solar_constant = 1361  # W/m²
    results = []

    for dt in datetimes:
        ha = hour_angle(dt, longitude)
        day_of_year = dt.timetuple().tm_yday
        declination = solar_declination(day_of_year)
        elevation_angle = solar_elevation(latitude, declination, ha)

        if elevation_angle > 0:  # Sun is above the horizon
            estimated_radiation = solar_constant * math.cos(math.radians(90 - elevation_angle)) * 0.75
            elevation_adjustment = 12 * (elevation / 1000)
            estimated_radiation += elevation_adjustment
        else:
            estimated_radiation = 0  # No solar radiation if the sun is below the horizon

        results.append(estimated_radiation)

    result_df = pd.DataFrame({"dswrf_max":results}, index=datetimes)
    return result_df

def compute_lapserate_geographically(ds_Z,ds_T,lv="depth"):
    """Compute lapse-rate in lowest X.x km for plotting purposes.

    level could be (not sure) but also "depth" to do average (over what
    levels, does this need to be defaulted? - see MM paper)
    """
    """
    Calculate the vertical lapse rate using least squares linear regression,
    ignoring outliers beyond a certain standard deviation threshold.

    :param elevation: List or array of elevations (in meters).
    :param temperature: Corresponding list or array of temperatures (in °C).
    :param std_dev_threshold: Number of standard deviations for outlier detection.
    :return: Lapse rate in °C per kilometer.
    """
    # Convert to numpy arrays
    # elevation = ds_Z
    # temperature = np.array(ds_T)
    return