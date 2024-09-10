"""Predicting Uinta Basin weather conditions.
"""

import os
import datetime

import pandas as pd
import numpy as np
import pytz

from src.obsdata import ObsData
import src.derive as derive
from src.lookups import elevations, hrrr_sfc_vrbls, lat_lon, obs_vars
import src.utils as utils
from src.hrrrdata import HRRRData
import src.plotting as plotting
from src.verification import Verification

### CONSTANTS

# Set initialisation time to be most recent

# init_dt = HRRRData.determine_latest_hrrr(long_only=True)
init_dt = datetime.datetime(2023,11,2,0,0,0)

init_hb = utils.herbie_from_datetime(init_dt)
init_pd = utils.pd_from_datetime(init_dt)

# obs_end_dt = ObsData.get_latest_hour()
obs_end_dt = init_dt

# Fetch last 72 hours of observations in case we want to plot trends, verification, etc
# Will end up throwing away most when concatenating to the archive
obs_start_dt = obs_end_dt - datetime.timedelta(hours=72)

# This should be dynamically based on date, and also can move to website repo
fig_root = "../figures"
utils.try_create(fig_root)

plot_obs_profile = True
plot_nwp_profile = True
plot_nwp_sfc = True
plot_nwp_500 = False
plot_meteograms = True
plot_gefs_probs = False
plot_verification = True
plot_ozone = False
plot_dev_models = False
do_database = False

# This needs to be somewhere backed up
data_root = "./local_data"
if not os.path.exists(data_root):
    os.makedirs(data_root)
obs_fpath = os.path.join(".", data_root, "basin_obs.h5")
# For now, not saving metadata separately as it's different every time
# metadata_fpath = os.path.join(".", data_root, "basin_ob_metadata.h5")
lapserate_fpath = os.path.join(".", data_root, "lapserates.h5")

force_do = False
if True in (plot_obs_profile,):
    do_obs = True
else:
    do_obs = False

if __name__ == "__main__":
    ## TODO: parallelise slow sections

    ### GET OBS DATA ###

    if do_obs:
        # Get all recent data - need to make this "latest"?
        obs = ObsData(obs_start_dt, obs_end_dt, tests=False)
        obs_df = obs.df
        print("Obs loaded")
        # TODO: plot distribution to scan for errors in each variable
        # Save this to disc and check as figures
        # for o in obs_vars:
            # How to get nearest to this time?
            # Generalise logic in get_profile_df to subsetting any df
            # plotting.plot_distribution(obs_df[obs_end_dt])

    ####################
    # For 2-by-2 plot of obs pseudo-lapse rate, analysed current HRRR profile, &
    # forecast at 6 and 12 hours

    if plot_obs_profile:
        # Plot profile and compute lapse rate somehow
        profile_dt = obs_end_dt - datetime.timedelta(minutes=30)
        profile_pdts = utils.pd_from_datetime(profile_dt)
        profile_df = obs.get_profile_df(dt=profile_pdts)
        print("Obs profile dataframe loaded")

        # Fitting lapse rate via least squares
        lapse_rate = derive.calculate_lapse_rate(profile_df["elevation"], profile_df["drybulb"])
        lapserate_str = f"Calculated Lapse Rate: {lapse_rate:.2f} °C/km"

        pass

        # Add to archive of lapse rates
        pass

        fname = utils.create_image_fname(dt=profile_dt,inittime=None,plot_type="profile_T",model="obs")
        fpath = os.path.join(fig_root,fname)
        title = utils.create_obs_title("Vertical profile of dry-bulb temperature",profile_dt,
                                       subtitle=lapserate_str)
        plotting.plot_profile(profile_df["drybulb"], profile_df["elevation"],
                              "obs", plot_levels=elevations, save=fpath, title=title)
        print("Obs profile plotted")




        # Could calculate from WRF too, and compare
        # Categorise into inversion strengths so we can verify hit/miss
        # Good probabilistic product to generate from RRFS and GEFS

    if plot_nwp_profile:
        # Two archives: NWP data (cropped and variables of interest) and obs data (dataframe)
        # Get sounding
        # hrrr = HRRRData()

        # Ouray 40.0891° N, 109.6774° W
        lon, lat = (360 - 109.6774, 40.0891)

        # Analysis
        # for fx in (0,12,24,36,48):
        for fx in (0,6,12,18):
            dt = init_dt + datetime.timedelta(hours=fx)
            # Eventually do this in parallel for big 3D fields:
            ds_Z = HRRRData.get_cropped_data(init_hb,fx,':HGT:.*hybrid') # / 1000 # km
            ds_T = HRRRData.get_cropped_data(init_hb,fx,':TMP:.*hybrid') # - 273.15 # Celsius
            profile_df = HRRRData.get_profile_df(ds_T,ds_Z,lat,lon,max_height=4000)

            # Find subset for only points below the highest ob if using >4000 for max_height
            lapse_rate = derive.calculate_lapse_rate(profile_df["height"], profile_df["temp"])
            lapserate_str = f"Calculated Lapse Rate: {lapse_rate:.2f} °C/km"

            fname = utils.create_image_fname(dt=dt, inittime=init_dt, plot_type="profile_T",
                                                  model="HRRR")
            fpath = os.path.join(fig_root, fname)

            title = utils.create_nwp_title("Vertical profile of potential temperature", "HRRR",
                                       init_dt, dt)

            print(f"Plotting HRRR fcst hour {fx} to", fpath)
            plotting.plot_profile(profile_df["temp"], profile_df["height"],
                                  "model", plot_levels=elevations,save=fpath,
                                  title=title)

            print("Visualising HRRR lapse-rate data as chart for Basin")
            # Compute NWP lapse rate geographically for Basin to see spatial variation?
            # data_arr = derive.compute_lapserate_geographically(ds_Z,ds_T)


    ########################
    # Surface plots
    ########################
    # hrrr_sfc_vrbls
    plot_sfc_vrbls = ["gust","blh","dswrf"]
    if plot_nwp_sfc:
        for fx in (0,6,12,18):
            dt = init_dt + datetime.timedelta(hours=fx)
            ds_sfc = HRRRData.get_cropped_data(init_hb,fx,':surface:')
            for vrbl in plot_sfc_vrbls:
                fname = utils.create_image_fname(dt=dt, inittime=init_dt, plot_type=vrbl,
                                                 model="HRRR")
                fpath = os.path.join(fig_root, fname)
                print("About to plot sfc map, saving to ",fpath)
                # clvs = np.linspace(0.5, ds_sfc[vrbl].max(), num=10)
                label = utils.reverse_lookup(hrrr_sfc_vrbls,vrbl)

                fig,ax = plotting.surface_plot(ds_sfc, vrbl, fchr=0, label=label, save=fpath,
                                                        # vlim=(2,None),
                                                        # levels=clvs,
                                               )

    plot_500_vrbls = ["gh",]
    if plot_nwp_500:
        # for fx in (0, 6, 12):
        for fx in (12,):
            dt = init_dt + datetime.timedelta(hours=fx)
            ds_sfc = HRRRData.get_cropped_data(init_hb, fx, ':500 mb:',product="prs")
            for vrbl in plot_500_vrbls:
                fname = utils.create_image_fname(dt=dt, inittime=init_dt, plot_type=vrbl+"-500hPa",
                                                 model="HRRR")
                fpath = os.path.join(fig_root, fname)
                print("About to plot gust analysis, saving to ", fpath)
                # clvs = np.linspace(0.5, ds_sfc[vrbl].max(), num=10)
                label = utils.reverse_lookup(hrrr_sfc_vrbls, vrbl)

                fig, ax = plotting.surface_plot(ds_sfc, vrbl, fchr=0, label=label, save=fpath,
                                                # vlim=(1, None),
                                                # levels=clvs,
                                                plot_type="contour"
                                                )

    if plot_meteograms:
        locations = ["Ouray","Vernal","Kings Peak"]

        for location in locations:
            # if long_only is False:
                # meteo_init = HRRRData.determine_latest_hrrr(long_only=True)
            meteo_init = init_dt
            print("Doing meteogram starting", utils.pd_from_datetime(meteo_init))
            # fcst_hrs = list(range(0,49,1))
            fcst_hrs = list(range(0,19,2))
            valid_times = [meteo_init + datetime.timedelta(hours=h) for h in fcst_hrs]

            lon, lat = lat_lon[location]

            # Can estimate max incoming solar for these times
            df_max = derive.max_solar_radiation(valid_times, lat, lon, elevations[location])
            pass

            # Solar incoming forecast
            solar_ts = HRRRData.generate_timeseries(fcst_hrs, meteo_init, "DSWRF","dswrf", lat, lon)
            df = pd.DataFrame({"dswrf":solar_ts},index=valid_times)

            fname = utils.create_meteogram_fname(inittime=meteo_init, loc=location, vrbl="DSWRF",model="HRRR")
            fpath = os.path.join(fig_root, fname)
            title = utils.create_meteogram_title(
                        "Incoming short-wave (W/m2)", meteo_init,"HRRR", location)


            # df_max = pd.DataFrame({"Max DSWRF":max_dswrf}, index=valid_times)
            plotting.plot_meteogram(df,"dswrf",save=fpath, title=title,
                                            second_df=df_max, second_col="dswrf_max")

            # snow depth forecast
            snow_ts = HRRRData.generate_timeseries(fcst_hrs, meteo_init, "SNOD","sde", lat, lon)
            df = pd.DataFrame({"sde":snow_ts},index=valid_times)

            fname = utils.create_meteogram_fname(inittime=meteo_init, loc=location, vrbl="SNOD",model="HRRR")
            fpath = os.path.join(fig_root, fname)
            title = utils.create_meteogram_title("Snow depth (m)", meteo_init,"HRRR", location)
            plotting.plot_meteogram(df,"sde",save=fpath, title=title,)

    if plot_gefs_probs:
        # Generate probabilities from the GEFS data
        # Download each cropped area? What is disc usage?

        # Time series

        # NWP Profiles

        # Snowfall forecasts - possibility theory correction

        # HRRR: RMSE minimisation

        # Can we use lapse-rate in HRRR, GEFS, and RFSS to correlate
        # with observed inversion strength?
        pass

    if plot_verification:

        # Draw last 24 hours of HRRR runs compared with observed values
        # Compute RMSE and DKL for deterministic/probabilistic

        # First, plot recent deterministic performance of HRRR

        # Do max number of hours for the init time minus this n hours
        # We want 48 hours

        # TESTING
        verif_end_dt = datetime.datetime(2023, 12, 2, 0, 0, 0,tzinfo=pytz.UTC)

        our_nhr = 48
        verif_start_dt = verif_end_dt - datetime.timedelta(hours=our_nhr)

        pass

        # Actual length of NWP forecast
        nhr = HRRRData.get_nhr(verif_start_dt)

        # Only do every sixth hour - for testing
        freq = 1

        # Will this work for naive timezone?
        # verif_start_dt = datetime.datetime(2023,12,2,0,0,0, tzinfo=pytz.UTC)

        obs_start_dt = verif_start_dt
        obs_end_dt = verif_start_dt + datetime.timedelta(hours=nhr + 1)

        fxx = list(range(0, nhr + 1, freq))

        # Create obs for this period
        # Testing with first 10 to speed up
        obs = ObsData(obs_start_dt, obs_end_dt, tests=10)
        obs_df = obs.df

        # In operations, use loaded obs at top of script - make sure it goes back nhr hours!
        print("Now extracting our observation of interest from whole obs dataframe")
        # Use staticmethods to get smaller dataframe plus nwp dataframe
        obs_sub_df = Verification.extract_obs(obs_df,"solar_radiation")

        # Lat/lons for a random station
        stids = Verification.find_stids_with_vrbl(obs_df,"solar_radiation")

        verif_vrbls = {
            "solar_radiation": ("DSWRF", "dswrf"),
            "air_temp":(":TMP:2m","t2m")
            # "snow_depth","2-m drybulb temperature","solar incoming",
            # "cloud cover", "gust or wind",
        }

        # Also do Vernal t2m
        stid_list = list(stids[0]) + ["KVEL",]
        for stid in stid_list:
            if stid == "KVEL":
                verif_vrbls = {
                    "air_temp": (":TMP:2m", "t2m")
                    # "snow_depth","2-m drybulb temperature","solar incoming",
                    # "cloud cover", "gust or wind",
                }
            else:
                verif_vrbls = {"solar_radiation": ("DSWRF", "dswrf")}

            latlon_dict = Verification.get_stid_latlon(obs,stid)
            fcst_sub_df = Verification.extract_nwp("DSWRF","dswrf",fxx,latlon_dict,verif_start_dt
                                                   ).rename(columns={stid:"solar_radiation"})

            fname = utils.create_meteogram_fname(inittime=verif_start_dt, loc=stid, vrbl="DSWRF",model="HRRR_verif")
            fpath = os.path.join(fig_root, fname)

            V = Verification(fcst_df=fcst_sub_df,obs_df=obs_sub_df)


            for v, (grib_q,ds_q) in verif_vrbls.items():
                if v == "solar_radiation":
                    title_desc = f"Downwelling Short-wave observed versus HRRR forecast"
                elif v == "air_temp":
                    title_desc = "Drybulb 2-m temperature observed versus HRRR forecast"


                # Plot the two

                # Subsample the observations to the NWP times
                subsample_obs_df = utils.select_nearest_neighbours(obs_sub_df,fcst_sub_df)

                rmse_scores = V.daily_evaluation(fcst_sub_df, subsample_obs_df, v, "RMSE")
                mae_scores = V.daily_evaluation(fcst_sub_df, subsample_obs_df, v,"MAE")
                mbe_scores = V.daily_evaluation(fcst_sub_df, subsample_obs_df, v,"MBE")

                metrics_dfs = [rmse_scores, mae_scores, mbe_scores]
                metric_names = ['RMSE', 'MAE', 'MBE']
                concatenated_metrics_df = V.concatenate_metrics(metrics_dfs, metric_names)
                print(concatenated_metrics_df)

                met_title = utils.create_meteogram_title(title_desc,
                                                         verif_start_dt, "Obs/HRRR", stid)
                plotting.plot_meteogram(fcst_sub_df, "solar_radiation", title=met_title, second_df=obs_sub_df,
                                        second_col="solar_radiation", save=fpath)



    if plot_ozone:
        # Show ozone for last 24-48 hours on sites
        # Do we have a way to predict this?
        # Dev models to evaluate probabilistic risk
        # Could be % of being in ~5 categories (no inversion to strong)
        pass

    if plot_dev_models:
        # A place to deploy AI-based and statistical forecasts
        pass

    ########################
    # Finally, we need to archive the images
    # Then copy ones for the website into the phpstorm folder
    # Uodate databases for NWP, dev model, time series, obs...

    # LOAD ARCHIVE & UPDATE & RE-WRITE TO DISC
    if do_database:
        old_combined_df = pd.read_hdf(obs_fpath)
        combined_df = ObsData.combine_dataframes(old_combined_df, obs_df)
        combined_df.to_hdf(obs_fpath, key='combined_df', mode='w')
        # We can ignore meta_df for now - don't bother saving to disc

        # FOR HRRR DATA:
        # Saving to disc allows lagged ensemble formation
        # First, compute ensembles (see above) using old data and new loaded
        # Save some prob data to disc for faster analysis

        # Then compute time series and charts from prob data

        # Then read to verify

        # Be ready to backfill for HRRR variables we introduce later!
    # TODO: Why is GEFS not working for 0.25 degree? RRFS?


    # Finally, copy all images in today's folder to where the website repo can see it
    print("Done.")