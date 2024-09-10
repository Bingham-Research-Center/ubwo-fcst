"""
Lookup tables and dictionaries for geography, variables, etc
"""

lat_lon = {
        "Vernal": (40.4555,-109.5287),
        "Duchesne":(40.1633,-110.4029),
        "Ouray":(40.0891,-109.6774),
        "Dragon":(39.78549,-109.07315),
        "Rock Creek Ranch":(39.53777,-110.03689),
        "Kings Peak":(40.7764,-110.3728),
}

elevations = {
        "Ouray":1425,
        "Vernal":1622,
        "Split Mtn":2294,
        "Kings Pk":4123,
}

hrrr_sfc_vrbls = {
                "Accum snowfall (m)":"acsno",
                "Temperature (K)":"t",
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

obs_vars = ["wind_speed", "wind_direction", "air_temp", "dew_point_temperature",
                "pressure", "snow_depth", "solar_radiation",
                "relative_humidity", "wind_gust", "altimeter", "soil_temp",
                "sea_level_pressure", "snow_accum", "road_temp",
                "cloud_layer_1_code", "cloud_layer_2_code",
                "cloud_layer_3_code", "cloud_low_symbol",
                "cloud_mid_symbol", "cloud_high_symbol",
                "sonic_wind_direction", "peak_wind_speed",
                "ceiling", "sonic_wind_speed", "soil_temp_ir",
                "snow_smoothed", "snow_accum_manual", "snow_water_equiv",
                "precipitable_water_vapor", "net_radiation_sw",
                "sonic_air_temp", "sonic_vertical_vel",
                "vertical_heat_flux", "outgoing_radiation_sw",
                "PM_25_concentration", "ozone_concentration",
                "derived_aerosol_boundary_layer_depth",
                "NOx_concentration", "PM_10_concentration",
                "visibility_code", "cloud_layer_1", "cloud_layer_2",
                "cloud_layer_3", "wet_bulb_temperature",
                # Boring but important ones:

                ]

