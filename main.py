"""Predicting Uinta Basin weather conditions.

Scope:
    * Downloading data (HRRRData, ObsData, GEFSData)
        * Surface observations
        * Satellite
        * Other sensors (?)
        * NWP data
            * HRRR
            * (GEFS, GEFS/R2)
    * Data extraction
        * Point time-series from NWP
        * Interpolation scheme for obs to create "contours"
    * Post-processing
        * Possibilistic PP (Le Carrer)
        * Lagged ensembles
        * Getting obs/NWP time/space aligned
    * Statistical analysis
        * Verification
        * Dataset characterists (obs, HRRR)
    * Maintaining database
        * sqlite3 w/ pandas for observations
        * numpy arrays for extracted HRRR fields
"""

import os

pass

