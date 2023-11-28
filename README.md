# Uintah Basin Winter Ozone prediction system 
## Version 0.0.1, purely experimental, Nov 2023
### Lead: John R. Lawson, Bingham Research Center, Utah State Univ., Vernal, Utah, United States 
#### john.lawson@usu.edu

The goal is to **provide guidance for predicting high levels of winter ozone in the Uintah Basin**. 

The package prerequisites are metpy (and dependencies), plus my ``infogain`` package for verification.

This prediction system ingests
* Surface meteorology observations (MesoWest; Synoptic)
* Air-quality observations (MesoWest; Synoptic; internal)
* Numerical Weather Prediction data (HRRR, GEFS)
* Climatology data (GEFS/R2)

The aims are:
* Ongoing evaluation against obs for numerous locations 
* Lagged ensemble in HRRR 
* Building archives for later full verification
* Traditional weather maps for public to use 

Future goals may include:
* Machine-learning 
* More NWP data for superensemble
* Possibility theory to account for uncertainty in probabilities
* Better visualisation
* Website or dashboard for forecast guidance
* LLMs and fuzzy logic to convert numbers into good communication 
* RRFS (successor of HRRR) incorporation 
* Technical documentation of generating products and how to interpret them