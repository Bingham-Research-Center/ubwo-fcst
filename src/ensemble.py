class Ensemble:
    def __init__(self,ens_dict):
        """Create ensembles from multiple runs or models.

        Will start by using RRFS and GEFS

        Could do lagged ensembles with HRRR too

        The dictionary should be
        ensemble[member] = {"fpath":fpath, "control":False, data_xr=xr.DataArray}

        The data can be provided, but will otherwise be loaded is fpath is None.
        """
        self.ens_dict = ens_dict

    def compute_prob_exceedance(self):
        pass

    def compute_ensemble_mean(self):
        pass

