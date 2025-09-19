import xarray
import pandas
# import rioxarray
import numpy as np
import rasterio.transform

from dataloader.utils import cache
import constants


# Core logic
@cache(use_cache=constants.use_cache)
def retrieve_features(glacier, year, retrieve_corrector_predictors=False, retrieve_facies=False):
    # always load all available smb, t, p and outlines
    # if retrieve_corrector_predictors is True, load and normalise elev, elev_stddev and monthly averaged t and p
    # if retrieve_facies is True and facies are available, load facies, otherwise prepare a placeholder

    # return a PyTree with a structure based on the smb records (annual vs winter+summer)
    pass


# Retrievers
# retrieve total_smb, point_smb, ...

@cache(use_cache=constants.use_cache)
def retrieve_outlines(glacier, year):
    outline_model = retrieve_outline_model(glacier)
    outlines = outline_model.sel(year=year)
    return outlines


@cache(use_cache=constants.use_cache)
def retrieve_outline_model(glacier):
    pass
    

# Conversions
def downscale_temperature(temperature, elevation, orography, temperature_lapse_rate=constants.temperature_lapse_rate):
    temperature_downscaled = temperature + temperature_lapse_rate * (elevation - orography)
    return temperature_downscaled


def convert_points_to_rowcol():
    pass
