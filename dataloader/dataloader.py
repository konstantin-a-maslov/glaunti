import xarray
import pandas
import rioxarray
import numpy as np
import rasterio.transform
import os

from dataloader.utils import cache
import constants


# Core logic
def retrieve_dataset_index():
    pass


def traverse_folds(dataset_index):
    pass


def traverse_glaciers(fold):
    pass


@cache(use_cache=constants.use_cache)
def retrieve_features(glacier, year, retrieve_corrector_predictors=False, retrieve_facies=False):
    if retrieve_facies and not retrieve_corrector_predictors:
        raise ValueError("If facies are fetched, whole corrector input fields should be constructed with retrieve_corrector_predictors=True!")
        
    # always load all available smb records, t, p and outlines
    y = retrieve_smb_records(glacier, year)
    begin_date, midseason_date, end_date = extract_season_dates(y)
    y = split_smb_records_by_seasons(y, begin_date, midseason_date, end_date)
    
    x = {
        "outlines": retrieve_outlines(glacier, year),
    }

    temperature = None
    precipitation = None
    
    # separate climate time series by season if seasons are measured
    if not pandas.isna(midseason_date):
        summer = slice()
        winter = slice()
        
        x["summer"] = {
            "temperature": temperature.sel(date=summer),
            "precipitation": precipitation.sel(date=summer),
        }
        x["winter"] = {
            "temperature": temperature.sel(date=winter),
            "precipitation": precipitation.sel(date=winter),
        }
    else:
        x["annual"] = {
            "temperature": temperature,
            "precipitation": precipitation,
        }
    
    # if retrieve_corrector_predictors is True, load and normalise elev, elev_stddev and monthly averaged t and p
    if retrieve_corrector_predictors:
        pass
        normalise_features(x)

    # if retrieve_facies is True and facies are available, load facies, otherwise prepare a placeholder (0 for all bands?)
    if retrieve_facies:
        pass
    else:
        pass

    # stack corrector inputs
    if retrieve_corrector_predictors:
        pass
    
    return x, y

    
def prefetch_features(glacier, year, retrieve_corrector_predictors=False, retrieve_facies=False): 
    raise NotImplementedError() # TODO?


def normalise_features(x):
    pass


# Elementary retrievers
# retrieve total_smb, point_smb, ...
@cache(use_cache=constants.use_cache)
def retrieve_smb_records(glacier, year):
    total_smb_path = f""
    point_smb_path = f""
    
    total_smb = pandas.read_csv(total_smb_path)
    total_smb = total_smb[total_smb.year == year]

    point_smb = None
    if os.path.exists(point_smb_path)
        point_smb = pandas.read_csv(point_smb_path)
        point_smb = point_smb[point_smb.year == year]
        if len(point_smb) == 0:
            point_smb = None

    return total_smb, point_smb


@cache(use_cache=constants.use_cache)
def retrieve_outlines(glacier, year):
    outline_model = retrieve_outline_model(glacier)
    outlines = outline_model.sel(year=year)
    return outlines







@cache(use_cache=constants.use_cache)
def retrieve_outline_model(glacier):
    outline_model = xarray.open_dataset(f"{constants.data_folder}/{glacier}/outlines.nc", decode_coords="all", engine="netcdf4")
    return outline_model


@cache(use_cache=constants.use_cache)
def retrieve_orography(glacier):
    orography = xarray.open_dataset(f"{constants.data_folder}/{glacier}/orography.nc", decode_coords="all", engine="netcdf4")
    return orography






@cache(use_cache=constants.use_cache)
def retrieve_crs_and_transform(glacier):
    outline_model = retrieve_outline_model(glacier)
    return outline_model.rio.crs, outline_model.rio.transform()


@cache(use_cache=constants.use_cache)
def retrieve_normalisation_constants():
    pass
    

# Conversions
def downscale_temperature(temperature, elevation, orography, temperature_lapse_rate=constants.temperature_lapse_rate):
    temperature_downscaled = temperature + temperature_lapse_rate * (elevation - orography)
    return temperature_downscaled


def convert_points_to_rowcol(point_smb_df, target_crs, transform, source_crs="EPSG:4326"):
    pass
