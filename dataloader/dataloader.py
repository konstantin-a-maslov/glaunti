import xarray
import pandas
import rioxarray
import numpy as np
import rasterio.transform
import pyproj
import os

import utils.serialise
from dataloader.utils import cache
import constants


# Core logic
def retrieve_dataset_index():
    dataset_index = pandas.read_csv(f"{constants.data_folder}/index.csv")
    return dataset_index


def traverse_train_val_folds(dataset_index):
    folds = dataset_index[dataset_index.fold != constants.test_fold]
    fold_ids = sorted(folds.fold.unique())
    for fold_id in fold_ids:
        val_fold = dataset_index[dataset_index.fold == fold_id]
        train_folds = dataset_index[dataset_index.fold != fold_id]
        yield val_fold, train_folds


def traverse_glaciers(fold):
    glaciers = sorted(fold.name.unique())
    for glacier in glaciers:
        yield glacier # [633] CONSIDER RETURNING A WHOLE RECORD


def traverse_years(glacier):
    start_year = constants.initialisation_period_start_year
    end_year = glacier.end_year # [633] BUG HERE, GLACIER IS A STRING, FINALISE
    for year in range(start_year, end_year + 1):
        yield year

    
@cache(use_cache=constants.use_cache)
def retrieve_features(glacier, year, retrieve_corrector_predictors=False, retrieve_facies=False):
    if retrieve_facies and not retrieve_corrector_predictors:
        raise ValueError("If facies are fetched, whole corrector input fields should be constructed with retrieve_corrector_predictors=True!")
        
    # always load all available smb records, t, p and outlines
    y = retrieve_smb_records(glacier, year)
    begin_date, midseason_date, end_date = extract_season_dates(y[0])
    y = split_smb_records_by_seasons(y, begin_date, midseason_date, end_date)
    
    outlines = retrieve_outlines(glacier, year)

    x = {
        "outlines": outlines,
    }
    
    temperature = retrieve_lapse_rate_corrected_temperature(glacier, year, begin_date, end_date)
    precipitation = retrieve_precipitation(glacier, begin_date, end_date)
    
    # separate climate time series by season if seasons are measured
    if not pandas.isna(midseason_date):
        winter = slice(begin_date, midseason_date)
        summer = slice(midseason_date, end_date)
        
        x["winter"] = {
            "temperature": temperature.sel(time=winter).data,
            "precipitation": precipitation.sel(time=winter).data,
        }
        x["summer"] = {
            "temperature": temperature.sel(time=summer).data,
            "precipitation": precipitation.sel(time=summer).data,
        }
    else:
        x["annual"] = {
            "temperature": temperature.data,
            "precipitation": precipitation.data,
        }
    
    # if retrieve_corrector_predictors is True, load and normalise elev, elev_stddev and monthly averaged t and p
    if retrieve_corrector_predictors:
        raise NotImplementedError()
        x = normalise_features(x)

    # if retrieve_facies is True and facies are available, load facies, otherwise prepare a placeholder (0 for all bands?)
    if retrieve_facies:
        raise NotImplementedError()
    else:
        pass

    # stack corrector inputs
    if retrieve_corrector_predictors:
        raise NotImplementedError()

    # TODO: CONSIDER MATERIALISATION IN PLACE FOR BETTER PERFORMANCE HERE
    
    return x, y

    
def prefetch_features(glacier, year, retrieve_corrector_predictors=False, retrieve_facies=False): 
    raise NotImplementedError() # TODO?


def normalise_features(x):
    normalisation_factors = retrieve_normalisation_factors()
    for feature in ["elevation", "elevation_std", "t_monthly", "p_monthly"]:
        x[feature] /= normalisation_factors[feature]
    return x


# Elementary retrievers
@cache(use_cache=constants.use_cache)
def retrieve_smb_records(glacier, year):
    total_smb_path = f"{constants.data_folder}/{glacier}/total_smb.csv"
    point_smb_path = f"{constants.data_folder}/{glacier}/point_smb.csv"
    
    total_smb = pandas.read_csv(total_smb_path)
    total_smb = total_smb[total_smb.year == year]

    if len(total_smb) == 0:
        total_smb = None
    
    point_smb = None
    if os.path.exists(point_smb_path):
        point_smb = pandas.read_csv(point_smb_path)
        point_smb = point_smb[point_smb.year == year]
        
        if len(point_smb) == 0:
            return total_smb, None

        crs, transform = retrieve_crs_and_transform(glacier)
        point_smb = convert_latlon_to_rowcol(point_smb, crs, transform)

    return total_smb, point_smb


@cache(use_cache=constants.use_cache)
def retrieve_outlines(glacier, year):
    outline_model = retrieve_outline_model(glacier)
    outlines = outline_model.sel(year=year)
    return outlines


@cache(use_cache=constants.use_cache)
def retrieve_lapse_rate_corrected_temperature(glacier, year, begin_date=None, end_date=None):
    temperature = retrieve_temperature(glacier, begin_date, end_date)
    elevation = retrieve_elevation(glacier, year)
    orography = retrieve_orography(glacier)
    lapse_rate_corrected_temperature = downscale_temperature(temperature, elevation, orography)
    return lapse_rate_corrected_temperature
    

@cache(use_cache=constants.use_cache)
def retrieve_temperature(glacier, begin_date=None, end_date=None):
    temperature = retrieve_complete_temperature_series(glacier)
    time_slice = slice(begin_date, end_date)
    temperature = temperature.sel(time=time_slice)
    return temperature


@cache(use_cache=constants.use_cache)
def retrieve_precipitation(glacier, begin_date=None, end_date=None):
    precipitation = retrieve_complete_precipitation_series(glacier)
    time_slice = slice(begin_date, end_date)
    precipitation = precipitation.sel(time=time_slice)
    return precipitation


@cache(use_cache=constants.use_cache)
def retrieve_elevation(glacier, year):
    elevation = retrieve_elevation_model(glacier)
    elevation = elevation["z"]
    elevation = elevation.sel(year=year)
    return elevation


@cache(use_cache=constants.use_cache)
def retrieve_elevation_std(glacier, year):
    elevation = retrieve_elevation_model(glacier)
    elevation_stddev = elevation["zstddev"]
    elevation_stddev = elevation_stddev.sel(year=year)
    return elevation_stddev


@cache(use_cache=constants.use_cache)
def retrieve_facies(glacier, year):
    facies_model = retrieve_facies_model(glacier)
    if facies_model is None or year not in facies_model.year:
        facies = retrieve_facies_placeholder(glacier)
    else:
        facies = facies_model.sel(year=year)
    return facies


@cache(use_cache=constants.use_cache)
def retrieve_facies_placeholder(glacier):
    raise NotImplementedError()
    

@cache(use_cache=constants.use_cache)
def retrieve_complete_temperature_series(glacier):
    temperature = xarray.open_dataset(f"{constants.data_folder}/{glacier}/temperature.nc", decode_coords="all", engine="netcdf4")
    temperature = temperature["temperature_2m"]
    return temperature.astype(np.float32)


@cache(use_cache=constants.use_cache)
def retrieve_complete_precipitation_series(glacier):
    precipitation = xarray.open_dataset(f"{constants.data_folder}/{glacier}/precipitation.nc", decode_coords="all", engine="netcdf4")
    precipitation = precipitation["total_precipitation_sum"]
    return precipitation.astype(np.float32)


@cache(use_cache=constants.use_cache)
def retrieve_elevation_model(glacier):
    elevation_model = xarray.open_dataset(f"{constants.data_folder}/{glacier}/dem.nc", decode_coords="all", engine="netcdf4")
    return elevation_model.astype(np.float32)


@cache(use_cache=constants.use_cache)
def retrieve_facies_model(glacier):
    facies_model_path = f"{constants.data_folder}/{glacier}/facies.nc"
    if not os.path.exists(facies_model_path):
        return None
    facies_model = xarray.open_dataset(facies_model_path, decode_coords="all", engine="netcdf4")
    facies_model = facies_model["facies"]
    return facies_model.astype(np.float32)


@cache(use_cache=constants.use_cache)
def retrieve_outline_model(glacier):
    outline_model = xarray.open_dataset(f"{constants.data_folder}/{glacier}/outlines.nc", decode_coords="all", engine="netcdf4")
    outline_model = outline_model["outlines"]
    return outline_model.astype(np.float32)


@cache(use_cache=constants.use_cache)
def retrieve_orography(glacier):
    orography = xarray.open_dataset(f"{constants.data_folder}/{glacier}/orography.nc", decode_coords="all", engine="netcdf4")
    orography = orography["elevation"]
    return orography.astype(np.float32)


@cache(use_cache=constants.use_cache)
def retrieve_crs_and_transform(glacier):
    outline_model = retrieve_outline_model(glacier)
    return outline_model.rio.crs, outline_model.rio.transform()


@cache(use_cache=constants.use_cache)
def retrieve_normalisation_factors():
    normalisation_factors = utils.serialise.load_pytree_with_pickle(f"{constants.data_folder}/normalisation_factors.pkl")
    return normalisation_factors
    

# Conversions/extractions
def extract_season_dates(total_smb_for_one_year):
    begin_date = total_smb_for_one_year.begin_date.iloc[0]
    midseason_date = total_smb_for_one_year.midseason_date.iloc[0]
    end_date = total_smb_for_one_year.end_date.iloc[0]
    return begin_date, midseason_date, end_date


def downscale_temperature(temperature, elevation, orography, temperature_lapse_rate=constants.temperature_lapse_rate):
    temperature_downscaled = temperature + temperature_lapse_rate * (elevation - orography)
    return temperature_downscaled


def convert_latlon_to_rowcol(point_smb_df, target_crs, rst_transform, source_crs="EPSG:4326"):
    crs_transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    point_smb_df[["row", "col"]] = point_smb_df.apply(
        lambda x: _latlon_to_rowcol(x["lat"], x["lon"], crs_transformer, rst_transform),
        axis=1, 
    ) # TODO: Consider vectorising instead of .apply
    return point_smb_df


def _latlon_to_rowcol(lat, lon, crs_transformer, rst_transform):
    x, y = crs_transformer.transform(lon, lat)
    row, col = rasterio.transform.rowcol(rst_transform, x, y)
    return pandas.Series({"row": row, "col": col})








######
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
import itertools

import constants
from . import retrieve_features  # your function

# One shared executor for all prefetch tasks
_PREFETCH_EXEC = ThreadPoolExecutor(max_workers=getattr(constants, "prefetch_workers", 2))

def _materialize(obj):
    """Optionally force-load xarray objects to make 'get()' return fully in-memory results."""
    try:
        return obj.load() if hasattr(obj, "load") else obj
    except Exception:
        return obj

def _walk_and_materialize(tree):
    if isinstance(tree, dict):
        return {k: _walk_and_materialize(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        t = type(tree)
        return t(_walk_and_materialize(v) for v in tree)
    return _materialize(tree)

@dataclass
class PrefetchHandle:
    _future: Future

    def get(self, timeout=None):
        """Block until ready and return (x, y). Propagates exceptions."""
        return self._future.result(timeout=timeout)

    def done(self):
        return self._future.done()

    def cancel(self):
        return self._future.cancel()

def prefetch_features(
    glacier,
    year,
    *,
    retrieve_corrector_predictors=False,
    retrieve_facies=False,
    realize=True,   # if True, .get() returns fully loaded (materialized) arrays
):
    """
    Submit a background task that calls retrieve_features(...) and returns a PrefetchHandle.
    Use handle.get() later to obtain (x, y).
    """
    def _task():
        x, y = retrieve_features(
            glacier,
            year,
            retrieve_corrector_predictors=retrieve_corrector_predictors,
            retrieve_facies=retrieve_facies,
        )
        if realize:
            x = _walk_and_materialize(x)
            y = _walk_and_materialize(y)
        return x, y

    fut = _PREFETCH_EXEC.submit(_task)
    return PrefetchHandle(fut)

def shutdown_prefetch():
    _PREFETCH_EXEC.shutdown(wait=False, cancel_futures=True)