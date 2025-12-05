import jax
import xarray
import pandas
import rioxarray
import numpy as np
import rasterio.transform
import pyproj
import os

import utils.serialise
import dataloader.prefetcher
from dataloader.utils import cache
import constants


# Core logic
def retrieve_dataset_index():
    dataset_index = pandas.read_csv(f"{constants.data_folder}/index.csv")
    return dataset_index


def get_train_subset(dataset_index):
    train_subset = dataset_index[~dataset_index.fold.isin({constants.test_fold, constants.val_fold})]
    return train_subset

    
def get_val_subset(dataset_index):
    val_subset = dataset_index[dataset_index.fold == constants.val_fold]
    return val_subset


def get_test_subset(dataset_index):
    test_subset = dataset_index[dataset_index.fold == constants.test_fold]
    return test_subset


def traverse_glaciers(fold):
    glaciers = fold.name
    for glacier in glaciers:
        yield fold[fold.name == glacier].iloc[0]


def retrieve_xy(glacier, year, geometry_year=None, retrieve_corrector_predictors=False, retrieve_facies=False, numpy=True):
    """
    No-cache variant of _retrieve_xy to avoid in-place edits of x in inverse modelling/initial guess.
    """
    x, y = _retrieve_xy(
        glacier, 
        year, 
        geometry_year, 
        retrieve_corrector_predictors, 
        retrieve_facies, 
        numpy=numpy,
    )
    x_copy = _copy_nested_dicts(x)
    return x_copy, y


@cache(use_cache=constants.use_cache)
def _retrieve_xy(glacier, year, geometry_year, retrieve_corrector_predictors, retrieve_facies, numpy):
    if geometry_year is None:
        geometry_year = year
        
    if retrieve_facies and not retrieve_corrector_predictors:
        raise ValueError("If facies are fetched, whole corrector input fields should be constructed with retrieve_corrector_predictors=True!")
        
    # always load all available smb records, t, p and outlines
    y = retrieve_smb_records(glacier, year)
    begin_date, midseason_date, end_date = extract_season_dates(y["total"])
    if y["point"] is not None:
        y["point"] = weight_point_smb(y["point"], begin_date, midseason_date, end_date)
        # y["point"] = group_point_smb_on_grid(y["point"])
    
    outlines = retrieve_outlines(glacier, geometry_year)
    x = {}
    x["outlines"] = outlines
    
    temperature = retrieve_lapse_rate_corrected_temperature(glacier, geometry_year, begin_date, end_date)
    precipitation = retrieve_precipitation(glacier, begin_date, end_date)
    
    # separate climate time series by season if seasons are measured
    if not pandas.isna(midseason_date):
        winter = slice(begin_date, midseason_date)
        summer = slice(midseason_date, end_date)
        
        x["winter"] = {
            "temperature": temperature.sel(time=winter),
            "precipitation": precipitation.sel(time=winter),
        }
        x["summer"] = {
            "temperature": temperature.sel(time=summer),
            "precipitation": precipitation.sel(time=summer),
        }
    else:
        x["annual"] = {
            "temperature": temperature,
            "precipitation": precipitation,
        }
    
    # if retrieve_corrector_predictors is True, load and normalise elev, elev_stddev and monthly averaged t and p
    if retrieve_corrector_predictors:
        x.update({
            "elevation": retrieve_elevation(glacier, geometry_year),
            "elevation_std": retrieve_elevation_std(glacier, geometry_year), 
            "delta_z": retrieve_elevation(glacier, geometry_year) - retrieve_orography(glacier), 
            "t_monthly": temperature.resample(time="MS").mean().weighted(outlines).mean(dim=("x", "y")), 
            "p_monthly": precipitation.resample(time="MS").mean().weighted(outlines).mean(dim=("x", "y")),
        })
        x = normalise_features(x)

    # if retrieve_facies is True and facies are available, load facies, otherwise prepare a placeholder (-1 for all bands)
    if retrieve_facies:
        x["facies"] = retrieve_glacier_facies(glacier, year)
    elif retrieve_corrector_predictors:
        x["facies"] = retrieve_facies_placeholder(glacier)

    # stack corrector inputs
    if retrieve_corrector_predictors:
        corrector_fields = xarray.concat(
            [
                x["outlines"].expand_dims(band=["outlines"]).rename("outlines"),
                x["elevation"].expand_dims(band=["elevation"]).rename("elevation"),
                x["elevation_std"].expand_dims(band=["elevation_std"]).rename("elevation_std"),
                x["delta_z"].expand_dims(band=["delta_z"]).rename("delta_z"),
                x["facies"].rename("facies")
            ],
            dim="band",
            coords="minimal",
        )
        corrector_fields = corrector_fields.transpose("band", "y", "x")
        corrector_fields = corrector_fields.rename("corrector_fields")
        x["corrector_fields"] = corrector_fields
        del x["elevation"], x["elevation_std"], x["delta_z"], x["facies"]
        
        climate_monthly = xarray.concat(
            [x["t_monthly"].rename("t"), x["p_monthly"].rename("p")],
            dim="var"
        )
        climate_monthly = climate_monthly.assign_coords(var=["t", "p"])
        climate_monthly = climate_monthly.rename("climate_monthly")
        x["climate_monthly"] = climate_monthly
        del x["t_monthly"], x["p_monthly"]

    if numpy:
        x = x_to_raw_numpy(x)
        
    x = dict(sorted(x.items()))
    return x, y

    
def prefetch_xy(glacier, year, geometry_year=None, retrieve_corrector_predictors=False, retrieve_facies=False, numpy=True, device=None):
    def _task():
        x, y = retrieve_xy(
            glacier,
            year,
            geometry_year=geometry_year,
            retrieve_corrector_predictors=retrieve_corrector_predictors,
            retrieve_facies=retrieve_facies,
            numpy=numpy,
        )
        if device is not None:
            x = to_device_tree(x, device)
        return x, y
    return dataloader.prefetcher.submit(_task)


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
            return {
                "total": total_smb, 
                "point": None,
            }

        crs, transform = retrieve_crs_and_transform(glacier)
        point_smb = convert_latlon_to_rowcol(point_smb, crs, transform)

    return {
        "total": total_smb, 
        "point": point_smb,
    }


@cache(use_cache=constants.use_cache)
def retrieve_outlines(glacier, year):
    outline_model = retrieve_outline_model(glacier)
    outlines = outline_model.sel(year=year)
    return outlines


@cache(use_cache=constants.use_cache)
def retrieve_lapse_rate_corrected_temperature(glacier, geometry_year, begin_date=None, end_date=None):
    temperature = retrieve_temperature(glacier, begin_date, end_date)
    elevation = retrieve_elevation(glacier, geometry_year)
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
def retrieve_glacier_facies(glacier, year):
    facies_model = retrieve_facies_model(glacier)
    if facies_model is None or year not in facies_model.year:
        facies = retrieve_facies_placeholder(glacier)
    else:
        facies = facies_model.sel(year=year)
    return facies


@cache(use_cache=constants.use_cache)
def retrieve_facies_placeholder(glacier):
    template = retrieve_outline_model(glacier).isel(year=0).reset_coords(names="year", drop=True)
    facies = xarray.full_like(template, -1.0, dtype=np.float32) \
        .expand_dims(band=constants.n_facies_classes + 1) \
        .assign_coords(band=[
            "ice", "snow", "debris", "firn", "shadow", 
            "superimposed-ice", "cloud", "water", "confidence",
        ]) \
        .rename("facies")
    facies = facies.rio.write_crs(template.rio.crs)
    facies = facies.rio.write_transform(template.rio.transform())
    return facies
    

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
    facies_model = facies_model["facies"].fillna(0)
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
    

# Conversions/extractions/helpers
def extract_season_dates(total_smb_for_one_year):
    begin_date = total_smb_for_one_year.begin_date.iloc[0]
    midseason_date = total_smb_for_one_year.midseason_date.iloc[0]
    end_date = total_smb_for_one_year.end_date.iloc[0]
    return begin_date, midseason_date, end_date


def weight_point_smb(point_smb, begin_date, midseason_date, end_date):
    point_smb["weight"] = 0.0

    point_smb["begin_date_tmstp"] = pandas.to_datetime(point_smb["begin_date"], format="%Y-%m-%d")
    point_smb["end_date_tmstp"] = pandas.to_datetime(point_smb["end_date"], format="%Y-%m-%d")
    
    begin_date = pandas.Timestamp(begin_date)
    end_date = pandas.Timestamp(end_date)
    if not pandas.isna(midseason_date):
        midseason_date = pandas.Timestamp(midseason_date)
    
    point_smb.loc[point_smb.balance_code == "annual", "weight"] = _gaussian_kernel((point_smb.begin_date_tmstp - begin_date).dt.days) * \
        _gaussian_kernel((point_smb.end_date_tmstp - end_date).dt.days)
    if not pandas.isna(midseason_date):
        point_smb.loc[point_smb.balance_code == "winter", "weight"] = _gaussian_kernel((point_smb.begin_date_tmstp - begin_date).dt.days) * \
            _gaussian_kernel((point_smb.end_date_tmstp - midseason_date).dt.days)
        point_smb.loc[point_smb.balance_code == "summer", "weight"] = _gaussian_kernel((point_smb.begin_date_tmstp - midseason_date).dt.days) * \
            _gaussian_kernel((point_smb.end_date_tmstp - end_date).dt.days)
    
    return point_smb
    

def normalise_features(x):
    normalisation_factors = retrieve_normalisation_factors()
    for feature in ["elevation", "elevation_std", "delta_z", "t_monthly", "p_monthly"]:
        v = x[feature].copy()
        x[feature] = v / normalisation_factors[feature]
    return x
    
    
def downscale_temperature(temperature, elevation, orography, temperature_lapse_rate=constants.temperature_lapse_rate):
    temperature_downscaled = temperature + temperature_lapse_rate * (elevation - orography)
    return temperature_downscaled


def convert_latlon_to_rowcol(point_smb_df, target_crs, rst_transform, source_crs="EPSG:4326"):
    crs_transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    point_smb_df[["row", "col"]] = point_smb_df.apply(
        lambda x: _latlon_to_rowcol(x["latitude"], x["longitude"], crs_transformer, rst_transform),
        axis=1, 
    ) # TODO: Consider vectorising instead of .apply
    return point_smb_df


# def group_point_smb_on_grid(point_smb_df, eps=1e-6):
#     grouped = []
#     for (row, col, balance_code), g in point_smb_df.groupby(["row", "col", "balance_code"]):
#         weight_sum = g["weight"].sum()
#         balance = (g["balance"] * g["weight"]).sum() / (weight_sum + eps)
#         grouped.append({
#             "row": row,
#             "col": col,
#             "balance_code": balance_code,
#             "weight": weight_sum,
#             "balance": balance,
#         })
#     return pandas.DataFrame(grouped)


def x_to_raw_numpy(x):
    if isinstance(x, dict):
        return {k: x_to_raw_numpy(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        t = [x_to_raw_numpy(v) for v in x]
        return type(x)(t)
    elif isinstance(x, xarray.DataArray):
        return np.ascontiguousarray(x.data)
    else:
        return x


def to_device_tree(x, device):
    return jax.tree.map(
        lambda a: jax.device_put(a, device) if isinstance(a, np.ndarray) else a,
        x,
    )


def _copy_nested_dicts(x):
    copy = x.copy()
    for k in list(copy.keys()):
        v = copy[k]
        if isinstance(v, dict):
            copy[k] = _copy_nested_dicts(v)
    return copy


def _gaussian_kernel(d, decay_rate=constants.point_smb_weight_decay_rate):
    return np.exp(-(d / decay_rate)**2)

    
def _latlon_to_rowcol(lat, lon, crs_transformer, rst_transform):
    x, y = crs_transformer.transform(lon, lat)
    row, col = rasterio.transform.rowcol(rst_transform, x, y)
    return pandas.Series({"row": row, "col": col})
    