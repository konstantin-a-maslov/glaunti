import os
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_compilation_cache_dir", "jax_xla_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

import rioxarray
import xarray
import pandas

import dataloader.dataloader as dataloader
import core.loss as loss
import utils.serialise
import constants

import datetime
import numpy as np
import pyproj

import sklearn.metrics
import scipy

import json
import os
from tqdm import tqdm
import argparse


def main():
    model, ti, ti_corr, facies, params_path, glacier, smb_path, eval_path = parse_args()
    smb = infer(model, ti, ti_corr, facies, params_path, glacier)
    save_smb(smb, smb_path)
    evaluation = evaluate(smb, glacier)
    save_evaluation(evaluation, eval_path)
    print(model, ti, ti_corr, facies, params_path, glacier, smb_path, eval_path)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["a", "b", "c", "d"], help="Model name")
    parser.add_argument("glacier", help="Glacier name")
    parser.add_argument("smb_path", help="Output SMB path (.nc)")
    parser.add_argument("eval_path", help="Output evaluation path (.json)")
    parser.add_argument("--params_path", help="Params path (.eqx)")
    args = parser.parse_args()

    ti, ti_corr, facies = False, False, False
    match args.model:
        case "a":
            import glaunti.ti_model as model
            ti = True
        case "b":
            import glaunti.gru_model as model
        case "c":
            import glaunti.ti_corr_model as model
            ti, ti_corr = True, True
        case "d":
            import glaunti.ti_corr_model as model
            ti, ti_corr, facies = True, True, True
    default_params_path = f"params/{args.model}.eqx"

    params_path = args.params_path or default_params_path
    smb_path = args.smb_path
    eval_path = args.eval_path

    dataset_index = dataloader.retrieve_dataset_index()
    glacier = dataset_index[dataset_index.name == args.glacier].iloc[0]
    
    return model, ti, ti_corr, facies, params_path, glacier, smb_path, eval_path


def infer(model, ti, ti_corr, facies, params_path, glacier):
    trainable_params, static_params = load_params(model, ti_corr, params_path)
    model_callable = get_model_callable(model, ti, ti_corr) 
    smb_results, ts = [], []

    swe_or_h, next_xy = loss.init_swe_or_h(
        trainable_params, static_params, model_callable, glacier["name"], 
        retrieve_corrector_predictors=ti_corr, retrieve_facies=facies, last_numpy=False,
    )
    
    for year in tqdm(range(constants.study_period_start_year, glacier.max_year + 1), desc="Inference"):
        x, y = next_xy.get()
        next_xy = dataloader.prefetch_xy(
            glacier["name"], year + 1, 
            retrieve_corrector_predictors=ti_corr, retrieve_facies=facies, numpy=False,
        )

        if "annual" in x:
            ts.append(x["annual"]["temperature"])
        else:
            ts.append(x["winter"]["temperature"])
            ts.append(x["summer"]["temperature"])
        
        x = dataloader.x_to_raw_numpy(x)
        if "annual" in x:
            x_annual = dict(x)
            x_annual.update(x_annual["annual"])
            ys = model_callable(trainable_params, static_params, x_annual, swe_or_h)
            smb, swe_or_h = ys[0], ys[1]
            smb_results.append(np.array(smb))
        else:
            x_winter = dict(x)
            x_winter.update(x_winter["winter"])
            ys = model_callable(trainable_params, static_params, x_winter, swe_or_h)
            smb, swe_or_h = ys[0], ys[1]
            smb_results.append(np.array(smb))
            x_summer = dict(x)
            x_summer.update(x_summer["summer"])
            ys = model_callable(trainable_params, static_params, x_summer, swe_or_h)
            smb, swe_or_h = ys[0], ys[1]
            smb_results.append(np.array(smb))

    smb_xrs = []
    for t, smb_r in zip(ts, smb_results):
        smb_xr = t.copy()
        smb_xr.data = smb_r
        smb_xr = smb_xr.rename("smb")
        smb_xrs.append(smb_xr)
    smb_xr = xarray.concat(smb_xrs, dim="time", join="exact").sortby("time").groupby("time").sum("time")
        
    return smb_xr
    

def save_smb(smb, smb_path):
    folder = os.path.dirname(smb_path)
    os.makedirs(folder, exist_ok=True)
    smb.to_netcdf(smb_path, engine="netcdf4")

    
def evaluate(smb, glacier):
    pass

    
def save_evaluation(evaluation, output_path):
    pass
    

def load_params(model, ti_corr, params_path):
    if ti_corr:
        params = model.get_initial_model_parameters(ti_params_static=True)
    else:
        params = model.get_initial_model_parameters()
    params = utils.serialise.load_pytree(params_path, template=params)
    return params


def get_model_callable(model, ti, ti_corr):
    if ti_corr: # c, d
        model_callable = jax.jit(
            lambda trainable_params, static_params, x, initial_swe, ds=None: model.run_model(
                trainable_params, 
                static_params, 
                x, 
                initial_swe=initial_swe, 
                return_series=True, 
                return_corrections=True,
                ds=ds,
            )
        )
    elif ti: # a
        model_callable = jax.jit(
            lambda trainable_params, static_params, x, initial_swe: model.run_model(
                trainable_params, 
                static_params, 
                x, 
                initial_swe=initial_swe, 
                return_series=True, 
            )
        )
    else: # b
        model_callable = jax.jit(
            lambda trainable_params, static_params, x, initial_h: model.run_model(
                trainable_params, 
                static_params, 
                x, 
                initial_h=initial_h, 
                return_series=True, 
            )
        )
    return model_callable

    
if __name__ == "__main__":
    main()
