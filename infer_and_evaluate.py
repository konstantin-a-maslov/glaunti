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

import numpy as np
import scipy
import json
import os
from tqdm import tqdm
import argparse


def main():
    model, ti, ti_corr, facies, params_path, glacier, smb_path, eval_path = resolve_args()
    smb = infer(model, ti, ti_corr, facies, params_path, glacier)
    save_smb(smb, smb_path)
    evaluation = evaluate(smb, glacier)
    save_evaluation(evaluation, eval_path)
    print_summary(evaluation)
    

def resolve_args():
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
            if ti_corr:
                ys = model_callable(trainable_params, static_params, x_summer, swe_or_h, ds=ys[2])
            else:
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
    evaluation = get_evaluation_template(glacier)

    for year in tqdm(range(constants.study_period_start_year, glacier.max_year + 1), desc="Evaluation"):
        smb_records = dataloader.retrieve_smb_records(glacier["name"], year)
        total_smb = smb_records["total"]
        point_smb = smb_records["point"]
        outlines = dataloader.retrieve_outlines(glacier["name"], year)
        begin_date, midseason_date, end_date = dataloader.extract_season_dates(total_smb)
        total_smb = total_smb.iloc[0]

        if not pandas.isna(total_smb.annual_balance):
            annual_smb_subset = smb.sel(time=slice(begin_date, end_date))
            annual_smb_model = (annual_smb_subset * outlines).sum() / outlines.sum()
            evaluation["true_pred"]["glacier-wide"]["overall"]["annual"]["true"].append(float(total_smb.annual_balance))
            evaluation["true_pred"]["glacier-wide"]["overall"]["annual"]["pred"].append(float(annual_smb_model))
        if not pandas.isna(midseason_date) and not pandas.isna(total_smb.winter_balance):
            winter_smb_subset = smb.sel(time=slice(begin_date, midseason_date))
            winter_smb_model = (winter_smb_subset * outlines).sum() / outlines.sum()
            evaluation["true_pred"]["glacier-wide"]["overall"]["winter"]["true"].append(float(total_smb.winter_balance))
            evaluation["true_pred"]["glacier-wide"]["overall"]["winter"]["pred"].append(float(winter_smb_model))
        if not pandas.isna(midseason_date) and not pandas.isna(total_smb.summer_balance):
            summer_smb_subset = smb.sel(time=slice(midseason_date, end_date))
            summer_smb_model = (summer_smb_subset * outlines).sum() / outlines.sum()
            evaluation["true_pred"]["glacier-wide"]["overall"]["summer"]["true"].append(float(total_smb.summer_balance))
            evaluation["true_pred"]["glacier-wide"]["overall"]["summer"]["pred"].append(float(summer_smb_model))
        
        if point_smb is not None:
            for m in point_smb.itertuples():
                if m.balance_code == "index":
                    continue
                smb_subset = smb.sel(time=slice(m.begin_date, m.end_date))
                smb_subset = smb_subset[:, m.row, m.col]
                smb_model = smb_subset.sum()
                
                evaluation["true_pred"]["point"]["overall"][m.balance_code]["true"].append(float(m.balance))
                evaluation["true_pred"]["point"]["overall"][m.balance_code]["pred"].append(float(smb_model))
                evaluation["true_pred"]["point"]["per_year"][m.balance_code][year]["true"].append(float(m.balance))
                evaluation["true_pred"]["point"]["per_year"][m.balance_code][year]["pred"].append(float(smb_model))

    stack = [(evaluation["true_pred"], evaluation["metrics"])]
    while len(stack) > 0:
        src, dst = stack.pop()
        if "true" in src.keys() and "pred" in src.keys():
            dst.update(populate_metrics(src))
            continue
        for k in src.keys():
            stack.append((src[k], dst[k]))
                    
    return evaluation

    
def save_evaluation(evaluation, eval_path):
    folder = os.path.dirname(eval_path)
    os.makedirs(folder, exist_ok=True)
    with open(eval_path, "w") as dst:
        json.dump(evaluation, dst, indent=4)


def print_summary(evaluation):
    for estimation_type in ["point", "glacier-wide"]:
        for season in ["annual", "winter", "summer"]:
            metrics = evaluation["metrics"][estimation_type]["overall"][season]
            if metrics["n"] > 1:
                print(f"{estimation_type}, {season}, n = {metrics['n']}")
                print(f"\tpearson r = {metrics['pearson_r']:.3f} (p = {metrics['pearson_r_p']:.6f})")
                print(f"\tbias = {metrics['bias']:.3f} m w.e.")
                print(f"\trmse = {metrics['rmse']:.3f} m w.e.")
                print(f"\tmae = {metrics['mae']:.3f} m w.e.")
                print()
    

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
                trainable_params, static_params, x, 
                initial_swe=initial_swe, 
                return_series=True, 
                return_corrections=True,
                ds=ds,
            )
        )
    elif ti: # a
        model_callable = jax.jit(
            lambda trainable_params, static_params, x, initial_swe: model.run_model(
                trainable_params, static_params, x, 
                initial_swe=initial_swe, 
                return_series=True, 
            )
        )
    else: # b
        model_callable = jax.jit(
            lambda trainable_params, static_params, x, initial_h: model.run_model(
                trainable_params, static_params, x, 
                initial_h=initial_h, 
                return_series=True, 
            )
        )
        
    return model_callable


def get_evaluation_template(glacier):
    evaluation = {"metrics": {}, "true_pred": {}}
    for estimation_type in ["point", "glacier-wide"]:
        for k in ["metrics", "true_pred"]:
            evaluation[k][estimation_type] = {"overall": {}, "per_year": {}}
            for season in ["annual", "winter", "summer"]:
                evaluation[k][estimation_type]["overall"][season] = {}
                evaluation[k][estimation_type]["per_year"][season] = {}
        for season in ["annual", "winter", "summer"]:
            evaluation["true_pred"][estimation_type]["overall"][season] = {"true": [], "pred": []}
            evaluation["true_pred"][estimation_type]["per_year"][season] = {}
            for year in range(constants.study_period_start_year, glacier.max_year + 1):
                evaluation["true_pred"][estimation_type]["per_year"][season][year] = {"true": [], "pred": []}
                evaluation["metrics"][estimation_type]["per_year"][season][year] = {}
    del evaluation["metrics"]["glacier-wide"]["per_year"], evaluation["true_pred"]["glacier-wide"]["per_year"]
    return evaluation


def populate_metrics(src):
    true = np.array(src["true"])
    pred = np.array(src["pred"])

    metrics = {
        "n": len(true),
    }
    if metrics["n"] > 1:
        r, p = scipy.stats.pearsonr(pred, true)
        metrics.update({
            "pearson_r": r,
            "pearson_r_p": p,
            "rmse": np.sqrt(np.mean((pred - true)**2)),
            "mae": np.mean(np.abs(pred - true)),
            "bias": np.mean(pred - true),
        })
    
    return metrics


if __name__ == "__main__":
    main()
