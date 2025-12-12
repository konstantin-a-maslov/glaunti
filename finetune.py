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

# The script is tied to model A as in the manuscript, yet one can easily adapt it to other models
import glaunti.ti_model as model
import train_c
import dataloader.dataloader as dataloader
import core.loss as loss
import core.training as training
import utils.serialise
import utils.logger
import constants

import numpy as np
import datetime
from tqdm import tqdm
import argparse


ti, ti_corr, facies = True, False, False
train_year_end, val_year_end = 2012, 2018
n_epochs, learning_rate = 250, 1e-2


def main():
    glacier, init_params_path, final_params_path, log_path = resolve_args()
    params = model.get_initial_model_parameters()
    trainable_params, static_params = utils.serialise.load_pytree(init_params_path, template=params)

    model_callable = jax.jit(
        lambda trainable_params, static_params, x, initial_swe: model.run_model(
            trainable_params, static_params, x, 
            initial_swe=initial_swe, 
            return_series=False, 
        )
    )

    loss_grad = jax.value_and_grad(finetuning_loss, argnums=0, has_aux=True)
    optimiser = training.get_optimiser(lr=learning_rate)
    opt_state = optimiser.init(trainable_params)
    logger, train_pbar_desc, val_pbar_desc = None, "", ""

    best_val_mse, best_epoch = np.inf, 0
    device = jax.devices()[0]

    with tqdm(total=n_epochs, desc="") as pbar:
        for epoch in range(n_epochs):
            (loss_value, (aux, swe_or_h, next_xy)), grads = loss_grad(
                trainable_params, static_params, model_callable, glacier, 
                end_year=train_year_end,
                ti=ti, ti_corr=ti_corr, device_to_prefetch=device,
            )            
            train_mse = train_c.extract_mse(aux)
            # log
            log_record = {"timestamp": str(datetime.datetime.now()), "epoch": epoch, "glacier": glacier["name"], "subset": "train", "loss": loss_value, **aux}
            if logger is None:
                logger = utils.logger.CSVLogger(log_path, log_record.keys())
            logger.log(log_record)
            trainable_params, opt_state = training.make_step(optimiser, grads, trainable_params, opt_state)
            train_pbar_desc = f"train_mse={train_mse:.3f}, "
            
            # checkpoint         
            loss_value, (aux, _, _) = finetuning_loss(
                trainable_params, static_params, model_callable, glacier, 
                end_year=val_year_end, start_year=train_year_end + 1,
                ti=ti, ti_corr=ti_corr, device_to_prefetch=device,
                swe_or_h=swe_or_h, next_xy=next_xy,
            ) 
            val_mse = train_c.extract_mse(aux)
            # log
            log_record = {"timestamp": str(datetime.datetime.now()), "epoch": epoch, "glacier": glacier["name"], "subset": "val", "loss": loss_value, **aux}
            logger.log(log_record)
            if val_mse < best_val_mse:
                best_val_mse, best_epoch = val_mse, epoch
                utils.serialise.save_pytree((trainable_params, static_params), final_params_path)
            val_pbar_desc = f"val_mse={val_mse:.3f} (best={best_val_mse:.3f} at #{best_epoch})"

            pbar.set_description(f"{train_pbar_desc}{val_pbar_desc} [m w.e.]")
            pbar.update(1)


def finetuning_loss(
    trainable_params, static_params, model_callable, 
    glacier, end_year, start_year=constants.study_period_start_year,
    ti=False, ti_corr=False, retrieve_facies=False, 
    swe_or_h=None, next_xy=None,
    lambda1=constants.lambda1, 
    lambda2=constants.default_lambda2, 
    lambda3=constants.default_lambda3, lambda4=constants.default_lambda4, lambda5=constants.default_lambda5, 
    device_to_prefetch=None,
):
    glacier_name, max_year, aux = glacier["name"], end_year, {}
    for balance_code in ["annual", "winter", "summer"]:
        aux.update({
            f"total_{balance_code}_error": 0, f"total_{balance_code}_n": 0, f"total_{balance_code}_mse": None, 
            f"point_{balance_code}_error": 0, f"point_{balance_code}_n": 0, f"point_{balance_code}_mse": None,
        })
    if ti_corr:
        aux["reg_ti_corr_acc"] = 0
        aux["reg_ti_corr_n"] = 0
        aux["reg_ti_corr"] = None

    if swe_or_h is None:
        swe_or_h, next_xy = loss.init_swe_or_h(
            trainable_params, static_params, model_callable, glacier_name, ti_corr, retrieve_facies, device_to_prefetch=device_to_prefetch
        )
    
    for year in range(start_year, max_year + 1):
        x, y = next_xy.get()
        next_xy = dataloader.prefetch_xy(
            glacier_name, year + 1, 
            retrieve_corrector_predictors=ti_corr, 
            retrieve_facies=retrieve_facies,
            device=device_to_prefetch,
        )
        
        if not "winter" in x:
            x_annual = dict(x)
            x_annual.update(x_annual["annual"])
            ys = model_callable(trainable_params, static_params, x_annual, swe_or_h)
            smb_annual, swe_or_h = ys[0], ys[1]

        else:
            x_winter = dict(x)
            x_winter.update(x_winter["winter"])
            ys = model_callable(trainable_params, static_params, x_winter, swe_or_h)
            smb_winter, swe_or_h = ys[0], ys[1]
            loss.update_metrics(aux, smb_winter, y, x["outlines"], "winter")

            x_summer = dict(x)
            x_summer.update(x_summer["summer"])
            if ti_corr:
                ys = model_callable(trainable_params, static_params, x_summer, swe_or_h, ds=ys[2])
            else:
                ys = model_callable(trainable_params, static_params, x_summer, swe_or_h)
            smb_summer, swe_or_h = ys[0], ys[1]
            loss.update_metrics(aux, smb_summer, y, x["outlines"], "summer")

            smb_annual = smb_winter + smb_summer
            
        loss.update_metrics(aux, smb_annual, y, x["outlines"], "annual")
        if ti_corr:
            ds = ys[2]
            reg_ti_corr_acc = aux["reg_ti_corr_acc"]
            reg_ti_corr_n = aux["reg_ti_corr_n"]
            reg_ti_corr_acc, reg_ti_corr_n = loss.update_ti_corr_regulariser(reg_ti_corr_acc, reg_ti_corr_n, ds, x["outlines"], lambda3, lambda4, lambda5)
            aux["reg_ti_corr_acc"] = reg_ti_corr_acc
            aux["reg_ti_corr_n"] = reg_ti_corr_n

    loss_v = 0.0
    for balance_code in ["annual", "winter", "summer"]:
        loss_v += (
            (aux[f"point_{balance_code}_mse"] if aux[f"point_{balance_code}_mse"] is not None else 0.0) + 
            lambda1 * (aux[f"total_{balance_code}_mse"] if aux[f"total_{balance_code}_mse"] is not None else 0.0)
        )
    
    reg = 0.0
    if ti:
        params = {**static_params, **trainable_params}
        aux["reg_ti"] = loss.ti_regulariser(params, lambda2)
        reg += aux["reg_ti"]
    if ti_corr:
        aux["reg_ti_corr"] = aux["reg_ti_corr_acc"] / aux["reg_ti_corr_n"]
        reg += aux["reg_ti_corr"]
    
    return loss_v + reg, (aux, swe_or_h, next_xy)


def resolve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("glacier", help="Glacier name")
    parser.add_argument("--init_params_path", help="Init params path (.eqx)")
    parser.add_argument("--final_params_path", help="Final params path (.eqx)")
    parser.add_argument("--log_path", help="Log path (.csv)")
    args = parser.parse_args()

    default_init_params_path = "params/a.eqx"
    init_params_path = args.init_params_path or default_init_params_path

    default_final_params_path = f"params/a_finetuned_{args.glacier}.eqx"
    final_params_path = args.final_params_path or default_final_params_path
    
    default_log_path = f"logs/a_finetuned_{args.glacier}.csv"
    log_path = args.log_path or default_log_path

    dataset_index = dataloader.retrieve_dataset_index()
    glacier = dataset_index[dataset_index.name == args.glacier].iloc[0]
    
    return glacier, init_params_path, final_params_path, log_path
    
    
if __name__ == "__main__":
    main()
