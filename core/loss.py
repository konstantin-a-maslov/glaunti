import jax
import jax.numpy as jnp
import numpy as np
import pandas

import glaunti.ti_model
import dataloader.dataloader as dataloader
import constants


def loss(
    trainable_params, static_params, model_callable, 
    glacier, 
    ti=False, ti_corr=False, retrieve_facies=False, 
    lambda1=constants.lambda1, 
    lambda2=constants.default_lambda2, 
    lambda3=constants.default_lambda3, lambda4=constants.default_lambda4, lambda5=constants.default_lambda5, 
    return_aux=True,
    device_to_prefetch=None,
):   
    glacier_name, max_year, aux = glacier["name"], glacier["max_year"], {}
    for balance_code in ["annual", "winter", "summer"]:
        aux.update({
            f"total_{balance_code}_error": 0, f"total_{balance_code}_n": 0, f"total_{balance_code}_mse": None, 
            f"point_{balance_code}_error": 0, f"point_{balance_code}_n": 0, f"point_{balance_code}_mse": None,
        })
    if ti_corr:
        aux["reg_ti_corr_acc"] = 0
        aux["reg_ti_corr_n"] = 0
        aux["reg_ti_corr"] = None
        
    swe_or_h, next_xy = init_swe_or_h(
        trainable_params, static_params, model_callable, glacier_name, ti_corr, retrieve_facies, device_to_prefetch=device_to_prefetch
    )
    
    for year in range(constants.study_period_start_year, max_year + 1):
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
            update_metrics(aux, smb_winter, y, x["outlines"], "winter")

            x_summer = dict(x)
            x_summer.update(x_summer["summer"])
            if ti_corr:
                ys = model_callable(trainable_params, static_params, x_summer, swe_or_h, ds=ys[2])
            else:
                ys = model_callable(trainable_params, static_params, x_summer, swe_or_h)
            smb_summer, swe_or_h = ys[0], ys[1]
            update_metrics(aux, smb_summer, y, x["outlines"], "summer")

            smb_annual = smb_winter + smb_summer
            
        update_metrics(aux, smb_annual, y, x["outlines"], "annual")
        if ti_corr:
            ds = ys[2]
            reg_ti_corr_acc = aux["reg_ti_corr_acc"]
            reg_ti_corr_n = aux["reg_ti_corr_n"]
            reg_ti_corr_acc, reg_ti_corr_n = update_ti_corr_regulariser(reg_ti_corr_acc, reg_ti_corr_n, ds, x["outlines"], lambda3, lambda4, lambda5)
            aux["reg_ti_corr_acc"] = reg_ti_corr_acc
            aux["reg_ti_corr_n"] = reg_ti_corr_n

    loss = 0.0
    for balance_code in ["annual", "winter", "summer"]:
        loss += (
            (aux[f"point_{balance_code}_mse"] if aux[f"point_{balance_code}_mse"] is not None else 0.0) + 
            lambda1 * (aux[f"total_{balance_code}_mse"] if aux[f"total_{balance_code}_mse"] is not None else 0.0)
        )
    
    reg = 0.0
    if ti:
        params = {**static_params, **trainable_params}
        aux["reg_ti"] = ti_regulariser(params, lambda2)
        reg += aux["reg_ti"]
    if ti_corr:
        aux["reg_ti_corr"] = aux["reg_ti_corr_acc"] / aux["reg_ti_corr_n"]
        reg += aux["reg_ti_corr"]
    
    if return_aux:
        return loss + reg, aux

    return loss + reg


def init_swe_or_h(
    trainable_params, 
    static_params, 
    model_callable, 
    glacier_name, 
    retrieve_corrector_predictors, 
    retrieve_facies, 
    last_numpy=True,
    device_to_prefetch=None,
):
    swe_or_h = None

    next_xy = dataloader.prefetch_xy(
        glacier_name, 
        constants.initialisation_period_start_year, 
        retrieve_corrector_predictors=retrieve_corrector_predictors, 
        retrieve_facies=retrieve_facies,
    )
    for year in range(constants.initialisation_period_start_year, constants.study_period_start_year):
        x, y = next_xy.get()
        next_xy = dataloader.prefetch_xy(
            glacier_name, year + 1, 
            retrieve_corrector_predictors=retrieve_corrector_predictors, 
            retrieve_facies=retrieve_facies,
            numpy=((year + 1 < constants.study_period_start_year) or last_numpy),
            device=device_to_prefetch,
        )

        if "annual" in x:
            x_annual = dict(x)
            x_annual.update(x_annual["annual"])
            ys = model_callable(trainable_params, static_params, x_annual, swe_or_h)
            swe_or_h = ys[1]
            
        else:
            x_winter = dict(x)
            x_winter.update(x_winter["winter"])
            ys = model_callable(trainable_params, static_params, x_winter, swe_or_h)
            swe_or_h = ys[1]

            x_summer = dict(x)
            x_summer.update(x_summer["summer"])
            ys = model_callable(trainable_params, static_params, x_summer, swe_or_h)
            swe_or_h = ys[1]
    
    return swe_or_h, next_xy


def update_metrics(aux, smb, y, outlines, balance_code):
    point_smb = y["point"]
    if point_smb is not None:
        point_smb = point_smb[point_smb.balance_code == balance_code]
        if len(point_smb) > 0:
            rows, cols = jnp.asarray(point_smb.row), jnp.asarray(point_smb.col)
            balances = jnp.asarray(point_smb.balance)
            weights = jnp.asarray(point_smb.weight)
            err_acc, w_acc = _point_mse_add(smb, rows, cols, balances, weights)
            aux[f"point_{balance_code}_n"] += w_acc
            aux[f"point_{balance_code}_error"] += err_acc
            
    total_smb = y["total"][f"{balance_code}_balance"].iloc[0]
    if not pandas.isna(total_smb):
        total_error = _total_mse_add(smb, outlines, total_smb)
        aux[f"total_{balance_code}_n"] += 1
        aux[f"total_{balance_code}_error"] += total_error

    if aux[f"point_{balance_code}_n"] > 0:
        aux[f"point_{balance_code}_mse"] = aux[f"point_{balance_code}_error"] / aux[f"point_{balance_code}_n"]
    else:
        aux[f"point_{balance_code}_mse"] = None
    
    if aux[f"total_{balance_code}_n"] > 0:
        aux[f"total_{balance_code}_mse"] = aux[f"total_{balance_code}_error"] / aux[f"total_{balance_code}_n"]
    else:
        aux[f"total_{balance_code}_mse"] = None

    return aux


@jax.jit
def _point_mse_add(smb, rows, cols, balances, weights):
    vals = smb[rows, cols]
    diff = vals - balances
    err_acc = jnp.sum(weights * jnp.square(diff))
    w_acc = jnp.sum(weights)
    return err_acc, w_acc


@jax.jit
def _total_mse_add(smb, outlines, target_total):
    denom = jnp.sum(outlines)
    num = jnp.sum(outlines * smb)
    err = jnp.square((num / denom - target_total))
    return err

    
# Elementary losses
def se(true, pred):
    return jnp.square(true - pred)


@jax.jit
def ti_regulariser(params, lambda2):
    init_ti_params = glaunti.ti_model.get_initial_model_parameters()
    init_ti_params = {**init_ti_params[0], **init_ti_params[1]}
    reg = 0.0
    for k, init_v in init_ti_params.items():
        reg += jnp.square((params[k] - init_v) / init_v)
    reg = lambda2 * reg
    return reg


@jax.jit
def update_ti_corr_regulariser(reg_ti_corr_acc, reg_ti_corr_n, ds, outlines, lambda3, lambda4, lambda5, eps=1e-6):
    d1, d2, d3, d4, d5 = ds[0], ds[1], ds[2], ds[3], ds[4]
    area = jnp.sum(outlines) + eps
    
    reg = lambda3 * (
        jnp.sum(outlines * jnp.square(d1)) / area + 
        jnp.sum(outlines * jnp.square(d2)) / area + 
        jnp.sum(outlines * jnp.square(d3)) / area +
        jnp.sum(outlines * jnp.square(d4)) / area +
        jnp.sum(outlines * jnp.square(d5)) / area 
    ) 

    d1_mean = jnp.sum(outlines * d1) / area
    d2_mean = jnp.sum(outlines * d2) / area
    d3_mean = jnp.sum(outlines * d3) / area
    d4_mean = jnp.sum(outlines * d4) / area
    d5_mean = jnp.sum(outlines * d5) / area
    reg += lambda4 * (
        jnp.sum(outlines * se(d1, d1_mean)) / area +
        jnp.sum(outlines * se(d2, d2_mean)) / area +
        jnp.sum(outlines * se(d3, d3_mean)) / area +
        jnp.sum(outlines * se(d4, d4_mean)) / area +
        jnp.sum(outlines * se(d5, d5_mean)) / area 
    )
                
    reg += lambda5 * jnp.sum(outlines * se(d2, d3)) / area
    
    reg_ti_corr_acc += reg
    reg_ti_corr_n += 1
    return reg_ti_corr_acc, reg_ti_corr_n
