import jax
import jax.numpy as jnp
import pandas

import glaunti.ti_model
import dataloader.dataloader as dataloader
import constants


def loss(
    trainable_params, static_params, model_callable, 
    glacier, 
    ti=False, ti_corr=False, retrieve_facies=False, 
    lambda1=constants.default_lambda1, 
    lambda2=constants.default_lambda2, lambda3=constants.default_lambda3, lambda4=constants.default_lambda4, 
    return_aux=True,
):   
    glacier_name, max_year, aux = glacier["name"], glacier["max_year"], {}
    for balance_code in ["annual", "winter", "summer"]:
        aux.update({
            f"total_{balance_code}_error": 0, f"total_{balance_code}_n": 0, 
            f"point_{balance_code}_error": 0, f"point_{balance_code}_n": 0,
        })
    swe_or_h, next_xy = init_swe_or_h(
        trainable_params, static_params, model_callable, glacier_name, ti_corr, retrieve_facies
    )
    
    for year in range(constants.study_period_start_year, max_year + 1):
        x, y = next_xy.get()
        next_xy = dataloader.prefetch_xy(
            glacier_name, year + 1, 
            retrieve_corrector_predictors=ti_corr, 
            retrieve_facies=retrieve_facies,
        )
        x = dataloader.x_to_raw_numpy(x)
        
        if "annual" in x:
            x_annual = {k: v for k, v in x.items() if k != "annual"}
            x_annual.update(x["annual"])
            if ti_corr:
                smb_annual, swe_or_h, ds = model_callable(trainable_params, static_params, x_annual, swe_or_h)
            else:
                smb_annual, swe_or_h = model_callable(trainable_params, static_params, x_annual, swe_or_h)
        else:
            x_winter = {k: v for k, v in x.items() if k not in {"winter", "summer"}}
            x_winter.update(x["winter"])
            if ti_corr:
                smb_winter, swe_or_h, _ = model_callable(trainable_params, static_params, x_winter, swe_or_h)
            else:
                smb_winter, swe_or_h = model_callable(trainable_params, static_params, x_winter, swe_or_h)
            update_metrics(aux, smb_winter, y, x["outlines"], "winter")

            x_summer = {k: v for k, v in x.items() if k not in {"winter", "summer"}}
            x_summer.update(x["summer"])
            if ti_corr:
                smb_summer, swe_or_h, ds = model_callable(trainable_params, static_params, x_summer, swe_or_h)
            else: 
                smb_summer, swe_or_h = model_callable(trainable_params, static_params, x_summer, swe_or_h)
            update_metrics(aux, smb_summer, y, x["outlines"], "summer")

            smb_annual = smb_winter + smb_summer
        update_metrics(aux, smb_annual, y, x["outlines"], "annual")
        if ti_corr:
            update_ti_corr_regulariser(aux, ds, lambda3, lambda4)

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
        reg += aux["reg_ti_corr"]
    
    if return_aux:
        return loss + reg, aux

    return loss + reg


def init_swe_or_h(trainable_params, static_params, model_callable, glacier_name, retrieve_corrector_predictors, retrieve_facies):
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
        )
        x = dataloader.x_to_raw_numpy(x)

        if "annual" in x:
            x_annual = {k: v for k, v in x.items() if k != "annual"}
            x_annual.update(x["annual"])
            if retrieve_corrector_predictors:
                _, swe_or_h, _ = model_callable(trainable_params, static_params, x_annual, swe_or_h)
            else:
                _, swe_or_h = model_callable(trainable_params, static_params, x_annual, swe_or_h)
            
        else:
            x_winter = {k: v for k, v in x.items() if k not in {"winter", "summer"}}
            x_winter.update(x["winter"])
            if retrieve_corrector_predictors:
                _, swe_or_h, _ = model_callable(trainable_params, static_params, x_winter, swe_or_h)
            else:
                _, swe_or_h = model_callable(trainable_params, static_params, x_winter, swe_or_h)

            x_summer = {k: v for k, v in x.items() if k not in {"winter", "summer"}}
            x_summer.update(x["summer"])
            if retrieve_corrector_predictors:
                _, swe_or_h, _ = model_callable(trainable_params, static_params, x_summer, swe_or_h)
            else:
                _, swe_or_h = model_callable(trainable_params, static_params, x_summer, swe_or_h)
    
    return swe_or_h, next_xy


def update_metrics(aux, smb, y, outlines, balance_code):
    point_smb = y["point"]
    if point_smb is not None:
        point_smb = point_smb[point_smb.balance_code == balance_code]
        if len(point_smb) > 0:
            for point in point_smb.itertuples():
                aux[f"point_{balance_code}_n"] += point.weight
                point_error = se(smb[point.row, point.col], point.balance)
                aux[f"point_{balance_code}_error"] += (point.weight * point_error)
            
    total_smb = y["total"][f"{balance_code}_balance"].iloc[0]
    if not pandas.isna(total_smb):
        aux[f"total_{balance_code}_n"] += 1
        total_error = se(jnp.sum(outlines * smb) / jnp.sum(outlines), total_smb)
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

    
# Elementary losses
def se(true, pred):
    return (true - pred)**2


def ti_regulariser(params, lambda2):
    init_ti_params = glaunti.ti_model.get_initial_model_parameters()
    init_ti_params = {**init_ti_params[0], **init_ti_params[1]}
    reg = 0.0
    for k, init_v in init_ti_params.items():
        reg += ((params[k] - init_v) / init_v)**2
    reg = lambda2 * reg
    return reg


def update_ti_corr_regulariser(aux, ds, lambda3, lambda4):
    d1, d2, d3, d4 = ds[0], ds[1], ds[2], ds[3]
    if not "reg_ti_corr" in aux:
        aux["reg_ti_corr_acc"] = 0
        aux["reg_ti_corr_n"] = 0
    reg = lambda3 * (jnp.mean(d1**2) + jnp.mean(d2**2) + jnp.mean(d3**2) + jnp.mean(d4**2)) + \
        lambda4 * jnp.mean(se(d2, d3))
    aux["reg_ti_corr_acc"] += reg
    aux["reg_ti_corr_n"] += 1
    aux["reg_ti_corr"] = aux["reg_ti_corr_acc"] / aux["reg_ti_corr_n"]
    return aux
