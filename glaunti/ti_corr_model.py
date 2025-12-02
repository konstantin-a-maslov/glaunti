import jax
import jax.numpy as jnp
import equinox as eqx

import glaunti.ti_model
from glaunti.corr_submodules import Corrector2dBranch, Corrector1dBranch, DoubleConv2d
import utils.activations
import constants


class Corrector(eqx.Module):
    branch_2d: Corrector2dBranch
    branch_1d: Corrector1dBranch
    reproj_1d_to_2d_conv: eqx.nn.Conv1d
    fuse_conv: DoubleConv2d
    out_conv: eqx.nn.Conv2d
    scaler_2d_branch: float = eqx.field(static=True)
    scaler: float = eqx.field(static=True)
    output_size: int = eqx.field(static=True)

    def __init__(
        self, 
        input_size_2d, 
        input_size_1d,
        output_size,
        n_filters_2d_branch,
        n_stages_2d_branch,
        n_filters_1d_branch,
        n_stages_1d_branch,
        key
    ):
        keys = jax.random.split(key, 5)
        keys = iter(keys)
        
        self.branch_2d = Corrector2dBranch(
            input_size_2d,
            n_filters_2d_branch,
            n_stages_2d_branch,
            key=next(keys),
        )
        self.branch_1d = Corrector1dBranch(
            input_size_1d,
            n_filters_1d_branch,
            n_stages_1d_branch,
            key=next(keys),
        )
        self.reproj_1d_to_2d_conv = eqx.nn.Conv1d(
            n_filters_1d_branch, n_filters_2d_branch, kernel_size=1, stride=1, use_bias=False, padding=0, key=next(keys)
        )
        self.fuse_conv = DoubleConv2d(
            n_filters_2d_branch, n_filters_2d_branch, key=next(keys)
        )
        self.out_conv = eqx.nn.Conv2d(
            n_filters_2d_branch, output_size, kernel_size=1, stride=1, use_bias=True, padding=0, key=next(keys)
        )
        self.scaler_2d_branch = constants.scaler_2d_branch
        self.scaler = constants.corrector_scaler
        self.output_size = output_size

    def __call__(self, x, initial_h=None, return_series=False):
        _, h, w = x["corrector_fields"].shape
        y_2d_branch = self._compute_2d_branch(x)
        y_1d_branch = self._compute_1d_branch(x)
        y_1d_branch_reproj = self.reproj_1d_to_2d_conv(y_1d_branch)
        y_1d_branch_reproj = jnp.expand_dims(y_1d_branch_reproj, 2)
        y_fused = self.scaler_2d_branch * y_2d_branch + y_1d_branch_reproj
        y_fused = self.fuse_conv(y_fused)
        ds = self.out_conv(y_fused) * self.scaler
        ds = jax.image.resize(ds, shape=(self.output_size, h, w), method="bilinear", antialias=False)
        return ds

    def _compute_2d_branch(self, x):
        corrector_fields = x["corrector_fields"]
        y_2d_branch = self.branch_2d(corrector_fields)
        return y_2d_branch

    def _compute_1d_branch(self, x):
        climate_monthly = x["climate_monthly"]
        y_1d_branch = self.branch_1d(climate_monthly)
        return y_1d_branch
    

def run_model(trainable_params, static_params, x, initial_swe=None, return_series=False, return_corrections=False, ds=None):
    """
    Run the model over time series of precipitation and temperature.

    Args:
        trainable_params (PyTree): Set of trainable parameters
        static_params (PyTree): Set of non-trainable parameters
        x (PyTree): Dictionary with keys 'precipitation' (T, H, W), 'temperature' (T, H, W), 'corrector_fields' (C2d, H, W) and 'climate_monthly' (C1d, T_sub)
        initial_swe (jnp.ndarray, Optional): Initial snow water equivalent (H, W)

    Returns:
        smb_series (jnp.ndarray): Surface mass balance predictions (H, W) or (T, H, W) if return_series
        swe (jnp.ndarray): Updated snow water equivalent (H, W)
        corrections (tuple of jnp.ndarray): Corrections of TI params (4, H, W)
    """
    params = {**static_params, **trainable_params}
    ti_params = glaunti.ti_model.resolve_param_constraints(params)

    if initial_swe is None:
        if "annual" in x:
            _, height, width = x["annual"]["temperature"].shape
        else:
            _, height, width = x["winter"]["temperature"].shape
        initial_swe = jnp.full((height, width), fill_value=ti_params["snow_depletion_centre"])

    if ds is None:
        extra = _get_extra_vars(ti_params, x, initial_swe)
        x = dict(x)
        x_corr = {
            "corrector_fields": jnp.concatenate([extra, x["corrector_fields"]]),
            "climate_monthly":  x["climate_monthly"],
        }
        ds = params["corrector"](x_corr)
    d1, d2, d3, d4, d5 = ds[0], ds[1], ds[2], ds[3], ds[4]

    # implement corrections    
    ti_params["prec_scale"] = ti_params["prec_scale"] * jnp.exp(d1)
    ti_params["ddf_snow"] = ti_params["ddf_snow"] * jnp.exp(d2)
    ti_params["ddf_relative_ice"] = ti_params["ddf_relative_ice"] * jnp.exp(d3) # explicitly allow < 1.0 here
    ti_params["snow_to_rain_centre"] = ti_params["snow_to_rain_centre"] + d4

    x_curr = {
        "temperature": x["temperature"], 
        "precipitation": x["precipitation"],
    }
    smb, swe = glaunti.ti_model.run_model_unconstrained(ti_params, x_curr, initial_swe, return_series, t_pos_shift=d5)

    if return_corrections:
        return smb, swe, (d1, d2, d3, d4, d5)

    return smb, swe


def get_initial_model_parameters(ti_params=None, ti_params_static=True, key=None):
    """
    Initialise TI_corrected parameters.

    Args:
        ti_params (PyTree): TI parameters
        ti_params_static (bool): Whether ti_params should be treated as static or trainable
        
    Returns:
        trainable_params (Corrector): Corrector model and ti_params if not ti_params_static
        static_params (dict): Default or provided TI parameters if ti_params_static
    """
    if key is None:
        key = jax.random.PRNGKey(constants.default_rng_key)

    model = Corrector(
        input_size_2d=constants.corrector_field_size, 
        input_size_1d=constants.climate_monthly_size,
        output_size=constants.corrector_output_size,
        n_filters_2d_branch=constants.n_filters_2d_branch,
        n_stages_2d_branch=constants.n_stages_2d_branch,
        n_filters_1d_branch=constants.n_filters_1d_branch,
        n_stages_1d_branch=constants.n_stages_1d_branch,
        key=key,
    )
    
    if ti_params is None:
        ti_trainable_params, ti_static_params = glaunti.ti_model.get_initial_model_parameters(key=key)
    else:
        ti_trainable_params, ti_static_params = ti_params

    if ti_params_static:
        return {"corrector": model}, {**ti_static_params, **ti_trainable_params}  # eqx's Module is PyTree

    params = {"corrector": model}
    params.update(ti_trainable_params)
    return params, ti_static_params


def _get_extra_vars(ti_params, x, initial_swe):
    if "annual" in x:
        smb0, _ = glaunti.ti_model.run_model_unconstrained(ti_params, x["annual"], initial_swe, return_series=False)
        x_with_dt = _add_dt(x["annual"])
        x_with_dp = _add_dp(x["annual"])
        smb0_dt, _ = glaunti.ti_model.run_model_unconstrained(ti_params, x_with_dt, initial_swe, return_series=False)
        smb0_dp, _ = glaunti.ti_model.run_model_unconstrained(ti_params, x_with_dp, initial_swe, return_series=False)
    else:
        smb0_w, swe_w = glaunti.ti_model.run_model_unconstrained(ti_params, x["winter"], initial_swe, return_series=False)
        smb0_s, _ = glaunti.ti_model.run_model_unconstrained(ti_params, x["summer"], swe_w, return_series=False)
        smb0 = smb0_w + smb0_s
        x_w_with_dt = _add_dt(x["winter"])
        x_s_with_dt = _add_dt(x["summer"])
        smb0_w_dt, swe_w_dt = glaunti.ti_model.run_model_unconstrained(ti_params, x_w_with_dt, initial_swe, return_series=False)
        smb0_s_dt, _ = glaunti.ti_model.run_model_unconstrained(ti_params, x_s_with_dt, swe_w_dt, return_series=False)
        smb0_dt = smb0_w_dt + smb0_s_dt
        x_w_with_dp = _add_dp(x["winter"])
        x_s_with_dp = _add_dp(x["summer"])
        smb0_w_dp, swe_w_dp = glaunti.ti_model.run_model_unconstrained(ti_params, x_w_with_dp, initial_swe, return_series=False)
        smb0_s_dp, _ = glaunti.ti_model.run_model_unconstrained(ti_params, x_s_with_dp, swe_w_dp, return_series=False)
        smb0_dp = smb0_w_dp + smb0_s_dp

    smb0_dt = smb0_dt - smb0
    smb0_dp = smb0_dp - smb0
    extra = jnp.stack([
        jax.lax.stop_gradient(smb0 / constants.initial_guess_normaliser), 
        jax.lax.stop_gradient(smb0_dt), 
        jax.lax.stop_gradient(smb0_dp),
    ])
    return extra
    
    
def _add_dt(x, dt=0.5):
    new_x = {
        "temperature": x["temperature"] + dt,
        "precipitation": x["precipitation"],
    }
    return new_x


def _add_dp(x, dp=0.05):
    new_x = {
        "temperature": x["temperature"],
        "precipitation": x["precipitation"] * (1 + dp),
    }
    return new_x
