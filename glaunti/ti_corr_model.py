import jax
import jax.numpy as jnp
import equinox as eqx

import glaunti.ti_model
from glaunti.corr_submodules import Corrector2dBranch, Corrector1dBranch
import utils.activations
import constants


class Corrector(eqx.Module):
    branch_2d: Corrector2dBranch
    branch_1d: Corrector1dBranch
    reproj_1d_to_2d_conv: eqx.nn.Conv1d
    out_conv: eqx.nn.Conv2d

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
        keys = jax.random.split(key, 4)
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
            n_filters_1d_branch, n_filters_2d_branch, kernel_size=1, stride=1, padding=0, key=next(keys)
        )
        self.out_conv = eqx.nn.Conv2d(
            n_filters_2d_branch, output_size, kernel_size=1, stride=1, padding=0, key=next(keys)
        )

    def __call__(self, x, initial_h=None, return_series=False):
        y_2d_branch = self._compute_2d_branch(x)
        y_1d_branch = self._compute_1d_branch(x)
        y_1d_branch_reproj = self.reproj_1d_to_2d_conv(y_1d_branch)
        y_1d_branch_reproj = jnp.expand_dims(y_1d_branch_reproj, 2)
        y_fused = y_2d_branch + y_1d_branch_reproj
        ds = self.out_conv(y_fused)
        return ds

    def _compute_2d_branch(self, x):
        corrector_fields = x["corrector_fields"]
        _, orig_h, orig_w = corrector_fields.shape
        corrector_fields_padded, _, _ = self._pad_to_multiple_hw(corrector_fields)
        y_2d_branch = self.branch_2d(corrector_fields_padded)
        y_2d_branch_cropped = self._crop_hw(y_2d_branch, orig_h, orig_w)
        return y_2d_branch_cropped

    def _compute_1d_branch(self, x):
        climate_monthly = x["climate_monthly"]
        y_1d_branch = self.branch_1d(climate_monthly)
        return y_1d_branch

    def _pad_to_multiple_hw(self, x):
        BUCKETS = (
            (32, 32), (64, 64), (128, 128),
            (256, 256), (384, 384), (512, 512), 
            (768, 768), (1024, 1024),
        )
        _, h, w = x.shape
        for bh, bw in BUCKETS:
            if (h <= bh) and (w <= bw):
                pad_h = bh - h
                pad_w = bw - w
                break
        pad_spec = ((0, 0), (0, pad_h), (0, pad_w))
        x = jnp.pad(x, pad_spec, mode="constant", constant_values=0)
        return x, pad_h, pad_w

    def _crop_hw(self, x, orig_h, orig_w):
        return x[:, :orig_h, :orig_w]
    

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
    if ds is None:
        ds = params["corrector"](x)
    d1, d2, d3, d4 = ds[0], ds[1], ds[2], ds[3]
    ti_params = glaunti.ti_model.resolve_param_constraints(params)
    
    if initial_swe is None:
        height, width = d1.shape
        initial_swe = jnp.full((height, width), fill_value=ti_params["snow_depletion_centre"])

    # implement corrections    
    ti_params["prec_scale"] = ti_params["prec_scale"] * jnp.exp(d1)
    ti_params["ddf_snow"] = ti_params["ddf_snow"] * jnp.exp(d2)
    ti_params["ddf_relative_ice"] = ti_params["ddf_relative_ice"] * jnp.exp(d3) # explicitly allow < 1.0 here
    initial_swe = utils.activations.softplus_t(ti_params["swe_softplus_sharpness"], initial_swe + d4)

    smb, swe = glaunti.ti_model.run_model_unconstrained(ti_params, x, initial_swe, return_series)

    if return_corrections:
        return smb, swe, (d1, d2, d3, d4)

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
