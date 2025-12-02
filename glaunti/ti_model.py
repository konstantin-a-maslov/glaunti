import jax
import jax.numpy as jnp
import utils.activations
import constants


def run_model(trainable_params, static_params, x, initial_swe=None, return_series=False, t_pos_shift=0.0):
    """
    Run the model over time series of precipitation and temperature with constrained parameters.

    Args:
        trainable_params (PyTree): Set of trainable parameters
        static_params (PyTree): Set of non-trainable parameters
        x (PyTree): Dictionary with keys 'precipitation' and 'temperature', each of shape (T, H, W)
        initial_swe (jnp.ndarray, Optional): Initial snow water equivalent (H, W)
        t_pos_shift (jnp.ndarray): Temperature shift for positive temperature computation (H, W)

    Returns:
        smb (jnp.ndarray): Surface mass balance predictions (H, W) or (T, H, W) if return_series
        swe (jnp.ndarray): Updated snow water equivalent (H, W)
    """
    params = {**static_params, **trainable_params}
    params = resolve_param_constraints(params) # Extract params, impose constraints where needed

    smb, swe = run_model_unconstrained(params, x, initial_swe, return_series, t_pos_shift=t_pos_shift)
    return smb, swe


def run_model_unconstrained(params, x, initial_swe=None, return_series=False, t_pos_shift=0.0):
    """
    Run the model over time series of precipitation and temperature with unconstrained parameters.

    Args:
        params (PyTree): Set of parameters
        x (PyTree): Dictionary with keys 'precipitation' and 'temperature', each of shape (T, H, W)
        initial_swe (jnp.ndarray, Optional): Initial snow water equivalent (H, W)
        t_pos_shift (jnp.ndarray): Temperature shift for positive temperature computation (H, W)

    Returns:
        smb (jnp.ndarray): Surface mass balance predictions (H, W) or (T, H, W) if return_series
        swe (jnp.ndarray): Updated snow water equivalent (H, W)
    """
    precipitation = x["precipitation"]
    temperature = x["temperature"]
    time, _, _ = temperature.shape
    w = jnp.ones((time, )).at[0].set(0.5).at[-1].set(0.5)
    inputs = (precipitation, temperature, w)

    if initial_swe is None:
        _, height, width = temperature.shape
        initial_swe = jnp.full((height, width), fill_value=params["snow_depletion_centre"])

    if return_series:
        def scan_step(prev_swe, inputs_d):
            precipitation_d, temperature_d, w_d = inputs_d
            smb, swe = run_one_day_iteration(params, precipitation_d, temperature_d, prev_swe, w_d, t_pos_shift=t_pos_shift)
            return swe, smb 
        scan_step = jax.remat(scan_step)
        
        swe, smb = jax.lax.scan(scan_step, initial_swe, inputs)
        
    else:
        def scan_step(carry, inputs_d):
            precipitation_d, temperature_d, w_d = inputs_d
            prev_swe, smb_acc = carry
            smb, swe = run_one_day_iteration(params, precipitation_d, temperature_d, prev_swe, w_d, t_pos_shift=t_pos_shift)
            return (swe, smb_acc + smb), None 
        scan_step = jax.remat(scan_step)
        
        carry = (initial_swe, jnp.zeros_like(initial_swe))
        (swe, smb), _ = jax.lax.scan(scan_step, carry, inputs)
        
    return smb, swe


def run_one_day_iteration(params, precipitation, temperature, prev_swe, w_d, t_pos_shift=0.0):
    """
    Make one-day model timestep.

    Args:
        params (PyTree): Set of parameters
        precipitation (jnp.ndarray): Precipitation (H, W)
        temperature (jnp.ndarray): Temperature (H, W)
        prev_swe (jnp.ndarray): Accumulated snow water equivalent (H, W)
        w_d (jnp.ndarray): Weight to avoid double-counting edge days (scalar 1 or 0.5)
        t_pos_shift (jnp.ndarray): Temperature shift for positive temperature computation (H, W)

    Returns:
        smb (jnp.ndarray): Surface mass balance prediction (H, W)
        swe (jnp.ndarray): Updated snow water equivalent (H, W)
    """ 
    prec_scale = params["prec_scale"]
    ddf_snow = params["ddf_snow"]
    ddf_relative_ice = params["ddf_relative_ice"]
    snow_to_rain_steepness = params["snow_to_rain_steepness"]
    snow_to_rain_centre = params["snow_to_rain_centre"]
    snow_depletion_steepness = params["snow_depletion_steepness"]
    snow_depletion_centre = params["snow_depletion_centre"]
    t_softplus_sharpness = params["t_softplus_sharpness"]
    swe_softplus_sharpness = params["swe_softplus_sharpness"]
    
    solid_precipitation = precipitation * jax.nn.sigmoid(snow_to_rain_steepness * (snow_to_rain_centre - temperature))

    t_pos = utils.activations.softplus_t(t_softplus_sharpness, temperature + t_pos_shift)
    fsc = utils.activations.hill_curve(snow_depletion_steepness, snow_depletion_centre, prev_swe)
    t_pos_snow = fsc * t_pos
    t_pos_ice = (1 - fsc) * t_pos
    
    swe = utils.activations.softplus_t(
        swe_softplus_sharpness, 
        prev_swe + w_d * (prec_scale * solid_precipitation - ddf_snow * t_pos_snow)
    )
    smb = w_d * (prec_scale * solid_precipitation - ddf_snow * (t_pos_snow + ddf_relative_ice * t_pos_ice))
    return smb, swe


def get_initial_model_parameters(key=None):
    """
    Initialise TI model parameters.
        
    Returns:
        PyTree: Set of initial parameter values.
    """
    trainable_params = dict(
        prec_scale_log=constants.prec_scale_log,
        ddf_snow_log=constants.ddf_snow_log,
        ddf_relative_ice_minus_one_log=constants.ddf_relative_ice_minus_one_log,
        snow_to_rain_steepness_log=constants.snow_to_rain_steepness_log,
        snow_to_rain_centre=constants.snow_to_rain_centre,
        snow_depletion_steepness_log=constants.snow_depletion_steepness_log,
        snow_depletion_centre_log=constants.snow_depletion_centre_log,
    )
    
    static_params = dict(
        t_softplus_sharpness_log=constants.t_softplus_sharpness_log,
        swe_softplus_sharpness_log=constants.swe_softplus_sharpness_log,
    )
    
    return trainable_params, static_params


def resolve_param_constraints(params):
    """
    Resolve parameter constraints.

    Args:
        params (PyTree): Set of log-transformed parameters

    Returns:
        PyTree: Set of constrained parameters
    """
    params = dict(
        # > 0
        prec_scale=jnp.exp(params["prec_scale_log"]), 
        ddf_snow=jnp.exp(params["ddf_snow_log"]),
        snow_to_rain_steepness=jnp.exp(params["snow_to_rain_steepness_log"]),
        snow_depletion_steepness=jnp.exp(params["snow_depletion_steepness_log"]),
        snow_depletion_centre=jnp.exp(params["snow_depletion_centre_log"]),
        t_softplus_sharpness=jnp.exp(params["t_softplus_sharpness_log"]),
        swe_softplus_sharpness=jnp.exp(params["swe_softplus_sharpness_log"]),
        # > 1
        ddf_relative_ice=jnp.exp(params["ddf_relative_ice_minus_one_log"]) + 1.0,
        # No constraints
        snow_to_rain_centre=params["snow_to_rain_centre"],
    )
    return params
