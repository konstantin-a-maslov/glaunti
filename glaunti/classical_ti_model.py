import jax
import jax.numpy as jnp
import constants


def run_model(trainable_params, static_params, x, initial_swe=None, return_series=False, t_pos_shift=0.0):
    """
    Run the model over time series of precipitation and temperature with unconstrained parameters.

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
    snow_to_rain_left_bound = params["snow_to_rain_left_bound"]
    snow_to_rain_length = params["snow_to_rain_length"]
    snow_depletion_centre = params["snow_depletion_centre"]
    
    solid_precipitation = precipitation * jnp.clip(
        (snow_to_rain_left_bound + snow_to_rain_length - temperature) / snow_to_rain_length, 
        0.0, 1.0
    )

    t_pos = jax.nn.relu(temperature + t_pos_shift)
    fsc = jnp.where(prev_swe > snow_depletion_centre, 1.0, 0.0)
    t_pos_snow = fsc * t_pos
    t_pos_ice = (1 - fsc) * t_pos
    
    swe = jax.nn.relu(prev_swe + w_d * (prec_scale * solid_precipitation - ddf_snow * t_pos_snow))
    smb = w_d * (prec_scale * solid_precipitation - ddf_snow * (t_pos_snow + ddf_relative_ice * t_pos_ice))
    return smb, swe


def get_initial_model_parameters(key=None):
    """
    Initialise classical TI model parameters.
        
    Returns:
        PyTree: Set of initial parameter values.
    """
    trainable_params = dict(
        prec_scale=jnp.exp(constants.prec_scale_log),
        ddf_snow=jnp.exp(constants.ddf_snow_log),
        ddf_relative_ice=jnp.exp(constants.ddf_relative_ice_minus_one_log) + 1.0,
        snow_to_rain_left_bound=constants.snow_to_rain_centre - 2.0 / jnp.exp(constants.snow_to_rain_steepness_log), # match centre and centre slope
        snow_to_rain_length=4.0 / jnp.exp(constants.snow_to_rain_steepness_log),
        snow_depletion_centre=jnp.exp(constants.snow_depletion_centre_log),
    )
    
    static_params = dict()
    
    return trainable_params, static_params
