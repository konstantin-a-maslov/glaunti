import jax
import jax.numpy as jnp
import utils.activations
import constants


def run_model(trainable_params, static_params, x, return_series=False, t_pos_shift=0.0):
    """
    Run the model over time series of precipitation and temperature with constrained parameters.

    Args:
        trainable_params (PyTree): Set of trainable parameters
        static_params (PyTree): Set of non-trainable parameters
        x (PyTree): Dictionary with keys 'precipitation' and 'temperature', each of shape (T, H, W)
        t_pos_shift (jnp.ndarray): Temperature shift for positive temperature computation (H, W)

    Returns:
        smb (jnp.ndarray): Surface mass balance predictions (H, W) or (T, H, W) if return_series
    """
    params = {**static_params, **trainable_params}
    params = resolve_param_constraints(params) # Extract params, impose constraints where needed

    smb = run_model_unconstrained(params, x, return_series, t_pos_shift=t_pos_shift)
    return smb


def run_model_unconstrained(params, x, return_series=False, t_pos_shift=0.0):
    """
    Run the model over time series of precipitation and temperature with unconstrained parameters.

    Args:
        params (PyTree): Set of parameters
        x (PyTree): Dictionary with keys 'precipitation' and 'temperature', each of shape (T, H, W)
        t_pos_shift (jnp.ndarray): Temperature shift for positive temperature computation (H, W)

    Returns:
        smb (jnp.ndarray): Surface mass balance predictions (H, W) or (T, H, W) if return_series
    """
    precipitation = x["precipitation"]
    temperature = x["temperature"]
    time, _, _ = temperature.shape
    w = jnp.ones((time, )).at[0].set(0.5).at[-1].set(0.5)


    if return_series:
        smb = jax.vmap(
            lambda precipitation_d, temperature_d, w_d: run_one_day_iteration(
                params, precipitation_d, temperature_d, w_d, t_pos_shift=t_pos_shift
            )
        )(precipitation, temperature, w)
        
    else:
        inputs = (precipitation, temperature, w)
        
        def scan_step(smb_acc, inputs_d):
            precipitation_d, temperature_d, w_d = inputs_d
            smb = run_one_day_iteration(params, precipitation_d, temperature_d, w_d, t_pos_shift=t_pos_shift)
            return smb_acc + smb, None 
        scan_step = jax.remat(scan_step)
        
        smb_acc = jnp.zeros_like(precipitation[0])
        smb, _ = jax.lax.scan(scan_step, smb_acc, inputs)
        
    return smb


def run_one_day_iteration(params, precipitation, temperature, w_d, t_pos_shift=0.0):
    """
    Make one-day model timestep.

    Args:
        params (PyTree): Set of parameters
        precipitation (jnp.ndarray): Precipitation (H, W)
        temperature (jnp.ndarray): Temperature (H, W)
        w_d (jnp.ndarray): Weight to avoid double-counting edge days (scalar 1 or 0.5)
        t_pos_shift (jnp.ndarray): Temperature shift for positive temperature computation (H, W)

    Returns:
        smb (jnp.ndarray): Surface mass balance prediction (H, W)
    """ 
    prec_scale = params["prec_scale"]
    ddf = params["ddf"]
    snow_to_rain_steepness = params["snow_to_rain_steepness"]
    snow_to_rain_centre = params["snow_to_rain_centre"]
    t_softplus_sharpness = params["t_softplus_sharpness"]
    
    solid_precipitation = precipitation * jax.nn.sigmoid(snow_to_rain_steepness * (snow_to_rain_centre - temperature))

    t_pos = utils.activations.softplus_t(t_softplus_sharpness, temperature + t_pos_shift)
    smb = w_d * (prec_scale * solid_precipitation - ddf * t_pos)
    return smb


def get_initial_model_parameters(key=None):
    """
    Initialise TI model parameters.
        
    Returns:
        PyTree: Set of initial parameter values.
    """
    ddf_snow = jnp.exp(constants.ddf_snow_log)
    ddf_relative_ice = jnp.exp(constants.ddf_relative_ice_minus_one_log) + 1.0
    ddf_ice = ddf_snow * ddf_relative_ice
    ddf_init = 0.5 * (ddf_snow + ddf_ice) # avg snow and ice DDFs
    
    trainable_params = dict(
        prec_scale_log=constants.prec_scale_log,
        ddf_log=jnp.log(ddf_init), 
        snow_to_rain_steepness_log=constants.snow_to_rain_steepness_log,
        snow_to_rain_centre=constants.snow_to_rain_centre,
    )
    
    static_params = dict(
        t_softplus_sharpness_log=constants.t_softplus_sharpness_log,
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
        ddf=jnp.exp(params["ddf_log"]),
        snow_to_rain_steepness=jnp.exp(params["snow_to_rain_steepness_log"]),
        t_softplus_sharpness=jnp.exp(params["t_softplus_sharpness_log"]),
        # No constraints
        snow_to_rain_centre=params["snow_to_rain_centre"],
    )
    return params
