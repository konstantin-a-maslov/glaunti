import jax
import jax.numpy as jnp
import utils.activations
import constants


@jax.jit
def run_model(trainable_params, non_trainable_params, x, initial_swe=None):
    """
    Run the model over time series of precipitation and temperature with constrained parameters.

    Args:
        trainable_params (PyTree): Set of trainable parameters
        non_trainable_params (PyTree): Set of non-trainable parameters
        x (PyTree): Model inputs, precipitation and temperature with shapes (T, H, W)

    Returns:
        jnp.ndarray: Surface mass balance predictions (T, H, W).
    """
    params = {**non_trainable_params, **trainable_params}
    # Extract params, impose constraints where needed
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
    
    return run_model_unconstrained(params, x, initial_swe=None)


@jax.jit
def run_model_unconstrained(params, x, initial_swe=None):
    """
    Run the model over time series of precipitation and temperature with unconstrained parameters.

    Args:
        params (PyTree): Set of parameters
        x (PyTree): Dictionary with keys 'precipitation' and 'temperature', each of shape (T, H, W)

    Returns:
        jnp.ndarray: Surface mass balance predictions (T, H, W).
    """
    precipitation = x["precipitation"]
    temperature = x["temperature"]

    if initial_swe is None:
        _, height, width = temperature.shape
        initial_swe = jnp.zeros((height, width))

    def scan_step(prev_swe, inputs_t):
        precipitation_t, temperature_t = inputs_t
        swe, smb = run_one_day_iteration(params, precipitation_t, temperature_t, prev_swe)
        return swe, smb 

    inputs = (precipitation, temperature)
    _, smb_series = jax.lax.scan(scan_step, initial_swe, inputs)
    
    return smb_series


@jax.jit
def run_one_day_iteration(params, precipitation, temperature, prev_swe):
    """
    Make one-day model timestep.

    Args:
        params (PyTree): Set of parameters
        precipitation (jnp.ndarray): Precipitation (H, W)
        temperature (jnp.ndarray): Temperature (H, W)
        prev_swe (jnp.ndarray): Accumulated snow water equivalent (H, W)

    Returns:
        jnp.ndarray: Surface mass balance prediction (H, W).
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

    t_pos = utils.activations.softplus_t(t_softplus_sharpness, temperature)
    fsc = utils.activations.hill_curve(snow_depletion_steepness, snow_depletion_centre, prev_swe)
    t_pos_snow = fsc * t_pos
    t_pos_ice = (1 - fsc) * t_pos
    
    swe = utils.activations.softplus_t(
        swe_softplus_sharpness, 
        prev_swe + prec_scale * solid_precipitation - ddf_snow * t_pos_snow
    )
    smb = prec_scale * solid_precipitation - ddf_snow * (t_pos_snow + ddf_relative_ice * t_pos_ice)
    
    return swe, smb


def get_initial_model_parameters():
    """
    Initialise TI model parameters.
    
    Based on physics-plausible and literature priors:
        https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2015.00054/full
        https://www.sciencedirect.com/science/article/pii/S0022169403002579
        https://www.science.org/doi/10.1126/science.abo1324
        https://www.nature.com/articles/s41467-018-03629-7
        https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2012JD018178

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
    
    non_trainable_params = dict(
        t_softplus_sharpness_log=constants.t_softplus_sharpness_log,
        swe_softplus_sharpness_log=constants.swe_softplus_sharpness_log,
    )
    
    return trainable_params, non_trainable_params
