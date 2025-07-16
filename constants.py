import jax.numpy as jnp


# Data
data_folder = "data"

# Conversions
temperature_lapse_rate = 0.0065 # deg C per km

# Initial TI parameter values
prec_scale_log = jnp.log(1.4)
ddf_snow_log = jnp.log(0.0049)
ddf_relative_ice_minus_one_log = jnp.log(0.5)
snow_to_rain_steepness_log = jnp.log(1.5)
snow_to_rain_centre = 1.0
snow_depletion_steepness_log = jnp.log(3.0)
snow_depletion_centre_log = jnp.log(0.03)
t_softplus_sharpness_log = jnp.log(10)
swe_softplus_sharpness_log = jnp.log(20)

# Calibration
initial_learning_rate = 1e-3
