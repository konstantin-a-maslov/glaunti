import jax.numpy as jnp
import pandas


# Data
data_folder = "data"
use_cache = True
test_fold = 0
val_fold = 1

prefetch_workers = 2


# Conventions/definitions
study_period_start_year = 1995
initialisation_period_start_year = 1990

point_smb_weight_decay_rate = 6.355375070295649 # days


# Conversions
temperature_lapse_rate = -0.0060 # deg C per m
gravitational_acceleration = 9.80665 # m per s2


# Initial TI parameter values
"""
Based on physics-plausible and literature priors:
    https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2015.00054/full
    https://www.sciencedirect.com/science/article/pii/S0022169403002579
    https://www.science.org/doi/10.1126/science.abo1324
    https://www.nature.com/articles/s41467-018-03629-7
    https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2012JD018178
"""
prec_scale_log = jnp.log(1.4)
ddf_snow_log = jnp.log(0.0049)
ddf_relative_ice_minus_one_log = jnp.log(0.5)
snow_to_rain_steepness_log = jnp.log(1.5)
snow_to_rain_centre = 1.0
snow_depletion_steepness_log = jnp.log(3.0)
snow_depletion_centre_log = jnp.log(0.03)
t_softplus_sharpness_log = jnp.log(10)
swe_softplus_sharpness_log = jnp.log(20)


# GRU parameters
gru_input_size = 2
gru_output_size = 1
gru_h_size = 16
gru_initial_h_scale = 0.01


# Corrector parameters
n_facies_classes = 8

corrector_field_size = 12 # outlines(1) + elev(1) + elev_stddev(1) + facies(8) + facies conf(1)
climate_monthly_size = 2 
corrector_output_size = 3
n_filters_2d_branch = 32
n_stages_2d_branch = 3
n_filters_1d_branch = 8
n_stages_1d_branch = 2
corrector_scaler = 0.1 # To stabilise first epochs


# Calibration
n_epochs = 100
learning_rate = 1e-3
grad_norm_clip = 5

lambda1 = 0.5
default_lambda2 = 0.1
default_lambda3 = 100
default_lambda4 = 100


# Misc
default_rng_key = 42
