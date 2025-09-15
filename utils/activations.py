import jax
import jax.numpy as jnp


def softplus_t(t, x): 
    """
    Steepness-parameterised softplus.

    Args:
        t (float): Softplus steepness
        x (jnp.ndarray): Input

    Returns:
        jnp.ndarray
    """
    return jax.nn.softplus(t * x) / t


def hill_curve(s, c, x):
    """
    Hill curve.

    Args:
        s (float): Curve steepness
        c (float): Curve centre (x at y = 0.5)
        x (jnp.ndarray): Input

    Returns:
        jnp.ndarray
    """
    x_term = jnp.power(x, s)
    c_term = jnp.power(c, s)
    return x_term / (x_term + c_term)
    