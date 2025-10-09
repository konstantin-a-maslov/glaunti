import jax
import jax.numpy as jnp
import equinox as eqx
import constants


class GRUBaseline(eqx.Module):
    rnn: eqx.nn.GRUCell
    out: eqx.nn.Linear
    init_h: jnp.ndarray 

    def __init__(self, input_size, output_size, h_size, h_scale, key):
        keys = jax.random.split(key, 3)
        keys = iter(keys)
        
        self.rnn = eqx.nn.GRUCell(input_size=input_size, hidden_size=h_size, key=next(keys))
        self.out = eqx.nn.Linear(h_size, output_size, key=next(keys))
        self.init_h = jax.random.normal(next(keys), shape=(h_size,)) * h_scale

    def __call__(self, x, initial_h=None, return_series=False):
        precipitation = x["precipitation"]
        temperature = x["temperature"]
        time, height, width = temperature.shape

        inputs = jnp.stack([precipitation, temperature], axis=-1)
        inputs_flat = inputs.reshape(time, height * width, -1)

        if initial_h is None:
            h0 = jnp.broadcast_to(self.init_h, (height, width, self.init_h.shape[0]))
            is_continuation = False
        else: 
            h0 = initial_h
            is_continuation = True
        h0_flat = h0.reshape(height * width, -1)

        w = jnp.ones((time, )).at[0].set(0.5).at[-1].set(0.5)
        m_step = jnp.ones((time, ))
        if is_continuation:
            m_step = m_step.at[0].set(0.0)
        
        batched_rnn = jax.vmap(self.rnn, in_axes=0)
        batched_out = jax.vmap(self.out)

        if return_series:
            def scan_step(h, inputs_t):
                x_t, w_t, m_t = inputs_t
                h_prop = batched_rnn(x_t, h)
                h_next = jnp.where(m_t > 0.5, h_prop, h)
                y = w_t * batched_out(h_next).squeeze(-1)
                return h_next, y
            # scan_step = jax.remat(scan_step)
            
            final_h_flat, smb_flat = jax.lax.scan(scan_step, h0_flat, (inputs_flat, w, m_step))
            smb = smb_flat.reshape(time, height, width)

        else:
            def scan_step(carry, inputs_t):
                h, y_acc = carry
                x_t, w_t, m_t = inputs_t
                h_prop = batched_rnn(x_t, h)
                h_next = jnp.where(m_t > 0.5, h_prop, h)
                y = batched_out(h_next).squeeze(-1)
                return (h_next, y_acc + w_t * y), None
            # scan_step = jax.remat(scan_step)

            y0 = batched_out(h0_flat).squeeze(-1)
            carry = (h0_flat, jnp.zeros_like(y0))
            (final_h_flat, smb_flat), _ = jax.lax.scan(scan_step, carry, (inputs_flat, w, m_step))
            smb = smb_flat.reshape(height, width)
        
        final_h = final_h_flat.reshape(height, width, -1)
        return smb, final_h


def run_model(trainable_params, static_params, x, initial_h=None, return_series=False):
    """
    Run the model over time series of precipitation and temperature.

    Args:
        trainable_params (PyTree): Full model
        static_params (PyTree): Empty dict, just for interface compatibility
        x (PyTree): Dictionary with keys 'precipitation' and 'temperature', each of shape (T, H, W)
        initial_h (jnp.ndarray, Optional): Initial GRU state (H, W, hidden)

    Returns:
        smb_series (jnp.ndarray): Surface mass balance predictions (H, W) or (T, H, W) if return_series
        h (jnp.ndarray): Updated hidden state (H, W, hidden)
    """
    smb, h = trainable_params["gru"](x, initial_h, return_series)
    return smb, h


def get_initial_model_parameters(key=None):
    """
    Initialise GRU parameters.
    
    Returns:
        trainable_params (GRUBaseline): Full model
        static_params (dict): Empty dict, just for interface compatibility
    """
    if key is None:
        key = jax.random.PRNGKey(constants.default_rng_key)

    model = GRUBaseline(
        constants.gru_input_size, 
        constants.gru_output_size, 
        constants.gru_h_size, 
        constants.gru_initial_h_scale, 
        key
    )
    return {"gru": model}, {}  # eqx's Module is PyTree
