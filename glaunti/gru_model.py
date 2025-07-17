import jax
import jax.numpy as jnp
import equinox as eqx
import constants


class GRUBaseline(eqx.Module):
    rnn: eqx.nn.GRUCell
    out: eqx.nn.Linear
    init_h: jnp.ndarray 

    def __init__(self, input_size, output_size, h_size, h_scale, key):
        key1, key2, key3 = jax.random.split(key, 3)
        self.rnn = eqx.nn.GRUCell(input_size=input_size, hidden_size=h_size, key=key1)
        self.out = eqx.nn.Linear(h_size, output_size, key=key2)
        self.init_h = jax.random.normal(key3, shape=(h_size,)) * h_scale

    def __call__(self, x, initial_h=None):
        precipitation = x["precipitation"]
        temperature = x["temperature"]
        time, height, width = temperature.shape

        inputs = jnp.stack([precipitation, temperature], axis=-1)
        inputs_flat = inputs.reshape(time, height * width, -1)

        if initial_h is None:
            h0 = jnp.broadcast_to(self.init_h, (height, width, self.init_h.shape[0]))
        else: 
            h0 = initial_h
        h0_flat = h0.reshape(height * width, -1)

        batched_rnn = jax.vmap(self.rnn, in_axes=0)
        batched_out = jax.vmap(self.out)
        
        def step(h, x_t):
            h_next = batched_rnn(x_t, h)
            y = batched_out(h_next).squeeze(-1)
            return h_next, y

        final_h_flat, smb_flat = jax.lax.scan(step, h0_flat, inputs_flat)
        smb = smb_flat.reshape(time, height, width)
        final_h = final_h_flat.reshape(height, width, -1)
        return smb, final_h

        
@jax.jit
def run_model(trainable_params, static_params, x, initial_h=None):
    """
    Run the model over time series of precipitation and temperature.

    Args:
        trainable_params (GRUBaseline): The full model
        static_params (PyTree): Empty dict, just for interface compatibility
        x (PyTree): Dictionary with keys 'precipitation' and 'temperature', each of shape (T, H, W)
        initial_h (jnp.ndarray, Optional): Initial GRU state (H, W)

    Returns:
        smb_series (jnp.ndarray): Surface mass balance predictions (T, H, W)
        h (jnp.ndarray): Updated hidden state (H, W)
    """
    smb_series, h = trainable_params(x, initial_h)
    return smb_series, h


def get_initial_model_parameters(key=None):
    """
    Initialise GRU parameters.
    
    Returns:
        trainable_params (GRUBaseline): The full model
        static_params (dict): Empty dict, just for interface compatibility
    """
    if key is None:
        key = jax.random.PRNGKey(constants.default_key)

    model = GRUBaseline(
        constants.gru_input_size, 
        constants.gru_output_size, 
        constants.gru_h_size, 
        constants.gru_initial_h_scale, 
        key
    )
    return model, {}  # eqx's Module is PyTree
