import jax
import optax
from functools import partial

import constants


def get_optimiser(lr=constants.learning_rate):
    opt_chain = [
        optax.clip_by_global_norm(constants.grad_norm_clip), 
        optax.scale_by_adam(),
        optax.scale(-lr),
    ]
    optimiser = optax.chain(*opt_chain)
    return optimiser


@partial(jax.jit, static_argnums=0, donate_argnums=3)
def make_step(optimiser, grads, params, opt_state):
    updates, opt_state = optimiser.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state
