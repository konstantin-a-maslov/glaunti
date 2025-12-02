import os
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_compilation_cache_dir", "jax_xla_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

import glaunti.gru_model as model
import dataloader.dataloader as dataloader
import core.loss as loss
import core.training as training
import utils.serialise
import utils.logger
import constants

import numpy as np
import datetime
from tqdm import tqdm


def main():
    dataset_index = dataloader.retrieve_dataset_index()
    train_subset = dataloader.get_train_subset(dataset_index)
    val_subset = dataloader.get_val_subset(dataset_index)

    model_callable = jax.jit(
        jax.remat(
            lambda trainable_params, static_params, x, initial_h: model.run_model(
                trainable_params, 
                static_params, 
                x, 
                initial_h=initial_h, 
                return_series=False, 
            ),
        ),
    )

    trainable_params, static_params = model.get_initial_model_parameters()

    loss_grad = jax.value_and_grad(loss.loss, argnums=0, has_aux=True)
    optimiser = training.get_optimiser()
    opt_state = optimiser.init(trainable_params)
    logger, train_pbar_desc, val_pbar_desc = None, "", ""

    best_val_mse, best_epoch = np.inf, 0
    device = jax.devices()[0]

    with tqdm(total=constants.n_epochs + 1, desc="") as pbar:
        for epoch in range(constants.n_epochs + 1):
            if epoch > 0:
                train_mse = 0.0
                accum_grads = jax.tree.map(jnp.zeros_like, trainable_params)
                for glacier in dataloader.traverse_glaciers(train_subset):
                    (loss_value, aux), grads = loss_grad(
                        trainable_params, static_params, model_callable, glacier, ti=False, device_to_prefetch=device,
                    )            
                    train_mse += extract_mse(aux)
                    accum_grads = jax.tree.map(lambda accum_grads, grads: accum_grads + grads, accum_grads, grads)
                    # log
                    log_record = {"timestamp": str(datetime.datetime.now()), "epoch": epoch, "glacier": glacier["name"], "subset": "train", "loss": loss_value, **aux}
                    logger.log(log_record)
                trainable_params, opt_state = training.make_step(optimiser, accum_grads, trainable_params, opt_state)
                train_mse /= len(train_subset)
                train_pbar_desc = f"train_mse={train_mse:.3f}, "
            
            # checkpoint
            val_mse = 0.0
            for glacier in dataloader.traverse_glaciers(val_subset):
                loss_value, aux = loss.loss(
                    trainable_params, static_params, model_callable, glacier, ti=False, device_to_prefetch=device,
                ) 
                val_mse += extract_mse(aux)
                # log
                log_record = {"timestamp": str(datetime.datetime.now()), "epoch": epoch, "glacier": glacier["name"], "subset": "val", "loss": loss_value, **aux}
                if logger is None:
                    logger = utils.logger.CSVLogger("logs/b.csv", log_record.keys())
                logger.log(log_record)
            val_mse /= len(val_subset)
            if val_mse < best_val_mse:
                best_val_mse, best_epoch = val_mse, epoch
                utils.serialise.save_pytree((trainable_params, static_params), "params/b.eqx")
            val_pbar_desc = f"val_mse={val_mse:.3f} (best={best_val_mse:.3f} at #{best_epoch})"

            pbar.set_description(f"{train_pbar_desc}{val_pbar_desc} [m w.e.]")
            pbar.update(1)


def extract_mse(aux):
    mse = [
        aux[k] for k in ["point_annual_mse", "point_winter_mse", "point_summer_mse"] 
        if ((k in aux) and (aux[k] is not None))
    ]
    mse.extend([
        constants.lambda1 * aux[k] for k in ["total_annual_mse", "total_winter_mse", "total_summer_mse"] 
        if ((k in aux) and (aux[k] is not None))
    ])
    mse = np.sum(mse)
    return mse


if __name__ == "__main__":
    main()
