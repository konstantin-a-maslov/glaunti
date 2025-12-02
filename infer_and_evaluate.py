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

import rioxarray
import xarray
import pandas

import jax.numpy as jnp
# import glaunti.ti_corr_model as model # Import dynamically instead!
import dataloader.dataloader as dataloader
# import core.loss as loss
# import core.training as training
import utils.serialise

import datetime
import numpy as np
import pyproj

import sklearn.metrics
import scipy

import json
import os
from tqdm import tqdm
import argparse


def main():
    model, ti, ti_corr, facies, params_path, glacier, output_path = parse_args()
    evaluation = evaluate(model, ti, ti_corr, facies, params_path, glacier)
    save_evaluation(output_path)
    print(model, ti, ti_corr, facies, params_path, glacier, output_path)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["a", "b", "c", "d"], help="Model name")
    parser.add_argument("glacier", help="Glacier name")
    parser.add_argument("output_path", help="Output path (.json)") # TODO: Add INFER! 
    parser.add_argument("--params_path", help="Params path (.eqx)")
    args = parser.parse_args()

    ti, ti_corr, facies = False, False, False
    match args.model:
        case "a":
            import glaunti.ti_model as model
            ti = True
        case "b":
            import glaunti.gru_model as model
        case "c":
            import glaunti.ti_corr_model as model
            ti, ti_corr = True, True
        case "d":
            import glaunti.ti_corr_model as model
            ti, ti_corr, facies = True, True, True
    default_params_path = f"params/{args.model}.eqx"

    params_path = args.params_path or default_params_path
    glacier = args.glacier
    output_path = args.output_path
    
    return model, ti, ti_corr, facies, params_path, glacier, output_path

    
def evaluate(model, ti, ti_corr, facies, params_path, glacier):
    trainable_params, static_params = load_params(model, ti_corr, params_path)
    print(trainable_params)
    print()
    print(static_params)


def load_params(model, ti_corr, params_path):
    if ti_corr:
        params = model.get_initial_model_parameters(ti_params_static=True)
    else:
        params = model.get_initial_model_parameters()
    params = utils.serialise.load_pytree(params_path, template=params)
    return params


def predict(ololo):
    pass

    
def save_evaluation(output_path):
    pass
    

if __name__ == "__main__":
    main()
