import jax
import pickle


def save_pytree(pytree, path): # TODO: Consider replacing with equinox serialisation
    """
    Serialise PyTree into file.

    Args:
        pytree (PyTree): PyTree to serialise, e.g., model parameters
        path (string): Output file
    """
    with open(path, "wb") as dst:
        pickle.dump(
            jax.tree_util.tree_map(lambda x: x, pytree),
            dst,
        )


def load_pytree(path): # TODO: Consider replacing with equinox serialisation
    """
    Deserialise file to PyTree.

    Args:
        path (string): Input file

    Returns:
        PyTree
    """
    with open(path, "rb") as src:
        return pickle.load(src)
