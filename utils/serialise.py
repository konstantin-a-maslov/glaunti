import jax
import equinox as eqx
import pickle


def save_pytree(pytree, path, use_pickle=False):
    """
    Serialise PyTree into file.

    Args:
        pytree (PyTree): PyTree to serialise, e.g., model parameters
        path (string): Output file
        use_pickle (bool): Use pickle backend for serialisation [!NOT RECOMMENDED!]
    """
    if use_pickle:
        save_pytree_with_pickle(pytree, path)
        return

    eqx.tree_serialise_leaves(path, pytree)
    

def load_pytree(path, template=None, use_pickle=False):
    """
    Deserialise file to PyTree.

    Args:
        path (string): Input file
        template (PyTree): Template to reconstruct serialised PyTree
        use_pickle (bool): Use pickle backend for serialisation [!NOT RECOMMENDED!]

    Returns:
        PyTree
    """
    if template is None and not use_pickle:
        raise ValueError("Pickle backend should be set explicitly with use_pickle=True if template is not provided!")

    if template is not None and use_pickle:
        raise ValueError("Only Equinox backend is available for serialisation when template is provided!")

    if use_pickle:
        return load_pytree_with_pickle(path)

    return eqx.tree_deserialise_leaves(path, template)


########################################## 
def save_pytree_with_pickle(pytree, path):
    """
    [!NOT RECOMMENDED!]
    Serialise PyTree into file with pickle.

    Args:
        pytree (PyTree): PyTree to serialise, e.g., model parameters
        path (string): Output file
    """
    with open(path, "wb") as dst:
        pickle.dump(
            jax.tree.map(lambda x: x, pytree),
            dst,
        )


def load_pytree_with_pickle(path):
    """
    [!NOT RECOMMENDED!]
    Deserialise file to PyTree with pickle.

    Args:
        path (string): Input file

    Returns:
        PyTree
    """
    with open(path, "rb") as src:
        return pickle.load(src)
