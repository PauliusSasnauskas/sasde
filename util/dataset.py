from jax import random, config
import jax.numpy as np
config.update('jax_platform_name', 'cpu') # for now

def batch_dataset(dataset, size=16):
    batch_count = dataset.shape[0] // size
    return np.split(dataset, batch_count)

def take_random(subkey, dataset):
    return random.choice(subkey, dataset)

def gen_dataset(subkey, bounds, size=256):
    return random.uniform(subkey, minval=bounds[0], maxval=bounds[1], shape=(size,))

def gen_analytical_dataset(subkey, x_bounds, k_bounds, y_func, size=256):
    xs = gen_dataset(subkey, x_bounds, size)
    ks = gen_dataset(subkey, k_bounds, size)
    ys = y_func(xs, ks)

    return np.concatenate((np.expand_dims(xs, 1), np.expand_dims(ys, 1)), axis=1)

def shuffle(subkey, dataset):
    indices = np.arange(dataset.shape[0])
    indices = random.permutation(subkey, indices)
    return dataset[indices]