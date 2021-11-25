from jax import random
import jax.numpy as np

_random_key = random.PRNGKey(712)

def batch_dataset(dataset, size=16):
    batch_count = len(dataset) // size
    batches = []
    for i in range(batch_count):
        batches += [dataset[i*size : (i+1)*size]]
    return batches

def take_random(dataset, key=_random_key):
    return random.choice(key, dataset)

def gen_dataset(bounds, size=256, key=_random_key):
    return random.uniform(key, minval=bounds[0], maxval=bounds[1], shape=(size,))

def gen_analytical_dataset(x_bounds, k_bounds, func_y_analytical, size=256):
    xs = gen_dataset(x_bounds, size)
    ks = gen_dataset(k_bounds, size, key=_random_key+9)
    ys = func_y_analytical(xs, ks)

    return np.concatenate((np.expand_dims(xs, 1), np.expand_dims(ys, 1)), axis=1)

def shuffle(dataset):
    indices = np.arange(dataset.shape[0])
    indices = random.permutation(_random_key, indices)
    return dataset[indices]