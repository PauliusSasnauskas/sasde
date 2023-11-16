import jax
from jax import numpy as np, Array
import haiku as hk

dtype = np.float32

class Network(hk.Module):
  
  def __init__(self, depth, preoperations, operations, name=None, sigma_weights=1., sigma_biases=0.1):
    super().__init__(name=name)
    self.sigma_weights = sigma_weights
    self.sigma_biases = sigma_biases
    self.depth = depth
    self.preoperations = preoperations
    self.operations = operations


  def link_forward(self, link_input, orig_inputs, operations, fr, to):
    alphas = hk.get_parameter(f"a__{fr}_{to}", shape=[len(operations)], dtype=dtype, init=hk.initializers.RandomNormal(self.sigma_weights))
    beta = hk.get_parameter(f"b__{fr}_{to}", shape=[1], dtype=dtype, init=hk.initializers.RandomNormal(self.sigma_biases))

    result = sum([alpha * operation(link_input, *orig_inputs) for alpha, operation in zip(alphas, operations)])
    return result + beta, alphas

  def link_forward_first(self, orig_inputs, preoperations, to):
    alphas = hk.get_parameter(f"a__0_{to}", shape=[len(preoperations)], dtype=dtype, init=hk.initializers.RandomNormal(self.sigma_weights))
    beta = hk.get_parameter(f"b__0_{to}", shape=[1,], dtype=dtype, init=hk.initializers.RandomNormal(self.sigma_biases))

    result = sum([alpha * operation(*orig_inputs) for alpha, operation in zip(alphas, preoperations)])
    return result + beta, alphas
  
  def __call__(self, x) -> Array:
    result = 0
    results_partial = {}

    for fr in range(self.depth):
      for to in range(fr+1, self.depth):
        if fr == 0:
          res, alphas = self.link_forward_first(x, self.preoperations, to)
          results_partial[to] = res
        else:
          res, alphas = self.link_forward(results_partial[fr], x, self.operations, fr, to)
          results_partial[to] += res

        constraint_sumtoone = (np.sum(alphas) - 1)**2
        constraint_nonnegative = np.sum(np.maximum(-alphas, 0.)**2)
        result += constraint_sumtoone + constraint_nonnegative

    result += results_partial[self.depth-1]
    return result
        



def get_network(depth, preoperations, operations):
  def network_forward(x):
    network = Network(depth, preoperations, operations)
    return network(x)

  network = hk.transform(network_forward)
  network_batch_apply = jax.vmap(network.apply, (None, None, 0))

  return network, network_batch_apply #, symbolic_model

# network_init, network_predict = get_network(network_size, sigmaweights, sigmabiases)