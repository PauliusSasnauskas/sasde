import sympy as sp
from jax import config as jax_config
from jax import random
from derivatives import make_derivatives
from util.asserts import validateConfig
from util.interfaces import Config
from util.dotdict import DotDict
from util.print import d, info
from network import Network
from train import train

jax_config.update('jax_platform_name', 'cpu')

def run(config: Config):
    validateConfig(config)

    symbols = DotDict()
    for name in config.vars.keys():
        symbols[name] = sp.symbols(name)
    symbols_d, exprs_d = make_derivatives(config.eq.name, config.vars.keys())
    symbols.update(symbols_d)

    # set up network by configuration
    network = Network(
        symbols,
        [symbols[varname] for varname in config.vars.keys()],
        exprs_d,
        config.preoperations,
        config.operations,
        config.hyperparameters.cellcount,
        config.eq,
        config.vars,
        config.conditions,
        config.verbosity
    )

    network.get_model()


    key = random.PRNGKey(7)
    key, subkey = random.split(key)
    W = random.uniform(subkey, shape=(len(network.alphas),), minval=0, maxval=0.001)

    if network.is_final:
        print('Network is final, no training')

    best = None

    while not network.is_final:
        train_results = train(
            network,
            config = config,
            key = key,
            W_init = W
        )

        best = train_results.best

        # loss_history = train_results.loss_history

        info('Pruning weights...')
        network.assign_weights(W)
        W, _, _ = network.prune_auto()

        next_var_index = list(network.variables.keys()).index(network.operating_var)
        if next_var_index >= len(network.variables):
            next_var_index = 0
        network.set_operating_var(list(network.variables.keys())[next_var_index])

    y_prediction_best = best.model_y.subs(zip(best.alphas, best.W))
    d(y_prediction_best)
