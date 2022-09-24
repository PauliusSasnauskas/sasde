import sympy as sp
from jax import random
from derivatives import make_derivatives
from util.asserts import validateConfig
from util.interfaces import Config
from util.dotdict import DotDict
from util.print import d, info
from network import Network
from train import train

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
        config.hyperparameters.penalty,
        config.verbosity
    )

    network.get_model()

    key = random.PRNGKey(2)
    key, subkey = random.split(key)

    if network.is_final:
        print('Network is final, no training')

    best = None

    W = random.uniform(subkey, shape=(len(network.alphas),), minval=0, maxval=0.001)

    try:
        while not network.is_final:
            train_results = train(
                network,
                config = config,
                key = key,
                W_init = W
            )

            W = train_results.W
            best = train_results.best

            info('Pruning weights...')
            network.assign_weights(W)
            W, _, _ = network.prune_auto()
    except KeyboardInterrupt:
        info("Stopping...")

    y_prediction_best = best.model_y.subs(zip(best.alphas, best.W))
    d(y_prediction_best)

    return network, best
