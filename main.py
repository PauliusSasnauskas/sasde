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
    symbols_d, exprs_d = make_derivatives(config.eq.name, config.vars.keys(), config.derivative_order)
    symbols.update(symbols_d)

    # set up network by configuration
    network = Network(
        symbols,
        [symbols[varname] for varname in config.vars.keys()],
        exprs_d,
        config.preoperations,
        config.operations,
        config.hyperparameters.nodecount,
        config.eq,
        config.vars,
        config.conditions,
        config.hyperparameters.penalty,
        config.verbosity
    )

    network.get_model()

    key = random.PRNGKey(config.seed)
    key, subkey = random.split(key)

    if network.is_final:
        print('Network is final, no training')

    best = None

    W = random.uniform(subkey, shape=(len(network.alphas),), minval=0, maxval=0.001)

    loss_histories = []

    pruneiter = 1
    prunemax = (len(config.preoperations)-1)*(config.hyperparameters.nodecount-1) + (len(config.operations)-1)*sp.binomial(config.hyperparameters.nodecount - 1, 2)

    try:
        while not network.is_final:
            key, subkey = random.split(key)
            train_results = train(
                network,
                config = config,
                key = subkey,
                W_init = W
            )

            W = train_results.W
            best = train_results.best
            loss_history = train_results.loss_history

            loss_histories += [loss_history]

            if pruneiter <= prunemax: info(f'Pruning weights ({pruneiter} / {prunemax})')
            pruneiter += 1

            network.assign_weights(W)
            W, _, _ = network.prune_auto()
    except KeyboardInterrupt:
        info("Stopping...")

    prediction_best = best.model.subs(zip(best.alphas, best.W))
    d(prediction_best)
    print(f"Best loss: {float(best.loss)}")

    return network, best, loss_histories
