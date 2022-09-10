import sympy as sp
from jax import config as jax_config
from jax import random
import jax.numpy as np
from derivatives import make_derivatives
from util.asserts import validateConfig
from util.interfaces import Config
from util.dotdict import DotDict
from util.plot import Plotting
from util.print import a, d, pad, info
from network import Network
from train import train

jax_config.update('jax_platform_name', 'cpu')

# Code

def run(config: Config):
    validateConfig(config)

    symbols = DotDict()
    for name in config.names.vars:
        symbols[name] = sp.symbols(name)
    symbols_d, exprs_d = make_derivatives(config.names)
    symbols.update(symbols_d)

    # set up network by parameters
    network = Network(
        symbols,
        [symbols[var] for var in config.names.vars],
        exprs_d,
        config.preoperations,
        config.operations,
        config.hyperparameters.cellcount,
        config.eq,
        config.verbosity
    )

    # train network
    # output progress

    # output final result
