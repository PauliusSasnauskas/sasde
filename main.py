import sympy as sp
from jax import random
from jax import config as jax_config
import jax.numpy as np

jax_config.update('jax_platform_name', 'cpu')

# !rm /etc/localtime
# !ln -s /usr/share/zoneinfo/Europe/Vilnius /etc/localtime

from util.plot import Plotting
from util.print import a, d, pad, info
from util.dotdict import DotDict
from util.asserts import validateConfig
from network import Network
from train import train

def run(config):
    validateConfig(config)

    network = Network(
        None,
        None,
        config.names,
        config.operations,
        config.hyperpars.cellcount,
        config.verbosity
    )