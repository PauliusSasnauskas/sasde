import sympy as sp
from util.interfaces import Config, ConfigEqInfo, ConfigHyperparameters, ConfigVarInfo
from main import run

config = Config(
  eq = ConfigEqInfo(
    name = 'y',
    function = lambda s: s.dydx - s.k * s.x,
  ),
  vars = {
    'k': ConfigVarInfo((1.4, 1.6), True),
    'x': ConfigVarInfo((0, 1), False),
  },
  conditions = [],
  preoperations = [
    lambda k, x: 0,
    lambda k, x: 1,
    lambda k, x: k,
    lambda k, x: x,
    lambda k, x: k + x,
    lambda k, x: k * x,
    lambda k, x: sp.exp(k),
  ],
  operations = [
    lambda z: 0,
    lambda z: 1,
    lambda z: z,
    lambda z: -z,
    lambda z: z*z,
    lambda z: sp.exp(z) + 0,
  ],
  hyperparameters = ConfigHyperparameters(
    lr = 0.0001,
    cellcount = 4,
  ),
  epochs = 10,
  batchsize = 16,
  verbosity = 1,
)

run(config)
