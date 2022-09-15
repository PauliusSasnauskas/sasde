import sympy as sp
from util.interfaces import Config, EqInfo, Hyperparameters, VarInfo

config = Config(
  eq = EqInfo(
    name = 'y',
    function = lambda s: s.dydx - s.k * s.y,
  ),
  vars = {
    'k': VarInfo(bounds=(1.4, 1.6), integrable=False),
    'x': VarInfo(bounds=(0, 1), integrable=False),
  },
  conditions = [
    (2, lambda s: s.y.subs(s.x, 1) - sp.exp(1.5))
  ],
  preoperations = [
    lambda k, x: 0,
    lambda k, x: 1,
    lambda k, x: k,
    lambda k, x: x,
    lambda k, x: k * x
  ],
  operations = [
    lambda z: 0,
    lambda z: 1,
    lambda z: z,
    lambda z: -z,
    lambda z: z*z,
    lambda z: sp.exp(z) + 0,
  ],
  hyperparameters = Hyperparameters(
    lr = 0.003,
    cellcount = 4,
  ),
  epochs = 24,
  batchsize = 32,
  verbosity = 1,
)
