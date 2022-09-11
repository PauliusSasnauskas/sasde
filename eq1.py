import sympy as sp
from util.interfaces import Config, EqInfo, Hyperparameters, VarInfo

config = Config(
  eq = EqInfo(
    name = 'y',
    function = lambda s: s.dydx - s.k * s.x,
  ),
  vars = {
    'k': VarInfo(bounds=(1.4, 1.6), integrable=True),
    'x': VarInfo(bounds=(0, 1), integrable=False),
  },
  conditions = [
    (0.001, lambda s: s.y.subs(s.x, 1) - 1)
  ],
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
  hyperparameters = Hyperparameters(
    lr = 0.0001,
    cellcount = 4,
  ),
  epochs = 16,
  batchsize = 16,
  verbosity = 1,
)
