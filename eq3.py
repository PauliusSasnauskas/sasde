import sympy as sp
from util.interfaces import Config, EqInfo, Hyperparameters, VarInfo

config = Config(
  eq = EqInfo(
    name = 'y',
    function = lambda s: s.d2ydx2 - 4 * s.dydx + 5 * s.y,
  ),
  vars = {
    'x': VarInfo(bounds=(0, 1), integrable=False)
  },
  conditions = [
    (64., lambda s: s.y.subs(s.x, 0) - 1),
    (64., lambda s: s.y.subs(s.x, 1) - 22),
  ],
  preoperations = [
    lambda z: 0,
    lambda z: 1,
    lambda z: z,
    lambda z: z + 1,
    lambda z: -z,
    lambda z: sp.sin(z) + 0,
    lambda z: sp.cos(z) + 0,
    lambda z: sp.exp(z) + 0,
  ],
  operations = [
    lambda z: 0,
    lambda z: 1,
    lambda z: z,
    lambda z: z + 1,
    lambda z: -z,
    lambda z: sp.sin(z) + 0,
    lambda z: sp.cos(z) + 0,
    lambda z: sp.exp(z) + 0,
  ],
  hyperparameters = Hyperparameters(
    lr = 0.000002,
    penalty = 1,
    cellcount = 5,
  ),
  epochs = 128,
  batchsize = 64,
  verbosity = 1,
)
