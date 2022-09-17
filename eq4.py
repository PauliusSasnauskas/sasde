import sympy as sp
from util.interfaces import Config, EqInfo, Hyperparameters, VarInfo

config = Config(
  eq = EqInfo(
    name = 'y',
    function = lambda s: s.d2ydx2 - s.k * s.y / (1 + s.y),
  ),
  vars = {
    'x': VarInfo(bounds=(0, 1), integrable=False),
    'k': VarInfo(bounds=(10, 20), integrable=False)
  },
  conditions = [
    (1., lambda s: s.y.subs(s.x, 1) - 1),
    (1., lambda s: s.dydx.subs(s.x, 0)),
  ],
  preoperations = [
    lambda x, k: 0,
    lambda x, k: 1,
    lambda x, k: x,
    lambda x, k: k,
    lambda x, k: x * k,
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
    lr = 0.0000001,
    cellcount = 4,
  ),
  epochs = 128,
  batchsize = 64,
  verbosity = 1,
)
