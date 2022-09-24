import sympy as sp
from util.interfaces import Config, EqInfo, Hyperparameters, VarInfo

config = Config(
  eq = EqInfo(
    name = 'U',
    function = lambda s: s.t * s.dUdx + s.b * s.U * (1 - s.U),
  ),
  vars = {
    't': VarInfo(bounds=(0.01, 1), integrable=False, symbol=r'\theta_0'),
    'x': VarInfo(bounds=(-20, 20), integrable=False, symbol=r'\xi'),
    'b': VarInfo(bounds=(0.01, 1), integrable=False)
  },
  conditions = [
    (0.01, lambda s: s.U.subs(s.x, -20) - 1),
    (0.01, lambda s: s.U.subs(s.x, 20))
  ],
  preoperations = [
    lambda t, x, b: 0,
    lambda t, x, b: 1,
    lambda t, x, b: t,
    lambda t, x, b: x,
    lambda t, x, b: b,
    lambda t, x, b: t * x,
    lambda t, x, b: t * b,
    lambda t, x, b: x * b,
  ],
  operations = [
    lambda z: 0,
    lambda z: 1,
    lambda z: z,
    lambda z: -z,
    lambda z: 1 / (1 + sp.exp(z)),
  ],
  hyperparameters = Hyperparameters(
    lr = 0.0003,
    penalty = 1,
    cellcount = 4,
  ),
  epochs = 128,
  batchsize = 64,
  verbosity = 1,
)
