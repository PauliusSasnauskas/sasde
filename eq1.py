import sympy as sp
from util.interfaces import Config, EqInfo, Hyperparameters, VarInfo

config = Config(
  eq = EqInfo(
    name = 'P',
    function = lambda s: s.dPdt - s.r * s.P,
  ),
  vars = {
    'r': VarInfo(bounds=(1, 2), integrable=False),
    't': VarInfo(bounds=(0, 1), integrable=True),
  },
  conditions = [
    (2, lambda s: s.P.subs(s.t, 1) - sp.exp(s.r))
  ],
  preoperations = [
    lambda r, t: 0,
    lambda r, t: 1,
    lambda r, t: r,
    lambda r, t: t,
    lambda r, t: r * t
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
    penalty = 2,
    cellcount = 4,
  ),
  epochs = 24,
  batchsize = 32,
  verbosity = 1,
)
