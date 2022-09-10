import sympy as sp
from util.interfaces import Config, ConfigHyperparameters, ConfigNames
from main import run

a = 1

config = Config(
  names = ConfigNames(
    eq = 'u',
    vars = ['x', 'y']
  ),
  bounds = [(-1, 1), (-1, 1)],
  eq = lambda s: s.dudx2 + s.dudy2 + s.dudy ** 2 - 2 * s.y + s.x ** 4,
  conds = [
    lambda s: s.u.subs(s.x, 0),
    lambda s: s.u.subs(s.x, 1) - s.y - a,
    lambda s: s.u.subs(s.y, 0) - a * s.x,
    lambda s: s.u.subs(s.y, 1) - s.x * (s.x + a)
  ],
  preoperations = [],
  operations = [
    lambda z: 0,
    lambda z: 1,
    lambda z: z,
    lambda z: -z,
    lambda z: 1 + z,
    lambda z: -z,
    lambda z: z**2,
    lambda z: z**3,
    lambda z: sp.exp(z) + 0,
  ],
  hyperparameters = ConfigHyperparameters(
    lr = 0.0001,
    cellcount = 4,
    conds = [1., 1., 1., 1.],
  ),
  epochs = 10,
  batchsize = 16,
  verbosity = 1,
)

run(config)
