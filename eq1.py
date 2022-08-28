import sympy as sp
from main import run
from util.dotdict import DotDict

a = 1

config = DotDict()
config.names = DotDict()
config.names.eq = 'u'
config.names.vars = ['x', 'y']
config.bounds = [(-1, 1), (-1, 1)]
config.eq = lambda s: s.dudx2 + s.dudy2 + s.dudy ** 2 - 2 * s.y + s.x ** 4
config.conds = [
  lambda s: s.u.subs(s.x, 0),
  lambda s: s.u.subs(s.x, 1) - s.y - a,
  lambda s: s.u.subs(s.y, 0) - a * s.x,
  lambda s: s.u.subs(s.y, 1) - s.x * (s.x + a)
]
config.basefuncs = [
  lambda z: 0,
  lambda z: 1,
  lambda z: z,
  lambda z: -z,
  lambda z: 1 + z,
  lambda z: -z,
  lambda z: z**2,
  lambda z: z**3,
  lambda z: sp.exp(z) + 0,
]
config.hyperpars = DotDict()
config.hyperpars.lr = 0.0001
config.hyperpars.cellcount = 4
config.hyperpars.conds = [1., 1., 1., 1.]
config.epochs = 10
config.batchsize = 16
config.verbosity = 1

run(config)
