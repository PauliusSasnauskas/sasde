import argparse
import sympy as sp
import jax.numpy as np

from util.interfaces import Config, EqInfo, Hyperparameters, VarInfo
from main import run

experiments = {}
experiments[1] = Config(eq = EqInfo(name = 'P', function = lambda s: s.dPdt - s.r * s.P), vars = { 'r': VarInfo(bounds=(1, 2)), 't': VarInfo(bounds=(0, 2)) }, conditions = [ (2, lambda s: s.P.subs(s.t, 1) - sp.exp(s.r)) ], preoperations = [ lambda r, t: 0, lambda r, t: 1, lambda r, t: r, lambda r, t: t, lambda r, t: r * t ], operations = [ lambda z, _: 0, lambda z, _: 1, lambda z, _: z, lambda z, _: -z, lambda z, _: z*z, lambda z, _: sp.exp(z) ], hyperparameters = Hyperparameters(lr = 0.01, penalty = 2, nodecount = 4), epochs = 32, samples = 512, batchsize = 64, verbosity = 1, seed = 1)
experiments[2] = Config(eq = EqInfo(name = 'u', function = lambda s: s.dudt + s.u * s.dudx - (0.01 / sp.pi) * s.d2udx2), vars = { 't': VarInfo(bounds=(0, 1)), 'x': VarInfo(bounds=(-1, 1))}, conditions = [ (2., lambda s: s.u.subs(s.t, 0) + sp.sin(sp.pi * s.x)), (2., lambda s: s.u.subs(s.x, -1)), (2., lambda s: s.u.subs(s.x, 1)) ], preoperations = [ lambda t, x: 0, lambda t, x: 1, lambda t, x: t, lambda t, x: x, lambda t, x: -x, lambda t, x: sp.exp(x), lambda t, x: x * t, lambda t, x: sp.exp(x) * t ], operations = [ lambda z, _: 0, lambda z, _: 1, lambda z, _: z, lambda z, _: z + 1, lambda z, _: -z, lambda z, _: sp.exp(z) ], hyperparameters = Hyperparameters(lr = 0.001, penalty = 1, nodecount = 5), epochs = 128, samples = 512, batchsize = 64, verbosity = 1)
experiments[3] = Config(eq = EqInfo(name = 'u', function = lambda s: s.x**2 * s.d2udx2 + s.y**2 * s.d2udy2), vars = { 'x': VarInfo(bounds=(0, 2)), 'y': VarInfo(bounds=(0, 2))}, conditions = [ (1., lambda s: s.u.subs(s.y, s.x**2) - sp.sin(s.x)), (1., lambda s: s.u.subs(s.x, 0) - s.y**2) ], preoperations = [ lambda x, y: 0, lambda x, y: 1, lambda x, y: x, lambda x, y: y, lambda x, y: x + y, lambda x, y: x * y ], operations = [ lambda z, _: 0, lambda z, _: 1, lambda z, _: z, lambda z, _: -z, lambda z, s: z * s.x, lambda z, s: z * s.y, lambda z, _: sp.exp(z), lambda z, _: sp.sin(z)], hyperparameters = Hyperparameters(lr = 0.002, penalty = 1, nodecount = 5), epochs = 256, samples = 512, batchsize = 128, verbosity = 1)

parser = argparse.ArgumentParser(description='Symbolic Neural Architecture Search for Differential Equations')
parser.add_argument('--experiment', default=0, type=int, choices=(0, 1, 2, 3),help='experiment number to run')

args = parser.parse_args()

if args.experiment == 0:
    print('Please specify a number of an experiment to run (--experiment n), or see notebooks.')
    quit(code=2)

network, best, loss_histories = run(experiments[args.experiment])

prediction_best = best.model.subs(zip(best.alphas, best.W))
print('\nBest symbolic model:')
print(sp.latex(prediction_best))
