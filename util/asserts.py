from sympy.core.expr import Expr
from util.dotdict import DotDict

def isNotEmpty(val, valname):
    if val is None:
        raise KeyError(f'{valname} is missing')

def isDict(val, valname):
    isNotEmpty(val, valname)
    if not isinstance(val, (dict, DotDict)):
        raise KeyError(f'{valname} is not a dict')

def isList(val, valname):
    isNotEmpty(val, valname)
    if not isinstance(val, list):
        raise KeyError(f'{valname} is not a list')

def isListOfSomething(isSomething, val, valname):
    isList(val, valname)
    for index, item in enumerate(val):
        isSomething(item, f'{valname}[{index}]')

def isFloat(val, valname):
    isNotEmpty(val, valname)
    if not isinstance(val, float):
        raise KeyError(f'{valname} is not a float')

def isInteger(val, valname):
    isNotEmpty(val, valname)
    if not isinstance(val, int):
        raise KeyError(f'{valname} is not an int')

def isString(val, valname):
    isNotEmpty(val, valname)
    if not isinstance(val, str):
        raise KeyError(f'{valname} is not a string')

def isStringList(val, valname):
    isListOfSomething(isString, val, valname)

def isFloatList(val, valname):
    isListOfSomething(isFloat, val, valname)

def isFloatPair(val, valname):
    isNotEmpty(val, valname)
    if not isinstance(val, tuple) or len(val) != 2:
        raise KeyError(f'{valname} is not a pair of floats (tuple of two floats)')

def isFloatPairList(val, valname):
    isListOfSomething(isFloatPair, val, valname)

def isSympyExpr(val, valname):
    isNotEmpty(val, valname)
    if not isinstance(val, Expr):
        raise KeyError(f'{valname} is not a SymPy Expr')

def isSympyExprList(val, valname):
    isListOfSomething(isSympyExpr, val, valname)

def isFunc(val, valname):
    isNotEmpty(val, valname)
    if not callable(val):
        raise KeyError(f'{valname} is not a function')

def isFuncList(val, valname):
    isListOfSomething(isFunc, val, valname)

def validateConfig(config):
    isDict(config, 'The config object')

    isDict(config.names, 'names')
    isString(config.names.eq, 'names.eq')
    isStringList(config.names.vars, 'names.vars')
    isFloatPairList(config.bounds, 'bounds')
    isFunc(config.eq, 'eq')
    isFuncList(config.conds, 'conds')
    isFuncList(config.basefuncs, 'basefuncs')
    isDict(config.hyperpars, 'hyperpars')
    isFloat(config.hyperpars.lr, 'hyperpars.lr')
    isInteger(config.hyperpars.cellcount, 'hyperpars.cellcount')
    isFloatList(config.hyperpars.conds, 'hyperpars.conds')
    isInteger(config.epochs, 'epochs')
    isInteger(config.batchsize, 'batchsize')
    isInteger(config.verbosity, 'verbosity')

    if len(config.names.vars) != len(config.bounds):
        raise KeyError('names.vars length should match bounds length (each named variable must have a range)')

    if len(config.conds) != len(config.hyperpars.conds):
        raise KeyError('conds length should match hyperpars.conds length (each condition should have its hyperparameter)')
