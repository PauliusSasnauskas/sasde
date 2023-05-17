from sympy.core.expr import Expr

from .interfaces import Config

def isNotEmpty(val, valname):
    if val is None:
        raise KeyError(f'{valname} is missing')

def isList(val, valname):
    isNotEmpty(val, valname)
    if not isinstance(val, list):
        raise KeyError(f'{valname} is not a list')

def isListOfType(isType, val, valname):
    isList(val, valname)
    for index, item in enumerate(val):
        isType(item, f'{valname}[{index}]')

def isStrDictOfType(isType, val, valname):
    isNotEmpty(val, valname)
    if not isinstance(val, dict):
        raise KeyError(f'{valname} is not a dict')
    if not all([isinstance(k, str) for k in val.keys()]):
        raise KeyError(f'{valname} keys are not strings')
    for key in val:
        isType(val[key], f'{valname}[{key}]')

def isNumeric(val, valname):
    isNotEmpty(val, valname)
    if not isinstance(val, float) and not isinstance(val, int):
        raise KeyError(f'{valname} is not a number')

def isFloat(val, valname):
    isNotEmpty(val, valname)
    if not isinstance(val, float):
        raise KeyError(f'{valname} is not a float')

def isInteger(val, valname):
    isNotEmpty(val, valname)
    if not isinstance(val, int):
        raise KeyError(f'{valname} is not an int')

def isString(val, valname, optional=False):
    if val is None and optional:
        return
    isNotEmpty(val, valname)
    if not isinstance(val, str):
        raise KeyError(f'{valname} is not a string')

def isStringList(val, valname):
    isListOfType(isString, val, valname)

def isNumericPair(val, valname):
    isNotEmpty(val, valname)
    isNumeric(val[0], f'{valname}[0]')
    isNumeric(val[1], f'{valname}[1]')

def isSympyExpr(val, valname):
    isNotEmpty(val, valname)
    if not isinstance(val, Expr):
        raise KeyError(f'{valname} is not a SymPy Expr')

def isSympyExprList(val, valname):
    isListOfType(isSympyExpr, val, valname)

def isFunc(val, valname):
    isNotEmpty(val, valname)
    if not callable(val):
        raise KeyError(f'{valname} is not a function')

def isFuncList(val, valname):
    isListOfType(isFunc, val, valname)

def isNumFunc(val, valname):
    isNotEmpty(val, valname)
    isNumeric(val[0], f'{valname}[0]')
    isFunc(val[1], f'{valname}[1]')

def isBool(val, valname):
    isNotEmpty(val, valname)
    if not isinstance(val, bool):
        raise KeyError(f'{valname} is not a bool')

def isVarInfo(val, valname):
    isNotEmpty(val, valname)
    isNumericPair(val.bounds, f'{valname}.bounds')
    isBool(val.integrable, f'{valname}.bool')
    isString(val.symbol, f'{valname}.symbol', optional=True)

def validateConfig(config: Config):
    isString(config.eq.name, 'eq.function')
    isFunc(config.eq.function, 'eq.function')

    isStrDictOfType(isVarInfo, config.vars, 'vars')

    isListOfType(isNumFunc, config.conditions, 'conditions')
    isFuncList(config.preoperations, 'preoperations')
    isFuncList(config.operations, 'operations')

    isFloat(config.hyperparameters.lr, 'hyperparameters.lr')
    isNumeric(config.hyperparameters.penalty, 'hyperparameters.penalty')
    isInteger(config.hyperparameters.nodecount, 'hyperparameters.nodecount')

    isInteger(config.epochs, 'epochs')
    isInteger(config.samples, 'samples')
    isInteger(config.batchsize, 'batchsize')
    isInteger(config.verbosity, 'verbosity')
    isInteger(config.seed, 'seed')
