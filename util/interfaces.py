from dataclasses import dataclass
from typing import Callable, Sequence, Tuple
import sympy as sp

Numeric = float | int # TODO: add jax numeric type
SymbolicNumeric = Numeric | sp.Expr

@dataclass
class ConfigNames:
    eq: str
    vars: Sequence[str]

@dataclass
class ConfigHyperparameters:
    lr: Numeric
    cellcount: int
    conds: Sequence[Numeric]

@dataclass
class Config:
    names: ConfigNames
    bounds: Sequence[Tuple[Numeric, ...]]
    eq: Callable[[dict[str, sp.Expr]], sp.Expr]
    conds: Sequence[Numeric]
    preoperations: Sequence[Callable[..., SymbolicNumeric]]
    operations: Sequence[Callable[[SymbolicNumeric], SymbolicNumeric]]
    hyperparameters: ConfigHyperparameters
    epochs: int
    batchsize: int
    verbosity: int
