from dataclasses import dataclass
from typing import Callable, Sequence, Tuple
import sympy as sp

Numeric = float | int # TODO: add jax numeric type
SymbolicNumeric = Numeric | sp.Expr

@dataclass
class ConfigEqInfo:
    name: str
    function: Callable[[dict[str, sp.Expr]], sp.Expr]

@dataclass
class ConfigVarInfo:
    bounds: Tuple[Numeric, Numeric]
    integrable: bool

@dataclass
class ConfigHyperparameters:
    lr: Numeric
    cellcount: int

@dataclass
class Config:
    eq: ConfigEqInfo
    vars: dict[str, ConfigVarInfo]
    conditions: Sequence[tuple[Numeric, Callable[[dict[str, sp.Expr]], sp.Expr]]]
    preoperations: Sequence[Callable[..., SymbolicNumeric]]
    operations: Sequence[Callable[[SymbolicNumeric], SymbolicNumeric]]
    hyperparameters: ConfigHyperparameters
    epochs: int
    batchsize: int
    verbosity: int
