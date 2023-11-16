from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Literal
from jaxtyping import Float
import sympy as sp

Numeric = float | int | Float
SymbolicNumeric = Numeric | sp.Expr

@dataclass
class EqInfo:
    name: str
    function: Callable[[dict[str, sp.Expr]], sp.Expr]

@dataclass
class VarInfo:
    bounds: Tuple[Numeric, Numeric]
    integrable: bool = False
    symbol: str | None = None

@dataclass
class Hyperparameters:
    lr: Numeric
    penalty: Numeric
    nodecount: int

@dataclass
class Config:
    eq: EqInfo
    vars: dict[str, VarInfo]
    conditions: Sequence[tuple[Numeric, Callable[[dict[str, sp.Expr]], sp.Expr]]]
    preoperations: Sequence[Callable[..., SymbolicNumeric]]
    operations: Sequence[Callable[[SymbolicNumeric], SymbolicNumeric]]
    hyperparameters: Hyperparameters
    epochs: int
    samples: int
    batchsize: int
    verbosity: int
    seed: int = 2
    derivative_order: int = 2

    def getSymbolsIntegrals(self):
        return [(symbol, *self.vars[symbol].bounds) for symbol in self.vars]
