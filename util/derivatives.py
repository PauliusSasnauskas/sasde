import itertools
from typing import Callable, Sequence
import sympy as sp
from .interfaces import ConfigNames
from .dotdict import DotDict

def make_derivative_name(eq: str, vars: Sequence[str], orders: Sequence[int]):
    var_names = ''
    for var, order in zip(vars, orders):
        if order == 1:
            var_names += f'd{var}'
        elif order >= 2:
            var_names += f'd{var}{order}'
    return f'd{eq}{var_names}'

def make_derivatives(names: ConfigNames, derivative_order: int = 3) -> tuple[dict[str, sp.Expr], dict[str, Callable[[sp.Expr], sp.Expr]]]:
    new_symbols = DotDict()
    lambda_exprs = DotDict()

    orders_list = itertools.product(range(derivative_order+1), repeat=len(names.vars))
    next(orders_list) # skip where all zeros

    for orders in orders_list:
        name = make_derivative_name(names.eq, names.vars, orders)
        lambda_exprs[name] = lambda eq_expr, orders=orders: sp.diff(eq_expr, *zip(names.vars, orders))
        new_symbols[name] = sp.Symbol(name)

    return (new_symbols, lambda_exprs)
