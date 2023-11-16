from enum import Enum
from sys import setrecursionlimit
from typing import Callable, Sequence
import sympy as sp
from jax import jit, vmap, value_and_grad
from jax import config as jax_config
import jax.numpy as np
from jaxtyping import Array
from util.dotdict import DotDict
from util.interfaces import EqInfo, VarInfo, Numeric, SymbolicNumeric
from util.print import pad, info
from signal import SIGALRM, signal, alarm

setrecursionlimit(10000) # for SymPy

class PruneStrategy(Enum):
    ALL = 0
    SINGLE = 1

class Link:
    is_pruned: bool = False
    inputs: Sequence[sp.Expr]
    operations: Sequence[Callable[..., SymbolicNumeric]]
    prune_strategy: PruneStrategy
    alphas: Sequence[sp.Expr]
    weights: Array
    forward: Callable[..., SymbolicNumeric]
    penalty: sp.Expr
    all_symbols: dict[str, sp.Expr]

    def __init__(self,
        operations: Callable[..., SymbolicNumeric],
        fr: int,
        to: int,
        param_count: int = 1,
        all_symbols: dict[str, sp.Expr] = {},
        prune_strategy: PruneStrategy = PruneStrategy.SINGLE
    ):
        self.inputs = sp.symbols([f'input{i}_{fr}__{to}' for i in range(param_count)])
        self.operations = operations
        self.all_symbols = all_symbols
        self.prune_strategy = prune_strategy

        # self.alphas = sp.symbols(' '.join(['\\alpha_{' + f'o_{i}' + '}^{' + f'({fr}\,{to})' + '}' for i in range(len(self.operations))]))
        self.alphas = sp.symbols(' '.join([(f'a_o{i}__{pad(fr)}__{pad(to)}') for i in range(len(operations))]))
        if len(operations) <= 1:
            self.alphas = (self.alphas,)
            self.is_pruned = True

        self.weights = np.zeros((len(self.operations),))

        self.make_forward()

    def make_forward(self, override_penalty_target: Numeric = 1):
        all_sum = sum([
            alpha * operation(*self.inputs) if len(self.inputs) != 1 else alpha * operation(*self.inputs, self.all_symbols)
            for alpha, operation in zip(self.alphas, self.operations)
        ])
        self.forward = lambda *inputs: all_sum.subs(zip(self.inputs, inputs))
        if isinstance(all_sum, float) or isinstance(all_sum, int):
            self.forward = lambda *_: all_sum
        if self.is_pruned:
            self.penalty = 0
        else:
            self.penalty = (sum(self.alphas) - override_penalty_target)**2

    def assign_weights(self, weights: Array):
        self.weights = weights

    def prune(self):
        if self.is_pruned:
            print("Link already pruned!")
            return

        if self.prune_strategy == PruneStrategy.ALL:
            keep_index = np.argmax(self.weights)

            weight_sum = np.sum(self.weights)

            self.alphas = [self.alphas[keep_index]]
            self.operations = [self.operations[keep_index]]
            self.weights = np.array([self.weights[keep_index]])

            info(f'Shed {weight_sum - self.weights[0]} weight')

            self.is_pruned = True
            self.make_forward()
        elif self.prune_strategy == PruneStrategy.SINGLE:
            remove_index = int(np.argmin(self.weights))
            info(f'Shed {self.weights[remove_index]} weight')

            self.alphas = self.alphas[:remove_index] + self.alphas[remove_index + 1:]
            self.operations = self.operations[:remove_index] + self.operations[remove_index + 1:]
            self.weights = np.delete(self.weights, remove_index)

            if len(self.alphas) <= 1:
                self.is_pruned = True

            self.make_forward(float(np.sum(self.weights)))

class TimeoutException(Exception):
    pass

def timeoutHandler(_, __):
    raise TimeoutException("Out of time")

class Network:
    symbols: dict[str, sp.Expr]
    symbols_input: dict[str, sp.Expr]
    derivative_replacements: dict[str, Callable[[sp.Expr], sp.Expr]]
    preoperations: Sequence[Callable[..., SymbolicNumeric]]
    operations: Sequence[Callable[[SymbolicNumeric], SymbolicNumeric]]
    node_count: int
    eq: EqInfo
    variables: dict[str, VarInfo]
    operating_var: str
    conditions: Sequence[tuple[Numeric, Callable[[dict[str, sp.Expr]], sp.Expr]]]
    verbose: int

    alphas: Sequence[sp.Expr] = []
    weights: Array | list[Numeric] = []
    penalties: SymbolicNumeric = 0
    penalty_hyperparameter: Numeric

    links: dict[str, dict[str, Link]] = {}

    debug: dict = DotDict()
    is_final: bool = False

    lambdify_modules = {
        "Min": np.minimum,
        "Max": np.maximum,
        "fmin": np.minimum,
        "fmax": np.maximum,
        "min": np.minimum,
        "max": np.maximum,
        "PINF": np.inf
    }

    def __init__(self,
        symbols: dict[str, sp.Expr],
        symbols_input: Sequence[sp.Expr],
        derivative_replacements: dict[str, Callable[[sp.Expr], sp.Expr]],
        preoperations: Sequence[Callable[..., SymbolicNumeric]],
        operations: Sequence[Callable[[SymbolicNumeric], SymbolicNumeric]],
        node_count: int,
        eq: EqInfo,
        variables: dict[str, VarInfo],
        conditions: Sequence[tuple[Numeric, Callable[[dict[str, sp.Expr]], sp.Expr]]],
        penalty_hyperparameter: Numeric,
        verbose: int = 0
    ):
        jax_config.update('jax_platform_name', 'cpu')

        self.symbols = symbols
        self.symbols_input = symbols_input
        self.derivative_replacements = derivative_replacements
        self.operations = operations
        self.node_count = node_count
        self.eq = eq
        self.variables = variables
        self.operating_var = list(self.variables.keys())[0]
        self.conditions = conditions
        self.penalty_hyperparameter = penalty_hyperparameter
        self.verbose = verbose

        self.model_y = None
        # self.func_y = None

        self.fmax = sp.Function('fmax')
        self.b_weight = np.zeros(1)[0]

        for fr in range(node_count):
            self.links[fr] = {}
            for to in range(fr+1, node_count):
                if fr == 0:
                    self.links[fr][to] = Link(preoperations[::], fr, to, param_count=len(symbols_input))
                else:
                    self.links[fr][to] = Link(operations[::], fr, to, all_symbols=symbols)

    def __get_symbolic_model(self):
        self.weights = []
        self.alphas = []
        self.penalties = 0

        b = sp.symbols('b')

        partial_results = {i: [] for i in range(self.node_count)}
        for fr, start_links in self.links.items():
            for to, link in start_links.items():
                if fr == 0:
                    partial_results[to] += [link.forward(*self.symbols_input)]
                else:
                    partial_results[to] += [link.forward(sum(partial_results[fr]))]
                self.alphas += link.alphas
                self.weights += list(link.weights)
                self.penalties += link.penalty

        self.penalties += sum(self.fmax(-alpha, 0.0) for alpha in self.alphas) # pylint: disable=not-callable
        self.alphas += [b]
        self.weights += [self.b_weight]

        last_index = self.node_count - 1
        symbolic_model = sum(partial_results[last_index]) + b

        self.symbols[self.eq.name] = symbolic_model

        return symbolic_model

    def get_loss_function(self, loss_integrated: sp.Expr):
        loss_lambdified = sp.lambdify(
            [self.alphas, *self.symbols_input],
            loss_integrated,
            modules=[self.lambdify_modules, 'jax'],
            cse=True
        )

        self.loss = lambda alphas, *inputs: loss_lambdified(alphas, *inputs)[0]

        # JAXify function
        loss_and_grad = jit(vmap(value_and_grad(loss_lambdified, reduce_axes=("batch",)), (None, *(0 for _ in self.symbols_input))))
        self.loss_and_grad = loss_and_grad

        return loss_and_grad

    def get_loss_model(self, model_y: sp.Expr):
        loss_base = self.eq.function(self.symbols)
        loss_model: sp.Expr = sp.Pow(loss_base, 2, evaluate=False)
        # loss_model = sp.Abs(loss_base)

        loss_model = self.run_dif_replacements(loss_model, model_y)

        operating_var = self.variables[self.operating_var]

        if operating_var.integrable:
            signal(SIGALRM, timeoutHandler)
            alarm(10)
            try:
                info(f'Trying to integrate w.r.t. {self.operating_var}')
                result = sp.integrate(loss_model, (self.operating_var, *operating_var.bounds))
                if isinstance(result, sp.Integral):
                    info(f'SymPy failed to integrate model w.r.t. {self.operating_var}')
                else:
                    loss_model = result
                    self.debug.integrated = True
                    if self.verbose >= 1:
                        info('Integrated')
            except TimeoutException:
                info(f'Out of time when integrating w.r.t. {self.operating_var}')
            finally:
                alarm(0)
        self.next_operating_var()
        return loss_model


    def get_model(self):
        symbolic_model = self.__get_symbolic_model()
        # self.func_y = sp.lambdify([self.alphas, *self.symbols_input], symbolic_model)
        # self.func_y = vmap(sp.lambdify([self.alphas, self.x], symbolic_model), (None, 0))
        self.model_y = symbolic_model
        if self.verbose >= 1:
            info('Constructed symbolic model')

        loss_and_grad = None

        loss_model = self.get_loss_model(symbolic_model)
        if self.verbose >= 1:
            info('Constructed loss equation')

        for cond_hyperparameter, cond_function in self.conditions:
            cond_squared_loss = sp.Pow(cond_function(self.symbols), 2)
            # cond_squared_loss = sp.Abs(cond_function(self.symbols))
            loss_model += cond_hyperparameter * cond_squared_loss
        if len(self.conditions) > 0 and self.verbose >= 2:
            info('Added boundary conditions')

        # TODO: 2nd call to run_dif_replacements, may be optimized?
        loss_model = self.run_dif_replacements(loss_model, symbolic_model)

        loss_model += self.penalty_hyperparameter * self.penalties # regularization
        self.debug['loss_constructed'] = loss_model

        loss_and_grad = self.get_loss_function(loss_model)
        if self.verbose >= 1:
            info('Constructed JAXified model')

        return np.array(self.weights), symbolic_model, loss_and_grad

    def run_dif_replacements(self, model, model_y):
        # 'run' all diff. replacements
        for dif_name, dif_function in self.derivative_replacements.items():
            dif_expr = dif_function(model_y)
            model = model.subs(dif_name, dif_expr)
        if self.verbose >= 2:
            info('Substituted y\'s with replacements')
        return model

    def assign_weights(self, weights: Array):
        self.weights = weights
        self.b_weight = weights[len(weights)-1]
        w_fr = 0
        for _, start_links in self.links.items():
            for _, link in start_links.items():
                w_len = len(link.alphas)
                link.assign_weights(weights[w_fr : w_fr + w_len])
                w_fr += w_len

    def prune_auto(self):
        for _, start_links in self.links.items():
            for _, link in start_links.items():
                if not link.is_pruned:
                    link.prune()
                    return self.get_model()
        self.is_final = True
        print("Nothing more to prune!")
        return self.get_model()

    def next_operating_var(self):
        next_var_index = 1 + list(self.variables.keys()).index(self.operating_var)
        if next_var_index >= len(self.variables):
            next_var_index = 0
        self.operating_var = list(self.variables.keys())[next_var_index]
