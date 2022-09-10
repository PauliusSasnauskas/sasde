from enum import Enum
from sys import setrecursionlimit
from typing import Callable, Sequence
import sympy as sp
from jax import jit, vmap, value_and_grad
import jax.numpy as np
from jaxtyping import Array
from util.dotdict import DotDict
from util.interfaces import ConfigEqInfo, ConfigVarInfo, Numeric, SymbolicNumeric
from util.print import pad, info

setrecursionlimit(10000)

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

    def __init__(self,
        operations: Callable[..., SymbolicNumeric],
        fr: int,
        to: int,
        parameter_count: int = 1,
        prune_strategy: PruneStrategy = PruneStrategy.SINGLE
    ):
        self.inputs = sp.symbols([f'input{i}_{fr}__{to}' for i in range(parameter_count)])
        self.operations = operations
        self.prune_strategy = prune_strategy

        # self.alphas = sp.symbols(' '.join(['\\alpha_{' + f'o_{i}' + '}^{' + f'({fr}\,{to})' + '}' for i in range(len(self.operations))]))
        self.alphas = sp.symbols(' '.join([(f'a_o{i}__{pad(fr)}__{pad(to)}') for i in range(len(operations))]))

        self.weights = np.zeros((len(self.operations),))

        self.make_forward()

    def make_forward(self, override_penalty_target: Numeric = 1):
        all_sum = sum([
            alpha * operation(*self.inputs)
            for alpha, operation in zip(self.alphas, self.operations)
        ])
        self.forward = lambda *inputs: all_sum.subs(zip(self.inputs, inputs))

        if self.is_pruned:
            self.penalty = 0
        else:
            self.penalty = (sum(self.alphas) - override_penalty_target)**2

    def assign_weights(self, weights):
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

            if len(self.alphas) == 1:
                self.is_pruned = True

            self.make_forward(float(np.sum(self.weights)))


class Network:
    symbols: dict[str, sp.Expr]
    symbols_input: dict[str, sp.Expr]
    derivative_replacements: dict[str, Callable[[sp.Expr], sp.Expr]]
    preoperations: Sequence[Callable[..., SymbolicNumeric]]
    operations: Sequence[Callable[[SymbolicNumeric], SymbolicNumeric]]
    node_count: int
    eq: ConfigEqInfo
    variables: dict[str, ConfigVarInfo]
    operating_var: str
    conditions: Sequence[tuple[Numeric, Callable[[dict[str, sp.Expr]], sp.Expr]]]
    verbose: int

    debug: dict = DotDict()
    is_final: bool = False

    lambdify_modules = {
        "exp": np.exp,
        "sin": np.sin,
        "cos": np.cos,
        "Min": np.minimum,
        "Max": np.maximum,
        "fmin": np.minimum,
        "fmax": np.maximum,
        "min": np.minimum,
        "max": np.maximum,
        "log": np.log,
        "Abs": np.abs,
        "abs": np.abs
    }

    def __init__(self,
        symbols: dict[str, sp.Expr],
        symbols_input: Sequence[sp.Expr],
        derivative_replacements: dict[str, Callable[[sp.Expr], sp.Expr]],
        preoperations: Sequence[Callable[..., SymbolicNumeric]],
        operations: Sequence[Callable[[SymbolicNumeric], SymbolicNumeric]],
        node_count: int,
        eq: ConfigEqInfo,
        variables: dict[str, ConfigVarInfo],
        conditions: Sequence[tuple[Numeric, Callable[[dict[str, sp.Expr]], sp.Expr]]],
        verbose: int = 0
    ):
        self.symbols = symbols
        self.symbols_input = symbols_input
        self.derivative_replacements = derivative_replacements
        self.operations = operations
        self.node_count = node_count
        self.eq = eq
        self.variables = variables
        self.operating_var = list(self.variables.keys())[0]
        self.conditions = conditions
        self.verbose = verbose

        self.model_y = None
        self.func_y = None

        self.fmax = sp.Function('fmax')
        self.b_weight = np.zeros(1)[0]

        self.links = {}

        for fr in range(node_count):
            self.links[fr] = {}
            for to in range(fr+1, node_count):
                if fr == 0:
                    self.links[fr][to] = Link(preoperations[::], fr, to, len(symbols_input))
                else:
                    self.links[fr][to] = Link(operations[::], fr, to)

    def __get_symbolic_model(self):
        self.alphas = []
        self.weights = []
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
                self.weights += link.weights
                self.penalties += link.penalty
        last_index = self.node_count - 1

        self.penalties += sum([self.fmax(-alpha, 0.0) for alpha in self.alphas]) # pylint: disable=not-callable
        self.alphas += [b]
        self.weights += [self.b_weight]

        symbolic_model = sum(partial_results[last_index]) + b

        self.symbols[self.eq.name] = symbolic_model

        return symbolic_model

    def lambdify(self, loss_integrated):
        loss_lambdified = sp.lambdify(
            [self.alphas, *self.symbols_input],
            loss_integrated,
            modules=self.lambdify_modules,
            cse=True
        )

        self.debug['loss_lambdified'] = loss_lambdified

        # JAXify function
        loss_and_grad = jit(vmap(value_and_grad(loss_lambdified, reduce_axes=("batch",)), (None, 0)))
        self.loss_and_grad = loss_and_grad

        return loss_and_grad

    def lambdify_no_alphas(self, loss_integrated):
        loss_lambdified = sp.lambdify(
            [*self.symbols_input],
            loss_integrated,
            modules=self.lambdify_modules,
            cse=True
        )

        self.debug['loss_lambdified'] = loss_lambdified

        # JAXify function
        loss_and_grad = jit(vmap(value_and_grad(loss_lambdified, reduce_axes=("batch",)), (0,)))
        self.loss_and_grad = loss_and_grad

        return loss_and_grad

    def get_constructed_model_nolambdify(self, model_y):
        loss_base = self.eq.function(self.symbols)
        loss_model = sp.Pow(loss_base, 2, evaluate=False)

        operating_var = self.variables[self.operating_var]
        if operating_var.integrable:
            loss_model = sp.integrate(loss_model, operating_var.bounds)
            if self.verbose >= 1:
                info('Integrated')

        # 'run' all diff. replacements
        for dif_name, dif_function in self.derivative_replacements.items():
            dif_expr = dif_function(model_y)
            loss_model = loss_model.subs(dif_name, dif_expr)
        # info('Substituted y\'s with replacements')

        return loss_model

    def __get_integrated_model(self, model_y):
        loss_constructed = self.get_constructed_model_nolambdify(model_y)

        for cond_hyperparameter, cond_function in self.conditions:
            cond_squared_loss = sp.Pow(cond_function(self.symbols), 2)
            loss_constructed += cond_hyperparameter * cond_squared_loss
            # if self.verbose >= 1:
            #     info('Added boundary condition')

        loss_constructed += self.penalties # regularization
        self.debug['loss_constructed'] = loss_constructed

        loss_and_grad = self.lambdify(loss_constructed)
        if self.verbose >= 1:
            info('Lambdified')
        return loss_and_grad

    def __get_secondary_model(self, model_y):
        y_actual = sp.symbols("y_actual")
        loss_secondary_model = (y_actual - model_y)**2
        self.debug['loss_secondary_model'] = loss_secondary_model

        loss_secondary_lambdified = sp.lambdify(
            [self.alphas, *self.symbols_input, y_actual],
            loss_secondary_model,
            modules=self.lambdify_modules,
            cse=True
        )
        self.debug['loss_secondary_lambdified'] = loss_secondary_lambdified

        self.loss_and_grad_secondary = jit(value_and_grad(loss_secondary_lambdified))


    def get_model(self, should_integrate=True):
        symbolic_model = self.__get_symbolic_model()
        self.func_y = sp.lambdify([self.alphas, *self.symbols_input], symbolic_model)
        # self.func_y = vmap(sp.lambdify([self.alphas, self.x], symbolic_model), (None, 0))
        self.model_y = symbolic_model
        if self.verbose >= 1:
            info('Constructed symbolic model')

        loss_and_grad = None

        if should_integrate:
            loss_and_grad = self.__get_integrated_model(symbolic_model)
            if self.verbose >= 1:
                info('Constructed JAXified model')

        self.__get_secondary_model(symbolic_model)

        return np.array(self.weights), symbolic_model, loss_and_grad, self.is_final

    def assign_weights(self, weights):
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
                if not link.is_pruned():
                    link.prune()
                    return self.get_model()
        self.is_final = True
        print("Nothing more to prune!")
        return self.get_model()
