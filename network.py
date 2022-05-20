import sympy as sp
from jax import jit, vmap, value_and_grad
import jax.numpy as np
from sys import setrecursionlimit
from util.print import pad, info

setrecursionlimit(10000)

class Link:
    def __init__(self, x, operations, fr, to):
        self.x = x
        self.operations = operations

        # self.alphas = sp.symbols(' '.join(['\\alpha_{' + f'o_{i}' + '}^{' + f'({fr}\,{to})' + '}' for i in range(len(self.operations))]))
        self.alphas = sp.symbols(' '.join([(f'a_o{i}__{pad(fr)}__{pad(to)}') for i in range(len(operations))]))

        self.weights = np.zeros((len(self.operations),))

        self.make_forward()
        self.make_penalty()

    def make_forward(self):
        all_sum = sum([
            alpha * operation(self.x)
            for alpha, operation in zip(self.alphas, self.operations)
        ])
        self.forward = lambda input: all_sum.subs(self.x, input)

    def make_penalty(self):
        self.penalty = (sum(self.alphas) - 1)**2

    def assign_weights(self, weights):
        self.weights = weights

    def prune(self):
        if self.is_pruned():
            print("Link already pruned!")
            return

        keep_index = np.argmax(self.weights)

        weight_sum = np.sum(self.weights)

        self.alphas = [self.alphas[keep_index]]
        self.operations = [self.operations[keep_index]]
        self.weights = np.array([self.weights[keep_index]])

        info(f'Shed {weight_sum - self.weights[0]} weight')

        self.make_forward()
        self.penalty = 0

    def is_pruned(self):
        return self.penalty == 0


class Network:
    def __init__(self, loss_model_func, loss_integration_func, operations, node_count = 4, x_bounds = (0, 1)):
        self.loss_model_func = loss_model_func
        self.loss_integration_func = loss_integration_func
        self.operations = operations
        self.node_count = node_count
        self.is_final = False

        self.debug = {}
        self.model_y = None
        self.func_y = None
        self.x_bounds = x_bounds

        self.x = sp.symbols('x')
        self.fmax = sp.Function('fmax')
        self.b_weight = np.zeros(1)[0]
        self.lambdify_modules = {
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

        self.links = {}

        for i in range(node_count):
            self.links[i] = {}
            for j in range(i+1, node_count):
                self.links[i][j] = Link(self.x, operations[::], i, j)

    def __get_symbolic_model(self):
        input = self.x
        self.alphas = []
        self.weights = []
        self.penalties = 0

        b = sp.symbols('b')

        partial_results = {i: [] for i in range(self.node_count)}
        for fr, start_links in self.links.items():
            for to, link in start_links.items():
                if fr == 0:
                    partial_results[to] += [link.forward(input)]
                else:
                    partial_results[to] += [link.forward(sum(partial_results[fr]))]
                self.alphas += link.alphas
                self.weights += link.weights
                self.penalties += link.penalty
        last_index = self.node_count - 1

        self.penalties += sum([self.fmax(-alpha, 0.0) for alpha in self.alphas])
        self.alphas += [b]
        self.weights += [self.b_weight]

        return sum(partial_results[last_index]) + b

    def lambdify(self, loss_integrated):
        loss_lambdified = sp.lambdify(
            [self.alphas, self.x],
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
            [self.x],
            loss_integrated,
            modules=self.lambdify_modules,
            cse=True
        )

        self.debug['loss_lambdified'] = loss_lambdified

        # JAXify function
        loss_and_grad = jit(vmap(value_and_grad(loss_lambdified, reduce_axes=("batch",)), (0,)))
        self.loss_and_grad = loss_and_grad

        return loss_and_grad

    def get_integrated_model_nolambdify(self, model_y):
        model_y_replacement = sp.symbols('y')
        model_d2y_replacement = sp.symbols('ddy')
        # loss_model = sp.Pow(sp.diff(model_y, self.x, 2) - c1 * (model_y) / (1 + model_y), 2, evaluate=False)
        # loss_integrated = sp.integrate(loss_model, c1s)
        loss_model = self.loss_model_func(model_y_replacement, self.x, model_d2y_replacement)



        loss_integrated = sp.integrate(*self.loss_integration_func(loss_model))
        info('Integrated')

        loss_integrated = loss_integrated.subs(model_y_replacement, model_y)
        loss_integrated = loss_integrated.subs(model_d2y_replacement, sp.diff(model_y, self.x, 2))
        info('Substituted y\'s with replacements')

        return loss_integrated

    def __get_integrated_model(self, model_y):
        loss_integrated = self.get_integrated_model_nolambdify(model_y)

        # TODO: Keep here or move?
        model_y_diff = sp.diff(model_y, self.x)
        model_y_diff_at0 = model_y_diff.subs(self.x, 0)
        hyperpar = 100.0
        loss_boundary_cond = hyperpar * model_y_diff_at0**2

        loss_integrated += loss_boundary_cond
        info('Added boundary condition')

        loss_integrated += self.penalties # regularization
        self.debug['loss_integrated'] = loss_integrated

        loss_and_grad = self.lambdify(loss_integrated)
        info('Lambdified')
        return loss_and_grad

    def __get_secondary_model(self, model_y):
        y_actual = sp.symbols("y_actual")
        loss_secondary_model = (y_actual - model_y)**2
        self.debug['loss_secondary_model'] = loss_secondary_model

        loss_secondary_lambdified = sp.lambdify(
            [self.alphas, self.x, y_actual],
            loss_secondary_model,
            modules=self.lambdify_modules,
            cse=True
        )
        self.debug['loss_secondary_lambdified'] = loss_secondary_lambdified

        self.loss_and_grad_secondary = jit(value_and_grad(loss_secondary_lambdified))


    def get_model(self, should_integrate=True):
        symbolic_model = self.__get_symbolic_model()
        self.func_y = sp.lambdify([self.alphas, self.x], symbolic_model)
        # self.func_y = vmap(sp.lambdify([self.alphas, self.x], symbolic_model), (None, 0))
        self.model_y = symbolic_model
        info('Constructed symbolic model')

        loss_and_grad = None

        if should_integrate:
            loss_and_grad = self.__get_integrated_model(symbolic_model)
            info('Constructed JAXified model')

        self.__get_secondary_model(symbolic_model)

        return np.array(self.weights), symbolic_model, loss_and_grad, self.is_final

    def assign_weights(self, weights):
        self.weights = weights
        self.b_weight = weights[len(weights)-1]
        w_fr = 0
        for fr, start_links in self.links.items():
            for to, link in start_links.items():
                w_len = len(link.alphas)
                link.assign_weights(weights[w_fr : w_fr + w_len])
                w_fr += w_len

    # def _prune_link(self, fr, to):
    #     self.links[fr][to].prune()

    #     return self.get_model()

    def prune_auto(self):
        for fr, start_links in self.links.items():
            for to, link in start_links.items():
                if not link.is_pruned():
                    link.prune()
                    return self.get_model()
        self.is_final = True
        print("Nothing more to prune!")
        return self.get_model()