import matplotlib.pyplot as plt
import jax.numpy as np
from util.print import pad

class Plotting:
    def __init__(self, actual_func, network, x_bounds, c2_bounds):
        self.actual_func = actual_func
        self.network = network
        self.x_bounds = x_bounds
        self.c2_bounds = c2_bounds
        self.funcs = []

    def init(self, W):
        self.funcs += [self.get_func(W)]

    def after_epoch(self, W, epoch, loss_epoch, show_plot=True):
        self.funcs += [self.get_func(W)]
        if show_plot:
            self.show_funcs(epoch, loss_epoch)

    def show_funcs(self, epoch, loss_epoch, xy_data=None, all=True):
        x = np.arange(*self.x_bounds, 0.05)

        plt.xlabel("x")
        plt.ylabel("y(x)")

        count = len(self.funcs)

        colors = plt.cm.viridis(np.linspace(0, 1, count))

        if all:
            for i, y in enumerate(self.funcs):
                plt.plot(x, y, color=colors[i], label=f"epoch {i}")
        else:
            last = self.funcs[-1]
            plt.plot(x, last, label=r"$y$")
        
        if xy_data is not None:
            plt.scatter(*list(zip(*xy_data))[:2], s=0.2)
            plt.scatter(*list(zip(*xy_data))[2:], s=0.2)

        if self.actual_func is not None:
            self.show_c2varied_plot(self.actual_func, x, self.c2_bounds)

        if not all or count < 10:
            plt.legend(loc=2)
        # plt.suptitle(text)
        plt.title(f"Epoch {epoch+1}, Loss: {loss_epoch:.5f}")
        # plt.ylim((0, 5))
        if not all:
            plt.savefig(f'imgs/img{pad(count, n=3)}.png', dpi=200)
        plt.show()

    def get_func(self, W):
        xrange = np.arange(*self.x_bounds, 0.05)
        val = self.network.func_y(W, xrange)

        if val.shape != xrange.shape:
            print('broken')
            val = np.array([])

            for item in xrange:
                itemval = self.network.func_y(W, item)
                val = np.concatenate((val, np.array([itemval])))
        return val

    def show_c2varied_plot(self, func, x, c2_bounds, label="expected"):
        c2_range = np.linspace(*c2_bounds)
        y_all = np.array([func(x, c_i) for c_i in c2_range])

        y_low = y_all.min(axis=0)
        y_up = y_all.max(axis=0)

        plt.fill_between(x, y_low, y_up, alpha=0.2, color="gray", label=label)


# def show_func(func_y, func_y_analytical, W, x_bounds, k_bounds):
#     x = np.arange(*x_bounds, 0.05)

#     show_c2varied_plot(func_y_analytical, x, k_bounds, "$k \\in [" + f"{k_bounds[0]}, {k_bounds[1]}]$")
#     plt.plot(x, func_y_analytical(x, 1), "--", color="tab:blue", label=f"$k = {1}$")
#     plt.plot(x, get_func(func_y, W, x_bounds), color="tab:orange", label="model")
#     plt.legend(
#         *([ x[i] for i in [2, 0, 1] ] for x in plt.gca().get_legend_handles_labels()),
#         handletextpad=0.75, loc='best')
#     plt.gcf().set_dpi(150)
#     plt.title(f"$k \\in [{k_bounds[0]}, {k_bounds[1]}]$")
#     plt.show()
