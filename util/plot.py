import matplotlib.pyplot as plt
import jax.numpy as np

def show_kvaried_plot(func, x, k_bounds, label="expected"):
    k_range = np.arange(*k_bounds, 0.01)
    y_all = np.array([func(x, k_i) for k_i in k_range])

    y_low = y_all.min(axis=0)
    y_up = y_all.max(axis=0)

    plt.fill_between(x, y_low, y_up, alpha=0.2, color="gray", label=label)

def get_func(func, W, x_bounds):
    return func(W, np.arange(*x_bounds, 0.05))

def show_funcs(y_history_all, xy_data, func_y_analytical, x_bounds, k_bounds):
    x = np.arange(*x_bounds, 0.05)

    plt.xlabel("x")
    plt.ylabel("y(x)")

    show_kvaried_plot(func_y_analytical, x, k_bounds)
    colors = plt.cm.viridis(np.linspace(0, 1, len(y_history_all)))
    for i, y_actual in enumerate(y_history_all):
        plt.plot(x, y_actual, color=colors[i], label=f"epoch {i}")
    
    plt.scatter(*list(zip(*xy_data)), s=0.2)

    if len(y_history_all) < 6:
        plt.legend(loc=2)
    plt.title(f"$k \\in [{k_bounds[0]}, {k_bounds[1]}]$")
    plt.show()

def show_func(func_y, func_y_analytical, W, x_bounds, k_bounds):
    x = np.arange(*x_bounds, 0.05)

    show_kvaried_plot(func_y_analytical, x, k_bounds, "$k \\in [" + f"{k_bounds[0]}, {k_bounds[1]}]$")
    plt.plot(x, func_y_analytical(x, 1), "--", color="tab:blue", label=f"$k = {1}$")
    plt.plot(x, get_func(func_y, W, x_bounds), color="tab:orange", label="model")
    plt.legend(
        *([ x[i] for i in [2, 0, 1] ] for x in plt.gca().get_legend_handles_labels()),
        handletextpad=0.75, loc='best')
    plt.gcf().set_dpi(150)
    plt.title(f"$k \\in [{k_bounds[0]}, {k_bounds[1]}]$")
    plt.show()
