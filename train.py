from dataclasses import dataclass
from typing import Sequence
from jax import random
import jax.numpy as np
from jax.scipy.optimize import minimize as min_bfgs
from jaxtyping import Array
import optax
import sympy as sp
from network import Network
from util.dataset import shuffle, batch_dataset, gen_dataset
from util.dotdict import DotDict
from util.interfaces import Config, Numeric
from util.print import a, info

@dataclass
class BestResult:
    loss: Numeric
    model: sp.Expr
    alphas: Sequence[sp.Expr]
    W: Array

@dataclass
class TrainOutput:
    W: Array
    loss_history: Sequence[Sequence[Numeric]]
    best: BestResult

def train(
    network: Network,
    config: Config,
    key = None,
    W_init = None
) -> TrainOutput:

    loss_history = []

    assert W_init is not None, 'W_init is None, please initialize'

    W = W_init

    best = BestResult(loss = np.inf, model = network.model_y, alphas = network.alphas, W = W)

    if config.verbosity >= 2:
        info(f"W₀ = {a(W)}")

    # plotting.init(W)

    # optimizer = optax.adam(config.hyperparameters.lr)
    linear_decay_schedule = optax.linear_schedule(init_value=config.hyperparameters.lr, end_value=config.hyperparameters.lr * 0.1, transition_steps=1000)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.),
        optax.scale_by_adam(),
        optax.scale_by_schedule(linear_decay_schedule),
        optax.scale(-1.0)
    )

    opt_state = optimizer.init(W)

    # train loop
    for epoch in range(config.epochs): # epochs
        loss_epoch = []
        batches = DotDict()

        input_datasets = DotDict()
        for varname, varinfo in config.vars.items():
            key, subkey = random.split(key)
            input_datasets[varname] = gen_dataset(subkey, varinfo.bounds, size=config.samples)
            batches[varname] = batch_dataset(input_datasets[varname], config.batchsize)

        for batch in zip(*batches.values()): # minibatches
            loss, grad = network.loss_and_grad(W, *batch)

            nan_locs = np.isnan(grad)
            grad = grad.at[nan_locs].set(0)

            grad_avg = np.average(grad, axis=0)
            loss_avg = np.average(loss)

            if np.any(nan_locs):
                info('Gradients are nan, mitigating...')
                W.at[np.isnan(grad_avg)].set(0)
                break

            updates, opt_state = optimizer.update(grad_avg, opt_state)
            W = optax.apply_updates(W, updates)

            # if np.any(np.isnan(loss)):
            #     info('Loss is nan...')

            loss_epoch += [loss_avg]

            if loss_avg < best.loss:
                best.loss = loss_avg
                best.model = network.model_y
                best.alphas = network.alphas
                best.W = np.array(W)
                if config.verbosity >= 3:
                    info(f'Found new best: {best.loss} on epoch {epoch}')
            else:
                del loss
                del grad
                del grad_avg
                del loss_avg

            if config.verbosity >= 2:
                info(f"\rΔWₛ = {a(-config.hyperparameters.lr * grad_avg)};\tℒₛ = {loss_avg:.6f};")
                info(f"W  = {a(W)}", end="")
        # elif config.optimizer == "bfgs":
        #     def avg_loss(W):
        #         total_loss = 0
        #         for batch in zip(*batches.values()): # minibatches
        #             loss = network.loss(W, *batch)
        #             total_loss += loss
        #         return total_loss / config.samples * config.batchsize
        #     opt_result = min_bfgs(avg_loss, W, method="BFGS")
        #     W = opt_result.x
        #     info(f"ℒₛ = {opt_result.fun:.6f} (success: {opt_result.success}) ")
        #     loss_epoch = opt_result.fun

        if config.verbosity >= 1:
            if epoch == 0 or epoch == config.epochs - 1 or \
                config.epochs < 100 or \
                (config.epochs >= 100 and config.epochs < 500 and epoch % 10 == 9) or \
                (config.epochs >= 500 and epoch % 100 == 99):
                info(f"Epoch: {epoch+1}, Loss: {np.mean(np.array(loss_epoch)):.10f}") #,\tW = {a(W)}")

        loss_history += [loss_epoch]
        # plotting.after_epoch(W, epoch, np.mean(np.array(loss_epoch)), show_plot=(epoch == epochs-1))

        network.next_operating_var()

    if config.verbosity >= 2:
        info(f"W = {a(W)}")
    return TrainOutput(W, loss_history, best)
