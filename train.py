from jax import random
import jax.numpy as np
from util.dataset import shuffle, batch_dataset, gen_dataset
from util.print import a, d, pad, info
from util.dotdict import DotDict

def train(
    network,
    dataset,
    plotting,
    key = None,
    lr: float = 1,
    lr_2: float = 0.5,
    epochs: int = 10,
    verbose: int = 0,
    batch_size: int = 16,
    W_init = None,
    best = None
):

    loss_history = []

    if W_init is None:
        print('W_init is None, please initialize')
        return W_init, loss_history, plotting

    W = W_init

    key, subkey = random.split(key)
    sampleset = gen_dataset(subkey, network.x_bounds, size=1024)

    if verbose >= 1:
        info(f"W₀ = {a(W)}")

    plotting.init(W)

    # train loop
    for epoch in range(epochs): # epochs
        try:
            loss_epoch = 0

            key, subkey = random.split(key)
            sampleset = shuffle(subkey, sampleset)

            key, subkey = random.split(key)
            batches = batch_dataset(sampleset, batch_size)

            for i, batch in enumerate(batches): # minibatches
                loss, grad = network.loss_and_grad(W, batch)
                grad_avg = np.average(grad, axis=0)
                loss_avg = np.average(loss)

                if loss_avg < best.loss:
                    best.loss = loss_avg
                    best.model_y = network.model_y
                    best.alphas = network.alphas
                    best.W = np.array(W)

                W -= lr * grad_avg
                loss_epoch += loss_avg

                if lr_2 > 0:
                    loss_2, grad_2 = network.loss_and_grad_secondary(W, *dataset[0])

                    W -= lr_2 * grad_2
                    loss_epoch += loss_2

                if verbose >= 2:
                    if lr_2 > 0:
                        print(f"\rΔWₛ = {a(-lr * grad_avg)};\tℒₛ = {loss_avg:.6f};\tΔWₐ = {a(-lr_2 * grad_2)};\tℒₐ = {loss_2:.6f};")
                    else:
                        print(f"\rΔWₛ = {a(-lr * grad_avg)};\tℒₛ = {loss_avg:.6f};")
                    print(f"W  = {a(W)}", end="")

            if verbose >= 1:
                info(f"Epoch: {epoch+1}, Loss: {loss_epoch},\tW = {a(W)}\n")

            loss_history += [loss_epoch]
            plotting.after_epoch(W, epoch, loss_epoch, show_plot=(epoch == epochs-1))

        except KeyboardInterrupt:
            info("Stopping...")
            info(f"Epoch: <{epoch+1}, Loss: {loss_epoch},\tW = {a(W)}")
            plotting.after_epoch(W, epoch, loss_epoch)
            break

    info(f"W = {a(W)}")
    return DotDict({"W": W, "loss_history": loss_history, "plotting": plotting, "best": best})