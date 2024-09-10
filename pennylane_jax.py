import jax
from jax import numpy as jnp
import optax
import pandas as pd

# from loader import *
from architecture import get_circuit, get_qcnn, a, b, g, universal, poolg, get_num_params, qcnn_12, qcnn_center
import time
import wandb

from biofind import run_random_search
from loader import get_spectrogram_dataset

import sys

from sklearn.metrics import balanced_accuracy_score

from functools import partial

# https://pennylane.ai/qml/demos/tutorial_How_to_optimize_QML_model_using_JAX_and_JAXopt/
# https://pennylane.ai/qml/demos/tutorial_How_to_optimize_QML_model_using_JAX_and_Optax/

device = 'default.qubit'
interface = 'jax'
# circuit = get_circuit(get_qcnn(universal, poolg), device, interface)
# circuit = get_circuit(qcnn_12(universal, poolg), device, interface)

# 12 qubits
conv = universal
pool = poolg
hierq = qcnn_12
wires = 12

# 8 qubits
conv = g
hierq = get_qcnn
wires = 8

# 9 qubits
# hierq = qcnn_center
# wires = 9

# # 10 qubits
# hierq = qcnn_12
# wires = 10


qcnn = hierq(conv, pool, wires=wires, share_weights=True)
circuit = get_circuit(qcnn, device, interface)

circuit_g = get_circuit(get_qcnn(g, poolg), device, interface)
circuit_u = get_circuit(get_qcnn(universal, poolg), device, interface)

circuit = jax.vmap(circuit, in_axes=(0, None))

opt = optax.adam(learning_rate=0.01)

@jax.jit
def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = jnp.clip(predictions, epsilon, 1. - epsilon)
    # N = predictions.shape[0]
    # print(predictions.shape, targets.shape)

    ce = -jnp.mean(targets*jnp.log(predictions) + (1 - targets)*jnp.log(1-predictions))
    return ce

@jax.jit
def loss_fn(params, data, targets):
    predictions = circuit(data, params)

    loss = cross_entropy(predictions, targets)
    # jax.debug.print("preds: {predictions}", predictions=predictions.shape)
    # loss = jnp.mean(optax.losses.sigmoid_binary_cross_entropy(predictions, targets))
    
    return loss

value_and_grad_fn = jax.value_and_grad(loss_fn)

## completely jitted
@jax.jit
def update_step_jit(i, args):
    params, opt_state, data, targets, print_training, data_test, metrics = args

    # jax.debug.print("nans: {params} ", params=params)

    loss_val, grads = value_and_grad_fn(params, data, targets)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    ### Loggear metricas
    # x = x.at[idx].set(y)
    metrics = metrics.at[0, i].set(loss_val)
    # metrics['loss'][i] = loss_val

    y_hat = circuit(data_test.x_test, params)
    y_hat = jnp.where(y_hat >= 0.5, 1, 0)
    test_accuracy = jnp.mean(jnp.where(y_hat == data_test.y_test, 1, 0))

    metrics = metrics.at[1, i].set(test_accuracy)
    # metrics['test_acc'][i] = test_accuracy

    y_hat = circuit(data, params)
    y_hat = jnp.where(y_hat >= 0.5, 1, 0)
    train_accuracy = jnp.mean(jnp.where(y_hat == targets, 1, 0))

    metrics = metrics.at[2, i].set(train_accuracy)
    # metrics['train_acc'][i] = train_accuracy
    ###

    def print_fn():
        jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)
        # res = circuit(data, params)

        # y_hat = jnp.where(res >= 0.5, 1, 0)
        # accuracy = jnp.mean(jnp.where(y_hat == targets, 1, 0))

        # jax.debug.print("Accuracy: {accuracy} - Loss: {loss}", accuracy=accuracy, loss=loss_val)

    # if print_training=True, print the loss every 5 steps
    jax.lax.cond((jnp.mod(i, 1) == 0) & print_training, print_fn, lambda: None)

    return (params, opt_state, data, targets, print_training, data_test, metrics)

@jax.jit
def optimization_jit(params, data, targets, epochs, data_test, metrics, print_training=False):
    opt_state = opt.init(params)
    args = (params, opt_state, data, targets, print_training, data_test, metrics)
    (params, opt_state, _, _, _, data_test, metrics) = jax.lax.fori_loop(0, epochs, update_step_jit, args)

    return params, metrics

def run_jax(ds, epochs, lr, key, qubits, verbose=False):
    key, subkey = jax.random.split(key)

    # jnp.set_printoptions(threshold=sys.maxsize)
    # dataset preparation
    data = ds.x_train
    targets = ds.y_train

    # metrics = { key : jnp.zeros(epochs) for key in ['loss', 'train_acc', 'test_acc'] }
    metrics = jnp.zeros((3, epochs))

    # initial parameters
    num_params = qcnn.n_symbols # 17*6 #(17*4=qcnn12) #
    # num_params = 238
    params = jax.random.uniform(subkey, shape=(num_params,))*jnp.pi

    t0 = time.time()
    # training loop
    params, metrics = optimization_jit(params, data, targets, epochs, ds, metrics, print_training=verbose)
    trtime = time.time() - t0

    ## Logging metrics
    circuit_log = {
        "hierarchy": hierq.__name__,
        "convolution": conv.__name__,
        "pooling": pool.__name__,
        "qubits": qubits,
        "num_params": num_params
    }
    wandb.run.config["circuit"] = circuit_log
    for i in range(epochs):
        wandb.log({ key : metrics[j][i] for j, key in enumerate(['loss', 'test_acc', 'train_acc'])})

    # get predictions
    y_hat = circuit(ds.x_test, params)

    # evaluate
    # y_hat = jnp.argmax(y_hat, axis=1)
    y_hat = jnp.where(y_hat >= 0.5, 1, 0)

    accuracy = jnp.mean(jnp.where(y_hat == ds.y_test, 1, 0))
    
    # bal_acc=balanced_accuracy_score(ds.y_test, y_hat)
    # print(bal_acc)

    return accuracy, params, trtime, key

def main():
    runs = 1
    epochs = 5

    from loader import dataset
    from itertools import combinations
    import seaborn as sns
    
    dataset = dataset()

    data = dataset.x_train
    targets = dataset.y_train

    rng = jax.random.PRNGKey(1234)
    key, subkey = jax.random.split(rng)
    weights = jax.random.uniform(subkey, shape=(36,))
    params = {"weights": weights}

    # Compilation execution
    t0 = time.time()
    optimization_jit(params, data, targets, epochs, print_training=False)
    t1 = time.time()
    print(f"\nTime elapsed: {t1- t0} s")

    # prf.run('reruns(params, data, targets, False, runs=10)', 'run-stats')

    # stats = pstats.Stats('run-stats')
    # stats.sort_stats(SortKey.CUMULATIVE).print_stats()

    # with jax.profiler.trace("./tmp/jax-trace", create_perfetto_link=True):
        # Run the operations to be profiled
        # optimization_jit(params, data, targets, print_training=False)

    for i in range(runs):
        t0 = time.time()
        optimization_jit(params, data, targets, epochs, print_training=False)
        t1 = time.time()
        print(f"Time elapsed: {t1- t0} s")

    # raise Exception(None)
    genres = ["blues", "classical", "country", "disco", "jazz", "metal", "pop", "reggae", "rock"] # "hiphop" da problemas??
    genre_combinations = combinations(genres, 2)

    res = pd.DataFrame(index=genres, columns=genres)

    for genre_pair in genre_combinations:
        cummulative_acc = 0
        cummulative_trtime = 0

        for i in range(runs):
            key, subkey = jax.random.split(key)
            
            # dataset premaration
            dataset = dataset(genre_pair)

            data = dataset.x_train
            targets = dataset.y_train

            # key = jax.random.PRNGKey(758493)  # Random seed is explicit in JAX
            weights = jax.random.uniform(subkey, shape=(36,))
            params = {"weights": weights}

            t0 = time.time()
            params = optimization_jit(params, data, targets, epochs, print_training=False)
            t1 = time.time()

            # get predictions
            y_hat = circuit(dataset.x_test, params["weights"])

            # evaluate
            y_hat = jnp.argmax(y_hat, axis=1)
            # y_hat = torch.round(y_hat).detach().numpy()

            accuracy = jnp.mean(jnp.where(y_hat == dataset.y_test, 1, 0))
            #     [y_hat[k] == ds.y_test[k] for k in range(len(y_hat))] # y_test.values en el original
            # ) / len(y_hat)
            

            trtime = t1 - t0
            cummulative_trtime += trtime
            print(f"\nTime elapsed: {trtime} s")
            print(f"Run {i+1} - accuracy: {accuracy}")
            cummulative_acc += accuracy
        
        final_accuracy = cummulative_acc / runs
        avr_trtime = cummulative_trtime / runs

        res.loc[genre_pair[1], genre_pair[0]] = final_accuracy
        # res[genre_pair[0]][genre_pair[1]] = final_accuracy
        print(f"{genre_pair[0]} vs {genre_pair[1]} - accuracy: {final_accuracy}\n")
        print(f"#####\nTrain time: {avr_trtime} s\n#####")

    # print(f"#####\nCummulative time: {cummulative_trtime} s\n#####")

    res = res.fillna(0)

    # heatmap mask
    mask = jnp.triu(jnp.ones_like(res, dtype=bool)) # No se si funciona con jnp

    heatmap = sns.heatmap(res, cmap = 'viridis', vmin=0, vmax=1, annot=True, mask=mask)

    figure = heatmap.get_figure()
    figure.savefig(f'heatmap_genres_jax.png', dpi=400)

## Testing grounds
if __name__ == "__main__":
    #main()

    hhds = get_spectrogram_dataset(['hiphop'])

    rng = jax.random.PRNGKey(0)
    key, _ = jax.random.split(rng)
    key, subkey = jax.random.split(key)

    params = jax.random.uniform(subkey, shape=(51,))*jnp.pi

    import tensorflow as tf
    import numpy as np
    X_resize = tf.image.resize(hhds[0][..., np.newaxis][:], (256, 1)).numpy()
    X_squeezed = tf.squeeze(X_resize).numpy()
    # print(X_squeezed)
    print(np.argwhere(np.isnan(circuit_g(X_squeezed, params))))
    print(X_squeezed[np.argwhere(np.isnan(circuit_g(X_squeezed, params)))])
