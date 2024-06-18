import jax
from jax import numpy as jnp
import optax
import pandas as pd

# from loader import *
from architecture import get_circuit, get_qcnn, a, b, g, poolg
import time

# https://pennylane.ai/qml/demos/tutorial_How_to_optimize_QML_model_using_JAX_and_JAXopt/
# https://pennylane.ai/qml/demos/tutorial_How_to_optimize_QML_model_using_JAX_and_Optax/

device = 'default.qubit'
interface = 'jax'
circuit = get_circuit(get_qcnn(g, poolg), device, interface)

# circuit = jax.jit(jax.vmap(circuit, in_axes=(0, None)))

@jax.jit
def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = jnp.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    # print(predictions.shape, targets.shape)

    ce = -jnp.sum(targets*jnp.log(predictions+1e-9))/N
    return ce

@jax.jit
def loss_fn(params, data, targets):
    predictions = circuit(data, params)

    loss = cross_entropy(predictions, targets)
    # loss = jnp.mean(optax.losses.sigmoid_binary_cross_entropy(predictions, targets))
    
    return loss

## completely jitted
@jax.jit
def update_step_jit(i, args):
    params, opt_state, data, targets, opt, print_training = args

    loss_val, grads = jax.value_and_grad(loss_fn)(params, data, targets)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    def print_fn():
        jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)
        y_hat = circuit(data, params)

        y_hat = jnp.where(y_hat >= 0.5, 1, 0)
        accuracy = jnp.mean(jnp.where(y_hat == targets, 1, 0))

        jax.debug.print("Accuracy: {accuracy}", accuracy=accuracy)

    # if print_training=True, print the loss every 5 steps
    jax.lax.cond((jnp.mod(i, 1) == 0) & print_training, print_fn, lambda: None)

    return (params, opt_state, data, targets, opt, print_training)

@jax.jit
def optimization_jit(params, data, targets, epochs, lr, print_training=False):
    opt = optax.adam(learning_rate=lr)

    opt_state = opt.init(params)
    args = (params, opt_state, data, targets, opt, print_training)
    (params, opt_state, _, _, _, _) = jax.lax.fori_loop(0, epochs, update_step_jit, args)

    return params

def run_jax(dataset, epochs, lr, key, verbose=False):
    key, subkey = jax.random.split(key)

    # dataset preparation
    data = dataset.x_train
    targets = dataset.y_train

    # initial parameters
    params = jax.random.uniform(subkey, shape=(36,))

    # training loop
    t0 = time.time()
    params = optimization_jit(params, data, targets, epochs, lr, print_training=verbose)
    trtime = time.time() - t0

    # get predictions
    y_hat = circuit(dataset.x_test, params)

    # evaluate
    # y_hat = jnp.argmax(y_hat, axis=1)
    y_hat = jnp.where(y_hat >= 0.5, 1, 0)

    accuracy = jnp.mean(jnp.where(y_hat == dataset.y_test, 1, 0))

    return accuracy, params, trtime, key


## Testing grounds
if __name__ == "__main__":
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
