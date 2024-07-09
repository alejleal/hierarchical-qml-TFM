from itertools import combinations
import pandas as pd
import numpy as np

import seaborn as sns
import jax

# self-libraries
from loader import dataset, full_dataset, dataset_yaseen
# from architecture import *
from pennylane_torch import run_torch
from pennylane_jax import run_jax

# TODO: 
# Preparar wandb para presentar los datos

genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"] # "hiphop" da problemas??
genre_combinations = combinations(genres, 2)

yaseen_types = ["N", "MS", "MR", "AS", "MVP"]
yaseen_types = ["N", "NOTN"]
yaseen_combinations = combinations(genres, 2)

# genre_combinations = [["country", "rock"]]
# genre_combinations = [["metal", "classical"]]

def experiment(runs, epochs, training, lr, combinations, verbose, **kwargs):
    # TODO: poner los index como parametros
    res = pd.DataFrame(index=genres, columns=genres, dtype=float)

    # rng and seed preparation (for jax)
    if training == 'jax':
        rng = jax.random.PRNGKey(0)
        key, _ = jax.random.split(rng)

    for genre_pair in combinations:
        cummulative_acc = 0
        cummulative_trtime = 0

        for i in range(runs):
            accuracy = 0
            trtime = 0

            ds = dataset(genre_pair)
            # fds = full_dataset(genre_pair)

            if training == 'jax':
                accuracy, params, trtime, key = run_jax(ds, epochs, key, verbose)
            elif training == 'torch':
                accuracy, params, trtime = run_torch(ds, epochs, lr, verbose)

            cummulative_acc += accuracy
            cummulative_trtime += trtime
            print(f"Run {i+1} - accuracy: {accuracy}\n    {trtime} s")
        
        final_accuracy = cummulative_acc / runs
        avr_trtime = cummulative_trtime / runs

        res.loc[genre_pair[1], genre_pair[0]] = final_accuracy
        # res[genre_pair[0]][genre_pair[1]] = final_accuracy
        
        print(f"{genre_pair[0]} vs {genre_pair[1]} - accuracy: {final_accuracy}\n")
        print(f"#####\nAvg. train time: {avr_trtime} s")
        print(f"Total time: {cummulative_trtime} s\n#####\n")

    res = res.fillna(0)

    if "heatmap_name" in kwargs:
        draw_heatmap(res, kwargs["heatmap_name"])

    return res

def draw_heatmap(res, name, cmap = 'viridis'):
    # heatmap mask
    mask = np.triu(np.ones_like(res, dtype=bool))

    heatmap = sns.heatmap(res.iloc[1:,:-1], cmap=cmap, vmin=0, vmax=1, annot=True, mask=mask[1:,:-1])

    figure = heatmap.get_figure()
    figure.savefig(f'./images/heatmap_{name}.png', dpi=400)

if __name__ == "__main__":
    experiments = [
        # {
        #     'runs': 5,
        #     'epochs': 50,
        #     'training': 'jax',
        #     'lr': 0.1,
        #     'combinations': genre_combinations,
        #     'heatmap_name': 'jax_gtzan',
        #     'verbose': False
        # }
        # {
        #     'runs': 3,
        #     'epochs': 1,
        #     'training': 'torch',
        #     'lr': 0.1,
        #     'combinations': [["country", "rock"]],
        #     'heatmap_name': '-',
        #     'verbose': True
        # }
        # {
        #     'runs': 5,
        #     'epochs': 50,
        #     'training': 'jax',
        #     'lr': 0.001, # no se utiliza en este caso
        #     'combinations': yaseen_combinations,
        #     'heatmap_name': 'jax_yaseen_binary',
        #     'verbose': True
        # }
        # {
        #     'runs': 5,
        #     'epochs': 40,
        #     'training': 'jax',
        #     'lr': 0.001, # no se utiliza en este caso
        #     'combinations': [["country", "jazz"]],
        #     'heatmap_name': 'jax_bio',
        #     'verbose': True
        # }
    ]

    for ep in experiments:
        experiment(**ep)
