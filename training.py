from itertools import combinations
import pandas as pd
import numpy as np

import seaborn as sns
import jax

import sys

# self-libraries
from loader import dataset, full_dataset, dataset_yaseen
# from architecture import *
from pennylane_torch import run_torch
from pennylane_jax import run_jax
from QuantumSpain.AutoQML.Hierarqcal.torch_cnn import run_torch_cnn
from logger import *

genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"] # "hiphop" da problemas??
genre_combinations = combinations(genres, 2)
genre_combinations_reduced = [["country", "jazz"]]

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

        # Crear un nuevo experimento
        name = f'{genre_pair[0]}_vs_{genre_pair[1]}'
        config = kwargs
        config['genres'] = genre_pair

        for i in range(runs):
            # Loggear nueva run
            run = init_wandb(kwargs['project'], f"run - {kwargs['id']}", name, name+f'_{i+1}', config)

            accuracy = 0
            trtime = 0

            ds = dataset(genre_pair)
            # fds = full_dataset(genre_pair)

            if training == 'jax':
                accuracy, params, trtime, key = run_jax(ds, epochs, lr, key, 8, verbose)
            elif training == 'torch':
                accuracy, params, trtime = run_torch(ds, epochs, lr, verbose)
            elif training == 'cnn':
                accuracy, trtime = run_torch_cnn(ds, epochs, lr, verbose)

            cummulative_acc += accuracy
            cummulative_trtime += trtime
            print(f"Run {i+1} - accuracy: {accuracy}\n    {trtime} s")

            run.finish()
        
        final_accuracy = cummulative_acc / runs
        avr_trtime = cummulative_trtime / runs

        res.loc[genre_pair[1], genre_pair[0]] = final_accuracy
        # res[genre_pair[0]][genre_pair[1]] = final_accuracy

        # draw_heatmap(res, kwargs["heatmap_name"])
        
        print(f"{genre_pair[0]} vs {genre_pair[1]} - accuracy: {final_accuracy}\n")
        print(f"#####\nAvg. train time: {avr_trtime} s")
        print(f"Total time: {cummulative_trtime} s\n#####\n")

    res = res.fillna(0)

    if "graph" in kwargs and kwargs['graph']:
        # run_img = wandb.init(project, f"run - {kwargs['id']}")
        draw_heatmap(res, kwargs["heatmap_name"])

        # run_img.finish()

    return res

if __name__ == "__main__":

    experiments = [
        {
            'project': 'GTZAN_JAX',
            'runs': 1,
            'epochs': 300,
            'training': 'jax',
            'lr': 0.1,
            'combinations': genre_combinations,
            'graph': True,
            'heatmap_name': 'share_weights_false',
            'verbose': True,
            'id': sys.argv[1] if len(sys.argv) > 1 else None
        }
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
        # {
        #     'runs': 5,
        #     'epochs': 100,
        #     'training': 'cnn',
        #     'lr': 0.001, # no se utiliza en este caso
        #     'combinations': genre_combinations,
        #     'graph': False,
        #     'heatmap_name': 'full_cnn_torch',
        #     'verbose': True
        # }
    ]

    for ep in experiments:
        # init_wandb('wandb-test', ep)
        experiment(**ep)
