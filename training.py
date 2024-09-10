from itertools import combinations
import pandas as pd
import numpy as np

import seaborn as sns
import jax

import sys

# self-libraries
from loader import dataset, full_dataset, dataset_yaseen, DatasetName
# from architecture import *
from pennylane_torch import run_torch
from pennylane_jax import run_jax
from torch_cnn import run_torch_cnn
from logger import *

genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"] # "hiphop" da problemas??
genre_combinations = combinations(genres, 2)
genre_combinations_reduced = [["country", "jazz"]]
genre_combinations_epochs = [['metal', 'pop'], ['country', 'jazz'], ['blues', 'classical']]

yaseen_types = ["N", "MS", "MR", "AS", "MVP"]
yaseen_types = ["N", "NOTN"]
yaseen_combinations = combinations(genres, 2)

# genre_combinations = [["country", "rock"]]
# genre_combinations = [["metal", "classical"]]

def experiment(runs, epochs, training, lr, combinations, verbose, **kwargs):
    subaudios = kwargs['subaudios'] if 'subaudios' in kwargs else 10
    features = kwargs['features'] if 'features' in kwargs else 256
    dataset_name = kwargs['dataset'] if 'dataset' in kwargs else 'GTZAN'
    filters = kwargs['filters'] if 'filters' in kwargs else (3, 7)
    method = kwargs['method'] if 'method' in kwargs else 'mel'
    qubits = kwargs['qubits'] if 'qubits' in kwargs else 8
    linear_sizes = kwargs['linear_sizes'] if 'linear_sizes' in kwargs else (120, 84)

    heatmap_name = f"{kwargs['id']}_{kwargs['project']}_f{features}_a{subaudios}_lr{lr}_e{epochs}_r{runs}"

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

            ds = dataset(dataset_name, genre_pair, nfeat=features, n=subaudios, method=method)
            # print(ds.x_train.shape)
            # fds = full_dataset(genre_pair)

            if training == 'jax':
                accuracy, params, trtime, key = run_jax(ds, epochs, lr, key, qubits, verbose)
            elif training == 'torch':
                accuracy, params, trtime = run_torch(ds, epochs, lr, verbose)
            elif training == 'cnn':
                accuracy, trtime = run_torch_cnn(ds, epochs, lr, filters, linear_sizes, verbose)

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
        draw_heatmap(res, heatmap_name)

        # run_img.finish()

    return res

if __name__ == "__main__":

    experiments = [
        ### CNNs

        {
            'project': 'CNN',
            'id': sys.argv[1] if len(sys.argv) > 1 else None,
            'runs': 5,
            'epochs': 250,
            'training': 'cnn',
            'lr': 0.001,
            'dataset': DatasetName.GTZAN.name,
            'combinations': genre_combinations,
            'features': 256,
            'subaudios': 1,
            'filters': (3,7),
            'linear_sizes': (120, 84),
            'method': 'mel',
            'graph': True,
            'verbose': True
        }

        ### QCNNs (Jax)

        # {
        #     'project': 'GTZAN',
        #     'id': sys.argv[1] if len(sys.argv) > 1 else None,
        #     'runs': 30,
        #     'epochs': 1000,
        #     'training': 'jax',
        #     'optim': 'adam',
        #     'lr': 0.01, # da igual
        #     'dataset': DatasetName.GTZAN.name,
        #     'combinations': genre_combinations_epochs,
        #     'features': 128,
        #     'qubits': 8,
        #     'subaudios': 1,
        #     'times': 1,
        #     'method': 'mel',
        #     'graph': True,
        #     'verbose': True
        # }

        # pruebas nuevas
        # {
        #     'project': 'GTZAN_JAX',
        #     'id': sys.argv[1] if len(sys.argv) > 1 else None,
        #     'runs': 5,
        #     'epochs': 400,
        #     'training': 'jax',
        #     'lr': 0.01, # da igual
        #     'dataset': DatasetName.GTZAN.name,
        #     'combinations': genre_combinations,
        #     'features': 256,
        #     'subaudios': 10,
        #     'qubits': 10,
        #     'method': 'mel',
        #     'graph': True,
        #     'verbose': True
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
        # init_wandb('wandb-test', ep)
        print(ep)
        experiment(**ep)
