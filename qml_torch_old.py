from collections import namedtuple
import pandas as pd
import numpy as np
import sympy as sp
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from hierarqcal import Qcycle, Qmask, Qinit, Qunitary
import pennylane as qml
from pennylane.templates.embeddings import AngleEmbedding
import torch
from torch import nn

import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from mel import get_spectrogram_dataset
from ansatz import a, b, g, panstz

import seaborn as sns

import time, sys

from common import *

# print(torch.cuda.get_device_name(0))

# TODO: 
# Parametrizar el numero de qubits y otros hiperparametros
# Parejas de generos para comparacion
# Preparar wandb para presentar los datos
# Seguir metiendo ansatz

# ANGLE_EMBEDDING = False

def get_tabular_dataset(genres=["country", "rock"]):
    PATH_DATA = f"/home/alejandrolc/QuantumSpain/AutoQML/Data/features_3_sec.csv"
    data = pd.read_csv(PATH_DATA)
    # remove filename and length columns
    data = data.drop(columns=["filename", "length"])
    # specify genre pair
    
    # filter data
    data = data[data["label"].isin(genres)]
    # set label to 0 or 1
    data["label"] = data["label"].map({genres[0]: 0, genres[1]: 1})
    # specify target and features
    target = "label"
    x, y = data.drop(columns=[target]), data[target]

    return x, y.values

# Comprobar mel.py para ver como se guardan los csv, como se leen y ver el formato en el que devuelvo la imagen para que tf se lo trague 
class ImageResize(BaseEstimator, TransformerMixin):
    """
    Resizes an image
    """

    def __init__(self, size=None):
        self.size = size

    def fit(self, X, y=None):
        """returns itself"""
        if self.size == None:
            # assume image is n * width * height np array
            self.size = X.shape[1] * X.shape[2]
        return self

    def transform(self, X, y=None):
        X_resize = tf.image.resize(X[..., np.newaxis][:], (self.size, 1)).numpy()
        X_squeezed = tf.squeeze(X_resize).numpy()
        # print(X_squeezed.shape)
        return X_squeezed


# set up pennylane circuit
def get_circuit(hierq, x=None, device="default.qubit.torch", angle_embdedding=False):
    dev = qml.device(device, wires=hierq.tail.Q) # default.qubit.torch

    @qml.qnode(dev, interface="torch")#, diff_method="adjoint")
    def circuit():
        if isinstance(next(hierq.get_symbols(), False), sp.Symbol):
            # Pennylane doesn't support symbolic parameters, so if no symbols were set (i.e. they are still symbolic), we initialize them randomly
            hierq.set_symbols(np.random.uniform(0, 2 * np.pi, hierq.n_symbols))
        if x is not None:
            # TODO: No hace falta que diga nada
            if angle_embdedding:
                AngleEmbedding(x, wires=hierq.tail.Q, rotation="Y")
            else:
                qml.AmplitudeEmbedding(features=x, wires=hierq.tail.Q, normalize=True)
        hierq(backend="pennylane")  # This executes the compute graph in order
        
        # o = [[1], [0]] * np.conj([[1], [0]]).T
        # return qml.expval(qml.Hermitian(o, wires=[0]))
        return qml.probs(wires=hierq.head.Q[0])

    return circuit


# set up train loop
# @torch.jit.script
def train(x, y, motif, N=70, lr=0.1, verbose=True, device="default.qubit.torch"):
    n_symbols = motif.n_symbols
    if n_symbols > 0:
        symbols = torch.rand(n_symbols, requires_grad=True)
        opt = torch.optim.Adam([symbols], lr=lr)
        for it in range(N):
            opt.zero_grad()
            loss = objective_function(motif, symbols, x, y, device)
            loss.backward()
            opt.step()
            if verbose:
                if it % 10 == 0:
                    print(f"Loss at step {it}: {loss}")
    else:
        symbols = None
        loss = objective_function(motif, [], x, y)
    return symbols, loss

# specify objective function
def objective_function(motif, symbols, x, y, device):
    motif.set_symbols(symbols)
    circuit = get_circuit(motif, x, device)
    y_hat = circuit()
    # cross entropy loss
    # m = nn.Sigmoid()
    loss = nn.BCELoss()
    # index 1 corresponds to predictions for being in class 1
    # use mse
    # loss = nn.MSELoss()
    # print(y_hat.shape)
    # print(y)
    loss = loss(y_hat[:, 1], torch.tensor(y, dtype=torch.double))       # y.values en el original
    # loss = loss(m(y_hat[:,1]),torch.tensor(y.values,dtype=torch.double))
    return loss

# TODO: conseguir el numero de simbolos directamente del ansatz
def get_qcnn(stride=1, step=1, offset=0, conv_ansatz=a, pool_ansatz=hierq_gates["CNOT"], filter="right", share_weights=True):
    panstz = Qunitary(function=pool_ansatz, n_symbols=2, arity=2)
    qcnn = (Qinit(range(8)) + 
            (Qcycle(
                stride=stride,
                step=step,
                offset=offset,
                mapping=Qunitary(conv_ansatz, n_symbols=10, arity=2),
                share_weights=share_weights
            )
            + Qmask(filter, mapping=panstz)
        )
        * 3
    )

    return qcnn

def get_qcnn_tabular(filter="right", sc=1, sp=0, N=8, conv_ansatz=a, pool_ansatz=hierq_gates["CNOT"]):
    # panstz = Qunitary(function=pool_ansatz, n_symbols=2, arity=2)
    qcnn = (Qinit(range(8)) + 
            (Qcycle(
                stride=sc,
                step=1,
                offset=0,
                mapping=Qunitary(conv_ansatz, n_symbols=2, arity=2),
                share_weights=True
            )
            + Qmask(filter, mapping=pool_ansatz, strides=sp)
        )
        * 3
    )

    return qcnn


def get_specs(circuit, theta=[0]*1000, shared_weights=True):
    specs = qml.specs(circuit)(theta)

    nwires = specs['num_used_wires']
    nparams = specs['num_trainable_params']

    if shared_weights:
        nparams = nparams/nwires

    return nwires, nparams

def tabular_test(runs=1, epochs=50):
    x, y = get_tabular_dataset()

    ANGLE_EMBEDDING = True
    
    # setup preprocessing pipeline
    pipeline = Pipeline(
        [
            ("scaler", preprocessing.MinMaxScaler((0, np.pi / 2))),
            ("pca", PCA(8)),
        ]
    )

    res = pd.DataFrame(index=range(24), columns=range(8))

    for sp in range(4):
        for sc in range(8):
            for j in range(len(PATTERNS)):
                cummulative_acc = 0

                filter = PATTERNS[j]

                for i in range(runs):
                    # dividir el dataset
                    Samples = namedtuple("samples", ["x_train", "x_test", "y_train", "y_test"])

                    samples_raw = Samples(*train_test_split(x, y, test_size=0.3))

                    samples_preprocessed = Samples(
                        pipeline.fit_transform(samples_raw.x_train),
                        pipeline.transform(samples_raw.x_test),
                        samples_raw.y_train,
                        samples_raw.y_test,
                    )

                    dataset = samples_preprocessed

                    # plot circuit
                    # fig, ax = qml.draw_mpl(get_circuit(qcnn))()
                    # plt.savefig(f'/home/alejandrolc/QuantumSpain/AutoQML/Hierarqcal/qml_ex.png')

                    qcnn = get_qcnn_tabular(filter=filter, sc=sc, sp=sp)

                    # train qcnn
                    symbols, loss = train(dataset.x_train, dataset.y_train, qcnn, N=epochs, verbose=False)

                    # get predictions
                    circuit = get_circuit(qcnn, dataset.x_test)
                    y_hat = circuit()

                    # evaluate
                    y_hat = torch.argmax(y_hat, axis=1).detach().numpy()
                    accuracy = sum(
                        [y_hat[k] == dataset.y_test[k] for k in range(len(y_hat))] # y_test.values en el original
                    ) / len(y_hat)

                    print(f"Run {i+1} - accuracy: {accuracy}")
                    cummulative_acc += accuracy
    
                final_accuracy = cummulative_acc / runs

                res[sp + 4*j][sc] = final_accuracy
    return res

def image_test(runs=1, epochs=50, device="default.qubit.torch"):
    res = pd.DataFrame(index=GENRES, columns=GENRES)

    for genre_pair in genre_combinations:
        x, y = get_spectrogram_dataset(genre_pair)

        pipeline_image = Pipeline([
            ("scaler", ImageResize(size=256))
        ])

        cummulative_acc = 0

        for i in range(runs):
            # dividir el dataset
            Samples = namedtuple("samples", ["x_train", "x_test", "y_train", "y_test"])

            image_samples_raw = Samples(*train_test_split(x, y, train_size=0.7))

            image_samples_preprocessed = Samples(
                pipeline_image.fit_transform(image_samples_raw.x_train),
                pipeline_image.transform(image_samples_raw.x_test),
                image_samples_raw.y_train,
                image_samples_raw.y_test,
            )

            dataset = image_samples_preprocessed

            # plot circuit
            # fig, ax = qml.draw_mpl(get_circuit(qcnn))()
            # plt.savefig(f'/home/alejandrolc/QuantumSpain/AutoQML/Hierarqcal/qml_ex.png')

            qcnn = get_qcnn(step=7, filter="10", conv_ansatz=g, pool_ansatz=panstz)

            # train qcnn
            symbols, loss = train(dataset.x_train, dataset.y_train, qcnn, N=epochs, verbose=False, device=device)

            # get predictions
            circuit = get_circuit(qcnn, dataset.x_test)
            y_hat = circuit()

            # evaluate
            y_hat = torch.argmax(y_hat, axis=1).detach().numpy()
            accuracy = sum(
                [y_hat[k] == dataset.y_test[k] for k in range(len(y_hat))] # y_test.values en el original
            ) / len(y_hat)

            print(f"Run {i+1} - accuracy: {accuracy}")
            cummulative_acc += accuracy
        
        final_accuracy = cummulative_acc / runs
        res[genre_pair[0]][genre_pair[1]] = final_accuracy
        print(f"{genre_pair[0]} vs {genre_pair[1]} - accuracy: {final_accuracy}\n")

    res = res.fillna(0)
    # print(res)

    # Mascara para que solo salga el mapa de calor para la matriz triangular inferior
    mask = np.triu(np.ones_like(res, dtype=bool))

    heatmap = sns.heatmap(res, cmap = 'viridis', vmin=0, vmax=1, annot=True, mask=mask)

    figure = heatmap.get_figure()
    figure.savefig('heatmap_genres_g-even.png', dpi=400)

    return res

def random_search_test(n=10):
    ANGLE_EMBEDDING = False

    for genre_pair in [["country", "rock"]]:
        x, y = get_spectrogram_dataset(genre_pair)

        pipeline_image = Pipeline([
            ("scaler", ImageResize(size=256))
        ])

        # dividir el dataset
        Samples = namedtuple("samples", ["x_train", "x_test", "y_train", "y_test"])

        image_samples_raw = Samples(*train_test_split(x, y, train_size=0.7))

        image_samples_preprocessed = Samples(
            pipeline_image.fit_transform(image_samples_raw.x_train),
            pipeline_image.transform(image_samples_raw.x_test),
            image_samples_raw.y_train,
            image_samples_raw.y_test,
        )

        dataset = image_samples_preprocessed

        # plot circuit
        # fig, ax = qml.draw_mpl(get_circuit(qcnn))()
        # plt.savefig(f'/home/alejandrolc/QuantumSpain/AutoQML/Hierarqcal/qml_ex.png')

        qcnn = get_qcnn(step=7, filter="10", conv_ansatz=g, pool_ansatz=panstz)

        # train qcnn
        # symbols, loss = train(dataset.x_train, dataset.y_train, qcnn, N=epochs, verbose=False)

        delta = 2*np.pi/n
        params = np.zeros(36)

        i = n**35

        mods = np.zeros(36)
        m = n
        for j in range(36):
            mods[j] = m
            m = m*n

        print(mods)
        print(delta)

        best = 0

        while i < n**36:
            # br_i = i*np.ones(36)
            # params = (delta * (br_i//mods).astype(type(delta)))%(2*np.pi)
            i += n**5
            symbols = torch.rand(36)
            # print(params)
            
            # get predictions
            qcnn.set_symbols(symbols)
            circuit = get_circuit(qcnn, dataset.x_test)
            y_hat = circuit()


            # evaluate
            y_hat = torch.argmax(y_hat, axis=1).detach().numpy()
            # print(y_hat)
            accuracy = sum(
                [y_hat[k] == dataset.y_test[k] for k in range(len(y_hat))] # y_test.values en el original
            ) / len(y_hat)

            if best < accuracy:
                best = accuracy

                print(f"Run {i} - accuracy: {accuracy}")
                print(symbols)

    # res = res.fillna(0)
    # print(res)

    return None

def test_params():
    ansatz = g

    maxarray = [0]*1000

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(theta):
        ansatz(list(range(1000)), maxarray)
        return qml.expval(qml.PauliZ(1))

    print(get_specs(circuit))

if __name__ == "__main__":
    # dev = qml.device("lightning.gpu", wires=2)
    runs = 10
    epochs = 50
    genres = ["country", "rock"]

    # device = sys.argv[1]

    # test_params()
    print(torch.cuda.is_available())
    try:
        print(torch.cuda.get_device_name(0))
    except:
        pass

    ### Imagenes
    t0 = time.time()
    res = image_test(runs=runs, epochs=epochs)
    print(f"#####\nTrain time: {time.time() - t0} s\n#####")

    ### Tabular
    # res = tabular_test(runs, epochs)

    ### Global
    # res = random_search_test()

    # print(res)
    # print("circ_specs: ", qml.specs(circuit)())