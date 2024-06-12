from collections import namedtuple
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from hierarqcal import Qcycle, Qmask, Qinit, Qunitary
import pennylane as qml
from pennylane.templates.embeddings import AngleEmbedding
import torch
from torch import nn

from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from mel import get_spectrogram_dataset
from ansatz import a, b, g, poolg

import seaborn as sns

import time

from common import *

# TODO: 
# Preparar wandb para presentar los datos

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
def get_circuit(hierq, device, angle_embdedding=False):
    # TODO: Manejar los keyword arguments: device, angle_embedding/embedding?, diff_method, interface, 
    #       Desempaquetar los parametros usando un diccionario con el operador ** i.e fun(**args) === fun(key1=arg1, key2=arg2, ...)
    #       Podria hacer tambien que aqui se llamase a una funcion en lugar de que los parametros vayan pasando de llamada en llamada...
    #       Debe tener todos los parametros necesarios para construir el circuito parametrizado
    dev = qml.device(device, wires=hierq.tail.Q) # default.qubit.torch

    @qml.qnode(dev, interface="torch") #, diff_method="adjoint")
    def circuit(x, symbols):
        hierq.set_symbols(symbols)

        if x is not None:
            if angle_embdedding:
                AngleEmbedding(x, wires=hierq.tail.Q, rotation="Y")
            else:
                qml.AmplitudeEmbedding(features=x, wires=hierq.tail.Q, normalize=True)
        hierq(backend="pennylane")  # This executes the compute graph in order
        
        # o = [[1], [0]] * np.conj([[1], [0]]).T
        # return qml.expval(qml.Hermitian(o, wires=hierq.head.Q[0]))
        return qml.probs(wires=hierq.head.Q[0])

    return circuit


# set up train loop
def train(x, y, circuit, symbols, epochs=50, lr=0.1, verbose=True):
    opt = torch.optim.Adam([symbols], lr)
    loss = nn.BCELoss()

    tensor_y = torch.tensor(y, dtype=torch.double)
    
    for it in range(epochs):
        opt.zero_grad()

        y_hat = circuit(x, symbols)
        loss_eval = loss(y_hat[:, 1], tensor_y)    # y_hat[:, 1]
        loss_eval.backward()

        opt.step()

        # if verbose:
        #     if it % 10 == 0:
        #         print(f"Loss at step {it}: {loss}")

    return symbols, loss

# TODO: conseguir el numero de simbolos directamente del ansatz
def get_qcnn(conv, pool, stride=1, step=1, offset=0, filter="right", share_weights=True):
    panstz = Qunitary(function=pool, n_symbols=2, arity=2)
    qcnn = (Qinit(range(8)) + 
            (Qcycle(
                stride=stride,
                step=step,
                offset=offset,
                mapping=Qunitary(conv, n_symbols=10, arity=2),
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
                    symbols, loss = train(dataset.x_train, dataset.y_train, qcnn, epochs=epochs, verbose=False)

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

def image_test(runs=1, epochs=50, **kwargs):
    res = pd.DataFrame(index=GENRES, columns=GENRES)

    hierq_params = {
        "conv": g,
        "pool": poolg,
        "step": 7,
        "filter": "right"
    }

    pipeline_image = Pipeline([
        ("scaler", ImageResize(size=256))
    ])

    for genre_pair in genre_combinations:
        x, y = get_spectrogram_dataset(genre_pair)

        cummulative_acc = 0
        cummulative_trtime = 0

        for i in range(runs):
            # dataset premaration
            Samples = namedtuple("samples", ["x_train", "x_test", "y_train", "y_test"])

            image_samples_raw = Samples(*train_test_split(x, y, train_size=0.7))

            image_samples_preprocessed = Samples(
                pipeline_image.fit_transform(image_samples_raw.x_train),
                pipeline_image.transform(image_samples_raw.x_test),
                image_samples_raw.y_train,
                image_samples_raw.y_test,
            )

            dataset = image_samples_preprocessed

            # x = torch.tensor(dataset.x_train, dtype=torch.double, device='cuda')
            # y = torch.tensor(dataset.y_train, dtype=torch.double, device='cuda')

            # x_test = torch.tensor(dataset.x_test, dtype=torch.double, device='cuda')

            # circuit and parameter preparation
            qcnn = get_qcnn(**hierq_params)

            # parameter initialization
            n_symbols = qcnn.n_symbols
            symbols = torch.rand(n_symbols, requires_grad=True)

            # build the circuit
            circuit = get_circuit(qcnn, device="lightning.qubit")

            # train qcnn
            t0 = time.time()
            symbols, loss = train(dataset.x_train, dataset.y_train, circuit, symbols, epochs=epochs, verbose=False)
            cummulative_trtime += (time.time() - t0)

            # get predictions
            y_hat = circuit(dataset.x_test, symbols)

            # evaluate
            y_hat = torch.argmax(y_hat, axis=1).detach().numpy()
            # y_hat = torch.round(y_hat).detach().numpy()

            accuracy = sum(
                [y_hat[k] == dataset.y_test[k] for k in range(len(y_hat))] # y_test.values en el original
            ) / len(y_hat)

            print(f"Run {i+1} - accuracy: {accuracy}")
            cummulative_acc += accuracy
        
        final_accuracy = cummulative_acc / runs
        avr_trtime = cummulative_trtime / runs

        res.loc[genre_pair[1], genre_pair[0]] = final_accuracy
        # res[genre_pair[0]][genre_pair[1]] = final_accuracy
        print(f"{genre_pair[0]} vs {genre_pair[1]} - accuracy: {final_accuracy}\n")
        print(f"#####\nTrain time: {avr_trtime} s\n#####")

    res = res.fillna(0)

    # heatmap mask
    mask = np.triu(np.ones_like(res, dtype=bool))

    heatmap = sns.heatmap(res, cmap = 'viridis', vmin=0, vmax=1, annot=True, mask=mask)

    figure = heatmap.get_figure()
    figure.savefig(f'heatmap_genres_{kwargs["heatmap_name"]}.png', dpi=400)

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

        qcnn = get_qcnn(step=7, filter="10", conv=g, pool=poolg)

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

if __name__ == "__main__":
    runs = 10
    epochs = 50

    # device = sys.argv[1]

    # test_params()
    print(torch.cuda.is_available())
    try:
        print(torch.cuda.get_device_name(0))
    except:
        pass

    ### Imagenes
    res = image_test(runs=runs, epochs=epochs, heatmap_name="test")

    ### Tabular
    # res = tabular_test(runs, epochs)

    ### Global
    # res = random_search_test()