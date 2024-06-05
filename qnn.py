import pennylane as qml
import numpy as np
import tensorflow as tf
# import jax

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

import sympy as sp
import matplotlib.pyplot as plt
from hierarqcal import (
    Qhierarchy,
    Qcycle,
    Qpermute,
    Qmask,
    Qunmask,
    Qpivot,
    Qinit,
    Qmotif,
    Qmotifs,
    plot_motif,
    plot_circuit,
    Qunitary,
)

from itertools import combinations

from ansatz import a, b
from mel import get_spectrogram_dataset

# Hierarqcal
from hierarqcal.pennylane.pennylane_circuits import V2, U2, V4
from hierarqcal.pennylane.pennylane_helper import execute_circuit_pennylane

def get_circuit(hierq):
    dev = qml.device("default.qubit", wires=hierq.tail.Q)

    @qml.qnode(dev)
    def circuit():
        if isinstance(next(hierq.get_symbols(), False), sp.Symbol):
            # Pennylane doesn't support symbolic parameters, so if no symbols were set (i.e. they are still symbolic), we initialize them randomly
            hierq.set_symbols(np.random.uniform(0, 2 * np.pi, hierq.n_symbols))
        hierq(
            backend="pennylane"
        )  # This executes the compute graph in order
        # return [qml.expval(qml.PauliZ(wire)) for wire in hierq.tail.Q]
        return qml.expval(qml.PauliZ(hierq.tail.Q[0]))

    return circuit

# Si se quiere guardar una imagen del circuito
def draw_circuit(circuit, **kwargs):
    fig, ax = qml.draw_mpl(circuit)(**kwargs)

    plt.savefig('/home/alejandrolc/QuantumSpain/AutoQML/Hierarqcal/test_cirq.png')

def plot_losses(history):
    tr_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = np.array(range(len(tr_loss))) + 1
    plt.plot(epochs, tr_loss, label = "Training loss")
    plt.plot(epochs, val_loss, label = "Validation loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.savefig('/home/alejandrolc/QuantumSpain/AutoQML/Hierarqcal/losses.png')

    # plt.show()

### Pruebas
# numero de qubits
layers = 3
nqubits = 2**layers
# # hierq es la jerarquia en terminos de la libreria hierarqcal
# hierq = Qinit(2**nq) + (Qcycle(mapping=u2) + Qmask("*!", mapping=v2))*nq

# # get_circuit te devuelve el circuito en pennylane dado por la arquitectura que te da hierarqcal (aparentemente)
# circuit = get_circuit(hierq)

# QML
seed = 4321
np.random.seed(seed)
tf.random.set_seed(seed)

tf.keras.backend.set_floatx('float64')

# Carga del dataset y separacion de los datos de entrenamiento, test y validacion
# x, y = load_breast_cancer(return_X_y=True)

# x_tr, x_test, y_tr, y_test = train_test_split(x, y, train_size=0.8)
# x_val, x_test, y_val, y_test = train_test_split(x, y, train_size=0.5)

x_tr, x_test, x_val, y_tr, y_test, y_val = get_spectrogram_dataset()

# Escalado y normalizado de los datos
# scaler = MaxAbsScaler()
# x_tr = scaler.fit_transform(x_tr)

# x_test = scaler.transform(x_test)
# x_val = scaler.transform(x_val)

# x_test = np.clip(x_test, 0, 1)
# x_val = np.clip(x_val, 0, 1)

# # Reduccion de dimensionalidad
# pca = PCA(n_components = 2**nqubits)

# xs_tr = pca.fit_transform(x_tr)
# xs_test = pca.transform(x_test)
# xs_val = pca.transform(x_val)

xs_tr = x_tr
xs_test = x_test
xs_val = x_val

# print(x_tr.shape, y_tr.shape)

def ZZFeatureMap(nqubits, data):
    nload = min(len(data), nqubits)

    for i in range(nload):
        qml.Hadamard(i)
        qml.RZ(2.0 * data[i], wires = i)

    for pair in list(combinations(range(nload), 2)):
        q0 = pair[0]
        q1 = pair[1]

        qml.CZ(wires = [q0, q1])
        qml.RZ(2.0 * (np.pi - data[q0]) * (np.pi - data[q1]), wires = q1)
        qml.CZ(wires = [q0, q1])

def TwoLocal(nqubits, theta, reps = 1):
    for r in range(reps):
        for i in range(nqubits):
            qml.RY(theta[r * nqubits + i], wires = i)
        for i in range(nqubits - 1):
            qml.CNOT(wires = [i, i + 1])

    for i in range(nqubits):
        qml.RY(theta[reps * nqubits + i], wires = i)


dev = qml.device("default.qubit", wires = nqubits)

state_0 = [[1], [0]]
M = state_0 * np.conj(state_0).T

def U3(bits, symbols=None):
    qml.CRY(symbols[0], wires=[bits[0], bits[1]])

# TODO: Falta parametrizarlo correctamente
def rev_binary_tree(s_conv, s_pool, f, ansatz, theta):
    u21 = Qunitary(ansatz, 1, 2, [theta[0], theta[1]])
    u22 = Qunitary(ansatz, 1, 2, [theta[2], theta[3]])
    u23 = Qunitary(ansatz, 1, 2, [theta[4], theta[5]])

    cycle1 = Qcycle(stride=s_conv, mapping=u21)
    cycle2 = Qcycle(stride=s_conv, mapping=u22)
    cycle3 = Qcycle(stride=s_conv, mapping=u23)

    v2 = Qunitary(V2, 0, 2)

    pool = Qmask(f, mapping=v2)

    hierq = Qinit(range(nqubits)) + cycle1 + pool \
                                  + cycle2 + pool \
                                  + cycle3 + pool

    return hierq

def qnn_circuit(inputs, theta):
    # Codificacion con amplitudes
    qml.AmplitudeEmbedding(features=inputs, wires=range(nqubits), normalize=True)

    # ZZFeatureMap(nqubits, inputs)
    # TwoLocal(nqubits = nqubits, theta = theta)

    ansatz = a

    # Parte de hierarqcal
    hierq = rev_binary_tree(1, 2, "right", ansatz, theta)       # El 2 no hace nada, es decir, el s_pool

    # execute_circuit_pennylane(hierq, theta)
    hierq(backend="pennylane", symbols=theta)

    # for i in range(nqubits):
    #     U3([i, (i + 1)%nqubits], [theta[0]])
    #     # qml.CRY(theta[0], wires=[i, (i + 1)%nqubits])

    # V2([2, 0])
    # V2([3, 1])

    # U2([0, 1], [theta[1]])
    # # qml.CRY(theta[1], wires=[0, 1])

    # V2([1, 0])

    return qml.expval(qml.Hermitian(M, wires = [0]))
    # return qml.expval(qml.PauliZ(0))

# def circuit(params):
#     qnn_circuit(params, [0.1, 0.3])

# params = [0, 1]

# drawer = qml.draw(qnn_circuit)
# print(drawer(circuit(params)))

# @qml.qnode(dev)
# def qnn():
#     qnn_circuit([0]*4, [0.1, 0.3])
#     return qml.expval(qml.Hermitian(M, wires = [0]))

# draw_circuit(qnn)

qnn = qml.QNode(qnn_circuit, dev, interface="tf")

# QML con TensorFlow
weights = {"theta": layers*2}
qlayer = qml.qnn.KerasLayer(qnn, weights, output_dim = 1)

model = tf.keras.models.Sequential([qlayer])

opt = tf.keras.optimizers.Adam(learning_rate = 0.1)
model.compile(opt, loss = tf.keras.losses.BinaryCrossentropy())

earlystop = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 5, verbose = 1, restore_best_weights = True)

history = model.fit(xs_tr, y_tr, epochs = 200, shuffle = True, validation_data = (xs_val, y_val), batch_size = 10, callbacks = [earlystop])

print(model.summary())
print(model.get_weights())

plot_losses(history)

tr_acc = accuracy_score(model.predict(xs_tr) >= 0.5, y_tr)
val_acc = accuracy_score(model.predict(xs_val) >= 0.5, y_val)
test_acc = accuracy_score(model.predict(xs_test) >= 0.5, y_test)

print("Train accuracy: ", tr_acc)
print("Validation accuracy: ", val_acc)
print("Test accuracy: ", test_acc)