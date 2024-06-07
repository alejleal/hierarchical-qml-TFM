import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

from common import *

from pathlib import Path
import os

# El parametro theta se utilizaria para entrenar posteriormente y hace referencia a los simbolos. Esta definido asi por defecto porque si
# Estan basados en como define U2, V2, ... en la libreria de hierarqcal
## TODO: No se como deberia hacerse bien esto pero esta practicamente copiado de la libreria
## Importante: los parametros tienen que ser con esos nombres, si no, se queja
def a(bits, symbols=None):
    qml.RY(symbols[0], wires=bits[0])
    qml.RY(symbols[1], wires=bits[1])
    qml.CNOT(wires=[bits[0], bits[1]])

def b(bits, symbols=None):
    qml.Hadamard(bits[0])
    qml.Hadamard(bits[1])

    qml.CZ(wires=[bits[0], bits[1]])

    qml.RX(symbols[0], wires=bits[0])
    qml.RX(symbols[1], wires=bits[1])

def g(bits, symbols=None):
    qml.RX(symbols[0], wires=bits[0])
    qml.RX(symbols[1], wires=bits[1])
    qml.RZ(symbols[2], wires=bits[0])
    qml.RZ(symbols[3], wires=bits[1])
    qml.CRZ(symbols[4], wires=[bits[1], bits[0]])
    qml.CRZ(symbols[5], wires=[bits[0], bits[1]])
    qml.RX(symbols[6], wires=bits[0])
    qml.RX(symbols[7], wires=bits[1])
    qml.RZ(symbols[8], wires=bits[0])
    qml.RZ(symbols[9], wires=bits[1])

# Pooling ansatz
def panstz(bits, symbols): 
    qml.CRZ(symbols[0], wires=[bits[0], bits[1]])
    qml.PauliX(wires=bits[0])
    qml.CRX(symbols[1], wires=[bits[0], bits[1]])

# Funcion auxiliar para dibujar circuitos
def draw_circuit(circuit, name="circuit", **kwargs):
    fig, ax = qml.draw_mpl(circuit)(**kwargs)

    if not Path("./images/").exists():
        os.makedirs("./images/")

    plt.savefig(f'./images/{name}.png')

if __name__ == "__main__":
    # Para probar que dibuja los circuitos bien
    nqubits = 2
    dev = qml.device("default.qubit", wires = nqubits)

    state_0 = [[1], [0]]
    M = state_0 * np.conj(state_0).T

    @qml.qnode(dev)
    def qnn():
        # Aqui se mete el circuito (qml.X() o anzatz() determinado)
        # g([0, 1], [0]*10)
        # hierq(backend="pennylane")
        panstz([0, 1], [0, 0])

        return qml.expval(qml.Hermitian(M, wires = [0]))

    draw_circuit(qnn, name=f"sample-circ")