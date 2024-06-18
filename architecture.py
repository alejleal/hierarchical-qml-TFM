import pennylane as qml
from hierarqcal import Qcycle, Qmask, Qinit, Qunitary

import jax
from jax import numpy as jnp

# n_wires = 8

# device = "default.qubit.torch"
# interface = 'torch'

# device = "default.qubit"
# interface = 'jax'
# dev = qml.device(device, wires=n_wires)

## Ansatzs

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
def poolg(bits, symbols): 
    qml.CRZ(symbols[0], wires=[bits[0], bits[1]])
    qml.PauliX(wires=bits[0])
    qml.CRX(symbols[1], wires=[bits[0], bits[1]])


## Hierarchies

def get_qcnn(conv, pool, stride=1, step=1, offset=0, filter="right", wires=8, share_weights=True):
    panstz = Qunitary(function=pool, n_symbols=2, arity=2)
    qcnn = (Qinit(range(wires)) + 
            (Qcycle(
                stride=stride,
                step=step,
                offset=offset,
                mapping=Qunitary(conv, n_symbols=10, arity=2),
                share_weights=share_weights
            )
            + Qmask(filter, mapping=panstz)
        )
        * 3 # jnp.log2(n_wires) -> a entero
    )

    return qcnn


def get_circuit(qcnn, device, interface):
    dev = qml.device(device, wires=qcnn.tail.Q)

    #@jax.jit   # future?
    @qml.qnode(dev, interface=interface)
    def circuit(x, symbols):
        qcnn.set_symbols(symbols)

        qml.AmplitudeEmbedding(features=x, wires=qcnn.tail.Q, normalize=True)
        qcnn(backend="pennylane")  # This executes the compute graph in order
        
        M = [[0, 0], [0, 1]]
        return qml.expval(qml.Hermitian(M, wires = qcnn.head.Q[0]))
        return qml.probs(wires=qcnn.head.Q[0])
    return circuit

# qcnn = get_qcnn(g, poolg)

# # @jax.jit
# @qml.qnode(dev, interface=interface)
# def circuit(data, weights):
#     # data embedding
#     qml.AmplitudeEmbedding(features=data, wires=range(n_wires), normalize=True)

#     qcnn.set_symbols(weights)
#     qcnn(backend="pennylane")

#     # state_0 = [[1], [0]]
#     M = [[0, 0], [0, 1]]
#     return qml.expval(qml.Hermitian(M, wires = qcnn.head.Q[0]))
#     return qml.probs(wires=qcnn.head.Q[0])

if __name__ == "__main__":
    pass