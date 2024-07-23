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
    # qml.Barrier([bits[0], bits[1]])

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

def genrot(bits, symbols=None):
    qml.RZ(symbols[0], wires=bits[0])
    qml.RY(symbols[1], wires=bits[0])
    qml.RZ(symbols[2], wires=bits[0])

def universal(bits, symbols=None):
    genrot([bits[0]], symbols[:3])
    genrot([bits[1]], symbols[3:6])

    qml.CNOT(wires=[bits[1], bits[0]])

    qml.RZ(symbols[6], wires=bits[0])
    qml.RY(symbols[7], wires=bits[1])

    qml.CNOT(wires=[bits[0], bits[1]])

    qml.RY(symbols[8], wires=bits[1])

    qml.CNOT(wires=[bits[1], bits[0]])

    genrot([bits[0]], symbols[9:12])
    genrot([bits[1]], symbols[12:])


# Pooling ansatz
def poolg(bits, symbols): 
    qml.CRZ(symbols[0], wires=[bits[0], bits[1]])
    qml.PauliX(wires=bits[0])
    qml.CRX(symbols[1], wires=[bits[0], bits[1]])

def cnot(bits, symbols):
    qml.CNOT(wires=[bits[0], bits[1]])

n_params = {
    'a': 2,
    'b': 2,
    'g': 10,
    'universal': 15,
    'poolg': 2,
    'cnot': 0
}


def num_layers(qubits):
    # Returns the number of layers assuming each layer halves its available qubits
    return (qubits - 1).bit_length()

def get_num_params(conv, pool, qubits):
    layers = num_layers(qubits)

    pool_n_symbols = n_params[pool.__name__]
    conv_n_symbols = n_params[conv.__name__]

    return layers * (pool_n_symbols + conv_n_symbols)


## Hierarchies
def get_qcnn(conv, pool, stride=1, step=1, offset=0, filter="01", wires=8, share_weights=True):
    # Por ahora todas los ansatzs son de aridad 2 asi que se queda asi de momento
    pool_n_symbols = n_params[pool.__name__]
    pool_map = Qunitary(function=pool, n_symbols=pool_n_symbols, arity=2)

    conv_n_symbols = n_params[conv.__name__]
    qcnn = (Qinit(range(wires)) + 
            (Qcycle(
                stride=stride,
                step=step,
                offset=offset,
                mapping=Qunitary(conv, n_symbols=conv_n_symbols, arity=2),
                share_weights=share_weights
            )
            + Qmask(filter, mapping=pool_map)
        )
        * (wires - 1).bit_length() # Consigue el ceil(log2(wires)) para el numero de capas
    )

    return qcnn


def get_circuit(qcnn, device, interface):
    dev = qml.device(device, wires=qcnn.tail.Q)

    #@jax.jit   # future?
    @qml.qnode(dev, interface=interface)
    def circuit(x, symbols):
        qcnn.set_symbols(symbols)

        qml.AmplitudeEmbedding(features=x, wires=qcnn.tail.Q, normalize=True, pad_with=0)
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
    device = "default.qubit.torch"
    interface = 'torch'
    wires = 4
    qcnn = get_qcnn(a, cnot, wires=wires)
    circuit = get_circuit(qcnn, device, interface)


    drawer = qml.drawer.MPLDrawer(n_wires=4, n_layers=8)

    drawer.label(range(4))

    embed_box_options = {'facecolor': 'green', 'edgecolor': 'darkgreen', 'linewidth': 3}
    conv_box_options = {'facecolor': 'lightcoral', 'edgecolor': 'maroon', 'linewidth': 3}
    pool_options = {'linewidth': 3, 'color': 'teal'}
    text_options = {'fontsize': 'xx-large', 'color': 'white'}

    # drawer.box_gate(layer=0, wires=[0, 1, 2, 3], text=r"$|\Psi\rangle$", box_options=embed_box_options)
    drawer.box_gate(layer=0, wires=[0, 1], text=r"$U(\theta)$", box_options=conv_box_options, text_options=text_options)
    drawer.box_gate(layer=1, wires=[1, 2], text=r"$U(\theta)$", box_options=conv_box_options, text_options=text_options)
    drawer.box_gate(layer=2, wires=[2, 3], text=r"$U(\theta)$", box_options=conv_box_options, text_options=text_options)
    drawer.box_gate(layer=3, wires=[0, 3], text=r"$U(\theta)$", box_options=conv_box_options, text_options=text_options)

    # drawer.Barrier()

    drawer.CNOT(layer=4, wires=(2, 0), options=pool_options)
    drawer.CNOT(layer=5, wires=(3, 1), options=pool_options)

    drawer.box_gate(layer=6, wires=[0, 1], text=r"$U(\theta)$", box_options=conv_box_options, text_options=text_options)

    drawer.CNOT(layer=7, wires=(1, 0), options=pool_options)

    # drawer.measure(layer=7, wires=0)

    # drawer.ctrl(layer=3, wires=[1, 3], control_values=[True, False])
    # drawer.box_gate(
    #     layer=3, wires=2, text="H", box_options={"zorder": 4}, text_options={"zorder": 5}
    # )

    # drawer.ctrl(layer=4, wires=[1, 2])

    # drawer.fig.suptitle('My Circuit', fontsize='xx-large')

    drawer.fig.savefig("./images/circuit_test.png")


    # fig, ax = qml.draw_mpl(circuit)(range(2**wires), range(36))
    # fig.savefig("./images/circuit_test.png")
    # # fig.show()