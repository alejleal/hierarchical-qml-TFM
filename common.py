import pennylane as qml
from hierarqcal import Qunitary

from itertools import combinations

PATTERNS = ["*!", "!*", "!*!", "*!*", "01", "10"]

"""
*! = 00001111 (right?)
!* = 11110000 (left?)
!*! = 11000011 (inside?)
*!* = 00111100 (outside?)
01 = 01010101 (odd?)
10 = 10101010 (even?)
"""

GENRES = ["blues", "classical", "country", "disco", "jazz", "metal", "pop", "reggae", "rock"] # "hiphop" da problemas??
genre_combinations = combinations(GENRES, 2)

QUBITS = 8

# Estos dependen de los qubits de los que se disponga
STRIDES = [] 
OFFSETS = []
STEPS = []

# Create Qcnn
def penny_gate_to_function(gate):
    return lambda bits, symbols: gate(*symbols, wires=[*bits])

primitive_gates = ["CRZ", "CRX", "CRY", "RZ", "RX", "RY", "Hadamard", "CNOT", "PauliX"]
penny_gates = [getattr(qml, gate_name) for gate_name in primitive_gates]
hierq_gates = {
    primitive_gate: Qunitary(
        penny_gate_to_function(penny_gate),
        n_symbols=penny_gate.num_params,
        arity=penny_gate.num_wires,
    )
    for primitive_gate, penny_gate in zip(primitive_gates, penny_gates)
}
