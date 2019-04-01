from forest_tools import *
import matplotlib.pyplot as plt
from pyquil import Program, get_qc
from pyquil.gates import *
import numpy as np
from pyquil.api import WavefunctionSimulator


def get_amplitudes(circuit):
    wf_sim = WavefunctionSimulator()
    wavefunction = wf_sim.wavefunction(circuit)
    amplitudes = wavefunction.amplitudes

    return amplitudes


ancilla_qubit = 0
index_qubit = 1
data_qubit = 2
class_qubit = 3

training_set = [[0, 1], [np.sqrt(2)/2, np.sqrt(2)/2]]
labels = [0, 1]
test_set = [[1, 0]]

test_angles = [2*np.arccos(test_set[0][0])/2]
training_angle = (2*np.arccos(training_set[1][0]))/4

angles = [test_angles[0], training_angle]

circuit = Program()
# Create uniform superpositions of the ancilla and index qubits
circuit += H(ancilla_qubit)
circuit += H(index_qubit)

# Entangle the test instance with ground state of ancilla
circuit += CNOT(ancilla_qubit, data_qubit)
circuit += X(ancilla_qubit)

# Apply Identity to Class state
circuit += I(class_qubit)

print('Input = \n', get_amplitudes(circuit))

# Extend the circuit to prepare the first training
# instance and entanle it with the excited
# state of the ancilla and ground state of the index qubit.

circuit += CCNOT(ancilla_qubit, index_qubit, data_qubit)
circuit += X(index_qubit)

print('First training instance \n', get_amplitudes(circuit))


######

# Extend the circuit to prepare the second training instance
# and entangle it with the excited state of the ancilla and
# the excited of the index qubit.

circuit += CCNOT(ancilla_qubit, index_qubit, data_qubit)
circuit += CNOT(index_qubit, data_qubit)
circuit += H(data_qubit)

circuit += CCNOT(ancilla_qubit, index_qubit, data_qubit)
circuit += CNOT(index_qubit, data_qubit)
circuit += H(data_qubit)

circuit += CCNOT(ancilla_qubit, index_qubit, data_qubit)
circuit += CNOT(index_qubit, data_qubit)
circuit += H(data_qubit)

circuit += CCNOT(ancilla_qubit, index_qubit, data_qubit)
circuit += CNOT(index_qubit, data_qubit)
circuit += H(data_qubit)

print('Second training instance \n', get_amplitudes(circuit))

amplitudes = get_amplitudes(circuit)

target = np.array([ 0. +0.j,  0. +0.j,
                    0. +0.j,  0.5+0.j,
                    0.5+0.j,  0.5+0.j,
                    -0.5+0.j,  0. +0.j,
                    0. +0.j,  0. +0.j,
                    0. +0.j,  0. +0.j,
                    0. +0.j,  0. +0.j,
                    0. +0.j,  0. +0.j])


if np.allclose(amplitudes, target):
    print('Yes')
else:
    print('No')
