# Defining Circuits -------------------------------------------
"""
Circuits are composed of qubit registers with gates acting on them and measurements on the registers
To store the outcome, QC libraries typically add classical registers to the circuits
Even by this implementation it is a very low level of programming and is remniscent of assembly

Qubit registers are indexed from 0. Do not confuse it with the actual state of the qubit

Any bit string can be achieved with just two gates, which makes universal computations possible

Some gates (All are unitary)
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]])/sqrt(2)
Rx(t) = Rotation around X = np.array([[cos(t/2), -i*sin(t/2)], [-i*sin(t/2), cos(t/2)]])
Ry(t) = Rotation around Y = np.array([[cos(t/2), -sin(t/2)], [-sin(t/2), cos(t/2)]])
CNOT, CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

There are infinitely many single qubit operations 
"""

# Create |phi +> = 1/sqrt(2) * (|00> + |11>)

import numpy as np
from pyquil import Program, get_qc
from pyquil.api import WavefunctionSimulator
from pyquil.gates import *
from forest_tools import *
import matplotlib.pyplot as plt 

np.set_printoptions(precision=3, suppress=True)

# Setup two qubit registers and two classical registers
qc = get_qc('2q-qvm')
#wf_sim = WavefunctionSimulator()
circuit = Program()
circuit += H(0)
circuit += CNOT(0, 1)

# Errors - No pages of output.
# Transcript written on circuit.log.
# plot_circuit(circuit)


# All registers are initialized in the |0> state and creating a desired
# state is part of the ckt. Arbitrary state prep is the same as universal quantum computation

# Hadamards create equal super position in qubit 0
# This qubit controls an X gate on qubit 1
# Since qubit 0 is in equal superposition after the hadamard gate 
# it will not apply the X gate for the first part of the superposition |0> and it will apply
# the X gate for the second part of the superposition |1>. thus the
# final state is 1/sqrt(2) (|00> + |11>) and we entangle the two qubit registers

ro = circuit.declare('ro', 'BIT', 2)
circuit += MEASURE(0, ro[0])
circuit += MEASURE(1, ro[1])
circuit.wrap_in_numshots_loop(100)
executable = qc.compile(circuit)
result = qc.run(executable)
plot_histogram(result)