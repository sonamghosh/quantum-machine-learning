import numpy as np
from pyquil import Program, get_qc
from pyquil.gates import *
from forest_tools import *
import matplotlib.pyplot as plt


"""
Quantum States evolve through unitary matrices.
Unitary evolution is true for a closed system aka
if the quantum system is perfectly isolated from the environment 

QC's today are in open systems which evolve differently
due to uncontrolled interactions with the environment
"""

# Unitary Evolution ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A unitary matrix has the property that its conjugate tranpose is its inverse
# U is unitary if UU^dag = U^dagU.= I

# For ex. the not operation (X gate) One can study the properties of the X gate
# X = [[0, 1], [1, 0]]

X = np.array([[0, 1], [1, 0]])
print("XX^dagger")
print(X.dot(X.T.conj()))
print("X^daggerX")
print(X.T.conj().dot(X))

# It is indeed Unitary 
# l_2 norm is preserved under unitary conditions
print("The norm of the state |0> before applying X")
zero_ket = np.array([[1], [0]])
print(np.linalg.norm(zero_ket))
print("The norm of the state after applying X")
print(np.linalg.norm(X.dot(zero_ket)))

# Unitary matrices are linear and measurements are represented by matrices
# This implies all qc implements are linear
# We also need some nonlinearity which requires classical intervention
# Consequence of unitary operations is reversibility
# Reversing a X gate is simply, apply it again since the conjugate tranpose is itself
# X^2 = I 

"""
qc = get_qc('1q-qvm')
# Calling a ndarray not collable error
circuit = Program()
circuit += H(0)
circuit += X(0)
results = qc.run_and_measure(circuit, trials=100)
plot_histogram(results)
"""

# Interaction with the environment: open systems -----------------------
# Actual quantum syetsm are seldom closed
# It constantly interacts with the environment in an uncontrolled fashion
# Causing loss of coherence
# This means actual time evolution is not described by a unitary matrix as one would want it
# The technical name of this operator is a positive trace preserving map
# Quantum computing libraries offer a variety of noise models that mimic diff types of interaction
# increasing the strength of interaction with the environment leads to faster decoherence
# timescale for decoherence is often called T2

# A cheap way of studying effects of coherence is mixing a pure state with a 
# maximally mixed state I /2^d, where d is the num of qubits with some visibility param [0, 1]
# Mix the |phi + > state with a maximally mixed state

def mixed_state(pure_state, visibility):
    density_matrix = pure_state.dot(pure_state.T.conj())
    maximally_mixed_state = np.eye(4)/2**2
    return visibility * density_matrix + (1 - visibility)*maximally_mixed_state

phi = np.array([[1], [0], [0], [1]])/np.sqrt(2)
print("maximally visibility is a pure state: ")
print(mixed_state(phi, 1.0))
print("The state is still entangled with visibility 0.8:")
print(mixed_state(phi, 0.8))
print("Entanglement is lost by 0.6")
print(mixed_state(phi, 0.6))
print("Barely any coherence remains by 0.2")
print(mixed_state(phi, 0.2))

"""

Another way to look at is what happens to quantum state is through equilibrium processes

Cup of coffee left alone.
It will interact with the environment and slowly reaching the temp of the environment
Including energy exchange

A quantum state does the same thing and the environment has a defined temperature, just the
environment of a cup of coffee

Equilibrium state is called the thermal state, it has a very specific structure and will be revisited
but for now suffice to say the energy of the samples pulled out of a thermal state
follow a Boltzmann distribution
given as P(E_i) = exp(-E_i / T) / sum exp(-E_j / T)
where sum goes from j = 1 to M the number of energy levels
Higher the temp, the closer to a uniform dist
At high temp all energy levels have equal prob.
at zero temp the entire prob mass is concentrated on the lowest energy level, ground state
plot a boltzmann dist

"""

temperatures = [.5, 5, 2000]
energies = np.linspace(0, 20, 100)
fig, ax = plt.subplots()
for i, T in enumerate(temperatures):
    probabilities = np.exp(-energies/T)
    Z = probabilities.sum()
    probabilities /= Z
    ax.plot(energies, probabilities, linewidth=3, label="$T_" + str(i+1)+"$")
ax.set_xlim(0, 20)
ax.set_ylim(0, 1.2*probabilities.max())
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Energy')
ax.set_ylabel('Probability')
ax.legend()
plt.show()