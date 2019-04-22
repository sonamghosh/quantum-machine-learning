import numpy as np
from pyquil import Program, get_qc
from pyquil.gates import *
from forest_tools import *

"""
The quantum fourier transform and quantum phase estimation techniques provide a 
foundation for many quantum algorithms such as quantum matrix inversion, which is used
extensively in quantum machine learning.

the algorithms presented are coherent quantum protocols which mean that
the input and output of an algorithm is a quantum state that we do not have classical information
about. 
The protocol itself might use measurements but are not fully coherent but 
gain some incomplete classical information about the quantum system.

State prep and tomography are resource intensive, and are likely to destroy any quantum advantage

QFT and other quantum algorithms similar in complexity require a very large number of gates on a large number
of high quality qubits
"""

qc = get_qc('3q-qvm')

# Quantum Fourier Transform

# 3 qubit circuit 
qft = Program()
qft += H(0)
qft += CPHASE(np.pi/2, 1, 0)
qft += H(1)
qft += CPHASE(np.pi/4, 2, 0)
qft += CPHASE(np.pi/2, 2, 1)
qft += H(2)

#plot_circuit(qft)
#plt.show()
# conditional rotations dominate complexity as O(N^2)

# Quantum Phase Estimation

"""
Given a unitary operator U and a eigenvector |psi> of U, to estimate theta in 
U|psi> = exp(2i*pi*theta) |psi>

Since U is unitary, all of its eigenvalues have an absolute value of 1.
By conventionm theta is taken to be in [0, 1] and is called the phase of U associated to psi

the eigenvector |psi> is ecndoed in one set of quantum registers. An additional set of n qubits
form an ancilla register, At the end of the procedure, the ancilla register should have an approximation
of the binary fraction associated to theta, with n-bits precision. 

A critical element is the ability to perform the controlled unitary C - U^2^k

First the uniform superposition is prepared in the ancilla register via the Hadamard, These qubits
act as controls for the unitary operators at different time steps. Our goal is to create
a superposition of U as the unitary is applied for different durations. Since the eigenvalues
are always situated on the complex unit circle, these differently evolved components in the
superposition reveal the eigenstructure 

Given the ancilla register, there is a superposition of all possible time steps
between 0 and 2^(n-1), we end up with a superposition of all posisble evolutions
to encode binary representations of the eigenvalues. at the end of you have the following state

\begin{array}{l}{\frac{1}{2^{\frac{n}{2}}}( | 0\rangle+ e^{2 i \pi \theta \cdot 2^{n-1}} | 1 \rangle ) \otimes \cdots( | 0\rangle+ e^{2 i \pi \theta \cdot 2^{1}} | 1 \rangle ) \otimes( | 0\rangle+ e^{2 i \pi \theta \cdot 2^{0}} | 1 \rangle )=} \\ {\frac{1}{2^{\frac{n}{2}}} \sum_{k=0}^{2^{n}-1} e^{2 i \pi \theta k} | k \rangle}\end{array}


Exploiting controlled unitary operations introduces a global phase, which is seen in the ancilla (phase kickback)

In the final step a inverse fourier transform is put on the ancilla
"""

# Example 
# the following matrix [[exp(0), 0], [0, exp(i*pi)]] = [[1, 0], [0, -1]]
# with eigenvectors |0> and |1> with phases 0 and 1/2 

qpe = Program()
qpe += H(0)
qpe += H(1)

# Controlled unitary operations
# Controlled-U0
qpe += CZ(1, 2)
# controlled U-1
# nothing: identity

# Apply inverse QFT to write phase to ancilla register
qpe += SWAP(0, 1)
qpe += H(1)
qpe += CPHASE(-np.pi/2, 0, 1)
qpe += H(0)

result = qc.run_and_measure(qpe, trials=2000)
plot_histogram(result)
plt.show()

# As expected |2 * theta_0> = |2* 0> = |00>