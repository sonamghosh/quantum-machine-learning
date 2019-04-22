import numpy as np
import scipy.linalg
from grove.alpha.phaseestimation.phase_estimation import controlled
from pyquil import Program, get_qc
from pyquil.gates import *
from forest_tools import *

qc = get_qc('6q-qvm')

"""
We will solve the equation Ax = b with A = 1/2 * [[3, 1], [1, 3]] and b = [1, 0]

The A matrix is encoded as an Hamiltonian and b in a register. 

With the ancillas, ther eis a need for a total of five qubits and one classical register for
post selection. 

An extra qubit and extra classical register is needed to create a swap test
to compare result to ideal state

"""

A = 0.5 * np.array([[3, 1], [1, 3]])
hhl = Program()

# vector b is encoded as |b> = sum_i=0 b_i |i> = |0>

# Quantum Phase Estimation 

"""
Next we encode the eigenvalues of A in an additional register 
this is done via quantum phase estimation of the evolution described by hamiltonian A
during some time t_0 , exp(i*At_0)

The protocol has three steps

prepare the ancilla state |psi_0> = sum_t=0 to T-1 |tau>

This state controls time evolution, like a clock it turns on evolution for a certain amount of time.

The OG HHL algo suggests using a weighted superposition of all states tau
that minimizes errors in the following steps, however , for this implementation
a uniform superposition already gives good results

The goal is to create a superposition of A as a hamiltonian applied for diff durations. 

Since the eigenvalues are always on the complex unit circle, these differently evolved
components in the superposition help reveal eigenstructure.

So we apply the conditional hamiltonian evolution sum_t=0 to T-1 |tau><tau| tensorprod exp(i*At_0/T)
on |psi_0> tensorprodu |b>. 

The operation evolves the state |b> according to Hamiltonian A for the time tau determined by 
the state |psi_0>

In |psi_0> we have a superposition of all possible time steps between 0 and T, 
so you have a superposition of all possible evolutions and a suitable choice of
number of timesteps T and total evolution time t_0 allow to encode binary representations of the eigenvalues. 

For the final step, you apply an inverse Fourier transform that writes the phases into new reg
"""

# In a 2 by 2 case, the circuit is simplified. Given that the matrix A has eigenvalues that are powers of 2
# We choose T = 4, t_0 = 2pi to obtain exact results with just two controlled evolutions

# Superposition
hhl += H(1)
hhl += H(2)

# Controlled-U0
hhl.defgate('CONTROLLED-U0', controlled(scipy.linalg.expm(2j*np.pi*A/4)))
hhl += ('CONTROLLED-U0', 2, 3)
# Controlled-U1
hhl.defgate('CONTROLLED-U1', controlled(scipy.linalg.expm(2j*np.pi*A/2)))
hhl += ('CONTROLLED-U1', 1, 3)

# Apply the Quantum Inverse fourier transform to write the phase to a register
hhl += SWAP(1, 2)
hhl += H(2)
hhl.defgate('CSdag', controlled(np.array([[1, 0], [0, -1j]])))
hhl += ('CSdag', 1, 2)
hhl += H(1)

hhl += SWAP(1, 2)
uncomputation = hhl.dagger()


# Conditional rotation of ancilla

# We need a conditional rotation to encode the information of thereciprocals of the
# eigenvalues in the amplitudes of a state 
# This is achieved by controled rotations in the same spirit of the hamiltonian evo

def rY(angle):
    """
    Generate a rotation matrix over the Y axis in the block sphere

    param angle: (float) The angle of rotation.

    return: (numpy.ndarray) The rotation matrix
    """
    return np.array([[np.cos(angle/2), -np.sin(angle/2)],
                     [np.sin(angle/2), np.cos(angle/2)]])


hhl.defgate('CRy0', controlled(rY(2*np.pi/2**4)))
hhl += ('CRy0', 1, 0)
hhl.defgate('CRy1', controlled(rY(np.pi/2**4)))
hhl += ('CRy1', 2, 0)


# Uncompute the eigenvalue register
# Need to uncompute all operations except those that store the information
# that we want to obtain from the algorithm in the final registers.
# We need to do this inc ase registers are entangled.

hhl += uncomputation

# Rejection sampling on the acnilla register and a swap test
# The state |x> = A^-1 |b> is prop to sum_J beta_j lambda_j^-1 |u_j>
# contains info about the solution to Ax = b when measuring 1 on the ancilla state. 

# We perform post selection by projecting onto the desired |1> 
# To check the solution is the expected one, we prepare the correct output state
# manually to perform a swap test with the outcome. 

from grove.alpha.arbitrary_state.arbitrary_state import create_arbitrary_state
reference_amplitudes = [0.9492929682, 0.3143928443]
# Target state prep
hhl += create_arbitrary_state(reference_amplitudes, [4])
# Swap test
hhl += H(5)
hhl += CSWAP(5, 4, 3)
hhl += H(5)
c = hhl.declare('ro', 'BIT', 2)
hhl += MEASURE(0, c[0])
hhl += MEASURE(5, c[1])

# it is a good ex to check if the right result is given by the state
# |x> = 0.949 |0> + 0.314 |1>

# There are tow measuresments performed, one of the ancilla register ( for doing post selection)
# and another one that gives the result of the swap test. To calculate success probabilities, define helper functions

def get_psuccess(counts):
    # Compute success probability of the HHL protocol from the stats

    try:
        succ_rotation_fail_swap = counts['11']
    except KeyError:
        succ_rotation_fail_swap = 0
    try:
        succ_rotation_succ_swap = counts['01']
    except KeyError:
        succ_rotation_succ_swap = 0
    succ_rotation = succ_rotation_succ_swap + succ_rotation_fail_swap
    try:
        prob_swap_test_success = succ_rotation_succ_swap / succ_rotation
    except ZeroDivisionError:
        prob_swap_test_success = 0
    return prob_swap_test_success

# Now run the ckt on simulator

hhl.wrap_in_numshots_loop(100)
executable = qc.compile(hhl)
result = qc.run(executable)
classical_bits = result.shape[1]
stats = {}
import itertools
for bits in itertools.product('01', repeat=classical_bits):
    stats["".join(str(bit) for bit in bits)] = 0
for i in range(100):
    stats["".join(str(bit) for bit in result[i])] += 1
print(get_psuccess(stats))

