import itertools
import numpy as np
import dimod



"""
ISING Model

Imagine having two magnets on the same axis (N -- S) (S -- N)

They will naturally anti-align, create two variables sigma_1 , sigma_2

If north, the value be +1, else -1

These variables can be called spin.

In the optimal config, the product is -1:
    sigma_1 * sigma_2 = -1

There are two physical configs , sigma_1 = +1 , sigma_2 = -1 and sigma_1 = -1 , sigma_2 = +1

If you add more magnets to the system, the pairwise interactions can be added up to get the total energy

If you have N magnets arranged in a straight line, the hamiltonian is given as

H = sum_{i=1}^{N-1} sigma_i * sigma_(i+1)

The above is a simplification where it is assumed the remote magnets dont interact with each other.

The interactions depends on the layout of the spins and assumptions of the physical model. 
Rewrite the hamiltonian as some graph

H = sum_<i,j> sigma_i sigma_j 

Now assume the distance is not the same between each pair, so some pairs interact more than others.
Can express a interaction strength param

H = - sum_<i,j> J_ij sigma_i sigma_j

where J_ij is a real num, the negative sign is by convention. I.e. If the spins are antiferromagnetic, then all Jij vals are negative

The model is fairly complicated. Imagine having a lot of spins but not all behave like magnet therefore
J_ij can be neg or pos. 

Nature wants to find the lowerst energy configuration. 
"""

# Calculate energy of spins on a line, given some couplings and spin config



def calculate_energy(J, sigma):
    return sum(J_ij*sigma[i]*sigma[i+1] for i, J_ij in enumerate(J))


# Ex . set of configs with a 3 site spin config
J = [1.0, -1.0]
sigma = [1, -1, 1]

# Energy
print(calculate_energy(J, sigma))

# We want the minimum but cant use a gradient based method to find since the variables are binary
# Optimization landscape is nonconvex

# Have to choose an exhaustive search of all possibilities

for sigma in itertools.product(*[{+1, -1} for _ in range(3)]):
    print(calculate_energy(J, sigma), sigma)

# -2 is the optimum with two optimal config.
# There are more clever ways to find the best solnm but in the general case this is not the case

# To get to the general case you need an external field (i.e. add a large magnet below each magnet)
# if the field is strong enough, it overrides the pairwise interaction and flip the magnets.
#  H = - sum_<i,j> J_ij sigma_i sigma_j - sum_i h_i sigma_i
# where h_i is the strength of the external field this is also knwon as the classical Ising model
# In physics, Hamitonian describes the energy but in CS it is the objective function to minimize

# these problems are known as the quadratic unconstrained binary optimization (QUBO) where the values take values 0, 1
# QUBOs are NP-hard in general so the generic strat is to use an exhaustive search which grows exponentially
# Imagine the energy diff between the gs and next lowest energy state (first excited state)
# If one starts from a random config, one could get stuck in a local optimum.

# To do this, we do a heuristic algorithm called simulated annealing
# This defines a temperature to be able to hop out of a local minimum
# The temp is lowered over time to find the actual minimum
# There are many implementations, we can use one in the dimod package

# Define couplings as a dict between spins
J = {(0, 1): 1.0, (1, 2): -1.0}
h = {0:0, 1:0, 2:0}

# Instatiate
model = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.SPIN)
# Create a simulated annealing sampler that pulls out potentially optimal solns and
# read out 10 possible solns
sampler = dimod.SimulatedAnnealingSampler()
response = sampler.sample(model, num_reads=10)

# Can see this config is easy, since -2 is the optimal soln most of the time
print([solution.energy for solution in response.data()].count(-2))

# Traverse-field Ising Model -------------------------------------------------------
# We can write the same Hamiltonian above in Quantum Mechanical form

from pyquil import Program, get_qc
from pyquil.api import WavefunctionSimulator
from pyquil.gates import *
from forest_tools import *
np.set_printoptions(precision=3, suppress=True)

wf_sim = WavefunctionSimulator()
circuit = Program()
# Pauli- Z matrix replicates the effect
# Use it on the elements of a computational basis
circuit += Z(0)
wavefunction = wf_sim.wavefunction(circuit)
print(wavefunction.amplitudes)  # prints |0> state

# It does not doing anything but it can be thought of as multiplying by +1 
# Try on |1> state instead 

circuit = Program()
circuit += X(0)
circuit += Z(0)
wavefunction = wf_sim.wavefunction(circuit)
print(wavefunction.amplitudes)   # prints - |1> , adds a minus sign, so now we get +1, -1 vals
# we write sigma_i^Z as the operator Z at site i 
# Hamiltonian thus becomes
# H = - sum_<i,j> J_ij sigma_i^Z sigma_j^Z - sum_i h_i sigma_i^Z
# The exp value <H> of the H is the energy of the system and the corresponding |psi> quantum state
# is the configuration of that energy level.
# Create the quantum mechanical version of calculating the energy, matching the function
# defined above for the classical mechanical variant

def calculate_energy_expectation(state, hamiltonian):
    return float(np.dot(state.T.conj(), np.dot(hamiltonian, state)).real)

# Tricky to define Hamiltonain with sigma_i^Z operators, bc acting on site i means
# it acts trivially on all other states
# So for two sites, we act on one site, the actual operator is sigma^Z \otimes I 
# and acting on site two, we have I \otimes sigma^Z 
# The above function to calculate the energy takes numpy arrays 
# We manually define sigma^Z and calculate  the energy of the Hamiltonian
# H = - sigma_1^Z sigma_2^Z - 0.5 * (sigma_1^Z + sigma_2^Z) on the state |00>

pauli_z = np.array([[1, 0], [0, -1]])
IZ = np.kron(np.eye(2), pauli_z)
ZI = np.kron(pauli_z, np.eye(2))
ZZ = np.kron(pauli_z, pauli_z)
H = -ZZ + -0.5*(ZI+IZ)
psi = np.kron([[1], [0]], [[1], [0]])
print(calculate_energy_expectation(psi, H))

# the hamiltonian commutes, so all of its operators are commutative, which is a sign of
# nothing much quantum going on 

# We need to add an non commutative term to make it quantum
# A transverse field is such, an on-site interaction like the external field 
# Its effect is described by the pauli-X operator, which is denoted as sigma_i^X for a site i

circuit = Program()
circuit += X(0)
circuit += Z(0)
wavefunction = wf_sim.wavefunction(circuit)
state = wavefunction.amplitudes
print("Pauli-X, then Pauli-Z: ", state)
circuit = Program()
circuit += Z(0)
circuit += X(0)
wavefunction = wf_sim.wavefunction(circuit)
state = wavefunction.amplitudes
print("Pauli-Z, then Pauli-X: ", state)

# There is a clear difference in sign
# There are other ways of making the Ising Hamiltonian noncommuting, but adding the onsite Pauli-X
# operations leads back to the transverse ising model
# H = - sum_<i,j> J_ij sigma_i^Z sigma_j^Z - sum_i h_i sigma_i^Z - sum_i g_i sigma_i^X
