"""
Quantum Annealing

Quantum annealing inspired gate-model algorithms that work on current and near-term
quantum computers. 

D-Wave offers a open source suite called Ocean. 
Vendor-independent solution is XACC, an extensible compilation framework
for hybrid quantum-classical computing architectures, the quantum annealer
it maps to is that of the D-Wave system

"""

# Unitary Evolution and the Hamiltonian -------------------------------

"""
Schrodinger Equation
i*hbar* d(|psi(t)>)/dt = H |psi(t)>

Time ind. soln U = exp(-iHT/hbar)

"""

# Adiabatic Theoream and adiabatic quantum computing
# Adiabatic process means that conditions change slowly enough for the system to
# adapt to the new configuration. 
# Start with some Hamiltonian H0 and change it slowly to some hamiltonian H1
# Linear sched: H(t) = (1-t)H0 + tH1
# t in [0, 1]
# Since the hamiltonian is time independent, the schrodinger eqn becomes hard to solve
# The adiabatic thrm says that if the change in the time-independent Hamiltonian occurs slowly
# the resulting dynamics remain simple, starting close to an eigenclose, the system remains close to an eigenstate
# This implies if the system started in the ground state, if certain conditions are met, the system stays in the ground state
# The energy diff between ground state and first excited state the gap
# If H(t) has a non negative gap for each t, the change happens slowly.
# Time depenedent gap by delta(t), a course approximation of the speed limit scales as 1 / min(delta(t))^2

# The theorem allows reaching the ground state of an easy to solve quantum many body system, and change
# the hamiltonian to a system we are interested in.
# one can start with an hamiltonian sum_i sigma_i^X , its ground ground state is just the equal superposition

import numpy as np 
np.set_printoptions(precision=3, suppress=True)

X = np.array([[0, 1], [1, 0]])
IX = np.kron(np.eye(2), X)
XI = np.kron(X, np.eye(2))
H_0 = - (IX + XI)
lamda , v = np.linalg.eigh(H_0)
print("Eigenvalues: ", lamda)
print("Eigenstate for lowest eigenvalue", v[:, 0])

# We could turn this Hamiltonian slowly into a classical Ising model and read out the Global soln
# adiabatic quantum computation exploits this phenomenon and is able to perform universal calculations
# with H = - sum_<i,j> Jij sigma_i^Z sigma_j^Z - sum_i h_i sigma_i^Z - sum_<i,j> gij sigma_i^X sigma_j^X
# Note that this is not the traverse field ising model since the last term is a X-X interaction
# If a quantum comp respects this speed limit, guratnees the finite gap, and implements this Hamiltonian
# it is equivalent to the gate model with some overhead. 

# Quantum Annealing -----------------------------------
# Theoretical obstacle to AQC is that calculating the speed limit is not trivial.
# It is harder than solving the original problem of finding the ground state of some Hamiltonian of interest.
# Engineering constraints apply: qubits decohere, the environment has finite temperature, and so on
# QA drops the strict requirements and instead of respecting speed limits, it repeats the annealing
# over and over
# After a number of samples, we pick the spin config with lowest energy as our soln
# There is no guarantee this is the ground state

# Theres a diff software stack versus gate model
# Instead of a quantum circuit, the level of abstraction is the classical ising model
# Superconducting annealers suffer from limited connectivity like superconducting qubits 
# We have to find a graph minor embedding , this will combine several physical qubits into a logical qubit


import dimod

J = {(0, 1): 1.0, (1, 2): -1.0}
h = {0:0 , 1:0, 2:0}
model = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.SPIN)
sampler = dimod.SimulatedAnnealingSampler()
response = sampler.sample(model, num_reads=10)
print("Energy of samples: ")
print([solution.energy for solution in response.data()])

# Look at a minor embedding problem. It's still NP-hard
# Usually need probablistic heuristics to find an embedding
# D-Wave has unit cells containing K4,4 bipartite fully connected graph
# with two remote connections from each qubit going to qubits in neighbouring unit cells. A unit
# cell is with its local and remote connections including indicated

# Chimera graph, largest hardware has 2048 qubits consisting of 16 by 16 unit cells of 8 qubits
# Chimera graph is available as a networkx graph in the package dwave_networkx
# draw a smaller one of 2 by 2 cells

import matplotlib.pyplot as plt
import dwave_networkx as dnx

connectivity_structure = dnx.chimera_graph(2, 2)
dnx.draw_chimera(connectivity_structure)
plt.show()

# Now create a graph that does not fit the connectivity structure. The complete Kn on nine nodes

import networkx as nx
G = nx.complete_graph(9)
plt.axis('off')
nx.draw_networkx(G, with_labels=False)

import minorminer
embedded_graph = minorminer.find_embedding(G.edges(), connectivity_structure.edges())

dnx.draw_chimera_embedding(connectivity_structure, embedded_graph)
plt.show()

# Qubits that have the same colour corresponding to a logical in the original problem defined by the K9 graph
# Qubits combined in such a way form a chain. Even though the problem has 9 variables(nodes), all 32 available
# were almost used by the toy Chimera graph, find the max chain length

max_chain_length = 0
for _, chain in embedded_graph.items():
    if len(chain) > max_chain_length:
        max_chain_length = len(chain)
print(max_chain_length)  # prints 4

# chain on the hardware is implemented by having strong couplings between
# the elements in a chain -- in fact, twice as strong as what the user can set.
# Long chains can break which gives inconsistent results. 
# in general, we prefer shorter trains so theres no wasting physical qubits
