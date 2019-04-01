import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
import itertools
import time

# Mapping clustering to discrete optimization

# Create a simple dataset to create two distinct classes
# First five belong to class 1 , second five to class 2

n_instances = 10
class_1 = np.random.rand(n_instances//2, 3)/5
class_2 = (0.6, 0.1, 0.05) + np.random.rand(n_instances//2, 3)/5
data = np.concatenate((class_1, class_2))
colors = ["red"] * (n_instances//2) + ["green"] * (n_instances//2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', xticks=[], yticks=[], zticks=[])
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors)

#plt.show()

# Calculate the euclidean distance (pairwise) between data points

w = np.zeros((n_instances, n_instances))
for i, j in itertools.product(*[range(n_instances)]*2):
    w[i, j] = np.linalg.norm(data[i] - data[j])

# W is known as the Gram or kernel matrix
# One can think of the Gram matrix as the weighted adjacency matrix of a graph: two nodes represent two data instances
# The distance contained in the Gram matrix is the weight on the edge that connects them. 
# If the distance is 0 , then they are not connected by an edge


# We no wlook at the max-cut, a collection of edges that would split a graph in exactly two if removed
# while maximizing the total weight of these edges. This is NP- hard but can map naturally to an Ising Model

# Pauli spins sig_1 in {-1, 1} take on value +1 if in cluster 1 (nodes V1 in the graph)
# -1 if in cluster 2 (nodes V2 in the grpah)

# cost of cut = sum over i in V1 and V2  w_ij

# given a fully connected graph , and accounting for symmetry of the adjacency matrix
# = 1/4 * sum_i,j w_ij - 1/4 * sum_ij w_ij * sig_i * sig_j

# Using the negative of this, one can solve the problem with a quantum optimizer

# Solving the max-cut problem by QAOA --------------------------
# Grove has a max-cut implementation for binary weights

# max-cut Hamiltonian can be seen as particular Ising model

start = time.time()

from pyquil import Program, api
from pyquil.paulis import PauliSum, PauliTerm
from scipy.optimize import fmin_bfgs
from grove.pyqaoa.qaoa import QAOA
from forest_tools import *

qvm = api.SyncConnection()

# Set p = 1 in QAOA , we can init it w/ max cut problem
maxcut_model = []
for i in range(n_instances):
    for j in range(i+1, n_instances):
        maxcut_model.append(PauliSum([PauliTerm("Z", i, 1/4 * w[i, j]) * PauliTerm("Z", j, 1.0)]))
        maxcut_model.append(PauliSum([PauliTerm("I", i, -1/4)]))

p = 1
Hm = [PauliSum([PauliTerm("X", i, 1.0)]) for i in range(n_instances)]
qaoa = QAOA(qvm,
            qubits=range(n_instances),
            steps=p,
            ref_ham=Hm,
            cost_ham=maxcut_model,
            store_basis=True,
            minimizer=fmin_bfgs,
            minimizer_kwargs={'maxiter': 50})

nu, gamma = qaoa.get_angles()
program = qaoa.get_parameterized_program()(np.hstack((nu, gamma)))
measures = qvm.run_and_measure(program, range(n_instances), trials=100)
measures = np.array(measures)

# Extract common soln
count = np.unique(measures, return_counts=True, axis=0)
weights = count[0][np.argmax(count[1])]

print(weights)

end = time.time()

print('Time elapsed for solving maxcut using QAOA', end - start, 's')
# Solve it using annealing


start = time.time()
import dimod

J, h = {}, {}
for i in range(n_instances):
    h[i] = 0
    for j in range(i+1, n_instances):
        J[(i, j)] = w[i, j]


model = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.SPIN)
sampler = dimod.SimulatedAnnealingSampler()
response = sampler.sample(model, num_reads=10)
end = time.time()
print("Energy of samples:")
for solution in response.data():
    print("Energy:", solution.energy, "Sample", solution.sample)

print("Time elapsed for solving maxcut using Annealing", end - start, 's')