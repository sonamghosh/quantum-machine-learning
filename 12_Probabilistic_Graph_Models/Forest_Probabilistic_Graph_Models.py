# Probabilistic graphical models ------------------

"""

Probabilistic graphical models give a compact representation
of a joint probability distribution [X_1, X_2,.., X_N] binary RV's
there are 2^N assignments

The factorization of the probabilities happens along
conditional indepences among rv's.
X is conditionally independent of Y given Z (X _|_ Y | Z)
P(X=x, Y=y|Z=z) = P(X=x|Z=z) P(Y=y|Z=z) for all x in X, y in Y, z in Z

The graph can be directed which in this case these are Bayesian networks
or undirected which in this case you have Markov networks.
Graphical models are pretty much generative.

In a markov random field, cycles are allowed in the graph and
switch from local normalization (conditional prob dist at each node) to
global normalization of probabilities (i.e. partition function)

The factorization is given as a sum
P(X_1,.., X_n) = 1/Z exp(-sum_k E[C_k]) where C_k are cliques of the graph
and E[.] is the energy defined over the cliquies
"""

# Define a markov field of binary fields
# This will be an Ising model over three nodes.
# There are three cliques of a single node (on site fields)
# and two cliques  of two nodes - the edges that connect the nodes

import matplotlib.pyplot as plt
import numpy as np
import dimod

n_spins = 3
h = {v: 1 for v in range(n_spins)}
J = {(0, 1):2,
     (1, 2): -1}

model = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.SPIN)
print(model.variables)
print(len(model.variables))
print(model.linear)
print(model.quadratic)
sampler = dimod.SimulatedAnnealingSampler()

# The prob distribution of the config does not explicityl define the temp
# But it is implicitly there in the constants defining the hamiltonian
# We can scale it a temp T = 1

# Now find the Probability P(E) of each energy level E
# It can be expressed as a sum over all states with energy E
# P(E) = sum_E(x_1,..,x-n) P(X1,...,X_N) = sum 1/Z  exp(-E/T)

# The term in the sum is constant (it doesnt depend on the RV's anymore)
# Just need to count the number of states such that
# E(X_1, ..., X_N) = E
# This is called the degeneracy of the energy level E, g(E)
# P(E) = 1/Z * g(E) * exp(-E/T)

temperature = 1
response = sampler.sample(model, beta_range=[1/temperature, 1/temperature], num_reads=100)

# Dict that associate to each energy E the degeneracy g(E)
g = {}
for solution in response.aggregate().data():
    if solution.energy in g.keys():
        g[solution.energy] += 1
    else:
        g[solution.energy] = 1

print('Degeneracy', g)
probabilities = np.array([g[E] * np.exp(-E/temperature) for E in g.keys()])
Z = probabilities.sum()
probabilities /= Z
fig, ax = plt.subplots()
ax.plot([E for E in g.keys()], probabilities, linewidth=3)
ax.set_xlim(min(g.keys()), max(g.keys()))
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Energy')
ax.set_ylabel('Probability')
#plt.show()

# In this case, the conditional probabilities are already encapsulated
# by the model
# Spins 0 and 2 do not interact
# In general it is hard to learn the strucutre of a probabilistic graph
# given a set of observed correlations in the sample S.
# We can only rely on heuristics
# The typical way is to define a socring function and do a global optimization


# Identify a graph G, learn the probabilities in the graph
# Rely on the sample and its correlations and use a max likelihood or
# maximum a posteriori estimate of the corresponding parameters
# t_G with likelihood P(S|t_G)

# Applying the learned model means probabilistic inference to answer
# queries of the following types
# Conditional prob: P(Y|E=e) = P(Y,e)/P(e)
# Max aposteriori: argmax_y P(y|e) = argmax_y sum_Z P(y, Z|e)


# In Deep learning, running a prediction is cheap after the network is traine
# Inference is computationally demanding even after training a model
# Instead of solving the inference problem directly, we use approximate inference
# with sampling, which is done with Monte carlo methods classically .

# Let's do a max a posteriori on the Ising Model
# clamp the first spin to -1 and run simulated annealing for the rest of them
# to find the optimal config

# simulated annealing routine is modified to account for clamping

from dimod.reference.samplers.simulated_annealing import greedy_coloring

clamped_spins = {0: -1}
num_sweeps = 1000
betas = [1.0 - 1.0*i / (num_sweeps - 1.) for i in range(num_sweeps)]

# Set up adjacency matrix
adj = {n: set() for n in h}
for n0, n1 in J:
    adj[n0].add(n1)
    adj[n1].add(n0)
# Use a vertex coloring for the graph and update the nodes by colour
__, colors = greedy_coloring(adj)

spins = {v: np.random.choice((-1, 1)) if v not in clamped_spins else clamped_spins[v]
         for v in h}

for beta in betas:
    energy_diff_h = {v: -2 * spins[v] * h[v] for v in h}
    # For each color, do updates
    for color in colors:
        nodes = colors[color]
        energy_diff_J = {}
        for v0 in nodes:
            ediff = 0
            for v1 in adj[v0]:
                if (v0, v1) in J:
                    ediff += spins[v0] * spins[v1] * J[(v0, v1)]
                if (v1, v0) in J:
                    ediff += spins[v0] * spins[v1] * J[(v1, v0)]

            energy_diff_J[v0] = -2. * ediff
        for v in filter(lambda x: x not in clamped_spins, nodes):
            logp = np.log(np.random.uniform(0, 1))
            if logp < -1. * beta * (energy_diff_h[v] + energy_diff_J[v]):
                spins[v] *= -1
print(spins)
