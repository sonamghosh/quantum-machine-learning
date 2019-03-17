"""
References:

1] Verdon, G., Broughton, M., Biamonte, J. (2017) A quantum algorithm to train neural networks using low-depth circuits. arXiv:1712.05304.

"""

"""
We want to approximate the mixed density matrix rho0 which follows a boltzmann distribution

We can use QAOA or Quantum Annealing
"""

# Quantum Annealing

import itertools
import matplotlib.pyplot as plt 
import numpy as np 
import dimod 
np.set_printoptions(precision=3, suppress=True)

# Thermal State of classical ising model

# Create random model over ten spins and sample a hundred states

n_spins = 10
n_samples = 1000
h = {v: np.random.uniform(-2, 2) for v in range(n_spins)}
J = {}
for u, v in itertools.combinations(h, 2):
    if np.random.random() < 0.05:
        J[(u, v)] = np.random.uniform(-1, 1)
model = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.SPIN)
sampler = dimod.SimulatedAnnealingSampler()

# Sample energies at diff temperatures
# dimod imp of simulated annealing allows us to set an initial and final temp.
# If set to sample value, the effect of finite temperature happens and there 
# will be a wider range of configs and energy levels in the samples

temperature_0 = 1
response = sampler.sample(model, beta_range=[1/temperature_0, 1/temperature_0], num_reads=n_samples)
energies_0 = [solution.energy for solution in response.data()]

temperature_1 = 10
response = sampler.sample(model, beta_range=[1/temperature_1, 1/temperature_1], num_reads=n_samples)
energies_1 = [solution.energy for solution in response.data()]

temperature_2 = 100
response = sampler.sample(model, beta_range=[1/temperature_2, 1/temperature_2], num_reads=n_samples)
energies_2 = [solution.energy for solution in response.data()]

# Plot distribution
def plot_probabilities(energy_samples, temperatures):
    fig, ax = plt.subplots()
    for i, (energies, T) in enumerate(zip(energy_samples, temperatures)):
        probabilities = np.exp(-np.array(sorted(energies))/T)
        Z = probabilities.sum()
        probabilities /= Z
        ax.plot(energies, probabilities, linewidth=3, label="$T_" + str(i+1) + "$")
    minimum_energy = min([min(energies) for energies in energy_samples])
    maximum_energy = max([max(energies) for energies in energy_samples])
    ax.set_xlim(minimum_energy, maximum_energy)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Energy')
    ax.set_ylabel('Probability')
    ax.legend()
    plt.show()

plot_probabilities([energies_0, energies_1, energies_2],
                   [temperature_0, temperature_1, temperature_2])

# Quantum Approximate Thermalization

import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import fmin_bfgs
from pyquil import get_qc, Program, api
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.gates import *
from grove.pyqaoa.qaoa import QAOA
from forest_tools import *
#qvm_server, quilc_server, fc = init_qvm_and_quilc('/home/local/bin/qvm', '/home/local/bin/quilc')
#qvm = api.QVMConnection(endpoint=fc.sync_endpoint, compiler_endpoint=fc.compiler_endpoint)
qvm = api.SyncConnection()

# High Temperature

# Create an example system of two qubits that needs two extra qubits for purification, set T=1000

n_qubits = 2
n_system = n_qubits * 2
T = 1000
p = 1  # QAOA parameter

# Define an Ising Hamiltonian Hc = -sum Jij sig_i sig_j
# Use the QAOA implementation found in Grove
# It takes params as a list of PauliSum (each paulisum is one term in the Hamiltonain)

# Weight Matrix of Ising Model , only (0, 1) coefficient is non-zero
J = np.array([[0, 1], [0, 0]])

Hc = []
for i in range(n_qubits):
    for j in range(n_qubits):
        Hc.append(PauliSum([PauliTerm("Z", i, -J[i, j]) * PauliTerm("Z", j, 1.0)]))

Hm = [PauliSum([PauliTerm("X", i, 1.0)]) for i in range(n_qubits)]

# Prepare the initial state
# |psi> = sqrt(2*cosh(1/T)) * sum_(z in -1, 1) exp(-z/T) |z>_H1 tensorprod |z>_H2

def prepare_init_state(T):
    init_state = Program()
    alpha = 2 * np.arctan(np.exp(-1/2*T))
    for i in range(n_qubits):
        init_state += RX(alpha, n_qubits+i)
        init_state += CNOT(n_qubits+i, i)

    return init_state

# QAOA Grove
def get_optimized_circuit(init_state):
    qaoa = QAOA(qvm,
                qubits=range(n_system),
                steps=p,
                ref_ham=Hm,
                cost_ham=Hc,
                driver_ref=init_state,
                store_basis=True,
                minimizer=fmin_bfgs,
                minimizer_kwargs={'maxiter': 50})
    beta, gamma = qaoa.get_angles()
    return qaoa.get_parameterized_program()(np.hstack((beta, gamma)))


def get_thermal_state(T):
    return get_optimized_circuit(prepare_init_state(T))

thermal_state = get_thermal_state(T)

# Reformat final results, measure out, and plot
def get_energy(x):
    return np.sum([[-J[i,j] * x[i] * x[j] for j in range(n_qubits)] for i in range(n_qubits)])

def get_energy_distribution(thermal_state):
    measures = np.array(qvm.run_and_measure(thermal_state, range(n_qubits), trials=1000))

    measures[measures == 0] = -1
    list_energy = np.array([get_energy(m) for m in measures])
    return list_energy


energy_distribution = get_energy_distribution(thermal_state)
hist = plt.hist(energy_distribution, density=True)
plt.show()

# The two eigenvalues (possible energies) of the hamiltonian H = sig1sig2
# are E = -1 and E = 1, at infinite temp, they should be assigned an equal prob.
# Which is the case in the histogram above o.0

T = 0.5
thermal_state = get_thermal_state(T)
energy_distribution = get_energy_distribution(thermal_state)
hist = plt.hist(energy_distribution, density=True)
plt.show()
