import numpy as np 
import matplotlib.pyplot as plt 
from pyquil import Program, get_qc
from pyquil.gates import *
from forest_tools import *
from pyquil.api import WavefunctionSimulator
 


# Classical Probability Distributions ---------------------------------------------

"""
Toss a biased coin. Let X be a random variable with the output
of 0 for heads and 1 for tails. 
P(X=0) = p0 and P(X = 1) = p1 for each toss of the coin
In classical Kolmogorovian probability theory p_i >= 0 for all i and sum of p's = 1
"""

# Sample distribution
n_samples = 100
p_1 = 0.2
x_data = np.random.binomial(1, p_1, (n_samples))
print(x_data)

# Check that the observed frequencies sum to one
frequency_of_zeros, frequency_of_ones = 0, 0
for x in x_data:
    if x:
        frequency_of_ones += 1/n_samples
    else:
        frequency_of_zeros += 1/n_samples
print('frequencies: ', frequency_of_zeros + frequency_of_ones)

# Given that p0 and p1 must be non-negative, all possible prob distributions msut be in Quadrant I
# Normalization constant puts all possible distributions on a straight line.
# Plot all possible probability distributions by biased and unbiased coins
p_0 = np.linspace(0, 1, 100)
p_1 = 1 - p_0
fig, ax = plt.subplots()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_xlabel("$p_0$")
ax.xaxis.set_label_coords(1.0, 0.5)
ax.set_ylabel("$p_1$")
ax.yaxis.set_label_coords(0.5, 1.0)
plt.plot(p_0, p_1)
# Uncomment to show plot from terminal
#plt.show()

# Arrange probabilities as a vector p = (p0, p1)^T
# Norm constraint allows for the norm of the vector to be 1 in l1-norm 
# sum of the p's = 1

p = np.array([[0.8], [0.2]])
print(np.linalg.norm(p, ord=1))  # prints 1

# The probability of heads is the first element of p
# Since it is a vector one can extract it by projecting the vector to the first axis
# Projection is described by [[1, 0],[1,0]] 
# Length in the l1-norm gives the probability
q_0 = np.array([[1, 0], [0, 0]])
print('Probability of heads: ', np.linalg.norm(q_0.dot(p), ord=1))  # prints 0.8
# Prob of tails
q_1 = np.array([[0, 0], [0, 1]])
print('Probability of tails: ', np.linalg.norm(q_1.dot(p), ord=1))  # prints 0.2

# The two projects take an equivalent role ot the values of 0 and 1
# Can define a new random variable Q , that cakes projections q0 and q1 as values
# and end up with a identical distribution

# Transform a prob distribution to another one
# E.g. change bias of a coin or describe transition of a markov chain
# A left stochastic matrix maps stochastic vectors to stochastic vectors when multiplied from the left
# e.g. unbiased coin, the map M will transform the distribution to a biased coin

p = np.array([[.5], [.5]])
M = np.array([[0.7, 0.6], [0.3, 0.4]])
print('Mapping: ', np.linalg.norm(M.dot(p), ord=1))  # prints 0.99999 ~ 1

# Entropy 
# Prob distributions entropy is given as H(p) = - sum_i p_i * log_2 p_i

eps = 10e-10
p_0 = np.linspace(eps, 1 - eps, 100)
p_1 = 1 - p_0
H =  -(p_0*np.log2(p_0) + p_1*np.log2(p_1))
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, -np.log2(0.5))
ax.set_xlabel("$p_0$")
ax.set_ylabel("$H$")
plt.plot(p_0, H)
plt.axvline(x=0.5, color='k', linestyle='--')
# Uncomment to show plot
#plt.show()  # Entropy is maximal for the unbiased coin
# Entropy peaks for the uniform distribution
# Entropy quantifies a notion of surprise and unpredictability


# Quantum States -----------------------------------------------------

"""
Classical coin is a two level system
Can write prob distribution as a column vector so we will write the quantum state as such
(Note, below are column vectors, I removed the tranpose for brevity)
|psi> = [a0, a1]
sqrt(|a0|^2 + |a1|^2) = 1

Expand a arbitrary qubit state in the following basis
|psi> = [a0 , a1] = a0 [1, 0] + a1 [0, 1] = a0 |0> + a1 |1>
"""

# Create a biased case where the heads is gotten with prob 1
# qubit |psi> = |0>
# Create a single qubit circle and single classical register where the sampling(measurements) go

circuit = Program()   # Any qubit is initialized in |0>
# Measure
ro = circuit.declare('ro', 'BIT', 1)
circuit += MEASURE(0, ro[0])
# Execute hundred times
qc = get_qc('1q-qvm')
circuit.wrap_in_numshots_loop(100)
executable = qc.compile(circuit)
result = qc.run(executable)
print('result = ', result.T)  # prints all 0 

# To understand possible quantum states, we need bloch sphere visualization
# Since there are two probability amplitudes for a single qubit, you need a 4-dim space
# Since vectors are normalized, a degree of freedom is removed giving a 3-dim rep.
# North pole with state |0>
# South pole with state |1>
# Two orthogonal vectors appear the same if they were on the same axis -- Z
# Computational basis is one basis, X and Y rep two other bases
# Any pt on the sphere is a valid quantum state
# Every pure quantum state is a point on the bloch sphere

# Plot |0> on the bloch sphere
wf_sim = WavefunctionSimulator()
wavefunction = wf_sim.wavefunction(circuit)
plot_quantum_state(wavefunction.amplitudes)

# Transform |0> to 1/sqrt(2) * (|0> + |1>)
# This corresponds to a unbiased coin due to equal prob of getting 0 and 1
# Pick a rotation around the Y - axis by pi/2
# Ry = 1/sqrt(2) * [[1, -1], [1, 1]]

circuit = Program()
circuit += RY(np.pi/2, 0)
results = qc.run_and_measure(circuit, trials=100)
plot_histogram(results)

# For an intuition for why it's a rotation around Y , plot on Bloch sphere
wavefunction = wf_sim.wavefunction(circuit)
plot_quantum_state(wavefunction.amplitudes)  # rotates from north pole of Bloch sphere

# Apply same rotation to |1>
# Flip |0> to |1> and applying a NOT gate (denoted as X) and then rotate
circuit = Program()
circuit += X(0)
circuit += RY(np.pi/2, 0)
wavefunction = wf_sim.wavefunction(circuit)
plot_quantum_state(wavefunction.amplitudes)
# One can verify the result is 1/sqrt(2) * (-|0> + |1>) 
# Note the diff cant be observed from statistics
results = qc.run_and_measure(circuit, trials=100)
plot_histogram(results)

# Looks like an approx biased coin, The negative sign or any complex value
# is what models interference, a phenomenon where probability amplitudes interact in
# a constructive or a destructive way. 
# Apply rotation twice on a row on |0> to get a determinstic output of |1>
# Between the two there was superposition
circuit = Program()
circuit += RY(np.pi/2, 0)
circuit += RY(np.pi/2, 0)
results = qc.run_and_measure(circuit, trials=100)
plot_histogram(results)

# A lot of quantum algorithms explot interference
# The simplest one to understand its significance is the Deutsch-Josza Algorithm

# More qubits and entanglement -----------------------------------------------
# Define Define the column vector for describing two qubits
# Tensor products
# Given two qubits |psi> = [a0, a1] and |psi '> = [b0, b1]
# The tensor prod gives [a0b0, a0b1, a1b0, a1b1]
# Imagine having two registers q0 and q1 each can hold a qubit and both qubits are in the state |0>

q0 = np.array([[1], [0]])
q1 = np.array([[1], [0]])
print('Tensor product of the two qubits: \n ', np.kron(q0, q1))
# This is the |00> state
# As such we get a four-dimensional complex space

# Interesting and Counter-intuitive part
# In Machine Learning, one works in high-dim space but one never constructs it as a tensor product
# it is normally R^d for some dimension d
# The interesting part of writing the high-dim space as a tensor product is that not all vectors
# in it can be written as a product of vectors in the component space.
# For ex: take the state |phi> = 1/sqrt(2) * (|00> + |11>)
# This vector is clearly in the C2 x C2 space as it's a linear combination of two of the basis vectors
# Howeber it cannot be written as |psi> x |psi '> 
# Proof
# Assume it holds true that you can write that state as the tensor product of those two states
# |phi> = 1/sqrt(2) (|00> + |11>) = [a0b0, a0b1, a1b0, a1b1] 
#       = a0b0 |00> a0b1 |01> + a1b0 |10> + a1b1 |11>
# Since |01> and |10> do not appear on the left side their coeffs must be 0
# This leads to a contradition as a1 cannot be 0 given that a1b1 = 1 so then it means b0 is 0 
# but a0b0 = 1, therefore phi cant be written as a product
# These states that cant be written as a product are called entangled states
# Phenomenon of strong correlation between RVs
# Plays a huge role in quantum algorithms, for e.g. Quantum teleportation and QML protocols

# Let's look at the measurement statistics of the |phi> state

#qvm_server, quilc_server, fc = init_qvm_and_quilc()


qc = get_qc('2q-qvm')
circuit = Program()
# Line below (229) returns a numpy.ndarray object is not callable error
# Demo for the last thing only works until you comment out everything above
# TODO: fix below
circuit += H(0)  
circuit += CNOT(0, 1)
ro = circuit.declare('ro', 'BIT', 2)
circuit += MEASURE(0, ro[0])
circuit += MEASURE(1, ro[1])
circuit.wrap_in_numshots_loop(100)
executable = qc.compile(circuit)
result = qc.run(executable)
plot_histogram(result)  # 01 or 10 never appear

# Extra reading
# Chapter 9 in Quantum Computing since Democritus by Scott Aaronson
