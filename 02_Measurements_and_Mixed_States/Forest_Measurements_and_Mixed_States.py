import numpy as np
from pyquil import Program, get_qc
from pyquil.gates import *
from forest_tools import *


# Bra-Ket
zero_ket = np.array([[1], [0]])
print("|0> ket: \n", zero_ket)
print("<0| bra: \n", zero_ket.T)

# Dot Products
#<0|0>
print('<0|0> = ', zero_ket.T.conj().dot(zero_ket))
# <0|1> should give zero since they are orthogonal
one_ket = np.array([[0], [1]])
print('<0|1> = ', zero_ket.T.conj().dot(one_ket))
# |0> <0|
print('|0><0| =  \n', zero_ket.dot(zero_ket.T.conj()))
# ^ Basically the projection to the first element of the canonical basis
# |psi><psi| is going to be a projector to |psi>
# Take another state |phi>
# apply the projector on it : |psi><psi|phi>
# The right most terms are a dot product the overlap between |psi> and |phi>
# Since it is a scalar, it scales the left most term a ket |psi> so we projected
# |phi> on this vector

# Measurements --------------------------------------------
"""
Measurement in QM is a operator-valued random variable.
Most qc's today only implement one specific measurement
Measurement in the canonical basis
The measure contains two projections |0><0| and |1><1| and they
can be applied to any of the qubits of the quantum computer

We saw how applying a projection on a vector works. now we want to
make a scalar value. We do that by adding a bra to the left
For some state |psi>, we get a scalar for <psi|0><0|psi> this is called
the expectation value of the operator |0><0|
Below we will apply the projection |0><0| on the superposition
1/sqrt(2) (|0> + |1>) which is the col vector 1/sqrt(2) [1,1]
"""

psi = np.array([[1], [1]])/np.sqrt(2)
proj_0 = zero_ket.dot(zero_ket.T.conj())
print('<psi|0><0|psi> = ', psi.T.conj().dot(proj_0.dot(psi)))  # prints 0.5

# The above should print 1/2 which is the square of the abs value of the prob amplitude
# corresponding to |0> in the superposition
# given |psi> = a0 |0> + a1 |1> , one gets an output i with probability |a_i}^2
# This is known as the Born Rule
# The above is a recipe to extract probabilities with projection
# The measurement in a quantum simulator is what is described here

# Create a equal superposition with the Hadamard gate, apply measurement and observe stats

qc = get_qc('1q-qvm')
circuit = Program()
circuit += H(0)
results = qc.run_and_measure(circuit, trials=100)
# Uncomment to view
#plot_histogram(results)  # roughly half of the outcomes are 0

# The measurement has a random outcome
# Once performed, the quantum state is in the corresponding basis vector
# The superposition is destroyed
# This is referred to as the collapse of the wavefunc
# Mathematically expressed by |i> <i|psi> divided by sqrt( <psi|i> <i|psi>) 
# If we observe zero after measuring the superpsotion the state after the measurement will be
psi = np.array([[np.sqrt(2)/2], [np.sqrt(2)/2]])
proj_0 = zero_ket.dot(zero_ket.T.conj())
probability_0 = psi.T.conj().dot(proj_0.dot(psi))
print('State collapse = \n', proj_0.dot(psi)/np.sqrt(probability_0))  # gives |0>

# One can also see this by putting two measurements in a sequence on the same qubit
# Second one will always give the same outcome as the first
# First one is random but the second is determined since no superposition

# Create two diff classical registers
circuit = Program()
circuit += H(0)
ro = circuit.declare('ro', 'BIT', 2)
circuit += MEASURE(0, ro[0])
circuit += MEASURE(0, ro[1])
circuit.wrap_in_numshots_loop(100)
executable = qc.compile(circuit)
result = qc.run(executable)
print(result)  # there is no 01 or 10

# Measuring multiqubit systems ------------------------------
# Most qc's implement local measurements which means each qubit is measured seperately
# If we have a two qubit system where the first qubit is in equal superposition and the second
# one is in |0>, that is we have the state 1/sqrt(2) (|00> + |01>) we will observe 0 and 0 as 
# outcomes of the measurements on the two qubits, or 0 and 1. 

qc = get_qc('2q-qvm')
circuit = Program()
circuit += H(0)
results = qc.run_and_measure(circuit, trials=100)
plot_histogram(results)  # 00 and 10 in equal super pos

# What happens if you take measurements on the entangled state? Looking at |phi +>
circuit = Program()
circuit += H(0)
circuit += CNOT(0, 1)
results = qc.run_and_measure(circuit, trials=100)
plot_histogram(results)  # 00 and 11
# The state is 1/sqrt(2) (|00> + |11>) 
# End of the last section gave the same results
#  but from measuremenrs on the same qubit
# Now we have two spatially seperate qubits exhibiting the same behaviour
# Very strong form of correlations. 
# If we measure just one qubit and get 0 as the outcome we know with certainty
# if we measured the other qubit, we would also get 0 even though second measurement is also a RV

# Imagine tossing two unbiased coins
# If you observe heads on one, there is nothing you can say about what the other might be
# Other than a wild guess that holds with prob 0.5
# If you play foul and you biased the coins, your guessing accuracy increases
# but you cant say with certainty will be based on the outcome you saw on the first coin
# other than the trivial case where the other coin deterministically gives the same face always

# Mixed States -------------------------------------------
# A ket and a bra is a density matrix
# it is another way of writing a quantum states instead of kets
# one could write rho = |psi><psi| where rho is the density matrix for |psi>
# Born rule still applies, now one has to take the trace to get the result
# Tr[|0><0|rho] is the prob of getting 0 

psi = np.array([[1], [1]])/np.sqrt(2)
rho = psi.dot(psi.T.conj())
proj_0 = zero_ket.dot(zero_ket.T.conj())
print('The probability of getting 0 using density matrix: \n', np.trace(proj_0.dot(rho)))
# gets 1/2 again
# Renormalization happens similarly (|0><0|rho|0> <0|) / Tr[|0><0|rho]
probability_0 = np.trace(proj_0.dot(rho))
print('Renormal \n', proj_0.dot(rho).dot(proj_0)/probability_0)

# Every state mentioned so far is a pure state - kets or a density matrix
# Mixed states are like classical probability distributions over pure states
# formally its written as sum_i p_i |psi_i><psi_i| where sum_i p_i = 1
# This reflects the classical ignorance over underlying quantum states
# Compare the density matrix of the equal super position 1/sqrt(2) (|0> + |1>) 
# And the mixed state 1/2 (|0><0| + |1><1|)

zero_ket = np.array([[1], [0]])
one_ket = np.array([[0], [1]])
psi = (zero_ket + one_ket)/np.sqrt(2)
print('Density matrix of the equal superposition')
print(psi.dot(psi.T.conj()))
print('Density matrix of the equally mixed state of |0><0| and |1><1|')
print((zero_ket.dot(zero_ket.T.conj()) + one_ket.dot(one_ket.T.conj()))/2)

# Off diagonals are gone in the second case
# The off diagonal elements are called coherences
# The presnece of them shows the state is quantum
# the smaller the values, the closer the quantum state is to a classicla probability distributiojn
# The second density matrix above has only diagonal elements and they are equal this is the equivalent
# of saying a uniform distribution and uniform distributions have maximum entropy
# This density matrix with that structure is a maximally mixed state
# We are perfectly ignorant of which elements of the canonical basis constitute the state
# We want the quantum state to be perfectly isoalted from the environment
# However in reality, its hard to achieve as coherences are slowly lost to the environment
# The speed at which decoherence happens determines the length of the quantum algoirthm can run on the qc
# If it happens fast, we have time to apply a handful of gates or do any form of calc
# Have to quickly measure the result
