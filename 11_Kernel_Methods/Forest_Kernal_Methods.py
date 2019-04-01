# Kernel Methods were built on the premise of
# introducing a notion of distance between high-dimensional data points
# by replacing the inner product by a function that retains many properties
# of the inner product but is nonlinear


# Learning methods based on what the hardware can do -------------------------

from pyquil import Program, get_qc
from pyquil.gates import *
from forest_tools import *

qc = get_qc('4q-qvm')

# Construct an instance-based classifier - calculating the kernal between
# all training instances and a test example.

# The algorithm itself is lazy as no learning is happening and each prediction
# includes the entire training set.

# State prep is critical to the protocol
# The training instances need to be encoded in a superposition in a register
# and test instances in another register

# Consider the following training instances of the iris 
# S = {([0; 1], 0), ([0.7886, 0.615], 1)} , one example from class 0 and one from class 1
# We have two test instances {[-0.549, 0.836], [0.053, 0.999]}

training_set = [[0, 1], [0.78861006, 0.61489363]]
labels = [0, 1]
test_set = [[-0.549, 0.836], [0.053, 0.999]]

# Use amplitude encoding
# i.e. second training vector will become 0.78861006 |0> + 0.61489363 |1>
# Preparing the vectors only needs a rotation, we need to specify the corresponding angles
# The first element does not need that, its just the |1> state

# To get the angle the following eqn needs to be solved
# a|0> + b|1> = cos(t/2)|0> + i*sin(t/2)|1>
# therefore we use t = 2arccos(a)

def get_angle(amplitude_0):
    return 2*np.arccos(amplitude_0)

# in practice, the state procedure we consider requires the application of several rotations
# by theta and -theta in order to prepare each data point in the good register
# As a consequence, the test angles need to be divided by 2 ( 2 rotations) and the training angles by 4

test_angles = [get_angle(test_set[0][0])/2, get_angle(test_set[1][0])/2]
training_angle = get_angle(training_set[1][0])/4

def prepare_state(angles):
    ancilla_qubit = 0
    index_qubit = 1
    data_qubit = 2
    class_qubit = 3
    circuit = Program()

    # Put the ancilla and the index qubits into uniform superposition
    circuit += H(ancilla_qubit)
    circuit += H(index_qubit)

    # Prepare the test vector
    circuit += CNOT(ancilla_qubit, data_qubit)
    circuit += RZ(-angles[0], data_qubit)
    circuit += CNOT(ancilla_qubit, data_qubit)
    circuit += RZ(angles[0], data_qubit)
    # Flip the ancilla qubit > this moves the input
    # vector to the |0> state of the ancilla
    circuit += X(ancilla_qubit)

    # Prepare the first training vector
    # [0, 1] -> class 0
    # We can prepare this with a Toffoli
    circuit += CCNOT(ancilla_qubit, index_qubit, data_qubit)
    # Flip the index qubit > moves the first training vector to the
    # |0> state of the index qubit
    circuit += X(index_qubit)

    # Prepare the second training vector
    # [0, 78861, 0.61489]

    circuit += CCNOT(ancilla_qubit, index_qubit, data_qubit)
    circuit += CNOT(index_qubit, data_qubit)
    circuit += RZ(angles[1], data_qubit)
    circuit += CNOT(index_qubit, data_qubit)
    circuit += RZ(-angles[1], data_qubit)
    circuit += CCNOT(ancilla_qubit, index_qubit, data_qubit)
    circuit += CNOT(index_qubit, data_qubit)
    circuit += RZ(-angles[1], data_qubit)
    circuit += CNOT(index_qubit, data_qubit)
    circuit += RZ(angles[1], data_qubit)

    # Flip the class label for training vector #2
    circuit += CNOT(index_qubit, class_qubit)

    return circuit

"""
The second training state (0.78861, 0.61489) is entangled with the excited
state of the ancilla and the excited state of the index qubit

angles[1] = 1.3245 is used becaused , the basis state |0> has to be rotated
to contain the vector that we want. 

One can write the generic state as (cos(t/2), sin(t/2)). The function argument
divides the angle by two, 

When you change the sign of t, you reverse the unitary. (check paper). By flipping
the sign you uncompute the register.

"""

# Prepare the state with the first instance

angles = [test_angles[0], training_angle]
state_preparation_0 = prepare_state(angles)
plot_circuit(state_preparation_0)
#plt.show()

"""
The test instance is prepared until the first barrier.

The ancilla and index qubits (registers 0 and 1) are put into uniform
superposition. The test instance is entangled with the ground state of the ancilla

Then between the first and second barriers, we prepare the state |1>, which
is the first training instance, and entangle it with the excited state of the ancilla
and the ground state of the index qubit with a Toffoli gate and a Pauli x gate

The toffooli gate is also called the controlled controlled not gate.

The third section prepares the second training instance and entangles it
with the excited state of the ancilla and the index qubit.

The final part flips the class qubit conditioned on the index qubit. 
This create a connection between the encoded training instances and the corresponding class label

"""


# A natural kernal on a shallow circuit

# the actual prediction is nothing but a Hadamard gate applied on the ancilla, followed
# by a measurements

# The ancilla is in a uniform superposition at the end of state prep and entangled
# with the registers encoding the test and training instances, applying a second Hadamard
# on the ancilla interferes the entangled registers.

# the state before the measurement is
# 1 / (2*sqrt(2)) sum_(i=0 to 1) |0>|i>(|x_t> + |x_i>)|y_i> + |1>|i>(|x_t> - |x_i>)|y_i>
# where |x_t> is the encoded test instance
# and |x_i> is the training instance

def interfere_data_and_test_instances(circuit, angles):
    ro = circuit.declare(name='ro', memory_type='BIT', memory_size=4)
    circuit += H(0)
    for q in range(4):
        circuit += MEASURE(q, ro[q])

    return circuit

# if we measure the ancilla, the outcome probability of observing 0 will be
# 1/(4N) * sum_(i=1 to N) |x_t + x_i|^2, 

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 2, 100)
plt.xlim(-2, 2)
plt.ylim(0, 1.1)
plt.plot(x, 1-x**2/4)

#plt.show()

# the kernel above will perform the classification
# post selection on observing 0 on the measuremennt on the ancilla
# calculate the probablities of the test instance belonging to either class

def postselect(result_counts):
    total_samples = sum(result_counts.values())

    # define lambda function that retrieves only results where the ancilla is in the |0> state
    post_select = lambda counts: [(state, occurences) for state, occurences in counts.items() if state[-1] == '0']

    # perform the post selection
    postselection = dict(post_select(result_counts))
    postselected_samples = sum(postselection.values())

    print(f'Ancilla post-selection probability was found to be {postselected_samples/total_samples}')

    retrieve_class = lambda binary_class: [occurences for state, occurences in postselection.items() if state[0] == str(binary_class)]

    prob_class0 = sum(retrieve_class[0])/postselected_samples
    prob_class1 = sum(retrieve_class[1])/postselected_samples

    print('probability for class 0 is', prob_class0)
    print('probability for class 1 is', prob_class1)


# first instance
circuit_0 = interfere_data_and_test_instances(state_preparation_0, angles)
circuit_0.wrap_in_numshots_loop(1024)
executable = qc.compile(circuit_0)
measures = qc.run(executable)

count = np.unique(measures, return_counts=True, axis=0)
count = dict(zip(list(map(lambda l: ''.join(list(map(str, 1))), count[0].tolist())), count[1]))
print(count)

postselect(count)

# second instance
angles = [test_angles[1], training_angle]
state_preparation_1 = prepare_state(angles)

circuit_1 = interfere_data_and_test_instances(state_preparation_1, angles)
executable = qc.compile(circuit_1)
measures = qc.run(executable)


count = np.unique(measures, return_counts=True, axis=0)
count = dict(zip(list(map(lambda l: ''.join(list(map(str, 1))), count[0].tolist())), count[1]))
print(count)

postselect(count)


