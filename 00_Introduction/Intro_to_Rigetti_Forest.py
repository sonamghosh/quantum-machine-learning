import socket
import subprocess
from pyquil.api import ForestConnection
from pyquil import Program  # describes the circuit
from pyquil import get_qc  # allows to define classical registers
from pyquil.gates import MEASURE, I, H
from pyquil.api import WavefunctionSimulator
from pyquil.latex import to_latex
import matplotlib.pyplot as plt
import numpy as np
import shutil
from tempfile import mkdtemp

# Initialize Quantum Virtual Machine and Quil Compiler using qvm -S and quilc -S
def get_free_port():
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def init_qvm_and_quilc(qvm_executable='qvm', quilc_executable='quilc'):
    qvm_port = get_free_port()
    quilc_port = get_free_port()
    qvm_server = subprocess.Popen([qvm_executable, '-S', '-p', str(qvm_port)])
    quilc_server = subprocess.Popen([quilc_executable, '-S', '-p', str(quilc_port)])
    fc = ForestConnection(sync_endpoint='http://127.0.0.1:' + str(qvm_port),
                          compiler_endpoint='http://127.0.0.1:' + str(quilc_port))

    return qvm_server, quilc_server, fc

def plot_circuit(circuit):
    latex_diagram = to_latex(circuit)
    tmp_folder = mkdtemp()
    print(tmp_folder)
    with open(tmp_folder + '/circuit.tex', 'w') as f:
        f.write(latex_diagram)
    proc = subprocess.Popen(['pdflatex', '-shell-escape', tmp_folder + '/circuit.tex'], cwd=tmp_folder)
    proc.communicate()
    image = plt.imread(tmp_folder + '/circuit.png')
    shutil.rmtree(tmp_folder)
    plt.axis('off')
    return plt.imshow(image)


if __name__ == "__main__":
    #qvm_server, quilc_server, fc = init_qvm_and_quilc()
    #qvm_server, quilc_server, fc = init_qvm_and_quilc('/home/local/bin/qvm', '/home/local/bin/quilc')

    ################
    # Backends
    ################

    # 1-qubit QVM
    qc = get_qc('1q-qvm')

    # Circuit with no gates and 1 qubit
    # Write out result to single classical register
    circuit = Program()
    ro = circuit.declare('ro', 'BIT', 1)
    circuit += MEASURE(0, ro[0])

    # Repeat the runs
    # Call Quantum Compiler
    # Run it on QVM
    circuit.wrap_in_numshots_loop(100)
    executable = qc.compile(circuit)
    result = qc.run(executable)

    # Inspect stats of 0 and 1
    print((result==0).sum())
    print((result==1).sum())
    print("-----------------------------------------------")
    # Easier method is to let simulator do the measurements
    circuit = Program()
    result = qc.run_and_measure(circuit, trials=100)
    print(result)
    print("-----------------------------------------------")
    # Qubit Registers are always initialized as |0>
    # Simulated Quantum State
    wf_sim = WavefunctionSimulator()
    # Circuit w.o a measurement and direct state inspection
    circuit = Program()
    wavefunction = wf_sim.wavefunction(circuit)
    print('state = ', wavefunction)
    print("-----------------------------------------------")

    ###############
    # Visualization
    ###############
    
    circuit = Program()
    circuit += H(0)
    plot_circuit(circuit)