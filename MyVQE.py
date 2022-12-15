import math
import cmath
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg
from scipy.optimize import minimize
from numpy import random
import qiskit
from qiskit import *
from qiskit import Aer
import numpy as np
from qiskit.opflow import X, Y, Z, I
import qiskit.opflow

n = 2 #number of qubits and classical bits for their measurement

def measure_1(qubit,QC): #function used to measure in the sigma_1 basis
    QC.h(qubit)
    QC.measure(qubit,qubit)
    
def measure_2(qubit,QC): #function used to measure in the sigma_2 basis
    QC.sdg(qubit)
    QC.h(qubit)
    QC.measure(qubit,qubit)
    
def measure_3(qubit,QC): #function used to measure in the sigma_3 basis
    QC.measure(qubit,qubit)
    
def circ(var,QC): #fucntion for initiating the circuit in the parameterised state
    QC.ry(var[4],0)
    QC.ry(var[5],1)
    QC.rz(var[0],0)
    QC.rz(var[1],1)
    QC.cx(0,1)
    QC.ry(var[6],0)
    QC.ry(var[7],1)
    QC.rz(var[2],0)
    QC.rz(var[3],1)
    return QC

#the next block of code simulates the system N times and stores the results
def sim(QC):
    backend_sim = Aer.get_backend('qasm_simulator')
    job_sim = backend_sim.run(transpile(QC, backend_sim), shots=N)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(QC)
    return counts

h = np.array([[0.0, 0.0, 0.0, 0.0],
             [2.0, 0.0, 7.2, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0]])

#generating the hamiltonian for finding the true eigenvalue
pauli_0 = np.array([[1, 0],
                   [0, 1]])
pauli_1 = np.array([[0, 1],
                   [1, 0]])
pauli_2 = np.array([[0, -1j],
                   [1j, 0]])
pauli_3 = np.array([[1, 0],
                   [0, -1]])
H = np.zeros((4,4))
for counter_1 in range(4):
    for counter_2 in range(4):
        exec(f'matrix1 = pauli_{counter_1}')
        exec(f'matrix2 = pauli_{counter_2}')
        Pauli_kron = np.kron(matrix1,matrix2)
        H = H + h[counter_1,counter_2]*Pauli_kron
E_true, EigenVectors = np.linalg.eig(H)
E_min_true = min(E_true)

guess = [2*math.pi,2*math.pi,2*math.pi,2*math.pi,2*math.pi,2*math.pi,2*math.pi,2*math.pi]

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#THE QUANTUM CIRCUIT
def VQE(var,h):
    GS = 0
    EV = 0
    q = QuantumRegister(n) #creating the qubits
    c = ClassicalRegister(n) #creating the classical bits
    for k in range(0,4):
        for s in range(0,4):
            QC = QuantumCircuit(q,c)
            circ(var,QC)
            if h[k,s] != 0:
                if k == 0 and s > 0:
                    measure_3(0,QC)
                    exec(f'measure_{s}(1,QC)')
                    test = 1
                elif s == 0 and k > 0:
                    QC.swap(0,1)
                    measure_3(0,QC)
                    exec(f'measure_{k}(1,QC)')
                    test = 1
                elif k > 0 and s > 0:
                    exec(f'measure_{k}(0,QC)')
                    exec(f'measure_{s}(1,QC)')
                    test = 0
                elif k == 0 and s == 0:
                    test = 2
                #this block of code finds the number of times the system was measured in each state
                if test == 2:
                    EV = 1
                    
                elif test != 2:
                    counts = sim(QC)
                    countslist = ['00','01','10','11']
                    counts_store = [0,0,0,0]
                    for x in range(2**n):
                        if countslist[x] in counts:
                            counts_store[x] = counts[countslist[x]]/N
                    EV = 0
                    if test == 0: #the test value allows for the fact that when a sigma_0 is included the formula changes
                        #then uses these numbers to calculate the expected value of the Hamiltonian - this is specific to 4x4
                        EV = counts_store[0] - counts_store[1] - counts_store[2] + counts_store[3]
                    else:
                        #then uses these numbers to calculate the expected value of the Hamiltonian - this is specific to 4x4
                        EV = counts_store[0] + counts_store[1] - counts_store[2] - counts_store[3]
                
                else:
                    EV = 0
            GS = GS + h[k,s]*EV
    return GS
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
N_list = [6000]
for N in N_list:
    E_min = minimize(VQE, guess, args = (h), method='powell',
                           options={'xtol': 1e-5, 'disp': True, 'maxfev': 10**4, 'return_all': True})     
    n_it = E_min.nit
    vecs = E_min.allvecs
    averages = []
    errors = []
    for index in range(n_it+1):
        exec(f'vec_{index}_{N} = []')
        for useless_counter in range(10):
            exec(f'vec_{index}_{N}.append(VQE(vecs[index],h))')
        exec(f'vec_{index}_{N}_mean = np.mean(vec_{index}_{N})')
        exec(f'averages.append(vec_{index}_{N}_mean)')
        exec(f'vec_{index}_{N}.sort()')
        exec(f'errors.append(abs(vec_{index}_{N}[useless_counter]-vec_{index}_{N}[0]))')
    x_pos = np.arange(n_it+1)
    true_lambda = [E_min_true]*(n_it+1)
    fig, ax = plt.subplots()
    ax.errorbar(x_pos, averages,
                yerr=errors,
                fmt='ro',
                label = 'Calculated expected value')
    ax.set_ylabel('Expected value')
    ax.set_xlabel('Iterations')
    plt.plot(x_pos,true_lambda, color = 'black', label = 'True eigenvalue')
    leg = ax.legend()
    fig.set_size_inches(10, 5)
    plt.show()
    

