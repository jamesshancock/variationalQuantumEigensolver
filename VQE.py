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
import qiskit.opflow

pauli_0 = np.array([[1, 0],
                   [0, 1]])
pauli_1 = np.array([[0, 1],
                   [1, 0]])
pauli_2 = np.array([[0, -1j],
                   [1j, 0]])
pauli_3 = np.array([[1, 0],
                   [0, -1]])
pauli = [pauli_0,pauli_1,pauli_2,pauli_3]

def Hf(h):
    H = np.zeros((2**n,2**n))
    K = int(len(h)/2)
    Coef = []
    matri = []
    for co in range(K):
        Coef.append(float(h[2*co]))
        matri.append([int(h[2*co+1][j]) for j in range(n)])

    for kj in range(K):
        H_temp = np.array([1])
        for km in range(n):
            H_temp = np.kron(H_temp,pauli[matri[kj][km]])
        H = H + Coef[kj]*H_temp
    return H, Coef, matri

def ryBlock(paray,QC):
    for ky in range(len(paray)):
        QC.ry(paray[ky],ky)
    return QC

def pBlock(varp,QC):
    for kp in range(len(varp)):
        QC.p(varp[kp],kp)
    return QC

def cnotBlock(QC):
    for kc in range(n-1):
        QC.cx(kc,kc+1)
    return QC
        
def measure_1(qubit,QC): #function used to measure in the sigma_1 basis
    QC.h(qubit)
    QC.measure(qubit,qubit)
    return QC
    
def measure_2(qubit,QC): #function used to measure in the sigma_2 basis
    QC.sdg(qubit)
    QC.h(qubit)
    QC.measure(qubit,qubit)
    return QC
    
def measure_3(qubit,QC): #function used to measure in the sigma_3 basis
    QC.measure(qubit,qubit)
    return QC

def measure(case,qubit,QC):
    if case == 1:
        measure_1(qubit,QC)
    elif case == 2:
        measure_2(qubit,QC)
    else:
        measure_3(qubit, QC)
    return QC

def S(sn):
    S = 0    
    while S*sn < 2**(sn):
        S = S + 1
    L = []
    for kl in range(S):
        if (kl % 2) == 0:
            L.append(0)
        else:
            L.append(1)
    return S, L

def para(var,QC):
    for kPara in range(S1):
        paras = var[kPara*n:(kPara+1)*n]
        if L[kPara] == 0:
            ryBlock(paras,QC)
        else:
            pBlock(paras,QC)
            cnotBlock(QC)  
    return QC

def binary_list(nb):
    return ['{:0{}b}'.format(i, nb) for i in range(nb*nb-1)]

def EV(counts,shot):
    b = binary_list(n)
    s = []
    for k in range(len(b)):
        b_temp = b[k]
        s_temp = 0
        for c in range(len(b_temp)):
            if b_temp[c] == '1':
                s_temp = s_temp + 1
        if (s_temp%2) == 1:
            s.append(-1)
        else:
            s.append(1)
    countslist = b
    counts_store = [0]*len(b)
    for x in range(len(b)):
        if b[x] in counts:
            counts_store[x] = counts[b[x]]/shot
    ev = 0
    for kcounts in range(len(counts_store)):
        ev = ev + s[kcounts]*counts_store[kcounts]
    return ev
                         
def sim(QC, shot):
    backend_sim = Aer.get_backend('qasm_simulator')
    job_sim = backend_sim.run(transpile(QC, backend_sim), shots=shot)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(QC)
    return counts

def VQE(Var,h,shot):    
    H, Coef, matri = Hf(h)
    GS = 0
    n = len(h[1])
    q = QuantumRegister(n)
    c = ClassicalRegister(n)
    for k_coef in range(len(Coef)):
        EV1 = 0
        circ = QuantumCircuit(q,c)
        para(Var,circ)
        for g in range(n):
            M = matri[k_coef][g]
            measure(M,g,circ)
        numbers = sim(circ,shot)
        EV1 = EV(numbers,shot)
        GS = GS + Coef[k_coef]*EV1
    return GS

#h below allows you to input the Hamiltonian
#it is of the form ['coefficient_1', 'cross product term_1' , 'coefficient_2', 'cross product term_2',...]

h = ['0.4', '1222', '0.5', '2233', '-1.9', '3311']
shot = 8000
global n
n = len(h[1])
N = 2**(n+1)

global S1
global L
S1, L = S(n)

H, Coef, matri = Hf(h)
E_true, EigenVectors = np.linalg.eig(H)
E_min_true = min(E_true.real)
print(E_min_true)

guess = [0.1]*(N+15)

#%% Realtive error fucntion
def VQEe(Var,h,shot,E):
    VQEee = abs(VQE(Var,h,shot) - E)
    return VQEee
E_min = minimize(VQEe, guess, args = (h,shot,E_min_true), method='powell',
                       options={'ftol': 1e-10, 'disp': True, 'maxfev': 10**4, 'return_all': True})

#%% Annealer
lw = [0]*(S1*n)
up = [4*math.pi]*(S1*n)
anneal = scipy.optimize.dual_annealing(VQE, bounds =list(zip(lw, up)), args = (h,8000))

    

#%% Code for making plots of multiple runs

Shots = [10000,10000,10000,10000,10000]
results = []
iterations = []

for s in range(len(Shots)):
    results_temp = []
    guess = [4*math.pi*random.uniform(0,1)]*(S1*n)
    E_min = minimize(VQE, guess, args = (h,Shots[s]), method='powell',
                           options={'ftol': 1e-10, 'disp': True, 'maxfev': 10**5, 'return_all': True})
    vecz = E_min.allvecs
    for V in range(len(vecz)):
        vqe = VQE(vecz[V],h,Shots[s])
        errorv = abs((vqe-E_min_true)/E_min_true)
        results_temp.append(errorv)
    iterations.append(E_min.nit+1)
    results.append(results_temp)

plt.figure(dpi=800)
plt.semilogy(list(range(iterations[0])),results[0], label = 'Attempt 1')
plt.semilogy(list(range(iterations[1])),results[1], label = 'Attempt 2')
plt.semilogy(list(range(iterations[2])),results[2], label = 'Attempt 3')
plt.semilogy(list(range(iterations[3])),results[3], label = 'Attempt 4')
plt.semilogy(list(range(iterations[4])),results[4], label = 'Attempt 5')
plt.xlabel('Iterations')
plt.ylabel('Relative error')
plt.legend()
plt.show()

