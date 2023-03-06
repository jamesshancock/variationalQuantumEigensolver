import math
import cmath
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg
from scipy.optimize import minimize
from numpy import random
import numpy as np

pauli_0 = np.array([[1, 0],
                   [0, 1]])
pauli_1 = np.array([[0, 1],
                   [1, 0]])
pauli_2 = np.array([[0, -1j],
                   [1j, 0]])
pauli_3 = np.array([[1, 0],
                   [0, -1]])
pauli = [pauli_0,pauli_1,pauli_2,pauli_3]

initial = np.array([[1],[0]])


def ry(theta):
    return np.array([[math.cos(theta/2),-math.sin(theta/2)],
                   [math.sin(theta/2),math.cos(theta/2)]])
def rz(phi):
    return np.array([[cmath.exp(-1j*phi/2),0],
                   [0,cmath.exp(1j*phi/2)]])

def p(lambda1):
    return np.array([[1,0],
                   [0,cmath.exp(1j*lambda1)]])

def CNOTf(ctrl,targ):
    zero = np.array([[1],[0]])
    one = np.array([[0],[1]])

    zero_matrix = np.outer(zero,zero)
    one_matrix = np.outer(one,one)

    CNOT_zerocase = np.array([1])
    CNOT_onecase = np.array([1])

    for k in range(n):
        if k == targ:
            CNOT_zerocase = np.kron(pauli_0,CNOT_zerocase)
            CNOT_onecase = np.kron(pauli_1,CNOT_onecase)
        elif k == ctrl:
            CNOT_zerocase = np.kron(zero_matrix,CNOT_zerocase)
            CNOT_onecase = np.kron(one_matrix,CNOT_onecase)
        else:
            CNOT_zerocase = np.kron(pauli_0,CNOT_zerocase)
            CNOT_onecase = np.kron(pauli_0,CNOT_onecase)
    CNOT = CNOT_zerocase+CNOT_onecase
    return CNOT


x = np.array([[0,1],[1,0]])

#h below allows you to input the Hamiltonian
#it is of the form ['coefficient_1', 'cross product term_1' , 'coefficient_2', 'cross product term_2',...]

h = ['0.4', '1222', '0.5', '2233', '-1.9', '3311']

n = len(h[1])

start = np.array([1])
for k in range(n):
    start = np.kron(initial,start)

def S(sn):
    S = 0    
    while S*sn < 2**(sn+1):
        S = S + 1
    return S

N = 2**(n+1) + 10
guess = [random.uniform(0,1)]*N

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
    return H
        

def ryBlock(theta):
    block = np.array([1])
    for kt in range(n):
        block = np.kron(ry(theta[kt]),block)
    return block

def pBlock(phi):
    block = np.array([1])
    for kp in range(n):
        block = np.kron(p(phi[kp]),block)
    return block

def cnotBlock():
    block = np.identity(2**n)
    for kc in range(n-1):
        block = np.dot(CNOTf(kc,kc+1),block)
    return block

def pcBlock(phi2):
    P = pBlock(phi2)
    CNOT = cnotBlock()
    return np.dot(P,CNOT)
   
blocks = []
for ks in range(S(n)):
    if (ks % 2) == 0:
        blocks.append('Ry')
    else:
        blocks.append('P')
    
def VQE(var,H):
    C = np.identity(2**n)
    B_l = len(blocks)
    for B in range(B_l):
        paras = var[B*n:(B+1)*n]
        if blocks[B] == 'Ry':
            C = np.dot(ryBlock(paras),C)
        else:
            C = np.dot(pcBlock(paras),C)
    vec = np.dot(C,start)
    vec2 = np.dot(H,vec)
    EV = np.vdot(vec,vec2)
    return EV.real

H = Hf(h)

E_true, EigenVectors = np.linalg.eig(H)
E_min_true = min(E_true.real)
print(E_min_true)

E_min = minimize(VQE, guess, args = (H), method='powell',
                       options={'ftol': 1e-10, 'disp': True, 'maxfev': 10**4, 'return_all': True})
error = abs(E_min_true-E_min.fun)/abs(E_min_true)
print('Relative error = '+str(error))
    
#%% Code for making plots of multiple runs
Tol = [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-20]
for tol in range(len(Tol)):
    results_temp = []
    guess = [0.1]*(S1*n)
    E_min = minimize(VQE, guess, args = (H), method='powell',
                           options={'ftol': tol, 'disp': True, 'maxfev': 10**5, 'return_all': True})
    vecz = E_min.allvecs
    for V in range(len(vecz)):
        vqe = VQE(vecz[V],H)
        errorv = abs((vqe-E_min_true)/E_min_true)
        results_temp.append(errorv)
    iterations.append(E_min.nit+1)
    results.append(results_temp)

plt.figure(dpi=800)
for kplots in range(len(Tol)):
    plt.semilogy(list(range(iterations[kplots])),results[kplots], label = str(Tol[kplots]))
plt.xlabel('Iterations')
plt.ylabel('Relative error')
plt.legend()
plt.show()
    
    
    
    
    
    
    
