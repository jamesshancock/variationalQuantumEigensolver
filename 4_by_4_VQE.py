import math
import cmath
import random
import numpy as np
import scipy
from scipy import linalg
from scipy.optimize import minimize
from numpy import random


pauli_0 = np.array([[1, 0],
                   [0, 1]])
pauli_1 = np.array([[0, 1],
                   [1, 0]])
pauli_2 = np.array([[0, -1j],
                   [1j, 0]])
pauli_3 = np.array([[1, 0],
                   [0, -1]])

#choosing the coefficients for the hamiltonian
h = np.array([[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]])

#h = random.rand(4,4)

H = np.zeros((4,4))

pauli = 'pauli'

#generating the hamiltonian
for k in range(4):
    for s in range(4):
        matrix1 = locals()[pauli +'_'+str(k)]
        matrix2 = locals()[pauli +'_'+str(s)]
        Pauli_kron = np.kron(matrix1,matrix2)
        H = H + h[k,s]*Pauli_kron
        
#showing the numerical value for E_min
E_true, EigenVectors = np.linalg.eig(H)
E_min_true = min(E_true)
print('The true value of the smallest eigenvalue is:', E_min_true.real)

#function for calculating the expectation value
def expected(vec,Hamil):
    vec2 = Hamil.dot(vec)
    v = np.vdot(vec,vec2)
    vv = np.vdot(vec,vec).real
    v = v.real/vv
    return v

#function that generates a vector guess and calculates the expectation value 
def VQE(var,Hamil):
    phi_1 = var[0]
    phi_2 = var[1]
    theta_1 = var[2]
    theta_2 = var[3]
    vec1 = ([[cmath.exp(-1j*phi_1)*math.cos(theta_1)],
                        [cmath.exp(1j*phi_1)*math.sin(theta_1)]])
    vec2 = ([[cmath.exp(-1j*phi_2)*math.cos(theta_2)],
                        [cmath.exp(1j*phi_2)*math.sin(theta_2)]])
    vec = np.kron(vec1, vec2)

    Est = expected(vec,Hamil)
    return Est

guess = [2*math.pi*random.uniform(0,1), 2*math.pi*random.uniform(0,1), 
         2*math.pi*random.uniform(0,1), 2*math.pi*random.uniform(0,1)]


M = ['Nelder-Mead', 'Powell', 'CG', 'BFGS']
E_min = minimize(VQE, guess, args = (H), method='Nelder-Mead',
               options={'xatol': 1e-10, 'disp': True})
error = abs(E_min.fun-E_min_true)
print('Error = '+str(error))
#Minimizer_kwargs = {'args': H}
#E_min2 = scipy.optimize.basinhopping(VQE, guess, minimizer_kwargs=Minimizer_kwargs)
#error = abs(E_min2.fun-E_min_true)
#print(error)
