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
a = random.rand()
h = np.array([[0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 3.4, 0.0]])


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

d = np.array([[1],[0]])

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
    phi_3 = var[2]
    phi_4 = var[3]
    theta_1 = var[4]
    theta_2 = var[5]
    theta_3 = var[6]
    theta_4 = var[7]
    Rytheta1 = np.array([[math.cos(theta_1), -math.sin(theta_1)],
                        [math.sin(theta_1), math.cos(theta_1)]])
    Rzphi1 = np.array([[cmath.exp(-1j*phi_1),0],
                       [0,cmath.exp(1j*phi_1)]])
    Rytheta2 = np.array([[math.cos(theta_2), -math.sin(theta_2)],
                        [math.sin(theta_2), math.cos(theta_2)]])
    Rzphi2 = np.array([[cmath.exp(-1j*phi_2),0],
                       [0,cmath.exp(1j*phi_2)]])
    Rytheta3 = np.array([[math.cos(theta_3), -math.sin(theta_3)],
                        [math.sin(theta_3), math.cos(theta_3)]])
    Rzphi3 = np.array([[cmath.exp(-1j*phi_3),0],
                       [0,cmath.exp(1j*phi_3)]])
    Rytheta4 = np.array([[math.cos(theta_4), -math.sin(theta_4)],
                        [math.sin(theta_4), math.cos(theta_4)]])
    Rzphi4 = np.array([[cmath.exp(-1j*phi_4),0],
                       [0,cmath.exp(1j*phi_4)]])

    CNOT = np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,0,1],
                     [0,0,1,0]])

    ini = np.kron(d,d)
    block1 = np.kron(Rytheta1, Rytheta2)
    block2 = np.kron(Rzphi1,Rzphi2)
    block3 = np.kron(Rytheta3, Rytheta4)
    block4 = np.kron(Rzphi3,Rzphi4)

    step1 = block1.dot(ini)
    step2 = block2.dot(step1)
    step3 = CNOT.dot(step2)
    step4 = block3.dot(step3)
    vec = block4.dot(step4)

    Est = expected(vec,Hamil)
    return Est


#
#   Mimization
#


guess = [2*math.pi*random.uniform(0,1), 2*math.pi*random.uniform(0,1), 
         2*math.pi*random.uniform(0,1), 2*math.pi*random.uniform(0,1),
         2*math.pi*random.uniform(0,1), 2*math.pi*random.uniform(0,1), 
         2*math.pi*random.uniform(0,1), 2*math.pi*random.uniform(0,1)]


M = ['Nelder-Mead', 'Powell', 'CG', 'BFGS']
E_min = minimize(VQE, guess, args = (H), method='Powell',
               options={'xatol': 1e-10, 'disp': True})

#
# Compute the errors on the eigenvalues
#
error_eig = abs(E_min.fun-E_min_true)
error_eig_rel = abs(E_min.fun-E_min_true) / abs(E_min_true)

print('Error |E - E^{true}| = ' , error_eig )
print('Relative Error |(E - E^{true}) / E^{true}| = ' , error_eig_rel )

#
# Compute the errors on the eigenvectors
#

print("Eigenvector analysis")
print(E_min.x)
