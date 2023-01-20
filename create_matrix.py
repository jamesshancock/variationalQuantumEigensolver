#
#  Create the Hamiltonan 
#
#

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
#a = random.rand()

def create_pauli_coeff_A() :
   h = np.array([[0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 3.4, 0.0]])
   return h,"H_store_A"

# select the matrix to use
create_pauli_coeff = create_pauli_coeff_A

#
#  create the matrix
#

h, filename = create_pauli_coeff()

H = np.zeros((4,4))

pauli = 'pauli'

#generating the hamiltonian
for k in range(4):
    for s in range(4):
        matrix1 = locals()[pauli +'_'+str(k)]
        matrix2 = locals()[pauli +'_'+str(s)]
        Pauli_kron = np.kron(matrix1,matrix2)
        H = H + h[k,s]*Pauli_kron
        

#
#  save the matrix to disk
#

np.save(filename, H)

print("The matrix is saved in the file" , filename)
