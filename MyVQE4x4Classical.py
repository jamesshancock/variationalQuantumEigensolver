#  Compute the lowest eigenvalue using the
#  Variational Quantum Algorithm.
#  This is a full clasical algorithm
#

import math
import cmath
import random
import numpy as np
import scipy
from scipy import linalg
from scipy.optimize import minimize
from numpy import random
import pickle

M = ['Nelder-Mead', 'Powell', 'CG', 'BFGS']
algorithm = 'Powell'
#algorithm = 'Nelder-Mead'
filename = "H_store_A.npy"

eig_store = [] 

try:
  H = np.load(filename)
  print("Loaded the matrix from ", filename)
except:
  print("Error loading matrix from " , filename)  
  sys.exit(0)

#showing the numerical value for E_min
E_true, EigenVectors = np.linalg.eig(H)
E_min_true = min(E_true)
print('The true value of the smallest eigenvalue is:', E_min_true.real)

#

#function for calculating the expectation value
def expected(vec,Hamil):
    vec2 = Hamil.dot(vec)
    v = np.vdot(vec,vec2)
    vv = np.vdot(vec,vec).real
    v = v.real/vv
    return v

#function that generates a vector guess and calculates the expectation value 
def create_vec(var):
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

    d = np.array([[1],[0]])
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

    return vec

def VQE(var,Hamil):
    """
    Create the vec from the parameters var

    Est = <vec, Hamil * vec>
    """
    vec = create_vec(var)
    Est = expected(vec,Hamil)
    return Est



#
#   Minization
#

Nfeval = 1

def callbackF(Xi):
    global Nfeval
    global eig_store 

    ff =  VQE(Xi,H)
    print("Iter", Nfeval, Xi, "eigenvalue_min=" , ff)
    Nfeval += 1
    eig_store.append(ff)


guess = [2*math.pi*random.uniform(0,1), 2*math.pi*random.uniform(0,1), 
         2*math.pi*random.uniform(0,1), 2*math.pi*random.uniform(0,1),
         2*math.pi*random.uniform(0,1), 2*math.pi*random.uniform(0,1), 
         2*math.pi*random.uniform(0,1), 2*math.pi*random.uniform(0,1)]


E_min = minimize(VQE, guess, args = (H), method=algorithm,
               callback=callbackF,
               tol=1e-10, options={'disp': True})

#
# Compute the errors on the eigenvalues
#
error_eig = abs(E_min.fun-E_min_true)
error_eig_rel = abs(E_min.fun-E_min_true) / abs(E_min_true)

print('The true value of the smallest eigenvalue is:', E_min_true.real)
print("Minimization algorithm = " , algorithm)
print('Eigenvalue Error |E - E^{true}| = ' , error_eig )
print('Eigenvalue Relative Error |(E - E^{true}) / E^{true}| = ' , error_eig_rel )

#
# Compute the errors on the eigenvectors
#

print("Eigenvector analysis")

lowest_eigenvec_est = create_vec(E_min.x)


print('Estimated lowest eigenvector = ' )
for i in range(0,4) :
  print(i, lowest_eigenvec_est[i])

print("Test of the eigenvector")
print("Eigenvalue   <EigenVec,EigenVec_min_est> ")
for i in range(0, 4) :
   dd_norm = np.vdot(EigenVectors[:,i] , EigenVectors[:,i])
   dd = np.vdot(EigenVectors[:,i] , lowest_eigenvec_est)

   # The eigenvectors from the two methods can be differ by a factor
   # of e^{i theta}. The normalization can be obtained by looking at the
   # ratio of the first elments
   nrm = EigenVectors[0,i] / lowest_eigenvec_est[0]
   tmp = dd / nrm[0]
   dd *= nrm[0]

   print(f"Exact Eigenvalue = {E_true[i].real:.3f}" , f"dot_prod = {dd:.3f}" )
#   print(EigenVectors[:,i])



print("DEBUG" , eig_store )

data = [E_min_true.real , eig_store] 

file_out = "VQE_store.pick"
with open(file_out, "wb") as f:
    pickle.dump(data, f)

print("Written the results to " , file_out)

