#  Power method for an arbitrary matrix
#
#

import numpy as np
def power_method(A,b,N):
    for k in range(N):
        top = np.dot(A,b)
        bottom = np.linalg.norm(top,1)
        b = top/bottom 
    return top/bottom
    
def RQ(A,b):
    b_star = np.transpose(np.conj(b))
    one = np.dot(A,b)
    top = np.dot(b_star,one)
    bottom = np.dot(b_star,b)
    return top/bottom


# Load the matrix from file

filename = "H_store_A.npy"

try:
  H = np.load(filename)
  print("Loaded the matrix from ", filename)
except:
  print("Error loading matrix from " , filename)  
  sys.exit(0)

print("Size of matrix " , H.size)
print("Shape of matrix " , H.shape)

E_true, EigenVectors = np.linalg.eig(H)
E_min_true = min(E_true)
print(E_min_true.real)

dim = H.shape[0]
b = np.random.rand(dim,1)
bN = power_method(H,b,300)
lambdaN = RQ(H,bN)
print(lambdaN)

