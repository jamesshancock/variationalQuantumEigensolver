#  Power method for an arbitrary matrix
#
#

import numpy as np
import pickle

def power_method(A,b,N, sss=0):
    """
    Compute the smallest eigenvalue by the power method
    """
    store = [] 
    for k in range(N):
        top = np.dot(A,b)
        bottom = np.linalg.norm(top,1)
        b = top/bottom 
        lambda_k = RQ(A,b) - sss
        store.append(lambda_k[0][0].real)
#        print(k, lambda_k.real)
    return top/bottom, store
    
def RQ(A,b):
    """
     lambda = <b, A b> / <b,b>
     return lambda
    """
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

print("H=", H)


E_true, EigenVectors = np.linalg.eig(H)
E_min_true = min(E_true)
print("Minimum eigenvalue from dense method", E_min_true.real)
E_true = sorted(E_true)
for i,ee in enumerate(E_true) :
   print("Eig[",i, "]=",  ee)


dim = H.shape[0]
con = -5
for i in range(0,dim) :
  H[i,i] += con



b = np.random.rand(dim,1)
bN, eig_iter = power_method(H,b,50, con)
lambdaN = RQ(H,bN) - con
print("Power estimate of lowest eigenvalue", lambdaN)

##print(eig_iter)

data = [E_min_true.real , eig_iter] 

file_out = "power_store.pick"
with open(file_out, "wb") as f:
    pickle.dump(data, f)

print("Written the results to " , file_out)


