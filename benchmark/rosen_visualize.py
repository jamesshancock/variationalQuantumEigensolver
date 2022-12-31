
#  Example for the Rosenbrock test function
#  https://en.wikipedia.org/wiki/Rosenbrock_function#Multidimensional_generalisations
# 
#  Explore the callback function
#  https://stackoverflow.com/questions/16739065/how-to-display-progress-of-scipy-optimize-function
#
#  This script examines the variables as the optinization algorithm
#  works.
#

#alg_  = 'Nelder-Mead' 
alg_  = 'Powell' 
use_mds = True

from scipy.optimize import minimize, rosen, rosen_der
from sklearn.manifold import MDS


import matplotlib.pyplot as plt
import numpy as np
import sys

x0 = [1.3, 0.75, 0.8, 1.9, 1.2]
seed = np.random.RandomState(seed=3)


Nfeval = 1

store = [] 
opt = []
store.append(x0)

def callbackF(Xi):
    global Nfeval, store_x, store_y
    ff = rosen(Xi)
    print("Iter", Nfeval, Xi, "func=" , ff)
    Nfeval += 1
    store.append( Xi  ) 
    opt.append(ff)

print("No derivative")
res = minimize(rosen, x0, method=alg_, tol=1e-6, callback=callbackF)

print("Algorthm = " , alg_ )
print("Iterations = " , res.nit)
print("Function evaluations = " , res.nfev)
print("Solutions, x = " , res.x )
print(" ") 

#
# reformat the solutions
#
dim = len(store) 
novar = len(x0)

#
#  project down into 2 dimensions
#
# Use Multi-dimensional Scaling (MDS) to project the
# data to a 2D space.
# https://scikit-learn.org/stable/modules/manifold.html#tips-on-practical-use

X = np.zeros((dim,novar))
for i in range(0,dim):
  tmp_ = store[i]
  for j in range(0,novar):
    X[i,j] = tmp_[j]

embedding = MDS(n_components=2,n_init=10)


X_init = np.full((dim,2),0.25)

# Fix the random seed, otherwise the matrix is randomly set
np.random.seed(0)
X_init = np.random.rand(dim,2)

X_transformed = embedding.fit_transform(X)
#X_transformed = embedding.fit_transform(X,init = X_init)
print("transform shape " , X_transformed.shape )

#
#  plot the solution
#


if use_mds :
   XX = X_transformed
else:
  XX = X

#  To plot the trajectory use the arrow function from matplotlib.
#  The function needs the start of the arrow and the direction.
#
for i in range(0,dim-1):
 x_dir = XX[i+1,0] - XX[i,0] 
 y_dir = XX[i+1,1] - XX[i,1] 
 plt.arrow(XX[i,0], XX[i,1] , x_dir , y_dir, head_width=0.01, head_length=0.01, fc='r', ec='r')



plt.title("Rosenbock cost function using " + alg_ + " algorithm")

if use_mds :
  plt.xlabel(r"$x_1$ (MDS)")
  plt.ylabel(r"$x_2$ (MDS)")
else:
  plt.xlabel(r"$x_1$")
  plt.ylabel(r"$x_2$")

plt.plot(XX[0,0] , XX[0,1]  , "go" , label = "Initial value")
plt.plot(XX[-1,0] , XX[-1,1]  , "bo" , label = "Final value")

plt.legend()

if use_mds :
  l_ = 0.9 
  plt.xlim(-1*l_,l_)
  plt.ylim(-1*l_,l_)
else:
  plt.xlim(0.7, 1.4)
  plt.ylim(0.7, 1.2)

print("Solution MDS ", XX[-1,0] , XX[-1,1])

if use_mds :
  fname = "MDS_Rosen_" + alg_ + ".png"
else: 
  fname = "Rosen_" + alg_ + ".png"

plt.savefig(fname)
print("The plot has been written to " , fname)

plt.show()

iter_opt = [i for i in range(0,len(opt) ) ] 

plt.plot(iter_opt , opt)
plt.xlabel("i")
plt.ylabel(r"$f_{obj}$")
plt.yscale("log")
plt.savefig("Rosen_" + alg_ +  "f.png")
plt.show()
