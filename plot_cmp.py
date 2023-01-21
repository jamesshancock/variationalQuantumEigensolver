\
import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np
import math


def load_eigen(file_in) :


  with open(file_in, "rb") as f:
      data = pickle.load(f)

  print("Loading data from " , file_in )

  eig_exact = data[0]
  eig_iter  = data[1]

#  print(eig_iter)
  print("Dense eigenvalue = ", eig_exact)

  iter_ = np.array(range(0, len(eig_iter) ))

  eig_iter_ = [ math.fabs((x-eig_exact)/eig_exact) for x in eig_iter ]

  return iter_, eig_iter_, eig_exact


# ----------------------------------------


p_iter, p_eig_iter, p_eig_exact = load_eigen("power_store.pick") 
plt.plot(p_iter, p_eig_iter, label= "power")

for alg_ in [ "BFGS" , "CG" , "Nelder-Mead" , "Powell" ]  :
  vqe_iter, vqe_eig_iter, vqe_eig_exact = load_eigen("VQE_" + alg_ +  "_store.pick") 
  plt.plot(vqe_iter, vqe_eig_iter, label= "VQE, " + alg_ )


plt.title("Error on eigenvalue with iteration for 4x4 matrix")
plt.xlabel("i")
plt.ylabel(r"$\mid (\lambda_i - \lambda_i^{exact}) / \lambda_i^{exact} \mid$")
plt.yscale('log')

plt.legend()
plt.savefig("power_Vs_VQE_4x4.png")

plt.show()
