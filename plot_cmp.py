\
import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np
import math

file_in = "power_store.pick"
with open(file_in, "rb") as f:
    data = pickle.load(f)

print("Loading data from " , file_in )

eig_exact = data[0]
eig_iter  = data[1]

print(eig_iter)

print("Dense eigenvalue = ", eig_exact)

iter_ = np.array(range(0, len(eig_iter) ))

eig_iter_ = [ math.fabs((x-eig_exact)/eig_exact) for x in eig_iter ]

plt.plot(iter_, eig_iter_, label= "power")

plt.xlabel("i")
plt.ylabel(r"$\mid (\lambda_i - \lambda_i^{exact}) / \lambda_i^{exact} \mid$")
plt.yscale('log')

plt.legend()

plt.show()
