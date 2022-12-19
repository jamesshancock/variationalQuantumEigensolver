#  Example for the Rosenbrock test function
#  https://en.wikipedia.org/wiki/Rosenbrock_function#Multidimensional_generalisations
# 
# Use jax to automaatically differentiate the python function
# https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
#

from scipy.optimize import minimize, check_grad
from jax import grad

def my_rosen(X): #Rosenbrock function
    return (1.0 - X[0])**2 + 100.0 * (X[1] - X[0]**2)**2 + \
           (1.0 - X[1])**2 + 100.0 * (X[2] - X[1]**2)**2

# Use jax to compute the derivative of the function
my_rosen_deriv = grad(my_rosen)

print("Check the differentiation")
ck = check_grad(my_rosen, my_rosen_deriv, [2, -1, 0.5])
print("Check optimization " , ck)

# Initial guess
x0 = [1.3, 0.7, 0.8 ]


alg_list  = [ 'Nelder-Mead' , 'CG' ] 


print("No derivative")
for alg_ in alg_list :
  res = minimize(my_rosen, x0, method=alg_, tol=1e-6)

  print("Algorthm = " , alg_ )
  print("Iterations = " , res.nit)
  print("Function evaluations = " , res.nfev)
  print("Solutons, x = " , res.x )
  print(" ") 

print("Use derivatives")

for alg_ in [ "CG" ] :
  res = minimize(my_rosen, x0, method=alg_, tol=1e-6, jac=my_rosen_deriv )

  print("Algorthm = " , alg_ )
  print("Iterations = " , res.nit)
  print("Function evaluations = " , res.nfev)
  print("Solutons, x = " , res.x )
  print(" ")
