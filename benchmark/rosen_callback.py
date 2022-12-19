#  Example for the Rosenbrock test function
#  https://en.wikipedia.org/wiki/Rosenbrock_function#Multidimensional_generalisations
# 
#  Explore the callback function
#  https://stackoverflow.com/questions/16739065/how-to-display-progress-of-scipy-optimize-function
#

from scipy.optimize import minimize, rosen, rosen_der


x0 = [1.3, 0.7, 0.8, 1.9, 1.2]

Nfeval = 1

def callbackF(Xi):
    global Nfeval
    ff = rosen(Xi)
    print("Iter", Nfeval, Xi, "func=" , ff)
    Nfeval += 1

alg_list  = [ 'Nelder-Mead' ] 


print("No derivative")
for alg_ in alg_list :
  res = minimize(rosen, x0, method=alg_, tol=1e-6, callback=callbackF)

  print("Algorthm = " , alg_ )
  print("Iterations = " , res.nit)
  print("Function evaluations = " , res.nfev)
  print("Solutons, x = " , res.x )
  print(" ") 

