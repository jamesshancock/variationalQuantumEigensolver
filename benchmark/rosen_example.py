#  Example for the Rosenbrock test function
#  https://en.wikipedia.org/wiki/Rosenbrock_function#Multidimensional_generalisations
# 
#  Try a variety of algorithms
#
#

from scipy.optimize import minimize, rosen, rosen_der


x0 = [1.3, 0.7, 0.8, 1.9, 1.2]


alg_list  = [ 'Nelder-Mead' , "Powell" , "CG" , "BFGS" ] 

print("No derivative")
for alg_ in alg_list :
  res = minimize(rosen, x0, method=alg_, tol=1e-6)

  print("Algorthm = " , alg_ )
  print("Iterations = " , res.nit)
  print("Function evaluations = " , res.nfev)
  print("Solutons, x = " , res.x )
  print(" ") 


#
#
#
print("Include derivative ")

alg_list_with_deriv  = [ "CG" , "BFGS" ] 

for alg_ in alg_list_with_deriv :
  res = minimize(rosen, x0, method=alg_, jac=rosen_der, tol=1e-6)

  print("Algorthm (with derivative) = " , alg_ )
  print("Iterations = " , res.nit)
  print("Function evaluations = " , res.nfev)
  print("Solutons, x = " , res.x )
  print(" ") 
