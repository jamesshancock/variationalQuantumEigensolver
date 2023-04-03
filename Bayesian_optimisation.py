import math
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from warnings import catch_warnings
from warnings import simplefilter
from scipy.stats import norm
from numpy.random import random
import cmath
import scipy
from scipy import linalg
from scipy.optimize import minimize
from numpy import random
import qiskit
from qiskit import *
from qiskit import Aer
import qiskit.opflow

pauli_0 = np.array([[1, 0],
                   [0, 1]])
pauli_1 = np.array([[0, 1],
                   [1, 0]])
pauli_2 = np.array([[0, -1j],
                   [1j, 0]])
pauli_3 = np.array([[1, 0],
                   [0, -1]])
pauli = [pauli_0,pauli_1,pauli_2,pauli_3]

def Hf(h):
    H = np.zeros((2**n,2**n))
    K = int(len(h)/2)
    Coef = []
    matri = []
    for co in range(K):
        Coef.append(float(h[2*co]))
        matri.append([int(h[2*co+1][j]) for j in range(n)])
    for kj in range(K):
        H_temp = np.array([1])
        for km in range(n):
            H_temp = np.kron(H_temp,pauli[matri[kj][km]])
        H = H + Coef[kj]*H_temp
    return H, Coef, matri

def ryBlock(paray,QC):
    for ky in range(len(paray)):
        QC.ry(paray[ky],ky)
    return QC

def pBlock(varp,QC):
    for kp in range(len(varp)):
        QC.p(varp[kp],kp)
    return QC

def cnotBlock(QC):
    for kc in range(n-1):
        QC.cx(kc,kc+1)
    return QC

def measure_0(qubit,QC): #function used to measure in the sigma_1 basis
    return QC
        
def measure_1(qubit,QC): #function used to measure in the sigma_1 basis
    QC.h(qubit)
    QC.measure(qubit,qubit)
    return QC
    
def measure_2(qubit,QC): #function used to measure in the sigma_2 basis
    QC.sdg(qubit)
    QC.h(qubit)
    QC.measure(qubit,qubit)
    return QC
    
def measure_3(qubit,QC): #function used to measure in the sigma_3 basis
    QC.measure(qubit,qubit)
    return QC

def measure(case,qubit,QC):
    if case == 1:
        measure_1(qubit,QC)
    elif case == 2:
        measure_2(qubit,QC)
    elif case == 0:
        measure_0(qubit, QC)
    else:
        measure_3(qubit,QC)
    return QC

def S(sn):
    S = 0    
    while S*sn < 2**(sn+1):
        S = S + 1
    L = []
    for kl in range(S):
        if (kl % 2) == 0:
            L.append(0)
        else:
            L.append(1)
    return S, L

def para(var,QC):
    for kPara in range(S1):
        paras = var[kPara*n:(kPara+1)*n]
        if L[kPara] == 0:
            ryBlock(paras,QC)
        else:
            pBlock(paras,QC)
            cnotBlock(QC)
    S2 = S1 + 1
    paras = var[S1*n:(S2)*n]
    ryBlock(paras,QC)
    return QC

def binary_list(nb):
    b = ['{:0{}b}'.format(i, nb) for i in range(nb*nb-1)]
    if (nb%2) == 0:
        x = ''
        for k in range(nb):
            x = x + '1'
        b.append(x)
    return b

def EV(counts,shot,M): #needs rewriting
    b = binary_list(n)
    signlist = {}
    for x1 in b:
        c1 = x1.count('1')
        if (c1%2) == 0:
            signlist[x1] = 1
        else:
            signlist[x1] = -1
    counts_store = []
    for x in b:
        if x in counts:
            temp = counts[x]/shot
            counts_store.append(signlist[x]*temp)
    EV = sum(counts_store)
    
    return EV
                         
def sim(QC, shot):
    backend_sim = Aer.get_backend('qasm_simulator')
    job_sim = backend_sim.run(transpile(QC, backend_sim), shots=shot)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(QC)
    return counts

def VQE(Var,h,shot):    
    H, Coef, matri = Hf(h)
    GS = 0
    n = len(h[1])
    q = QuantumRegister(n)
    c = ClassicalRegister(n)
    for k_coef in range(len(Coef)):
        if matri[k_coef] == [0]*n:
            EV1 = 1
        else:
            EV1 = 0
            circ = QuantumCircuit(q,c)
            para(Var,circ)
            for g in range(n):
                M = matri[k_coef][g]
                measure(M,g,circ)
            numbers = sim(circ,shot)
            
            EV1 = EV(numbers,shot,matri[k_coef])
        GS = GS + Coef[k_coef]*EV1
    return GS

def hmaker(n,K):
    H = []
    for k in range(K):
        h = str(round(random.uniform(-10,10),1))
        if h == '0.0' or '-0.0':
            h = str(round(random.uniform(-10,10),1))
        g = ''
        for i in range(n):
            g = g+str(random.randint(0,4))
        H.append(h)
        H.append(g)
    return H

h = ['1.4','102', '-3.3', '123', '9.0', '012', '-2.7', '221', '7.0', '222']
H, Coef, matri = Hf(h)
E_true, EigenVectors = np.linalg.eig(H)
E_min_true = min(E_true.real)

shot = 16
shot0 = 3000
global n
n = len(h[1])
global S1
global L
S1, L = S(n)
npara = S1*n + n

X = []
samples = 100
for J in range(1,samples+1):
    xtemp = []
    for kx in range(npara):
        xtemp.append(2*math.pi*random.uniform(0,1))
    X.append(xtemp)
Y = []
N = len(X)
for k in range(N):
    Y.append(VQE(X[k],h,shot))

model = GaussianProcessRegressor(ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed"))
model.fit(X,Y)

def kernel(x,xprime,hyper):
    t = 0
    for k in range(len(x)):
        t = t - 2/(hyper[1]**2)*math.sin((x[k]-xprime[k])/2)
    kern = (hyper[0]**2)*math.exp(t)

def surrogate(model, x):
    with catch_warnings():
        simplefilter("ignore")
    return model.predict(x, return_std=True)

def EI(X,Xsamples,model):
    ei = []
    Yhat, std = surrogate(model,X)
    mu_best = min(Yhat)
    mu, std = surrogate(model,Xsamples)
    mu = mu[:]
    probs = []   
    K = 4
    Ei = []
    for k in range(len(mu)):
        probs.append(max(0,mu_best-mu[k]))     
    if len(model_store) > 4:
        for Kc in range(K):
            if Kc == 0: 
                ei.append(probs)
            else:
                probs = []
                randk = random.randint(0,len(model_store))
                model0 = model_store[randk]
                mu, std = surrogate(model0,Xsamples)
                for k0 in range(len(mu)):
                    probs.append(max(0,mu_best-mu[k0])) 
                ei.append(probs)
        for i in range(K):
            temp = 0
            for j in range(len(ei)):
                temp = temp + (1/K)*ei[i][j]
            Ei.append(temp)
    else:
        Ei = probs
    return Ei

def opt_sample(X, y, model):
    Xsamples = []
    for kx in range(100):
        Xsamples.append(np.array([2*math.pi*random.uniform(0,1)]*(npara)))
    scores = EI(X, Xsamples, model)
    ix = np.argmin(scores) #finds the most optimum value to sample next
    return Xsamples[ix]

global model_store
model_store = []
BITS = 600
for i in range(BITS):
    x_opt = opt_sample(X, Y, model)
    actual = VQE(x_opt,h,shot)
    est, _ = surrogate(model, [x_opt])
    X = np.vstack((X, [x_opt]))
    Y.append(actual)
    model.fit(X, Y)
    model_store.append(model)
xi = np.argmin(Y)
guess = X[xi]
E_min = minimize(VQE, guess, args = (h,shot0), method='powell',
                       options={'ftol': 1e-15, 'disp': False, 'maxfev': 10**4, 'return_all': True})
Est = E_min.fun
E_true = round(E_min_true,6)
print('=====================================')
Estimate = round(float(Est),6)
print('True = '+str(E_true))
print('Bayesian minimum = '+str(VQE(X[xi],h,1000)))
print('Final estimate = '+str(Estimate))
error = abs(E_true-Estimate)
print('Error = '+str(round(error,6)))
print('Relative error = '+str(round(abs(error/E_true),6)))
print('Iterations of BO = '+str(BITS))
print('Iterations of Powell = '+str(E_min.nit))
shots_total = BITS*shot + E_min.nfev*n*shot0
print('Total number of shots = '+str(shots_total))
print('=====================================')
