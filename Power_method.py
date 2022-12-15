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

pauli_0 = np.array([[1, 0],
                   [0, 1]])
pauli_1 = np.array([[0, 1],
                   [1, 0]])
pauli_2 = np.array([[0, -1j],
                   [1j, 0]])
pauli_3 = np.array([[1, 0],
                   [0, -1]])

H = 5/2*pauli_0 + -1/2*pauli_3
E_true, EigenVectors = np.linalg.eig(H)
E_min_true = min(E_true)
print(E_min_true.real)

b = np.random.rand(2,1)
bN = power_method(H,b,300)
lambdaN = RQ(H,bN)
print(lambdaN)

