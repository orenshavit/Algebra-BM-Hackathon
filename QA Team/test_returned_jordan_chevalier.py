import numpy as np
def isDiagnosiable(M):
    values, vectors = np.linalg.eig(np.array(M))
    if len(vectors) != len(M):
        print("S not diagoniasble")
        return False
    return True
def test_returned_jordan_chevallier (T,S,N):
    shape_S = S.shape
    shape_N = N.shape
    shape_T = T.shape
    if shape_T != shape_S :
        print("Unmatching dimensions of T,S")
        return False
    if shape_S != shape_N :
        print("Unmatching dimensions of S,N")
        return False
    if shape_T != shape_N:
        print("Unmatching dimensions of T,N")
        return False
    X = np.add(S, N)
    for i in range(len(S)):
        for j in range(len(S)):
            if X[i, j] != T[i, j]:
                print("N+S != T")
                return False
    A = N.dot(S)
    B = S.dot(N)
    for i in range(len(A)):
        for j in range(len(B)):
            if X[i, j] != T[i, j]:
                print("NS != SN")
                return False
    if not isDiagnosiable(S):
        print("S is not diagonal")

    return True
T = np.matrix('1 1;0 1')
S = np.matrix('1 0;0 1')
N = np.matrix('0 1;0 0')
test_returned_jordan_chevallier(T, S,  N)
