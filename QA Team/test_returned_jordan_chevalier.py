import numpy as np
def is_diagnosiable(M):
    values, vectors = np.linalg.eig(np.array(M))
    if len(vectors) != len(M):
        print("S not diagoniasble")
        return False
    return True
def is_nilpotent(M):
    values = np.linalg.eigvals(np.array(M))
    for i in values:
        if i != 0:
            return False
    return True
def test_returned_jordan_chevallier(T, S, N):
    shape_s = S.shape
    shape_n = N.shape
    shape_t = T.shape
    if shape_t != shape_s:
        print("Unmatching dimensions of T,S")
        return False
    if shape_s != shape_n:
        print("Unmatching dimensions of S,N")
        return False
    if shape_t != shape_n:
        print("Unmatching dimensions of T,N")
        return False
    if shape_s[0] != shape_s[1]:
        print("T not square matrix")
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
    if not is_diagnosiable(S):
        print("S is not diagonal")
        return False
    if not is_nilpotent(N):
        print("N is not Nilpotent")
        return False
    return True

