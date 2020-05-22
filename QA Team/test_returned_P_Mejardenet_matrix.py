import numpy as np
from numpy import matrix


def almost_equal(a, b, threshold=0.01):
    c = a - b
    if (c < threshold and c > -threshold).all:
        return True
    else:
        return False


def has_jordan_form(j):
    m,n = j.shape
    if m != n:
        return False

    for x in 0..m:
        for y in 0..n:
            if x > y and j.item((x, y)) != 0:
                return False
            if x + 1 < y and j.item((x, y)) != 0:
                return False
    for x in 0..m-1:
        if j.item((x, x)) != j.item((x+1, x+1)):
            if j.item((x, x+1)) != 0:
                return False
        elif j.item((x, x+1)) != 1 and j.item((x, x+1)) != 0:
            return False
    return True


def test_returned_P_Mejardenet_matrix(a, j, p):
    b = p.I @ a @ p
    if almost_equal(b, j) and has_jordan_form(j):
        return True
    return False
