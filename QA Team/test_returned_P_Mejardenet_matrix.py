import numpy as np
from numpy import matrix

def almost_equal(a,b,threshold=0.01):
    c=a-b
    if (c<threshold and c>-threshold).all:
        return true
    else:
        return false
        
def test_returned_P_Mejardenet_matrix(a, j, p):
    b=p.I @ a @ p
    if almost_equal(b,j) and hasJordanForm(j):
        return true
    return false




        
        