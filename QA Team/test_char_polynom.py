#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
from numpy import matrix
from numpy import linalg as LA
"""
get a charachteristic polynom and a matrix
returns true\false if the polynom is correct.
"""

def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x

def test_returned_characteristic_polynom(polynom,matrix):
    eignValues = LA.eigvals(matrix)
    coefArray = polynom.coef
    polyRoots = np.polyroots(polynom)
    
    if(len(polyRoots) != len(eignValues)) : return False
    
    arr1 = selection_sort(eignValus)
    arr2 = selection_sort(polyRoots)
    
    if(arr1 != arr2):
        return False
    
    return True


"""
# In[0]:
a = np.matrix('1 0; 0 1')
p = (1,-2,1)
test_returned_characteristic_polynom(p,a)
# In[ ]:
"""



