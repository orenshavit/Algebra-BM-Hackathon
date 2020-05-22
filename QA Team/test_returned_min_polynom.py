#!/usr/bin/env python
# coding: utf-8

import numpy.polynomial.polynomial as np
import scipy.linalg as la

#not working
#calc the poly with the matrix as x
def mat_pol_solve (arr_coef, matrix):
    poly_of_matirces = []
    for i in len(arr_coef):
        temp = matrix
        for j in range(0,i):
            temp = temp@matrix
        poly_of_matirces[i] = temp*arr_coef[i]
    return poly_of_matirces.sum()
        

#checks if the poly is the right minimal poly of the matrix
def test_returned_min_polynom(matrix, poly):
    arr_coef = poly.coef #arr of the coef of the matrix
    
    #checks if the leading coef is 1
    if(arr_coef[sizeof(arr_coef)-1] != 1):
        return false
    
    #if the poly does not become 0 with the matrix as x than it isnt the minimal poly
    if(mat_pol_solve(arr_coef, matrix) != 0):
        return false
    
    arr_mat_eigenvals = la.eig(matrix)
    arr_poly_roots = np.polyroots(poly)
    
    #checks if all the eigenvalues are roots of the poly and the poly has no other roots
    j=0
    for i in len(arr_poly_roots):
        if(j >= len(arr_mat_eigenvals)): return false #means that there are more roots than eigenvals
        while(j+1 < len(arr_mat_eigenvals) and arr_mat_eigenvals[j] == arr_mat_eigenvals[j+1]): #skips the same eigenvals
            j+=1
        if(arr_mat_eigenvals[j] != arr_poly_roots[i]): # means that there is a diff between eigenvals and roots
            return false
        j+=1
    if(j < len(arr_mat_eigenvals)-1): #means there are eigenvals that are not roots
        return false
    
    #checks if the poly is minimal
    for i in len(arr_mat_eigenvals):
        while(i+1 < len(arr_mat_eigenvals) and arr_mat_eigenvals[i] == arr_mat_eigenvals[i+1]): #skips the same eigenvals
            i+=1
        if(mat_pol_solve(np.polydiv(arr_coef, (-arr_mat_eigenvals[i], 1)), matrix) == 0):
            return false
    
    return true
        
