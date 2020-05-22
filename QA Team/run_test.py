# %%

import random

from create_testable_matrix 			  import create_testable_matrix
from test_returned_min_polynom 			  import test_returned_min_polynom
from test_returned_characteristic_polynom import test_returned_characteristic_polynom
from test_returned_jordan_chevalier 	  import test_returned_jordan_chevalier
from test_returned_P_Mejardenet_matrix    import test_returned_P_Mejardenet_matrix

def run_test(l):	
	J, size = create_jordan_matrix(l)
	P = create_mejarden_matrix(size)
	A = create_testable_matrix(P, J)
	M = Matrix(A)
	
	if (test_returned_P_Mejardenet_matrix(A, M.getJordanForm(), M.getPmejardent())):
		print("P and J matricies are good")
	else:
		print("P and J matricies are bad")
		return False
	
	if (test_returned_min_polynom(A, M.getCharacteristicPolynomial())):
		print("min polynom is good")
	else:
		print("min polynom is bad")
		return False
	
	if (test_returned_characteristic_polynom(A, M.getMinimalPolynomial())):
		print("characteristic polynom is good")
	else:
		print("characteristic polynom is bad")
		return False
	
	if (test_returned_jordan_chevalier(A, M.getSmatrix(), M.getNmatrix())):
		print("jordan chevalier matricies are good")
	else:
		print("jordan chevalier matricies are bad")
		return False
		
	return True

def random_realNumbers_tests(num_tests):
	for i in range(num_tests):
		# set how many jordan-blocks
		blocks = random.randint(1,5)
		l = []
		for b in range(blocks):
			# eigenvalue to be from range -10 to 10, and jordan block to be in size 1-8
			l.append((random.randint(-10,10), random.randint(1,8)))
		run_test(l)


d = [(1,1),(2,1),(3,1)]
# d = diag with 1,2,3 on the diag
run_test(d)

l = [(1,2),(0,2),(3,1)]
# l = jordan matrix that looks like this:
'''
1 1
0 1
    0 1
	0 0
	    3
'''
run_test(l)

l = [(i,2),(-i,2),(3,3)]
# l = jordan matrix that looks like this:
'''
i 1
0 i
    -i 1
	0  i
	    3 1
		0 3 1
		0 0 3 
'''
run_test(l)

random_realNumbers_tests(50)


