import numpy

def run_test():
	l = [(1,1),(2,1),(3,1)]
	# l = diag with 1,2,3 on the diag
	J, size = create_jordan_matrix(l)
	P = create_mejarden_matrix(size)
	A = create_testable_matrix(P, J)
	M = Matrix(A)
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
