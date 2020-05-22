import numpy as np
import numpy.linalg
from numpy.linalg import matrix_rank

class Matrix():
    def __init__(self, matrix : np.ndarray):
        '''
        :param matrix: ndarray matrix (C)
        '''
        self.matrix = matrix
        self.S = None
        self.N = None

    def __call__(self, *args, **kwargs):
        '''

        :param args:
        :param kwargs:
        :return:
        '''
        J = []
        P = []


        return J, P

    def getCharacteristicPolynomial(self):
        '''

        :return:
        '''
        pass

    def getMinimalPolynomial(self):
        '''

        :return:
        '''
        pass

    def isDiagonalizableMatrix(self):
        '''

        :return:
        '''
        pass

    def getEigenvectors(self):
        '''

        :return:
        '''
        pass

    def getPmejardent(self):
        '''
        Algorith:
            1.check if A is Diagonalizable if so, P = stack_col(eig_vectors)
            2.find eig vectors muhlalim and jordan chains:
                2.1 for each eig value a:
                    2.1.1 find v (eig_vector_muhlal) in ker(A-aI)^k so that,
                          v not in ker(A-aI)^(k-1),
                          when k is the pow of (x-a) in the min_poly
                    2.1.2 find jordan chain for each v ,example jordan chain for v:
                        [v, (A-aI)v, ((A-aI)^2)v,..., ((A-aI)^k-1)v]
                2.2 P = stack_col(all_jordan chains)
        '''

        eig_vectors_map = self.getEigenvectors()

        # if matrix is Diagonalizable, no need to find eig vectors muhlalim
        if self.isDiagonalizableMatrix:
            P = np.zeros_like(self.matrix)
            for i, vector in enumerate(eig_vectors_map):
                P[:, i] = eig_vectors_map[i]
            return P

        char_poly = self.getCharacteristicPolynomial()
        min_poly = self.getMinimalPolynomial()

        # list to store all the jordan chains
        base_list = []

        for eig_value, eig_vectors in eig_vectors_map.items():
            # geometric multiplicity = num of eig_vectros
            r_g = len(eig_vectors)
            # algebraic multiplicity, need to find from char_poly
            r_a = -1
            # the pow of (x-eig_value*I) in the min_poly
            min_poly_pow = -1

            # finding r_a and min_poly_pow
            for row in char_poly:
                if row[0] == eig_value:
                    r_a = row[1]
                    break
            for row in min_poly:
                if row[0] == eig_value:
                    min_poly_pow = row[1]
                    break

            num_of_eig_muhlalim = r_a - r_g
            #finding eig muhlalim
            for _ in range(num_of_eig_muhlalim):
                # A = (matrix - eig_value*I)
                A = (self.matrix - (eig_value * np.eye(self.matrix.shape[0])))
                # A_pow = A^min_poly_pow
                A_pow = np.linalg.matrix_power(A, min_poly_pow)

                # compute A_pow null_space (no pre made function in numpy)
                # null_space = all eig_vectors of eig_value 0
                tmp, eig_muhlalim = np.linalg.eig(A_pow)
                null_space = [vector for i, vector in enumerate(eig_muhlalim) if tmp[i] == 0]
                for vector in null_space:
                    if self.checkIfMuhlal(A, vector, min_poly_pow):
                        self.findJordanChain(A, vector, min_poly_pow, base_list)

        # col_stack all jordan chains
        P = numpy.stack(base_list, axis=1)
        return P

    # check if vector is in ker((matrix - eig_value*I)^(min_poly_pow-1))
    def checkIfMuhlal(self, A, vector, min_poly_pow):
        A_pow_tmp = np.linalg.matrix_power(A, min_poly_pow-1)
        if A_pow_tmp @ vector == 0:
            return False
        return True

    # find jordan chains and add them to base_list
    def findJordanChain(self, A, vector, min_poly_pow, base_list):
        min_poly_pow -= 1
        base_list.append(vector)
        for tmp_pow in range(1, min_poly_pow, 1):
            A_pow_tmp = np.linalg.matrix_power(A, tmp_pow)
            base_list.append(A_pow_tmp @ vector)
        return True

    def getGeoMul(self):
        '''

        :return:
        '''

    @staticmethod
    def combine_jordan_blocks(blocks, size):
        res = np.zeros((size, size))
        i = 0
        for block in blocks:
            for j in range(block[1] - 1):
                res[i + j][i + j] = block[0]
                res[i + j][i + j + 1] = 1

            res[i + block[1] - 1][i + block[1] - 1] = block[0]

            i += block[1]

        return res


    def getJordanForm(self):
        """

        :return: List of tuples
        """

        # ALGORITHM:
        # 1. Get eigenvalue from minimal polynom
        # 2.
        #           Loop1:  for each ev:
        #           Loop2:      for power in range(1 to power_of_ev_in_min_pol):
        #
        #                           # Calculating number of blocks of size power for ev
        #                           num = 2*dimKer(A-lambda*I)^block_size - dimKer(A-lambda*I)^(block_size+1) -
        #                                                                            - dimKer(A-lambda*I)^(block_size-1)
        #                           # Adding blocks to output as tuple of (ev and block size)
        # 3. Construct the output matrix out of Jordan blocks

        result = []
        minimal_pol = self.getMinimalPolynomial()

        for row in minimal_pol:
            ev = row[0]
            n = self.matrix[0].size

            identity_m = np.identity(n)
            tmp_matrix = self.matrix - ev * identity_m  # A - lambda*I

            mat_1 = np.identity(n)
            mat_2 = tmp_matrix
            mat_3 = tmp_matrix @ tmp_matrix

            dim_ker_1 = n - matrix_rank(mat_1)
            dim_ker_2 = n - matrix_rank(mat_2)
            dim_ker_3 = n - matrix_rank(mat_3)

            for block_size in range(1, row[1] + 1):
                number_of_blocks = 2 * dim_ker_2 - dim_ker_1 - dim_ker_3

                # Updates the arguments for the next iteration
                mat_3 = mat_3 @ tmp_matrix
                dim_ker_1 = dim_ker_2
                dim_ker_2 = dim_ker_3
                dim_ker_3 = n - matrix_rank(mat_3)

                [result.append((ev, block_size)) for _ in range(number_of_blocks)]

        return Matrix.combine_jordan_blocks(result, n)

    def getGCD(self):
        '''

        :return:
        '''
        pass

    def getSmatrix(self):
        '''

        :return: returns only the S matrix
        '''
        if self.S is None:
            self.getSNMatrices()
        return self.S

    def getNmatrix(self):
        '''

        :return: returns only the S matrix
        '''
        if self.N is None:
            self.getSNMatrices()
        return self.N

    def getSNMatrices(self):

        '''
        the jordan form A = P*J*(P^-1)
        the jordan form can be decomposed to the sum of two matrices:
        a diagonal matrix, and a nilpotent matrix (J - diag(J))
        A = P*(diagonal + nilpotent)*(P^-1)
        A = P*diagonal*(P^-1) + P*nilpotent*(P^-1)
        S = P*diagonal*(P^-1)
        N = P*nilpotent*(P^-1)

        :return: returns S,N of self.matrix
        '''

        J, P = self.getJordanForm() , self.getPmejardent()
        diag_index = np.array([[i, i] for i in range(J.shape[0])])
        diag_matrix = np.zeros_like(J)
        diag_matrix[diag_index[:, 0], diag_index[:, 1]] = J[diag_index[:, 0], diag_index[:, 1]]
        second_diagonal = np.array([[i, i + 1] for i in range(J.shape[0] - 1)])
        nil_matrix = np.zeros_like(J)
        nil_matrix[second_diagonal[:, 0], second_diagonal[:, 1]] = J[second_diagonal[:, 0], second_diagonal[:, 1]]
        P_inv = np.linalg.inv(P)
        self.S = P.dot(diag_matrix).dot(P_inv)
        self.N = P.dot(nil_matrix).dot(P_inv)
        return self.S, self.N


if __name__ == '__main__':
    '''
    Can do here some plays
    '''
    numpy_matrix = np.ndarray([1,2,3])

    mat = Matrix(numpy_matrix)
    J, P = mat() # will call "__call__"
    pass


