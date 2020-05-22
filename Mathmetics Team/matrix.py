import numpy as np
import numpy.linalg

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
        # what getEigenvectors returs?
        eag_vectors_map = self.getEigenvectors()
        eag_muhlalim_map = []
        P = np.zeros_like(self)
        if self.isDiagonalizableMatrix:
            for i, vector in enumerate(eag_vectors_map):
                P[:, i] = eag_vectors_map[i]
            return P

        char_poly = self.getCharacteristicPolynomial()
        base_list = []
        for eag_value, eag_vectors in eag_vectors_map.items():
            r_g = len(eag_vectors)
            r_a = -1
            min_poly_pow = -1
            for line in char_poly:
                if line[0] == eag_value:
                    r_a = line[1]
                    break
            for line in min_poly:
                if line[0] == eag_value:
                    min_poly_pow = line[1]
                    break
            num_of_eag_muhlalim = r_g - r_a
            min_poly = self.getMinimalPolynomial()
            while num_of_eag_muhlalim > 0:
                # pow should be power from min_poly
                A = (self.matrix - (eag_value * np.eye(self.matrix.shape[0])))
                A_pow = np.linalg.matrix_power(A, min_poly_pow)
                tmp, eag_muhlalim = np.linalg.eig(A_pow)
                for vector in eag_muhlalim:
                    if self.checkIfMuhlal(A, vector, min_poly_pow):
                        self.findJordanChain(A, vector, min_poly_pow, base_list)
                num_of_eag_muhlalim -= 1
        P = numpy.stack(base_list, axis=1)
        return P


    def checkIfMuhlal(self, A, vector, min_poly_pow):
        min_poly_pow -= 1
        while min_poly_pow > 0:
            A_pow_tmp = np.linalg.matrix_power(A, min_poly_pow)
            if A_pow_tmp @ vector == 0:
                return False
            min_poly_pow -= 1
        return True

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

    def getJordanForm(self):
        '''

        :return:
        '''

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


