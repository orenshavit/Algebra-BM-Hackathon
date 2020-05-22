import numpy as np
import numpy.linalg
import itertools

class Matrix():
    
    def __init__(self, matrix : np.ndarray):
        '''
        :param matrix: ndarray matrix (C)
        '''
        self.matrix = matrix
        self.eig_val,_ = np.linalg.eig(self.matrix)
        self.charPoly = self.getCharacteristicPolynomial()
        self.minPoly = None
        self.isDiagonal = None

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
        unique_elements, counts_elements = np.unique(self.eig_val, return_counts=True)
        charPoly = np.array([[unique_elements],[counts_elements]])
        return charPoly.transpose()

    def getMinimalPolynomial(self):
        '''
        
        :return:
        '''
        
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

        :return:
        '''
        pass

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

        :return:
        '''
        pass

    def getNmatrix(self):
        '''

        :return:
        '''
        pass

if __name__ == '__main__':
    '''
    Can do here some plays
    '''
    numpy_matrix = np.ndarray([1,2,3])

    mat = Matrix(numpy_matrix)
    J, P = mat() # will call "__call__"
    pass


