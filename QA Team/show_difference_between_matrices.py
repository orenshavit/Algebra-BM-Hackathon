import numpy

def show_difference_between_matrices(matrix1, matrix2):
    size = len(matrix1);
    if size != len(matrix2):
        print("matrix are not the same size")
    print("matrix 1:")
    print(matrix1)
    print("matrix 2:")
    print(matrix2)
    print("diff:")
    for i in range(size):
        for j in range(size):
            if matrix1[i, j] != matrix2[i, j]:
                print('X', ' ', end='')
            else:
                print(matrix1[i, j], ' ', end='')
        print('')