import numpy as np
# Hopfield Network Implementation

# activation function is a simple step function or sign function
# with threshold 0
def activation_function(x):
    threshold=0
    if x < threshold:
        return -1
    return 1

class Matrix():

    @staticmethod
    def matrix_vector_multiplication(matrix, vector):
        #The output vector will have the same dimension as the input vector
        return matrix.dot(vector)

    @staticmethod
    def clear_diagonal(matrix):
        #set the diagonal elements to zero in order to construct a symmetric matrix
        #and to signify that no neuron element is connected to itself (zeroes along the matrix main diagonal)
        np.fill_diagonal(matrix, 0)
        return matrix

    @staticmethod
    def outer_product(pattern):
        return np.outer(pattern,pattern)

    @staticmethod
    def add_matrices(matrix1, matrix2):
        return matrix1 + matrix2

class HopfieldNetwork():

    def __init__(self, dimension):
        self.weight_matrix = np.zeros((dimension, dimension))

    def train(self, pattern):
        #step 1: transform the pattern into bipolar format (1 or -1)
        pattern_bipolar = HopfieldNetwork.transform(pattern)
        #step 2: Get a matrix out of the 1-dimensional array using the outer product
        pattern_weight_matrix = Matrix.outer_product(pattern_bipolar)
        print("(training) Pattern Weight Matrix: ",pattern_weight_matrix)
        #step 3: Clear (reinitialize to zeroes) the diagonal values of the matrix to create a symmetric matrix
        pattern_weight_matrix = Matrix.clear_diagonal(pattern_weight_matrix)
        print("(training) Pattern Weight Matrix Made Symmetric and Zeroed along Diagonal: ",pattern_weight_matrix)
        #step 4: Add this new matrix to the original weight matrix ... why ?
        self.weight_matrix = Matrix.add_matrices(self.weight_matrix, pattern_weight_matrix)
        print("(training) Weight Matrices added: ",self.weight_matrix)

    def recall(self, pattern):
        # we now recall the pattern that has been embedded within the matrix using the prior encoding steps
        #1st: encode out of bipolar format
        pattern_bipolar = HopfieldNetwork.transform(pattern)
        print("Bipolar Pattern: ",pattern_bipolar)
        result = Matrix.matrix_vector_multiplication(self.weight_matrix, pattern_bipolar)
        print("Result of Matrix Vector Multiplication: ",result)
        result = np.array([activation_function(x) for x in result])
        result = HopfieldNetwork.re_transform(result)
        print("Result of Recall: ",result)

    @staticmethod
    def transform(pattern):
        return np.where(pattern == 0,-1,pattern)

    @staticmethod
    def re_transform(pattern):
        return np.where(pattern == -1,0,pattern)


if __name__ == '__main__':
    network = HopfieldNetwork(9)
    network.train(np.array([1,1,1,1,0,0,1,1,1]))
    network.train(np.array([1,1,1,0,1,0,0,1,0]))
    network.recall(np.array([1,0,1,0,1,0,0,1,0]))
