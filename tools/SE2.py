# Importing necessary libraries
import numpy as np
import scipy.linalg

# Implementation of the Special Euclidean Group SE(2) class
class SE2:
    def __init__(self, vector=np.array([0, 0, 0])):
        vector = np.array(vector)
        self.setxytheta(vector)

    def __repr__(self):
        return f"{self.vec_SE2()}"

    def __call__(self):
        return self.vec_SE2()

    def __mul__(self, other):
        # Multiplication operation for SE2 group
        SE2_matrix = self.mat_SE2()
        se2_matrix = SE2.SE2_to_se2(SE2_matrix)
        ret_matrix = SE2.se2_to_SE2(se2_matrix * other)
        ret = SE2(ret_matrix)
        return ret

    def __matmul__(self, other):
        # Matrix multiplication operation for SE2 group
        ret = SE2(self.mat_SE2() @ other.mat_SE2())
        return ret

    def __rmatmul__(self, other):
        # Right matrix multiplication operation for SE2 group
        ret = SE2(other.mat_SE2() @ self.mat_SE2())
        return ret

    def __pow__(self, other):
        # Exponentiation operation for SE2 group
        ret = SE2(np.linalg.matrix_power(self.mat_SE2(), other))
        return ret

    def setxytheta(self, value):
        # Setter method for setting x, y, and theta values
        if value.ndim == 1:
            self.__x = value[0]
            self.__y = value[1]
            self.__theta = value[2]
        elif value.ndim == 2:
            self.__x = value[0, 2]
            self.__y = value[1, 2]
            self.__theta = np.arctan2(value[1, 0], value[0, 0])

    def mat_SE2(self):
        # Returns the SE2 matrix representation of the SE2 instance
        cos_theta = np.cos(self.__theta)
        sin_theta = np.sin(self.__theta)

        matrix = np.zeros([3, 3])
        matrix[0, 0] = cos_theta
        matrix[0, 1] = -sin_theta
        matrix[0, 2] = self.__x
        matrix[1, 0] = sin_theta
        matrix[1, 1] = cos_theta
        matrix[1, 2] = self.__y
        matrix[2, 2] = 1

        return matrix

    def vec_SE2(self, ndim=0):
        # Returns the vector representation of the SE2 instance
        vector = np.array([self.__x, self.__y, self.__theta])

        if ndim == 1:
            vector = np.reshape(vector, [3, 1])

        return vector
    
    def vec_se2(self, ndim=0):
        # Returns the vector representation of the SE2 instance
        vector = np.array([self.__x, self.__y, self.__theta])
        vector = SE2.SE2_to_se2(vector)

        if ndim == 1:
            vector = np.reshape(vector, [3, 1])

        return vector    

    @staticmethod
    def vec_to_mat_SE2(vector):
        # Converts a vector to the SE(2) matrix representation
        x = vector[0]
        y = vector[1]
        cos_theta = np.cos(vector[2])
        sin_theta = np.sin(vector[2])

        matrix = np.zeros([3, 3])
        matrix[0, 0] = cos_theta
        matrix[0, 1] = -sin_theta
        matrix[0, 2] = x
        matrix[1, 0] = sin_theta
        matrix[1, 1] = cos_theta
        matrix[1, 2] = y
        matrix[2, 2] = 1

        return matrix

    def vec_to_mat_se2(vector):
        x = vector[0]
        y = vector[1]
        w = vector[2]

        matrix = np.zeros([3,3])
        matrix[0, 0] = 0
        matrix[0, 1] = -w
        matrix[0, 2] = x
        matrix[1, 0] = w
        matrix[1, 1] = 0
        matrix[1, 2] = y
        matrix[2, 2] = 0
        
        return matrix

    @staticmethod
    def mat_to_vec_SE2(matrix):
        # Converts the SE(2) matrix representation to a vector
        vector = np.zeros(3)
        vector[0] = matrix[0, 2]
        vector[1] = matrix[1, 2]
        vector[2] = np.arctan2(matrix[1, 0], matrix[0, 0])

        return vector

    @staticmethod
    def se2_to_SE2(vec_or_mat):
        # Converts Lie algebra (se2) to Lie group (SE2) representation
        if vec_or_mat.ndim == 1: 
            if vec_or_mat[2] == 0:
                return vec_or_mat
            else:
                matrix = SE2.vec_to_mat_se2(vec_or_mat)
                matrix = scipy.linalg.expm(matrix)
                matrix = SE2.mat_to_vec_SE2(matrix)
        elif vec_or_mat.ndim == 2:
            matrix = vec_or_mat         
            if vec_or_mat[0,0] == 0:		
                matrix = np.identity(3)
                matrix[0,2] = vec_or_mat[0,2]
                matrix[1,2] = vec_or_mat[1,2]
            else:
                matrix = scipy.linalg.expm(matrix)
        return matrix

    @staticmethod
    def SE2_to_se2(vec_or_mat):
        # Converts Lie group (SE2) to Lie algebra (se2) representation
        if vec_or_mat.ndim == 1:
            matrix = SE2.vec_to_mat_SE2(vec_or_mat)
            matrix = scipy.linalg.logm(matrix)
            return SE2.mat_to_vec_SE2(matrix)
        elif vec_or_mat.ndim == 2:
            matrix = vec_or_mat
            matrix = scipy.linalg.logm(matrix)
            return matrix

# Example usage of the SE2 class
if __name__ == "__main__":
    a = SE2(np.array([5, 0, 1]))
    b = SE2(np.array([1, 1, 0]))
    print(a @ b)
    print(b @ a)
    print(a ** -1)
