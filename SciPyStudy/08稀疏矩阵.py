from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import numpy as np

array = np.array([[2, 0, 0, 3, 0, 0], [1, 0, 1, 0, 0, 2], [0, 0, 1, 2, 0, 0]])
csr = csr_matrix(array)
print(csr, '\n')

csc = csc_matrix(array)
print(csc)