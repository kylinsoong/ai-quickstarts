import numpy as np
from scipy.sparse import csr_matrix

# Non-sparse (dense) vector
dense_vector = np.array([0, 0, 0, 4, 0, 0, 5, 0, 0, 0, 6])
print("Dense Vector:")
print(dense_vector)

# Sparse vector
sparse_vector = csr_matrix(dense_vector)
print("\nSparse Vector (CSR format):")
print(sparse_vector)

