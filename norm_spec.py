import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.linalg import qr


def read_arxiv(file_path):
    """
    Read arxiv edge list file and construct sparse adjacency matrix.

    Args:
        file_path: Path to edge list file (CSV format, 0-indexed)

    Returns:
        scipy.sparse.csr_matrix: Symmetric adjacency matrix
    """
    I = []
    J = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            data = line.strip().split(',')
            I.append(int(data[0]))
            J.append(int(data[1]))

    I = np.array(I)
    J = np.array(J)

    # No need to add 1 (already 0-indexed in Python)
    n = max(I.max(), J.max()) + 1

    # Create sparse adjacency matrix
    A = sparse.csr_matrix((np.ones(len(I)), (I, J)), shape=(n, n))

    # Make symmetric
    A = A.maximum(A.T)

    # Ensure binary (0 or 1)
    A.data = np.minimum(A.data, 1)

    return A


def spectral_embedding(adj, k=128):
    """
    Compute spectral embedding using normalized regularized Laplacian.

    This is the Python equivalent of the Julia main() function.

    Args:
        adj: scipy.sparse matrix (CSR format) - adjacency matrix
        k: int - number of eigenvectors to compute

    Returns:
        numpy.ndarray: N x k matrix of spectral embeddings
    """
    # Ensure CSR format for efficient operations
    if not sparse.isspmatrix_csr(adj):
        adj = adj.tocsr()

    N = adj.shape[0]

    # Compute degree vector
    d = np.array(adj.sum(axis=1)).flatten()

    # Regularization parameter τ (tau)
    tau = d.sum() / len(d)

    # Normalized regularized Laplacian: I + D^(-1/2) * (A + τ/N * J) * D^(-1/2)
    # where J is the all-ones matrix

    # D^(-1/2) with regularization
    d_reg = d + tau
    d_inv_sqrt = 1.0 / np.sqrt(d_reg)
    D_inv_sqrt = sparse.diags(d_inv_sqrt)

    # Create LinearOperator for (A + τ/N * J)
    # This avoids explicitly creating the dense all-ones matrix
    def matvec(X):
        if X.ndim == 1:
            # A * X + (τ/N) * sum(X) * ones_vector
            result = adj.dot(X) + (tau / N) * X.sum()
        else:
            # For multiple columns
            result = adj.dot(X) + (tau / N) * X.sum(axis=0)
        return result

    A_op = LinearOperator((N, N), matvec=matvec, dtype=np.float64)

    # Compute normalized operator: D^(-1/2) * A_op * D^(-1/2)
    def nrl_matvec(X):
        if X.ndim == 1:
            # D^(-1/2) * (A_op * (D^(-1/2) * X))
            temp = d_inv_sqrt * X
            temp = matvec(temp)
            result = d_inv_sqrt * temp
            # Add identity: I + normalized_A_op
            result = X + result
        else:
            # For multiple columns
            temp = d_inv_sqrt[:, np.newaxis] * X
            temp = matvec(temp)
            result = d_inv_sqrt[:, np.newaxis] * temp
            result = X + result
        return result

    NRL_op = LinearOperator((N, N), matvec=nrl_matvec, dtype=np.float64)

    # Compute k largest eigenvalues and eigenvectors
    # which='LM' = largest magnitude (equivalent to Julia's :LM)
    Lambda, V = eigsh(NRL_op, k=k, which='LM', tol=1e-6, ncv=min(2*k+1, N))

    # eigsh returns eigenvalues in ascending order, but we computed largest magnitude
    # so they should already be the largest ones

    # SCDM axis rotation (optional but helpful)
    # QR decomposition with column pivoting
    Q, R, piv = qr(V.T, pivoting=True, mode='economic')

    # Select top k pivot columns
    piv_indices = piv[:k]

    # SVD of V[piv_indices, :]
    U_svd, s_svd, Vt_svd = np.linalg.svd(V[piv_indices, :].T, full_matrices=False)

    # Rotate V
    SCDM_V = V @ (U_svd @ Vt_svd)

    return SCDM_V


# For backward compatibility if anyone imports this directly
main = spectral_embedding
