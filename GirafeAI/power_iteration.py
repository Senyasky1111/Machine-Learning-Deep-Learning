import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    r = np.random.rand(data.shape[0])
    
    for _ in range(num_steps):
        Ar = np.dot(data, r)
        r = Ar
        r = r/np.max(np.abs(r))
    sob = np.dot(Ar, r)/np.dot(r.T, r)
    sob_vector = r/np.linalg.norm(r)
    return float(sob), sob_vector