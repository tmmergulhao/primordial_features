from typing import Callable, List
import numpy as np
from dataclasses import dataclass

def QuadraticForm(A,B,M):
    """Compute a quadratic form: A.M.B^{T}

    Args:
        A (np.array): The first vector, should be n x 1
        B (np.array): The second vector, should be n x 1
        invCOV (np.array): The matrix. Should be n X n

    Returns:
        float: The result of the quadratic form
    """
    return np.dot(A, np.dot(M,B.T))

@dataclass
class likelihoods:
    theory: Callable[[List[float]], np.ndarray]
    data: np.ndarray
    invcov: np.ndarray


    def logGaussian(self, theta: List[float]) -> float:
        """
        Calculate the Gaussian log-likelihood.

        Parameters:
        theta (List[float]): The parameter vector.

        Returns:
        float: The log-likelihood value.
        """
        theory_result = self.theory(theta)
        assert len(theory_result) == len(self.data), f"Data and theory with different shape: data: {len(self.data)}, theory: {len(theory_result)}"
        diff = theory_result - self.data
        return -0.5 * QuadraticForm(diff, diff, self.invcov)