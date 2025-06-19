import numpy as np

class QLDPCGenerator:
    """A class for generating different Density Parity-Check (QLDPC) codes."""
    
    def __init__(self):
        """
        Initialize the QLDPC generator.
        """
    
    def show_code_parameters(self):
        """
        Show all the information about a QLDPC code.
        
        Returns:
            dict(dictionary): A dictionary containing code parameters.
        """
    
    def show_parity_matrices(self):
        """
        Generate a QLDPC code with its parity matrices.
        
        Returns:
            Hx(nd.array): A parity matrix.
            Hz(nd.array): A parity matrix.
        """
    
class HGPGenerator(QLDPCGenerator):
    """A child class for generating HyperGraph Product Codes."""

    def __init__(self, n1, k1, H1, n2, k2, H2):
        """
        Initialize the HGP codes.
        
        Args:
            n1 (int): Length of the first classical code.
            k1 (int): Dimension of the first classical code.
            H1 (nd.array): The parity check matrix of the first classical code.
            n2 (int): Length of the second classical code.
            k2 (int): Dimension of the second classical code.
            H2 (nd.array): The parity check matrix of the second classical code.
        """
        self._code_parameter_check(n1, H1)
        self._code_parameter_check(n2, H2)
        self.n1 = n1
        self.k1 = k1
        self.H1 = H1
        self.n2 = n2
        self.k2 = k2
        self.H2 = H2
        self.r1, _ = H1.shape
        self.r2, _ = H2.shape
        #TODO: distance
        self.n, self.k = self._calculate_HGP_parameters(n1, k1, H1, n2, k2, H2)
    
    def _code_parameter_check(n, H):
        """
        Check the dimensions of the parameters.
        
        Raise Errors.
        """
        _, n1 = H.shape
        if n1 != n:
            raise Exception()
    
    def _calculate_HGP_parameters(self):
        """
        Calculate the HGP parameters [[n,k]].
        
        Returns:
            n (int): Length of the first classical code.
            k (int): Dimension of the first classical code.
        """
        k1T = rank(self.H1.T)
        k2T = rank(self.H2.T)
        n = self.n1*self.n2 + self.r1*self.r2
        k = self.k1*self.k2 + k1T*k2T
        return n, k
        
    def get_parity_matrices(self):
        """
        Get a QLDPC code with its parity matrices.
        
        Returns:
            Hx(nd.array): A parity matrix.
            Hz(nd.array): A parity matrix.
        """
        if self.Hx and self. Hz:
            return self.Hx, self.Hz
        
        I_n1 = np.eye(self.n1, dtype=int)
        I_n2 = np.eye(self.n2, dtype=int)
        I_r1 = np.eye(self.r1, dtype=int)
        I_r2 = np.eye(self.r2, dtype=int)
        
        self.Hx = np.block([np.kron(self.H1, I_n2), np.kron(I_r1, self.H2.T)])
        self.Hz = np.block([np.kron(I_n1. self.H2), np.kron(self.H1.T, I_r2)])
        return self.Hx, self.Hz
