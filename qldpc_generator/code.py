import numpy as np
import random

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
        Get the parity matrices of HGP codes.
        
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


class FBGenerator(QLDPCGenerator):
    """A child class for generating Fiber Bundle Codes."""

    def __init__(self, n1, k1, H1, n2, k2, H2, G):
        """
        Initialize the FB codes.
        
        Args:
            n1 (int): Length of the first classical code.
            k1 (int): Dimension of the first classical code.
            H1 (nd.array): The parity check matrix of the first classical code.
            n2 (int): Length of the second classical code.
            k2 (int): Dimension of the second classical code.
            H2 (nd.array): The parity check matrix of the second classical code.
            G (List[nd.array]): The automorphism group G, requiring g \partial_f = \partial g f.
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
        self.G = G
    
    def get_parity_matrices(self):
        """
        Get the parity matrices of FB Codes.
        
        Returns:
            Hx(nd.array): A parity matrix.
            Hz(nd.array): A parity matrix.
        """
        if self.Hx and self. Hz:
            return self.Hx, self.Hz
        
        I_n1 = np.eye(self.n1, dtype=int) #fiber vector space 1
        I_n2 = np.eye(self.n2, dtype=int) #fiber vector space 2
        I_r1 = np.eye(self.r1, dtype=int) #base vector space 1
        I_r2 = np.eye(self.r2, dtype=int) #base vector space 2

        Hz_part1 = np.zeros((self.r2, self.n1, self.r2, self.n1))
        #b2 \otimes f1
        for i in range(self.r2):
            for j in range(self.n1):
                Hz_part1[i,j,:,:] = np.kron(I_r2[i, :], I_n1[j, :])
        
        Hz_part2 = np.zeros((self.r1, self.n2, self.r1, self.n2))
        #b1 \otimes ZYY f2
        ZYY = self._generate_ZYY_mask_matrix()
        for i in range(self.r1):
            for j in range(self.n2):
                #TODO: let G can choose different group representation with different size
                Hz_part2[i,j,:,:] = np.kron(I_r1[i, :], self.G[ZYY[i,j]].matrix(self.n2)@I_n2[j, :])
        Hz_part2[0,j,:,:] = np.sum(Hz_part2)
        for i in range(1,self.r2):
            Hz_part2[i,j,:,:] = Hz_part2[0,j,:,:]
        self.Hz = np.block([Hz_part1, Hz_part2])
        
        Hx_part1 = np.zeros((self.r1, self.n1, self.r1, self.n1))
        #b1 \otimes f1
        for i in range(self.r1):
            for j in range(self.n1):
                Hx_part1[i,j,:,:] = np.kron(I_r1[i, :], I_n1[j, :])

        Hx_part2 = np.zeros((self.r1, self.n1, self.r1, self.n1))
        #b1 \otimes ZYY f1
        ZYY = self._generate_ZYY_mask_matrix()
        for i in range(self.r1):
            for j in range(self.n1):
                #TODO: let G can choose different group representation with different size
                Hx_part2[i,j,:,:] = np.kron(I_r1[i, :], self.G[ZYY[i,j]].matrix(self.n1)@I_n1[j, :])
        Hx_part2[0,j,:,:] = np.sum(Hx_part2)
        for i in range(1,self.r1):
            Hx_part2[i,j,:,:] = Hx_part2[0,j,:,:]
        self.Hx = np.block([Hx_part1, Hx_part2])

        return self.Hx, self.Hz
    
    def _generate_ZYY_mask_matrix(self):
        #spread the number of elements in G!
        ZYY = np.zeros(self.n2, self.r2)
        r = len(self.G)
        
        # Fill matrix by cycling through [1, 2, ..., r]
        for i in range(self.n2*self.r2):
            row = i // r
            col = i % r
            ZYY[row][col] = (i % r) + 1
        
        # Flatten matrix to a list for shuffling
        flat_matrix = [ZYY[i][j] for i in range(self.n2) for j in range(self.r2)]
        
        # Shuffle the flattened list
        random.shuffle(flat_matrix)
        
        # Reconstruct the matrix from the shuffled list
        for i in range(self.n2, self.r2):
            row = i // r
            col = i % r
            ZYY[row][col] = flat_matrix[i] * random.randint(0, 1) #random for 0 or \varphi_\tau in the paper
        return ZYY


class BPGenerator(QLDPCGenerator):
    """A child class for generating Balanced Product Codes."""

    def __init__(self, n1, k1, H1, n2, k2, H2, H):
        """
        Initialize the BP codes.
        
        Args:
            n1 (int): Length of the first classical code.
            k1 (int): Dimension of the first classical code.
            H1 (nd.array): The parity check matrix of the first classical code.
            n2 (int): Length of the second classical code.
            k2 (int): Dimension of the second classical code.
            H2 (nd.array): The parity check matrix of the second classical code.
            H (List[nd.array]): The group H is a common symmetry group of the two classical codes.
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
        self.H = H
        self.Hinv = self._calculate_H_inv(self.H)
    
    def _calculate_H_inv(self, H):
        #TODO: calculate the inverse element of the H group.
        pass
    
    def get_parity_matrices(self):
        """
        Get the parity matrices of BP Codes.
        
        Returns:
            Hx(nd.array): A parity matrix.
            Hz(nd.array): A parity matrix.
        """
        if self.Hx and self. Hz:
            return self.Hx, self.Hz
        
        I_n1 = np.eye(self.n1, dtype=int) #C1
        I_n2 = np.eye(self.n2, dtype=int) #C1'
        I_r1 = np.eye(self.r1, dtype=int) #C2
        I_r2 = np.eye(self.r2, dtype=int) #C2'

        #Construct the vector spaces by cancelling out the symmetry H
        # C1 \otimes_H C1'
        L11 =  self._calculate_vector_spaces(self.n1, self.n2, I_n1, I_n2)
        # C2 \otimes_H C2'
        L22 =  self._calculate_vector_spaces(self.r1, self.r2, I_r1, I_r2)
        # C1 \otimes_H C2'
        L12 =  self._calculate_vector_spaces(self.n1, self.r2, I_n1, I_r2)
        # C2 \otimes_H C1'
        L21 =  self._calculate_vector_spaces(self.r1, self.n2, I_r1, I_n2)

        #Calculate the H matrix if it is a HGP code
        hgp = HGPGenerator(self.n1, self.k1, self.H1, self.n2, self.k2, self.H2)
        Hxprime, Hzprime = hgp.get_parity_matrices()

        #Construct the final H matrix
        Hxl = self._calculate_final_H_matrix_from_HGP_H_matrix(Hxprime[:, self.r1*self.n2:], I_n1, I_r1, I_n1, I_r2, L11, L22, L12, L21)
        Hxr = self._calculate_final_H_matrix_from_HGP_H_matrix(Hxprime[:, :self.r1*self.n2], I_n1, I_r1, I_n2, I_r1, L11, L22, L12, L21)
        Hzl = self._calculate_final_H_matrix_from_HGP_H_matrix(Hzprime[:, self.r1*self.n2:], I_n1, I_r1, I_n2, I_r2, L11, L22, L12, L21)
        Hzr = self._calculate_final_H_matrix_from_HGP_H_matrix(Hzprime[:, :self.r1*self.n2], I_n2, I_r1, I_n2, I_r2, L11, L22, L12, L21)
        return np.block([Hxl, Hxr]), np.block([Hzl, Hzr])
    
    def _calculate_vector_spaces(self, p: int, q:int, w: np.ndarray, v: np.ndarray):
        L = []
        for i in range(p):
            for j in range(q):
                for h, h_inv in zip(self.H, self.Hinv):
                    L.append(np.kron(w[i]@h, h_inv@v[j]))
        return L
    
    def _calculate_final_H_matrix_from_HGP_H_matrix(self, H: np.ndarray, w: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray, L11, L22, L12, L21):
        Hout = np.zeros((self.n1*self.n2, self.r1*self.r2))
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                if H[i,j] == 1:
                    m1, m2, m3, m4 = self._find_the_indices_from_old_indices(i,j)
                    for k, nk in enumerate(L11+L22):
                        for l, ml in enumerate(L12+L21):
                            if (np.kron(w[i]@h, h_inv@v[j]) == nk) and (np.kron(w[i]@h, h_inv@v[j]) == nk):
                                Hout[k,l] = 1
        return Hout

    
    def _find_the_indices_from_old_indices(self, i: int, j: int):
        m1 = 1
        m2 = 1
        m3 = 1
        m4 = 1
        return m1, m2, m3, m4
