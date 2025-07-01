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

    def __init__(self, n1, k1, H1, n2, k2, H2T):
        """
        Initialize the HGP codes.
        
        Args:
            n1 (int): Length of the first classical code.
            k1 (int): Dimension of the first classical code.
            H1 (nd.array): The parity check matrix of the first classical code.
            n2 (int): Length of the second classical code.
            k2 (int): Dimension of the second classical code.
            H2T (nd.array): The parity check matrix of the second classical code.
        """
        # self._code_parameter_check(n1, H1)
        # self._code_parameter_check(n2, H2T)
        self.n1 = n1
        self.k1 = k1
        self.H1 = H1
        self.n2 = n2
        self.k2 = k2
        self.H2T = H2T
        #C2(n1) -> C1(r1) H1 (r1 \times n1)
        #C2'(r2) -> C1'(n2) H2T (n2 \times r2)
        self.r1, _ = H1.shape
        _, self.r2 = H2T.shape
        #TODO: distance
        # self.n, self.k = self._calculate_HGP_parameters(n1, k1, H1, n2, k2, H2T)
        self.HxT = None
        self.HZ = None
    
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
            HxT(nd.array): A parity matrix.
            Hz(nd.array): A parity matrix.
        """
        if self.HxT and self. Hz:
            return self.HxT, self.Hz
        
        I_n1 = np.eye(self.n1, dtype=int)
        I_n2 = np.eye(self.n2, dtype=int)
        I_r1 = np.eye(self.r1, dtype=int)
        I_r2 = np.eye(self.r2, dtype=int)
        
        #HXT = [[H1 \otimes Ir2],[In1 \otimes H2T]]
        #HZ = [Ir1 \otimes H2T, H1 \otimes In2]
        self.HxT = np.block([[np.kron(self.H1, I_r2)], [np.kron(I_n1, self.H2T)]])
        self.Hz = np.block([np.kron(I_r1, self.H2T), np.kron(self.H1, I_n2)])
        return self.HxT, self.Hz


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
        self.HxT = None
        self.HZ = None
    
    def get_parity_matrices(self):
        """
        Get the parity matrices of FB Codes.
        
        Returns:
            Hx(nd.array): A parity matrix.
            Hz(nd.array): A parity matrix.
        """
        if self.HxT and self. Hz:
            return self.HxT, self.Hz
        
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

    def __init__(self, n1, k1, H1, n2, k2, H2T, h1, h1inv=None, h2=None):
        """
        Initialize the BP codes.
        
        Args:
            n1 (int): Length of the first classical code.
            k1 (int): Dimension of the first classical code.
            H1 (nd.array): The parity check matrix of the first classical code.
            n2 (int): Length of the second classical code.
            k2 (int): Dimension of the second classical code.
            H2T (nd.array): The parity check matrix of the second classical code.
            h1 (List[nd.array]): The group h1 is a common symmetry group of the two classical codes.
            h2 (List[nd.array]): The group h1's another size group.
        """
        #TODO: check function
        # self._code_parameter_check(n1, H1)
        # self._code_parameter_check(n2, H2T)
        self.n1 = n1
        self.k1 = k1
        self.H1 = H1
        self.n2 = n2
        self.k2 = k2
        self.H2T = H2T
        self.r1, _ = H1.shape
        self.r2, _ = H2T.shape
        self.h1 = h1
        if h1inv is None:
            self.h1inv = self.calculate_H_inv(self.h1)
        else:
            self.h1inv = h1inv
        if h2 is None:
            self.h2 = h1
            self.h2inv = self.calculate_H_inv(self.h1)
        else:
            self.h2 = h2
            self.h2inv = self.calculate_H_inv(self.h2)
        self.HxT = None
        self.HZ = None
    
    def calculate_H_inv(self, H):
        h_inv = []
        for h in H:
            h_inv.append(np.linalg.inv(h))
        return h_inv
    
    def get_parity_matrices(self):
        """
        Get the parity matrices of BP Codes.
        
        Returns:
            Hx(nd.array): A parity matrix.
            Hz(nd.array): A parity matrix.
        """
        if self.HxT and self.Hz:
            return self.HxT, self.Hz
        
        #C2(n1) -> C1(r1) H1 (r1 \times n1)
        #C2'(r2) -> C1'(n2) H2T (n2 \times r2)
         
        I_n1 = np.eye(self.n1, dtype=int) #C2
        I_r1 = np.eye(self.r1, dtype=int) #C1
        I_r2 = np.eye(self.r2, dtype=int) #C2'
        I_n2 = np.eye(self.n2, dtype=int) #C1'


        #Construct the vector spaces 
        # L: tensor product
        # Lh: by cancelling out the symmetry group h
        # Ldict: the dictionary representing the corresponding information because of the cancelling of h
    
        # C1 \otimes_H C1'
        L11, L11h, L11dict =  self.calculate_vector_spaces_with_h(self.r1, self.n2, I_r1, I_n2)
        print("Ldict11", L11dict)
        # C2 \otimes_H C2'
        L22, L22h, L22dict =  self.calculate_vector_spaces_with_h(self.n1, self.r2, I_n1, I_r2)
        print("Ldict22", L22dict)
        # C1 \otimes_H C2'
        L12, L12h, L12dict =  self.calculate_vector_spaces_with_h(self.r1, self.r2, I_r1, I_r2)
        print("Ldict12", L12dict)
        # C2 \otimes_H C1'
        L21, L21h, L21dict =  self.calculate_vector_spaces_with_h(self.n1, self.n2, I_n1, I_n2)
        print("Ldict21", L21dict)

        #Calculate the H matrix if it is a HGP code
        hgp = HGPGenerator(self.n1, self.k1, self.H1, self.n2, self.k2, self.H2T)
        Hxprime, Hzprime = hgp.get_parity_matrices()

        #Construct the final H matrix
        Hxup = self.calculate_final_H_matrix_from_HGP_H_matrix(Hxprime[:self.r1*self.r2], L22h, L22dict, L12h, L12dict)
        Hxdown = self.calculate_final_H_matrix_from_HGP_H_matrix(Hxprime[self.r1*self.r2:], L22h, L22dict, L21h, L21dict)
        Hzl = self.calculate_final_H_matrix_from_HGP_H_matrix(Hzprime[:, :self.r1*self.r2], L12h, L12dict, L11h, L11dict)
        Hzr = self.calculate_final_H_matrix_from_HGP_H_matrix(Hzprime[:, self.r1*self.r2:], L21h, L21dict, L11h, L11dict)
        return np.block([[Hxup], [Hxdown]]), np.block([Hzl, Hzr])
    
    def show_easy_func(self, lh_list):
        numbers = []
        for j, lh in enumerate(lh_list):
            if isinstance(lh, np.ndarray):
                for i, llh in enumerate(lh):
                    if llh == 1.:
                        numbers.append(i)
            else:
                if lh == 1.:
                    numbers.append(j)
        return numbers
                

    def calculate_vector_spaces_with_h(self, p: int, q:int, w: np.ndarray, v: np.ndarray):
        L = []
        Lh = []
        Ldict = {}
        for i in range(p):
            for j in range(q):
                current_L = np.kron(w[i], v[j])
                L.append(current_L)
                current_Lh = []
                #construct Lh
                for h, h_inv in zip(self.h1, self.h2inv):
                    current_kron = np.kron(w[i]@h, h_inv@v[j].T)
                    #找这个kron的结果在不在现有的lh里
                    # matching_index = next((idx for idx, lh in enumerate(current_Lh) if np.allclose(current_kron, lh)), None)
                    matching_index = next((idx for idx, lh in enumerate(current_Lh) if (current_kron==lh).all()), None)
                    if matching_index is None:
                        #不在的话就加入
                        current_Lh.append(current_kron)
                #找这个lh的小list是不是已经存在 没有存在的话就存一下
                # if next((idx for idx, compare_Lh in enumerate(Lh) if self.two_list_equal(current_Lh, compare_Lh)), None) is None:
                matching_index = None
                for idx, compare_Lh in enumerate(Lh):
                    if set(self.show_easy_func(current_Lh)) == set(self.show_easy_func(compare_Lh)):
                        matching_index = idx
                if matching_index is None:
                    Lh.append(current_Lh)
        #construct dictionary
        for l, current_Lh_list in enumerate(Lh):
            Ldict[l] = []
            for k, current_L in enumerate(L):
                flag = False
                for lh in current_Lh_list:
                    if np.allclose(current_L, lh):
                        flag = True
                if flag:
                    Ldict[l].append(k)
        return L, Lh, Ldict
    
    def two_list_equal(self, list1, list2):
        if len(list1) != len(list2):
            return False
        list1copy = list1.copy()
        list2copy = list2.copy()
        for i, list1ele in enumerate(list1):
            for j, list2ele in enumerate(list2):
                if (list1ele==list2ele).all():
                    np.delete(list1copy, i)
                    np.delete(list2copy, j)
        if (len(list1copy)==0) & (len(list2copy)==0):
            return True
        return False
    
    def calculate_final_H_matrix_from_HGP_H_matrix(self, HGPH: np.ndarray, L1H, L1dict, L2H, L2dict):
        Hout = np.zeros((len(L1H),len(L2H)))
        for i in range(HGPH.shape[0]):
            for j in range(HGPH.shape[1]):
                if HGPH[i,j] == 1:
                    L1Hindices = self.corresponding_key(i, L1dict)
                    L2Hindices = self.corresponding_key(j, L2dict)
                    Hout[L1Hindices,L2Hindices] = 1
        return Hout
    
    def corresponding_key(self, val, dictionary):
            for k, v in dictionary.items():
                if val in v:
                    return k
