import pytest
import numpy as np
from sympy import symbols
from qldpc_generator.code import QLDPCGenerator, HGPGenerator, BPGenerator
from qldpc_generator.helper_func import polynomial_to_circulant, subgroup

#============hgp test==========================
# Fixture to create a simple HGPGenerator instance
@pytest.fixture
def hgp_generator():
    # Example classical parity-check matrices
    H1 = np.array([[1, 1, 0], [0, 1, 1]], dtype=int)  # 2x3 matrix, r1=2, n1=3
    H2 = np.array([[1, 0, 1], [0, 1, 1]], dtype=int)  # 2x3 matrix, r2=2, n2=3
    n1, k1 = 3, 1  # k1 = n1 - rank(H1) = 3 - 2 = 1
    n2, k2 = 3, 1  # k2 = n2 - rank(H2) = 3 - 2 = 1
    return HGPGenerator(n1, k1, H1, n2, k2, H2.T)

def test_hgp_generator_init(hgp_generator):
    """Test initialization of HGPGenerator."""
    assert hgp_generator.n1 == 3
    assert hgp_generator.k1 == 1
    assert hgp_generator.n2 == 3
    assert hgp_generator.k2 == 1
    assert hgp_generator.r1 == 2
    assert hgp_generator.r2 == 2
    assert np.array_equal(hgp_generator.H1, np.array([[1, 1, 0], [0, 1, 1]]))
    assert np.array_equal(hgp_generator.H2T, np.array([[1, 0, 1], [0, 1, 1]]).T)
    assert hasattr(hgp_generator, 'Hx') is False  # Not initialized yet
    assert hasattr(hgp_generator, 'HzT') is False  # Not initialized yet

def test_get_parity_matrices(hgp_generator):
    """Test the get_parity_matrices method."""
    HxT, Hz = hgp_generator.get_parity_matrices()
    print(hgp_generator.get_parity_matrices())
    # Check shapes
    # Hx = [H1 ⊗ I_n2 | I_r1 ⊗ H2.T], shape: (r1*n2) x (n1*n2 + r1*r2)
    # Hz = [I_n1 ⊗ H2 | H1.T ⊗ I_r2], shape: (n1*r2) x (n1*n2 + r1*r2)
    assert HxT.shape == (13, 6)  
    assert Hz.shape == (6, 13)  
    # Check binary matrices
    assert np.all((HxT == 0) | (HxT == 1))
    assert np.all((Hz == 0) | (Hz == 1))
    # Check quantum condition: Hx * Hz.T = 0 mod 2
    assert np.all(np.dot(Hz, HxT) % 2 == 0)
    # Check attributes are set
    assert np.array_equal(hgp_generator.HxT, HxT)
    assert np.array_equal(hgp_generator.Hz, Hz)
#============================================

#============bp test=========================
def bp_generator():
    # Example classical parity-check matrices
    H1 = np.array([[1, 1, 0], [0, 1, 1]], dtype=int)  # 2x3 matrix, r1=2, n1=3
    H2T = np.array([[1, 0, 1], [0, 1, 1]], dtype=int)  # 2x3 matrix, r2=2, n2=3
    n1, k1 = 3, 1  # k1 = n1 - rank(H1) = 3 - 2 = 1
    n2, k2 = 3, 1  # k2 = n2 - rank(H2) = 3 - 2 = 1
    h1 = [np.array([[1, 0], [0, 1]]), # Identity element
          np.array([[0, 1], [1, 0]])  # Permutation matrix
          ]
    return BPGenerator(n1, k1, H1, n2, k2, H2T, h1)

def test_bp_parity_check_matrix():
    m_val = 2
    l_val = 4
    Hmatrices, Hinvmatrices = subgroup(m_val, l_val)
    x = symbols('x')
    # p1x= 1+x+x**5
    p1x = 1+x
    # p2x=1+x**2+x**7
    p2x=1+x
    H_1= polynomial_to_circulant(p1x, 4)
    H_2= polynomial_to_circulant(p2x, 4)
    n1, k1 = H_1.shape[0], 0
    n2, k2 = H_2.shape[1], 0
    bpcode1 = BPGenerator(n1=n1, k1=k1, H1=H_1, n2=n2, k2=k2, H2T=H_2, h1 = Hmatrices, h1inv=Hinvmatrices)
    HxT1 , HZ = bpcode1.get_parity_matrices()
    assert np.allclose(HZ@HxT1 % 2, np.zeros((HZ@HxT1).shape))
#============================================
