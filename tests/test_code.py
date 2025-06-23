import pytest
import numpy as np
from qldpc_generator.code import QLDPCGenerator, HGPGenerator, BPGenerator

def test_qldpc_generator_base_class():
    """Test the base QLDPCGenerator class instantiation."""
    qldpc = QLDPCGenerator()
    assert isinstance(qldpc, QLDPCGenerator)
    # Check that base class methods raise NotImplementedError if not overridden
    with pytest.raises(NotImplementedError):
        qldpc.show_code_parameters()
    with pytest.raises(NotImplementedError):
        qldpc.show_parity_matrices()

#============hgp test==========================
# Fixture to create a simple HGPGenerator instance
@pytest.fixture
def hgp_generator():
    # Example classical parity-check matrices
    H1 = np.array([[1, 1, 0], [0, 1, 1]], dtype=int)  # 2x3 matrix, r1=2, n1=3
    H2 = np.array([[1, 0, 1], [0, 1, 1]], dtype=int)  # 2x3 matrix, r2=2, n2=3
    n1, k1 = 3, 1  # k1 = n1 - rank(H1) = 3 - 2 = 1
    n2, k2 = 3, 1  # k2 = n2 - rank(H2) = 3 - 2 = 1
    return HGPGenerator(n1, k1, H1, n2, k2, H2)

def test_hgp_generator_init(hgp_generator):
    """Test initialization of HGPGenerator."""
    assert hgp_generator.n1 == 3
    assert hgp_generator.k1 == 1
    assert hgp_generator.n2 == 3
    assert hgp_generator.k2 == 1
    assert hgp_generator.r1 == 2
    assert hgp_generator.r2 == 2
    assert np.array_equal(hgp_generator.H1, np.array([[1, 1, 0], [0, 1, 1]]))
    assert np.array_equal(hgp_generator.H2, np.array([[1, 0, 1], [0, 1, 1]]))
    assert hgp_generator.n == 3 * 3 + 2 * 2  # n1*n2 + r1*r2 = 9 + 4 = 13
    assert hasattr(hgp_generator, 'Hx') is False  # Not initialized yet
    assert hasattr(hgp_generator, 'Hz') is False  # Not initialized yet

def test_code_parameter_check():
    """Test the _code_parameter_check method."""
    H_valid = np.array([[1, 0, 1], [0, 1, 0]], dtype=int)  # 2x3
    H_invalid = np.array([[1, 0], [0, 1]], dtype=int)  # 2x2
    # Valid case: n = 3 matches H columns
    HGPGenerator._code_parameter_check(3, H_valid)  # Should not raise
    # Invalid case: n = 3 does not match H columns (2)
    with pytest.raises(ValueError, match="Number of columns in H does not match n"):
        HGPGenerator._code_parameter_check(3, H_invalid)

def test_calculate_hgp_parameters(hgp_generator):
    """Test the _calculate_HGP_parameters method."""
    n, k = hgp_generator._calculate_HGP_parameters()
    assert n == 3 * 3 + 2 * 2  # n1*n2 + r1*r2 = 9 + 4 = 13
    # k = k1*k2 + rank(H1.T)*rank(H2.T)
    # H1 rank = 2, H1.T rank = 2, H2 rank = 2, H2.T rank = 2
    # k1 = 1, k2 = 1, so k = 1*1 + 2*2 = 1 + 4 = 5
    assert k == 1 * 1 + 2 * 2  # k1*k2 + rank(H1.T)*rank(H2.T)

def test_get_parity_matrices(hgp_generator):
    """Test the get_parity_matrices method."""
    Hx, Hz = hgp_generator.get_parity_matrices()
    # Check shapes
    # Hx = [H1 ⊗ I_n2 | I_r1 ⊗ H2.T], shape: (r1*n2) x (n1*n2 + r1*r2)
    # Hz = [I_n1 ⊗ H2 | H1.T ⊗ I_r2], shape: (n1*r2) x (n1*n2 + r1*r2)
    assert Hx.shape == (2 * 3, 3 * 3 + 2 * 2)  # (6, 13)
    assert Hz.shape == (3 * 2, 3 * 3 + 2 * 2)  # (6, 13)
    # Check binary matrices
    assert np.all((Hx == 0) | (Hx == 1))
    assert np.all((Hz == 0) | (Hz == 1))
    # Check quantum condition: Hx * Hz.T = 0 mod 2
    assert np.all(np.dot(Hx, Hz.T) % 2 == 0)
    # Check attributes are set
    assert np.array_equal(hgp_generator.Hx, Hx)
    assert np.array_equal(hgp_generator.Hz, Hz)

def test_show_code_parameters(hgp_generator):
    """Test the show_code_parameters method."""
    params = hgp_generator.show_code_parameters()
    assert isinstance(params, dict)
    assert params['n'] == 13
    assert params['k'] == 5
    # Add more checks if show_code_parameters is extended

def test_show_parity_matrices(hgp_generator):
    """Test the show_parity_matrices method."""
    Hx, Hz = hgp_generator.show_parity_matrices()
    # Should call get_parity_matrices
    assert Hx.shape == (2 * 3, 3 * 3 + 2 * 2)  # (6, 13)
    assert Hz.shape == (3 * 2, 3 * 3 + 2 * 2)  # (6, 13)
    assert np.all((Hx == 0) | (Hx == 1))
    assert np.all((Hz == 0) | (Hz == 1))
    assert np.all(np.dot(Hx, Hz.T) % 2 == 0)

def test_invalid_initialization():
    """Test initialization with invalid parameters."""
    H1 = np.array([[1, 1, 0], [0, 1, 1]])  # 2x3
    H2 = np.array([[1, 0], [0, 1]])  # 2x2
    # n1=4 does not match H1 columns (3)
    with pytest.raises(ValueError, match="Number of columns in H does not match n"):
        HGPGenerator(n1=4, k1=1, H1=H1, n2=2, k2=1, H2=H2)
#============================================

#============bp test=========================
