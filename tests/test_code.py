import pytest
import numpy as np
from qldpc_generator.code import QLDPCGenerator

def test_qldpc_generator_init():
    """Test initialization of QLDPCGenerator with random matrices."""
    gen = QLDPCGenerator(m1=4, n1=6, m2=3, n2=5, density=0.2)
    assert gen.H1.shape == (4, 6)
    assert gen.H2.shape == (3, 5)
    assert np.all((gen.H1 == 0) | (gen.H1 == 1))  # Binary matrix
    assert np.all((gen.H2 == 0) | (gen.H2 == 1))  # Binary matrix

def test_qldpc_generator_hgp():
    """Test HGP code generation."""
    gen = QLDPCGenerator(m1=4, n1=6, m2=3, n2=5, density=0.2)
    result = gen.generate_hgp()
    H_X, H_Z = result["H_X"], result["H_Z"]
    n, k = result["n"], result["k"]
    
    # Check shapes
    assert H_X.shape == (4 * 5, 6 * 5 + 4 * 3)  # (m1 * n2) x (n1 * n2 + m1 * m2)
    assert H_Z.shape == (6 * 3, 6 * 5 + 4 * 3)  # (n1 * m2) x (n1 * n2 + m1 * m2)
    
    # Check quantum condition
    assert np.all(np.dot(H_X, H_Z.T) % 2 == 0)
    
    # Check code parameters
    assert n == 6 * 5 + 4 * 3  # n1 * n2 + m1 * m2
    assert k >= 0