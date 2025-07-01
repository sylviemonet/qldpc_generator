import math
import numpy as np
from sympy import Poly, symbols, ZZ

def polynomial_to_circulant(poly, l):
    """Given a polynomials with symbol 'x' and the circulant matrix size l, get the circulant matrix."""
    
    x = symbols('x')

   
    if not isinstance(poly, Poly):
        poly = Poly(poly, x,domain=ZZ)

    
    coeffs = poly.all_coeffs()
    coeffs.reverse()
    circulant_matrix = np.zeros((l, l), dtype=int)
    for i in range(l):
        a1 = np.pad( coeffs, (0, l - len( coeffs)), 'constant')
        circulant_matrix[i] = np.roll(a1, i)

    return circulant_matrix