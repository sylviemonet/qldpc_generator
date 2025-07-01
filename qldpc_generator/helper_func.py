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

def subgroup(m,l):
    """Get the subgroup of the symmetry vector space for bp code.
        l: size of the square matrix.
        m: 
    """
    H=[]
    Hinv =[]
    if l % m != 0:
        print(f"警告: l ({l}) 不是 m ({m}) 的倍数。整数除法 l // m 可能不符合数学预期。")
    for i in range(1, (l // m) + 1):
        x = symbols('x')
        poly_H = x**(m * i)
        poly_Hinv = x**(l - m * i)
        if m*i < l:
           H.append(polynomial_to_circulant(poly_H, l))
           Hinv.append(polynomial_to_circulant(poly_Hinv, l))
        if m*i == l:
           H.append(polynomial_to_circulant(1, l))
           Hinv.append(polynomial_to_circulant(1, l))
            
    return H,Hinv