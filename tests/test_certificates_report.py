import numpy as np
import pytest
from qldpc_generator.certificates import classical_distance,hgp_distance_witness
from qldpc_generator.verify_paper import verify_all,distance_contradicts_claim
from qldpc_generator.code import HGPGenerator,BPGenerator,FBGenerator
from qldpc_generator.algebra import x,subgroup,polynomial_to_circulant,cyclic_group


def test_classical_zero_code_and_hgp_certificate_hypotheses():
    assert classical_distance(np.eye(3,dtype=np.uint8))["value"]==float("inf")
    h=np.array([[1,1,0],[0,1,1]],dtype=np.uint8)
    c=classical_distance(h)
    assert c["value"]==3 and c["witness"].tolist()==[1,1,1]
    cert=hgp_distance_witness(h)
    assert cert["upper_bound"]==3 and cert["code"].is_logical(cert["logical"])
    for bad in [np.eye(2,dtype=np.uint8),np.ones((2,3),dtype=np.uint8)]:
        with pytest.raises(ValueError,match="full row rank"):
            hgp_distance_witness(bad)
    with pytest.raises(ValueError,match="dimension"):
        classical_distance(np.zeros((1,21),dtype=np.uint8))


def test_legacy_adapters_and_new_conventions():
    h=np.array([[1,1,0],[0,1,1]],dtype=np.uint8)
    generator=HGPGenerator(3,1,h,3,1,h.T)
    a,b=generator.get_parity_matrices()
    assert generator.get_parity_matrices()[0] is a
    assert np.array_equal(a,generator.get_css_code().hz.T)
    assert np.array_equal(b,generator.get_css_code().hx)
    assert generator._calculate_HGP_parameters()==(13,1)
    acts,invs=subgroup(2,4)
    m=polynomial_to_circulant(1+x,4)
    bp=BPGenerator(4,0,m,4,0,m,acts,invs)
    a,b=bp.get_parity_matrices()
    assert bp.get_parity_matrices()[0] is a
    assert not np.any(b @ a % 2)
    g=cyclic_group(2)
    fb=FBGenerator([[1]],np.eye(2,dtype=np.uint8),g,g.regular(),g.regular())
    assert np.array_equal(fb.get_parity_matrices()[0],fb.get_css_code().hx)


def test_report_does_not_promote_literature_or_conjecture_to_verified_distance():
    r=verify_all(distance=False)
    assert r["passed"] and len(r["cases"])==12 and all(r["relations"].values())
    lp=next(c for c in r["cases"] if c["id"]=="lp544")
    assert lp["distance_claim"]=={"kind":"reported_upper_bound","value":12}
    assert lp["distance_computation"]["status"]=="not_searched"
    assert r["unresolved_inputs"] and r["outside_finite_validation"]


def test_lower_bound_and_upper_certificate_can_disprove_a_claim_without_exact_distance():
    assert distance_contradicts_claim(6,"reported",{"status":"bounded","lower_bound":7})
    assert distance_contradicts_claim(6,"reported_upper_bound",{"status":"bounded","lower_bound":7})
    assert distance_contradicts_claim(10,"reported",{"status":"bounded","lower_bound":5,"upper_bound":8})
    assert not distance_contradicts_claim(10,"reported_upper_bound",{"status":"exact","value":8})
