"""Independent enumeration and solver-bound checks for the optional MILP backend."""
import json
import os
import sys
import numpy as np
import pytest
from qldpc_generator import CSSCode, hgp
from qldpc_generator.gf2 import nullspace
from qldpc_generator.gurobi_distance import GurobiUnavailableError, _combine
from qldpc_generator.verify_paper import verify_all
from tests.test_algebra_css import brute_distance


def repetition_hgp():
    h = [[1, 1, 0], [0, 1, 1]]
    return hgp(h, h)


@pytest.fixture(scope='module')
def solver():
    try:
        repetition_hgp().distance(method='gurobi', time_limit=10)
    except GurobiUnavailableError as exc:
        if os.environ.get('QLDPC_REQUIRE_GUROBI') == '1':
            pytest.fail(str(exc))
        pytest.skip(str(exc))


def test_missing_dependency_is_explicit_and_mitm_still_works(monkeypatch):
    monkeypatch.setitem(sys.modules, 'gurobipy', None)
    with pytest.raises(GurobiUnavailableError, match='pip install'):
        repetition_hgp().distance(method='gurobi')
    assert repetition_hgp().distance(3)['value'] == 3


@pytest.mark.parametrize('kwargs', [
    {'method': 'unknown'}, {'method': 'gurobi', 'max_weight': 3},
    {'method': 'gurobi', 'max_subsets': 100}, {'time_limit': 1},
    {'method': 'gurobi', 'time_limit': -1}, {'method': 'gurobi', 'time_limit': float('nan')},
    {'method': 'gurobi', 'time_limit': float('inf')}, {'method': 'gurobi', 'threads': 0},
    {'method': 'gurobi', 'threads': 1.5},
])
def test_reject_ambiguous_or_invalid_solver_options(kwargs):
    with pytest.raises(ValueError):
        repetition_hgp().distance(**kwargs)


def test_trivial_logical_space_needs_no_solver(monkeypatch):
    monkeypatch.setitem(sys.modules, 'gurobipy', None)
    code = CSSCode(np.eye(2, dtype=np.uint8), np.zeros((0, 2), dtype=np.uint8))
    assert code.distance(method='gurobi')['status'] == 'no_logicals'


def test_random_codes_match_independent_full_space_enumeration(solver):
    rng = np.random.default_rng(2026)
    for n in range(1, 8):
        for _ in range(3):
            hx = rng.integers(0, 2, (2, n), dtype=np.uint8)
            kernel = nullspace(hx)
            mix = rng.integers(0, 2, (2, len(kernel)), dtype=np.uint8)
            code = CSSCode(hx, mix @ kernel % 2)
            result = code.distance(method='gurobi', time_limit=10)
            assert result.get('value') == brute_distance(code)
            if code.k:
                assert result['status'] == 'exact'
                assert result['value'] == code.distance(n)['value']
                for sector, data in result['sectors'].items():
                    witness = np.zeros(n, dtype=np.uint8)
                    witness[data['witness']] = 1
                    assert code.is_logical(witness, sector)
                    assert int(witness.sum()) == data['upper_bound']


def test_sector_orientation_and_asymmetric_distances(solver):
    code = CSSCode(np.zeros((0, 3), dtype=np.uint8), [[1, 1, 0], [0, 1, 1]])
    result = code.distance(method='gurobi', time_limit=10)
    assert result['value'] == 1
    assert result['sectors']['X']['value'] == 3
    assert result['sectors']['Z']['value'] == 1


def test_weight_one_stabilizers_are_excluded_in_both_sectors(solver):
    base = repetition_hgp()
    hx = np.zeros((7, 15), dtype=np.uint8)
    hz = np.zeros((7, 15), dtype=np.uint8)
    hx[:6, :13], hz[:6, :13] = base.hx, base.hz
    hx[6, 13], hz[6, 14] = 1, 1
    code = CSSCode(hx, hz)
    assert code.distance(method='gurobi', time_limit=10)['value'] == 3


def test_zero_time_limit_is_a_bound_not_an_exact_distance(solver):
    code = repetition_hgp()
    result = code.distance(method='gurobi', time_limit=0)
    assert result['status'] == 'bounded' and 'value' not in result
    assert result['lower_bound'] == 1 and result['upper_bound'] == 3
    for sector, data in result['sectors'].items():
        assert data['solver_status'] == 'TIME_LIMIT'
        assert data['solution_count'] == 0 and data['witness_source'] == 'logical_basis'
        v = np.zeros(code.n, dtype=np.uint8)
        v[data['witness']] = 1
        assert code.is_logical(v, sector)
    json.dumps(result, allow_nan=False)


def test_global_minimum_uses_both_sector_bounds():
    bounded = _combine({'X': {'lower_bound': 1, 'upper_bound': 8},
                        'Z': {'lower_bound': 3, 'upper_bound': 3}})
    assert bounded['status'] == 'bounded' and bounded['lower_bound'] == 1 and bounded['upper_bound'] == 3
    exact = _combine({'X': {'lower_bound': 4, 'upper_bound': 8},
                      'Z': {'lower_bound': 3, 'upper_bound': 3}})
    assert exact['status'] == 'exact' and exact['value'] == 3


def test_report_solver_errors_are_visible_and_fail_the_report(monkeypatch):
    monkeypatch.setitem(sys.modules, 'gurobipy', None)
    report = verify_all(distance_method='gurobi', case_ids=['hgp13'])
    assert not report['passed'] and report['cases'][0]['algebra_verified']
    assert report['cases'][0]['distance_computation']['status'] == 'unavailable'
    skipped = verify_all(distance=False, distance_method='gurobi', case_ids=['hgp13'])
    assert skipped['passed'] and skipped['cases'][0]['distance_computation']['status'] == 'not_searched'


def test_report_checks_time_limited_witnesses(solver):
    report = verify_all(distance_method='gurobi', time_limit=0, case_ids=['hgp13'])
    assert report['passed'] and len(report['cases']) == 1
    assert report['cases'][0]['distance_computation']['status'] == 'bounded'
    assert report['configuration']['time_limit_per_sector'] == 0
    assert report['configuration']['case_ids'] == ['hgp13']
    json.dumps(report, allow_nan=False)


def test_license_error_is_not_silently_replaced_by_enumeration(solver, monkeypatch):
    import gurobipy as gp
    def unavailable_env(*args, **kwargs):
        raise gp.GurobiError(10009, 'Test license unavailable')
    monkeypatch.setattr(gp, 'Env', unavailable_env)
    with pytest.raises(GurobiUnavailableError, match='10009'):
        repetition_hgp().distance(method='gurobi')
    report = verify_all(distance_method='gurobi', case_ids=['hgp13'])
    assert not report['passed']
    assert report['cases'][0]['distance_computation']['status'] == 'unavailable'


def test_report_rejects_a_stabilizer_presented_as_a_bounded_witness(monkeypatch):
    code = repetition_hgp()
    support = np.flatnonzero(code.hz[0]).tolist()
    fake = {'status': 'bounded', 'lower_bound': 1, 'upper_bound': len(support),
            'sectors': {'Z': {'status': 'bounded', 'lower_bound': 1,
                              'upper_bound': len(support), 'witness': support}}}
    monkeypatch.setattr(CSSCode, 'distance', lambda *args, **kwargs: fake)
    report = verify_all(distance_method='gurobi', case_ids=['hgp13'])
    assert not report['passed'] and not report['cases'][0]['algebra_verified']


def test_report_rejects_unknown_backend_and_cases():
    with pytest.raises(ValueError):
        verify_all(distance_method='typo')
    with pytest.raises(ValueError):
        verify_all(case_ids=['missing'])
