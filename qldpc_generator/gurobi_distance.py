"""Optional MILP distance minimization with explicit logical-quotient constraints.

For Z operators, minimize |z| with HX z = 2a, LX z = 2b+s,
z,s binary, a,b nonnegative integer, and sum(s)>=1. LX is a basis
of ker(HZ)/row(HX), so s!=0 excludes every Z stabilizer, not just zero.
Exchange X and Z for the other sector. No literature distance is assumed.
"""
from math import ceil, isfinite
from numbers import Real, Integral
import numpy as np


class GurobiUnavailableError(RuntimeError):
    """The optional package, license, or solver environment cannot run the model."""


def _load_gurobi():
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise GurobiUnavailableError(
            "Install Gurobi with: python -m pip install -e '.[gurobi]'. "
            "A license valid for the model size is also required."
        ) from exc
    return gp


def _combine(sectors):
    lower = min(result['lower_bound'] for result in sectors.values())
    upper = min(result['upper_bound'] for result in sectors.values())
    if lower > upper:
        raise RuntimeError("Inconsistent distance bounds returned by Gurobi")
    result = {'status': 'exact' if lower == upper else 'bounded',
              'lower_bound': lower, 'upper_bound': upper, 'sectors': sectors}
    if lower == upper:
        result['value'] = upper
    return result


def _solve_sector(code, sector, signatures, logical_basis, gp, env, time_limit, threads):
    checks, _ = code._sector(sector)
    # A verified logical basis row supplies a feasible warm start and a valid
    # upper bound even if the solver stops before accepting an incumbent.
    seed = logical_basis[np.argmin(logical_basis.sum(axis=1))].copy()
    if not code.is_logical(seed, sector):
        raise RuntimeError("Invalid logical basis supplied to the Gurobi model")
    with gp.Model(f'css_distance_{sector}', env=env) as model:
        model.Params.TimeLimit = float(time_limit)
        model.Params.Threads = int(threads)
        model.Params.Seed = 0
        model.Params.MIPGap = 0.0
        model.Params.MIPGapAbs = 0.0
        model.Params.IntFeasTol = 1e-9
        model.Params.FeasibilityTol = 1e-9
        z = model.addVars(code.n, vtype=gp.GRB.BINARY, name='support')
        s = model.addVars(len(signatures), vtype=gp.GRB.BINARY, name='logical_parity')
        for i, row in enumerate(checks):
            support = np.flatnonzero(row).tolist()
            slack = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=len(support)//2,
                                 name=f'check_half_{i}')
            model.addConstr(gp.quicksum(z[j] for j in support) == 2*slack)
            slack.Start = int(np.dot(row.astype(np.int64), seed))//2
        for i, row in enumerate(signatures):
            support = np.flatnonzero(row).tolist()
            slack = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=len(support)//2,
                                 name=f'logical_half_{i}')
            model.addConstr(gp.quicksum(z[j] for j in support) == 2*slack+s[i])
            total = int(np.dot(row.astype(np.int64), seed))
            slack.Start, s[i].Start = total//2, total % 2
        model.addConstr(gp.quicksum(s.values()) >= 1, name='nontrivial_logical')
        model.setObjective(gp.quicksum(z.values()), gp.GRB.MINIMIZE)
        for j in range(code.n):
            z[j].Start = int(seed[j])
        model.optimize()

        status_names = {getattr(gp.GRB, name): name for name in (
            'OPTIMAL', 'TIME_LIMIT', 'NODE_LIMIT', 'ITERATION_LIMIT', 'SOLUTION_LIMIT',
            'INTERRUPTED', 'WORK_LIMIT', 'MEM_LIMIT', 'SUBOPTIMAL', 'NUMERIC',
            'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED') if hasattr(gp.GRB, name)}
        status = status_names.get(model.Status, f'STATUS_{model.Status}')
        if status in ('INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED'):
            raise RuntimeError(f"Unexpected {status} for a feasible bounded logical-distance model")
        witness, witness_source = seed, 'logical_basis'
        if model.SolCount:
            values = np.array([z[j].X for j in range(code.n)])
            candidate = np.rint(values).astype(np.uint8)
            if (np.any(np.abs(values-candidate) > 1e-6)
                    or not code.is_logical(candidate, sector)
                    or abs(float(model.ObjVal)-int(candidate.sum())) > 1e-6):
                raise RuntimeError("Gurobi incumbent failed exact binary logical-witness validation")
            if int(candidate.sum()) <= int(witness.sum()):
                witness, witness_source = candidate, 'gurobi_incumbent'
        upper = int(witness.sum())
        raw_bound = float(model.ObjBound)
        # Numerical termination does not justify a solver lower bound.
        trusted = status in ('OPTIMAL', 'TIME_LIMIT', 'NODE_LIMIT', 'ITERATION_LIMIT',
                             'SOLUTION_LIMIT', 'INTERRUPTED', 'WORK_LIMIT', 'MEM_LIMIT')
        lower = max(1, ceil(raw_bound-1e-6)) if trusted and isfinite(raw_bound) else 1
        if status == 'OPTIMAL' and (not model.SolCount or witness_source != 'gurobi_incumbent'):
            raise RuntimeError("Gurobi reported optimality without a matching validated incumbent")
        if lower > upper:
            raise RuntimeError("Gurobi lower bound exceeds the validated witness weight")
        if status == 'OPTIMAL' and lower != upper:
            raise RuntimeError("Gurobi optimality status conflicts with the returned bounds")
        result = {'status': 'exact' if lower == upper else 'bounded',
                  'lower_bound': lower, 'upper_bound': upper,
                  'witness': np.flatnonzero(witness).tolist(), 'witness_source': witness_source,
                  'solver_status': status, 'solver_bound': raw_bound if isfinite(raw_bound) else None,
                  'runtime_seconds': float(model.Runtime), 'node_count': float(model.NodeCount),
                  'solution_count': int(model.SolCount), 'variables': model.NumVars,
                  'constraints': model.NumConstrs}
        if lower == upper:
            result['value'] = upper
        return result


def gurobi_distance(code, *, time_limit=60, threads=1, solver_log=False):
    """Compute min(dX,dZ). TimeLimit is per sector, excluding model setup.

    ``exact`` uses Gurobi's optimality bounds with zero MIP gaps and a witness
    rechecked over GF(2). It is not an independently checkable infeasibility
    proof. Resource limits return bounds; package/license errors are explicit.
    """
    if isinstance(time_limit, bool) or not isinstance(time_limit, Real) or not isfinite(time_limit) or time_limit < 0:
        raise ValueError('time_limit must be a finite nonnegative number of seconds per sector')
    if isinstance(threads, bool) or not isinstance(threads, Integral) or not 1 <= threads <= 1024:
        raise ValueError('threads must be an integer between 1 and 1024')
    if code.k == 0:
        return {'status': 'no_logicals', 'value': None, 'sectors': {}, 'method': 'gurobi'}
    gp = _load_gurobi()
    lx, lz = code.logical_operators()
    try:
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', int(bool(solver_log)))
            env.start()
            sectors = {
                sector: _solve_sector(code, sector, signatures, basis, gp, env, time_limit, threads)
                for sector, signatures, basis in [('X', lz, lx), ('Z', lx, lz)]
            }
    except gp.GurobiError as exc:
        raise GurobiUnavailableError(
            f'Gurobi error {exc.errno}: {exc}. Check installation, license, and model-size limits; '
            "no fallback was performed. Select method='mitm' explicitly to use enumeration."
        ) from exc
    result = _combine(sectors)
    result.update(method='gurobi', solver_version='.'.join(map(str, gp.gurobi.version())),
                  time_limit_per_sector=float(time_limit), threads=int(threads))
    return result
