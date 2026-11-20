"""Run with python -m qldpc_generator.verify_paper --output reports/paper_validation.json."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
import sympy as sp
from .paper_cases import paper_cases,bb_direct_sum_equivalence,bb144_case,PAPER_SOURCE
from .algebra import (x,y,cyclic_group,symmetric_group_3,group_algebra_matrix)
from .equivalences import lp_to_bp,bp_to_lp,fb_to_bp,bp_to_fb,monomial_bb_equivalence
from .constructions import bb_quotient_dimension,generalized_toric
from .representations import repetition_checks,classical_soundness
from .certificates import hgp_distance_witness
from .gurobi_distance import GurobiUnavailableError


def relation_checks():
    results={}
    for group,name in [(cyclic_group(2),"C2"),(cyclic_group(4),"C4"),(symmetric_group_3(),"S3")]:
        a=group_algebra_matrix(group,[[[0,1],[0]]])
        b=group_algebra_matrix(group,[[[0,group.order-1]],[[1]]])
        bp,eq=lp_to_bp(group,a,b)
        results[f"lp_bp_{name}"]=eq.verify() and bp_to_lp(bp)[2].verify()
    group=symmetric_group_3()
    bp,eq=fb_to_bp([[1,1]],[[1,1,1]],group,np.zeros((6,1),dtype=int),np.array(group.labels),[[1,3]])
    results["fb_bp_S3_nonfree_fiber"]=eq.verify() and bp_to_fb(bp)[2].verify()
    results["bb_direct_sum_t3"]=bb_direct_sum_equivalence(3).verify()
    results["bb144_quotient_dimension"]=bb144_case().k==2*bb_quotient_dimension(y**2*(x+x**2+y**3),x**2*(y+y**2+x**3),12,6)
    results["bb144_generalized_toric"]=monomial_bb_equivalence(bb144_case(),generalized_toric(-1,-3,3,-1,12,6),12,6,(-1,-2),(-2,-1)).verify()
    witness=hgp_distance_witness([[1,1,0],[0,1,1]])
    results["hgp_distance_quotient_witness"]=witness["upper_bound"]==3
    from fractions import Fraction
    results["classical_path_star_soundness"]=classical_soundness(repetition_checks(8))==Fraction(2,7) and classical_soundness(repetition_checks(8,"star"))==Fraction(8,7)
    return results


def provenance():
    root=Path(__file__).resolve().parent.parent
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.joinpath("qldpc_generator").glob("*.py"))}
    def git(*args):
        try:
            return subprocess.check_output(["git",*args],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError):
            return None
    return {"generated_at":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),
            "numpy":np.__version__,"sympy":sp.__version__,"git_head":git("rev-parse","HEAD"),
            "working_tree_dirty":bool(git("status","--porcelain")),"source_sha256":hashes,
            "paper":PAPER_SOURCE}


def distance_contradicts_claim(value, kind, computation):
    if value is None or computation["status"] in ("not_searched", "no_logicals", "unavailable"):
        return False
    lower = computation.get("value") if computation["status"] == "exact" else computation.get("lower_bound")
    upper = computation.get("value") if computation["status"] == "exact" else computation.get("upper_bound")
    if lower is not None and lower > value:
        return True
    return kind != "reported_upper_bound" and upper is not None and upper < value


def verify_all(distance=True, *, distance_method="mitm", time_limit=60, threads=1, case_ids=None):
    if distance_method not in ("mitm", "gurobi"):
        raise ValueError("distance_method must be mitm or gurobi")
    cases = paper_cases()
    if case_ids is not None:
        selected = set(case_ids)
        if not selected or selected - {case.identifier for case in cases}:
            raise ValueError("case_ids must select known paper instances")
        cases = [case for case in cases if case.identifier in selected]
    rows=[]
    limits={"hgp13":3,"lcs52":4,"bicycle6":2,"gb48":8,"bb72":6,
            "bp90":6,"bp108":6,"bb144":6,"mitten150":6}
    for case in cases:
        code=case.build(); p=code.parameters()
        valid=(p["n"],p["k"],p["rank_x"],p["rank_z"])==(case.n,case.k,case.rank_x,case.rank_z)
        if not distance:
            d = {"status": "not_searched"}
        elif distance_method == "gurobi":
            try:
                d = code.distance(method="gurobi", time_limit=time_limit, threads=threads)
            except GurobiUnavailableError as exc:
                d = {"status": "unavailable", "method": "gurobi", "error": str(exc)}
        else:
            d = code.distance(limits[case.identifier]) if case.identifier in limits else {"status":"not_searched"}
        # All witnesses are checked again directly against both matrices and the stabilizer row space.
        for sector,data in d.get("sectors",{}).items():
            if "witness" in data:
                v=np.zeros(code.n,dtype=np.uint8); v[data["witness"]]=1
                weight = data.get("upper_bound", data.get("value"))
                valid=valid and code.is_logical(v,sector) and int(v.sum())==weight
        contradicted=distance_contradicts_claim(case.distance_value,case.distance_kind,d)
        rows.append({"id":case.identifier,"paper_label":case.paper_label,"source":case.source,
                     "parameters":p,"algebra_verified":bool(valid),"distance_claim":{"kind":case.distance_kind,"value":case.distance_value},
                     "distance_computation":d,"contradiction_found":bool(contradicted)})
    relations=relation_checks()
    return {"provenance":provenance(),"cases":rows,"relations":relations,
            "configuration": {"distance_method": distance_method, "distance_enabled": bool(distance),
                              "time_limit_per_sector": time_limit if distance_method == "gurobi" else None,
                              "threads": threads if distance_method == "gurobi" else None,
                              "case_ids": [case.identifier for case in cases]},
            "passed":all(r["algebra_verified"] and not r["contradiction_found"]
                         and r["distance_computation"]["status"] != "unavailable" for r in rows) and all(relations.values()),
            "meaning_of_passed":"All requested solver runs were available and executed algebra/certificate checks passed; bounded, reported, or not_searched distances are not certified exact.",
            "unresolved_inputs":[
                {"claim":"GB [[126,28,8]]", "reason":"Paper mentions parameters but does not specify its two polynomials."},
                {"claim":"Planar [[88,6,6]] and tile [[288,8,12]], [[512,18,19]]", "reason":"Full open-boundary check data are not supplied in this manuscript; periodic templates do not determine them."}],
            "outside_finite_validation":["General asymptotic rate/distance theorems", "General LCS distance conjecture", "Decoder thresholds and single-shot guarantees", "Hardware experimental results"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("reports/paper_validation.json"))
    parser.add_argument("--skip-distance",action="store_true")
    parser.add_argument("--distance-method", choices=("mitm", "gurobi"), default="mitm")
    parser.add_argument("--time-limit", type=float, default=60,
                        help="Gurobi seconds per X/Z sector (default 60; setup time excluded)")
    parser.add_argument("--threads", type=int, default=1, help="Gurobi thread count (default 1)")
    parser.add_argument("--cases", nargs="+", choices=[case.identifier for case in paper_cases()],
                        help="Select instances; the default runs all twelve")
    args=parser.parse_args()
    report=verify_all(not args.skip_distance, distance_method=args.distance_method,
                      time_limit=args.time_limit, threads=args.threads, case_ids=args.cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for r in report["cases"]:
        p=r["parameters"]; d=r["distance_computation"]
        if d['status'] == 'exact':
            distance_text = f"D={d['value']} (exact)"
        elif d['status'] == 'unavailable':
            distance_text = f"D unavailable: {d['error']}"
        elif 'lower_bound' in d:
            distance_text = (f"{d['lower_bound']} <= D <= {d['upper_bound']} (bounded)"
                             if d.get('upper_bound') is not None else f"D>={d['lower_bound']} (lower bound)")
        else:
            distance_text = "D not searched"
        print(f"{r['id']:12} N={p['n']:4} K={p['k']:3} ranks=({p['rank_x']},{p['rank_z']}) {distance_text}")
    print(f"Report: {args.output}; executed checks passed: {report['passed']}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__=="__main__":
    main()
