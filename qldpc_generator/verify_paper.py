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
    if value is None or computation["status"] in ("not_searched", "no_logicals"):
        return False
    lower = computation.get("value") if computation["status"] == "exact" else computation.get("lower_bound")
    upper = computation.get("value") if computation["status"] == "exact" else computation.get("upper_bound")
    if lower is not None and lower > value:
        return True
    return kind != "reported_upper_bound" and upper is not None and upper < value


def verify_all(distance=True):
    rows=[]
    limits={"hgp13":3,"lcs52":4,"bicycle6":2,"gb48":8,"bb72":6,
            "bp90":6,"bp108":6,"bb144":6,"mitten150":6}
    for case in paper_cases():
        code=case.build(); p=code.parameters()
        valid=(p["n"],p["k"],p["rank_x"],p["rank_z"])==(case.n,case.k,case.rank_x,case.rank_z)
        d=code.distance(limits[case.identifier]) if distance and case.identifier in limits else {"status":"not_searched"}
        # All witnesses are checked again directly against both matrices and the stabilizer row space.
        for sector,data in d.get("sectors",{}).items():
            if data["status"]=="exact":
                v=np.zeros(code.n,dtype=np.uint8); v[data["witness"]]=1
                valid=valid and code.is_logical(v,sector) and int(v.sum())==data["value"]
        contradicted=distance_contradicts_claim(case.distance_value,case.distance_kind,d)
        rows.append({"id":case.identifier,"paper_label":case.paper_label,"source":case.source,
                     "parameters":p,"algebra_verified":bool(valid),"distance_claim":{"kind":case.distance_kind,"value":case.distance_value},
                     "distance_computation":d,"contradiction_found":bool(contradicted)})
    relations=relation_checks()
    return {"provenance":provenance(),"cases":rows,"relations":relations,
            "passed":all(r["algebra_verified"] and not r["contradiction_found"] for r in rows) and all(relations.values()),
            "meaning_of_passed":"All executed algebra/certificate checks passed; reported or not_searched distance claims are not certified.",
            "unresolved_inputs":[
                {"claim":"GB [[126,28,8]]", "reason":"Paper mentions parameters but does not specify its two polynomials."},
                {"claim":"Planar [[88,6,6]] and tile [[288,8,12]], [[512,18,19]]", "reason":"Full open-boundary check data are not supplied in this manuscript; periodic templates do not determine them."}],
            "outside_finite_validation":["General asymptotic rate/distance theorems", "General LCS distance conjecture", "Decoder thresholds and single-shot guarantees", "Hardware experimental results"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("reports/paper_validation.json"))
    parser.add_argument("--skip-distance",action="store_true")
    args=parser.parse_args()
    report=verify_all(not args.skip_distance)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for r in report["cases"]:
        p=r["parameters"]; d=r["distance_computation"]
        distance_text=f"D={d['value']} (exact)" if d['status']=="exact" else (f"D>={d['lower_bound']} (lower bound)" if 'lower_bound' in d else "D not searched")
        print(f"{r['id']:12} N={p['n']:4} K={p['k']:3} ranks=({p['rank_x']},{p['rank_z']}) {distance_text}")
    print(f"Report: {args.output}; executed checks passed: {report['passed']}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__=="__main__":
    main()
