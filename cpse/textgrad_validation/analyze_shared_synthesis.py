"""Scan all gold docs for shared-synthesis structure.

Quantifies per-doc: number of polymers, process-flow entries per polymer, and
cross-polymer process similarity (the "same-condition shared synthesis" pattern
that collapses 4aa3b3a7). Also reports property counts. Used to pick a training
doc that teaches the optimizer to replicate shared conditions per-sample instead
of omitting inherited feed amounts.
"""

import glob
import io
import json
import os
import sys
from difflib import SequenceMatcher

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA = "test/data"


def _descs(flow_entry):
    """Collect free-text pieces of a flow entry for similarity comparison."""
    parts = []
    rc = flow_entry.get("反应条件") or []  # may be str or list[dict]
    if isinstance(rc, str):
        parts.append(rc)
    elif isinstance(rc, list):
        for item in rc:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            for key in ("制备过程", "反应装置", "反应气氛", "溶剂"):
                v = item.get(key)
                if isinstance(v, str):
                    parts.append(v)
                elif isinstance(v, dict):
                    parts.append(str(v.get("单值", "")))
    parts += flow_entry.get("后处理步骤", []) if isinstance(flow_entry.get("后处理步骤"), list) else []
    return " ".join(str(x) for x in parts).strip()


def _similarity(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def analyze(doc):
    d = json.load(open(os.path.join(DATA, doc), encoding="utf-8"))
    polys = d.get("聚合物", [])
    n_poly = len(polys)
    flows = []  # list of (poly_name, [entry_descs...])
    props_per_poly = []
    for p in polys:
        fl = p.get("工艺流程") or []
        flows.append((p.get("名称", ""), [_descs(e) for e in fl]))
        props_per_poly.append(len(p.get("性质") or []))
    # Cross-polymer flow similarity: for each polymer, max similarity to any other polymer's flow.
    cross_sim = []
    for i, (_, fi) in enumerate(flows):
        for j, (_, fj) in enumerate(flows):
            if i >= j:
                continue
            for a in fi:
                for b in fj:
                    s = _similarity(a, b)
                    if s > 0.6:
                        cross_sim.append((i, j, s))
    return {
        "doc": doc,
        "n_poly": n_poly,
        "flow_entry_counts": [len(f) for _, f in flows],
        "props_per_poly": props_per_poly,
        "n_sim_pairs": len(cross_sim),
        "max_sim": max((s for _, _, s in cross_sim), default=0.0),
        "has_same_cond_shared": len(cross_sim) > 0,
    }


def main():
    docs = sorted(glob.glob(os.path.join(DATA, "*.json")))
    rows = []
    for f in docs:
        doc = os.path.basename(f)
        rows.append(analyze(doc))
    rows.sort(key=lambda r: -r["n_sim_pairs"])
    print(f"{'doc':<38} poly flow#  sim  maxsim  props")
    for r in rows:
        print(
            f"{r['doc']:<38} {r['n_poly']:>4} {str(r['flow_entry_counts']):<12} "
            f"{str(r['n_sim_pairs']):>3} {r['max_sim']:>5.2f} {str(r['props_per_poly'])}"
        )


if __name__ == "__main__":
    main()
