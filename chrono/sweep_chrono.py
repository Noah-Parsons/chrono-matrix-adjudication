"""
Chrono as the fourth engine: the three families, on the study's own probes.

Same cases, seeds and probe states as the existing sweeps:

  chain        N = 1..5, 16 probes, seed 20260822, U(-0.5, 0.5)   (adapter_drake)
  slidercrank  l/r sweep at m_s/m_c = 1e2, mass sweep at l/r = 3,
               16 probes on the manifold + both dead centres,
               seed 20260825                                     (sweep_slidercrank)
  cartpole     M/m = 10 .. 1e-6, 16 probes, seed 20260825        (sweep_families)
               (the cart-pole probe states do not depend on th0, so each
               mass ratio is run once rather than once per starting angle)

Columns, worst over the probes of each case:
  gen      the study's metric on generalised accelerations, matrix method
  dead     the same, at the slider-crank dead centres
  body     body-level acceleration (Negrut: compare at the body level)
  force    body-level constraint force, Cq^T lam vs closed form
  native   Chrono's own DoAssembly + GetPosDt2 (chrono#846), for contrast
  guard    reference guard: C, dC/dt, d2C/dt2 of the reference
  res      closed form substituted into Chrono's equations (max of dyn, con)
  qc_fd    analytic Qc vs a central difference of Chrono's own Cq
           (limited by the 13-figure export to about 1e-8)

Run in the chrono conda environment, from this folder:

    python sweep_chrono.py --family chain
    python sweep_chrono.py --family slidercrank
    python sweep_chrono.py --family cartpole
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, SUITE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import adapter_chrono as AC  # noqa: E402

K_PROBE = 16
TOL = 1e-8


def chain_cases():
    import reference
    for N in range(1, 6):
        rng = np.random.default_rng(20260822)
        states = [rng.uniform(-0.5, 0.5, size=2 * N) for _ in range(K_PROBE)]
        yield f"N={N}", AC.ChainFamily(reference.NLinkChain(N)), states, []


def slidercrank_cases():
    import reference_slidercrank as RSC
    cases = ([("l/r", ratio, 1e2) for ratio in RSC.RATIOS]
             + [("m_s/m_c", 3.0, mr) for mr in RSC.MASS_RATIOS])
    for knob, ratio, mr in cases:
        sc = RSC.SliderCrank(ratio, mass_ratio=mr)
        rng = np.random.default_rng(20260825)
        states = []
        for _ in range(K_PROBE):
            th = rng.uniform(-math.pi, math.pi)
            thd = rng.uniform(-4.0, 4.0)
            states.append([th, thd, sc.slider_position(th),
                           sc.slider_velocity(th, thd)])
        dead = [[th, thd, sc.slider_position(th), sc.slider_velocity(th, thd)]
                for th in (0.0, math.pi) for thd in (1.0, 4.0)]
        yield (f"{knob:<8} l/r={ratio:<5g} m={mr:<6g}",
               AC.SliderCrankFamily(sc), states, dead)


def cartpole_cases():
    import reference_cartpole as RCP
    for mr in RCP.MASS_RATIOS:
        cp = RCP.CartPole(mass_ratio=mr)
        rng = np.random.default_rng(20260825)
        states = [np.array([rng.uniform(-1, 1), rng.uniform(-2, 2),
                            rng.uniform(-math.pi, math.pi),
                            rng.uniform(-3, 3)]) for _ in range(K_PROBE)]
        yield f"M/m={mr:g}", AC.CartPoleFamily(cp), states, []


FAMILIES = {"chain": chain_cases, "slidercrank": slidercrank_cases,
            "cartpole": cartpole_cases}


def run_family(name, args):
    print("=" * 112)
    print(f"Chrono (matrix method) -- {name}")
    print("=" * 112)
    hdr = (f"{'case':<30}{'gen':>9}{'dead':>9}{'body':>9}{'force':>9}"
           f"{'native':>9}{'guard':>9}{'res':>9}{'qc_fd':>9}  status")
    print(hdr)
    print("-" * len(hdr))
    rows, shown = [], False
    for label, fam, states, dead in FAMILIES[name]():
        t0 = time.time()
        worst, dworst = {}, 0.0
        try:
            for s in states:
                o = AC.probe(fam, s, assembly=args.assembly,
                             fd_h=0.0 if args.no_fd else 1e-5,
                             native=not args.no_native)
                if not shown:
                    print(f"  [first probe] state applied by {o['refresh']}; "
                          f"Cq has {o['rows']} rows, {o['distance_rows']} "
                          f"matched to distance links; M_check "
                          f"{o['M_check']:.1e}; Cq_check {o['Cq_check']:.1e}")
                    shown = True
                for k, v in o.items():
                    if isinstance(v, (int, float)):
                        worst[k] = max(worst.get(k, 0.0), float(v))
            for s in dead:
                o = AC.probe(fam, s, assembly=args.assembly, fd_h=0.0,
                             native=False)
                dworst = max(dworst, o["gen"])
                for k in ("body", "force", "guard", "res_dyn", "res_con",
                          "Cq_check", "M_check", "shift"):
                    worst[k] = max(worst.get(k, 0.0), o[k])
            res = max(worst["res_dyn"], worst["res_con"])
            ok = worst["gen"] <= TOL and (not dead or dworst <= TOL)
            health = []
            if worst["Cq_check"] > 1e-9:
                health.append("Cq form")
            if worst.get("qc_fd", 0.0) > 1e-6:
                health.append("Qc")
            if worst["M_check"] > 1e-12:
                health.append("M")
            if worst["shift"] > 1e-12:
                health.append(f"state moved {worst['shift']:.0e}")
            status = ("agree" if ok else "DISAGREE") + \
                     (f"  CHECK: {', '.join(health)}" if health else "")
            nat = worst.get("native")
            qcf = worst.get("qc_fd")
            print(f"{label:<30}{worst['gen']:>9.1e}"
                  f"{(f'{dworst:.1e}' if dead else '--'):>9}"
                  f"{worst['body']:>9.1e}{worst['force']:>9.1e}"
                  f"{(f'{nat:.1e}' if nat is not None else '--'):>9}"
                  f"{worst['guard']:>9.1e}{res:>9.1e}"
                  f"{(f'{qcf:.1e}' if qcf is not None else '--'):>9}  {status}"
                  f"   ({time.time() - t0:.1f}s)")
            rows.append({"family": name, "case": label.strip(), "status": status,
                         "dead": dworst if dead else None, **worst})
        except Exception as e:
            print(f"{label:<30}  refused: {type(e).__name__}: {str(e)[:70]}")
            rows.append({"family": name, "case": label.strip(),
                         "status": "refused",
                         "error": f"{type(e).__name__}: {e}"[:200]})
    print()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="all",
                    choices=["all"] + list(FAMILIES))
    ap.add_argument("--assembly", action="store_true",
                    help="apply the state with DoAssembly instead of Update")
    ap.add_argument("--no-fd", action="store_true", help="skip the Qc check")
    ap.add_argument("--no-native", action="store_true",
                    help="skip the DoAssembly/GetPosDt2 contrast column")
    args = ap.parse_args()

    import pychrono
    print(f"pychrono from {os.path.dirname(pychrono.__file__)}")
    print(f"probes {K_PROBE}/case, tolerance {TOL:g} relative "
          f"(gen and dead decide the status)\n")
    print(f"native contrast column: DoAssembly with solver {AC.NATIVE_SOLVER}")

    names = list(FAMILIES) if args.family == "all" else [args.family]
    rows = []
    for name in names:
        rows += run_family(name, args)

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    out = os.path.join(HERE, "out", f"chrono_{args.family}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    dis = [r for r in rows if r["status"].startswith("DISAGREE")]
    ref = [r for r in rows if r["status"] == "refused"]
    print(f"cases {len(rows)}, disagreements {len(dis)}, refusals {len(ref)}")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
