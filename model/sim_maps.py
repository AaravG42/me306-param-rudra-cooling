"""
Multi-variable optimisation maps on the C-DAC configuration (619 nodes, in-row coolers, tower, wet bulb 25 C,
full load). Every map cell is a full coupled solve; where noted the chilled-water set-point is optimised in
the cell (inner problem), so each map shows the best achievable cooling power for its two coordinates.

  map_psi_beta : psi (recirculation) x beta (leakage),   T_in = 27 C, T_chw optimised  -> seal-quality sweet spot
  map_psi_tin  : psi x server inlet T,                    beta = 0.05, T_chw optimised
  map_psi_tchw : psi x chilled-water T (no inner opt.),   T_in = 27 C, beta = 0.05      -> feasibility boundary
Run: python3 sim_maps.py   (~3 min on 2 cores) -> results/sim_maps.json
"""
import json, time
import numpy as np
from multiprocessing import Pool
import s5_plant as S
import s5_optimize as O

PSI = np.round(np.arange(0.0, 0.401, 0.025), 4)
BETA = np.round(np.arange(0.0, 0.2001, 0.0125), 4)
TIN = np.round(np.arange(18.0, 27.01, 0.5), 2)
TCHW = np.round(np.arange(8.0, 20.01, 0.5), 2)

def cell(args):
    kind, a, b = args
    x = S.cdac_x()
    if kind == "psi_beta":
        r = O.best_chw({**x, "T_in": 27.0, "psi": float(a), "beta": float(b)})
        return args, (None if r is None else (r["W_cool_total"], r["x"]["T_chw"]))
    if kind == "psi_tin":
        r = O.best_chw({**x, "T_in": float(b), "psi": float(a), "beta": 0.05})
        return args, (None if r is None else (r["W_cool_total"], r["x"]["T_chw"]))
    if kind == "psi_tchw":
        r = S.system({**x, "T_in": 27.0, "psi": float(a), "beta": 0.05, "T_chw": float(b)}, detail=False)
        return args, ((r["W_cool_total"], float(b)) if r["feasible"] else None)

def main():
    t0 = time.time()
    jobs = ([("psi_beta", p, b) for b in BETA for p in PSI] + [("psi_tin", p, t) for t in TIN for p in PSI]
            + [("psi_tchw", p, c) for c in TCHW for p in PSI])
    with Pool(2) as pool:
        res = dict(pool.map(cell, jobs, chunksize=4))
    out = {}
    for kind, ys in [("psi_beta", BETA), ("psi_tin", TIN), ("psi_tchw", TCHW)]:
        W = [[(res[(kind, p, y)][0] if res[(kind, p, y)] else None) for p in PSI] for y in ys]
        C = [[(res[(kind, p, y)][1] if res[(kind, p, y)] else None) for p in PSI] for y in ys]
        out[kind] = dict(psi=PSI.tolist(), y=ys.tolist(), W=W, T_chw=C)
    json.dump(out, open("results/sim_maps.json", "w"), default=float)
    print(f"{len(jobs)} cells in {time.time()-t0:.0f} s")

if __name__ == "__main__":
    main()
