"""
S5 — multi-variable optimisation maps, sweet-spot search and sensitivities.
Writes results/s5_*.json|csv used by the report. Run: python3 s5_optimize.py
"""
import json, time, math
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize_scalar, differential_evolution
import params as P
import s5_plant as S

OUT = "results/"
BIG = 1e9

def w_at(x):
    try:
        r = S.system(x, detail=False)
    except Exception:
        return BIG, None
    return (r["W_cool_total"] if r["feasible"] else BIG), r

def best_chw(x, lo=6.0, hi=20.0):
    """Inner problem: CHW supply temperature is a free plant set-point -> pick the one minimising W_cool.
    Coarse grid (0.5 K) then bounded Brent refinement inside the best bracket."""
    grid = np.arange(lo, hi + 1e-9, 0.5)
    vals = [w_at({**x, "T_chw": t})[0] for t in grid]
    i = int(np.argmin(vals))
    if vals[i] >= BIG:
        return None
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(lambda t: w_at({**x, "T_chw": t})[0], bounds=(a, b), method="bounded",
                          options=dict(xatol=0.02))
    t = res.x if res.fun <= vals[i] else grid[i]
    _, r = w_at({**x, "T_chw": t})
    return S.system({**x, "T_chw": t}, detail=True)

def _cell(args):
    x = args
    r = best_chw(x)
    if r is None:
        return dict(x=x, feasible=False)
    keep = ["W_cool_total", "W_cool_facility", "COP_cool", "pPUE", "T_c", "T_ret", "m_c", "m_w", "T_w_out",
            "COP_chiller", "SHI", "RHI", "RTI", "irc_speed", "X_dest_total", "f_econ"]
    d = {k: float(r[k]) for k in keep}
    d.update(T_chw=float(r["x"]["T_chw"]), W={k: float(v) for k, v in r["W"].items()},
             X={k: float(v) for k, v in r["X"].items()}, feasible=True, x={k: v for k, v in x.items()})
    return d

def grid_map(var1, vals1, var2, vals2, base, pool):
    jobs = [{**base, var1: float(a), var2: float(b)} for b in vals2 for a in vals1]
    res = pool.map(_cell, jobs, chunksize=4)
    return res

def main():
    t0 = time.time()
    base = dict(S.DEFAULT_X)
    out = {}
    with Pool(2) as pool:
        # 1) psi x beta at T_in = 24
        psis = np.round(np.arange(0.0, 0.401, 0.025), 3)
        betas = np.round(np.array([0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25]), 3)
        out["map_psi_beta"] = dict(psi=psis.tolist(), beta=betas.tolist(),
                                   cells=grid_map("psi", psis, "beta", betas, {**base, "T_in": 24.0}, pool))
        print("map psi-beta", time.time() - t0)
        # 2) psi x T_in at beta = 0.05
        tins = np.round(np.arange(18.0, 32.01, 1.0), 2)
        out["map_psi_tin"] = dict(psi=psis.tolist(), T_in=tins.tolist(),
                                  cells=grid_map("psi", psis, "T_in", tins, {**base, "beta": 0.05}, pool))
        print("map psi-Tin", time.time() - t0)
        # 3) part load slices: optimum psi at load 0.3/0.6/1.0, beta 0.05, T_in 24 and 27
        out["load"] = {}
        for L in (0.3, 0.6, 1.0):
            out["load"][str(L)] = grid_map("psi", psis, "T_in", np.array([24.0, 27.0]),
                                           {**base, "beta": 0.05, "load": L}, pool)
        print("load", time.time() - t0)
    json.dump(out, open(OUT + "s5_maps.json", "w"))
    # 4) global optimiser (DE) per beta, recommended envelope T_in <= 27 and allowable <= 32
    opt = {}
    for env, tmax in (("recommended", 27.0), ("allowable_A1", 32.0)):
        for b in (0.0, 0.02, 0.05, 0.10, 0.20):
            res = differential_evolution(S.objective, ((18.0, tmax), (0.0, 0.40), (6.0, 20.0)),
                                         args=({**base, "beta": b},), seed=7, tol=1e-8, maxiter=40,
                                         popsize=10, polish=True, init="sobol")
            r = S.system({**base, "beta": b, "T_in": res.x[0], "psi": res.x[1], "T_chw": res.x[2]})
            opt[f"{env}|{b}"] = dict(T_in=res.x[0], psi=res.x[1], T_chw=res.x[2], W=res.fun,
                                     COP=r["COP_cool"], pPUE=r["pPUE"], T_c=r["T_c"], m_c=r["m_c"],
                                     Wb={k: float(v) for k, v in r["W"].items()},
                                     X={k: float(v) for k, v in r["X"].items()}, nfev=res.nfev)
            print(env, b, np.round(res.x, 3), round(res.fun, 1), time.time() - t0)
    json.dump(opt, open(OUT + "s5_opt.json", "w"), indent=1)
    # 5) liquid side
    liq = []
    for T in np.arange(30.0, 45.01, 1.0):
        for l in (0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
            r = S.system({**base, "T_liq_in": float(T), "lpm": l}, detail=True)
            liq.append(dict(T_liq_in=float(T), lpm=l, W_tcs=r["W"]["tcs_pump"], W_trim=r["W"]["trim_chiller"],
                            W_twr_pumps=r["W"]["cw_fw_pumps"], W_twr_fans=r["W"]["tower_fans"],
                            W_total=r["W_cool_total"], f_econ=r["f_econ"], T_case=r["T_case_max"],
                            T_tcs_ret=r["T_tcs_ret"], X_tcs_ret=r["X_tcs_return"],
                            X_cp=r["X"]["server: cold plates (S1)"]))
    json.dump(liq, open(OUT + "s5_liquid.json", "w"))
    print("done", time.time() - t0)

if __name__ == "__main__":
    main()
