"""
Workload cases at the design wet bulb (25 C) on the C-DAC configuration (Student 2 / Student 5).
  thermal virus : every core at TDP, memory idle            (load 1.0, mem 0.0)
  LINPACK/DGEMM : cores at TDP (AVX-512), memory moderate   (load 1.0, mem 0.4)  -- compute-bound
  STREAM        : cores partly stalled, memory saturated    (load 0.6, mem 1.0)  -- memory-bound
  sporadic      : general users                             (load 0.3, mem 0.3)
For each: node power, liquid/air split, C-DAC set-points, best psi = 0, best overall (T_in 27 C).
Run: python3 sim_workloads.py -> results/sim_workloads.json
"""
import json, numpy as np
import s5_plant as S, s5_optimize as O
WL = {"thermal virus": (1.0, 0.0), "LINPACK / DGEMM": (1.0, 0.4), "STREAM": (0.6, 1.0), "sporadic (30 %)": (0.3, 0.3)}
out = []
for name, (l, m) in WL.items():
    x = S.cdac_x(load=l, mem_load=m)
    base = S.system({**x, "T_in": 24.0, "psi": 0.0, "T_chw": 14.0})
    trad = O.best_chw({**x, "T_in": 27.0, "psi": 0.0})
    best = None
    for p in np.arange(0, 0.326, 0.025):
        r = O.best_chw({**x, "T_in": 27.0, "psi": float(p)})
        if r is not None and (best is None or r["W_cool_total"] < best["W_cool_total"]): best = r
    out.append(dict(workload=name, load=l, mem_load=m, P_it_kW=base["P_it_wall"] / 1e3,
                    Q_liq_kW=base["Q_liquid"] / 1e3 if "Q_liquid" in base else None,
                    Q_air_kW=base["Q_air"] / 1e3 if "Q_air" in base else None,
                    W_base=base["W_cool_total"], W_trad=trad["W_cool_total"], W_opt=best["W_cool_total"],
                    psi_opt=best["x"]["psi"], T_chw_opt=best["x"]["T_chw"], T_case_max=base.get("T_case_max")))
    print(out[-1])
json.dump(out, open("results/sim_workloads.json", "w"), indent=1, default=float)
