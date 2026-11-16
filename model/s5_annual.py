"""S5 - Mumbai climate year: monthly wet bulb (IMD Santacruz 1991-2020 normals), economiser fraction,
annual cooling energy for baseline vs optimised set-points. Monthly-mean wet bulb (no diurnal bins) -> state."""
import json, calendar, numpy as np
import s5_plant as S, s5_optimize as O
wb = S.mumbai_wetbulb()
days = [31,28,31,30,31,30,31,31,30,31,30,31]
cases = {
 "Baseline (24 C, psi 0, CHW 14, TCS 35 C, 0.5 L/min)": dict(T_in=24.0, psi=0.0, T_chw=14.0, T_liq_in=35.0, lpm=0.5, fixed_chw=True),
 "Optimised air side, TCS 35 C": dict(T_in=27.0, psi=None, T_liq_in=35.0, lpm=0.5),
 "Optimised air side, TCS 40 C (W40), 0.35 L/min": dict(T_in=27.0, psi=None, T_liq_in=40.0, lpm=0.35),
}
out = {"wb": wb, "cases": {}}
for name, c in cases.items():
    rows = []
    for (m, d), nd in zip(wb.items(), days):
        base = {**S.DEFAULT_X, "T_wb": d["wb_mean"], "T_in": c["T_in"], "T_liq_in": c["T_liq_in"], "lpm": c["lpm"], "beta": 0.05}
        if c.get("fixed_chw"):
            r = S.system({**base, "psi": c["psi"], "T_chw": c["T_chw"]})
        else:
            best = None
            for psi in np.arange(0.0, 0.31, 0.025):
                rr = O.best_chw({**base, "psi": float(psi)})
                if rr is not None and (best is None or rr["W_cool_total"] < best["W_cool_total"]):
                    best = rr
            r = best
        rows.append(dict(month=m, wb=d["wb_mean"], W=r["W_cool_total"], Wfac=r["W_cool_facility"], f_econ=r["f_econ"],
                         psi=r["x"]["psi"], T_chw=r["x"]["T_chw"], COP=r["COP_cool"], pPUE=r["pPUE"],
                         P_it=r["P_it_wall"], hours=nd*24, Wb={k: float(v) for k, v in r["W"].items()}))
        print(name[:30], m, round(d["wb_mean"],2), round(r["W_cool_total"]), round(r["f_econ"],2), round(r["x"]["psi"],3), round(r["x"]["T_chw"],2))
    E = sum(x["W"]*x["hours"] for x in rows)/1e6; Eit = sum(x["P_it"]*x["hours"] for x in rows)/1e6
    Efac = sum(x["Wfac"]*x["hours"] for x in rows)/1e6
    out["cases"][name] = dict(rows=rows, E_cool_MWh=E, E_it_MWh=Eit, pPUE_annual=(Eit+Efac)/Eit)
    print(name, "E_cool MWh/yr", round(E,1), "annual pPUE", round((Eit+Efac)/Eit,4))
json.dump(out, open("results/s5_annual.json","w"), indent=1, default=float)
