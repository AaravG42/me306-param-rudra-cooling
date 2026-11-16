"""
SIM B - Hour-by-hour Mumbai year for the C-DAC IIT Bombay configuration, by workload and heat-rejection type.

What the monthly analysis cannot see: monthly-mean wet bulbs hide the humid monsoon hours in which the
tower can no longer deliver the facility water the CDU needs, so the trim chiller runs. Those hours, the
peak cooling power and the gap between hourly and monthly energy are what this run adds.

Weather: Mumbai Shivaji Intl AP TMYx 2009-2023 (climate.onebuilding.org, via the ideas-lab-nus GitHub
mirror), 8760 h of dry bulb, RH and pressure; wet bulb from CoolProp psychrometrics.

Quasi-steady: each hour is a steady state of s5_plant.system() at that hour's sink temperature. The plant
response depends on the weather only through the sink (wet bulb for a tower, dry bulb for a dry cooler),
so each case is solved on a 0.5 K sink grid (CHW set-point optimised at each point where the case allows)
and interpolated to the 8760 hours.

Workloads (per node, applied to all 619 nodes; CPU share via the S2 utilisation model, DIMM share decoupled):
  thermal virus : CPU at TDP (u = 1), memory near idle (cache-resident)            -> load 1.0, mem 0.0
  STREAM        : memory bandwidth bound, DIMMs at full activity; CPU u = 0.6 is an
                  ASSUMPTION to be replaced by a measured package power            -> load 0.6, mem 1.0
  sporadic      : general users, 30 % CPU and memory                                -> load 0.3, mem 0.3
Set-points:
  C-DAC design  : server inlet 24 degC, psi 0, CHW 14 degC, secondary 37 degC at 0.5 L/min per node
  recommended   : server inlet 27 degC, psi 0.15, CHW optimised per hour, secondary 37 degC (C-DAC fixed)
Run: python3 sim_hourly.py
"""
import csv, json, os, sys, time
import numpy as np
from multiprocessing import Pool
import CoolProp.CoolProp as CP
import s5_plant as S
import s5_optimize as O

RES = "results/"; FIG = "figures/"
EPW = "weather/IND_MH_Mumbai-Shivaji.Intl.AP.430030_TMYx.2009-2023.epw"

WORKLOADS = {"thermal virus": dict(load=1.0, mem_load=0.0),
             "STREAM": dict(load=0.6, mem_load=1.0),
             "sporadic (30 %)": dict(load=0.3, mem_load=0.3)}
SETPOINTS = {"C-DAC design": dict(T_in=24.0, psi=0.0, T_chw=14.0, opt_chw=False),
             "recommended": dict(T_in=27.0, psi=0.15, opt_chw=True)}
SINKS = ["tower", "dry"]

def read_weather():
    rows = list(csv.reader(open(EPW)))[8:]
    tdb = np.array([float(r[6]) for r in rows]); rh = np.array([float(r[8]) for r in rows])
    pa = np.array([float(r[9]) for r in rows]); mon = np.array([int(r[1]) for r in rows])
    twb = np.array([CP.HAPropsSI("B", "T", t + 273.15, "P", p, "R", min(max(h, 1.0), 100.0) / 100) - 273.15
                    for t, h, p in zip(tdb, rh, pa)])
    return dict(tdb=tdb, twb=twb, rh=rh, month=mon)

KEYS = ["W_cool_total", "W_cool_facility", "P_it_wall", "f_econ", "Q_trim", "T_fws", "T_case_max", "T_tcs_ret"]

def point(args):
    sp_name, wl_name, sink, T = args
    sp, wl = SETPOINTS[sp_name], WORKLOADS[wl_name]
    x = S.cdac_x(T_in=sp["T_in"], psi=sp["psi"], sink=sink, **wl)
    if sink == "tower":
        x["T_wb"] = T
    else:
        x["T_db"] = T; x["T_wb"] = T            # T_wb unused for a dry sink
    if sp["opt_chw"]:
        r = O.best_chw(x)
        if r is None:
            return args, None
    else:
        r = S.system({**x, "T_chw": sp["T_chw"]}, detail=False)
        if not r["feasible"]:
            return args, None
    d = {k: float(r[k]) for k in KEYS}
    d.update(T_chw=float(r["x"]["T_chw"]), W={k: float(v) for k, v in r["W"].items()})
    return args, d

def main():
    os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    w = read_weather()
    print(f"weather: Tdb {w['tdb'].min():.1f}..{w['tdb'].max():.1f} C, Twb {w['twb'].min():.1f}..{w['twb'].max():.1f} C, "
          f"Twb 99.6 % {np.percentile(w['twb'], 99.6):.1f} C")
    grids = {"tower": np.arange(np.floor(w["twb"].min()) - 0.5, np.ceil(w["twb"].max()) + 1.0, 0.5),
             "dry": np.arange(np.floor(w["tdb"].min()) - 0.5, np.ceil(w["tdb"].max()) + 1.0, 0.5)}
    jobs = [(sp, wl, sk, float(T)) for sp in SETPOINTS for wl in WORKLOADS for sk in SINKS for T in grids[sk]]
    t0 = time.time()
    with Pool(2) as pool:
        res = dict(pool.map(point, jobs, chunksize=4))
    print(f"{len(jobs)} grid points in {time.time()-t0:.0f} s")
    out = dict(weather=dict(source=EPW, tdb_max=float(w["tdb"].max()), twb_max=float(w["twb"].max()),
                            twb_p996=float(np.percentile(w["twb"], 99.6)),
                            monthly_twb=[float(w["twb"][w["month"] == m].mean()) for m in range(1, 13)],
                            monthly_tdb=[float(w["tdb"][w["month"] == m].mean()) for m in range(1, 13)]),
               cases={})
    hourly_store = {}
    for sp in SETPOINTS:
        for wl in WORKLOADS:
            for sk in SINKS:
                g = grids[sk]
                pts = [res[(sp, wl, sk, float(T))] for T in g]
                ok = np.array([p is not None for p in pts])
                T_h = w["twb"] if sk == "tower" else w["tdb"]
                if not ok.all():
                    bad = g[~ok]
                    print(f"  ! {sp}/{wl}/{sk}: infeasible at sink {bad.min():.1f}..{bad.max():.1f} C")
                gg = g[ok]
                arr = {k: np.array([p[k] for p in pts if p is not None]) for k in KEYS + ["T_chw"]}
                comp = {k: np.array([p["W"][k] for p in pts if p is not None]) for k in pts[int(np.argmax(ok))]["W"]}
                hr = {k: np.interp(T_h, gg, v) for k, v in arr.items()}
                hc = {k: np.interp(T_h, gg, v) for k, v in comp.items()}
                infeasible_h = int(np.sum((T_h > gg.max()) | (T_h < gg.min())))
                trim_h = int(np.sum(hr["Q_trim"] > 100.0))
                E = hr["W_cool_total"].sum() / 1e6; E_it = hr["P_it_wall"].sum() / 1e6
                E_fac = hr["W_cool_facility"].sum() / 1e6
                # monthly-mean method (what the October report did), same model and grid
                E_mon = 0.0
                for m in range(1, 13):
                    sel = w["month"] == m
                    E_mon += np.interp(T_h[sel].mean(), gg, arr["W_cool_total"]) * sel.sum() / 1e6
                key = f"{sp} | {wl} | {sk}"
                out["cases"][key] = dict(
                    setpoints=sp, workload=wl, sink=sk,
                    E_cool_MWh=float(E), E_it_MWh=float(E_it), pPUE_annual=float((E_it + E_fac) / E_it),
                    E_cool_monthly_method_MWh=float(E_mon), hourly_minus_monthly_pct=float((E / E_mon - 1) * 100),
                    W_cool_mean_kW=float(hr["W_cool_total"].mean() / 1e3), W_cool_peak_kW=float(hr["W_cool_total"].max() / 1e3),
                    W_cool_p99_kW=float(np.percentile(hr["W_cool_total"], 99) / 1e3),
                    trim_hours=trim_h, econ_full_hours=int(np.sum(hr["f_econ"] >= 0.999)),
                    E_trim_MWh=float(hc["trim_chiller"].sum() / 1e6),
                    T_case_max=float(hr["T_case_max"].max()), hours_outside_grid=infeasible_h,
                    E_by_component_MWh={k: float(v.sum() / 1e6) for k, v in hc.items()},
                    monthly_E_MWh=[float(hr["W_cool_total"][w["month"] == m].sum() / 1e6) for m in range(1, 13)],
                    monthly_trim_hours=[int(np.sum((hr["Q_trim"] > 100.0) & (w["month"] == m))) for m in range(1, 13)])
                hourly_store[key] = hr["W_cool_total"] / 1e3
                c = out["cases"][key]
                print(f"{key:45s} E {E:6.1f} MWh (monthly method {E_mon:6.1f}, {c['hourly_minus_monthly_pct']:+.1f} %), "
                      f"pPUE {c['pPUE_annual']:.3f}, peak {c['W_cool_peak_kW']:.1f} kW, trim h {trim_h}, "
                      f"E_trim {c['E_trim_MWh']:.1f}")
    json.dump(out, open(RES + "simB_hourly.json", "w"), indent=1, default=float)
    np.savez_compressed(RES + "simB_hourly_series.npz", tdb=w["tdb"], twb=w["twb"],
                        **{k.replace(" | ", "__").replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct"): v
                           for k, v in hourly_store.items()})
    figs(out, w, hourly_store)

def figs(out, w, hs):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e3"})
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    days = np.arange(8760) / 24
    for key, col in [("C-DAC design | thermal virus | tower", "#2a78d6"), ("C-DAC design | thermal virus | dry", "#eb6834")]:
        d = hs[key].reshape(365, 24)
        ax[0].plot(np.arange(365), d.mean(1), color=col, lw=1.2, label=key.split(" | ")[2])
        ax[0].fill_between(np.arange(365), d.min(1), d.max(1), color=col, alpha=0.18, lw=0)
    ax[0].set_xlabel("Day of year"); ax[0].set_ylabel("Cooling power, kW (daily mean, min-max)")
    ax[0].set_title("C-DAC set-points, thermal virus"); ax[0].legend(fontsize=7)
    for key, col in [("C-DAC design | thermal virus | tower", "#2a78d6"), ("C-DAC design | thermal virus | dry", "#eb6834"),
                     ("recommended | thermal virus | tower", "#1baf7a")]:
        ax[1].plot(np.arange(1, 8761), np.sort(hs[key])[::-1], color=col, label=key.replace(" | thermal virus", ""))
    ax[1].set_xlabel("Hours in the year (sorted)"); ax[1].set_ylabel("Cooling power, kW")
    ax[1].set_title("Load-duration curve"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG + "simB_hourly.png", dpi=160)

if __name__ == "__main__":
    main()
