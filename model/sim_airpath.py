"""
v3 study - C-DAC configuration re-optimised, raised-floor air path, and the air-moving-power argument.

Configurations (all at the C-DAC IIT Bombay installation: 619 nodes, 37 degC secondary, CHx750, tower):
  hybrid | in-row          : CPUs on cold plates; in-row coolers on a slab (October model)
  hybrid | raised, wall    : same coolers acting as CRAHs: underfloor plenum + 25 % tiles + return path;
                             recirculation by fans in the containment wall (hot -> cold aisle directly)
  hybrid | raised, plenum  : as above, recirculation ducted into the plenum, so it also crosses the tiles
  air-only | in-row / raised, wall : reference machine with the SAME nodes but no cold plates (capture 0),
                             chassis air rise capped at 20 K for the CPU heat sinks, coolers re-counted.
For each: baseline (24 degC inlet, psi 0, CHW 14 degC); "traditional enclosed aisle" = best set-points with
psi = 0 (inlet 27 degC, CHW optimised); "bypass recirculation" = psi also optimised. Loads 100/60/30 %.
Leakage sweep beta 0..0.20 for hybrid in-row and raised-wall. One DE check that 27 degC inlet is optimal.
Run: python3 sim_airpath.py   (~10-15 min on 2 cores)
"""
import json, time, os
import numpy as np
from multiprocessing import Pool
import s5_plant as S
import s5_optimize as O

RES = "results/"; FIG = "figures/"
PSI = np.round(np.arange(0.0, 0.401, 0.025), 3)
GROUPS = dict(
    air_moving=["server_fans", "irc_fans", "bypass_fans"],
    liquid_moving=["tcs_pump", "chw_pump", "cw_fw_pumps"],
    refrigeration=["chiller", "trim_chiller"],
    heat_rejection_fans=["tower_fans"])

CONFIGS = {
    "hybrid | in-row": dict(),
    "hybrid | raised floor, wall bypass": dict(air_path="raised_floor", bypass_route="wall"),
    "hybrid | raised floor, plenum bypass": dict(air_path="raised_floor", bypass_route="plenum"),
    "air-only | in-row": dict(capture=0.003, dt_max_air=20.0),
    "air-only | raised floor, wall bypass": dict(capture=0.003, dt_max_air=20.0, air_path="raised_floor", bypass_route="wall"),
}

def base_x(cfg, load=1.0, beta=0.05, n_irc=None):
    return S.cdac_x(**CONFIGS[cfg], load=load, beta=beta, **({"n_irc": n_irc} if n_irc else {}))

def n_irc_for(cfg):
    """Coolers: as many as the October rule (minimum feasible at baseline + 1 redundant)."""
    for n in range(4, 40):
        r = S.system({**base_x(cfg, n_irc=n), "T_in": 24.0, "psi": 0.0, "T_chw": 14.0}, detail=False)
        if r["feasible"]:
            return n + 1
    raise RuntimeError(cfg)

def summarise(r):
    W = r["W"]
    g = {k: sum(W[c] for c in v) for k, v in GROUPS.items()}
    return dict(W_cool_total=r["W_cool_total"], W=dict(W), groups=g, T_in=r["x"]["T_in"], psi=r["x"]["psi"],
                T_chw=r["x"]["T_chw"], COP=r["COP_cool"], pPUE=r["pPUE"], P_it=r["P_it_wall"], Q_air=r["Q_air"],
                Q_liq=r["Q_liq"], m_s=r["m_s"], m_c=r["m_c"], dp_crah_ext=r.get("dp_crah_ext", 0.0),
                X_dest_total=r.get("X_dest_total", float("nan")), T_case_max=r["T_case_max"])

def job(args):
    cfg, load, beta, n_irc, psi, T_in = args
    x = {**base_x(cfg, load, beta, n_irc), "T_in": T_in, "psi": float(psi)}
    r = O.best_chw(x)
    return args, (None if r is None else summarise(r))

def baseline(cfg, load, beta, n_irc):
    r = S.system({**base_x(cfg, load, beta, n_irc), "T_in": 24.0, "psi": 0.0, "T_chw": 14.0}, detail=True)
    return summarise(r) if r["feasible"] else None

def main():
    os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    t0 = time.time()
    nirc = {c: n_irc_for(c) for c in CONFIGS}
    print("coolers:", nirc)
    jobs = []
    for cfg in CONFIGS:
        for load in (1.0, 0.6, 0.3):
            for psi in PSI:
                jobs.append((cfg, load, 0.05, nirc[cfg], psi, 27.0))
    for cfg in ("hybrid | in-row", "hybrid | raised floor, wall bypass"):
        for beta in (0.0, 0.02, 0.10, 0.20):
            for psi in PSI:
                jobs.append((cfg, 1.0, beta, nirc[cfg], psi, 27.0))
    with Pool(2) as pool:
        res = dict(pool.map(job, jobs, chunksize=3))
    print(f"{len(jobs)} psi points in {time.time()-t0:.0f} s")

    out = dict(n_irc=nirc, cases={}, beta_sweep={})
    for cfg in CONFIGS:
        for load in (1.0, 0.6, 0.3):
            curve = [(p, res[(cfg, load, 0.05, nirc[cfg], p, 27.0)]) for p in PSI]
            ok = [(p, v) for p, v in curve if v is not None]
            trad = res[(cfg, load, 0.05, nirc[cfg], 0.0, 27.0)]
            best_p, best = min(ok, key=lambda t: t[1]["W_cool_total"])
            base = baseline(cfg, load, 0.05, nirc[cfg])
            key = f"{cfg} | load {int(load*100)}"
            out["cases"][key] = dict(
                config=cfg, load=load, n_irc=nirc[cfg], baseline=base, traditional=trad, bypass=best,
                psi_opt=float(best_p), psi_max_feasible=float(max(p for p, _ in ok)),
                saving_bypass_W=trad["W_cool_total"] - best["W_cool_total"],
                saving_bypass_pct=100 * (trad["W_cool_total"] - best["W_cool_total"]) / trad["W_cool_total"],
                saving_total_W=(base["W_cool_total"] - best["W_cool_total"]) if base else None,
                saving_total_pct=(100 * (base["W_cool_total"] - best["W_cool_total"]) / base["W_cool_total"]) if base else None,
                psi_curve=[dict(psi=float(p), W=(v["W_cool_total"] if v else None),
                                air=(v["groups"]["air_moving"] if v else None),
                                refr=(v["groups"]["refrigeration"] if v else None)) for p, v in curve])
            c = out["cases"][key]
            print(f"{key:48s} base {base['W_cool_total']/1e3 if base else float('nan'):6.2f}  trad {trad['W_cool_total']/1e3:6.2f}  "
                  f"bypass {best['W_cool_total']/1e3:6.2f} kW (psi {best_p:.3f}, max {c['psi_max_feasible']:.3f})  "
                  f"bypass saves {c['saving_bypass_W']/1e3:5.2f} kW ({c['saving_bypass_pct']:4.1f} %), total {c['saving_total_pct'] or float('nan'):4.1f} %  "
                  f"air share base {100*base['groups']['air_moving']/base['W_cool_total'] if base else float('nan'):4.1f} %")
    for cfg in ("hybrid | in-row", "hybrid | raised floor, wall bypass"):
        rows = []
        for beta in (0.0, 0.02, 0.05, 0.10, 0.20):
            curve = [(p, res.get((cfg, 1.0, beta, nirc[cfg], p, 27.0))) for p in PSI]
            ok = [(p, v) for p, v in curve if v is not None]
            bp, bv = min(ok, key=lambda t: t[1]["W_cool_total"])
            tv = dict(curve)[0.0]
            rows.append(dict(beta=beta, psi_opt=float(bp), T_chw=bv["T_chw"], W_bypass=bv["W_cool_total"],
                             W_trad=tv["W_cool_total"] if tv else None))
            print(f"  beta {beta:.2f} {cfg:36s} psi* {bp:.3f}  T_chw {bv['T_chw']:.1f}  W {bv['W_cool_total']/1e3:.2f}  trad {tv['W_cool_total']/1e3:.2f}")
        out["beta_sweep"][cfg] = rows
    # DE check that 27 degC is the right inlet (hybrid in-row and raised wall, full load)
    out["de_check"] = {}
    for cfg in ("hybrid | in-row", "hybrid | raised floor, wall bypass"):
        res_de = S.optimise_air(fixed=base_x(cfg, n_irc=nirc[cfg]), seed=1)
        out["de_check"][cfg] = dict(T_in=float(res_de.x[0]), psi=float(res_de.x[1]), T_chw=float(res_de.x[2]), W=float(res_de.fun))
        print("DE", cfg, out["de_check"][cfg])
    # liquid-side lever inside C-DAC's envelope: secondary 37 -> 40 degC (W4 primary allows it)
    lq = {}
    for T in (33.0, 35.0, 37.0, 40.0):
        r = O.best_chw({**base_x("hybrid | in-row", n_irc=nirc["hybrid | in-row"]), "T_in": 27.0, "psi": out["cases"]["hybrid | in-row | load 100"]["psi_opt"], "T_liq_in": T})
        lq[T] = dict(W=r["W_cool_total"], T_case_max=r["T_case_max"], f_econ=r["f_econ"], T_fws=r["T_fws"])
        print(f"  TCS {T}: W {r['W_cool_total']/1e3:.2f} kW, T_case {r['T_case_max']:.1f}, f_econ {r['f_econ']:.2f}, facility supply {r['T_fws']:.1f}")
    out["liquid_side"] = lq
    json.dump(out, open(RES + "sim_airpath.json", "w"), indent=1, default=float)
    print(f"done in {time.time()-t0:.0f} s")
    figs(out)

def figs(out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e3", "axes.axisbelow": True})
    cfgs = list(CONFIGS)
    short = ["Hybrid\nin-row", "Hybrid\nraised, wall", "Hybrid\nraised, plenum", "Air-only\nin-row", "Air-only\nraised, wall"]
    cols = {"air_moving": "#2a78d6", "refrigeration": "#e34948", "heat_rejection_fans": "#eda100", "liquid_moving": "#1baf7a"}
    lab = {"air_moving": "Air moving (server, cooler, bypass fans)", "refrigeration": "Refrigeration (chillers)",
           "heat_rejection_fans": "Tower fans", "liquid_moving": "Pumps (TCS, CHW, CW/FW)"}
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
    x = np.arange(len(cfgs))
    bottom = np.zeros(len(cfgs))
    for g in ["air_moving", "liquid_moving", "heat_rejection_fans", "refrigeration"]:
        v = np.array([out["cases"][f"{c} | load 100"]["baseline"]["groups"][g] / 1e3 for c in cfgs])
        ax[0].bar(x, v, bottom=bottom, color=cols[g], label=lab[g], width=0.62)
        bottom += v
    for i, c in enumerate(cfgs):
        b = out["cases"][f"{c} | load 100"]["baseline"]
        ax[0].text(i, bottom[i] + 1, f"{100*b['groups']['air_moving']/b['W_cool_total']:.0f} % air", ha="center", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(short, fontsize=8); ax[0].set_ylabel("Cooling power at baseline, kW")
    ax[0].set_title("Where cooling power goes (full load, baseline set-points)"); ax[0].legend(fontsize=7, loc="upper left")
    w = 0.25
    for j, (load, col) in enumerate([(1.0, "#2a78d6"), (0.6, "#6250d6"), (0.3, "#e87ba4")]):
        v = [out["cases"][f"{c} | load {int(load*100)}"]["saving_bypass_pct"] for c in cfgs]
        ax[1].bar(x + (j - 1) * w, v, width=w, color=col, label=f"{int(load*100)} % load")
    for i, c in enumerate(cfgs):
        if max(out["cases"][f"{c} | load {L}"]["saving_bypass_pct"] for L in (100, 60, 30)) < 0.5:
            ax[1].text(i, 0.3, "≈ 0", ha="center", fontsize=8, color="#52514e")
    ax[1].set_xticks(x); ax[1].set_xticklabels(short, fontsize=8)
    ax[1].set_ylabel("Saving from bypass recirculation, % of cooling power")
    ax[1].set_title("Bypass recirculation vs traditional enclosed aisle"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG + "sim_airpath.png", dpi=160)



def dt_sensitivity():
    """Air-only reference: value of bypass recirculation vs the chassis air rise the servers run at."""
    out = []
    for dt in (10.0, 12.0, 15.0, 20.0):
        CONFIGS["air-only | in-row"]["dt_max_air"] = dt
        n = n_irc_for("air-only | in-row")
        curve = [(float(p), job(("air-only | in-row", 1.0, 0.05, n, p, 27.0))[1]) for p in PSI]
        ok = [(p, v) for p, v in curve if v is not None]
        bp, bv = min(ok, key=lambda t: t[1]["W_cool_total"])
        tv = dict(curve)[0.0]
        b = baseline("air-only | in-row", 1.0, 0.05, n)
        out.append(dict(dT_cap=dt, n_irc=n, air_share_base=b["groups"]["air_moving"] / b["W_cool_total"],
                        W_base=b["W_cool_total"], W_trad=tv["W_cool_total"], W_bypass=bv["W_cool_total"], psi_opt=bp,
                        saving_bypass_pct=100 * (tv["W_cool_total"] - bv["W_cool_total"]) / tv["W_cool_total"]))
        print(out[-1])
    CONFIGS["air-only | in-row"]["dt_max_air"] = 20.0
    d = json.load(open(RES + "sim_airpath.json")); d["air_only_dT_sensitivity"] = out
    json.dump(d, open(RES + "sim_airpath.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
    dt_sensitivity()
