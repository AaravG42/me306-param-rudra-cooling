"""Generate every data figure in the ME 306 report from the v3 model and its saved results.

Run from anywhere:  python3 make_figures.py
Model code: rudra_cooling_model_v3/dc_project (imported directly, so every number is the model's own).
Outputs: ../figures/fig_<topic>.pdf (+ .png previews). Naming: fig_<section-topic>, lower case, one per figure.
"""
import json, os, sys
from pathlib import Path
import numpy as np

MODEL = Path(os.environ.get("RUDRA_MODEL", Path(__file__).resolve().parent.parent / "model"))
OUT = Path(__file__).resolve().parent.parent / "figures"
sys.path.insert(0, str(MODEL)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import os; os.chdir(MODEL)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from report_style import (apply_publication_style, finalize_figure, panel_label, PALETTE,
                          OPT, BASE, AIR, LIQ, REFR, REJ, PUMP, GREY, FULL, FULL_TALL, HALF)
import params as P
import s1_liquid as s1
import s2_air as s2
import s3_aisle as s3
import s4_irc as s4
import s5_plant as S
import s5_optimize as O

apply_publication_style()

def place_clabels(ax, cs, fmt, avoid=(), tol=0.09, edge=0.07, spacing=0.16, fontsize=6.0):
    """Label contours only where the label is clear of the axes edges, of the given
    polylines (data coords) and of each other."""
    T = ax.transData + ax.transAxes.inverted()
    tmp = ax.clabel(cs, fmt=fmt, fontsize=fontsize, inline=False)
    cand = [t.get_position() for t in tmp]
    for t in tmp: t.remove()
    A = [T.transform(np.asarray(a, float)) for a in avoid if len(a)]
    keep, kept_ax = [], []
    for xy in cand:
        x, y = T.transform(xy)
        if not (edge < x < 1 - edge and edge < y < 1 - edge): continue
        if any(np.min(np.hypot(P[:, 0] - x, P[:, 1] - y)) < tol for P in A): continue
        if any(np.hypot(x - u, y - v) < spacing for u, v in kept_ax): continue
        keep.append(xy); kept_ax.append((x, y))
    if keep:
        ax.clabel(cs, fmt=fmt, fontsize=fontsize, inline=True, inline_spacing=2, manual=keep)

RES = MODEL / "results"
N = P.CDAC["n_nodes"]


# ----------------------------------------------------------------------------- 1. cold plate
def fig_liquid_coldplate():
    fig, ax = plt.subplots(1, 2, figsize=FULL)
    q = np.linspace(0.15, 2.0, 40)
    Q = 165.0 * 0.92
    temps = [(30, PALETTE["blue_secondary"]), (37, OPT), (45, BASE)]
    for T, c in temps:
        tc = [s1.cold_plate(Q, x / 60000 * s1.coolant_props(T)["rho"], T)["T_case"] for x in q]
        ax[0].plot(q, tc, color=c, label=f"inlet {T} °C")
    # C-DAC operating point: 0.5 L/min through both plates in series -> second plate sees a warmer inlet
    m = 0.5 / 60000 * s1.coolant_props(37)["rho"]
    p1 = s1.cold_plate(Q, m, 37.0); p2 = s1.cold_plate(Q, m, p1["T_out"])
    ax[0].plot([0.5], [p2["T_case"]], "o", color=PALETTE["ink"], ms=4.5, zorder=5)
    ax[0].annotate(f"C-DAC design\n(2nd plate)\n{p2['T_case']:.1f} °C", (0.5, p2["T_case"]), xytext=(0.32, 63),
                   fontsize=7.2, arrowprops=dict(arrowstyle="-", lw=0.7, color=PALETTE["ink"]))
    ax[0].set_xlabel("Coolant flow per cold plate (L/min)"); ax[0].set_ylabel("CPU case temperature (°C)")
    ax[0].set_ylim(30, 74); ax[0].set_xlim(0, 2.05); ax[0].legend(loc="upper right")
    panel_label(ax[0], "a")
    # resistance chain
    flows = [0.25, 0.5, 1.0, 2.0]
    parts = [("R_jc", "die to lid", PALETTE["neutral_dark"]), ("R_tim2", "lid to plate", PALETTE["neutral"]),
             ("R_base", "plate base", PALETTE["green_1"]), ("R_conv_eff", "convection", PALETTE["green_3"]),
             ("R_cal", "coolant warming", LIQ)]
    bottom = np.zeros(len(flows))
    for k, lab, c in parts:
        v = np.array([s1.cold_plate(Q, f / 60000 * s1.coolant_props(37)["rho"], 37.0)[k] for f in flows])
        ax[1].bar(range(len(flows)), v, bottom=bottom, color=c, edgecolor="white", lw=0.6, width=0.62, label=lab)
        bottom += v
    ax[1].set_xticks(range(len(flows))); ax[1].set_xticklabels([f"{f:g}" for f in flows])
    ax[1].set_xlabel("Coolant flow per cold plate (L/min)"); ax[1].set_ylabel("Thermal resistance (K/W)")
    ax[1].legend(loc="upper right", ncol=1, fontsize=6.8, labelspacing=0.3, handlelength=1.2)
    ax[1].set_ylim(0, 0.14)
    panel_label(ax[1], "b")
    return finalize_figure(fig, OUT, "fig_liquid_coldplate")


# ----------------------------------------------------------------------------- 2. server air side
def fig_server_air():
    fig, ax = plt.subplots(1, 2, figsize=FULL, gridspec_kw=dict(width_ratios=[1.15, 1]))
    nb = s2.node_heat_breakdown(1.0, 24.0)
    items = [("CPUs to cold plates", nb["Q_liquid"], LIQ),
             ("CPU heat missed by plates", nb["cpu_residual_to_air"], AIR),
             ("Memory (12 DIMMs)", nb["dimms_to_air"], AIR),
             ("Voltage regulators", nb["vrm_cpu_loss"] + nb["vrm_mem_loss"], AIR),
             ("Power supply loss", nb["psu_loss"], AIR),
             ("Network card", nb["nic_hca"], AIR),
             ("Chipset, SSD, BMC, misc.", nb["pch"] + nb["ssd"] + nb["bmc"] + nb["misc"], AIR),
             ("Server fans", nb["fans"], AIR)]
    y = np.arange(len(items))[::-1]
    for yi, (lab, v, c) in zip(y, items):
        ax[0].barh(yi, v, color=c, height=0.62)
        ax[0].text(v + 4, yi, f"{v:.0f} W", va="center", fontsize=7.5)
    ax[0].set_yticks(y); ax[0].set_yticklabels([i[0] for i in items], fontsize=7.5)
    ax[0].set_xlabel("Heat per node at full load (W)"); ax[0].set_xlim(0, 360)
    ax[0].legend(handles=[Line2D([], [], color=LIQ, lw=6, label=f"to liquid: {nb['Q_liquid']:.0f} W"),
                          Line2D([], [], color=AIR, lw=6, label=f"to air: {nb['Q_air']:.0f} W")],
                 loc="lower right", fontsize=7.2)
    panel_label(ax[0], "a", x=-0.62)
    Tin = np.arange(18, 40.1, 1.0)
    W, D = [], []
    for T in Tin:
        nbT = s2.node_heat_breakdown(1.0, T)
        sa = s2.server_air(T, nbT["Q_air"])
        W.append(sa["W_fan"] * N / 1e3); D.append(sa["T_dimm_case"])
    ax[1].plot(Tin, W, color=AIR, label="Server fan power (pod)")
    ax[1].set_xlabel("Server inlet air temperature (°C)"); ax[1].set_ylabel("Server fan power, all nodes (kW)")
    ax[1].set_ylim(0, max(W) * 1.15)
    a2 = ax[1].twinx(); a2.spines["right"].set_visible(True)
    a2.plot(Tin, D, color=BASE, ls="--", label="Hottest DIMM")
    a2.axhline(85, color=GREY, lw=1.0, ls=":"); a2.set_ylabel("Hottest DIMM case (°C)"); a2.set_ylim(60, 92)
    ax[1].axvspan(18, 27, color=PALETTE["green_1"], lw=0, zorder=0)
    ax[1].text(22.5, max(W) * 1.13, "ASHRAE A1\nrecommended", ha="center", fontsize=7, va="top", color=PUMP)
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, loc="lower right", fontsize=7.0)
    panel_label(ax[1], "b")
    return finalize_figure(fig, OUT, "fig_server_air")


# ----------------------------------------------------------------------------- 3. aisle mixing
def fig_aisle_mixing():
    pa = s2.pod_air(24.0, 1.0, n_nodes=N)
    m_s, Q_air = pa["m_dot_servers"], pa["Q_air_pod"]
    psis = np.linspace(0, 0.4, 41)
    fig, ax = plt.subplots(1, 2, figsize=FULL)
    for beta, ls in [(0.0, ":"), (0.05, "-"), (0.20, "--")]:
        r = [s3.solve_for_inlet(m_s, Q_air, 24.0, p, beta) for p in psis]
        ax[0].plot(psis, [x["T_c"] for x in r], color=OPT, ls=ls)
        ax[1].plot(psis, [x["m_c"] / m_s for x in r], color=AIR, ls=ls, label=f"leakage β = {beta:.2f}")
        if beta == 0.05:
            ax[0].plot(psis, [x["T_ret"] for x in r], color=BASE, ls=ls)
    ax[0].text(0.02, 33.4, "air returning to the coolers", color=BASE, fontsize=7.5)
    ax[0].text(0.02, 16.5, "air the coolers must supply", color=OPT, fontsize=7.5)
    ax[0].axhline(24, color=GREY, lw=0.9); ax[0].text(0.395, 24.3, "server inlet 24 °C", ha="right", fontsize=7, color=PUMP)
    ax[0].set_xlabel("Recirculation fraction ψ"); ax[0].set_ylabel("Air temperature (°C)"); ax[0].set_ylim(14, 38)
    ax[1].set_xlabel("Recirculation fraction ψ"); ax[1].set_ylabel("Cooler air flow / server air flow")
    ax[1].legend(loc="upper right")
    panel_label(ax[0], "a"); panel_label(ax[1], "b")
    return finalize_figure(fig, OUT, "fig_aisle_mixing")


# ----------------------------------------------------------------------------- 4. coil validation
def fig_coil():
    rows = s4.validation_table()
    fig, ax = plt.subplots(1, 2, figsize=FULL)
    ds = np.array([r["Q_ds_gross"] for r in rows]) / 1e3; md = np.array([r["Q_model"] for r in rows]) / 1e3
    lo, hi = 10, 45
    ax[0].fill_between([lo, hi], [lo * 0.95, hi * 0.95], [lo * 1.05, hi * 1.05], color=PALETTE["green_1"], lw=0)
    ax[0].plot([lo, hi], [lo, hi], color=PUMP, lw=0.9)
    ax[0].plot(ds, md, "o", color=OPT, ms=4.5, mec="white", mew=0.6)
    ax[0].text(lo + 1, hi - 4, "±5 % band", fontsize=7.5, color=PUMP)
    ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi)
    ax[0].set_xlabel("Datasheet capacity (kW)"); ax[0].set_ylabel("Model capacity (kW)")
    err = np.max(np.abs(md / ds - 1)) * 100
    ax[0].text(hi - 1, lo + 1.5, f"13 rating points\nmax error {err:.1f} %", ha="right", fontsize=7.5)
    panel_label(ax[0], "a")
    m_air = 1.52
    itd = np.linspace(6, 24, 19)
    for mw, c in [(0.4, PALETTE["blue_secondary"]), (0.66, OPT), (1.0, PALETTE["ink"])]:
        Qs = [s4.coil(m_air, 14.0 + d, mw, 14.0)["Q"] / 1e3 for d in itd]
        ax[1].plot(itd, Qs, color=c, label=f"water {mw:g} kg/s")
    ax[1].set_xlabel("Return air minus chilled-water temperature (K)"); ax[1].set_ylabel("Capacity per cooler (kW)")
    ax[1].legend(loc="upper left")
    panel_label(ax[1], "b")
    return finalize_figure(fig, OUT, "fig_coil")


# ----------------------------------------------------------------------------- 5. savings steps (waterfall)
def fig_savings_steps():
    X = S.cdac_x()
    b = S.system({**X, "T_in": 24.0, "psi": 0.0, "T_chw": 14.0}, detail=False)["W_cool_total"] / 1e3
    s1v = S.system({**X, "T_in": 27.0, "psi": 0.0, "T_chw": 14.0}, detail=False)["W_cool_total"] / 1e3
    t = O.best_chw({**X, "T_in": 27.0, "psi": 0.0})["W_cool_total"] / 1e3
    o = O.best_chw({**X, "T_in": 27.0, "psi": 0.175})["W_cool_total"] / 1e3
    steps = [("C-DAC design\nset-points", b, None), ("Inlet\n24 → 27 °C", s1v - b, None),
             ("Chilled water\n14 → 17.6 °C", t - s1v, None), ("Bypass\nrecirculation", o - t, None),
             ("Optimised", o, None)]
    fig, ax = plt.subplots(figsize=(5.2, 2.3))
    run = 0
    for i, (lab, v, _) in enumerate(steps):
        if i in (0, len(steps) - 1):
            ax.bar(i, v, color=BASE if i == 0 else OPT, width=0.6)
            ax.text(i, v + 0.12, f"{v:.2f}", ha="center", fontsize=7.8)
            run = v
        else:
            ax.bar(i, -v, bottom=run + v, color=PALETTE["green_3"], width=0.6)
            ax.text(i, run + 0.12, f"{v:+.2f}".replace("-", "\u2212"), ha="center", fontsize=7.5)
            ax.plot([i - 0.7, i + 0.3], [run, run], color=PUMP, lw=0.6)
            run += v
    ax.set_xticks(range(len(steps))); ax.set_xticklabels([s[0] for s in steps], fontsize=7.6)
    ax.set_ylabel("Cooling power (kW)"); ax.set_ylim(28, 33.5); ax.set_xlim(-0.6, 4.6)
    return finalize_figure(fig, OUT, "fig_savings_steps")


# ----------------------------------------------------------------------------- 6. recirculation trade-off
def fig_recirc_tradeoff():
    d = json.load(open(RES / "sim_airpath.json"))
    fig, ax = plt.subplots(1, 2, figsize=FULL)
    c = d["cases"]["hybrid | in-row | load 100"]["psi_curve"]
    psi = np.array([p["psi"] for p in c]); W = np.array([p["W"] for p in c]) / 1e3
    air = np.array([p["air"] for p in c]) / 1e3; refr = np.array([p["refr"] for p in c]) / 1e3
    ax[0].plot(psi, air - air[0], color=AIR, label="air moving (fans)")
    ax[0].plot(psi, refr - refr[0], color=REFR, label="refrigeration (chiller)")
    ax[0].plot(psi, W - W[0], color=PALETTE["ink"], lw=2.0, label="total cooling power")
    i = int(np.argmin(W)); ax[0].plot(psi[i], W[i] - W[0], "o", color=OPT, ms=5, zorder=5)
    ax[0].annotate(f"optimum ψ = {psi[i]:.3f}\n{W[i]-W[0]:+.2f} kW".replace("-", "\u2212"), (psi[i], W[i] - W[0]), xytext=(0.005, -3.1),
                   fontsize=7.5, arrowprops=dict(arrowstyle="-", lw=0.7))
    ax[0].axhline(0, color=GREY, lw=0.8)
    ax[0].set_xlabel("Recirculation fraction ψ"); ax[0].set_ylabel("Change vs traditional aisle (kW)")
    ax[0].legend(loc="upper left"); panel_label(ax[0], "a")
    for L, col, ls in [(100, OPT, "-"), (60, AIR, "--"), (30, LIQ, ":")]:
        cc = d["cases"][f"hybrid | in-row | load {L}"]["psi_curve"]
        p = np.array([x["psi"] for x in cc]); w = np.array([x["W"] for x in cc])
        ax[1].plot(p, 100 * (w / w[0] - 1), color=col, ls=ls, label=f"{L} % CPU load")
    ax[1].axhline(0, color=GREY, lw=0.8)
    ax[1].set_xlabel("Recirculation fraction ψ"); ax[1].set_ylabel("Change in cooling power (%)")
    ax[1].legend(loc="upper left"); panel_label(ax[1], "b")
    return finalize_figure(fig, OUT, "fig_recirc_tradeoff")


# ----------------------------------------------------------------------------- 7. leakage
def fig_leakage():
    d = json.load(open(RES / "sim_airpath.json"))
    fig, ax = plt.subplots(figsize=(3.12, 2.6))
    for cfg, lab, ls in [("hybrid | in-row", "in-row coolers", "-"), ("hybrid | raised floor, wall bypass", "raised floor", "--")]:
        rows = d["beta_sweep"][cfg]
        b = [r["beta"] * 100 for r in rows]
        ax.plot(b, [r["W_trad"] / 1e3 for r in rows], color=BASE, ls=ls, marker="o", ms=3, label=f"traditional, {lab}")
        ax.plot(b, [r["W_bypass"] / 1e3 for r in rows], color=OPT, ls=ls, marker="o", ms=3, label=f"recirculation, {lab}")
    ax.set_xlabel("Containment leakage β (% of cooler air)"); ax.set_ylabel("Cooling power (kW)")
    ax.set_ylim(29.3, 39.5)
    ax.legend(loc="upper left", fontsize=7, ncol=1)
    return finalize_figure(fig, OUT, "fig_leakage")


# ----------------------------------------------------------------------------- 8. air moving argument
def fig_air_moving():
    d = json.load(open(RES / "sim_airpath.json"))
    cfgs = ["hybrid | in-row", "hybrid | raised floor, wall bypass", "air-only | in-row", "air-only | raised floor, wall bypass"]
    short = ["Hybrid\nin-row", "Hybrid\nraised floor", "Air-only\nin-row", "Air-only\nraised floor"]
    fig, ax = plt.subplots(1, 2, figsize=FULL_TALL, gridspec_kw=dict(width_ratios=[1, 1.1]))
    groups = [("air_moving", "air moving", AIR), ("liquid_moving", "pumps", PUMP),
              ("heat_rejection_fans", "tower fans", REJ), ("refrigeration", "refrigeration", REFR)]
    bottom = np.zeros(len(cfgs))
    for g, lab, c in groups:
        v = np.array([d["cases"][f"{k} | load 100"]["baseline"]["groups"][g] / 1e3 for k in cfgs])
        ax[0].bar(range(len(cfgs)), v, bottom=bottom, color=c, width=0.62, label=lab, edgecolor="white", lw=0.5)
        bottom += v
    for i, k in enumerate(cfgs):
        b = d["cases"][f"{k} | load 100"]["baseline"]
        ax[0].text(i, bottom[i] + 1.2, f"{100*b['groups']['air_moving']/b['W_cool_total']:.0f} % air", ha="center", fontsize=7.2)
    ax[0].set_xticks(range(len(cfgs))); ax[0].set_xticklabels(short, fontsize=7.2)
    ax[0].set_ylabel("Cooling power, C-DAC set-points (kW)"); ax[0].set_ylim(0, 82)
    ax[0].legend(loc="upper left", fontsize=7); panel_label(ax[0], "a")
    # saving vs server air rise
    sens = d["air_only_dT_sensitivity"]
    ax[1].plot([s["dT_cap"] for s in sens], [s["saving_bypass_pct"] for s in sens], "-o", color=GREY, mfc="white",
               mec=PUMP, label="air-only reference (air rise set)")
    for k, lab, col, mk in [("hybrid | in-row", "hybrid, in-row", OPT, "s"), ("hybrid | raised floor, wall bypass", "hybrid, raised floor", AIR, "D")]:
        xs, ys = [], []
        for L in (100, 60, 30):
            c = d["cases"][f"{k} | load {L}"]
            b = c["baseline"]; xs.append(b["Q_air"] / (b["m_s"] * 1006)); ys.append(c["saving_bypass_pct"])
        ax[1].plot(xs, ys, mk, color=col, ms=5, ls="none", label=lab)
        if k == "hybrid | in-row":
            for x, y, L in zip(xs, ys, (100, 60, 30)):
                ax[1].text(x + 0.3, y - 0.2, f"{L} %", fontsize=6.8, color=col, va="top")
    ax[1].set_xlabel("Server air temperature rise (K)"); ax[1].set_ylabel("Saving from recirculation (%)")
    ax[1].set_xlim(7.5, 21); ax[1].set_ylim(-0.5, 12)
    ax[1].legend(loc="upper right", fontsize=7); panel_label(ax[1], "b")
    return finalize_figure(fig, OUT, "fig_air_moving")


# ----------------------------------------------------------------------------- 9. coolant distribution
def fig_coolant_distribution():
    d = json.load(open(RES / "simA_flow_network.json"))
    fig, ax = plt.subplots(1, 2, figsize=FULL, gridspec_kw=dict(width_ratios=[1.35, 1]))
    x = np.arange(1, 13)
    for key, col, off, lab in [("nominal", BASE, -0.17, "as installed (no balancing)"), ("balanced", OPT, 0.17, "balancing valves trimmed")]:
        r = d["cases"][key]["racks"]
        m = np.array([k["q_node_mean"] for k in r]); lo = np.array([k["q_node_min"] for k in r]); hi = np.array([k["q_node_max"] for k in r])
        ax[0].errorbar(x + off, m, yerr=[m - lo, hi - m], fmt="o", ms=3.5, color=col, capsize=1.8, lw=1.0, label=lab)
    ax[0].axhline(0.5, color=GREY, lw=0.9, ls="--")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{i}\n{n}" for i, n in zip(x, P.CDAC["rack_nodes"])], fontsize=6.8)
    ax[0].set_xlabel("Rack number / nodes in rack"); ax[0].set_ylabel("Coolant flow per node (L/min)")
    ax[0].legend(loc="upper left", fontsize=7); ax[0].set_ylim(0.38, 0.78); panel_label(ax[0], "a", x=-0.1)
    t = d["turndown"]
    ax[1].plot([s["q_total"] for s in t], [s["T_case_max"] for s in t], "-o", color=BASE, ms=3.5)
    ax[1].axvline(309.5, color=GREY, lw=0.9, ls="--"); ax[1].text(314, 66, "C-DAC design\n309.5 L/min", fontsize=7, color=PUMP)
    ax[1].set_xlabel("Total secondary flow (L/min)"); ax[1].set_ylabel("Hottest CPU case (°C)")
    panel_label(ax[1], "b")
    return finalize_figure(fig, OUT, "fig_coolant_distribution")


# ----------------------------------------------------------------------------- 10. annual
def fig_annual():
    o = json.load(open(RES / "simB_hourly.json"))
    z = np.load(RES / "simB_hourly_series.npz")
    fig, ax = plt.subplots(1, 2, figsize=FULL, gridspec_kw=dict(width_ratios=[1.3, 1]))
    days = np.arange(365)
    for key, col, lab in [("C-DAC_design__thermal_virus__dry", BASE, "dry cooler"), ("C-DAC_design__thermal_virus__tower", OPT, "cooling tower")]:
        dd = z[key].reshape(365, 24)
        ax[0].fill_between(days, dd.min(1), dd.max(1), color=col, alpha=0.16, lw=0)
        ax[0].plot(days, dd.mean(1), color=col, lw=1.0, label=lab)
    ax[0].set_xlabel("Day of the year"); ax[0].set_ylabel("Cooling power (kW)")
    ax[0].set_xlim(0, 364); ax[0].legend(loc="upper right", ncol=2); panel_label(ax[0], "a", x=-0.1)
    months = list("JFMAMJJASOND")
    a = o["cases"]["C-DAC design | thermal virus | tower"]["monthly_E_MWh"]
    b = o["cases"]["recommended | thermal virus | tower"]["monthly_E_MWh"]
    xi = np.arange(12)
    ax[1].bar(xi - 0.2, a, width=0.4, color=BASE, label="C-DAC set-points")
    ax[1].bar(xi + 0.2, b, width=0.4, color=OPT, label="recommended")
    ax[1].set_xticks(xi); ax[1].set_xticklabels(months)
    ax[1].set_ylabel("Cooling energy (MWh/month)"); ax[1].set_ylim(0, 30)
    ax[1].legend(loc="upper center", ncol=2, fontsize=7, columnspacing=1.0, handlelength=1.2); panel_label(ax[1], "b")
    return finalize_figure(fig, OUT, "fig_annual")


# ----------------------------------------------------------------------------- 11. exergy
def fig_exergy():
    X = S.cdac_x()
    b = S.system({**X, "T_in": 24.0, "psi": 0.0, "T_chw": 14.0})
    o = O.best_chw({**X, "T_in": 27.0, "psi": 0.175})
    names = {"server: cold plates (S1)": "Cold plates", "server: chassis air heat transfer (S2)": "Board to air",
             "server: fans (S2)": "Server fans", "server: TCS pumping/friction (S1)": "TCS pumping",
             "aisle mixing + bypass fans (S3)": "Aisle mixing + bypass fans",
             "in-row coolers: coil + fans (S4)": "In-row coolers", "CHW pumping": "CHW pumping",
             "CDU plate HX": "CDU", "chiller (main, CHW)": "Chiller", "trim chiller (TCS)": "Trim chiller",
             "cooling towers + CW/FW pumps": "Towers + plant pumps"}
    keys = [k for k in names if abs(b["X"].get(k, 0)) > 20 or abs(o["X"].get(k, 0)) > 20]
    keys = sorted(keys, key=lambda k: b["X"][k])
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    y = np.arange(len(keys))
    ax.barh(y + 0.19, [b["X"][k] / 1e3 for k in keys], height=0.36, color=BASE, label="C-DAC set-points")
    ax.barh(y - 0.19, [o["X"][k] / 1e3 for k in keys], height=0.36, color=OPT, label="optimised")
    ax.set_yticks(y); ax.set_yticklabels([names[k] for k in keys], fontsize=7.3)
    ax.set_xlabel("Exergy destroyed (kW)")
    el_b = b["X"]["server: electronics, electricity to heat"] / 1e3
    ax.set_xlim(0, 13)
    ax.legend(loc="lower right", fontsize=7.5)
    return finalize_figure(fig, OUT, "fig_exergy")


# ----------------------------------------------------------------------------- 12. optimisation maps
def fig_optimisation_maps():
    from matplotlib.colors import LinearSegmentedColormap
    d = json.load(open(RES / "sim_maps.json"))
    cmap = LinearSegmentedColormap.from_list("w", [PALETTE["blue_main"], PALETTE["blue_secondary"], "#DCE7F3", "#FFFFFF"])
    fig, ax = plt.subplots(1, 3, figsize=(6.0, 2.55))
    specs = [("psi_beta", "Leakage β (%)", 100.0, "a"), ("psi_tin", "Server inlet (°C)", 1.0, "b"),
             ("psi_tchw", "Chilled water (°C)", 1.0, "c")]
    for axi, (k, ylab, ys, letter) in zip(ax, specs):
        m = d[k]
        psi = np.array(m["psi"]); y = np.array(m["y"]) * ys
        W = np.array([[np.nan if v is None else v for v in row] for row in m["W"]]) / 1e3
        lo, hi = np.nanmin(W), np.nanpercentile(W, 95)
        lev = np.linspace(lo, hi, 13)
        axi.contourf(psi, y, np.clip(W, lo, hi), levels=lev, cmap=cmap, extend="max")
        cs = axi.contour(psi, y, W, levels=lev[1::2], colors=PALETTE["ink"], linewidths=0.4, alpha=0.55)
        bad = np.isnan(W)
        avoid = []
        if bad.any():
            axi.contourf(psi, y, bad.astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"])
            cb_ = axi.contour(psi, y, bad.astype(float), levels=[0.5], colors=PUMP, linewidths=0.6)
            avoid += [seg for seg in cb_.allsegs[0]]
        # optimum psi along each row (the "ridge")
        rows = [(np.nanargmin(r), yy) for r, yy in zip(W, y) if np.isfinite(r).any()]
        if k != "psi_tchw":
            ridge = np.array([[psi[i], yy] for i, yy in rows])
            axi.plot(ridge[:, 0], ridge[:, 1], color=BASE, lw=1.3, label="best ψ")
        else:
            ridge = np.array([[p, y[np.nanargmin(W[:, j])]] for j, p in enumerate(psi) if np.isfinite(W[:, j]).any()])
            axi.plot(ridge[:, 0], ridge[:, 1], color=BASE, lw=1.3, label="best chilled water")
        avoid.append(ridge)
        if k == "psi_beta":
            avoid.append(np.array([[0.25, 5.0], [0.42, 5.0]]))
        place_clabels(axi, cs, "%.1f", avoid=avoid, tol=0.08, spacing=0.14, fontsize=6.0)
        i, j = np.unravel_index(np.nanargmin(W), W.shape)
        if k == "psi_beta":
            r = list(np.round(y, 4)).index(5.0); i, j = r, int(np.nanargmin(W[r]))
            axi.axhline(5.0, color=PALETTE["ink"], lw=0.6, ls=":")
            axi.text(0.395, 5.6, "as built", ha="right", fontsize=6.5)
        axi.plot(psi[j], y[i], marker="*", ms=8.5, color=PALETTE["highlight"], mec=PALETTE["ink"], mew=0.6, zorder=6)
        axi.set_xlabel("Recirculation ψ"); axi.set_ylabel(ylab)
        axi.set_xlim(0, 0.4); axi.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
        if k == "psi_tin":
            axi.set_ylim(18, 27.4)
        panel_label(axi, letter, x=-0.3)
    ax[1].text(0.30, 19.0, "coolers cannot\nreach supply", fontsize=6.3, ha="center", color=PALETTE["ink"],
               bbox=dict(fc="white", ec="none", pad=0.6))
    ax[2].text(0.31, 18.6, "chilled water\ntoo warm", fontsize=6.3, ha="center", color=PALETTE["ink"],
               bbox=dict(fc="white", ec="none", pad=0.6))
    return finalize_figure(fig, OUT, "fig_optimisation_maps", pad=0.4)


# ----------------------------------------------------------------------------- 13. coolant properties
FLUID_COL = {"water": AIR, "PG25": OPT, "EG25": LIQ, "Novec7500": BASE}
FLUID_LAB = {"water": "water", "PG25": "PG25 (used)", "EG25": "EG25", "Novec7500": "Novec 7500"}
def fig_coolant_properties():
    g = json.load(open(RES / "sim_gaps.json"))["coolant_props"]; T = np.array(g["T"])
    fig, ax = plt.subplots(1, 3, figsize=(6.0, 2.3))
    for f in ("water", "PG25", "EG25", "Novec7500"):
        d = g[f]; kw = dict(color=FLUID_COL[f], lw=1.5 if f == "PG25" else 1.1, label=FLUID_LAB[f])
        ax[0].plot(T, np.array(d["rho"]) * np.array(d["cp"]) / 1e6, **kw)
        ax[1].plot(T, np.array(d["mu"]) * 1e3, **kw)
        ax[2].plot(T, d["k"], **kw)
    ax[0].set_ylabel("ρc$_p$ (MJ/m³·K)"); ax[1].set_ylabel("Viscosity (mPa·s)"); ax[2].set_ylabel("Conductivity (W/m·K)")
    ax[1].set_yscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
    ax[1].yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 3])); ax[1].yaxis.set_minor_locator(NullLocator())
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%g"))
    for a, l in zip(ax, "abc"):
        a.set_xlabel("Temperature (°C)"); a.axvspan(37, 47, color=PALETTE["green_1"], lw=0, zorder=0); panel_label(a, l, x=-0.32)
    ax[0].text(42, 2.35, "loop\nrange", ha="center", fontsize=6.5, color=PUMP)
    ax[0].legend(loc="center left", fontsize=6.2, handlelength=1.2, bbox_to_anchor=(0.0, 0.55))
    return finalize_figure(fig, OUT, "fig_coolant_properties", pad=0.4)

# ----------------------------------------------------------------------------- 14. cold plate: junction + exergy map
def fig_coldplate_junction():
    from matplotlib.colors import LinearSegmentedColormap
    g = json.load(open(RES / "sim_gaps.json"))
    fig, ax = plt.subplots(1, 2, figsize=FULL)
    L = g["coldplate_lines"]; q = np.array(L["flow"])
    for t, c in (("30", PALETTE["blue_secondary"]), ("37", OPT), ("45", BASE)):
        ax[0].plot(q, L[t], color=c, label=f"inlet {t} °C")
    m = 0.5 / 60000 * s1.coolant_props(37)["rho"]; Qp = g["coldplate_map"]["Q_plate"]
    p1 = s1.cold_plate(Qp, m, 37.0); p2 = s1.cold_plate(Qp, m, p1["T_out"])
    ax[0].plot([0.5], [p2["T_j"]], "o", color=PALETTE["ink"], ms=4.5, zorder=5)
    ax[0].annotate(f"C-DAC design\n(2nd plate)\n{p2['T_j']:.1f} °C", (0.5, p2["T_j"]), xytext=(0.3, 70), fontsize=7.2,
                   arrowprops=dict(arrowstyle="-", lw=0.7))
    ax[0].set_xlabel("Coolant flow per cold plate (L/min)"); ax[0].set_ylabel("Junction temperature (°C)")
    ax[0].set_ylim(30, 80); ax[0].set_xlim(0, 2.05); ax[0].legend(loc="upper right")
    ax[0].text(2.0, 32, "limit: 100 °C junction, 90 °C case", ha="right", fontsize=6.6, color=PUMP)
    panel_label(ax[0], "a")
    cm = g["coldplate_map"]; F = np.array(cm["flow"]); Ti = np.array(cm["T_in"]); X = np.array(cm["X_dest"]); Be = np.array(cm["Bejan"])
    cmap = LinearSegmentedColormap.from_list("x", ["#FFFFFF", "#DCE7F3", PALETTE["blue_secondary"], PALETTE["blue_main"]])
    cf = ax[1].contourf(F, Ti, X, levels=12, cmap=cmap)
    cs = ax[1].contour(F, Ti, X, levels=cf.levels[1::2], colors=PALETTE["ink"], linewidths=0.4, alpha=0.6)
    cb = ax[1].contour(F, Ti, Be, levels=[0.95, 0.98, 0.99], colors=BASE, linewidths=0.9, linestyles="--")
    place_clabels(ax[1], cs, "%.1f W", avoid=[s_ for s_ in cb.allsegs[0] + cb.allsegs[1] + cb.allsegs[2]] + [np.array([[0.5, 37.0]])],
                  tol=0.07, edge=0.1, spacing=0.18, fontsize=6.0)
    ax[1].clabel(cb, fmt="Be %.2f", fontsize=6.0)
    ax[1].plot([0.5], [37], "o", color=PALETTE["ink"], ms=4)
    cbar = fig.colorbar(cf, ax=ax[1], pad=0.02, fraction=0.05); cbar.set_label("Exergy destroyed per plate (W)", fontsize=7); cbar.ax.tick_params(labelsize=6.5)
    ax[1].set_xlabel("Coolant flow per cold plate (L/min)"); ax[1].set_ylabel("Coolant inlet (°C)")
    panel_label(ax[1], "b")
    return finalize_figure(fig, OUT, "fig_coldplate_junction")

# ----------------------------------------------------------------------------- 15. server fans: curves, power vs flow, chassis profile
def fig_server_fans():
    g = json.load(open(RES / "sim_gaps.json"))
    fc = g["fan_curves"]; fig, ax = plt.subplots(1, 3, figsize=(6.0, 2.4))
    cfm = 1 / 4.719474e-4
    for n, c in zip(("0.25", "0.5", "0.75", "1.0"), (PALETTE["green_2"], PALETTE["green_3"], PALETTE["teal"], OPT)):
        d = fc["curves"][n]; ax[0].plot(np.array(d["V"]) * cfm, d["dP"], color=c, lw=1.1, label=f"{int(float(n)*100)} %")
    V = np.linspace(0, max(fc["curves"]["1.0"]["V"]), 50)
    ax[0].plot(V * cfm, fc["k_sys"] * V ** 2, color=PALETTE["ink"], lw=1.3, ls="--", label="chassis")
    ops = fc["op"]; ax[0].plot([o["V"] * cfm for o in ops], [o["dP"] for o in ops], "o", color=BASE, ms=2.5, zorder=5)
    ax[0].set_xlabel("Air flow per node (CFM)"); ax[0].set_ylabel("Pressure (Pa)"); ax[0].set_ylim(0, 600)
    ax[0].legend(fontsize=6.4, loc="upper right", bbox_to_anchor=(1.06, 1.0), handlelength=1.3, title="fan speed", title_fontsize=6.4); panel_label(ax[0], "a", x=-0.33)
    Vo = np.array([o["V"] for o in ops]) * cfm; Wo = np.array([o["W"] for o in ops])
    ax[1].plot(Vo, Wo, "-o", color=AIR, ms=2.5)
    ax[1].plot(Vo, Wo[-1] * (Vo / Vo[-1]) ** 3, color=GREY, lw=1.0, ls=":", label="pure cube law")
    ax[1].set_xlabel("Air flow per node (CFM)"); ax[1].set_ylabel("Fan power per node (W)")
    ax[1].legend(fontsize=6.4, loc="upper left"); panel_label(ax[1], "b", x=-0.3)
    zones = ["inlet", "drives", "fans", "DIMMs", "NIC", "exhaust"]
    keys = ["T_in", "T_air_z1", "T_air_fan", "T_air_z2", "T_air_z3", "T_out"]
    for t, c in (("24.0", AIR), ("27.0", OPT), ("32.0", BASE)):
        p = g["chassis_profile"][t]; ax[2].plot(range(6), [p[k] for k in keys], "-o", color=c, ms=2.5, label=f"inlet {float(t):.0f} °C")
    ax[2].set_xticks(range(6)); ax[2].set_xticklabels(zones, fontsize=6.2, rotation=35, ha="right")
    ax[2].set_ylabel("Air temperature (°C)"); ax[2].legend(fontsize=6.2, loc="upper left"); panel_label(ax[2], "c", x=-0.3)
    return finalize_figure(fig, OUT, "fig_server_fans", pad=0.4)

# ----------------------------------------------------------------------------- 16. rear door
def fig_rdhx():
    g = json.load(open(RES / "sim_gaps.json"))["rdhx"]
    fig, ax = plt.subplots(1, 2, figsize=FULL)
    for geom, c in (("2-row", OPT), ("4-row dense", BASE)):
        sw = [s for s in g["sweep"] if s["geom"] == geom]
        V = [s["V"] for s in sw]
        ax[0].plot(V, [s["dP"] for s in sw], color=c, label=f"{geom} door")
        ax[1].plot(V, [s["W_passive"] for s in sw], color=c, label=f"{geom}, passive")
        ax[1].plot(V, [s["W_active"] for s in sw], color=c, ls="--", label=f"{geom}, active fans")
    lo, hi = g["dP_avail_range"]
    ax[0].axhspan(lo, hi, color=PALETTE["green_1"], lw=0, zorder=0)
    Vr = g["m_air_rack"] / P.rho_air(g["T_in"])
    for a in ax:
        a.axvline(Vr, color=PALETTE["ink"], lw=0.7, ls=":")
    ax[0].text(Vr + 0.06, 46, "average\nRudra rack", fontsize=6.8)
    ax[0].set_xlabel("Air flow through door (m³/s)"); ax[0].set_ylabel("Door pressure drop (Pa)")
    from matplotlib.patches import Patch
    h0, l0 = ax[0].get_legend_handles_labels()
    ax[0].legend(h0 + [Patch(color=PALETTE["green_1"])], l0 + ["server-fan margin"], loc="upper left", bbox_to_anchor=(0.2, 1.0))
    ax[1].set_xlabel("Air flow through door (m³/s)"); ax[1].set_ylabel("Parasitic fan power per rack (W)"); ax[1].legend(loc="upper left", fontsize=6.8, bbox_to_anchor=(0.2, 1.0))
    panel_label(ax[0], "a"); panel_label(ax[1], "b")
    return finalize_figure(fig, OUT, "fig_rdhx")

# ----------------------------------------------------------------------------- 17. chilled water optimisation
def fig_chw():
    g = json.load(open(RES / "sim_gaps.json"))["chw_optimisation"]
    fig, ax = plt.subplots(1, 2, figsize=FULL)
    for key, lab, c, ls in (("psi0.0_Tin24.0", "C-DAC (24 °C, ψ = 0)", BASE, "-"),
                            ("psi0.0_Tin27.0", "27 °C inlet, ψ = 0", AIR, "--"),
                            ("psi0.175_Tin27.0", "27 °C inlet, ψ = 0.175", OPT, "-")):
        rows = [r for r in g[key] if r["feasible"]]
        t = [r["T_chw"] for r in rows]
        ax[0].plot(t, [r["W"] / 1e3 for r in rows], color=c, ls=ls, label=lab)
        i = int(np.argmin([r["W"] for r in rows])); ax[0].plot(t[i], rows[i]["W"] / 1e3, "o", color=c, ms=4)
        ax[1].plot(t, [r["m_w"] for r in rows], color=c, ls=ls, label=lab)
    ax[0].set_xlabel("Chilled-water supply (°C)"); ax[0].set_ylabel("Cooling power (kW)")
    ax[1].set_xlabel("Chilled-water supply (°C)"); ax[1].set_ylabel("Chilled-water flow (kg/s)"); ax[1].legend(loc="upper left", fontsize=6.4)
    panel_label(ax[0], "a"); panel_label(ax[1], "b")
    return finalize_figure(fig, OUT, "fig_chw")


if __name__ == "__main__":
    for f in [fig_liquid_coldplate, fig_server_air, fig_aisle_mixing, fig_coil, fig_savings_steps,
              fig_recirc_tradeoff, fig_leakage, fig_air_moving, fig_coolant_distribution, fig_annual, fig_exergy, fig_optimisation_maps,
              fig_coolant_properties, fig_coldplate_junction, fig_server_fans, fig_rdhx, fig_chw]:
        p = f(); print("wrote", p[0].name)
