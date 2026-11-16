"""S5 figures: optimisation maps and trade-off curves."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from s2_air import _style
_style(plt)
M = json.load(open("results/s5_maps.json"))
INK, INK2 = "#0b0b0b", "#52514e"

def grid(mp, ykey):
    psi = np.array(mp["psi"]); ys = np.array(mp[ykey]); W = np.full((len(ys), len(psi)), np.nan); C = {}
    for c in mp["cells"]:
        if not c.get("feasible"): continue
        i = int(np.argmin(abs(ys - c["x"][ykey if ykey != "T_in" else "T_in"])))
        j = int(np.argmin(abs(psi - c["x"]["psi"])))
        W[i, j] = c["W_cool_total"] / 1e3; C[(i, j)] = c
    return psi, ys, W, C

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for k, (mp, yk, ylab) in enumerate(((M["map_psi_beta"], "beta", "Containment leakage (bypass) fraction β [-]"),
                                    (M["map_psi_tin"], "T_in", "Server inlet temperature T_in [°C]"))):
    psi, ys, W, C = grid(mp, yk)
    lv = np.arange(np.floor(np.nanmin(W) * 4) / 4, np.nanmax(W) + 0.25, 0.25)
    cf = ax[k].contourf(psi, ys, W, levels=lv, cmap="Blues_r")
    cs = ax[k].contour(psi, ys, W, levels=lv[::2], colors=INK2, linewidths=0.5)
    ax[k].clabel(cs, fmt="%.1f", fontsize=7)
    opt = [(psi[np.nanargmin(W[i])], ys[i]) for i in range(len(ys)) if np.isfinite(W[i]).any()]
    ax[k].plot([o[0] for o in opt], [o[1] for o in opt], "o-", color="#e8590c", ms=3.5, lw=1.4, label="optimal ψ (CHW re-optimised)")
    inf = np.isnan(W)
    if inf.any():
        ax[k].contourf(psi, ys, inf.astype(float), levels=[0.5, 1.5], colors="none", hatches=["///"])
    ax[k].set_xlabel("Intentional recirculation fraction ψ [-]"); ax[k].set_ylabel(ylab)
    plt.colorbar(cf, ax=ax[k], label="Total cooling power [kW]")
    ax[k].legend(loc="upper right", fontsize=8)
ax[0].set_title("ψ–β at T_in = 24 °C: leakage costs ~0.09–0.10 kW per 1 % β; optimal ψ rises with β")
ax[1].set_title("ψ–T_in at β = 0.05: warmer inlet dominates; minimum at the T_in ceiling, ψ ≈ 0.15–0.28")
fig.suptitle("Pod cooling power (server fans + IRC + pumps + chillers + towers), CHW set-point optimised at every point", fontsize=10.5)
fig.tight_layout(); fig.savefig("figures/s5_maps.png", dpi=200)

# trade-off along psi at T_in 24 and 27, beta .05 : components
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
for L, ls in (("1.0", "-"),):
    pass
mp = M["map_psi_tin"]; psi = np.array(mp["psi"])
for k, T in enumerate((24.0, 27.0)):
    cells = sorted([c for c in mp["cells"] if c.get("feasible") and abs(c["x"]["T_in"] - T) < 1e-6], key=lambda c: c["x"]["psi"])
    p = [c["x"]["psi"] for c in cells]
    W0 = cells[0]
    comps = [("IRC fans", lambda c: c["W"]["irc_fans"]), ("Chillers", lambda c: c["W"]["chiller"] + c["W"]["trim_chiller"]),
             ("CHW pumps", lambda c: c["W"]["chw_pump"]), ("Bypass fans", lambda c: c["W"]["bypass_fans"]),
             ("Towers + CW/FW pumps", lambda c: c["W"]["tower_fans"] + c["W"]["cw_fw_pumps"])]
    cols = ["#1c7ed6", "#e8590c", "#7048e8", "#2b8a3e", "#868e96"]
    for (n, f), col in zip(comps, cols):
        ax[k].plot(p, [(f(c) - f(W0)) / 1e3 for c in cells], color=col, label=n)
    tot = [(c["W_cool_total"] - W0["W_cool_total"]) / 1e3 for c in cells]
    ax[k].plot(p, tot, color=INK, lw=2.6, label="Total")
    i = int(np.argmin(tot)); ax[k].plot(p[i], tot[i], "o", color=INK)
    ax[k].annotate(f"optimum ψ={p[i]:.3f}\nΔW={tot[i]*1e3:+.0f} W", (p[i], tot[i]), xytext=(10, -28), textcoords="offset points", fontsize=8)
    ax2 = ax[k].twinx(); ax2.plot(p, [c["T_chw"] for c in cells], ":", color="#e8590c", lw=1.3); ax2.set_ylabel("Optimal CHW supply [°C] (dotted)", color="#e8590c")
    ax2.spines["right"].set_visible(True); ax2.grid(False)
    ax[k].axhline(0, color=INK2, lw=0.6); ax[k].set_xlabel("Recirculation fraction ψ [-]"); ax[k].set_ylabel("Change vs ψ = 0 [kW]")
    ax[k].set_title(f"T_in = {T:.0f} °C, β = 0.05: IRC fans fall as (1−ψ)^~2.6, chiller rises with colder CHW")
ax[0].legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig("figures/s5_tradeoff.png", dpi=200)
print("ok")
