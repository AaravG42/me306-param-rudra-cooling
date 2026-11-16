"""
s3_aisle.py -- Student 3: enclosed-aisle flow network & bypass/recirculation modeller
=====================================================================================
PARAM Rudra hybrid-cooled pod (2 rows x 8 racks, in-row coolers = IRC).

Notation (follows Demetriou & Khalifa, J. Electron. Packag. 134(2) 020904, 2012, "bypass
recirculation"; and the lumped, well-mixed-aisle thermodynamic model of Demetriou's 2012 Syracuse
PhD thesis, surface.syr.edu/mae_etd/65):

    m_s   total server (IT) air mass flow                       [kg/s]
    m_c   cooler (IRC) supply = return air mass flow             [kg/s]
    T_c   cooler supply (leaving-air) temperature                [degC]
    beta  cold-air bypass / leakage fraction:  m_b = beta * m_c
          (cold air that leaves the cold side and reaches the IRC return without passing servers)
    psi   recirculation fraction:              m_r = psi * m_s
          (server exhaust re-admitted to the cold side: intentional "bypass recirculation" through
           dampers / low-lift fans, plus any uncontrolled hot-air infiltration)
    T_in  server inlet (cold-aisle) temperature                  [degC]
    T_h   server exhaust temperature                             [degC]
    T_ret IRC return (air-on-coil) temperature                   [degC]

Control volumes (see figures/s3_schematic.png):
  CAC: CV1 = contained cold aisle (IRC supply -> CV1 -> servers).  CV2 = open room = hot side.
  HAC: CV1 = open room = cold side.  CV2 = contained hot aisle (servers -> CV2 -> IRC return).
  In both, the IRC draws from the hot side and discharges to the cold side, so the lumped balances
  are IDENTICAL; config changes only which side is contained (and hence which leakage surface the
  flow-network model uses).

Key lumped assumptions (Demetriou-type model): steady state; each side perfectly mixed (single
temperature); constant cp (params.CP_AIR); dry air (no condensation in the aisle); no heat gain
through containment walls; leakage air leaves the cold side at T_c (leak paths at the cooler
discharge region) -- this is exact when psi=0 and an O(beta*psi) approximation otherwise.

Public API (INTERFACE.md):
    aisle_mixing(m_s, Q_air, T_c, psi, beta, config="CAC")         -> dict
    solve_for_inlet(m_s, Q_air, T_in_target, psi, beta, config)    -> dict (+ required T_c)
    flow_network(m_s0, m_c=None, A_L=..., A_r=0, m_bypass_fan=0, dP_set=None, ...) -> dict
Helpers: containment_surface(config), leak_area_from_fraction(frac, config).
"""
import json
import os

import numpy as np
from scipy.optimize import brentq

import params as P

CP = P.CP_AIR
R = P.R_AIR

# ---------------------------------------------------------------- geometry for the flow network
# Pod: 2 facing rows x 8 racks; rack 600 mm wide, 42U ~ 2.0 m tall; contained aisle 1.2 m wide.
RACK_W = 0.60          # m   (standard 600 mm rack)
RACK_H = 2.00          # m   (42U + plinth, ~2.0 m)
AISLE_W = 1.20         # m   (two-tile contained aisle)
CD_ORIFICE = 0.61      # discharge coefficient of sharp-edged gaps (classic orifice value; ASHRAE
                       # Fundamentals ch.16 uses Cd~0.6-0.65 for effective-leakage-area cracks)
DP_SHUTOFF_SERVER = 150.0  # Pa, net (fan - chassis) shut-off pressure of a 1U node fan curve.
                           # Placeholder; S2 may replace with the real fan curve.


def containment_surface(config="CAC"):
    """Leakage-relevant boundary area of the contained aisle [m^2].

    Roof + two end doors + the rack faces bounding the aisle (blanking panels, rail gaps,
    inter-rack seams all live on the rack faces). Same geometry for CAC and HAC with IRCs
    (no raised floor in PARAM Rudra's slab-floor in-row layout -> no floor cut-outs).
    """
    L = P.RACKS_PER_ROW * RACK_W                 # aisle length 4.8 m
    roof = L * AISLE_W                           # 5.76 m^2
    doors = 2 * AISLE_W * RACK_H                 # 4.8 m^2
    rack_faces = P.RACKS_PER_POD * RACK_W * RACK_H   # 19.2 m^2
    return {"A_roof": roof, "A_doors": doors, "A_rack_faces": rack_faces,
            "A_surface": roof + doors + rack_faces}


def leak_area_from_fraction(frac, config="CAC"):
    """Leakage area [m^2] for an open-area fraction of the containment surface (0.01 = 1 %)."""
    A = containment_surface(config)["A_surface"]
    AL = frac * A
    return {"A_L": AL, "A_L_cm2_per_rack": AL * 1e4 / P.RACKS_PER_POD, "A_surface": A}


def _f(d):
    """Cast every value to a plain Python float (INTERFACE.md: dicts of floats)."""
    return {k: float(v) for k, v in d.items()}


# ================================================================ lumped mixing model
def aisle_mixing(m_s, Q_air, T_c, psi, beta, config="CAC", dP=0.0):
    """Well-mixed two-zone aisle model with leakage beta and recirculation psi.

    Parameters
    ----------
    m_s   : total server air flow [kg/s]
    Q_air : air-side heat picked up by servers [W]
    T_c   : IRC supply temperature [degC]
    psi   : recirculation fraction, m_r = psi*m_s  (0 <= psi < 1)
    beta  : leakage/bypass fraction, m_b = beta*m_c (0 <= beta < 1)
    config: "CAC" or "HAC" (balances identical; label only)
    dP    : |pressure difference| across leak/recirculation openings [Pa] (optional) used only
            for the throttling-exergy term.

    Returns dict of floats (temperatures degC, flows kg/s, exergy W).
    """
    if not (0 <= psi < 1 and 0 <= beta < 1):
        raise ValueError("need 0<=psi<1, 0<=beta<1")
    dTs = Q_air / (m_s * CP)                          # server air temperature rise, Q = m_s cp dTs
    # --- mass balances
    # cold side: (1-beta) m_c + m_r = m_s ; m_r = psi m_s   ->  m_c = m_s (1-psi)/(1-beta)
    m_c = m_s * (1 - psi) / (1 - beta)
    m_r = psi * m_s
    m_b = beta * m_c
    m_hot_to_irc = m_s - m_r                          # exhaust reaching the IRC = (1-beta) m_c
    # hot side check: m_s - m_r + m_b = m_c
    mass_err = (m_s - m_r + m_b) - m_c
    # --- energy balances (perfect mixing, constant cp)
    # cold side: m_s T_in = (1-beta) m_c T_c + m_r T_h = m_s[(1-psi) T_c + psi T_h]
    # server   : T_h = T_in + dTs
    # => T_in = T_c + psi/(1-psi) dTs ;  T_h = T_c + dTs/(1-psi)
    T_in = T_c + psi / (1 - psi) * dTs
    T_h = T_in + dTs
    # hot side / return mixing: m_c T_ret = (m_s - m_r) T_h + m_b T_c  ->  T_ret = (1-beta)T_h + beta T_c
    T_ret = ((m_s - m_r) * T_h + m_b * T_c) / m_c
    Q_coil = m_c * CP * (T_ret - T_c)                 # closure: must equal Q_air
    # --- passive-scalar view: c = mass fraction of "exhaust" air (c=0 at IRC outlet, 1 at server outlet)
    c_in = (m_r * 1.0 + (1 - beta) * m_c * 0.0) / m_s  # = psi
    c_ret = ((m_s - m_r) * 1.0 + m_b * 0.0) / m_c      # = 1-beta
    # --- thermal indices
    # Sharma, Bash & Patel (2002) AIAA 2002-3091: SHI = dQ/(Q+dQ), RHI = Q/(Q+dQ), T_ref = supply T_c
    dQ = m_s * CP * (T_in - T_c)                      # enthalpy rise of cold air before rack inlet
    SHI = dQ / (Q_air + dQ)                           # = (T_in-T_c)/(T_h-T_c) = psi
    RHI = Q_air / (Q_air + dQ)                        # = (T_h-T_in)/(T_h-T_c) = 1-psi
    # Herrlin (2008) RTI = (T_ret - T_sup)/dT_equip * 100 %  (=m_s/m_c in a closed balance)
    RTI = (T_ret - T_c) / dTs * 100.0
    # --- exergy destruction by adiabatic isobaric mixing of ideal gas: S_gen = cp[sum m_out lnT - sum m_in lnT]
    K = 273.15
    S_rec = CP * (m_s * np.log(T_in + K) - (1 - beta) * m_c * np.log(T_c + K) - m_r * np.log(T_h + K))
    S_byp = CP * (m_c * np.log(T_ret + K) - (m_s - m_r) * np.log(T_h + K) - m_b * np.log(T_c + K))
    X_rec = P.T0_K * S_rec                            # Gouy-Stodola
    X_byp = P.T0_K * S_byp
    # pressure (throttling) term through leaks/dampers, isothermal ideal gas:
    # S_gen = m R ln(P_hi/P_lo) ~ m R dP/P0  ->  X = T0 m R dP / P0  (shown separately; tiny)
    X_thr = P.T0_K * (m_r + m_b) * R * np.log((P.P0 + abs(dP)) / P.P0)
    return _f({
        "m_s": m_s, "Q_air": Q_air, "T_c": T_c, "psi": psi, "beta": beta,
        "dT_server": dTs, "T_in": T_in, "T_h": T_h, "T_ret": T_ret,
        "m_c": m_c, "m_r": m_r, "m_b": m_b, "m_exhaust_to_irc": m_hot_to_irc,
        "flow_ratio_mc_ms": m_c / m_s, "dT_coil": T_ret - T_c,
        "Q_coil": Q_coil, "closure_err_W": Q_coil - Q_air, "mass_err": mass_err,
        "c_in": c_in, "c_ret": c_ret,
        "SHI": SHI, "RHI": RHI, "SHI_plus_RHI": SHI + RHI, "RTI": RTI,
        "S_gen_rec": S_rec, "S_gen_byp": S_byp,
        "X_dest_rec": X_rec, "X_dest_byp": X_byp, "X_dest_mix": X_rec + X_byp,
        "X_dest_throttle": X_thr,
        "config": 0.0 if config == "CAC" else 1.0,
    })


def solve_for_inlet(m_s, Q_air, T_in_target, psi, beta, config="CAC", dP=0.0,
                    T_chw_in=P.BASE["T_chw_in"], coil_approach=3.0):
    """Required IRC supply temperature for a target server inlet temperature.

    T_c = T_in - psi/(1-psi) * Q_air/(m_s cp)       (inverse of aisle_mixing)
    Adds feasibility flags: T_c >= T_chw_in + approach (coil can reach it; approach is a
    placeholder for S4) and T_c > room dew point (sensible-only coil, no condensate).
    """
    dTs = Q_air / (m_s * CP)
    T_c = T_in_target - psi / (1 - psi) * dTs
    out = aisle_mixing(m_s, Q_air, T_c, psi, beta, config, dP)
    T_c_min = T_chw_in + coil_approach
    # psi limit from T_c >= T_c_min : psi/(1-psi) <= (T_in - T_c_min)/dTs
    a = max(T_in_target - T_c_min, 0.0) / dTs
    a_dp = max(T_in_target - P.ROOM_DEWPOINT_C, 0.0) / dTs
    out.update({
        "T_c_required": T_c, "T_in_target": T_in_target,
        "T_c_min_coil": T_c_min,
        "feasible_coil": float(T_c >= T_c_min),
        "feasible_dewpoint": float(T_c > P.ROOM_DEWPOINT_C),
        "psi_max_coil": a / (1 + a),
        "psi_max_dewpoint": a_dp / (1 + a_dp),
    })
    return out


# ================================================================ pressure-based flow network
def _orifice(A, dP, rho, Cd=CD_ORIFICE):
    """Signed orifice mass flow, m = Cd A sqrt(2 rho |dP|) sign(dP)  (positive = cold -> hot)."""
    return np.sign(dP) * Cd * A * np.sqrt(2.0 * rho * abs(dP))


def flow_network(m_s0, m_c=None, A_L=None, A_r=0.0, m_bypass_fan=0.0, dP_set=None,
                 T_cold=24.0, T_hot=37.0, dP_shutoff=DP_SHUTOFF_SERVER, Cd=CD_ORIFICE,
                 leak_frac=0.005, config="CAC"):
    """Two-node pressure flow network model (FNM) of the contained pod.

    Nodes: cold side (pressure P_c) and hot side (P_h); unknown dP = P_c - P_h.
    Elements:
      * servers: net fan curve (fan minus chassis), quadratic:  P_h - P_c = dP_sh [1 - (V/V_f)^2]
        with V_f = free-delivery flow at dP = 0 (= nominal m_s0)  ->  m_s = m_s0 sqrt(1 + dP/dP_sh)
      * IRC fans: speed-controlled, deliver commanded m_c (hot -> cold), or -- if dP_set is given --
        the controller finds the m_c that holds the cold side at dP_set (common practice:
        slight positive pressurisation, a few Pa).
      * containment leakage area A_L and recirculation damper A_r: orifices,
        m = Cd A sqrt(2 rho |dP|), flow direction set by sign(dP).
      * optional low-lift bypass fans (Demetriou & Khalifa 2012) moving m_bypass_fan hot -> cold.
    Cold-side mass balance:  m_c + m_bypass_fan - m_s(dP) - m_open(dP) = 0.

    Mapping to the lumped model:
      dP > 0 : openings carry COLD air out  -> beta = m_open/m_c, psi = m_bypass_fan/m_s
      dP < 0 : openings draw HOT air in     -> beta = 0, psi = (m_open_in + m_bypass_fan)/m_s
    A single dP means passive leakage and passive recirculation cannot coexist through the same
    openings: passive bypass-recirculation REQUIRES cold-side under-pressure (cooler under-supply);
    with a pressurised aisle, recirculation needs fans.
    """
    if A_L is None:
        A_L = leak_area_from_fraction(leak_frac, config)["A_L"]
    rho_c, rho_h = P.rho_air(T_cold), P.rho_air(T_hot)
    A_tot = A_L + A_r

    def m_srv(dP):
        return m_s0 * np.sqrt(max(1.0 + dP / dP_shutoff, 0.0))

    def m_open(dP):  # positive = out of cold side
        return _orifice(A_tot, dP, rho_c if dP > 0 else rho_h, Cd)

    if dP_set is not None:
        dP = float(dP_set)
        m_c = m_srv(dP) + m_open(dP) - m_bypass_fan
    else:
        f = lambda dp: m_c + m_bypass_fan - m_srv(dp) - m_open(dp)   # decreasing in dp
        lo, hi = -0.999 * dP_shutoff, 2000.0
        dP = brentq(f, lo, hi, xtol=1e-10, rtol=1e-12)
    ms = m_srv(dP)
    mo = m_open(dP)
    if dP >= 0:
        m_leak_out, m_in = mo, 0.0
        beta = m_leak_out / m_c if m_c > 0 else 0.0
        psi = m_bypass_fan / ms
    else:
        m_leak_out, m_in = 0.0, -mo
        beta = 0.0
        psi = (m_in + m_bypass_fan) / ms
    # split inflow between uncontrolled leaks and the damper (same dP, proportional to area)
    psi_damper = (m_in * A_r / A_tot) / ms if A_tot > 0 else 0.0
    # flow work dissipated in the openings (exergy destroyed by throttling)
    rho_o = rho_c if dP > 0 else rho_h
    X_thr = abs(mo) * abs(dP) / rho_o
    return _f({
        "dP": dP, "m_c": m_c, "m_s": ms, "m_s0": m_s0, "m_server_change": ms / m_s0 - 1,
        "m_leak_out": m_leak_out, "m_infiltration": m_in, "m_bypass_fan": m_bypass_fan,
        "beta": beta, "psi": psi, "psi_damper": psi_damper,
        "psi_leaks": (m_in * A_L / A_tot) / ms if A_tot > 0 else 0.0,
        "psi_fan": m_bypass_fan / ms,
        "A_L": A_L, "A_r": A_r, "A_L_cm2_per_rack": A_L * 1e4 / P.RACKS_PER_POD,
        "A_L_frac_surface": A_L / containment_surface(config)["A_surface"],
        "supply_ratio": m_c / m_s0, "X_dest_throttle": X_thr,
        "mass_residual": m_c + m_bypass_fan - ms - mo,
    })


# ================================================================ self-test / figures
def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
        "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "font.size": 9, "axes.titlesize": 9.5, "legend.frameon": False,
    })


SEQ = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]       # ordinal blue ramp (beta levels)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]        # categorical slots 1-4
INK, INK2 = "#0b0b0b", "#52514e"


def _schematic(path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, cfg in zip(axes, ["CAC", "HAC"]):
        ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off"); ax.grid(False)
        cold_cont = cfg == "CAC"

        def box(x, y, w, h, label, fc, ec, ls="-", lw=1.5, fs=8.5):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                        fc=fc, ec=ec, lw=lw, ls=ls))
            ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=INK)

        def arr(p, q, lab, col, off=(0, 0.18), rad=0.0):
            ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=2, color=col,
                                         connectionstyle=f"arc3,rad={rad}"))
            ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], lab, ha="center",
                    fontsize=8, color=INK)
        # CVs
        box(0.3, 0.6, 3.6, 5.2, "", "#e8f1fc", SEQ[2], ls="-" if cold_cont else "--",
            lw=2.4 if cold_cont else 1.2)
        ax.text(2.1, 5.45, ("CV1: CONTAINED cold aisle" if cold_cont else "CV1: open room (cold side)")
                + "\nwell-mixed, $T_{in}$", ha="center", fontsize=8.5, color=INK)
        box(6.1, 0.6, 3.6, 5.2, "", "#fdece4", "#d95926", ls="--" if cold_cont else "-",
            lw=1.2 if cold_cont else 2.4)
        ax.text(7.9, 5.45, ("CV2: open room (hot side)" if cold_cont else "CV2: CONTAINED hot aisle")
                + "\nwell-mixed, $T_h$ -> $T_{ret}$", ha="center", fontsize=8.5, color=INK)
        # elements
        box(4.2, 3.6, 1.6, 1.0, "Servers\n$m_s$, $Q_{air}$", "#f0efec", INK2)
        box(4.2, 0.9, 1.6, 1.0, "IRC coil+fans\n$m_c$", "#f0efec", INK2)
        # streams
        arr((3.9, 4.1), (4.2, 4.1), "", CAT[0])
        ax.text(3.0, 4.25, "$m_s$, $T_{in}$", fontsize=8, color=INK)
        arr((5.8, 4.1), (6.1, 4.1), "", CAT[1]); ax.text(6.2, 4.25, "$m_s$, $T_h$", fontsize=8)
        arr((6.1, 1.4), (5.8, 1.4), "", CAT[1]); ax.text(6.2, 1.0, "$m_c$, $T_{ret}$", fontsize=8)
        arr((4.2, 1.4), (3.9, 1.4), "", CAT[0]); ax.text(2.6, 1.0, "$m_c$, $T_c$", fontsize=8)
        arr((3.9, 2.3), (6.1, 2.3), r"leakage $m_b=\beta m_c$ ($T_c$)", SEQ[1])
        arr((6.1, 3.0), (3.9, 3.0), r"recirc. $m_r=\psi m_s$ ($T_h$)", "#d95926")
        ax.text(5.0, 6.45, f"{cfg} with in-row coolers", ha="center", fontsize=11, weight="bold")
        ax.text(5.0, 0.15, r"cold: $(1-\beta)m_c+m_r=m_s$   hot: $m_s-m_r+\beta m_c=m_c$",
                ha="center", fontsize=8, color=INK2)
    fig.suptitle("Control volumes: IRC draws from the hot side, supplies the cold side; "
                 "the same balances hold for CAC and HAC", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()
    here = os.path.dirname(os.path.abspath(__file__))
    fdir, rdir = os.path.join(here, "figures"), os.path.join(here, "results")
    os.makedirs(fdir, exist_ok=True); os.makedirs(rdir, exist_ok=True)

    import s2_air as _s2; _pa = _s2.pod_air(P.BASE["T_in_server"]); m_s, Q_air, T_in = _pa["m_dot_servers"], _pa["Q_air_pod"], P.BASE["T_in_server"]   # integrated S2 values (was placeholder 6 kg/s, 80 kW)
    beta0 = P.BASE["beta"]
    psis = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # ---------------- baseline table
    print(f"S3 baseline: m_s={m_s} kg/s, Q_air={Q_air/1e3:.0f} kW, T_in={T_in} C, beta={beta0}")
    hdr = f"{'psi':>5}{'T_c':>7}{'T_h':>7}{'T_ret':>7}{'m_c':>7}{'mc/ms':>7}{'SHI':>6}{'RHI':>6}" \
          f"{'RTI%':>7}{'Xrec W':>8}{'Xbyp W':>8}{'Qerr W':>9}{'coilOK':>7}"
    print(hdr)
    table = []
    for psi in psis:
        r = solve_for_inlet(m_s, Q_air, T_in, psi, beta0)
        table.append(r)
        print(f"{psi:5.2f}{r['T_c']:7.2f}{r['T_h']:7.2f}{r['T_ret']:7.2f}{r['m_c']:7.3f}"
              f"{r['flow_ratio_mc_ms']:7.3f}{r['SHI']:6.2f}{r['RHI']:6.2f}{r['RTI']:7.1f}"
              f"{r['X_dest_rec']:8.1f}{r['X_dest_byp']:8.1f}{r['closure_err_W']:9.2e}"
              f"{int(r['feasible_coil']):7d}")
    base = table[0]
    print(f"psi_max (T_c >= T_chw+3K = {base['T_c_min_coil']:.1f} C): {base['psi_max_coil']:.3f};"
          f"  psi_max (T_c > dew point {P.ROOM_DEWPOINT_C} C): {base['psi_max_dewpoint']:.3f}")

    # ---------------- FNM baseline
    geo = containment_surface()
    fn_base = flow_network(m_s, dP_set=2.5, leak_frac=0.005, T_cold=T_in, T_hot=base["T_h"])
    print(f"FNM: surface {geo['A_surface']:.1f} m2; 0.5 % open area = {fn_base['A_L']*1e4:.0f} cm2 "
          f"({fn_base['A_L_cm2_per_rack']:.0f} cm2/rack) at dP=+2.5 Pa -> beta={fn_base['beta']:.3f}, "
          f"m_c={fn_base['m_c']:.3f} kg/s, server flow change {fn_base['m_server_change']*100:+.2f} %")
    # beta = 0.05 at 2.5 Pa -> required leakage area
    f5 = lambda fr: flow_network(m_s, dP_set=2.5, leak_frac=fr, T_cold=T_in, T_hot=base["T_h"])["beta"] - 0.05
    fr05 = brentq(f5, 1e-5, 0.2)
    A05 = leak_area_from_fraction(fr05)
    print(f"FNM: beta=0.05 at +2.5 Pa needs open area {fr05*100:.2f} % of surface "
          f"= {A05['A_L']*1e4:.0f} cm2 ({A05['A_L_cm2_per_rack']:.0f} cm2/rack)")
    # passive recirculation psi=0.2 at -2 Pa: damper area
    A_need = 0.2 * m_s / (CD_ORIFICE * np.sqrt(2 * P.rho_air(base['T_h']) * 2.0))
    print(f"FNM: passive psi=0.20 at -2 Pa needs total opening {A_need:.2f} m2 "
          f"({A_need/P.RACKS_PER_POD*1e4:.0f} cm2/rack)")

    # ---------------- Figure 1: schematic
    _schematic(os.path.join(fdir, "s3_schematic.png"))

    # ---------------- Figure 2: T_c, T_ret, m_c vs psi for several beta
    psi_v = np.linspace(0, 0.5, 101)
    betas = [0.0, 0.05, 0.10, 0.20]
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))
    for b, col in zip(betas, SEQ):
        rr = [solve_for_inlet(m_s, Q_air, T_in, p, b) for p in psi_v]
        lab = rf"$\beta$={b:.2f}"
        ax[0, 0].plot(psi_v, [r["T_c"] for r in rr], color=col, label=lab)
        ax[0, 1].plot(psi_v, [r["T_ret"] for r in rr], color=col, label=lab)
        ax[1, 0].plot(psi_v, [r["m_c"] for r in rr], color=col, label=lab)
        mc0 = rr[0]["m_c"]
        ax[1, 1].plot(psi_v, [(r["m_c"] / mc0) ** 3 for r in rr], color=col, label=lab)
    ax[0, 0].axhline(base["T_c_min_coil"], color="#e34948", lw=1, ls="--")
    ax[0, 0].text(0.01, base["T_c_min_coil"] + 0.3, f"coil limit T_chw+3 K = {base['T_c_min_coil']:.0f} °C",
                  fontsize=7.5, color=INK2)
    ax[0, 0].axhline(P.ROOM_DEWPOINT_C, color=INK2, lw=1, ls=":")
    ax[0, 0].text(0.01, P.ROOM_DEWPOINT_C + 0.3, f"room dew point {P.ROOM_DEWPOINT_C:.0f} °C",
                  fontsize=7.5, color=INK2)
    ax[0, 0].axvline(base["psi_max_coil"], color="#e34948", lw=0.8, ls="--")
    ax[0, 0].set(title=f"Required T_c (independent of β; curves overlap): ψ_max≈{base['psi_max_coil']:.2f} at coil limit",
                 xlabel="recirculation fraction ψ [-]", ylabel="IRC supply T_c [°C]")
    ax[0, 1].set(title="T_ret=(1−β)T_h+βT_c: leakage dilutes the return, worse as T_c drops",
                 xlabel="recirculation fraction ψ [-]", ylabel="IRC return T_ret [°C]")
    ax[1, 0].set(title="Cooler flow m_c = m_s(1-ψ)/(1-β): recirculation unloads the IRC fans",
                 xlabel="recirculation fraction ψ [-]", ylabel="IRC air flow m_c [kg/s]")
    ax[1, 1].set(title="Fan-affinity ratio (m_c/m_c,ψ=0)³ = (1−ψ)³, any β: −49 % at ψ=0.2",
                 xlabel="recirculation fraction ψ [-]", ylabel="IRC fan power ratio [-]")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"Bypass recirculation at fixed T_in={T_in:.0f} °C (m_s={m_s} kg/s, Q_air={Q_air/1e3:.0f} kW): "
                 "less IRC air, colder supply — the fan/chiller trade-off", fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(fdir, "s3_mixing_vs_psi.png"), dpi=200); plt.close(fig)

    # ---------------- Figure 3: indices
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    shi = [aisle_mixing(m_s, Q_air, 18.0, p, beta0)["SHI"] for p in psi_v]
    rhi = [aisle_mixing(m_s, Q_air, 18.0, p, beta0)["RHI"] for p in psi_v]
    ax[0].plot(psi_v, shi, color=CAT[1], label="SHI")
    ax[0].plot(psi_v, rhi, color=CAT[0], label="RHI")
    ax[0].plot(psi_v, np.array(shi) + np.array(rhi), color=INK2, ls="--", lw=1.2, label="SHI+RHI")
    for b, mk in zip([0.0, 0.2], ["o", "s"]):
        ss = [aisle_mixing(m_s, Q_air, 18.0, p, b)["SHI"] for p in psis]
        ax[0].plot(psis, ss, mk, ms=6, mfc="none", color=INK2, label=rf"SHI at β={b}")
    ax[0].set(title="SHI = ψ and RHI = 1−ψ exactly (independent of β); sum ≡ 1",
              xlabel="recirculation fraction ψ [-]", ylabel="index [-]")
    ax[0].legend(fontsize=7.5)
    for b, col in zip(betas, SEQ):
        ax[1].plot(psi_v, [aisle_mixing(m_s, Q_air, 18.0, p, b)["RTI"] for p in psi_v], color=col,
                   label=rf"β={b:.2f}")
    ax[1].axhline(100, color=INK2, lw=1, ls="--")
    ax[1].text(0.30, 103, "RTI>100 %: net recirculation", fontsize=7.5, color=INK2)
    ax[1].text(0.30, 94, "RTI<100 %: net bypass", fontsize=7.5, color=INK2)
    ax[1].set(title="RTI = (1−β)/(1−ψ) = m_s/m_c: balanced (100 %) when ψ = β",
              xlabel="recirculation fraction ψ [-]", ylabel="RTI [%]")
    ax[1].legend(fontsize=7.5)
    fig.suptitle("Thermal indices under perfect mixing — SHI sees only recirculation, RTI sees both", fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(fdir, "s3_indices.png"), dpi=200); plt.close(fig)

    # ---------------- Figure 4: exergy destruction map
    pg = np.linspace(0, 0.5, 81); bg = np.linspace(0, 0.30, 61)
    X = np.zeros((bg.size, pg.size))
    for i, b in enumerate(bg):
        for j, p in enumerate(pg):
            X[i, j] = solve_for_inlet(m_s, Q_air, T_in, p, b)["X_dest_mix"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3))
    cs = ax[0].contourf(pg, bg, X / 1e3, levels=14, cmap="Blues")
    cl = ax[0].contour(pg, bg, X / 1e3, levels=[0.1, 0.25, 0.5, 1.0, 1.5], colors=INK2, linewidths=0.7)
    ax[0].clabel(cl, fmt="%.2f kW", fontsize=7)
    fig.colorbar(cs, ax=ax[0], label="X_dest,mix [kW]")
    ax[0].plot([0.0], [beta0], "o", color="#e34948", ms=6)
    ax[0].set(title="Mixing exergy destruction grows ~quadratically with ψ (fixed T_in)",
              xlabel="recirculation fraction ψ [-]", ylabel="leakage fraction β [-]")
    rr = [solve_for_inlet(m_s, Q_air, T_in, p, beta0) for p in psi_v]
    ax[1].plot(psi_v, [r["X_dest_rec"] for r in rr], color=CAT[1], label="cold-aisle recirculation mixing")
    ax[1].plot(psi_v, [r["X_dest_byp"] for r in rr], color=CAT[0], label="return (leakage) mixing")
    ax[1].plot(psi_v, [r["X_dest_mix"] for r in rr], color=INK, lw=1.4, ls="--", label="total")
    ax[1].set(title=f"At β={beta0}: X_dest,mix = {rr[0]['X_dest_mix']:.0f} W (ψ=0) → "
                    f"{rr[40]['X_dest_mix']:.0f} W (ψ=0.2), ≤{rr[40]['X_dest_mix']/Q_air*100:.1f} % of Q_air",
              xlabel="recirculation fraction ψ [-]", ylabel="exergy destroyed [W]")
    ax[1].legend(fontsize=7.5)
    fig.suptitle(f"Exergy destroyed by mixing, T0={P.T0_C:.0f} °C (marker: baseline ψ=0, β={beta0})", fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(fdir, "s3_exergy_map.png"), dpi=200); plt.close(fig)

    # ---------------- Figure 5: flow network
    fr = np.linspace(0, 0.02, 81)
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for dps, col in zip([1.0, 2.5, 5.0, 10.0], SEQ):
        bb = [flow_network(m_s, dP_set=dps, leak_frac=f, T_cold=T_in, T_hot=base["T_h"])["beta"] for f in fr]
        ax[0].plot(fr * 100, bb, color=col, label=f"ΔP = +{dps:g} Pa")
    ax[0].axhspan(0.03, 0.10, color="#f0efec", zorder=0)
    ax[0].text(0.05, 0.105, "typical containment 3–10 % (Schneider WP135)", fontsize=7, color=INK2)
    sec = ax[0].secondary_xaxis("top", functions=(lambda x: x / 100 * geo["A_surface"] * 1e4 / P.RACKS_PER_POD,
                                                    lambda y: y * 100 / (geo["A_surface"] * 1e4 / P.RACKS_PER_POD)))
    sec.set_xlabel("leakage area [cm² per rack]")
    ax[0].set(title="Seal quality → β: 1 % open area ≈ 7 % leakage at +2.5 Pa",
              xlabel=f"open area [% of {geo['A_surface']:.1f} m² containment surface]", ylabel="β [-]")
    ax[0].legend(fontsize=7.5)
    sr = np.linspace(0.7, 1.3, 121)
    for (lf, ar, lab), col in zip([(0.005, 0.0, "0.5 % leaks, damper shut"),
                                   (0.005, 0.5, "0.5 % leaks + 0.5 m² damper"),
                                   (0.02, 0.0, "2 % leaks, damper shut")], CAT[:3]):
        rr = [flow_network(m_s, m_c=s * m_s, A_r=ar, leak_frac=lf, T_cold=T_in, T_hot=base["T_h"]) for s in sr]
        ax[1].plot(sr, [r["dP"] for r in rr], color=col, label=lab)
        ax[2].plot(sr, [r["beta"] for r in rr], color=col, label=lab + " — β")
        ax[2].plot(sr, [r["psi"] for r in rr], color=col, ls="--", label=lab + " — ψ")
    ax[1].axhline(0, color=INK2, lw=0.8); ax[1].axvline(1, color=INK2, lw=0.8)
    ax[1].set_yscale("symlog", linthresh=1.0)
    ax[1].set(title="Over-supply pressurises the cold aisle; under-supply depressurises it",
              xlabel="IRC supply ratio m_c/m_s0 [-]", ylabel="ΔP = P_cold − P_hot [Pa]")
    ax[1].legend(fontsize=7.5)
    ax[2].axvline(1, color=INK2, lw=0.8)
    ax[2].set(title="ΔP>0 → leakage β; ΔP<0 → hot ingress ψ\n(never both passively)",
              xlabel="IRC supply ratio m_c/m_s0 [-]", ylabel="β (solid), ψ (dashed) [-]")
    ax[2].legend(fontsize=6.5)
    fig.suptitle("Two-node pressure flow-network model of the contained pod (orifice leaks, Cd=0.61; "
                 f"server net fan shut-off {DP_SHUTOFF_SERVER:.0f} Pa)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(fdir, "s3_fnm.png"), dpi=200); plt.close(fig)

    # ---------------- JSON
    keep = ["psi", "beta", "T_c", "T_in", "T_h", "T_ret", "m_c", "m_r", "m_b", "flow_ratio_mc_ms",
            "SHI", "RHI", "RTI", "X_dest_rec", "X_dest_byp", "X_dest_mix", "closure_err_W", "feasible_coil"]
    out = {
        "inputs": {"m_s": m_s, "Q_air_W": Q_air, "T_in_target": T_in, "beta": beta0,
                   "note": "m_s, Q_air are placeholders pending S2 pod_air()"},
        "baseline_psi0": {k: float(base[k]) for k in keep},
        "psi_sweep": [{k: float(r[k]) for k in keep} for r in table],
        "psi_max_coil_limit": float(base["psi_max_coil"]),
        "psi_max_dewpoint": float(base["psi_max_dewpoint"]),
        "T_c_min_coil": float(base["T_c_min_coil"]),
        "fnm": {"A_surface_m2": geo["A_surface"],
                "leak_frac_for_beta0.05_at_2.5Pa": fr05, "A_L_cm2_per_rack_for_beta0.05": A05["A_L_cm2_per_rack"],
                "beta_at_0.5pct_2.5Pa": fn_base["beta"], "server_flow_change_at_2.5Pa": fn_base["m_server_change"],
                "passive_psi0.2_open_area_m2_at_-2Pa": float(A_need),
                "Cd": CD_ORIFICE, "dP_shutoff_server_Pa": DP_SHUTOFF_SERVER},
    }
    with open(os.path.join(rdir, "s3_baseline.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote figures/s3_*.png and results/s3_baseline.json")


if __name__ == "__main__":
    main()
