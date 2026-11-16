"""
Component- and system-level analyses that complete the project brief, all on the C-DAC configuration
where a pod total is involved. Writes results/sim_gaps.json.

 1 coolant_props      : rho, cp, k, mu, Pr of water / PG25 / EG25 / Novec 7500, 15-60 degC      (Student 1)
 2 coldplate_map      : junction temp, exergy destruction and Bejan number over flow x inlet T   (Student 1)
 3 pod_loop           : two plates per node in series vs parallel at C-DAC flows                 (Student 1)
 4 fan_curves         : server fan curves at several speeds, chassis system curve, power vs flow  (Student 2)
 5 chassis_profile    : air and component temperatures through the chassis zones               (Student 2)
 6 board_measures     : effect of board thermal management on air share and air-to-liquid ratio (Student 2)
 7 loop_first_law     : first-law comparison of the liquid and air heat paths                     (scope)
 8 rdhx               : rear-door coil by eps-NTU, passive vs active fans                          (Student 4)
 9 chw_optimisation   : chilled-water temperature / flow trade-off at psi = 0 and optimum psi      (Student 4)
10 sensitivity        : optimum cooling power under changed assumptions; dead-state effect         (Student 5)
Run: python3 sim_gaps.py  (~2 min)
"""
import json, math, copy
import numpy as np
import params as P
import s1_liquid as s1
import s2_air as s2
import s3_aisle as s3
import s4_irc as s4
import s5_plant as S
import s5_optimize as O

N = P.CDAC["n_nodes"]
out = {}

# 1 ------------------------------------------------------------------------------------------
T = np.arange(15, 60.1, 2.5)
out["coolant_props"] = {f: {k: [s1.coolant_props(t, f)[k] for t in T] for k in ("rho", "cp", "k", "mu", "Pr")}
                        for f in ("water", "PG25", "EG25", "Novec7500")}
out["coolant_props"]["T"] = T.tolist()

# 2 ------------------------------------------------------------------------------------------
Q = 165.0 * s1.CAPTURE_FRAC
flows = np.round(np.arange(0.2, 2.01, 0.1), 2); tins = np.arange(17.0, 50.1, 1.5)
Tj = np.zeros((len(tins), len(flows))); Xd = np.zeros_like(Tj); Be = np.zeros_like(Tj); Tc = np.zeros_like(Tj)
for i, t in enumerate(tins):
    for j, f in enumerate(flows):
        r = s1.cold_plate(Q, f / 60000 * s1.coolant_props(t)["rho"], t)
        Tj[i, j], Xd[i, j], Be[i, j], Tc[i, j] = r["T_j"], r["X_dest_total"], r["Bejan"], r["T_case"]
out["coldplate_map"] = dict(flow=flows.tolist(), T_in=tins.tolist(), T_j=Tj.tolist(), X_dest=Xd.tolist(),
                            Bejan=Be.tolist(), T_case=Tc.tolist(), Q_plate=Q)
# junction temperature against flow for the line plot
out["coldplate_lines"] = {str(t): [s1.cold_plate(Q, f / 60000 * s1.coolant_props(t)["rho"], t)["T_j"]
                                   for f in np.linspace(0.15, 2.0, 40)] for t in (30, 37, 45)}
out["coldplate_lines"]["flow"] = np.linspace(0.15, 2.0, 40).tolist()

# 3 ------------------------------------------------------------------------------------------
rows = []
for cfg, lpm_node in [("series", 0.5), ("parallel", 0.5), ("series", 1.0), ("parallel", 1.0)]:
    m_plate = (lpm_node if cfg == "series" else lpm_node / 2) / 60000 * s1.coolant_props(37.0)["rho"]
    lq = s1.liquid_loop(2 * Q, m_plate, 37.0, N, config=cfg)
    rows.append(dict(config=cfg, lpm_node=lpm_node, W_pump=lq["W_pump"], dP_loop=lq["dP_loop"], T_return=lq["T_return"],
                     T_case_max=lq["T_case_max"], T_j_max=lq["T_j_max"], X_cp=lq["X_dest_coldplates"]))
out["pod_loop"] = rows

# 4 ------------------------------------------------------------------------------------------
V0t = s2.N_FANS * s2.FAN_V0
def fan_dp(V, Nn):
    x = V / (Nn * V0t)
    return Nn ** 2 * s2.FAN_DP0 * (1 - s2.FAN_SHAPE_A * x - (1 - s2.FAN_SHAPE_A) * x ** 2)
curves = {}
for Nn in (0.25, 0.5, 0.75, 1.0):
    V = np.linspace(0, Nn * V0t, 60)
    curves[str(Nn)] = dict(V=V.tolist(), dP=[max(fan_dp(v, Nn), 0) for v in V])
ops = [s2.operating_point(Nn) for Nn in np.linspace(0.25, 1.0, 16)]
out["fan_curves"] = dict(curves=curves, k_sys=s2.K_SYS,
                         op=[dict(N=float(n), V=o["V_dot"], dP=o["dP"], W=s2.fan_power(n)) for n, o in zip(np.linspace(0.25, 1.0, 16), ops)],
                         check_op_on_fan_curve=[abs(fan_dp(o["V_dot"], n) - o["dP"]) for n, o in zip(np.linspace(0.25, 1.0, 16), ops)])

# 5 ------------------------------------------------------------------------------------------
prof = {}
for Tin in (24.0, 27.0, 32.0):
    nb = s2.node_heat_breakdown(1.0, Tin); sa = s2.server_air(Tin, nb["Q_air"])
    prof[str(Tin)] = {k: sa[k] for k in ("T_in", "T_air_z1", "T_air_fan", "T_air_z2", "T_air_z3", "T_out",
                                          "T_dimm_case", "T_vrm_j", "T_pch_case", "T_nic_case", "T_ssd_case",
                                          "fan_speed_frac", "V_dot_cfm", "W_fan")}
out["chassis_profile"] = prof

# 6 ------------------------------------------------------------------------------------------
cp_liq = s1.coolant_props(41.0)["cp"]; m_liq_node = 0.5 / 60000 * s1.coolant_props(37.0)["rho"]
def measure(name, **kw):
    res = {}
    for Tin in (27.0, 32.0):
        nb = s2.node_heat_breakdown(1.0, Tin, **kw)
        comp = s2._components(1.0, nb["fans"], nb["capture_frac"], **{k: v for k, v in kw.items() if k != "capture"})
        sa = s2.server_air(Tin, nb["Q_air"], comp=comp)
        res[str(Tin)] = dict(Q_air=nb["Q_air"], Q_liq=nb["Q_liquid"], air_frac=nb["air_frac"], cfm=sa["V_dot_cfm"],
                             fan_N=sa["fan_speed_frac"], W_fan=sa["W_fan"], m_air=sa["m_dot"],
                             C_ratio=sa["m_dot"] * P.CP_AIR / (m_liq_node * cp_liq),
                             V_ratio=sa["V_dot"] / (m_liq_node / s1.coolant_props(37.0)["rho"]),
                             Q_node=nb["Q_node_total"])
    return dict(name=name, **res)
out["board_measures"] = [
    measure("Baseline (CPU plates only, capture 0.92)"),
    measure("Capture 0.85", capture=0.85),
    measure("Capture 0.98", capture=0.98),
    measure("DIMM cold plates", dimm_liquid_frac=1.0),
    measure("Better VRs (0.95 / 0.93)", eta_vr_cpu=0.95, eta_vr_mem=0.93),
    measure("Titanium PSU", psu_table=s2.PSU_EFF_TITANIUM),
]

# 7 ------------------------------------------------------------------------------------------
b = S.system({**S.cdac_x(), "T_in": 24.0, "psi": 0.0, "T_chw": 14.0})
o = O.best_chw({**S.cdac_x(), "T_in": 27.0, "psi": 0.175})
def loops(r):
    W = r["W"]
    fac_share_liq = r["Q_econ"] / (r["Q_econ"] + (r["Q_coil"] + W["chw_pump"] + W["chiller"]))  # tower heat share
    liq = dict(Q=r["Q_liq"], flow_kg_s=r["m_fw"] and S.P.CDAC["lpm_node"] * N / 60000 * s1.coolant_props(37.0)["rho"],
               dT=r["T_tcs_ret"] - r["x"]["T_liq_in"], transport=W["tcs_pump"],
               refrigeration=W["trim_chiller"], rejection=(W["tower_fans"] + W["cw_fw_pumps"]) * fac_share_liq)
    air = dict(Q=r["Q_air"], flow_kg_s=r["m_s"], dT=r["T_h"] - r["x"]["T_in"],
               transport=W["server_fans"] + W["irc_fans"] + W["bypass_fans"], chw_pump=W["chw_pump"],
               refrigeration=W["chiller"], rejection=(W["tower_fans"] + W["cw_fw_pumps"]) * (1 - fac_share_liq))
    for d in (liq, air):
        d["total"] = d["transport"] + d["refrigeration"] + d["rejection"] + d.get("chw_pump", 0.0)
        d["W_per_kW"] = d["total"] / (d["Q"] / 1e3)
        d["transport_per_kW"] = d["transport"] / (d["Q"] / 1e3)
    return dict(liquid=liq, air=air, X=r["X"], X_total=r["X_dest_total"])
out["loop_first_law"] = dict(baseline=loops(b), optimised=loops(o))

# 8 ------------------------------------------------------------------------------------------
pa = s2.pod_air(24.0, 1.0, n_nodes=N)
m_rack = pa["m_dot_servers"] / 12; Q_rack = pa["Q_air_pod"] / 12
rd = []
for mw in (0.3, 0.5, 0.8):
    r = s4.rdhx(m_rack, pa["T_out"], mw, 14.0)
    rd.append(dict(m_w=mw, Q=r["Q"], eps=r.get("eps", float("nan")), NTU=r.get("NTU", float("nan")), T_air_out=r["T_air_out"],
                   dP_door=r["dP_door"], W_par=r["W_parasitic"], passive=r["passive_ok"]))
sweep = []
for V in np.linspace(0.3, 4.0, 30):
    m = V * P.rho_air(pa["T_out"])
    for gname, g in (("2-row", s4.RDHX_GEOM), ("4-row dense", s4.RDHX_DENSE_GEOM)):
        r = s4.rdhx(m, pa["T_out"], 0.8, 14.0, geom=g)
        sweep.append(dict(V=V, geom=gname, dP=r["dP_door"], W_passive=r["W_parasitic_passive"], W_active=r["W_parasitic_active"]))
out["rdhx"] = dict(m_air_rack=m_rack, Q_air_rack=Q_rack, T_in=pa["T_out"], cases=rd, sweep=sweep,
                   dP_avail=s4.RDHX["dP_avail_nominal"], dP_avail_range=s4.RDHX["dP_avail_range"])

# 9 ------------------------------------------------------------------------------------------
chw = {}
for psi, Tin in ((0.0, 24.0), (0.0, 27.0), (0.175, 27.0)):
    rows = []
    for t in np.arange(8.0, 20.01, 0.5):
        r = S.system({**S.cdac_x(), "T_in": Tin, "psi": psi, "T_chw": float(t)}, detail=False)
        rows.append(dict(T_chw=float(t), feasible=bool(r["feasible"]), W=r["W_cool_total"] if r["feasible"] else None,
                         m_w=r["m_w"] if r["feasible"] else None, T_w_out=r["T_w_out"] if r["feasible"] else None,
                         W_chiller=r["W"]["chiller"] if r["feasible"] else None, W_chw_pump=r["W"]["chw_pump"] if r["feasible"] else None))
    chw[f"psi{psi}_Tin{Tin}"] = rows
out["chw_optimisation"] = chw

# 10 -----------------------------------------------------------------------------------------
def best(xo=None, psis=np.arange(0, 0.326, 0.025), Tin=27.0):
    bw = None
    for p in psis:
        r = O.best_chw({**S.cdac_x(), **(xo or {}), "T_in": Tin, "psi": float(p)})
        if r is not None and (bw is None or r["W_cool_total"] < bw["W_cool_total"]):
            bw = r
    trad = O.best_chw({**S.cdac_x(), **(xo or {}), "T_in": Tin, "psi": 0.0})
    base = S.system({**S.cdac_x(), **(xo or {}), "T_in": 24.0, "psi": 0.0, "T_chw": 14.0}, detail=False)
    return dict(W_opt=bw["W_cool_total"], psi=bw["x"]["psi"], T_chw=bw["x"]["T_chw"], W_trad=trad["W_cool_total"],
                W_base=base["W_cool_total"] if base["feasible"] else None)
sens = []
sens.append(dict(case="Reference (wet bulb 25 °C, fan eff. 0.32, capture 0.92)", **best()))
sens.append(dict(case="Wet bulb 22 °C", **best({"T_wb": 22.0})))
sens.append(dict(case="Wet bulb 28 °C (design summer)", **best({"T_wb": 28.0})))
sens.append(dict(case="Cold-plate capture 1.00 (C-DAC sizing)", **best({"capture": 1.0})))
sens.append(dict(case="Leakage 2 %", **best({"beta": 0.02})))
s4.FAN["eta_tot"] = 0.45; s4._CAL.clear() if hasattr(s4, "_CAL") and isinstance(s4._CAL, dict) else None
sens.append(dict(case="Cooler fan efficiency 0.45 (modern EC)", **best()))
s4.FAN["eta_tot"] = None; s4._CAL.clear() if isinstance(getattr(s4, "_CAL", None), dict) else None
GN0 = S.GN.copy()
for pct in (1.0, 2.5):
    p2 = S._KW_PER_TON / (0.560 * (1 - pct / 100) ** 16)
    (q1, e1, c1, p1) = S._CAL[0]; e2, c2 = 15.56, 29.44
    x1, x2 = (c1 + 273.15) / (e1 + 273.15) - 1, (c2 + 273.15) / (e2 + 273.15) - 1
    k1 = (1 / p1 - 1 / p2) / (x1 - x2); S.GN = np.array([1 / p1 - k1 * x1, k1])
    sens.append(dict(case=f"Chiller gain {pct:g} % per °F of warmer CHW (ref. 1.7)", **best()))
S.GN = GN0
sens.append(dict(case="Server inlet up to 32 °C (A1 allowable)", **best(Tin=32.0)))
out["sensitivity"] = sens
# dead state: total destruction is fixed by the work; the split moves
ds = []
for T0 in (25.0, 30.0, 34.0):
    P.T0_C = T0; P.T0_K = T0 + 273.15; S.T0 = T0; S.T0K = T0 + 273.15
    rb = S.system({**S.cdac_x(), "T_in": 24.0, "psi": 0.0, "T_chw": 14.0})
    ro = O.best_chw({**S.cdac_x(), "T_in": 27.0, "psi": 0.175})
    el = "server: electronics, electricity to heat"
    m = s1.coolant_props(37.0)["rho"] * 0.5 * N / 60000; cpl = s1.coolant_props(41.5)["cp"]
    ds.append(dict(T0=T0, X_total_base=rb["X_dest_total"], X_total_opt=ro["X_dest_total"],
                   X_cool_base=rb["X_dest_total"] - rb["X"][el], X_cool_opt=ro["X_dest_total"] - ro["X"][el],
                   X_recoverable=S.x_flow(m, cpl, ro["T_tcs_ret"]) - S.x_flow(m, cpl, 37.0),
                   closure=ro["X_closure"]))
P.T0_C = 30.0; P.T0_K = 303.15; S.T0 = 30.0; S.T0K = 303.15
out["dead_state"] = ds

json.dump(out, open("results/sim_gaps.json", "w"), indent=1, default=float)
print("ok")
for s in sens: print(f"{s['case']:58s} base {s['W_base'] and s['W_base']/1e3:6.2f} trad {s['W_trad']/1e3:6.2f} opt {s['W_opt']/1e3:6.2f} psi {s['psi']:.3f} chw {s['T_chw']:.1f}")
for d in ds: print(d)
