"""
S5 — Plant models, system integration, exergy ledger, COP and optimisation.
PARAM Rudra (IIT Bombay) hybrid-cooled enclosed-aisle pod.

Couples S1 (cold-plate/TCS loop), S2 (node heat + server fans), S3 (aisle mixing,
bypass beta / recirculation psi), S4 (in-row coolers, eps-NTU) with:
  * a water-cooled centrifugal chiller: two-parameter lift model (Gordon-Ng family) x DOE-2 EIR-fPLR,
    calibrated to ASHRAE 90.1 Path A (>=300 t: 0.560 kW/ton full load, 0.500 IPLV);
  * a cooling tower (approach model, fan power from the 90.1 38.2 gpm/hp minimum);
  * a CDU (balanced counterflow plate HX) with water-side economiser / trim chiller;
  * a stream-exergy ledger (Gouy-Stodola) whose component boxes are closed by
    stream balances and cross-checked against each module's internal X_dest.

All temperatures degC unless suffixed _K. Powers W. Run: python3 s5_plant.py
"""
import json, time
import numpy as np
from scipy.optimize import differential_evolution
import CoolProp.CoolProp as CP

import params as P
import s1_liquid as s1
import s2_air as s2
import s3_aisle as s3
import s4_irc as s4

T0 = P.T0_C
T0K = P.T0_K
CP_W = 4186.0          # J/kg/K, water (plant loops); PG25 properties come from S1
K = 273.15

# --------------------------------------------------------------------------- plant assumptions
PLANT = dict(
    tower_approach=4.0,        # K, leaving CW - ambient wet bulb (typical 3-5 K design, CTI)
    cw_range=5.0,              # K, condenser water range (AHRI 3 gpm/ton -> ~5.6 K)
    tower_fan_kw_per_kw=0.0133,# kW/kW rejected: ASHRAE 90.1 Table 6.8.1-7 open axial tower >=38.2 gpm/hp
                               # at 95/85/75 F: 38.2 gpm*1.465 kW/gpm per 0.746 kW -> 0.0133
    cw_pump_dp=200e3,          # Pa, condenser loop head (tower static + condenser + piping), assumption
    fw_pump_dp=150e3,          # Pa, facility-water loop to CDU primary, assumption
    pump_eta=0.70,             # wire-to-water, large plant pumps
    UA_cdu=60e3,               # W/K, CDU plate HX: balanced counterflow -> approach = Q/UA ~ 2.9 K at 175 kW
    Q_rated_chiller=100e3,     # W, pod share of ONE chiller module; plant stages modules (see chiller())
    stage_up_plr=0.90,         # add a module when running modules would exceed 90 % PLR
    bypass_fan_dp=15.0,        # Pa, low-lift fans driving intentional recirculation (Demetriou & Khalifa)
    bypass_fan_eta=0.35,       # small axial fan total efficiency, assumption
    T_chw_min=6.0,
    # v2: dry-cooler heat rejection (sink = ambient DRY bulb). Assumptions, not C-DAC data:
    dry_approach=6.0,          # K, leaving water - ambient dry bulb at design (typical 5-8 K)
    dry_fan_kw_per_kw=0.025,   # kW/kW rejected at design approach (EC-fan dry coolers ~0.015-0.035)
)

def sink_temp(x):
    """Heat-rejection sink temperature for the active plant option: tower -> wet bulb, dry -> dry bulb."""
    return x["T_wb"] if x.get("sink", "tower") == "tower" else x["T_db"]

def _rej_params(x):
    if x.get("sink", "tower") == "tower":
        return PLANT["tower_approach"], PLANT["tower_fan_kw_per_kw"]
    return PLANT["dry_approach"], PLANT["dry_fan_kw_per_kw"]

# --------------------------------------------------------------------------- chiller
# Full-load COP: two-parameter lift model (Gordon-Ng family with the heat-exchanger term absorbed),
#     1/COP_full = k1 (T_cdi/T_evo - 1) + k0          (T in K; T_cdi condenser water in, T_evo CHW out)
# fitted exactly to two full-load points consistent with ASHRAE 90.1 Path A (>=300 t centrifugal):
#   AHRI 550/590 (CHW 6.67 C, ECWT 29.44 C): 0.560 kW/ton -> COP 6.28
#   elevated CHW 15.56 C (60 F), same ECWT: 0.560*(1-0.017)^16 kW/ton (rule of thumb ~1.7 %/F) -> COP 8.26
# Check (not fitted): ECWT 23.89 C, CHW 6.67 C -> COP 7.33 = AHRI IPLV 75 % point 0.480 kW/ton (7.33).
# Part load at fixed temperatures: DOE-2.1E default centrifugal EIR-fPLR curve (also EnergyPlus
# Chiller:Electric:EIR reference): EIR/EIR_full = 0.17149 + 0.58955 PLR + 0.23898 PLR^2  ->  COP = COP_full*PLR/EIRfPLR.
# (An earlier 3-parameter Gordon-Ng fit to IPLV points confounded PLR with ECWT and gave a ~33 % COP
#  drop at 50 % PLR at constant ECWT, which is unrealistic; replaced.)
_KW_PER_TON = 3.5169
_CAL = [  # (PLR, T_evo C, T_cdi C, COP) full-load fit points + one IPLV check point
    (1.00, 6.67, 29.44, _KW_PER_TON / 0.560),
    (1.00, 15.56, 29.44, _KW_PER_TON / (0.560 * (1 - 0.017) ** 16)),
    (0.75, 6.67, 23.89, _KW_PER_TON / 0.480),
]
def _fit_lift():
    (q1, e1, c1, p1), (q2, e2, c2, p2) = _CAL[0], _CAL[1]
    x1, x2 = (c1 + K) / (e1 + K) - 1, (c2 + K) / (e2 + K) - 1
    k1 = (1 / p1 - 1 / p2) / (x1 - x2)
    return np.array([1 / p1 - k1 * x1, k1])
GN = _fit_lift()                       # (k0, k1)
EIR_PLR = (0.17149, 0.58955, 0.23898)  # DOE-2.1E centrifugal

def cop_full(T_evo_C, T_cdi_C):
    return 1.0 / (GN[1] * ((T_cdi_C + K) / (T_evo_C + K) - 1) + GN[0])

def eir_fplr(plr):
    a, b, c = EIR_PLR
    return a + b * plr + c * plr ** 2

def chiller(Q_evap_W, T_chw_supply_C, T_cdi_C, Q_rated=None):
    """Electric chiller with equal-module staging. Returns W (compressor electrical), COP, Q_cond, PLR, eta_II."""
    Qu = Q_rated or PLANT["Q_rated_chiller"]
    if Q_evap_W <= 0:
        return dict(W=0.0, COP=np.inf, Q_cond=0.0, PLR=0.0, COP_carnot=np.inf, eta_II=np.nan, n_on=0)
    n_on = int(np.ceil(Q_evap_W / (PLANT["stage_up_plr"] * Qu)))
    q = max(Q_evap_W / (n_on * Qu), 0.15)               # below ~15 % PLR chillers cycle / hot-gas bypass
    cop = cop_full(T_chw_supply_C, T_cdi_C) * q / eir_fplr(q)
    T_ev_ref, T_cd_ref = T_chw_supply_C + K - 2.0, T_cdi_C + K + PLANT["cw_range"] + 2.0   # refrigerant temps, 2 K approaches
    cop_c = T_ev_ref / (T_cd_ref - T_ev_ref)
    W = Q_evap_W / cop
    return dict(W=float(W), COP=float(cop), Q_cond=float(Q_evap_W + W), PLR=float(Q_evap_W / (n_on * Qu)),
                COP_carnot=float(cop_c), eta_II=float(cop / cop_c), n_on=n_on)

def cooling_tower(Q_rej_W, T_wb_C, dp_pump=None, T_out_target=None, rng=None, a_des=None, fan_kw_per_kw=None):
    """Approach model. Returns leaving/entering water T, fan + pump electrical power.
    If a warmer leaving temperature is acceptable (economiser serving warm TCS water) the fans are
    slowed: required air flow ~ 1/approach (Merkel, fixed range), fan power ~ air flow^2 for a tower
    whose static pressure is dominated by the fill (conservative vs the cubic affinity law).
    v2: a_des / fan_kw_per_kw override the tower values so the same model serves a dry cooler
    (T_wb_C is then the ambient dry bulb)."""
    a_des = a_des or PLANT["tower_approach"]
    a = a_des if T_out_target is None else max(T_out_target - T_wb_C, a_des)
    T_out = T_wb_C + a
    rng = rng or PLANT["cw_range"]
    m = Q_rej_W / (CP_W * rng)
    W_fan = (fan_kw_per_kw or PLANT["tower_fan_kw_per_kw"]) * Q_rej_W * (a_des / a) ** 2
    W_pump = m / 996.0 * (dp_pump or PLANT["cw_pump_dp"]) / PLANT["pump_eta"]
    return dict(T_out=T_out, T_in=T_out + rng, m=m, W_fan=W_fan, W_pump=W_pump, approach=a)

# --------------------------------------------------------------------------- exergy helpers
def x_flow(m, cp, T_C):
    """Thermomechanical flow exergy rate of an incompressible / ideal-gas stream (pressure term neglected)."""
    T = T_C + K
    return m * cp * ((T - T0K) - T0K * np.log(T / T0K))

def x_heat(Q, T_C):
    return Q * (1 - T0K / (T_C + K))

# --------------------------------------------------------------------------- climate (Mumbai, IMD Santacruz 1991-2020)
MUMBAI = dict(  # Tmax, Tmean, Tmin (degC), RH at 17:30 IST (%) - IMD Santacruz 1991-2020 (Wikipedia "Mumbai" climate box)
    Jan=(31.2, 24.6, 16.9, 49), Feb=(31.7, 25.3, 18.1, 47), Mar=(33.1, 27.6, 21.1, 51), Apr=(33.4, 28.8, 24.2, 59),
    May=(33.7, 30.2, 27.0, 65), Jun=(32.5, 29.3, 26.6, 74), Jul=(30.4, 27.9, 25.5, 81), Aug=(30.2, 27.8, 25.2, 81),
    Sep=(30.9, 27.9, 24.9, 76), Oct=(33.6, 29.0, 23.9, 63), Nov=(34.1, 28.0, 21.4, 54), Dec=(32.6, 25.8, 18.4, 51))

def mumbai_wetbulb():
    """Dew point from the 17:30 RH taken at T ~ Tmax-1 (late afternoon), assumed constant through the day;
    wet bulb at daily mean T=(Tmax+Tmin)/2 and at Tmax. CoolProp HAPropsSI."""
    out = {}
    for mth, (tx, tm, tn, rh) in MUMBAI.items():
        t530 = tx - 1.0
        w = CP.HAPropsSI("W", "T", t530 + K, "P", P.P0, "R", rh / 100)
        dp = CP.HAPropsSI("D", "T", t530 + K, "P", P.P0, "W", w) - K
        wb_m = CP.HAPropsSI("B", "T", tm + K, "P", P.P0, "W", w) - K
        wb_x = CP.HAPropsSI("B", "T", tx + K, "P", P.P0, "W", w) - K
        out[mth] = dict(T_mean=tm, T_max=tx, dew=dp, wb_mean=wb_m, wb_max=wb_x)
    return out

# --------------------------------------------------------------------------- system integration
DEFAULT_X = dict(T_in=24.0, psi=0.0, beta=0.05, T_liq_in=35.0, lpm=0.5, T_chw=14.0, load=1.0,
                 T_wb=P.T_WB_C, n_irc=P.N_IRC, fluid="PG25",
                 # v2 options (defaults reproduce the October model exactly)
                 n_nodes=None, mem_load=None, tcs_config="parallel", cdu="balanced", sink="tower", T_db=None,
                 lpm_primary=None,
                 # v3: air-only reference machine and raised-floor air path
                 capture=None, dt_max_air=None,
                 air_path="inrow", bypass_route="wall",
                 dp_plenum_ref=12.0, dp_tile_ref=22.0, dp_return_ref=6.0, eta_crah=None)

# v3 raised-floor air path (CRAH -> underfloor plenum -> perforated tiles -> contained cold aisle;
# hot aisle -> ceiling/room return -> CRAH). Pressure drops are given at V_REF_AIR, the server air flow of
# the C-DAC pod at 24 degC inlet and full load, and scale with flow squared:
#   plenum distribution (incl. underfloor piping obstruction) 12 Pa  -> scales with CRAH flow
#   perforated 25 %-open tiles, 24 tiles at ~0.4 m3/s each   22 Pa  -> scales with TILE flow
#   return path to the CRAH top                               6 Pa  -> scales with CRAH flow
# Typical values: 25 % tile ~7.5 Pa at 500 CFM (vendor curves, ~V^2); plenum static 5-25 Pa (ASHRAE TC9.9).
# Assumptions, varied in sim_airpath.py.
V_REF_AIR = None   # set lazily from the C-DAC baseline

def cdac_x(**over):
    """Design vector for the C-DAC IIT Bombay installation: 619 nodes, 0.5 L/min per node with the two
    cold plates in series (assumption), 37 degC secondary supply, CHx750 eps-NTU CDU at 400 L/min primary."""
    c = P.CDAC
    return {**DEFAULT_X, "n_nodes": c["n_nodes"], "tcs_config": "series", "lpm": c["lpm_node"],
            "T_liq_in": c["T_sec_supply"], "cdu": "cdac", "lpm_primary": c["lpm_primary"], **over}

def system(x=None, detail=True):
    """Coupled steady-state solution of the pod + plant for design vector x (dict, see DEFAULT_X).
    Order: S2 node/pod air at T_in -> S3 aisle (required T_c, T_ret, m_c) -> S4 coil (CHW flow)
           -> S1 TCS loop -> CDU / economiser -> chillers -> towers -> energy & exergy ledgers."""
    x = {**DEFAULT_X, **(x or {})}
    N = x["n_nodes"] or P.NODES_PER_POD
    if x["sink"] != "tower" and x["T_db"] is None:
        raise ValueError("dry-cooler sink needs T_db")
    a_rej, fan_rej = _rej_params(x)
    T_sink = sink_temp(x)
    r = dict(x=x)
    s2.DT_MAX_AIR = x["dt_max_air"]
    # ---- S2: node heat split and server air
    pa = s2.pod_air(x["T_in"], x["load"], n_nodes=N, mem_load=x["mem_load"], capture=x["capture"])
    m_s, Q_air = pa["m_dot_servers"], pa["Q_air_pod"]
    Q_liq = pa["Q_liquid_pod"]
    P_it_wall = pa["Q_it_pod"]                       # all node wall power incl. server fans
    W_srv_fans = pa["W_fans_pod"]
    # ---- S3: aisle mixing -> required cooler supply
    ai = s3.solve_for_inlet(m_s, Q_air, x["T_in"], x["psi"], x["beta"], T_chw_in=x["T_chw"])
    T_c, T_ret, m_c, T_h = ai["T_c"], ai["T_ret"], ai["m_c"], ai["T_h"]
    m_r = ai["m_r"]
    # ---- v3 air path. Tile flow: recirculation through the plenum also leaves through the tiles.
    raised = x["air_path"] == "raised_floor"
    W_ext = 0.0; dp_tile = 0.0; dp_crah_ext = 0.0
    if raised:
        Vref = _v_ref_air()
        V_c = m_c / P.rho_air(T_c); V_s = m_s / P.rho_air(x["T_in"])
        V_tile = V_s if x["bypass_route"] == "plenum" else V_s * (1 - x["psi"])
        dp_tile = x["dp_tile_ref"] * (V_tile / Vref) ** 2
        dp_crah_ext = (x["dp_plenum_ref"] + x["dp_return_ref"]) * (V_c / Vref) ** 2 + dp_tile
        eta_c = x["eta_crah"] or s4.calibrate()["eta_fan"]
        W_ext = V_c * dp_crah_ext / eta_c
    dp_bp = PLANT["bypass_fan_dp"] + (dp_tile if (raised and x["bypass_route"] == "plenum") else 0.0)
    W_bp = m_r / P.rho_air(T_h) * dp_bp / PLANT["bypass_fan_eta"]
    # ---- S4: in-row coolers (equal share)
    n = x["n_irc"]
    co = s4.solve_chw(m_c / n, T_ret, T_c, x["T_chw"])
    feas_coil = bool(co.get("feasible", 0) >= 0.5) and np.isfinite(co.get("m_w", np.nan))
    if feas_coil:
        m_w = co["m_w"] * n
        W_irc_fans = co["W_fan"] * n + W_ext          # v3: + raised-floor external pressure (0 for in-row)
        W_chw_pump = co["W_pump_chw"] * n
        # Energy-consistent coil duty from the aisle balance (S2/S3 use dry-air cp 1006; S4 uses moist-air
        # cp_m ~1022 J/kg/K -> its Q is ~1.6 % high). Keep S4's leaving-water T, rescale the water flow.
        Q_coil = Q_air + W_irc_fans + W_bp
        cp_ratio = Q_coil / (co["Q"] * n)
        m_w = m_w * cp_ratio
        T_w_out = co["T_w_out"]
        dry = bool(co.get("dry", 1) >= 0.5)
        X_coil_mod = co["X_dest_coil"] * n
    else:
        m_w = W_irc_fans = W_chw_pump = Q_coil = X_coil_mod = np.nan
        T_w_out = np.nan; dry = False
    # ---- S1: TCS loop
    m_plate = s1.lpm_to_kg_s(x["lpm"], x["T_liq_in"], x["fluid"])
    air_only = (x["capture"] is not None and x["capture"] < 0.01)
    Q_liq_loop = Q_liq                               # v3: air-only reference uses capture 0.003 (1 W/node token)
    lq = s1.liquid_loop(Q_liq_loop / N, m_plate, x["T_liq_in"], N, fluid=x["fluid"], config=x["tcs_config"])
    W_tcs_pump = 0.0 if air_only else lq["W_pump"]   # air-only machine has no TCS loop
    Q_tcs = Q_liq_loop + W_tcs_pump                       # pump work ends up as heat in the TCS fluid
    cp_l = s1.coolant_props(0.5 * (x["T_liq_in"] + lq["T_return"]), x["fluid"])["cp"]
    m_l = lq["m_dot_total"]
    T_tcs_ret = x["T_liq_in"] + Q_tcs / (m_l * cp_l)
    # ---- CDU + economiser
    if x["cdu"] == "cdac":
        # v2: counterflow eps-NTU with the CHx750 UA and a FIXED primary flow (C-DAC: 400 L/min).
        # Required facility supply so the secondary leaves at T_liq_in:  Q = eps C_min (T_tcs_ret - T_fws)
        m_fw = x["lpm_primary"] / 60000.0 * 994.0
        C_h, C_c = m_l * cp_l, m_fw * CP_W
        Cmin, Cmax = min(C_h, C_c), max(C_h, C_c); Cr = Cmin / Cmax
        NTU = P.CDAC["UA_cdu"] / Cmin
        eps_cdu = (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr)))
        T_fws = T_tcs_ret - Q_tcs / (eps_cdu * Cmin)
        T_fwr = T_fws + Q_tcs / C_c
        r.update(eps_cdu=float(eps_cdu), NTU_cdu=float(NTU))
    else:
        T_fws = x["T_liq_in"] - Q_tcs / PLANT["UA_cdu"]  # balanced counterflow: approach = Q/UA
        T_fwr = T_tcs_ret - Q_tcs / PLANT["UA_cdu"]
        m_fw = Q_tcs / (CP_W * (T_fwr - T_fws))
    T_tw = T_sink + a_rej
    f_econ = float(np.clip((T_fwr - T_tw) / (T_fwr - T_fws), 0.0, 1.0))
    Q_econ = f_econ * Q_tcs
    Q_trim = Q_tcs - Q_econ
    T_trim_in = T_fws if f_econ >= 1 else min(T_tw, T_fwr)   # full econ: tower modulated to deliver T_fws
    # ---- chillers
    T_cdi = T_tw
    if feas_coil:
        Q_ev_main = Q_coil + W_chw_pump
        chm = chiller(Q_ev_main, x["T_chw"], T_cdi)
    else:
        Q_ev_main = np.nan; chm = dict(W=np.nan, COP=np.nan, Q_cond=np.nan, PLR=np.nan, COP_carnot=np.nan, eta_II=np.nan)
    cht = chiller(Q_trim, T_fws, T_cdi, Q_rated=PLANT["Q_rated_chiller"]) if Q_trim > 1 else \
        dict(W=0.0, COP=np.inf, Q_cond=0.0, PLR=0.0, COP_carnot=np.inf, eta_II=np.nan, n_on=0)
    # ---- towers: condenser heat of both chillers + economiser heat
    Q_rej_cond = (chm["Q_cond"] if feas_coil else np.nan) + cht["Q_cond"]
    tw_c = cooling_tower(Q_rej_cond, T_sink, a_des=a_rej, fan_kw_per_kw=fan_rej)
    tw_e = cooling_tower(Q_econ, T_sink, dp_pump=PLANT["fw_pump_dp"],
                         T_out_target=(T_fws if f_econ >= 1 else None),
                         rng=max(T_fwr - (T_fws if f_econ >= 1 else T_tw), 0.5),
                         a_des=a_rej, fan_kw_per_kw=fan_rej) if Q_econ > 0 else \
        dict(T_out=T_tw, T_in=T_tw, m=0.0, W_fan=0.0, W_pump=0.0)
    # ---- power ledger
    W = dict(server_fans=W_srv_fans, irc_fans=W_irc_fans, bypass_fans=W_bp, tcs_pump=W_tcs_pump,
             chw_pump=W_chw_pump, chiller=chm["W"], trim_chiller=cht["W"],
             tower_fans=tw_c["W_fan"] + tw_e["W_fan"], cw_fw_pumps=tw_c["W_pump"] + tw_e["W_pump"])
    W_cool_fac = sum(v for k, v in W.items() if k != "server_fans")
    W_cool_all = W_cool_fac + W_srv_fans
    P_compute = P_it_wall - W_srv_fans
    r.update(
        W_ext_airpath=W_ext, dp_crah_ext=dp_crah_ext, dp_tile=dp_tile,
        m_s=m_s, Q_air=Q_air, Q_liq=Q_liq, P_it_wall=P_it_wall, P_compute=P_compute, T_h=T_h,
        T_c=T_c, T_ret=T_ret, m_c=m_c, m_r=m_r, SHI=ai["SHI"], RHI=ai["RHI"], RTI=ai["RTI"],
        m_w=m_w, T_w_out=T_w_out, Q_coil=Q_coil, irc_speed=co.get("speed_frac", np.nan), dry=dry,
        T_tcs_ret=T_tcs_ret, T_case_max=lq["T_case_max"], T_j_max=lq["T_j_max"],
        T_fws=T_fws, T_fwr=T_fwr, f_econ=f_econ, Q_econ=Q_econ, Q_trim=Q_trim,
        COP_chiller=chm["COP"], PLR=chm["PLR"], eta_II_chiller=chm["eta_II"], COP_trim=cht["COP"],
        W=W, W_cool_facility=W_cool_fac, W_cool_total=W_cool_all,
        COP_cool=P_compute / W_cool_all,                # compute heat removed per unit of ALL cooling work
        COP_facility=P_it_wall / W_cool_fac,            # server fans counted as IT (PUE convention)
        pPUE=(P_it_wall + W_cool_fac) / P_it_wall,
        feasible=bool(feas_coil and dry and pa["feasible"] >= 0.5 and lq["feasible"] >= 0.5
                      and x["T_chw"] >= PLANT["T_chw_min"] and T_fws > 5.0),
        flags=dict(coil=feas_coil, dry=dry, server=bool(pa["feasible"] >= 0.5), liquid=bool(lq["feasible"] >= 0.5)),
    )
    if not detail or not feas_coil:
        return r
    # ---- exergy ledger: each box closed by its stream balance (Gouy-Stodola: X_dest = sum X_in - sum X_out)
    ca = P.CP_AIR
    Xa = lambda m, T: x_flow(m, ca, T)
    Xl = lambda m, T: x_flow(m, cp_l, T)
    Xw = lambda m, T: x_flow(m, CP_W, T)
    T_chw_ret = x["T_chw"] + Q_ev_main / (m_w * CP_W)
    X = {}
    # servers + TCS pump/piping box: electricity in, air and TCS liquid streams through
    X_servers_box = (P_it_wall + W_tcs_pump + Xa(m_s, x["T_in"]) - Xa(m_s, T_h)
                     + Xl(m_l, x["T_liq_in"]) - Xl(m_l, T_tcs_ret))   # v3: TCS supply = set-point (was the balanced-CDU identity)
    X["server: cold plates (S1)"] = lq["X_dest_coldplates_heat"]
    X["server: TCS pumping/friction (S1)"] = W_tcs_pump
    X["server: chassis air heat transfer (S2)"] = pa["X_dest_chassis_ht"]
    X["server: fans (S2)"] = W_srv_fans
    X["server: electronics, electricity to heat"] = (X_servers_box - X["server: cold plates (S1)"]
        - X["server: TCS pumping/friction (S1)"] - X["server: chassis air heat transfer (S2)"] - X["server: fans (S2)"])
    # aisle box (mixing + bypass fans)
    X_aisle = (Xa(m_c, T_c) + Xa(m_s, T_h) - Xa(m_s, x["T_in"]) - Xa(m_c, T_ret)) + W_bp
    X["aisle mixing + bypass fans (S3)"] = X_aisle
    # IRC box (fans + coil)
    X["in-row coolers: coil + fans (S4)"] = (W_irc_fans + Xa(m_c, T_ret) - Xa(m_c, T_c)
                                            + Xw(m_w, x["T_chw"]) - Xw(m_w, T_w_out))
    # CHW distribution (pump work -> heat; mixing of unit returns is trivial: equal units)
    X["CHW pumping"] = W_chw_pump + Xw(m_w, T_w_out) - Xw(m_w, T_chw_ret)
    # CDU box
    X["CDU plate HX"] = (Xl(m_l, T_tcs_ret) - Xl(m_l, x["T_liq_in"])
                        + Xw(m_fw, T_fws) - Xw(m_fw, T_fwr))
    # TCS stream: pump sits between CDU outlet and servers; temperature rise from pump heat inside server box
    # main chiller
    m_cw = tw_c["m"]
    m_cw_main = chm["Q_cond"] / (CP_W * PLANT["cw_range"])
    X["chiller (main, CHW)"] = (chm["W"] + Xw(m_w, T_chw_ret) - Xw(m_w, x["T_chw"])
                                + Xw(m_cw_main, T_cdi) - Xw(m_cw_main, T_cdi + PLANT["cw_range"]))
    # trim chiller + economiser HX on the facility-water side
    m_fw_trim = Q_trim / (CP_W * max(T_trim_in - T_fws, 1e-9)) if Q_trim > 1 else 0.0
    m_cw_trim = cht["Q_cond"] / (CP_W * PLANT["cw_range"])
    X["trim chiller (TCS)"] = (cht["W"] + Xw(m_fw_trim, T_trim_in) - Xw(m_fw_trim, T_fws)
                               + Xw(m_cw_trim, T_cdi) - Xw(m_cw_trim, T_cdi + PLANT["cw_range"])) if Q_trim > 1 else 0.0
    # economiser tower path: facility water T_fwr -> T_trim_in (tower), then trim to T_fws
    X["cooling towers + CW/FW pumps"] = (W["tower_fans"] + W["cw_fw_pumps"]
        + Xw(m_cw, T_cdi + PLANT["cw_range"]) - Xw(m_cw, T_cdi)
        + Xw(m_fw, T_fwr) - Xw(m_fw, T_trim_in))
    X_chem = -(Xw(m_cw, T_cdi) if T_cdi < T0 else 0.0) - (Xw(m_fw, T_trim_in) if T_trim_in < T0 else 0.0)   # exhaust air leaves at dead state (chemical exergy of humid air not tracked)
    X_total = sum(X.values())
    X_in = P_it_wall + W_cool_fac
    r.update(X=X, X_dest_total=X_total, X_in=X_in, X_closure=X_in - X_total,
             X_cross=dict(aisle_S3=ai["X_dest_mix"] + W_bp,
                          coil_S4=n * (co["X_dest_heat"] + co["X_dest_fric_air"] + co["X_dest_fan_system"]),
                          chassis_S2=pa["X_dest_chassis_ht"]),
             X_tcs_return=Xl(m_l, T_tcs_ret), X_heat_IT_liq=x_heat(Q_liq, lq["T_case_max"]),
             E_closure=(P_it_wall + W_cool_fac) - (tw_c["m"] * CP_W * PLANT["cw_range"] + Q_econ
                                                   + W["tower_fans"] + W["cw_fw_pumps"]),
             n_chillers=chm["n_on"],
             T_chw_ret=T_chw_ret, m_cw=m_cw, m_fw=m_fw)
    return r

def _v_ref_air():
    global V_REF_AIR
    if V_REF_AIR is None:
        save = s2.DT_MAX_AIR; s2.DT_MAX_AIR = None
        pa = s2.pod_air(24.0, 1.0, n_nodes=P.CDAC["n_nodes"])
        s2.DT_MAX_AIR = save
        V_REF_AIR = pa["m_dot_servers"] / P.rho_air(24.0)
    return V_REF_AIR

# --------------------------------------------------------------------------- optimisation
def objective(v, fixed, key="W_cool_total"):
    x = {**fixed, "T_in": v[0], "psi": v[1], "T_chw": v[2]}
    try:
        r = system(x, detail=False)
    except Exception:
        return 1e9
    if not r["feasible"]:
        return 1e9
    return r[key]

def optimise_air(fixed=None, bounds=((18, 27), (0.0, 0.40), (6.0, 20.0)), seed=1, key="W_cool_total"):
    fixed = {**DEFAULT_X, **(fixed or {})}
    res = differential_evolution(objective, bounds, args=(fixed, key), seed=seed, tol=1e-7, maxiter=60,
                                 popsize=12, polish=False, init="sobol")
    return res

def optimise_liquid(fixed=None):
    """TCS side decouples from the air side (only Q_liq enters it): grid over supply T and flow."""
    fixed = {**DEFAULT_X, **(fixed or {})}
    best = None
    for T in np.arange(30.0, 45.01, 0.5):
        for l in np.arange(0.25, 1.001, 0.05):
            m = s1.lpm_to_kg_s(l, T)
            lq = s1.liquid_loop(s2.pod_air(fixed["T_in"], fixed["load"])["Q_liquid_pod"] / P.NODES_PER_POD, m, T, P.NODES_PER_POD)
            if lq["feasible"] < 0.5:
                continue
            r = system({**fixed, "T_liq_in": T, "lpm": l}, detail=False)
            w = r["W"]["tcs_pump"] + r["W"]["trim_chiller"] + r["W"]["cw_fw_pumps"] + r["W"]["tower_fans"]
            if best is None or w < best[0]:
                best = (w, T, l)
    return best


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    print("lift model k0, k1:", GN)
    for q, te, tc, cop in _CAL:
        m = cop_full(te, tc) * q / eir_fplr(q)
        print(f"  calib PLR {q:.2f} CHW {te:5.2f} ECWT {tc:5.2f}: model COP {m:.3f} vs target {cop:.3f} ({100*(m/cop-1):+.1f} %)")
    for q in (0.3, 0.5, 0.7, 0.9, 1.0):
        print(f"  COP @ CHW 16 / ECWT 29, PLR {q}: {cop_full(16, 29) * q / eir_fplr(q):.2f}")
    t = time.time(); base = system(); print(f"system() {time.time()-t:.2f}s")
    print(json.dumps({k: (v if not isinstance(v, (np.floating,)) else float(v)) for k, v in base.items()
                      if k not in ("X", "W", "x", "flags", "X_cross")}, indent=1, default=float))
    print("W:", {k: round(v, 1) for k, v in base["W"].items()})
    print("X:", {k: round(v, 1) for k, v in base["X"].items()})
    print("X cross:", base["X_cross"])
