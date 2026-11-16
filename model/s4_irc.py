"""
s4_irc.py -- Student 4: in-row cooler (IRC) chilled-water coil, fans, CHW flow optimisation,
             rear-door heat-exchanger (RDHX) comparison and coil exergy destruction
=============================================================================================
PARAM Rudra hybrid-cooled enclosed-aisle pod (576 nodes, params.N_IRC = 4 in-row coolers).

Reference product (rating data used for calibration / validation):
  Schneider Electric / APC Uniflair InRow RC ACRC301S, 300 mm wide, chilled water, 8 x 200 mm
  mixed-flow direct-drive DC fans, 1510 L/s (3200 CFM) nominal air flow, 805 W input at max
  operating condition, 3/8" Cu tube / raised-lance Al fin counterflow coil.
    - guide spec  : https://www.criticalpowersupplies.co.uk/downloads/apc/1636405034-8405-download3.pdf
    - performance : https://objects.eanixter.com/PD425008.PDF  (Technical Specifications, InRow CW)
    - product page: https://www.se.com/us/en/product/ACRC301S ;
                    https://www.mouser.lt/datasheet/3/187/14/ACRC301S_document.pdf

Model chain (per IRC unit)
  geometry  -> air-side area A_a, fin area A_f, min. free-flow area A_min, water-side area A_w
  air side  -> Wang, Chi & Chang (2000) plain-fin j & f  (IJHMT 43:2693-2700), x C_CAL (calibrated)
  fins      -> Schmidt (1949) equivalent-circular-fin (sector) efficiency, staggered tubes
  water side-> Gnielinski (1976) turbulent / Nu=3.66 laminar, linear blend 2300<Re<3000
  UA        -> 1/UA = 1/(eta_o h_a A_a) + R_wall + R_foul + 1/(h_w A_w)
  eps-NTU   -> N-pass cross-counterflow (each row a crossflow pass, passes in counterflow)
  wet coil  -> Braun-Mitchell-Klein (1989) enthalpy effectiveness; "max(dry, wet)" for partly wet
  fans      -> dP = coil (Wang f) + filter + casing;  W = V dP / eta_tot;  30 % min speed
  CHW       -> solve_chw: required water flow for a target supply-air temperature (brentq)
  RDHX      -> same coil engine with a door geometry; passive vs active (fan matrix) parasitics
  exergy    -> Gouy-Stodola: X = T0 * S_gen(heat) + T0 * V dP / T (friction), both streams

Public API (INTERFACE.md):
  coil(m_air, T_air_in_C, m_w, T_w_in_C, ...)            -> dict
  irc_unit(m_air, T_air_in_C, m_w, T_w_in_C, ...)        -> dict (+ W_fan, W_pump_chw)
  solve_chw(m_air, T_air_in_C, T_air_out_target_C, T_w_in_C, ...) -> dict (m_w or infeasible)
  irc_fan(m_air, T_air_C)                                -> dict (W_fan, dP breakdown)
  pod_irc(m_air_pod, T_ret_C, T_supply_C, T_w_in_C)      -> dict (pod totals over N_IRC units)
  rdhx(m_air_rack, T_air_in_C, m_w, T_w_in_C, ...)        -> dict (passive/active parasitics)
  X_dest_coil(...)                                       -> dict (exergy split)
All temperatures in degC unless suffixed _K; SI units; all returns are plain dicts of floats.
"""
import json
import os
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
import CoolProp.CoolProp as CP

import params as P

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
K0 = 273.15
CPA = P.CP_AIR                     # J/kg/K dry air (params)
CPV = 1860.0                       # J/kg/K water vapour (for moist-air cp_m = cpa + W cpv)

# =================================================================================== geometry
INCH = 0.0254
IRC_GEOM = dict(
    name="IRC 300 mm (ACRC301S-like)",
    H=1.80,            # m  coil face height (unit is 1991 mm tall; brief: ~1.8 m)
    W=0.25,            # m  coil face width = finned tube length (300 mm cabinet minus casing/sheet)
    rows=4,            # tube rows in air-flow direction (assumed; 4-6 typical for CW IRC coils)
    Do=3 / 8 * INCH,   # m  tube OD 9.525 mm -- ACRC301S guide spec: 9.5 mm (3/8") Cu tube
    t_wall=0.32e-3,    # m  tube wall (typical light-wall 3/8" HVAC coil tube) [assumption]
    Pt=1.00 * INCH,    # m  transverse pitch 25.4 mm  (standard 1" x 0.866" 3/8" coil pattern)
    Pl=0.866 * INCH,   # m  longitudinal pitch 22.0 mm (staggered, equilateral)
    FPI=14.0,          # fins per inch -> Fp = 1.814 mm (brief 12-14 FPI) [assumption]
    t_fin=0.115e-3,    # m  Al fin thickness [assumption, typical 0.10-0.13 mm]
    k_fin=200.0,       # W/m/K aluminium fin (alloy 1100/8011 ~ 200-220)
    k_tube=390.0,      # W/m/K copper
    n_circ=14,         # water circuits in parallel (sets tube velocity ~1.3 m/s at rated flow)
    K_bend=1.0,        # loss coefficient per 180-deg return bend [assumption]
    fin="raised-lance (modelled with plain-fin Wang 2000 + calibration)",
)

# Rear-door heat exchanger for a 600 mm x 42U rack (comparison variant).  Low fin density and only
# 2 rows to keep the air pressure drop within what server fans can supply.
RDHX_GEOM = dict(IRC_GEOM, name="RDHX passive door 600 mm rack", H=1.90, W=0.55, rows=2,
                 FPI=10.0, n_circ=12)
# Denser door (4 rows, 14 FPI) -- representative of high-capacity (35-55 kW) doors; sensitivity only
RDHX_DENSE_GEOM = dict(RDHX_GEOM, name="RDHX dense door (4 rows, 14 FPI)", rows=4, FPI=14.0,
                       n_circ=24)


def geometry(g=None):
    """Derived coil areas for a staggered fin-and-tube coil.

    Equations (ACHP fin-tube documentation, achp.sourceforge.net FinTubeHX):
      N_fin   = W / Fp
      A_fin   = 2 N_fin (H D_depth - N_tubes pi Dc^2/4)          (both fin faces, holes removed)
      A_tube  = N_tubes pi Dc (W - N_fin t)                       (exposed tube between fins)
      A_a     = A_fin + A_tube
      sigma   = (Pt - Dc)(Fp - t) / (Pt Fp)                       (min free flow / face area)
      Dh      = 4 A_min D_depth / A_a                             (Wang 2000 hydraulic diameter)
      A_w     = N_tubes pi Di W                                   (water side, bends ignored)
    Dc = Do + 2 t_fin (fin-collar outside diameter, Wang et al. definition).
    """
    g = dict(IRC_GEOM if g is None else g)
    Fp = INCH / g["FPI"]
    Dc = g["Do"] + 2 * g["t_fin"]
    Di = g["Do"] - 2 * g["t_wall"]
    n_tr = int(np.floor(g["H"] / g["Pt"]))            # tubes per row
    n_t = n_tr * g["rows"]
    depth = g["rows"] * g["Pl"]
    N_fin = g["W"] / Fp
    A_face = g["H"] * g["W"]
    A_fin = 2 * N_fin * (g["H"] * depth - n_t * np.pi * Dc ** 2 / 4)
    A_tube = n_t * np.pi * Dc * (g["W"] - N_fin * g["t_fin"])
    A_a = A_fin + A_tube
    sigma = (g["Pt"] - Dc) * (Fp - g["t_fin"]) / (g["Pt"] * Fp)
    A_min = sigma * A_face
    Dh = 4 * A_min * depth / A_a
    A_w = n_t * np.pi * Di * g["W"]
    passes = n_t / g["n_circ"]                         # tube passes per circuit
    L_circ = passes * g["W"]
    g.update(Fp=Fp, Dc=Dc, Di=Di, n_tr=n_tr, n_t=n_t, depth=depth, N_fin=N_fin, A_face=A_face,
             A_fin=A_fin, A_tube=A_tube, A_a=A_a, sigma=sigma, A_min=A_min, Dh=Dh, A_w=A_w,
             passes=passes, L_circ=L_circ, alpha_a=A_a / (A_face * depth))
    return g


# =================================================================================== properties
@lru_cache(maxsize=4096)
def _air(Tr):
    """Air mu [Pa s], k [W/m/K], Pr at T (degC, rounded key) via CoolProp 'Air' (dry)."""
    T = Tr + K0
    mu = CP.PropsSI("V", "T", T, "P", P.P0, "Air")
    k = CP.PropsSI("L", "T", T, "P", P.P0, "Air")
    return mu, k, mu * CPA / k


@lru_cache(maxsize=4096)
def _water(Tr):
    """Liquid water rho, cp, mu, k, Pr at T (degC, rounded key) via CoolProp 'Water'."""
    T = Tr + K0
    rho, cp, mu, k = (CP.PropsSI(q, "T", T, "P", 3e5, "Water") for q in ("D", "C", "V", "L"))
    return rho, cp, mu, k, mu * cp / k


def air_props(T_C):
    mu, k, Pr = _air(round(float(T_C), 1))
    return dict(rho=P.rho_air(T_C), mu=mu, k=k, Pr=Pr)


def water_props(T_C):
    rho, cp, mu, k, Pr = _water(round(float(T_C), 1))
    return dict(rho=rho, cp=cp, mu=mu, k=k, Pr=Pr)


@lru_cache(maxsize=8192)
def _h_sat(Tr):
    """Saturated moist-air enthalpy [J/kg dry air] at T (degC) -- CoolProp HAPropsSI."""
    return CP.HAPropsSI("H", "T", Tr + K0, "P", P.P0, "R", 1.0)


def h_sat(T_C):
    # linear interpolation on a 0.05 K cached grid (keeps calls cheap and smooth)
    a = np.floor(T_C / 0.05) * 0.05
    f = (T_C - a) / 0.05
    return (1 - f) * _h_sat(round(a, 2)) + f * _h_sat(round(a + 0.05, 2))


def T_sat_from_h(h):
    return brentq(lambda T: h_sat(T) - h, -10.0, 60.0, xtol=1e-4)


def W_from_dewpoint(T_dp_C):
    """Humidity ratio [kg/kg] from dew point (CoolProp)."""
    return CP.HAPropsSI("W", "T", T_dp_C + 0.5 + K0, "P", P.P0, "D", T_dp_C + K0)


def h_moist(T_C, W):
    """Moist-air enthalpy per kg dry air, ASHRAE Fundamentals ch.1: h = 1006 T + W(2.501e6 + 1860 T)."""
    return CPA * T_C + W * (2.501e6 + CPV * T_C)


# =================================================================================== correlations
def wang2000_plain(Re, g):
    """Colburn j and Fanning-type f for plain fin-and-tube coils, N >= 2 rows.

    Wang, C.-C., Chi, K.-Y., Chang, C.-J. (2000) "Heat transfer and friction characteristics of
    plain fin-and-tube heat exchangers, part II: Correlation", IJHMT 43:2693-2700.
    (equation set as reproduced in ENCIT 2008 paper 11_5312, abcm.org.br, Eqs. 22-30)
      j = 0.086 Re^P3 N^P4 (Fp/Dc)^P5 (Fp/Dh)^P6 (Fp/Pt)^-0.93
      P3 = -0.361 - 0.042 N/ln Re + 0.158 ln[N (Fp/Dc)^0.41]
      P4 = -1.224 - 0.076 (Pl/Dh)^1.42 / ln Re
      P5 = -0.083 + 0.058 N/ln Re
      P6 = -5.735 + 1.21 ln(Re/N)
      f  = 0.0267 Re^F1 (Pt/Pl)^F2 (Fp/Dc)^F3
      F1 = -0.764 + 0.739 Pt/Pl + 0.177 Fp/Dc - 0.00758/N
      F2 = -15.689 + 64.021/ln Re
      F3 = 1.696 - 15.695/ln Re
    NOTE: the third P3 term (+0.158 ln[N(Fp/Dc)^0.41]) is in the original paper but is dropped in
    the ENCIT text extraction we opened; we keep it (original form).  Re = Re_Dc (collar dia.).
    Validity (paper): 300 < Re_Dc < 20000; any bias is absorbed into C_CAL.
    """
    N = g["rows"]
    Fp, Dc, Dh, Pt, Pl = g["Fp"], g["Dc"], g["Dh"], g["Pt"], g["Pl"]
    lnR = np.log(Re)
    P3 = -0.361 - 0.042 * N / lnR + 0.158 * np.log(N * (Fp / Dc) ** 0.41)
    P4 = -1.224 - 0.076 * (Pl / Dh) ** 1.42 / lnR
    P5 = -0.083 + 0.058 * N / lnR
    P6 = -5.735 + 1.21 * np.log(Re / N)
    j = 0.086 * Re ** P3 * N ** P4 * (Fp / Dc) ** P5 * (Fp / Dh) ** P6 * (Fp / Pt) ** -0.93
    F1 = -0.764 + 0.739 * Pt / Pl + 0.177 * Fp / Dc - 0.00758 / N
    F2 = -15.689 + 64.021 / lnR
    F3 = 1.696 - 15.695 / lnR
    f = 0.0267 * Re ** F1 * (Pt / Pl) ** F2 * (Fp / Dc) ** F3
    return j, f


def schmidt_fin_eff(h, g):
    """Schmidt (1949) equivalent circular fin, staggered layout (as used by Wang et al.; ACHP docs):
      r = Dc/2 ; X_M = Pt/2 ; X_L = sqrt((Pt/2)^2 + Pl^2)/2
      Re/r = 1.27 (X_M/r) (X_L/X_M - 0.3)^0.5
      phi  = (Re/r - 1)(1 + 0.35 ln(Re/r))
      m    = sqrt(2 h / (k_f t))           eta_f = tanh(m r phi)/(m r phi)
    """
    r = g["Dc"] / 2
    XM = g["Pt"] / 2
    XL = np.sqrt((g["Pt"] / 2) ** 2 + g["Pl"] ** 2) / 2
    Rr = 1.27 * XM / r * np.sqrt(XL / XM - 0.3)
    phi = (Rr - 1) * (1 + 0.35 * np.log(Rr))
    m = np.sqrt(2 * h / (g["k_fin"] * g["t_fin"]))
    x = m * r * phi
    return float(np.tanh(x) / x)


def water_side(m_w, T_w_C, g):
    """Tube-side h_w [W/m2K], Darcy f, velocity, Re and coil-tube dP [Pa] for one unit.

    Nu: laminar fully developed, constant T_wall: Nu = 3.66 (Incropera); turbulent: Gnielinski
      Nu = (f/8)(Re-1000)Pr / (1 + 12.7 (f/8)^0.5 (Pr^(2/3)-1)), 3000<=Re<=5e6, 0.5<=Pr<=2000
      (tacodoc.readthedocs.io/en/latest/Nusselt.html); f = (0.79 ln Re - 1.64)^-2 (Petukhov).
      2300 < Re < 3000 linearly blended (Gnielinski 1995 intermittency approach).
    dP_coil = f (L_circ/Di) rho v^2/2 + n_bends K_bend rho v^2/2   (Darcy-Weisbach + minor losses)
    """
    wp = water_props(T_w_C)
    A1 = np.pi * g["Di"] ** 2 / 4
    v = m_w / (wp["rho"] * A1 * g["n_circ"])
    Re = wp["rho"] * v * g["Di"] / wp["mu"]
    Pr = wp["Pr"]

    def turb(R):
        f = (0.79 * np.log(R) - 1.64) ** -2
        return f, (f / 8) * (R - 1000) * Pr / (1 + 12.7 * np.sqrt(f / 8) * (Pr ** (2 / 3) - 1))

    if Re < 2300:
        f, Nu = 64 / max(Re, 1e-6), 3.66
    elif Re > 3000:
        f, Nu = turb(Re)
    else:
        w = (Re - 2300) / 700
        f3, Nu3 = turb(3000.0)
        f, Nu = (1 - w) * 64 / 2300 + w * f3, (1 - w) * 3.66 + w * Nu3
    h = Nu * wp["k"] / g["Di"]
    q = wp["rho"] * v ** 2 / 2
    dP = f * g["L_circ"] / g["Di"] * q + (g["passes"] - 1) * g["K_bend"] * q
    return dict(h_w=h, f_w=f, v_w=v, Re_w=Re, Nu_w=Nu, dP_coil_w=dP, rho_w=wp["rho"], cp_w=wp["cp"])


# =================================================================================== eps-NTU
def eps_counterflow(NTU, Cr):
    """Counterflow (Incropera Table 11.3): eps = (1-e)/(1-Cr e), e = exp(-NTU(1-Cr)); Cr=1: NTU/(1+NTU)."""
    if abs(1 - Cr) < 1e-9:
        return NTU / (1 + NTU)
    e = np.exp(-NTU * (1 - Cr))
    return (1 - e) / (1 - Cr * e)


def eps_npass(NTU, Cr, N, water_is_min):
    """N crossflow passes connected in overall counterflow (Incropera Eq. 11.31a/Kays & London):
      eps = [((1-eps_p Cr)/(1-eps_p))^N - 1] / [((1-eps_p Cr)/(1-eps_p))^N - Cr]
    per pass NTU_p = NTU/N; each pass = one tube row, water (inside one row) treated as mixed,
    air (between fins) unmixed (Incropera Table 11.3):
      air = Cmin (unmixed): eps_p = (1/Cr)(1 - exp(-Cr(1 - exp(-NTU_p))))
      water = Cmin (mixed): eps_p = 1 - exp(-(1/Cr)(1 - exp(-Cr NTU_p)))
    For N >= 4 this is within ~1 % of pure counterflow (see summary), justifying the common
    'counterflow' approximation for multi-row CW coils."""
    if Cr < 1e-9:
        return 1 - np.exp(-NTU)
    Np = NTU / N
    if water_is_min:
        ep = 1 - np.exp(-(1 / Cr) * (1 - np.exp(-Cr * Np)))
    else:
        ep = (1 / Cr) * (1 - np.exp(-Cr * (1 - np.exp(-Np))))
    if abs(1 - Cr) < 1e-9:
        return N * ep / (1 + (N - 1) * ep)
    a = ((1 - ep * Cr) / (1 - ep)) ** N
    return (a - 1) / (a - Cr)


def _eps(NTU, Cr, g, arrangement, water_is_min):
    if arrangement == "counterflow":
        return eps_counterflow(NTU, Cr)
    return eps_npass(NTU, Cr, g["rows"], water_is_min)


# =================================================================================== calibration
# Datasheet rating points, ACRC301S (objects.eanixter.com/PD425008.PDF).  Air flow 1510 L/s
# (3200 CFM, nominal; ratings assumed at 100 % fan speed).  (EWT, DB, WB, dT_w, total net kW,
# sensible net kW, CW flow L/s, total CW dP kPa)
DATASHEET = [
    (7.2, 37.78, 20.7, 6.6, 30.4, 30.1, 1.13, 65.0),   # <- calibration point
    (7.2, 35.0, 19.8, 5.5, 28.7, 28.4, 1.29, 81.7),
    (7.2, 35.0, 19.8, 8.8, 24.2, 24.0, 0.69, 27.0),
    (7.2, 35.0, 19.8, 11.1, 21.7, 21.5, 0.49, 14.7),
    (7.2, 37.78, 20.7, 10.0, 26.3, 26.0, 0.66, 25.0),
    (7.2, 40.5, 21.5, 6.6, 33.6, 33.3, 1.25, 77.3),
    (7.2, 40.5, 21.5, 11.1, 28.4, 28.1, 0.64, 23.8),
    (10.0, 35.0, 19.8, 5.5, 24.9, 24.9, 1.13, 64.5),
    (10.0, 35.0, 19.8, 6.6, 23.5, 23.5, 0.87, 40.8),
    (10.0, 35.0, 19.8, 7.7, 22.1, 22.1, 0.72, 29.1),
    (5.0, 26.7, 17.1, 5.5, 21.3, 20.9, 0.97, 49.8),
    (5.0, 26.7, 17.1, 10.0, 15.3, 15.0, 0.39, 9.1),
    (7.2, 26.7, 17.1, 5.5, 18.4, 18.2, 0.84, 38.7),
]
V_RATED = 1.5102          # m3/s (3200 CFM) ACRC301S nominal air flow
W_RATED_INPUT = 805.0     # W, input power at max operating condition (se.com ACRC301S page)
M_W_MAX_BASE = 1.3        # L/s, max base-load CW flow for ACRC301S (guide spec); peak +25 %
_CAL = {}


def _W_from_db_wb(DB, WB):
    return CP.HAPropsSI("W", "T", DB + K0, "P", P.P0, "B", WB + K0)


def calibrate():
    """Fit the single air-side multiplier C_CAL to the rated point and the CW-side external loss
    coefficient K_EXT (valve + headers + internal piping) to the rated CW dP.  Cached."""
    if _CAL:
        return _CAL
    ewt, db, wb, dTw, Qt, Qs, Vw, dPk = DATASHEET[0]
    W = _W_from_db_wb(db, wb)
    m_a = V_RATED * P.rho_air(db)
    rho_w = water_props(ewt + dTw / 2)["rho"]
    m_w = Vw * 1e-3 * rho_w
    cp_w = water_props(ewt + dTw / 2)["cp"]
    Q_gross = m_w * cp_w * dTw                        # water-side gross coil load (total)

    def res(C):
        return coil(m_a, db, m_w, ewt, W_in=W, C_cal=C, K_ext=0.0)["Q"] - Q_gross
    C = brentq(res, 0.2, 6.0, xtol=1e-5)
    g = geometry(IRC_GEOM)
    ws = water_side(m_w, ewt + dTw / 2, g)
    K_ext = (dPk * 1e3 - ws["dP_coil_w"]) / (Vw * 1e-3) ** 2   # Pa/(m3/s)^2
    # fan wire-to-air efficiency from the datasheet input power: 805 W at max operating condition
    # minus W_CONTROLS (controller/display/sensors) must equal V dP_total/eta at rated flow.
    # Cross-check: gross(m_w cp dT_w) - net(total) at the rating point = fan heat ~ 870 W.
    dPc = coil(m_a, db, m_w, ewt, W_in=W, C_cal=C, K_ext=0.0)["dP_air"]
    dP_tot = dPc + FAN["dP_filter_rated"] + FAN["dP_casing_rated"]
    eta_fan = V_RATED * dP_tot / (W_RATED_INPUT - W_CONTROLS)
    _CAL.update(C_cal=C, K_ext=K_ext, eta_fan=eta_fan, dP_air_total_rated=dP_tot, Q_gross_rated=Q_gross, m_a_rated=m_a, m_w_rated=m_w,
                fan_heat_implied=Q_gross - Qt * 1e3, dP_coil_w_rated=ws["dP_coil_w"])
    return _CAL


# =================================================================================== coil
def coil(m_air, T_air_in_C, m_w, T_w_in_C, T_dp_C=None, W_in=None, geom=None,
         arrangement="n_pass", C_cal=None, wet_model=True, K_ext=None):
    """Chilled-water fin-and-tube coil, one unit (eps-NTU, dry + Braun-Mitchell-Klein wet check).

    Parameters
    ----------
    m_air      : dry-air mass flow through the coil [kg/s]
    T_air_in_C : entering air temperature (EAT = IRC return, T_ret) [degC]
    m_w        : chilled-water mass flow [kg/s]
    T_w_in_C   : entering water temperature (EWT) [degC]
    T_dp_C     : entering-air dew point [degC]; default params.ROOM_DEWPOINT_C
    W_in       : entering humidity ratio [kg/kg] (overrides T_dp_C)
    geom       : geometry dict (IRC_GEOM default; RDHX_GEOM for doors)
    arrangement: "n_pass" (rows as crossflow passes in counterflow, default) or "counterflow"
    C_cal      : air-side h multiplier (default: calibrated value for IRC, 1.0 for other geoms)

    Returns dict: Q (total, W), Q_sens, SHR, T_air_out, T_w_out, eps, NTU, UA, Cr, C_air, C_w,
    h_air, h_w, eta_f, eta_o, dP_air (coil only), dP_w (coil tubes + calibrated unit externals),
    T_surf_min, dry (1/0), Re_Dc, Re_w, v_face, ...
    """
    g = geometry(IRC_GEOM if geom is None else geom)
    is_irc = geom is None or geom.get("name") == IRC_GEOM["name"]
    if C_cal is None:
        C_cal = calibrate()["C_cal"] if is_irc else 1.0
    if W_in is None:
        W_in = W_from_dewpoint(P.ROOM_DEWPOINT_C if T_dp_C is None else T_dp_C)
    T_dp = CP.HAPropsSI("D", "T", T_air_in_C + K0, "P", P.P0, "W", W_in) - K0
    cp_m = CPA + W_in * CPV                                          # moist-air cp per kg dry air

    # ---- first pass dry: outlet guesses for property temperatures
    T_a_m, T_w_m = T_air_in_C - 6.0, T_w_in_C + 3.0
    for _ in range(3):
        ap = air_props(T_a_m)
        G = m_air / g["A_min"]                                       # kg/m2/s at min. free flow
        Re = G * g["Dc"] / ap["mu"]                                  # Re_Dc (Wang 2000)
        j, f_a = wang2000_plain(Re, g)
        h_a = C_cal * j * G * cp_m / ap["Pr"] ** (2 / 3)             # Colburn: h = j G cp Pr^-2/3
        eta_f = schmidt_fin_eff(h_a, g)
        eta_o = 1 - g["A_fin"] / g["A_a"] * (1 - eta_f)              # overall surface efficiency
        ws = water_side(m_w, T_w_m, g)
        R_a = 1 / (eta_o * h_a * g["A_a"])
        R_wall = np.log(g["Do"] / g["Di"]) / (2 * np.pi * g["k_tube"] * g["W"] * g["n_t"])
        R_foul = RF_W / g["A_w"]
        R_wi = 1 / (ws["h_w"] * g["A_w"])
        R_wside = R_wall + R_foul + R_wi
        UA = 1 / (R_a + R_wside)                                     # overall conductance
        C_a, C_w = m_air * cp_m, m_w * ws["cp_w"]
        Cmin, Cmax = min(C_a, C_w), max(C_a, C_w)
        Cr = Cmin / Cmax
        NTU = UA / Cmin
        eps = _eps(NTU, Cr, g, arrangement, water_is_min=C_w < C_a)
        Q_dry = eps * Cmin * (T_air_in_C - T_w_in_C)
        T_a_out = T_air_in_C - Q_dry / C_a
        T_w_out = T_w_in_C + Q_dry / C_w
        T_a_m, T_w_m = 0.5 * (T_air_in_C + T_a_out), 0.5 * (T_w_in_C + T_w_out)

    # coldest wetted-surface point: air outlet / water inlet end (counterflow), resistance divider
    T_s_min = T_w_in_C + (T_a_out - T_w_in_C) * R_wside / (R_a + R_wside)
    dry = T_s_min >= T_dp
    Q, Q_sens, T_ao, T_wo, SHR = Q_dry, Q_dry, T_a_out, T_w_out, 1.0
    eps_used, NTU_used, wet_info = eps, NTU, {}
    if (not dry) and wet_model:
        # ---- Braun, Mitchell & Klein (1989) enthalpy-based effectiveness (fully wet coil)
        # c_s = dh_sat/dT over water range; UA* = 1/(cp_m R_a + c_s R_wside) [kg/s]
        # m* = m_a c_s/(m_w cp_w); Q = eps* Cmin* (h_a,in - h_sat(T_w,in))
        h_ai = h_moist(T_air_in_C, W_in)
        Two = T_w_out
        for _ in range(30):
            c_s = (h_sat(Two) - h_sat(T_w_in_C)) / max(Two - T_w_in_C, 1e-3)
            UAs = 1 / (cp_m * R_a + c_s * R_wside)
            Ca_s, Cw_s = m_air, m_w * ws["cp_w"] / c_s
            Cmin_s, Cmax_s = min(Ca_s, Cw_s), max(Ca_s, Cw_s)
            NTU_s = UAs / Cmin_s
            eps_s = _eps(NTU_s, Cmin_s / Cmax_s, g, arrangement, water_is_min=Cw_s < Ca_s)
            Q_wet = eps_s * Cmin_s * (h_ai - h_sat(T_w_in_C))
            Two_new = T_w_in_C + Q_wet / (m_w * ws["cp_w"])
            if abs(Two_new - Two) < 1e-4:
                break
            Two = 0.5 * Two + 0.5 * Two_new
        if Q_wet > Q_dry:                                    # partly-wet: take max(dry, wet)
            h_ao = h_ai - Q_wet / m_air
            NTU_ao = 1 / (R_a * m_air * cp_m)                 # air-side NTU (Purdue ME418 notes)
            h_s_eff = h_ai - (h_ai - h_ao) / (1 - np.exp(-NTU_ao))
            T_s_eff = T_sat_from_h(h_s_eff)
            T_ao = T_s_eff + (T_air_in_C - T_s_eff) * np.exp(-NTU_ao)
            Q = Q_wet
            Q_sens = min(m_air * cp_m * (T_air_in_C - T_ao), Q)
            SHR = Q_sens / Q
            T_wo = T_w_in_C + Q / (m_w * ws["cp_w"])
            eps_used, NTU_used = eps_s, NTU_s
            wet_info = dict(c_s=c_s, m_star=Ca_s * c_s / (m_w * ws["cp_w"]), T_s_eff=T_s_eff,
                            Q_latent=Q - Q_sens)

    # ---- air-side coil dP: Kays & London core equation with Wang f definition
    #   dP = G^2/(2 rho_i) [ (1+sigma^2)(rho_i/rho_o - 1) + f (A_a/A_min)(rho_i/rho_m) ]
    rho_i, rho_o = P.rho_air(T_air_in_C), P.rho_air(T_ao)
    rho_m = 2 / (1 / rho_i + 1 / rho_o)
    dP_air = G ** 2 / (2 * rho_i) * ((1 + g["sigma"] ** 2) * (rho_i / rho_o - 1)
                                     + f_a * g["A_a"] / g["A_min"] * rho_i / rho_m)
    # ---- water dP: tube friction + bends (computed) + unit externals (valve/headers, calibrated)
    V_w = m_w / ws["rho_w"]
    if K_ext is None:
        K_ext = calibrate()["K_ext"] if is_irc else K_EXT_RDHX
    dP_w = ws["dP_coil_w"] + K_ext * V_w ** 2
    out = dict(Q=Q, Q_sens=Q_sens, Q_latent=Q - Q_sens, SHR=SHR, T_air_out=T_ao, T_w_out=T_wo,
               dT_air=T_air_in_C - T_ao, dT_w=T_wo - T_w_in_C, ITD=T_air_in_C - T_w_in_C,
               approach=T_ao - T_w_in_C, eps=eps_used, eps_dry=eps, NTU=NTU_used, NTU_dry=NTU,
               UA=UA, Cr=Cr, C_air=C_a, C_w=C_w, h_air=h_a, h_w=ws["h_w"], j=j, f_air=f_a,
               eta_f=eta_f, eta_o=eta_o, R_air=R_a, R_wall=R_wall, R_foul=R_foul, R_w_conv=R_wi,
               dP_air=dP_air, dP_w=dP_w, dP_w_coil=ws["dP_coil_w"], V_w=V_w, v_w=ws["v_w"],
               Re_w=ws["Re_w"], Re_Dc=Re, v_face=m_air / rho_i / g["A_face"],
               V_air=m_air / rho_i, T_surf_min=T_s_min, T_dewpoint=T_dp, dry=float(dry),
               wet_risk=float(not dry), C_cal=C_cal, Q_dry_model=Q_dry, cp_m=cp_m,
               cp_w=ws["cp_w"], W_in=W_in)
    out.update(wet_info)
    return {k: float(v) for k, v in out.items()}


# =================================================================================== fans
FAN = dict(
    n_fans=8,             # ACRC301S: eight 200 mm mixed-flow direct-drive DC fans (guide spec)
    eta_tot=None,         # wire-to-air efficiency: CALIBRATED in calibrate() from the 805 W rated
                          # input (result ~0.32).  Brief suggests 0.45-0.6; ebm-papst quotes >60 %
                          # static-total only for large GreenTech EC radials.  Set a float here
                          # to override (e.g. 0.45 for a modern EC retrofit sensitivity).
    dP_filter_rated=25.0,  # Pa at V_RATED, 1/2" washable MERV-1 filter (guide spec) [assumption]
    n_filter=1.5,         # dP_filter ~ V^1.5 (mesh/fibrous, between laminar 1 and turbulent 2)
    dP_casing_rated=40.0,  # Pa at V_RATED: rear grille + fan plate + front discharge [assumption]
    min_speed=0.30,       # 30-100 % variable speed (guide spec)
)
W_CONTROLS = 50.0         # W, controller + display + sensors + idle condensate pump [assumption]
RF_W = 0.000044           # m2K/W water-side fouling (closed CW loop; ~0.00025 h ft2 F/Btu) [assumption]
K_EXT_RDHX = 2.0e10       # Pa/(m3/s)^2 door hoses+valve: 20 kPa at 1 L/s [assumption]
DP_BRANCH_DESIGN = 60e3   # Pa CW branch piping/valves/strainer between riser and unit at rated flow
ETA_PUMP = 0.55           # CHW pump wire-to-water efficiency [assumption]


def irc_fan(m_air, T_air_C, geom=None, coil_dP=None):
    """IRC fan power for one unit at a given air flow.

    dP_total = dP_coil (Wang f, Kays-London core eq.) + dP_filter (~V^1.5) + dP_casing (~V^2)
    W_fan    = V dP_total / eta_tot           (air power / wire-to-air efficiency)
    Below min speed the fans cannot turn down further (floor at 30 % of V_RATED; flow then
    held at that minimum -- the IRC would short-cycle / need staging).  Affinity scaling follows
    automatically: V ~ n, dP ~ n^~1.8-2, W ~ n^~2.8-3 (Vertiv EC plug-fan technical note).
    """
    rho = P.rho_air(T_air_C)
    V = m_air / rho
    s = V / V_RATED
    if coil_dP is None:
        # coil dP at a nominal 14 degC EWT, 1 kg/s water (air dP only weakly depends on water side)
        coil_dP = coil(m_air, T_air_C, 1.0, 14.0, geom=geom)["dP_air"]
    dP_f = FAN["dP_filter_rated"] * s ** FAN["n_filter"]
    dP_c = FAN["dP_casing_rated"] * s ** 2
    dP = coil_dP + dP_f + dP_c
    eta = FAN["eta_tot"] if FAN["eta_tot"] is not None else calibrate()["eta_fan"]
    W = V * dP / eta
    return dict(eta_fan=eta, V_air=V, speed_frac=s, dP_coil_air=coil_dP, dP_filter=dP_f, dP_casing=dP_c,
                dP_air_total=dP, W_air=V * dP, W_fan=W, below_min_speed=float(s < FAN["min_speed"]),
                above_rated_flow=float(s > 1.0 + 1e-9))


def chw_pump(m_w, dP_unit, T_w_C=14.0):
    """CHW pumping power attributable to one IRC: W = V (dP_unit + dP_branch)/eta_pump,
    dP_branch = DP_BRANCH_DESIGN (V/V_rated)^2.  Plant-side (evaporator, header) dP is S5's."""
    rho = water_props(T_w_C)["rho"]
    V = m_w / rho
    dP_br = DP_BRANCH_DESIGN * (V / (1.13e-3)) ** 2
    return dict(V_w=V, dP_branch=dP_br, dP_w_total=dP_unit + dP_br,
                W_pump_chw=V * (dP_unit + dP_br) / ETA_PUMP)


def X_dest_coil(m_air, T_a_in, T_a_out, m_w, T_w_in, T_w_out, dP_air, dP_w, cp_w=4190.0,
                Q_latent=0.0, cp_a=CPA):
    """Coil exergy destruction (Gouy-Stodola), sensible streams + friction terms.

      X_heat = T0 [ m_a cp_m ln(T_a,out/T_a,in) + m_w cp_w ln(T_w,out/T_w,in) ]   (temperatures K)
      X_fric = T0 [ V_a dP_a / T_a,m + V_w dP_w / T_w,m ]   (isothermal throttling, ideal gas /
               incompressible: S_gen = V dP / T)
    A latent load is not included in X_heat (sensible-only form); it is reported separately.
    """
    T0 = P.T0_K
    Tai, Tao, Twi, Two = (x + K0 for x in (T_a_in, T_a_out, T_w_in, T_w_out))
    S_heat = m_air * cp_a * np.log(Tao / Tai) + m_w * cp_w * np.log(Two / Twi)
    Va = m_air / P.rho_air(0.5 * (T_a_in + T_a_out))
    Vw = m_w / water_props(0.5 * (T_w_in + T_w_out))["rho"]
    Xa = T0 * Va * dP_air / (0.5 * (Tai + Tao))
    Xw = T0 * Vw * dP_w / (0.5 * (Twi + Two))
    Xh = T0 * S_heat
    return dict(X_dest_heat=Xh, X_dest_fric_air=Xa, X_dest_fric_w=Xw,
                X_dest_coil=Xh + Xa + Xw, S_gen_heat=S_heat,
                # exergy given up by the air / gained by the water (flow exergy changes, thermal)
                dEx_air=m_air * cp_a * ((Tai - Tao) - T0 * np.log(Tai / Tao)),
                dEx_w=m_w * cp_w * ((Two - Twi) - T0 * np.log(Two / Twi)),
                Q_latent_excluded=Q_latent)


def irc_unit(m_air, T_air_in_C, m_w, T_w_in_C, **kw):
    """One in-row cooler: coil + EC fans + CHW pumping + exergy.

    Fans sit downstream of the coil (rear return -> coil -> fans -> front discharge), so the
    supply temperature is T_supply = T_air_out(coil) + W_fan/(m_a cp).  Net sensible capacity
    delivered to the room = Q_sens - W_fan.
    """
    c = coil(m_air, T_air_in_C, m_w, T_w_in_C, **kw)
    fan = irc_fan(m_air, T_air_in_C, geom=kw.get("geom"), coil_dP=c["dP_air"])
    pump = chw_pump(m_w, c["dP_w"], T_w_in_C)
    x = X_dest_coil(m_air, T_air_in_C, c["T_air_out"], m_w, T_w_in_C, c["T_w_out"],
                    c["dP_air"], c["dP_w"], cp_w=c["cp_w"], Q_latent=c["Q_latent"],
                    cp_a=c["cp_m"])
    out = dict(c)
    out.update(fan)
    out.update(pump)
    out.update(x)
    out["T_supply"] = c["T_air_out"] + fan["W_fan"] / (m_air * CPA)
    out["Q_net_sens"] = c["Q_sens"] - fan["W_fan"]
    # fan-system exergy destruction: shaft/wire losses + filter/casing throttling (approx.)
    out["X_dest_fan_system"] = fan["W_fan"] - fan["V_air"] * c["dP_air"]
    return {k: float(v) for k, v in out.items()}


def solve_chw(m_air, T_air_in_C, T_air_out_target_C, T_w_in_C, include_fan_heat=True,
              m_w_max=None, **kw):
    """Required chilled-water flow for one IRC to deliver a target SUPPLY air temperature.

    If include_fan_heat, the coil must leave T_target - W_fan/(m_a cp) (fans after the coil).
    m_w_max default: 1.25 x 1.3 L/s (ACRC301S base-load limit + 25 % peak allowance).
    Returns dict with feasible (1/0), m_w [kg/s] (nan if infeasible), T_w_out, dT_w, dP_w,
    W_pump_chw, W_fan, T_air_out_min (at m_w_max), reason code (0 ok, 1 below EWT+approach,
    2 needs > m_w_max), plus everything in irc_unit at the solution.
    """
    rho_w = water_props(T_w_in_C)["rho"]
    if m_w_max is None:
        m_w_max = 1.25 * M_W_MAX_BASE * 1e-3 * rho_w
    m_w_min = 0.02
    fanW = irc_fan(m_air, T_air_in_C, geom=kw.get("geom"))["W_fan"] if include_fan_heat else 0.0
    T_coil_target = T_air_out_target_C - fanW / (m_air * CPA)
    base = dict(T_air_out_target=T_air_out_target_C, T_coil_out_target=T_coil_target,
                m_w_max=m_w_max)
    if T_coil_target >= T_air_in_C:
        r = irc_unit(m_air, T_air_in_C, m_w_min, T_w_in_C, **kw)
        r.update(base, feasible=1.0, m_w=m_w_min, reason=0.0, T_air_out_min=np.nan)
        return r
    hi = coil(m_air, T_air_in_C, m_w_max, T_w_in_C, **kw)
    if hi["T_air_out"] > T_coil_target:
        r = irc_unit(m_air, T_air_in_C, m_w_max, T_w_in_C, **kw)
        r.update(base, feasible=0.0, m_w=np.nan, reason=1.0 if T_coil_target <= T_w_in_C else 2.0,
                 T_air_out_min=hi["T_air_out"] + fanW / (m_air * CPA))
        return r
    fun = lambda mw: coil(m_air, T_air_in_C, mw, T_w_in_C, **kw)["T_air_out"] - T_coil_target
    if fun(m_w_min) <= 0:
        mw = m_w_min
    else:
        mw = brentq(fun, m_w_min, m_w_max, xtol=1e-5, rtol=1e-6)
    r = irc_unit(m_air, T_air_in_C, mw, T_w_in_C, **kw)
    r.update(base, feasible=1.0, m_w=mw, reason=0.0,
             T_air_out_min=hi["T_air_out"] + fanW / (m_air * CPA))
    return r


def pod_irc(m_air_pod, T_ret_C, T_supply_C, T_w_in_C, n_irc=P.N_IRC, **kw):
    """Pod-level convenience for S5: split air equally over n_irc units, solve CHW flow each,
    return pod totals (Q, m_w, W_fan, W_pump_chw, X_dest) + per-unit values."""
    u = solve_chw(m_air_pod / n_irc, T_ret_C, T_supply_C, T_w_in_C, **kw)
    out = {k + "_unit": v for k, v in u.items()}
    out.update(feasible=u["feasible"], n_irc=float(n_irc), Q_pod=n_irc * u["Q"],
               Q_sens_pod=n_irc * u["Q_sens"], m_w_pod=n_irc * u["m_w"],
               T_w_out=u["T_w_out"], dT_w=u["dT_w"], W_fan_pod=n_irc * u["W_fan"],
               W_pump_chw_pod=n_irc * u["W_pump_chw"], X_dest_coil_pod=n_irc * u["X_dest_coil"],
               T_supply=u["T_supply"])
    return out


# =================================================================================== RDHX
RDHX = dict(
    dP_avail_nominal=30.0,     # Pa usable server-fan margin at nominal speed (brief: 20-40 Pa)
    dP_avail_range=(20.0, 40.0),
    eta_server_fan=0.25,       # small 40-60 mm server fans, wire-to-air [assumption]
    eta_door_fan=0.40,         # active-door EC fan matrix, wire-to-air [assumption]
    dP_fanplane=10.0,          # Pa extra loss of the active fan plate/guard at rated door flow
    V_door_rated=1.0,          # m3/s reference for fan-plane loss scaling
)


def rdhx(m_air_rack, T_air_in_C, m_w, T_w_in_C, mode="auto", geom=None, **kw):
    """Rear-door heat exchanger on one rack (passive or active).

    Passive: server fans push air through the door; extra server fan power
        W_par = V dP_door / eta_server_fan (valid while dP_door <= dP_avail).
    Active : fan matrix on the door supplies dP_door (+ fan-plane loss); servers see ~0 Pa:
        W_par = V (dP_door + dP_fanplane) / eta_door_fan.
    mode="auto" picks passive if dP_door <= dP_avail_nominal else active.
    Uses the same coil engine (C_cal = 1, plain fins assumed for doors).
    """
    g = RDHX_GEOM if geom is None else geom
    c = coil(m_air_rack, T_air_in_C, m_w, T_w_in_C, geom=g, **kw)
    V = c["V_air"]
    dP = c["dP_air"]
    passive_ok = dP <= RDHX["dP_avail_nominal"]
    W_passive = V * dP / RDHX["eta_server_fan"]
    dP_fp = RDHX["dP_fanplane"] * (V / RDHX["V_door_rated"]) ** 2
    W_active = V * (dP + dP_fp) / RDHX["eta_door_fan"]
    use_active = (mode == "active") or (mode == "auto" and not passive_ok)
    out = dict(c)
    out.update(dP_door=dP, passive_ok=float(passive_ok), W_parasitic_passive=W_passive,
               W_parasitic_active=W_active, active=float(use_active),
               W_parasitic=W_active if use_active else W_passive,
               dP_avail=RDHX["dP_avail_nominal"])
    return {k: float(v) for k, v in out.items()}


# =================================================================================== self-test
def _style(plt):
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e3",
                         "grid.linewidth": 0.6, "axes.edgecolor": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "font.size": 9, "axes.titlesize": 10,
                         "lines.linewidth": 2.0, "legend.frameon": False})


COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK2 = "#52514e"


def validation_table():
    """Model vs ACRC301S datasheet rows: gross total (net + modelled fan heat) and CW dP."""
    rows = []
    for ewt, db, wb, dTw, Qt, Qs, Vw, dPk in DATASHEET:
        W = _W_from_db_wb(db, wb)
        m_a = V_RATED * P.rho_air(db)
        m_w = Vw * 1e-3 * water_props(ewt + dTw / 2)["rho"]
        c = coil(m_a, db, m_w, ewt, W_in=W)
        fan = irc_fan(m_a, db, coil_dP=c["dP_air"])
        Q_ds_gross = m_w * water_props(ewt + dTw / 2)["cp"] * dTw
        rows.append(dict(EWT=ewt, DB=db, WB=wb, Vw=Vw, Q_ds_gross=Q_ds_gross,
                         Q_ds_net_sens=Qs * 1e3, Q_model=c["Q"],
                         Q_model_net_sens=c["Q_sens"] - fan["W_fan"],
                         SHR_ds=Qs / Qt, SHR_model=c["SHR"], dP_ds=dPk * 1e3, dP_model=c["dP_w"],
                         dry=c["dry"]))
    return rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    cal = calibrate()
    g = geometry(IRC_GEOM)
    B = P.BASE
    # ------------------------------------------------ baseline (placeholders, brief)
    m_pod, T_ret, T_sup, EWT = 10.634, 35.82, 24.0, B["T_chw_in"]   # integrated S2+S3 baseline (psi=0, beta=0.05)
    m_unit = m_pod / P.N_IRC
    base = solve_chw(m_unit, T_ret, T_sup, EWT)
    podb = pod_irc(m_pod, T_ret, T_sup, EWT)
    val = validation_table()
    err_Q = np.array([(r["Q_model"] - r["Q_ds_gross"]) / r["Q_ds_gross"] for r in val])
    err_dP = np.array([(r["dP_model"] - r["dP_ds"]) / r["dP_ds"] for r in val])
    fan_rated = irc_fan(cal["m_a_rated"], 37.78)
    cmp_cf = coil(m_unit, T_ret, base["m_w"], EWT, arrangement="counterflow")

    print("=" * 78)
    print("S4  In-row cooler (IRC) -- geometry (per unit)")
    print("=" * 78)
    for k, u, s in [("A_face", "m2", 1), ("A_a", "m2", 1), ("A_fin", "m2", 1), ("A_w", "m2", 1),
                    ("A_min", "m2", 1), ("sigma", "-", 1), ("Dh", "mm", 1e3), ("Dc", "mm", 1e3),
                    ("Fp", "mm", 1e3), ("n_t", "-", 1), ("passes", "-", 1), ("L_circ", "m", 1)]:
        print(f"  {k:10s} = {g[k]*s:10.4g} {u}")
    print(f"  A_a/A_w    = {g['A_a']/g['A_w']:.1f}")
    print(f"\nCalibration: C_cal (air-side h multiplier) = {cal['C_cal']:.3f}; "
          f"K_ext = {cal['K_ext']:.3g} Pa/(m3/s)^2; implied fan heat {cal['fan_heat_implied']:.0f} W")
    print(f"Validation over {len(val)} datasheet rows: Q error mean {100*err_Q.mean():+.1f} %, "
          f"max |err| {100*np.abs(err_Q).max():.1f} %; CW dP mean {100*err_dP.mean():+.1f} %, "
          f"max |err| {100*np.abs(err_dP).max():.1f} %")
    print(f"Fan at rated 1.51 m3/s: dP_total {fan_rated['dP_air_total']:.0f} Pa, "
          f"W_fan {fan_rated['W_fan']:.0f} W (+{W_CONTROLS:.0f} W controls = datasheet "
          f"{W_RATED_INPUT:.0f} W) -> calibrated eta_fan = {cal['eta_fan']:.3f}")
    print("\nBaseline (per IRC): m_air %.3f kg/s, T_ret %.1f C, T_supply target %.1f C, EWT %.1f C"
          % (m_unit, T_ret, T_sup, EWT))
    keys = [("feasible", "-"), ("m_w", "kg/s"), ("Q", "W"), ("Q_sens", "W"), ("SHR", "-"),
            ("T_air_out", "C"), ("T_supply", "C"), ("T_w_out", "C"), ("dT_w", "K"),
            ("eps", "-"), ("NTU", "-"), ("Cr", "-"), ("UA", "W/K"), ("h_air", "W/m2K"),
            ("h_w", "W/m2K"), ("eta_f", "-"), ("eta_o", "-"), ("Re_Dc", "-"), ("Re_w", "-"),
            ("v_face", "m/s"), ("T_surf_min", "C"), ("dry", "-"), ("dP_air", "Pa"),
            ("dP_air_total", "Pa"), ("W_fan", "W"), ("dP_w", "Pa"), ("W_pump_chw", "W"),
            ("X_dest_heat", "W"), ("X_dest_fric_air", "W"), ("X_dest_fric_w", "W"),
            ("X_dest_coil", "W")]
    for k, u in keys:
        print(f"  {k:16s} = {base[k]:12.4g} {u}")
    print(f"\nPod ({P.N_IRC} IRC): Q {podb['Q_pod']/1e3:.1f} kW, m_w {podb['m_w_pod']:.2f} kg/s, "
          f"W_fan {podb['W_fan_pod']:.0f} W, W_pump_chw {podb['W_pump_chw_pod']:.0f} W, "
          f"X_dest_coil {podb['X_dest_coil_pod']:.0f} W")
    print(f"eps n-pass {base['eps']:.4f} vs counterflow {cmp_cf['eps']:.4f} at same flows "
          f"(Q diff {100*(cmp_cf['Q']-base['Q'])/base['Q']:+.2f} %)")

    # ------------------------------------------------ FIG 1: validation parity
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.0))
    q_ds = np.array([r["Q_ds_gross"] for r in val]) / 1e3
    q_m = np.array([r["Q_model"] for r in val]) / 1e3
    wet = np.array([r["dry"] == 0 for r in val])
    lim = [12, 38]
    ax[0].plot(lim, lim, color=INK2, lw=1)
    ax[0].fill_between(lim, [l * 0.95 for l in lim], [l * 1.05 for l in lim], color="#e6e6e3")
    ax[0].scatter(q_ds[~wet], q_m[~wet], s=36, color=COLS[0], label="surface above dew point (dry)", zorder=3,
                  edgecolor="white", lw=1)
    ax[0].scatter(q_ds[wet], q_m[wet], s=36, color=COLS[1], marker="s", label="surface < dew point (BMK checked)",
                  zorder=3, edgecolor="white", lw=1)
    ax[0].scatter(q_ds[0], q_m[0], s=120, facecolor="none", edgecolor="#0b0b0b", lw=1.2,
                  label="calibration point", zorder=4)
    ax[0].set(xlim=lim, ylim=lim, xlabel="Datasheet gross capacity  m_w c_p ΔT_w  [kW]",
              ylabel="Model coil capacity [kW]", title="Capacity (band ±5 %)")
    ax[0].legend(loc="upper left")
    dps = np.array([r["dP_ds"] for r in val]) / 1e3
    dpm = np.array([r["dP_model"] for r in val]) / 1e3
    lim2 = [0, 95]
    ax[1].plot(lim2, lim2, color=INK2, lw=1)
    ax[1].scatter(dps, dpm, s=36, color=COLS[0], edgecolor="white", lw=1, zorder=3)
    ax[1].set(xlim=lim2, ylim=lim2, xlabel="Datasheet total CW pressure drop [kPa]",
              ylabel="Model CW pressure drop [kPa]", title="Chilled-water pressure drop")
    fig.suptitle(f"One air-side multiplier C_cal = {cal['C_cal']:.2f} reproduces ACRC301S ratings "
                 f"within {100*np.abs(err_Q).max():.0f} % (13 rows, EWT 5-10 °C)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "s4_validation.png"), dpi=200)
    plt.close(fig)

    # ------------------------------------------------ FIG 2: capacity vs EAT
    EATs = np.linspace(26, 46, 21)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))
    for i, mw in enumerate([0.3, 0.6, 1.0, 1.4]):
        Q = [coil(m_unit, t, mw, EWT)["Q_sens"] / 1e3 for t in EATs]
        ax[0].plot(EATs, Q, color=COLS[i], label=f"m_w = {mw:.1f} kg/s")
    ax[0].axvline(T_ret, color=INK2, lw=1, ls="--")
    ax[0].text(T_ret + 0.3, ax[0].get_ylim()[0] + 1, "baseline T_ret", color=INK2, fontsize=8)
    ax[0].set(xlabel="Entering air temperature EAT = T_ret [°C]", ylabel="Sensible capacity [kW]",
              title=f"EWT = {EWT:.0f} °C, m_air = {m_unit:.2f} kg/s")
    ax[0].legend()
    for i, ewt in enumerate([7.0, 10.0, 14.0, 18.0]):
        Q = [coil(m_unit, t, 0.8, ewt)["Q_sens"] / 1e3 for t in EATs]
        ax[1].plot(EATs, Q, color=COLS[i], label=f"EWT = {ewt:.0f} °C")
    ax[1].set(xlabel="Entering air temperature EAT [°C]", ylabel="Sensible capacity [kW]",
              title="m_w = 0.8 kg/s; EWT ≲ 9 °C: coil surface < 12 °C dew point")
    ax[1].legend()
    for i, ewt in enumerate([7.0, 10.0, 14.0, 18.0]):
        itd = EATs - ewt
        Qd = [coil(m_unit, t, 0.8, ewt, wet_model=False)["Q"] / 1e3 for t in EATs]
        ax[2].plot(itd, Qd, color=COLS[i], label=f"EWT = {ewt:.0f} °C")
    ax[2].set(xlabel="ITD = EAT − EWT [K]", ylabel="Dry-coil capacity [kW]",
              title="Dry capacity collapses onto Q ≈ ε C_min · ITD")
    ax[2].legend()
    fig.suptitle("IRC capacity is set by ITD: each +1 K of return air (less bypass) adds "
                 f"≈{(coil(m_unit, 38, 0.8, EWT)['Q_sens']-coil(m_unit, 37, 0.8, EWT)['Q_sens'])/1e3:.2f}"
                 " kW per unit at 0.8 kg/s", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "s4_capacity_vs_EAT.png"), dpi=200)
    plt.close(fig)

    # ------------------------------------------------ FIG 3: CHW map (EWT x T_supply)
    ewts = np.linspace(6, 22, 33)
    tsups = np.linspace(16, 30, 29)
    MW = np.full((len(tsups), len(ewts)), np.nan)
    TWO = np.full_like(MW, np.nan)
    WET = np.zeros_like(MW)
    for a, ts in enumerate(tsups):
        for b, ew in enumerate(ewts):
            r = solve_chw(m_unit, T_ret, ts, ew)
            if r["feasible"]:
                MW[a, b] = r["m_w"]
                TWO[a, b] = r["T_w_out"]
                WET[a, b] = 1 - r["dry"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    cs = ax[0].contourf(ewts, tsups, MW, levels=np.linspace(0, 1.65, 12), cmap="Blues")
    fig.colorbar(cs, ax=ax[0], label="Required CHW flow per IRC [kg/s]")
    ax[0].contourf(ewts, tsups, np.isnan(MW).astype(float), levels=[0.5, 1.5], colors="none",
                   hatches=["///"])
    ax[0].contour(ewts, tsups, WET, levels=[0.5], colors=[COLS[1]], linewidths=1.5)
    ax[0].plot(EWT, T_sup, "o", color="#0b0b0b", ms=6)
    ax[0].annotate("baseline", (EWT, T_sup), xytext=(5, 5), textcoords="offset points",
                   fontsize=8)
    ax[0].text(15.0, 16.6, "hatched = infeasible (needs m_w > 1.62 kg/s)", fontsize=8,
               color="#0b0b0b", bbox=dict(fc="white", ec="none", alpha=0.9))
    ax[0].text(6.4, 28.5, "orange line: wet-coil boundary\n(left/below = condensation)",
               fontsize=8, color="#0b0b0b", bbox=dict(fc="white", ec="none", alpha=0.8))
    ax[0].set(xlabel="Entering water temperature EWT [°C]", ylabel="Target supply air [°C]",
              title=f"Required CHW flow (T_ret {T_ret:.0f} °C, m_air {m_unit:.2f} kg/s)")
    cs2 = ax[1].contourf(ewts, tsups, TWO, levels=12, cmap="Oranges")
    fig.colorbar(cs2, ax=ax[1], label="Leaving water temperature [°C]")
    cl = ax[1].contour(ewts, tsups, TWO - ewts[None, :], levels=[3, 5, 8, 12, 16], colors=INK2,
                       linewidths=0.8)
    ax[1].clabel(cl, fmt="ΔT_w=%g K", fontsize=7)
    ax[1].plot(EWT, T_sup, "o", color="#0b0b0b", ms=6)
    ax[1].set(xlabel="Entering water temperature EWT [°C]", ylabel="Target supply air [°C]",
              title="Return-water temperature & ΔT_w (high ΔT_w helps the chiller plant)")
    fig.suptitle("Raising EWT toward T_supply − ~4 K costs CHW flow but lifts LWT; "
                 f"baseline needs {base['m_w']:.2f} kg/s per IRC at ΔT_w = {base['dT_w']:.1f} K",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "s4_chw_map.png"), dpi=200)
    plt.close(fig)

    # ------------------------------------------------ FIG 4: EWT trade-off line (S5 input)
    ew_line = np.linspace(8, 20, 25)
    tr = [solve_chw(m_unit, T_ret, T_sup, e) for e in ew_line]
    ok = np.array([t["feasible"] for t in tr]) > 0
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(ew_line[ok], [P.N_IRC * t["m_w"] for t, o in zip(tr, ok) if o], color=COLS[0])
    ax[0].set(xlabel="EWT [°C]", ylabel="Pod CHW flow [kg/s]", title="Flow rises steeply near EWT_max")
    ax[1].plot(ew_line[ok], [t["T_w_out"] for t, o in zip(tr, ok) if o], color=COLS[1],
               label="leaving water T")
    ax[1].plot(ew_line[ok], [t["dT_w"] for t, o in zip(tr, ok) if o], color=COLS[2],
               label="ΔT_w")
    ax[1].set(xlabel="EWT [°C]", ylabel="[°C] / [K]", title="Return water temperature & ΔT_w")
    ax[1].legend()
    ax[2].plot(ew_line[ok], [P.N_IRC * t["W_pump_chw"] for t, o in zip(tr, ok) if o],
               color=COLS[6], label="CHW pump (IRC branches)")
    ax[2].plot(ew_line[ok], [P.N_IRC * t["X_dest_coil"] for t, o in zip(tr, ok) if o],
               color=COLS[3], label="coil X_dest")
    ax[2].set(xlabel="EWT [°C]", ylabel="Pod power / exergy [W]",
              title="Pump power vs coil exergy destruction")
    ax[2].legend()
    ewmax = ew_line[ok].max() if ok.any() else np.nan
    fig.suptitle(f"Baseline load (T_ret {T_ret:.0f} → {T_sup:.0f} °C): feasible up to EWT ≈ "
                 f"{ewmax:.1f} °C; higher EWT cuts coil X_dest but multiplies pump power",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "s4_ewt_tradeoff.png"), dpi=200)
    plt.close(fig)

    # ------------------------------------------------ FIG 5: fans
    Vs = np.linspace(0.3, 1.6, 40)
    rho = P.rho_air(T_ret)
    F = [irc_fan(v * rho, T_ret) for v in Vs]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.0))
    ax[0].plot(Vs, [f["dP_coil_air"] for f in F], color=COLS[0], label="coil (Wang 2000 f)")
    ax[0].plot(Vs, [f["dP_filter"] for f in F], color=COLS[1], label="filter")
    ax[0].plot(Vs, [f["dP_casing"] for f in F], color=COLS[2], label="casing / grilles")
    ax[0].plot(Vs, [f["dP_air_total"] for f in F], color="#0b0b0b", label="total")
    ax[0].set(xlabel="Air flow per IRC [m³/s]", ylabel="Pressure drop [Pa]",
              title="Air-side pressure drop build-up")
    ax[0].legend()
    Wf = np.array([f["W_fan"] for f in F])
    ax[1].plot(Vs, Wf, color=COLS[0], label="model, η_tot = %.2f (calibrated)" % F[0]["eta_fan"])
    ax[1].plot([V_RATED], [W_RATED_INPUT - W_CONTROLS], "s", color=COLS[1], ms=7,
               label="ACRC301S 805 W input − 50 W controls")
    vb = base["V_air"]
    ax[1].plot([vb], [base["W_fan"]], "o", color="#0b0b0b", ms=6)
    ax[1].annotate(f"baseline {base['W_fan']:.0f} W", (vb, base["W_fan"]), xytext=(-90, 10),
                   textcoords="offset points", fontsize=8)
    ax[1].axvspan(0, FAN["min_speed"] * V_RATED, color="#e6e6e3")
    ax[1].text(0.03, Wf.max() * 0.45, "below\n30 % speed", fontsize=8, color=INK2)
    ax[1].set(xlabel="Air flow per IRC [m³/s]", ylabel="Fan electrical power [W]",
              title="IRC fan power (affinity ~V^%.1f)" %
                    np.polyfit(np.log(Vs[10:]), np.log(Wf[10:]), 1)[0])
    ax[1].legend(loc="upper left")
    fig.suptitle(f"Coil is {100*base['dP_air']/base['dP_air_total']:.0f} % of IRC air-side dP; "
                 f"fan power scales ~V³ so splitting flow over more units saves energy", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "s4_fan.png"), dpi=200)
    plt.close(fig)

    # ------------------------------------------------ FIG 6: RDHX vs IRC
    gd = geometry(RDHX_GEOM)
    m_rack_base = m_pod / P.RACKS_PER_POD
    Vr = np.linspace(0.05, 3.0, 60)
    Rr = [rdhx(v * rho, T_ret, 0.5, EWT) for v in Vr]
    Rd = [rdhx(v * rho, T_ret, 0.5, EWT, geom=RDHX_DENSE_GEOM) for v in Vr]
    dPd = np.array([r["dP_door"] for r in Rr])
    dPdd = np.array([r["dP_door"] for r in Rd])
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].plot(Vr, dPd, color=COLS[0], label="RDHX door coil (2 rows, 10 FPI)")
    ax[0].plot(Vr, dPdd, color=COLS[6], label="dense door (4 rows, 14 FPI)")
    ax[0].axhspan(*RDHX["dP_avail_range"], color="#e6e6e3", label="server-fan margin 20–40 Pa")
    nv = np.array([3462, 4773, 6600]) / 3600
    ax[0].plot(nv, [9, 16, 28], "s", color=COLS[1], ms=6,
               label="nVent RackChiller passive (larger door)")
    vb_r = m_rack_base / rho
    ax[0].axvline(vb_r, color=INK2, ls="--", lw=1)
    ax[0].text(vb_r + 0.02, dPd.max() * 0.9, "Rudra rack\nbaseline", fontsize=8, color=INK2)
    ax[0].set(xlabel="Rack air flow [m³/s]", ylabel="Door air-side ΔP [Pa]",
              title="Passive-door impedance vs available server-fan pressure")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[1].plot(Vr, [r["W_parasitic_passive"] for r in Rr], color=COLS[0],
               label="passive: extra server-fan power")
    ax[1].plot(Vr, [r["W_parasitic_active"] for r in Rr], color=COLS[1],
               label="active fan matrix")
    ax[1].plot(Vr, [r["W_parasitic"] for r in Rd], color=COLS[6], ls="--",
               label="dense door, auto passive→active")
    lim30 = lambda d: Vr[np.argmax(d > RDHX["dP_avail_nominal"])] if (d > 30).any() else np.nan
    Vlim = lim30(dPd)
    Vlim_txt = f"{Vlim:.2f}" if np.isfinite(Vlim) else f"> {Vr.max():.1f}"
    Vlim_dense = lim30(dPdd)
    ax[1].axvline(Vlim_dense, color=INK2, ls=":", lw=1)
    ax[1].text(Vlim_dense + 0.03, 440, f"dense door hits\n30 Pa at {Vlim_dense:.2f} m³/s",
               fontsize=8, color=INK2)
    # IRC equivalent per rack (4 IRC fans shared by 16 racks) at same per-rack flow
    Vr_irc = Vr[Vr * P.RACKS_PER_POD / P.N_IRC <= 1.25 * V_RATED]   # stay within IRC fan range
    irc_per_rack = [P.N_IRC * irc_fan(v * rho * P.RACKS_PER_POD / P.N_IRC, T_ret)["W_fan"]
                    / P.RACKS_PER_POD for v in Vr_irc]
    ax[1].plot(Vr_irc, irc_per_rack, color=COLS[2], label="IRC fans, share per rack (≤125 % rated)")
    ax[1].set(xlabel="Rack air flow [m³/s]", ylabel="Parasitic fan power per rack [W]",
              title="Air-moving parasitic per rack", ylim=(0, 500))
    ax[1].legend(loc="upper left", fontsize=8)
    rb = rdhx(m_rack_base, T_ret, 0.25, EWT)
    fig.suptitle(f"At Rudra's low air load (≈{vb_r:.2f} m³/s/rack) a passive door costs "
                 f"{rb['dP_door']:.1f} Pa / {rb['W_parasitic']:.1f} W per rack; active fans needed "
                 f"above {Vlim_txt} m³/s (dense door: {Vlim_dense:.2f} m³/s)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "s4_rdhx.png"), dpi=200)
    plt.close(fig)

    # ------------------------------------------------ FIG 7: exergy split vs EWT and aisle coupling
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    xs = [(e, t) for e, t in zip(ew_line, tr) if t["feasible"]]
    ax[0].plot([e for e, _ in xs], [P.N_IRC * t["X_dest_heat"] for _, t in xs], color=COLS[0],
               label="heat transfer (finite ΔT)")
    ax[0].plot([e for e, _ in xs], [P.N_IRC * t["X_dest_fric_air"] for _, t in xs], color=COLS[1],
               label="air friction (coil)")
    ax[0].plot([e for e, _ in xs], [P.N_IRC * t["X_dest_fric_w"] for _, t in xs], color=COLS[2],
               label="water friction (unit)")
    ax[0].set(xlabel="EWT [°C]", ylabel="Pod coil exergy destruction [W]",
              title="Coil X_dest split (baseline air side)")
    ax[0].legend()
    # aisle coupling: beta (cold-air bypass) and psi (recirculation) -> T_ret, T_c, required flow
    Q_air_pod = m_pod * CPA * (T_ret - T_sup)
    betas = np.linspace(0.0, 0.30, 13)
    try:
        import s3_aisle as S3
        mix = lambda ms, Q, Tc, ps, be: S3.aisle_mixing(ms, Q, Tc, ps, be)
    except Exception:          # fallback: same lumped balances (Demetriou & Khalifa 2012)
        def mix(ms, Q, Tc, ps, be):
            dT = Q / (ms * CPA)
            mc = ms * (1 - ps) / (1 - be)
            Th = Tc + dT / (1 - ps)
            return dict(m_c=mc, T_ret=(1 - be) * Th + be * Tc)
    ms = 0.95 * m_pod                                    # server flow (BASE beta 0.05 -> m_c=6)
    for i, psi in enumerate([0.0, 0.1, 0.2]):
        mwp = []
        for be in betas:
            Tc = 24.0 - psi / (1 - psi) * Q_air_pod / (ms * CPA)  # keep server inlet at 24 degC
            mm = mix(ms, Q_air_pod, Tc, psi, be)
            rr = solve_chw(mm["m_c"] / P.N_IRC, mm["T_ret"], Tc, EWT)
            mwp.append(P.N_IRC * rr["m_w"] if rr["feasible"] else np.nan)
        lab = f"ψ = {psi:.1f}" + ("" if np.isfinite(mwp).any() else " (infeasible: T_c < 21 °C)")
        ax[1].plot(betas, mwp, color=COLS[i], label=lab)
    ax[1].set(xlabel="Cold-air bypass / leakage β [-]", ylabel="Pod CHW flow [kg/s]",
              title=f"Aisle coupling (S3 model), server inlet 24 °C, EWT {EWT:.0f} °C")
    ax[1].legend()
    fig.suptitle("Heat-transfer irreversibility dominates coil X_dest; bypass (β) and "
                 "recirculation (ψ) both raise the CHW flow needed", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "s4_exergy_aisle.png"), dpi=200)
    plt.close(fig)

    # ------------------------------------------------ wet-coil check at low EWT
    wet7 = solve_chw(m_unit, T_ret, T_sup, 7.0)
    # max EWT
    ewmax_fine = brentq(lambda e: (solve_chw(m_unit, T_ret, T_sup, e)["feasible"] - 0.5),
                        14.0, 22.0, xtol=0.01) if not np.isnan(ewmax) else np.nan
    rdhx_base = rdhx(m_rack_base, T_ret, 0.25, EWT)
    pod_rdhx_W = P.RACKS_PER_POD * rdhx_base["W_parasitic"]
    print(f"\nWet check @EWT 7 C: dry={wet7['dry']:.0f}, SHR={wet7['SHR']:.3f}, "
          f"T_surf_min={wet7['T_surf_min']:.1f} C vs dew point {P.ROOM_DEWPOINT_C} C")
    print(f"Max feasible EWT at baseline duty: {ewmax_fine:.2f} C")
    print(f"RDHX per rack (m_air {m_rack_base:.3f} kg/s, m_w 0.25 kg/s, EWT {EWT}): "
          f"dP {rdhx_base['dP_door']:.2f} Pa, T_out {rdhx_base['T_air_out']:.2f} C, "
          f"Q {rdhx_base['Q']/1e3:.2f} kW, parasitic {rdhx_base['W_parasitic']:.2f} W; "
          f"pod {pod_rdhx_W:.0f} W vs IRC fans {podb['W_fan_pod']:.0f} W; door V for 30 Pa "
          f"{Vlim_txt} m3/s (dense {Vlim_dense:.2f})")
    print(f"RDHX geometry: A_a {gd['A_a']:.1f} m2, face {gd['A_face']:.2f} m2")

    js = dict(
        C_cal=cal["C_cal"], K_ext_Pa_per_m3s2=cal["K_ext"],
        validation_Q_err_mean_pct=100 * err_Q.mean(), validation_Q_err_max_pct=100 * np.abs(err_Q).max(),
        validation_dPw_err_mean_pct=100 * err_dP.mean(),
        validation_dPw_err_max_pct=100 * np.abs(err_dP).max(),
        fan_W_rated_model=fan_rated["W_fan"], fan_W_rated_datasheet=W_RATED_INPUT,
        eta_fan_calibrated=cal["eta_fan"], fan_heat_implied_by_rating=cal["fan_heat_implied"],
        geometry=dict(A_face=g["A_face"], A_air=g["A_a"], A_fin=g["A_fin"], A_water=g["A_w"],
                      A_min=g["A_min"], sigma=g["sigma"], Dh=g["Dh"], Dc=g["Dc"], rows=g["rows"],
                      FPI=g["FPI"], n_tubes=g["n_t"], n_circuits=g["n_circ"]),
        baseline_inputs=dict(m_air_pod=m_pod, m_air_unit=m_unit, T_ret=T_ret, T_supply=T_sup,
                             EWT=EWT, n_irc=P.N_IRC, T_dewpoint=P.ROOM_DEWPOINT_C),
        baseline_unit={k: base[k] for k, _ in keys},
        baseline_unit_extra=dict(T_w_out=base["T_w_out"], V_air=base["V_air"],
                                 speed_frac=base["speed_frac"], Q_net_sens=base["Q_net_sens"],
                                 X_dest_fan_system=base["X_dest_fan_system"],
                                 eps_counterflow_same_flows=cmp_cf["eps"]),
        baseline_pod=dict(Q=podb["Q_pod"], m_w=podb["m_w_pod"], W_fan=podb["W_fan_pod"],
                          W_pump_chw=podb["W_pump_chw_pod"], X_dest_coil=podb["X_dest_coil_pod"],
                          T_w_out=podb["T_w_out"]),
        EWT_max_feasible_baseline=ewmax_fine,
        wet_at_EWT7=dict(dry=wet7["dry"], SHR=wet7["SHR"], T_surf_min=wet7["T_surf_min"],
                         m_w=wet7["m_w"]),
        rdhx_baseline_rack=dict(m_air=m_rack_base, dP_door=rdhx_base["dP_door"],
                                T_air_out=rdhx_base["T_air_out"], Q=rdhx_base["Q"],
                                W_parasitic=rdhx_base["W_parasitic"], pod_W=pod_rdhx_W,
                                V_at_30Pa=(Vlim if np.isfinite(Vlim) else None), V_at_30Pa_dense=Vlim_dense,
                                dP_dense_at_base=rdhx(m_rack_base, T_ret, 0.25, EWT,
                                                      geom=RDHX_DENSE_GEOM)["dP_door"]),
    )
    with open(os.path.join(RES, "s4_baseline.json"), "w") as fh:
        json.dump(js, fh, indent=2, default=float)
    print("\nwrote results/s4_baseline.json and figures/s4_*.png")
    return js


if __name__ == "__main__":
    main()
