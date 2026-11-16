"""
S1 -- Processor direct-to-chip liquid-cooling loop for the PARAM Rudra pod.

Scope (see INTERFACE.md):
  coolant_props(T_C, fluid)                 thermophysical properties (PG25, EG25, water, Novec 7500)
  cold_plate(Q_W, m_dot, T_in_C, fluid)     one microchannel Cu cold plate on one Xeon Gold 6240R
  liquid_loop(Q_liq_node_W, m_dot_plate, T_in_C, n_nodes, fluid)
                                            TCS (secondary) loop: plates -> node hoses/QDs -> rack
                                            manifolds -> row piping -> CDU secondary side, pump power.

Conventions: SI units, temperatures in degC unless suffixed _K; exergy dead state params.T0_K.
All public functions return plain dicts of floats.

Key references (opened 2026-10-13, full list in results/s1_summary.md):
 [ARK]  Intel ARK, Xeon Gold 6240R: TDP 165 W, T_CASE 90 degC, package 76.0 x 56.5 mm, FCLGA3647.
 [SL78] Shah & London (1978), Laminar Flow Forced Convection in Ducts (fRe, Nu_H1 polynomials).
 [SK06] Steinke & Kandlikar (2006), J. Fluids Eng. -- K(inf), C(alpha) fits for Shah's apparent-f model.
 [LEE95] Lee, Song, Au & Moran (1995), "Constriction/spreading resistance model for electronics packaging".
 [CU72] Churchill & Usagi (1972) asymptotic blending.
 [KL]  Kays & London, Compact Heat Exchangers -- contraction/expansion loss coefficients.
 [N7500] 3M Novec 7500 product information (property correlations).
 [FITOK] FITOK UQD datasheet -- quick-disconnect Cv.
 [NSF-CDU] Heydari/Sammakia group, PG25 TCS test: liquid-to-liquid CDU differential ~8.7 psi (~60 kPa).
"""
import json
import math
import os

import numpy as np
import CoolProp.CoolProp as CP

import params as P

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")

# ============================================================================ constants / assumptions
CPU_TDP_W = P.CPU_TDP_W           # 165 W [ARK]
T_CASE_MAX_C = 90.0               # [ARK] T_CASE spec at TDP
T_J_MAX_C = 100.0                 # assumed TCC/DTS-max class value; Intel quotes 102 degC DTS_max for Gold 6244
                                  # (same Cascade Lake family) -- not listed on ARK for 6240R; 100 used (conservative)
CAPTURE_FRAC = 0.92               # fraction of CPU package power that leaves through the cold plate
                                  # (remainder: socket pins/substrate -> PCB -> air). Assumption 0.90-0.95.
ETA_PUMP_W2W = 0.55               # wire-to-water pump efficiency (motor x VFD x hydraulic), 0.5-0.6 typical
P_LOOP_PA = 2.0e5                 # TCS static pressure for property evaluation (2 bar abs)

K_CU = 390.0                      # W/m/K, C110/C101 copper (cold plate base + fins, IHS is Ni-plated Cu)
K_SI = 120.0                      # W/m/K, silicon at ~80 degC (148 at 300 K, falls ~T^-1.3)

# ---- package (LGA3647, Cascade Lake XCC die)
PKG = dict(
    A_die=694e-6,         # m^2  XCC die (~21.6 x 32.1 mm); Gold 6240R (24 C) ships on XCC -- WikiChip/AnandTech est.
    t_si=0.78e-3,         # m    die thickness (typical 775 um wafer, assumption)
    Rpp_tim1=0.05e-4,     # K m^2/W  soldered (indium) STIM, 0.05 K cm^2/W (assumption, solder TIM 0.02-0.08)
    t_ihs=3.0e-3,         # m    IHS copper thickness (assumption)
)
# 45 deg spreading footprint of the die at the IHS top surface (classic spreading-angle rule)
PKG["A_spread"] = (math.sqrt(PKG["A_die"]) + 2 * PKG["t_ihs"]) ** 2

# ---- TIM2 (IHS -> cold plate)
RPP_TIM2 = 0.08e-4                # K m^2/W = 0.08 K cm^2/W: production grease / PCM pad after bake-in
                                  # (Honeywell PTM7950 datasheet 0.04 K cm^2/W lab value; x2 for field/ageing)

# ---- microchannel cold plate (C-DAC style skived/microchannel Cu plate; geometry assumed)
GEOM = dict(
    w_c=0.30e-3,          # channel width, m
    w_f=0.20e-3,          # fin thickness, m
    H=2.5e-3,             # channel (fin) height, m
    L=50e-3,              # flow length, m
    W=50e-3,              # array width (normal to flow), m  -> N = W/(w_c+w_f) = 100 channels
    t_b=2.0e-3,           # base thickness under the fins, m
    d_port=6.0e-3,        # inlet/outlet barb ID, m
    K_ports=2.0,          # inlet turn into plenum (1.0) + outlet plenum exit (1.0), on port velocity
    K_c=0.5, K_e=1.0,     # plenum -> channel contraction / channel -> plenum expansion [KL], sigma << 1
)

# ---- secondary (TCS) loop components, referenced at 1.0 L/min per plate (parallel pair => 2 L/min/node)
LOOP = dict(
    V_ref_plate=1.0 / 60000.0,  # m^3/s per plate at which the reference dP values below apply
    dP_rack_manifold=10e3,      # Pa supply+return vertical manifold @ 72 L/min/rack (5-15 kPa typical)
    dP_row_piping=20e3,         # Pa row headers, valves, balancing, strainer (assumption)
    dP_cdu_sec=60e3,            # Pa CDU plate HX secondary + filter + valve: 8.7 psi set-point [NSF-CDU]
    Cv_qd=1.12,                 # UQD04 coupling flow coefficient (US gpm/psi^0.5) [FITOK]
    n_qd=2,                     # supply + return coupling per node
    d_hose=6.0e-3, L_hose=1.5,  # m, node hose (each of supply/return 0.75 m) EPDM 6 mm ID
    K_hose=3.0,                 # bends + barbs + Y-split inside node
)

# Novec 7500 correlations [N7500] (T in degC; viscosity correlation in K)
def _novec7500(T_C):
    """3M Novec 7500 single-phase dielectric (hydrofluoroether) properties from 3M product information."""
    T_K = T_C + 273.15
    rho = -2.0845 * T_C + 1665.8                          # kg/m^3
    cp = 1.4982 * T_C + 1091.0                            # J/kg/K
    k = 0.069 - 1.798e-4 * T_C + 4.24e-7 * T_C ** 2       # W/m/K
    Z = 10 ** (10 ** (11.843 - 5.0874 * math.log10(T_K)))  # ASTM D341-type fit
    nu_cSt = (Z - 0.7) - math.exp(-0.7487 - 3.295 * (Z - 0.7) + 0.6119 * (Z - 0.7) ** 2
                                  - 0.3193 * (Z - 0.7) ** 3)
    nu = nu_cSt * 1e-6
    return rho, cp, k, nu * rho


_COOLPROP = {"pg25": "INCOMP::APG-25%", "eg25": "INCOMP::AEG-25%", "water": "Water"}
FLUID_LABEL = {"pg25": "PG25 (25 % propylene glycol)", "eg25": "EG25 (25 % ethylene glycol)",
               "water": "Water", "novec7500": "Novec 7500 (dielectric HFE)"}


def coolant_props(T_C, fluid="PG25", P_Pa=P_LOOP_PA):
    """Liquid coolant properties at T_C [degC].

    fluid: 'PG25' (CoolProp INCOMP::APG-25%, mass-based), 'EG25' (INCOMP::AEG-25%), 'water', 'Novec7500'.
    Returns dict {rho [kg/m3], cp [J/kg/K], k [W/m/K], mu [Pa s], Pr [-], nu [m2/s]}.
    """
    f = fluid.lower().replace("-", "").replace(" ", "")
    if f in _COOLPROP:
        name, T_K = _COOLPROP[f], T_C + 273.15
        rho = CP.PropsSI("D", "T", T_K, "P", P_Pa, name)
        cp = CP.PropsSI("C", "T", T_K, "P", P_Pa, name)
        k = CP.PropsSI("L", "T", T_K, "P", P_Pa, name)
        mu = CP.PropsSI("V", "T", T_K, "P", P_Pa, name)
    elif f in ("novec7500", "novec", "hfe7500"):
        rho, cp, k, mu = _novec7500(T_C)
    else:
        raise ValueError(f"unknown fluid {fluid}")
    return dict(rho=float(rho), cp=float(cp), k=float(k), mu=float(mu),
                Pr=float(cp * mu / k), nu=float(mu / rho))


def figures_of_merit(T_C, fluid="PG25"):
    """Mouromtseff-type figures of merit (higher = better heat transfer at equal velocity & geometry).

    Mo_turb = rho^0.8 k^0.6 cp^0.4 / mu^0.4      (Dittus-Boelter, h ~ k Re^0.8 Pr^0.4 / D)
    Mo_lam  = (rho cp)^(1/3) k^(2/3)             (thermally developing laminar, Leveque: h ~ k (Re Pr D/L)^(1/3)/D)
    rho_cp  volumetric heat capacity (sets caloric resistance per unit volumetric flow).
    """
    p = coolant_props(T_C, fluid)
    return dict(Mo_turb=p["rho"] ** 0.8 * p["k"] ** 0.6 * p["cp"] ** 0.4 / p["mu"] ** 0.4,
                Mo_lam=(p["rho"] * p["cp"]) ** (1 / 3) * p["k"] ** (2 / 3),
                rho_cp=p["rho"] * p["cp"])


# ============================================================================ correlations
def _fRe_rect(a):
    """Fanning fRe, fully developed laminar flow, rectangular duct, aspect ratio a<=1 [SL78]."""
    return 24 * (1 - 1.3553 * a + 1.9467 * a ** 2 - 1.7012 * a ** 3 + 0.9564 * a ** 4 - 0.2537 * a ** 5)


def _NuH1_rect(a):
    """Fully developed Nu (H1, four walls heated), rectangular duct [SL78]."""
    return 8.235 * (1 - 2.0421 * a + 3.0853 * a ** 2 - 2.4765 * a ** 3 + 1.0578 * a ** 4 - 0.1861 * a ** 5)


def _fapp_Re(a, Lplus):
    """Apparent Fanning fRe incl. hydrodynamic entrance (Shah 1978 form, K(inf), C fits from [SK06])."""
    fRe = _fRe_rect(a)
    Kinf = 0.6796 + 1.2197 * a + 3.3089 * a ** 2 - 9.5921 * a ** 3 + 8.9089 * a ** 4 - 2.9959 * a ** 5
    C = (0.1811 + 4.3488 * a - 1.6027 * a ** 2) * 1e-4
    s = 3.44 / math.sqrt(Lplus)
    return s + (Kinf / (4 * Lplus) + fRe - s) / (1 + C / Lplus ** 2)


def _lee_spreading(A_s, A_p, t, k, h_eff):
    """Lee et al. (1995) closed-form spreading resistance of a centred source A_s on plate A_p.

    eps = r1/r2, tau = t/r2, Bi = h r2/k, lambda_c = pi + 1/(sqrt(pi) eps)
    Phi = (tanh(lc tau) + lc/Bi) / (1 + lc/Bi tanh(lc tau))
    psi_avg = 0.5 (1-eps)^1.5 Phi ; psi_max = eps tau/sqrt(pi) + (1-eps) Phi/sqrt(pi)  (the eps*tau term is 1-D)
    R_sp = psi/(k r1 sqrt(pi)).  Returns (R_sp_avg, R_sp_max_excl_1D).
    """
    r1, r2 = math.sqrt(A_s / math.pi), math.sqrt(A_p / math.pi)
    eps, tau, Bi = r1 / r2, t / r2, h_eff * r2 / k
    lc = math.pi + 1 / (math.sqrt(math.pi) * eps)
    th = math.tanh(lc * tau)
    Phi = (th + lc / Bi) / (1 + lc / Bi * th)
    psi_avg = 0.5 * (1 - eps) ** 1.5 * Phi
    psi_max = (1 - eps) * Phi / math.sqrt(math.pi)
    den = k * r1 * math.sqrt(math.pi)
    return psi_avg / den, psi_max / den


# ============================================================================ cold plate
def cold_plate(Q_W, m_dot, T_in_C, fluid="PG25", geom=None):
    """One microchannel copper cold plate on one Xeon Gold 6240R (LGA3647).

    Q_W    heat entering the cold plate from this CPU package [W]
    m_dot  coolant mass flow through this plate [kg/s]
    T_in_C plate inlet temperature [degC]

    Thermal chain (junction -> inlet):
      R_jc   = t_si/(k_si A_die) + R''_TIM1/A_die + t_ihs/(k_Cu sqrt(A_die A_spread))   (1-D, 45-deg spreading)
      R_tim2 = R''_TIM2 / A_spread           (flux under the die shadow governs T_case at IHS centre)
      R_base = t_b/(k_Cu A_p) + R_sp(Lee)    (A_spread -> fin footprint A_p, Bi from fluid side)
      R_fl   = 1/(m cp eps),  eps = 1 - exp(-NTU),  NTU = (h A_eff)/(m cp)   (isothermal-base HX form)
             = R_conv,eff + R_cal ; R_conv = 1/(h A_eff), R_cal = 1/(m cp) reported separately
    T_case = T_in + Q (R_fl + R_base + R_tim2);  T_j = T_case + Q R_jc ;  T_out = T_in + Q/(m cp)

    Pressure drop: dP = [4 f_app L/Dh + K_c + K_e] rho u_ch^2/2 + K_ports rho u_port^2/2 (+ Shah f_app).
    Exergy (dead state T0):
      X_heat = T0 [m cp ln(T_out/T_in) - Q/T_j]      (heat crossing T_j -> fluid; chip-to-fluid)
      X_fric = T0 m dP/(rho T_m)                      (incompressible-liquid friction)
      Be = X_heat / (X_heat + X_fric)
    """
    g = dict(GEOM if geom is None else geom)
    N = int(round(g["W"] / (g["w_c"] + g["w_f"])))
    a = min(g["w_c"], g["H"]) / max(g["w_c"], g["H"])
    Dh = 2 * g["w_c"] * g["H"] / (g["w_c"] + g["H"])
    A_flow = N * g["w_c"] * g["H"]
    A_p = g["W"] * g["L"]

    T_m = T_in_C
    for _ in range(3):                                   # props at mean bulk temperature (fixed point)
        p = coolant_props(T_m, fluid)
        mcp = m_dot * p["cp"]
        T_out = T_in_C + Q_W / mcp
        T_m = 0.5 * (T_in_C + T_out)
    rho, mu, k, cp, Pr = p["rho"], p["mu"], p["k"], p["cp"], p["Pr"]

    V_dot = m_dot / rho
    u = V_dot / A_flow
    Re = rho * u * Dh / mu
    # Heat transfer: Churchill-Usagi blend of fully developed Nu_H1 and thermally developing (Leveque, H bc)
    Gz = Re * Pr * Dh / g["L"]
    Nu_fd = _NuH1_rect(a)                                         # [SL78]
    Nu_dev = 1.953 * Gz ** (1 / 3)                                # [SL78] mean Nu, developing, H bc
    Nu = (Nu_fd ** 3 + Nu_dev ** 3) ** (1 / 3)                    # [CU72]
    h = Nu * k / Dh
    # Fin efficiency, straight fin, adiabatic tip (lid) -- Incropera eq. 3.89
    mfin = math.sqrt(2 * h / (K_CU * g["w_f"]))
    eta_f = math.tanh(mfin * g["H"]) / (mfin * g["H"])
    A_eff = N * g["L"] * (g["w_c"] + 2 * eta_f * g["H"])          # base between fins + 2 fin faces
    hA = h * A_eff
    NTU = hA / mcp
    eff = 1 - math.exp(-NTU)                                      # isothermal base, single stream
    R_fl = 1 / (mcp * eff)
    R_cal = 1 / mcp
    R_conv = 1 / hA
    # Base conduction + spreading (Lee 1995), Bi from fluid-side conductance spread over A_p
    h_eff = 1 / (R_fl * A_p)
    R_1d = g["t_b"] / (K_CU * A_p)
    R_sp_avg, _ = _lee_spreading(PKG["A_spread"], A_p, g["t_b"], K_CU, h_eff)
    R_base = R_1d + R_sp_avg
    R_tim2 = RPP_TIM2 / PKG["A_spread"]
    R_jc = (PKG["t_si"] / (K_SI * PKG["A_die"]) + PKG["Rpp_tim1"] / PKG["A_die"]
            + PKG["t_ihs"] / (K_CU * math.sqrt(PKG["A_die"] * PKG["A_spread"])))
    R_case_in = R_fl + R_base + R_tim2
    R_total = R_case_in + R_jc

    T_case = T_in_C + Q_W * R_case_in
    T_j = T_case + Q_W * R_jc
    T_base = T_in_C + Q_W * R_fl

    # Hydraulics
    Lplus = g["L"] / (Dh * Re)
    fapp = _fapp_Re(a, Lplus) / Re
    dP_ch = 4 * fapp * g["L"] / Dh * rho * u ** 2 / 2               # Darcy = 4 x Fanning
    dP_minor_ch = (g["K_c"] + g["K_e"]) * rho * u ** 2 / 2          # [KL]
    u_port = V_dot / (math.pi * g["d_port"] ** 2 / 4)
    dP_ports = g["K_ports"] * rho * u_port ** 2 / 2
    dP = dP_ch + dP_minor_ch + dP_ports
    W_pump_ideal = V_dot * dP

    # Exergy destruction
    T0 = P.T0_K
    Tin_K, Tout_K, Tj_K, Tc_K = T_in_C + 273.15, T_out + 273.15, T_j + 273.15, T_case + 273.15
    S_heat = mcp * math.log(Tout_K / Tin_K) - Q_W / Tj_K
    S_heat_case = mcp * math.log(Tout_K / Tin_K) - Q_W / Tc_K
    Tm_K = (Tout_K - Tin_K) / math.log(Tout_K / Tin_K)             # log-mean bulk temperature
    S_fric = m_dot * dP / (rho * Tm_K)
    X_heat, X_fric = T0 * S_heat, T0 * S_fric
    X_in = Q_W * (1 - T0 / Tj_K)                                    # exergy of heat released at T_j
    X_fluid = mcp * ((Tout_K - Tin_K) - T0 * math.log(Tout_K / Tin_K))   # exergy gained by coolant
    return dict(
        T_j=T_j, T_case=T_case, T_base=T_base, T_out=T_out, T_mean=T_m,
        R_total=R_total, R_case_in=R_case_in, R_jc=R_jc, R_tim2=R_tim2, R_base=R_base, R_1d=R_1d,
        R_sp=R_sp_avg, R_cond=R_jc + R_tim2 + R_base, R_fl=R_fl, R_conv=R_conv, R_cal=R_cal,
        R_conv_eff=R_fl - R_cal, h=h, Nu=Nu, Re=Re, Pr=Pr, Gz=Gz, eta_fin=eta_f, NTU=NTU, eps=eff,
        u_ch=u, V_dot_Lpm=V_dot * 60000.0, n_channels=float(N), Dh=Dh,
        dP=dP, dP_channels=dP_ch, dP_minor=dP_minor_ch + dP_ports, W_pump_ideal=W_pump_ideal,
        X_dest_total=X_heat + X_fric, X_dest_heat=X_heat, X_dest_heat_case=T0 * S_heat_case,
        X_dest_fric=X_fric, Bejan=X_heat / (X_heat + X_fric), X_in=X_in, X_fluid=X_fluid,
        T_case_margin=T_CASE_MAX_C - T_case, T_j_margin=T_J_MAX_C - T_j,
        feasible=float(T_case <= T_CASE_MAX_C and T_j <= T_J_MAX_C),
    )


# ============================================================================ loop
def default_Q_liq_node(load=1.0, capture=CAPTURE_FRAC):
    """Default liquid-captured heat per node [W]: 2 sockets x TDP x load x capture fraction.
    (Student 2's node_heat_breakdown()['Q_liquid'] supersedes this in integration.)"""
    return P.SOCKETS_PER_NODE * CPU_TDP_W * load * capture


def _dp_hose_qd(V_node, p):
    """Node-level hose + quick-disconnect pressure drop [Pa] at node flow V_node [m^3/s]."""
    gpm = V_node * 15850.3                                          # m^3/s -> US gpm
    dP_qd = LOOP["n_qd"] * (p["rho"] / 998.0) * (gpm / LOOP["Cv_qd"]) ** 2 * 6894.76   # dP = SG (Q/Cv)^2
    A = math.pi * LOOP["d_hose"] ** 2 / 4
    u = V_node / A
    Re = p["rho"] * u * LOOP["d_hose"] / p["mu"]
    fD = 64 / Re if Re < 2300 else 0.316 * Re ** -0.25             # Hagen-Poiseuille / Blasius
    dP_hose = (fD * LOOP["L_hose"] / LOOP["d_hose"] + LOOP["K_hose"]) * p["rho"] * u ** 2 / 2
    return dP_qd, dP_hose


def liquid_loop(Q_liq_node_W, m_dot_plate, T_in_C, n_nodes, fluid="PG25", config="parallel",
                eta_pump=ETA_PUMP_W2W):
    """Secondary (TCS) liquid loop serving n_nodes 2-socket nodes.

    Q_liq_node_W heat into liquid per node (both cold plates) [W]; split equally between plates.
    m_dot_plate  mass flow through EACH cold plate [kg/s].
    config 'parallel' (default): the two plates of a node in parallel (node flow 2 m_dot_plate), equal T_in.
           'series': plate 2 is fed by plate-1 outlet (node flow = m_dot_plate).
    Loop dP (critical path) = node (plates + hoses + 2 QDs) + rack manifold + row piping + CDU secondary,
    with rack/row/CDU terms scaled as (V/V_ref)^2 from reference values at 1 L/min per plate (parallel).
    W_pump (electrical) = V_total dP_loop / eta_w2w.
    X_dest_pumping = W_pump - W_hyd (1 - T0/T_m)  (all shaft work is dissipated; tiny exergy of friction heat
    returned to the fluid is credited).
    """
    Qp = Q_liq_node_W / 2.0
    if config == "parallel":
        cp1 = cold_plate(Qp, m_dot_plate, T_in_C, fluid)
        plates = [cp1, cp1]
        m_node = 2 * m_dot_plate
        dP_plates = cp1["dP"]
    elif config == "series":
        cp1 = cold_plate(Qp, m_dot_plate, T_in_C, fluid)
        cp2 = cold_plate(Qp, m_dot_plate, cp1["T_out"], fluid)
        plates = [cp1, cp2]
        m_node = m_dot_plate
        dP_plates = cp1["dP"] + cp2["dP"]
    else:
        raise ValueError(config)
    p = coolant_props(T_in_C, fluid)
    V_node = m_node / p["rho"]
    dP_qd, dP_hose = _dp_hose_qd(V_node, p)
    scale = (V_node / (2 * LOOP["V_ref_plate"])) ** 2                 # referenced to 2 L/min/node
    dP_rack = LOOP["dP_rack_manifold"] * scale
    dP_row = LOOP["dP_row_piping"] * scale
    dP_cdu = LOOP["dP_cdu_sec"] * scale
    dP_node = dP_plates + dP_qd + dP_hose
    dP_loop = dP_node + dP_rack + dP_row + dP_cdu
    m_tot = m_node * n_nodes
    V_tot = V_node * n_nodes
    W_hyd = V_tot * dP_loop
    W_pump = W_hyd / eta_pump
    Q_liq = Q_liq_node_W * n_nodes
    T_return = T_in_C + (Q_liq + W_hyd) / (m_tot * p["cp"])             # 1st law, pump work ends as heat
    Tm_K = 0.5 * (T_in_C + T_return) + 273.15
    X_cp = n_nodes * sum(pl["X_dest_total"] for pl in plates)
    X_pump = W_pump - W_hyd * (1 - P.T0_K / Tm_K)
    T_j_max = max(pl["T_j"] for pl in plates)
    T_case_max = max(pl["T_case"] for pl in plates)
    return dict(
        Q_liq=Q_liq, T_return=T_return, m_dot_total=m_tot, V_dot_total_Lpm=V_tot * 60000.0,
        dP_loop=dP_loop, dP_node=dP_node, dP_plates=dP_plates, dP_qd=dP_qd, dP_hose=dP_hose,
        dP_rack=dP_rack, dP_row=dP_row, dP_cdu=dP_cdu, W_hyd=W_hyd, W_pump=W_pump,
        X_dest_coldplates=X_cp, X_dest_coldplates_heat=n_nodes * sum(pl["X_dest_heat"] for pl in plates),
        X_dest_pumping=X_pump, X_fluid_out=n_nodes * sum(pl["X_fluid"] for pl in plates),
        T_j_max=T_j_max, T_case_max=T_case_max, R_total_plate=plates[0]["R_total"],
        feasible=float(T_case_max <= T_CASE_MAX_C and T_j_max <= T_J_MAX_C),
    )


# ============================================================================ self-test / figures
def lpm_to_kg_s(lpm, T_C, fluid="PG25"):
    return lpm / 60000.0 * coolant_props(T_C, fluid)["rho"]


def _style(plt):
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "font.size": 9, "axes.titlesize": 10, "lines.linewidth": 2.0, "legend.frameon": False})


COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FLUIDS = ["water", "PG25", "EG25", "Novec7500"]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    B = P.BASE
    fl = "PG25"
    T_in = B["T_liq_in"]
    m_b = B["m_liq_plate"]
    Q_plate = CPU_TDP_W * CAPTURE_FRAC
    Q_node = default_Q_liq_node()

    # ---------------- 1. coolant properties
    Ts = np.linspace(15, 60, 46)
    props = {f: [coolant_props(t, f) for t in Ts] for f in FLUIDS}
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.2))
    keys = [("rho", "Density ρ [kg/m³]", False), ("cp", "Specific heat c_p [J/(kg·K)]", False),
            ("k", "Conductivity k [W/(m·K)]", False), ("mu", "Viscosity μ [mPa·s]", True),
            ("Pr", "Prandtl number Pr [–]", True), ("Mo", "Mouromtseff Mo (laminar), rel. to water [–]", False)]
    for i, (key, lab, logy) in enumerate(keys):
        a_ = ax.flat[i]
        for j, f in enumerate(FLUIDS):
            if key == "Mo":
                y = [figures_of_merit(t, f)["Mo_lam"] / figures_of_merit(t, "water")["Mo_lam"] for t in Ts]
            else:
                y = [pp[key] * (1e3 if key == "mu" else 1) for pp in props[f]]
            a_.plot(Ts, y, color=COLS[j], label=FLUID_LABEL[f.lower()])
        if logy:
            a_.set_yscale("log")
        a_.set_xlabel("Temperature [°C]")
        a_.set_ylabel(lab)
    ax.flat[0].legend(fontsize=7.5, loc="center left")
    mo = {f: figures_of_merit(35.0, f)["Mo_lam"] / figures_of_merit(35.0, "water")["Mo_lam"] for f in FLUIDS}
    fig.suptitle(f"Coolant properties 15–60 °C: at 35 °C PG25 keeps {mo['PG25']*100:.0f} % of water's laminar "
                 f"h-merit; Novec 7500 only {mo['Novec7500']*100:.0f} % (k ≈ 0.065 W/m·K)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "s1_coolant_props.png"), dpi=200)
    plt.close(fig)

    fom_rows = []
    for f in FLUIDS:
        for t in (20.0, 35.0, 50.0):
            pp, mo = coolant_props(t, f), figures_of_merit(t, f)
            mw = figures_of_merit(t, "water")
            fom_rows.append(dict(fluid=f, T=t, **pp, Mo_lam_rel=mo["Mo_lam"] / mw["Mo_lam"],
                                 Mo_turb_rel=mo["Mo_turb"] / mw["Mo_turb"], rho_cp_rel=mo["rho_cp"] / mw["rho_cp"]))

    # ---------------- baseline plate & fluid comparison (same volumetric flow 1 L/min)
    base = cold_plate(Q_plate, m_b, T_in, fl)
    fl_cmp = {}
    for f in FLUIDS:
        r = cold_plate(Q_plate, lpm_to_kg_s(1.0, T_in, f), T_in, f)
        fl_cmp[f] = {k_: r[k_] for k_ in ("R_total", "R_case_in", "T_case", "T_j", "dP", "W_pump_ideal",
                                          "Re", "h", "X_dest_total")}

    # ---------------- 2. maps vs flow & inlet T
    lpm = np.linspace(0.2, 3.0, 57)
    Tin = np.linspace(17, 50, 34)
    TJ, TC, XT, BE, QMAX = (np.zeros((len(Tin), len(lpm))) for _ in range(5))
    for i, t in enumerate(Tin):
        for j, v in enumerate(lpm):
            r = cold_plate(Q_plate, lpm_to_kg_s(v, t), t, fl)
            TJ[i, j], TC[i, j], XT[i, j], BE[i, j] = r["T_j"], r["T_case"], r["X_dest_total"], r["Bejan"]
            QMAX[i, j] = (T_CASE_MAX_C - t) / r["R_case_in"]   # linear in Q at fixed R (R weakly Q-dependent)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4))
    lv = np.arange(20, 72, 4)
    cs = ax[0].contourf(lpm, Tin, TC, levels=lv, cmap="Blues")
    ax[0].contour(lpm, Tin, TC, levels=lv, colors="#52514e", linewidths=0.4)
    fig.colorbar(cs, ax=ax[0], label="T_case [°C]")
    ax[0].set_title(f"T_case at Q = {Q_plate:.0f} W (max {TC.max():.1f} °C < 90 °C spec)")
    cs = ax[1].contourf(lpm, Tin, TJ, levels=np.arange(20, 80, 4), cmap="Oranges")
    fig.colorbar(cs, ax=ax[1], label="T_j [°C]")
    c2 = ax[1].contour(lpm, Tin, TJ, levels=[60, 70], colors="#0b0b0b", linewidths=1.0)
    ax[1].clabel(c2, fmt="%d °C")
    ax[1].set_title(f"Junction temperature (max {TJ.max():.1f} °C vs T_j,max {T_J_MAX_C:.0f} °C)")
    cs = ax[2].contourf(lpm, Tin, QMAX, levels=np.arange(200, 2200, 200), cmap="Greens", extend="max")
    fig.colorbar(cs, ax=ax[2], label="Q allowed at T_case = 90 °C [W]")
    c3 = ax[2].contour(lpm, Tin, QMAX, levels=[CPU_TDP_W, 2 * CPU_TDP_W, 3 * CPU_TDP_W],
                       colors="#e34948", linewidths=1.4)
    ax[2].clabel(c3, fmt={CPU_TDP_W: "1×TDP", 2 * CPU_TDP_W: "2×TDP", 3 * CPU_TDP_W: "3×TDP"})
    ax[2].set_title(f"Plate heat that would reach T_case = 90 °C (min {QMAX.min():.0f} W ≈ {QMAX.min()/CPU_TDP_W:.1f}×TDP)")
    for a_ in ax:
        a_.set_xlabel("Flow per plate [L/min]")
        a_.set_ylabel("Inlet temperature [°C]")
        a_.grid(False)
    fig.suptitle("Xeon Gold 6240R cold plate (PG25): T_case limit is never reached for 17–50 °C inlet; "
                 "≥0.4 L/min the map is set by T_in, not flow", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "s1_temperature_map.png"), dpi=200)
    plt.close(fig)

    # ---------------- 3. resistance breakdown vs flow
    rs = [cold_plate(Q_plate, lpm_to_kg_s(v, T_in), T_in, fl) for v in lpm]
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.2))
    comps = [("R_jc", "Junction→case (die, STIM, IHS)"), ("R_tim2", "TIM2"), ("R_base", "Base + spreading"),
             ("R_conv_eff", "Convection (ε-form)"), ("R_cal", "Caloric 1/(ṁc_p)"), ("R_total", "Total j→inlet")]
    for j, (key, lab) in enumerate(comps):
        ax.plot(lpm, [r[key] for r in rs], color=COLS[j] if key != "R_total" else "#0b0b0b",
                lw=2.4 if key == "R_total" else 1.8, label=lab)
    ax.axvline(1.0, color="#9a9994", lw=0.8, ls="--")
    ax.set_xlabel("Flow per plate [L/min]")
    ax.set_ylabel("Thermal resistance [K/W]")
    ax.set_ylim(0, 0.16)
    ax.legend(fontsize=7.5)
    ax.set_title(f"Above ~1 L/min the fixed conduction chain dominates; R_total(1 L/min) = "
                 f"{base['R_total']:.3f} K/W", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "s1_resistance_vs_flow.png"), dpi=200)
    plt.close(fig)

    # ---------------- 4. pressure drop & pumping
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for j, t in enumerate((20.0, 35.0, 50.0)):
        rr = [cold_plate(Q_plate, lpm_to_kg_s(v, t), t, fl) for v in lpm]
        ax[0].plot(lpm, [r["dP"] / 1e3 for r in rr], color=COLS[j], label=f"T_in = {t:.0f} °C")
        ax[1].plot(lpm, [r["W_pump_ideal"] for r in rr], color=COLS[j], label=f"T_in = {t:.0f} °C")
    ax[0].set_ylabel("Cold-plate ΔP [kPa]")
    ax[1].set_ylabel("Ideal pumping power per plate [W]")
    ax[1].set_yscale("log")
    for a_ in ax:
        a_.set_xlabel("Flow per plate [L/min]")
        a_.legend(fontsize=8)
    b1 = base
    ax[0].set_title(f"ΔP ≈ linear (laminar) + quadratic (ports): {b1['dP']/1e3:.1f} kPa at 1 L/min, 35 °C",
                    fontsize=9.5)
    ax[1].set_title("Plate hydraulic power stays < 1 W (≪ 0.5 % of 152 W)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "s1_pressure_pumping.png"), dpi=200)
    plt.close(fig)

    # ---------------- 5. exergy
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))
    cs = ax[0].contourf(lpm, Tin, XT, levels=14, cmap="Purples")
    fig.colorbar(cs, ax=ax[0], label="X_dest per plate [W]")
    ax[0].set_title("Total exergy destruction per plate")
    cs = ax[1].contourf(lpm, Tin, BE, levels=np.linspace(np.floor(BE.min() * 50) / 50, 1.0, 11), cmap="Greys")
    fig.colorbar(cs, ax=ax[1], label="Bejan number [–]")
    ax[1].set_title(f"Bejan number {BE.min():.2f}–{BE.max():.4f}: thermal-dominated")
    for a_ in ax[:2]:
        a_.set_xlabel("Flow per plate [L/min]")
        a_.set_ylabel("Inlet temperature [°C]")
        a_.grid(False)
    ax[2].plot(lpm, [r["X_dest_heat"] for r in rs], color=COLS[0], label="Heat transfer (T_j → fluid)")
    ax[2].plot(lpm, [r["X_dest_heat_case"] for r in rs], color=COLS[2], label="Heat transfer (T_case → fluid)")
    ax[2].plot(lpm, [r["X_dest_fric"] for r in rs], color=COLS[1], label="Fluid friction")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("Flow per plate [L/min]")
    ax[2].set_ylabel("Exergy destruction per plate [W]")
    ax[2].set_title(f"T_in = {T_in:.0f} °C: friction ≤ {rs[-1]['X_dest_fric']:.2f} W vs heat ≈ "
                    f"{rs[-1]['X_dest_heat']:.1f} W", fontsize=9.5)
    ax[2].legend(fontsize=7.5)
    fig.suptitle("Cold-plate exergy destruction is mostly thermal (Be > 0.98 below 1 L/min); it falls with flow (smaller ΔT) "
                 "and with inlet temperature (higher T level)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "s1_exergy_map.png"), dpi=200)
    plt.close(fig)

    # ---------------- 6. loop
    n = P.NODES_PER_POD
    lpm2 = np.linspace(0.3, 3.0, 28)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    loops = {}
    for j, cfg in enumerate(("parallel", "series")):
        Ls = [liquid_loop(Q_node, lpm_to_kg_s(v, T_in), T_in, n, fl, cfg) for v in lpm2]
        loops[cfg] = Ls
        ax[0].plot(lpm2, [L["W_pump"] / 1e3 for L in Ls], color=COLS[j], label=f"{cfg} plates")
        ax[1].plot(lpm2, [L["T_return"] for L in Ls], color=COLS[j], label=f"{cfg} plates")
        ax[2].plot(lpm2, [L["T_case_max"] for L in Ls], color=COLS[j], label=f"{cfg} plates")
    ax[0].set_ylabel("Pod TCS pump electrical power [kW]")
    ax[0].set_yscale("log")
    ax[1].set_ylabel("Loop return temperature [°C]")
    ax[2].set_ylabel("Hottest T_case [°C]")
    for a_ in ax:
        a_.set_xlabel("Flow per plate [L/min]")
        a_.legend(fontsize=8)
    Lb = liquid_loop(Q_node, m_b, T_in, n, fl, "parallel")
    Ls_b = liquid_loop(Q_node, m_b, T_in, n, fl, "series")
    ax[0].set_title(f"Pump power ∝ V³: {Lb['W_pump']/1e3:.2f} kW at 1 L/min (parallel)", fontsize=9.5)
    ax[1].set_title("Return T rises fast below 0.5 L/min", fontsize=9.5)
    ax[2].set_title(f"Series: 2nd CPU +{Ls_b['T_case_max']-Lb['T_case_max']:.1f} K at 1 L/min, but half the loop flow",
                    fontsize=9.5)
    fig.suptitle(f"576-node pod TCS loop (PG25, T_in = {T_in:.0f} °C, {Q_node:.0f} W/node to liquid)",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "s1_loop.png"), dpi=200)
    plt.close(fig)

    # ---------------- print baseline
    print("=" * 72)
    print(f"S1 baseline: {P.CPU_MODEL}, Q_plate = {Q_plate:.1f} W (TDP {CPU_TDP_W:.0f} W x capture "
          f"{CAPTURE_FRAC}), fluid {fl}, T_in = {T_in} C, m_dot = {m_b*1e3:.2f} g/s ({base['V_dot_Lpm']:.2f} L/min)")
    print("-" * 72)
    rows = [("T_j [C]", base["T_j"]), ("T_case [C]", base["T_case"]), ("T_out [C]", base["T_out"]),
            ("R_total j->inlet [K/W]", base["R_total"]), ("R_case->inlet [K/W]", base["R_case_in"]),
            ("  R_jc", base["R_jc"]), ("  R_TIM2", base["R_tim2"]), ("  R_base+spread", base["R_base"]),
            ("  R_fluid=1/(m cp eps)", base["R_fl"]), ("  R_conv=1/(hA)", base["R_conv"]),
            ("  R_cal=1/(m cp)", base["R_cal"]), ("Re", base["Re"]), ("Nu", base["Nu"]),
            ("h [W/m2K]", base["h"]), ("fin efficiency", base["eta_fin"]), ("NTU", base["NTU"]),
            ("dP plate [kPa]", base["dP"] / 1e3), ("W_pump_ideal plate [W]", base["W_pump_ideal"]),
            ("X_dest heat [W]", base["X_dest_heat"]), ("X_dest fric [W]", base["X_dest_fric"]),
            ("Bejan [-]", base["Bejan"])]
    for k_, v in rows:
        print(f"{k_:28s} {v:12.5g}")
    print("-" * 72)
    print(f"Pod loop ({n} nodes, parallel / series):")
    for k_ in ("Q_liq", "V_dot_total_Lpm", "T_return", "dP_plates", "dP_node", "dP_qd", "dP_hose", "dP_rack",
               "dP_row", "dP_cdu", "dP_loop", "W_hyd", "W_pump", "X_dest_coldplates", "X_dest_pumping",
               "X_fluid_out", "T_case_max", "T_j_max", "feasible"):
        print(f"{k_:28s} {Lb[k_]:12.5g} {Ls_b[k_]:12.5g}")
    print("-" * 72)
    print("Fluid comparison at 1.0 L/min, 35 C:")
    for f, d in fl_cmp.items():
        print(f"  {f:10s} R_tot={d['R_total']:.4f} K/W  T_case={d['T_case']:.1f}  dP={d['dP']/1e3:.2f} kPa "
              f"Re={d['Re']:.0f}")

    out = dict(
        inputs=dict(Q_plate_W=Q_plate, capture_frac=CAPTURE_FRAC, TDP_W=CPU_TDP_W, fluid=fl, T_in_C=T_in,
                    m_dot_plate_kg_s=m_b, n_nodes=n, T_case_max_C=T_CASE_MAX_C, T_j_max_C=T_J_MAX_C,
                    eta_pump=ETA_PUMP_W2W, geom=GEOM, pkg=PKG, Rpp_tim2_Km2_W=RPP_TIM2, loop=LOOP),
        cold_plate={k_: v for k_, v in base.items()},
        loop_parallel=Lb, loop_series=Ls_b,
        fluid_comparison_1Lpm_35C=fl_cmp,
        map_ranges=dict(T_case_max_over_map=float(TC.max()), T_j_max_over_map=float(TJ.max()),
                        Bejan_min_over_map=float(BE.min()), Q_allow_min_W=float(QMAX.min())),
        coolant_table=fom_rows,
    )
    with open(os.path.join(RES, "s1_baseline.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    return out


if __name__ == "__main__":
    main()
