"""
S2 -- Air-side board components and server fans for the PARAM Rudra hybrid-cooled pod.

Scope (INTERFACE.md):
  node_heat_breakdown(load=1.0, T_in_C=None, capture=CAPTURE_FRAC)
        secondary (air-side) heat loads of one 1U Rudra node, PSU loss, fans (computed),
        cold-plate capture fraction -> Q_node_total, Q_cpu_package, Q_liquid, Q_air.
  server_air(T_in_C, Q_air_W, V_dot=None, load=1.0, ...)
        1U chassis air-side model: fan curve x system curve, serial-zone air temperatures,
        component case/junction temperatures via h*A, minimum-speed fan control law, exergy.
  pod_air(T_in_C, load=1.0)
        576 identical nodes (params.NODES_PER_POD).

Conventions: SI units, temperatures in degC unless suffixed _K, exergy dead state params.T0_K.
Every public function returns a plain dict of floats.

Sources (all opened 2026-10-13; full list and URLs in results/s2_summary.md):
 [ARK-C621] Intel ARK, C621 chipset: TDP 15 W ("max TDP is worst case, actual may be lower if not all I/Os used").
 [MU-RDIMM] Micron MTA18ASF2G72PZ 16 GB 1Rx4 DDR4 RDIMM datasheet: IDD3N 0.720 A, IDD4R 2.556 A,
            IDD4W 2.214 A, IDD5R 0.882 A (2933 MT/s, E-die); T_OPER 0..95 degC, IDD valid 0..85 degC.
 [ADV-RDIMM] Advantech AQD-D4U16R32 DDR4 RDIMM datasheet: T_C 85..95 degC requires 2x refresh (3.9 us).
 [STH-DIMM] ServeTheHome "DDR4 DIMMs and system power consumption": ~2 W per 8 GB RDIMM at the wall.
 [CX6]  NVIDIA ConnectX-6 VPI user manual, specs: MCX653105A-HDAT typical 19.3 W; airflow 350 LFM @55 degC
        (heatsink->port), 250 LFM @35 degC (port->heatsink); operating 0..55 degC.
 [CX7]  NVIDIA ConnectX-7 user manual, specs: MCX75310AAS-NEAT (1-port NDR OSFP) typical 24.9 W; 0..55 degC.
 [S3610] Intel SSD DC S3610 800 GB spec: write 6.3 W, read 2.9 W, idle 0.61 W; case 0..70 degC.
 [M5300] Micron 5300 product brief: read <3 W, write <3.9 W, 0..70 degC.
 [80PLUS] Wikipedia "80 Plus", 230 V internal-redundant table: Platinum 88/90/94/91 % at 10/20/50/100 % load.
 [SM-PSU] Supermicro PWS-751P-1R: 750 W, Platinum, 94 % peak.
 [SM-1029] Supermicro SYS-1029U-E1CRTP (C621, 2S LGA3647, 1U): 8 x 40x56 mm PWM fans, 2 x 750 W Platinum.
 [DELTA] Delta GFB 40x40x56 mm datasheet: GFB0412EHS 12 V, 1.35 A, 16.2 W, 31.18 CFM free air,
         56.93 mmH2O (558 Pa) max static pressure.
 [ROT]  Comair Rotron "Establishing cooling requirements: air flow vs pressure": system impedance
        P = K Q^n with n = 2 (turbulent); one operating point; flow ~ speed, power ~ speed^3.
 [TEC]  tec-science, forced flow over plates: Nu_lam = 0.664 Re^0.5 Pr^(1/3) (Re < 5e5, 0.6 < Pr < 2000).
 [BEL]  Bel Fuse VRP4-80A1Ax 12 V POL datasheet: efficiency 86-90 % at 1.8 V, 81-86 % at 1.2 V (full load).
 [TI-PS] TI CSD95490Q5MC smart power stage: >95 % stage efficiency at 30 A; operating -55..150 degC.
 [LEN-DWC] Lenovo Press LP0636 (SD650 DWC): 85-90 % of node heat removed by water (CPU+DIMM+drives plates).
 [NREL] NREL/Sandia Aquarius test: cold plates captured up to 98 % of compute-node heat.
 [SPEC] SPECpower_ssj2008 result 20220523-01177 (2x Xeon Gold 6226R): 377 W @100 %, 96.6 W active idle.
"""
import functools
import json
import math
import os

import numpy as np
import CoolProp.CoolProp as CP

import params as P

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")

# ============================================================================ node power assumptions
CAPTURE_FRAC = 0.92        # fraction of CPU PACKAGE power leaving through the cold plate. Same value as S1
                           # (s1_liquid.CAPTURE_FRAC) so the ledger closes. Remaining 8 % leaves the package
                           # through LGA pins/substrate into the PCB and the socket/loading-plate surfaces and
                           # is picked up by the air. Bounds: Lenovo SD650 85-90 % of NODE heat with DIMMs on
                           # water too [LEN-DWC]; NREL Aquarius up to 98 % of node heat [NREL]. For the CPU
                           # alone 0.90-0.95 is the usual range; 0.92 is mid-range.
CPU_IDLE_FRAC = 0.30       # CPU package power at 0 % utilisation as fraction of TDP (Cascade Lake with C-states;
                           # cf. [SPEC] 2x6226R server: active idle 96.6 W = 26 % of 377 W full-load wall power).
N_DIMM = 12                # 12 x 16 GB RDIMM (6 channels/socket x 1 DPC)
# DIMM power from Micron IDD (module-level currents) [MU-RDIMM] at VDD = 1.2 V:
#   standby IDD3N 0.72 A -> 0.86 W; read IDD4R 2.556 A -> 3.07 W; write IDD4W 2.214 A -> 2.66 W.
#   Add RCD (register) ~0.6 W idle / 0.9 W active, VPP (2.5 V) ~0.15 W and refresh ~0.15 W (assumption).
#   Active mix (60 % read / 40 % write, ~85 % bus duty under STREAM/HPC) ->
#   0.85*(0.6*3.07+0.4*2.66) + 0.15*0.86 + 0.9 + 0.15 + 0.15 = 3.8 W ; idle 0.86+0.6+0.15+0.15 = 1.8 W.
#   Cross-check [STH-DIMM]: ~2 W per 8 GB RDIMM at the wall (lighter load, half the dies).
P_DIMM_IDLE = 1.8          # W per 16 GB RDIMM
P_DIMM_ACT = 3.8           # W per 16 GB RDIMM at full memory load
ETA_VR_CPU = 0.90          # VR13 VCCIN (12 V -> 1.8 V; FIVR loss already inside TDP). [BEL] gives 86-90 %
                           # for 12->1.8 V POL modules; multiphase power stages [TI-PS] reach >95 % per stage,
                           # system incl. inductor/PCB ~ 90 %. -> loss = P_out (1/eta - 1)
ETA_VR_MEM = 0.88          # VDDQ 1.2 V rail; [BEL] 81-86 % for 1.2 V modules, multiphase better -> 0.88
P_VR_FIXED = 1.5           # W per socket VR fixed loss (controller, gate drive, quiescent) -- assumption
P_PCH_FULL = 12.0          # W; ARK TDP 15 W is worst-case with all I/O [ARK-C621]; HPC node uses few PCH I/Os
P_PCH_IDLE = 8.0           # W  (assumption, ~55 % TDP)
P_NIC_TYP = 24.9           # W  ConnectX-7 1-port NDR OSFP, passive cable, typical [CX7] (HDR CX-6: 19.3 W [CX6])
NIC_ASIC_FRAC = 0.80       # share of card power dissipated in the ASIC under the heat sink (assumption)
P_NIC_IDLE_FRAC = 0.80     # SerDes/PHY dominate -> weak load dependence (assumption)
P_SSD_IDLE = 0.61          # W  [S3610] 800 GB SATA idle
P_SSD_ACT = 3.0            # W  mixed read/write average (read 2.9, write 6.3 W [S3610]; HPC node SSD is OS/scratch)
P_BMC = 4.0                # W  BMC SoC (AST2500-class) + its DRAM + mgmt PHY + standby rails -- assumption
P_MISC = 8.0               # W  clocks, CPLD, USB/VGA, PCIe retimer/riser, 3.3/5 V POLs, LEDs -- assumption
PSU_RATED = 750.0          # W per PSU [SM-PSU]; 1+1 redundant, current sharing (each PSU carries half)
N_PSU_SHARING = 2
# 80 PLUS Platinum, 230 V internal redundant (minimum required values) [80PLUS]; actual units exceed these
# slightly ([SM-PSU] 94 % peak). Linear interpolation in load fraction.
PSU_EFF_TABLE = ([0.10, 0.20, 0.50, 1.00], [0.88, 0.90, 0.94, 0.91])
PSU_EFF_TITANIUM = ([0.10, 0.20, 0.50, 1.00], [0.90, 0.94, 0.96, 0.91])

# ============================================================================ chassis / fan assumptions
N_FANS = 8                 # [SM-1029]: 8 x 40x56 mm fans in a comparable C621 1U 2S node
FAN_V0 = 31.18 * 4.719474e-4   # m^3/s free-air flow of one GFB0412EHS at 100 % speed [DELTA]
FAN_DP0 = 56.93 * 9.80665      # Pa max static pressure (shut-off) at 100 % speed [DELTA] = 558 Pa
FAN_P_RATED = 16.2         # W electrical at rated speed [DELTA]
FAN_SHAPE_A = 0.5          # dP/dP0 = 1 - a x - (1-a) x^2, x = V/V_free: concave curve typical of dual-rotor
                           # axial fans (datasheet P-Q chart not machine-readable; shape is an assumption,
                           # end points are datasheet values)
FAN_P_FIXED = 0.03         # fraction of rated power drawn regardless of speed (driver/hall/PWM electronics)
FAN_FLOOR = 0.25           # minimum fan speed fraction (BMC fan tables typically floor at 20-30 %)
# System impedance dP = k V^2 [ROT]. Calibrated: 80 Pa at 60 CFM per node. Justification: a 1U chassis with
# drive bay, fan wall, DIMM banks, riser and PSU has ~0.3-0.5 inH2O at 60-80 CFM when it carries two 1U CPU
# heat sinks; here the CPU heat sinks are replaced by low-profile cold plates, so the lower end
# (80 Pa = 0.32 inH2O) is used. Engineering estimate -- flagged in summary.
V_NOM = 60.0 * 4.719474e-4     # m^3/s
DP_NOM = 80.0                  # Pa
K_SYS = DP_NOM / V_NOM ** 2    # Pa/(m^3/s)^2

W_INT, H_INT = 0.430, 0.040    # m internal width / height of a 1U 19" chassis
POROSITY = 0.60                # free-flow fraction of the cross-section in the board region (assumption)
A_FLOW = W_INT * H_INT * POROSITY

# component geometry (heat-transfer surfaces); h from laminar flat plate [TEC] over flow length L
GEO = dict(
    # 2.5" 7 mm drive: top+bottom faces 100 x 70 mm, L = 0.10 m along flow
    ssd=dict(A=2 * 0.100 * 0.070, L=0.100, eta=1.0, n=1),
    # RDIMM: 133.35 x 31.25 mm, both faces, flow along 133 mm (DIMMs parallel to airflow)
    dimm=dict(A=2 * 0.13335 * 0.03125, L=0.13335, eta=1.0, n=N_DIMM),
    # VRM bank per socket: finned sink 70 mm wide, 30 mm deep (flow), 14 fins 18 mm tall, plus heat spread
    # into the PCB copper planes (power stages are designed to sink most heat through their thermal pad);
    # effective PCB area 2 faces x 60 x 80 mm at 50 % plane-fin efficiency (assumption)
    vrm=dict(A=2 * 14 * 0.030 * 0.018 + 0.070 * 0.030 + 0.5 * 2 * 0.060 * 0.080, L=0.030, eta=0.90, n=2,
             n_phase=6, R_jc=2.0),
    # PCH: 45 x 45 mm sink, 12 fins 25 mm tall (1U), flow length 45 mm
    pch=dict(A=2 * 12 * 0.045 * 0.025 + 0.045 * 0.045, L=0.045, eta=0.85, n=1, R_jc=0.6),
    # HCA ASIC: ConnectX-class HHHL card sink ~55 x 55 mm, 20 fins 15 mm tall (fits a 1U riser);
    # only the ASIC share of card power (NIC_ASIC_FRAC) goes through the sink, rest = PHY/regulators on the card
    nic=dict(A=2 * 20 * 0.055 * 0.015 + 0.055 * 0.055, L=0.055, eta=0.85, n=1, R_jc=0.3),
)
# DIMM channel: 1 DPC on a 2 DPC board -> every other slot empty, slot pitch 8 mm -> 16 mm between populated
# modules, module thickness incl. DRAM packages ~4 mm -> 12 mm x 31 mm gap shared by two half-faces.
A_DIMM_CH = 0.012 * 0.03125    # m^2 flow area associated with one DIMM
DIMM_HOTSPOT = 0.453 / 0.664   # h(x=L, UHF) / h_avg(isothermal) -- trailing-edge DRAM hotspot factor

LIMITS = dict(                # component limits used by the control law
    dimm_case=85.0,           # degC JEDEC normal refresh (85..95 needs 2x refresh) [ADV-RDIMM], [MU-RDIMM]
    ssd_case=70.0,            # degC [S3610], [M5300]
    vrm_junction=110.0,       # degC design target for power stages (abs. max 150 degC [TI-PS]) -- assumption
    pch_case=100.0,           # degC assumed (Intel PCH TDG not public on ARK) -- UNVERIFIED, flagged
    nic_air=55.0,             # degC max operating ambient at the card [CX6], [CX7] (also optics/cables)
    nic_case=105.0,           # degC ASIC case -- assumption, not published
    psu_inlet=50.0,           # degC typical PSU ambient rating -- assumption
)
# HCA airflow requirement [CX6]: 250 LFM at 35 degC, 350 LFM at 55 degC (linear between; flat below 35 degC)
LFM = 0.00508                 # m/s per LFM


# ============================================================================ helpers
def air_props(T_C):
    """Dry air at P0: rho (ideal gas, params), mu, k, Pr (CoolProp 'Air'); cached at 0.1 K resolution."""
    return _air_props(round(float(T_C), 1))


@functools.lru_cache(maxsize=4096)
def _air_props(T_C):
    T = T_C + 273.15
    return dict(rho=P.rho_air(T_C), mu=CP.PropsSI("V", "T", T, "P", P.P0, "Air"),
                k=CP.PropsSI("L", "T", T, "P", P.P0, "Air"),
                Pr=CP.PropsSI("Prandtl", "T", T, "P", P.P0, "Air"), cp=P.CP_AIR)


def h_plate(u, L, T_film_C):
    """Average h over a plate of length L in parallel flow at velocity u [TEC]:
    Nu_L = 0.664 Re_L^0.5 Pr^(1/3) (laminar, Re < 5e5); h = Nu k / L.
    Conservative: ignores free-stream turbulence from upstream fans/components (real h higher by ~1.2-1.5x)."""
    p = air_props(T_film_C)
    Re = p["rho"] * u * L / p["mu"]
    Nu = 0.664 * math.sqrt(max(Re, 1.0)) * p["Pr"] ** (1.0 / 3.0)
    return Nu * p["k"] / L


def psu_eff(load_frac, table=PSU_EFF_TABLE):
    """PSU efficiency vs per-PSU load fraction, linear interpolation of 80 PLUS points [80PLUS]."""
    x, y = table
    return float(np.interp(load_frac, x, y))


def cpu_package_power(load):
    """CPU package power per socket [W]: P = TDP (f_idle + (1 - f_idle) u)  (linear utilisation model)."""
    return P.CPU_TDP_W * (CPU_IDLE_FRAC + (1.0 - CPU_IDLE_FRAC) * load)


def nic_lfm_required(T_air_C):
    """Required approach velocity at the HCA [LFM] vs local air temperature [CX6] (interpolated)."""
    return float(np.interp(T_air_C, [35.0, 55.0], [250.0, 350.0]))


# ---------------------------------------------------------------------------- fans
def fan_dp(V, N):
    """Fan-wall static pressure [Pa] for N_FANS parallel fans at speed fraction N and total flow V [m^3/s].
    Affinity laws [ROT]: V_free ~ N, dP0 ~ N^2.  dP = N^2 dP0 [1 - a x - (1-a) x^2], x = V/(N n V0)."""
    x = V / (N * N_FANS * FAN_V0)
    return N ** 2 * FAN_DP0 * (1.0 - FAN_SHAPE_A * x - (1.0 - FAN_SHAPE_A) * x ** 2)


def operating_point(N=1.0, k=K_SYS):
    """Intersection of fan curve and system curve dP = k V^2. Because both scale with N^2 the
    operating flow is exactly V = N V_op(1) (affinity law). Solves
    (1 - a + k V0'^2/dP0) x^2 + a x - 1 = 0 with V0' = n V0 (x at N = 1)."""
    V0t = N_FANS * FAN_V0
    A = (1.0 - FAN_SHAPE_A) + k * V0t ** 2 / FAN_DP0
    x = (-FAN_SHAPE_A + math.sqrt(FAN_SHAPE_A ** 2 + 4 * A)) / (2 * A)
    V1 = x * V0t
    return dict(V_dot=N * V1, dP=k * (N * V1) ** 2, x=x)


V_OP_FULL = operating_point(1.0)["V_dot"]


def fan_power(N):
    """Electrical fan power [W]: datasheet rated power at 100 % scaled with N^3 (affinity) plus a small
    speed-independent electronics share: W = n P_rated [c0 + (1 - c0) N^3]."""
    return N_FANS * FAN_P_RATED * (FAN_P_FIXED + (1.0 - FAN_P_FIXED) * N ** 3)


# ============================================================================ node heat breakdown
def _components(load, W_fan, capture=CAPTURE_FRAC, eta_vr_cpu=ETA_VR_CPU, eta_vr_mem=ETA_VR_MEM,
                dimm_liquid_frac=0.0, psu_table=PSU_EFF_TABLE, mem_load=None):
    """Component powers [W] at CPU utilisation `load` for a given fan power (all end up as heat).
    mem_load (v2, workload cases): memory utilisation 0..1, decoupled from CPU load (None -> = load)."""
    P_cpu1 = cpu_package_power(load)
    P_cpu = P.SOCKETS_PER_NODE * P_cpu1
    ml = load if mem_load is None else mem_load
    P_dimm = N_DIMM * (P_DIMM_IDLE + (P_DIMM_ACT - P_DIMM_IDLE) * ml)
    # VR conversion loss: P_loss = P_out (1/eta - 1) + fixed  [BEL], [TI-PS]
    vr_cpu = P_cpu * (1.0 / eta_vr_cpu - 1.0) + P.SOCKETS_PER_NODE * P_VR_FIXED
    vr_mem = P_dimm * (1.0 / eta_vr_mem - 1.0)
    pch = P_PCH_IDLE + (P_PCH_FULL - P_PCH_IDLE) * load
    nic = P_NIC_TYP * (P_NIC_IDLE_FRAC + (1.0 - P_NIC_IDLE_FRAC) * load)
    ssd = P_SSD_IDLE + (P_SSD_ACT - P_SSD_IDLE) * load
    P_dc = P_cpu + P_dimm + vr_cpu + vr_mem + pch + nic + ssd + P_BMC + P_MISC + W_fan
    # PSU: P_in = P_dc / eta(P_dc / (n_share * rated))  [80PLUS]
    eta_psu = psu_eff(P_dc / (N_PSU_SHARING * PSU_RATED), psu_table)
    P_in = P_dc / eta_psu
    psu_loss = P_in - P_dc
    Q_liq = capture * P_cpu + dimm_liquid_frac * P_dimm
    return dict(cpu_package=P_cpu, cpu_residual_to_air=(1 - capture) * P_cpu, dimms=P_dimm,
                dimms_to_air=(1 - dimm_liquid_frac) * P_dimm, vrm_cpu_loss=vr_cpu, vrm_mem_loss=vr_mem,
                pch=pch, nic_hca=nic, ssd=ssd, bmc=P_BMC, misc=P_MISC, fans=W_fan, psu_loss=psu_loss,
                eta_psu=eta_psu, P_dc=P_dc, Q_node_total=P_in, Q_liquid=Q_liq, Q_air=P_in - Q_liq)


def _zone_heats(c):
    """Map component heats to the serial air zones (W, excluding fans)."""
    return dict(
        z1_drive=c["ssd"],
        z2_mem_vrm=c["dimms_to_air"] + c["vrm_cpu_loss"] + c["vrm_mem_loss"] + c["cpu_residual_to_air"],
        z3_pch_nic=c["pch"] + c["nic_hca"] + c["bmc"] + c["misc"],
        z4_psu=c["psu_loss"],
    )


def node_heat_breakdown(load=1.0, T_in_C=None, capture=CAPTURE_FRAC, **kw):
    """Heat ledger of one 1U Rudra node at CPU utilisation `load` (0..1).

    Fans are computed (not fixed): fixed-point iteration W_fan -> Q_air -> server_air(T_in) -> W_fan.
    T_in_C defaults to params.BASE['T_in_server'].  kw: eta_vr_cpu, eta_vr_mem, dimm_liquid_frac, psu_table.
    Returns {component W..., Q_node_total (wall/IT power), Q_cpu_package, Q_liquid, Q_air, liquid_frac,
    air_frac, capture_frac, eta_psu, fan_speed_frac}."""
    T_in = P.BASE["T_in_server"] if T_in_C is None else T_in_C
    W_fan = fan_power(FAN_FLOOR)
    for _ in range(50):
        c = _components(load, W_fan, capture, **kw)
        sa = server_air(T_in, c["Q_air"], load=load, comp=c)
        if abs(sa["W_fan"] - W_fan) < 1e-4:
            break
        W_fan = sa["W_fan"]
    c = _components(load, sa["W_fan"], capture, **kw)
    out = {k: float(v) for k, v in c.items()}
    out.update(Q_cpu_package=c["cpu_package"], liquid_frac=c["Q_liquid"] / c["Q_node_total"],
               air_frac=c["Q_air"] / c["Q_node_total"], capture_frac=float(capture),
               fan_speed_frac=sa["fan_speed_frac"], load=float(load), T_in_C=float(T_in))
    return out


# ============================================================================ chassis air model
def _evaluate(T_in, V_dot, Q_air, shares, per_dimm_share, W_fan):
    """Serial-zone model at a given volumetric flow (at inlet density). Returns temperatures and margins.

    Energy balance per zone (steady flow, cp const):  T_k+1 = T_k + Q_k / (m cp)
    Component case temperature:  T_case = T_air,local + Q / (h A eta_fin)   [Newton cooling, h from TEC]
    Power-stage/ASIC junction:   T_j = T_case + Q_dev R_jc
    """
    m = P.rho_air(T_in) * V_dot
    cp = P.CP_AIR
    Q_board = max(Q_air - W_fan, 0.0)
    Qz = {k: s * Q_board for k, s in shares.items()}
    u = V_dot * P.rho_air(T_in) / (A_FLOW * P.rho_air(T_in + 5.0))   # mean core velocity (slightly heated air)

    T0 = T_in
    T1 = T0 + Qz["z1_drive"] / (m * cp)                 # after drive bay
    T1f = T1 + W_fan / (m * cp)                          # fan wall (fan work dissipated in the air)
    T2 = T1f + Qz["z2_mem_vrm"] / (m * cp)               # after DIMM/VRM/socket region
    T3 = T2 + Qz["z3_pch_nic"] / (m * cp)                # after PCH / HCA / BMC
    T4 = T3 + Qz["z4_psu"] / (m * cp)                    # exhaust (PSU loss)

    res = dict(T_air_z0=T0, T_air_z1=T1, T_air_fan=T1f, T_air_z2=T2, T_air_z3=T3, T_out=T4, u_core=u, m_dot=m)
    comps = {}
    # SSD (front): local air = inlet
    g = GEO["ssd"]; q = Qz["z1_drive"]
    h = h_plate(u, g["L"], T0 + 10)
    comps["ssd"] = dict(Q=q, T_air=T0, T_case=T0 + q / (h * g["A"]), h=h)
    # DIMM: own channel, air enters at T1f; channel flow = rho u A_ch; DRAM case = channel exit + q/(hA)
    g = GEO["dimm"]; q = per_dimm_share * Q_board
    m_ch = P.rho_air(T1f) * u * A_DIMM_CH
    T_ch_out = T1f + q / (m_ch * cp)
    h = h_plate(u, g["L"], T1f + 20)
    # hottest (trailing-edge) DRAM: local Nu_x for uniform heat flux, Nu_x = 0.453 Re_x^0.5 Pr^(1/3)
    # (Incropera & DeWitt, eq. 7.45) -> h_L = (0.453/0.664) h_avg ; JEDEC T_case refers to the hottest DRAM
    h_L = DIMM_HOTSPOT * h
    comps["dimm"] = dict(Q=q, T_air=T_ch_out, T_case=T_ch_out + q / (h_L * g["A"]), h=h,
                         T_case_mean=T_ch_out - 0.5 * q / (m_ch * cp) + q / (h * g["A"]))
    # VRM bank per socket: local air = zone-2 exit (conservative), loss split over n_phase power stages
    g = GEO["vrm"]
    q = (Qz["z2_mem_vrm"] - N_DIMM * per_dimm_share * Q_board - shares["_cpu_res"] * Q_board) / g["n"]
    h = h_plate(u, g["L"], T2 + 20)
    T_hs = T2 + q / (h * g["A"] * g["eta"])
    comps["vrm"] = dict(Q=q, T_air=T2, T_case=T_hs, T_j=T_hs + q / g["n_phase"] * g["R_jc"], h=h)
    # PCH: local air = zone-3 inlet
    g = GEO["pch"]; q = shares["_pch"] * Q_board
    h = h_plate(u, g["L"], T2 + 20)
    comps["pch"] = dict(Q=q, T_air=T2, T_case=T2 + q / (h * g["A"] * g["eta"]), h=h)
    # HCA at the rear riser: local air = zone-3 exit (conservative)
    g = GEO["nic"]; q = NIC_ASIC_FRAC * shares["_nic"] * Q_board
    h = h_plate(u, g["L"], T3 + 20)
    comps["nic"] = dict(Q=q, T_air=T3, T_case=T3 + q / (h * g["A"] * g["eta"]), h=h)
    # residual CPU heat leaves through PCB/socket; board surface taken ~15 K above local air (assumption,
    # only used for the exergy ledger)
    comps["cpu_res"] = dict(Q=shares["_cpu_res"] * Q_board, T_air=0.5 * (T1f + T2),
                            T_case=0.5 * (T1f + T2) + 15.0)
    comps["psu"] = dict(Q=Qz["z4_psu"], T_air=T3, T_case=T3 + 20.0)      # PSU internals ~20 K over inlet (assump.)
    comps["misc"] = dict(Q=(shares["_bmc_misc"]) * Q_board, T_air=T2, T_case=T2 + 15.0)

    lfm_req = nic_lfm_required(T3)
    margins = dict(
        dimm=LIMITS["dimm_case"] - comps["dimm"]["T_case"],
        ssd=LIMITS["ssd_case"] - comps["ssd"]["T_case"],
        vrm=LIMITS["vrm_junction"] - comps["vrm"]["T_j"],
        pch=LIMITS["pch_case"] - comps["pch"]["T_case"],
        nic_air=LIMITS["nic_air"] - T3,
        nic_case=LIMITS["nic_case"] - comps["nic"]["T_case"],
        nic_lfm=(u / LFM - lfm_req) / 10.0,     # scaled: 10 LFM ~ 1 K margin (only the sign matters)
        psu=LIMITS["psu_inlet"] - T3,
    )
    res.update(comps=comps, margins=margins, lfm_req=lfm_req, Qz=Qz)
    return res


def _shares(comp):
    """Zone heat shares of the board (non-fan) air heat from a component dict."""
    z = _zone_heats(comp)
    tot = sum(z.values())
    s = {k: v / tot for k, v in z.items()}
    s["_cpu_res"] = comp["cpu_residual_to_air"] / tot
    s["_pch"] = comp["pch"] / tot
    s["_nic"] = comp["nic_hca"] / tot
    s["_bmc_misc"] = (comp["bmc"] + comp["misc"]) / tot
    return s, comp["dimms_to_air"] / N_DIMM / tot


def server_air(T_in_C, Q_air_W, V_dot=None, load=1.0, comp=None, k_sys=K_SYS, floor=FAN_FLOOR):
    """Air side of ONE 1U node.

    Q_air_W : total air-side heat [W] INCLUDING the server-fan dissipation (as returned by
              node_heat_breakdown()['Q_air']). The fan power is solved self-consistently; the board part
              (Q_air_W - W_fan) is split over the zones with the component shares of node_heat_breakdown at
              `load` (or of `comp` if given).
    V_dot   : volumetric flow at inlet conditions [m^3/s]. None -> fan control law: the minimum fan speed
              N in [floor, 1] such that every air-cooled component is within its limit (bisection; all
              margins increase monotonically with N).
    Returns dict: V_dot, V_dot_cfm, m_dot, T_out, dT, W_fan, fan_speed_frac, dP_sys, T_component_max,
              limiting_component (code: 0 dimm,1 ssd,2 vrm,3 pch,4 nic_air,5 nic_case,6 nic_lfm,7 psu, -1 floor),
              feasible, zone temperatures, component temperatures, X_dest_* [W].
    """
    if comp is None:
        comp = _components(load, fan_power(FAN_FLOOR))
    shares, per_dimm = _shares(comp)
    V1 = operating_point(1.0, k_sys)["V_dot"]
    names = list(_evaluate(T_in_C, V1, Q_air_W, shares, per_dimm, fan_power(1.0))["margins"].keys())

    def ev(N):
        W = fan_power(N)
        return _evaluate(T_in_C, N * V1, Q_air_W, shares, per_dimm, W), W

    if V_dot is None:
        r_floor, _ = ev(floor)
        if min(r_floor["margins"].values()) >= 0:
            N = floor; lim = -1
        else:
            r_full, _ = ev(1.0)
            if min(r_full["margins"].values()) < 0:
                N = 1.0
            else:
                lo, hi = floor, 1.0
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if min(ev(mid)[0]["margins"].values()) >= 0:
                        hi = mid
                    else:
                        lo = mid
                N = hi
            r_N, _ = ev(N)
            lim = int(np.argmin([r_N["margins"][n] for n in names]))
        if DT_MAX_AIR is not None:
            # v2, air-only reference: a CPU heat sink needs air flow too. Cap the chassis air rise at
            # DT_MAX_AIR (typical 1U 2-socket design 15-20 K) -> raise N until dT <= cap.
            for _ in range(30):
                rN, _ = ev(N)
                dT_N = rN["T_out"] - T_in_C
                if dT_N <= DT_MAX_AIR + 1e-3 or N >= 1.0:
                    break
                N = min(1.0, N * dT_N / DT_MAX_AIR)
                lim = 8
    else:
        N = V_dot / V1
        lim = None
    r, W = ev(N)
    if lim is None:
        lim = int(np.argmin([r["margins"][n] for n in names]))
    feasible = float(min(r["margins"].values()) >= -1e-6)
    if DT_MAX_AIR is not None and V_dot is None and (r["T_out"] - T_in_C) > DT_MAX_AIR + 0.05:
        feasible = 0.0

    # Exergy (simple form): heat Q_i crossing from component surface T_s,i to local air T_a,i
    #   S_gen = sum Q_i (1/T_a - 1/T_s);  X_dest = T0 S_gen   (Bejan, Gouy-Stodola)
    T0 = P.T0_K
    S_ht = sum(c["Q"] * (1.0 / (c["T_air"] + 273.15) - 1.0 / (c["T_case"] + 273.15))
               for c in r["comps"].values() if c["Q"] > 0)
    m, cp = r["m_dot"], P.CP_AIR
    Tin_K, Tout_K = T_in_C + 273.15, r["T_out"] + 273.15
    # flow exergy gained by the air stream between inlet and exhaust (ideal gas, dP neglected)
    X_air = m * cp * ((Tout_K - Tin_K) - T0 * math.log(Tout_K / Tin_K))
    cm = r["comps"]
    out = dict(
        V_dot=N * V1, V_dot_cfm=N * V1 / 4.719474e-4, m_dot=m, T_in=T_in_C, T_out=r["T_out"],
        dT=r["T_out"] - T_in_C, W_fan=W, fan_speed_frac=N, dP_sys=k_sys * (N * V1) ** 2,
        eta_fan_total=k_sys * (N * V1) ** 3 / W,
        u_core=r["u_core"], u_core_lfm=r["u_core"] / LFM, nic_lfm_req=r["lfm_req"],
        T_air_z1=r["T_air_z1"], T_air_fan=r["T_air_fan"], T_air_z2=r["T_air_z2"], T_air_z3=r["T_air_z3"],
        T_dimm_case=cm["dimm"]["T_case"], T_ssd_case=cm["ssd"]["T_case"], T_vrm_hs=cm["vrm"]["T_case"],
        T_vrm_j=cm["vrm"]["T_j"], T_pch_case=cm["pch"]["T_case"], T_nic_case=cm["nic"]["T_case"],
        T_nic_air=r["T_air_z3"], T_psu_in=r["T_air_z3"],
        h_dimm=cm["dimm"]["h"], h_vrm=cm["vrm"]["h"],
        T_component_max=max(cm["dimm"]["T_case"], cm["ssd"]["T_case"], cm["vrm"]["T_j"],
                            cm["pch"]["T_case"], cm["nic"]["T_case"]),
        limiting_component=float(lim), feasible=feasible,
        X_dest_heat_transfer=T0 * S_ht, X_dest_fan=W, X_air_gain=X_air,
        S_gen_heat_transfer=S_ht,
    )
    for n in names:
        out["margin_" + n] = r["margins"][n]
    return {k: float(v) for k, v in out.items()}


LIMIT_NAMES = {-1: "fan floor", 0: "DIMM T_case", 1: "SSD T_case", 2: "VRM T_j", 3: "PCH T_case",
               4: "HCA air 55 C", 5: "HCA case", 6: "HCA LFM", 7: "PSU inlet", 8: "air rise cap (CPU heat sink)"}
DT_MAX_AIR = None   # v2: set (K) only for the air-only reference machine; None = hybrid node (CPUs on liquid)


# ============================================================================ pod
def pod_air(T_in_C, load=1.0, n_nodes=None, **kw):
    """576 identical nodes (params.NODES_PER_POD). Returns pod totals:
    m_dot_servers [kg/s], V_dot_servers [m^3/s at inlet], cfm_servers, Q_air_pod, Q_liquid_pod, Q_it_pod [W],
    T_out (= server exhaust, all nodes identical), W_fans_pod [W], X_dest_chassis_ht [W], X_dest_fans [W],
    plus per-node values."""
    n = P.NODES_PER_POD if n_nodes is None else n_nodes
    cap = kw.pop("capture", None)
    cap = CAPTURE_FRAC if cap is None else cap
    nb = node_heat_breakdown(load, T_in_C, capture=cap, **kw)
    sa = server_air(T_in_C, nb["Q_air"], load=load,
                    comp=_components(load, nb["fans"], nb["capture_frac"],
                                     **{k: v for k, v in kw.items()}))
    return dict(
        n_nodes=float(n), T_in=float(T_in_C), load=float(load),
        m_dot_servers=n * sa["m_dot"], V_dot_servers=n * sa["V_dot"], cfm_servers=n * sa["V_dot_cfm"],
        Q_air_pod=n * nb["Q_air"], Q_liquid_pod=n * nb["Q_liquid"], Q_it_pod=n * nb["Q_node_total"],
        T_out=sa["T_out"], dT=sa["dT"], W_fans_pod=n * sa["W_fan"], fan_speed_frac=sa["fan_speed_frac"],
        X_dest_chassis_ht=n * sa["X_dest_heat_transfer"], X_dest_fans=n * sa["W_fan"],
        X_air_gain=n * sa["X_air_gain"], limiting_component=sa["limiting_component"],
        feasible=sa["feasible"], Q_air_node=nb["Q_air"], Q_node=nb["Q_node_total"], m_dot_node=sa["m_dot"],
        air_frac=nb["air_frac"],
    )


# ============================================================================ self-test / figures
COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _style(plt):
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "font.size": 9, "axes.titlesize": 10, "lines.linewidth": 2.0, "legend.frameon": False})


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    T_b = P.BASE["T_in_server"]
    m_liq_node = P.SOCKETS_PER_NODE * P.BASE["m_liq_plate"]
    cp_liq = CP.PropsSI("C", "T", P.BASE["T_liq_in"] + 273.15, "P", 2e5, "INCOMP::APG-25%")

    nb = node_heat_breakdown(1.0, T_b)
    sa = server_air(T_b, nb["Q_air"], load=1.0)
    pod = pod_air(T_b, 1.0)
    C_air = sa["m_dot"] * P.CP_AIR
    C_liq = m_liq_node * cp_liq

    # ---------------- print baseline table
    print(f"\nS2 baseline: T_in = {T_b} C, load = 1.0, capture = {CAPTURE_FRAC}, {N_FANS} x Delta GFB0412EHS")
    print("-" * 64)
    order = ["cpu_package", "dimms", "vrm_cpu_loss", "vrm_mem_loss", "pch", "nic_hca", "ssd", "bmc", "misc",
             "fans", "psu_loss"]
    for k in order:
        print(f"  {k:22s} {nb[k]:8.1f} W")
    print(f"  {'Q_node_total (wall)':22s} {nb['Q_node_total']:8.1f} W   (PSU eta {nb['eta_psu']:.3f})")
    print(f"  {'Q_liquid':22s} {nb['Q_liquid']:8.1f} W   ({100*nb['liquid_frac']:.1f} %)")
    print(f"  {'Q_air':22s} {nb['Q_air']:8.1f} W   ({100*nb['air_frac']:.1f} %)")
    print("-" * 64)
    for k in ["fan_speed_frac", "V_dot_cfm", "m_dot", "dT", "T_out", "W_fan", "dP_sys", "eta_fan_total",
              "u_core_lfm", "nic_lfm_req", "T_dimm_case", "T_ssd_case", "T_vrm_j", "T_pch_case", "T_nic_case",
              "T_nic_air", "X_dest_heat_transfer", "X_air_gain"]:
        print(f"  {k:22s} {sa[k]:10.4g}")
    print(f"  limiting component     {LIMIT_NAMES[int(sa['limiting_component'])]}")
    print(f"  full-speed op. point   {V_OP_FULL/4.719474e-4:.1f} CFM at {operating_point(1)['dP']:.0f} Pa")
    print(f"  C_air/C_liq            {C_air/C_liq:.3f}  (C_air {C_air:.1f}, C_liq {C_liq:.1f} W/K)")
    print(f"  air flow per kW air    {1000*sa['m_dot']/nb['Q_air']*1000:.1f} g/s per kW")
    print("-" * 64)
    print(f"  POD ({P.NODES_PER_POD} nodes): m_air {pod['m_dot_servers']:.2f} kg/s, {pod['V_dot_servers']:.2f} m3/s,"
          f" {pod['cfm_servers']:.0f} CFM; Q_air {pod['Q_air_pod']/1e3:.1f} kW, Q_liq {pod['Q_liquid_pod']/1e3:.1f} kW,"
          f" IT {pod['Q_it_pod']/1e3:.1f} kW; T_out {pod['T_out']:.1f} C; fans {pod['W_fans_pod']/1e3:.2f} kW;"
          f" X_dest_ht {pod['X_dest_chassis_ht']/1e3:.2f} kW")

    # ---------------- Fig 1: node breakdown + load sweep
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = ["CPU pkg → liquid", "CPU pkg → air", "DIMMs", "CPU VR loss", "Mem VR loss", "PCH", "HCA (NDR)",
              "SSD", "BMC", "Misc", "Fans", "PSU loss"]
    vals = [nb["Q_liquid"], nb["cpu_residual_to_air"], nb["dimms"], nb["vrm_cpu_loss"], nb["vrm_mem_loss"],
            nb["pch"], nb["nic_hca"], nb["ssd"], nb["bmc"], nb["misc"], nb["fans"], nb["psu_loss"]]
    colors = [COLS[0]] + [COLS[1]] * 11
    y = np.arange(len(vals))[::-1]
    ax[0].barh(y, vals, color=colors, height=0.6)
    for yi, v in zip(y, vals):
        ax[0].text(v + 3, yi, f"{v:.1f}", va="center", fontsize=8, color="#52514e")
    ax[0].set_yticks(y); ax[0].set_yticklabels(labels)
    ax[0].set_xlabel("Heat per node [W]"); ax[0].grid(axis="y", visible=False)
    ax[0].set_title(f"Full load: {nb['Q_node_total']:.0f} W/node, {100*nb['air_frac']:.0f} % to air "
                    f"(blue = liquid)")
    L = np.linspace(0, 1, 21)
    rows = [node_heat_breakdown(l, T_b) for l in L]
    ax[1].plot(L * 100, [r["Q_node_total"] for r in rows], color="#52514e", label="Node total (wall)")
    ax[1].plot(L * 100, [r["Q_liquid"] for r in rows], color=COLS[0], label="Liquid (cold plates)")
    ax[1].plot(L * 100, [r["Q_air"] for r in rows], color=COLS[1], label="Air")
    ax[1].set_xlabel("CPU utilisation [%]"); ax[1].set_ylabel("Heat per node [W]")
    ax[1].legend(loc="upper left")
    ax[1].set_title(f"Air share falls from {100*rows[0]['air_frac']:.0f} % (idle) to "
                    f"{100*rows[-1]['air_frac']:.0f} % (full load)")
    ax[1].set_ylim(0, None)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "s2_node_breakdown.png"), dpi=200); plt.close(fig)

    # ---------------- Fig 2: fan curves & system curve
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    Vg = np.linspace(0, N_FANS * FAN_V0, 200)
    for i, N in enumerate([1.0, 0.75, 0.5, FAN_FLOOR]):
        Vn = Vg[Vg <= N * N_FANS * FAN_V0]
        ax.plot(Vn / 4.719474e-4, fan_dp(Vn, N), color=COLS[i], label=f"Fan wall, N = {N:.0%}")
        op = operating_point(N)
        ax.plot(op["V_dot"] / 4.719474e-4, op["dP"], "o", color=COLS[i], ms=7, mec="white", mew=1.5)
    ax.plot(Vg / 4.719474e-4, K_SYS * Vg ** 2, color="#52514e", ls="--", label="Chassis ΔP = kV²")
    ax.plot(sa["V_dot_cfm"], sa["dP_sys"], "D", color=COLS[7], ms=8, mec="white", label="Baseline control point")
    ax.set_ylim(0, 600); ax.set_xlim(0, N_FANS * FAN_V0 / 4.719474e-4)
    ax.set_xlabel("Node air flow [CFM]"); ax.set_ylabel("Static pressure [Pa]")
    ax.set_title(f"8×GFB0412EHS: {V_OP_FULL/4.719474e-4:.0f} CFM at full speed; baseline needs "
                 f"{sa['V_dot_cfm']:.0f} CFM (N = {sa['fan_speed_frac']:.0%})")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "s2_fan_curve.png"), dpi=200); plt.close(fig)

    # ---------------- Fig 3: required flow & fan power vs inlet temperature
    Tins = np.linspace(18, 45, 55)     # brief asks 18-35 C (shaded band = ASHRAE recommended 18-27 C);
                                       # extended to 45 C to expose where the cubic fan law starts to bite
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))
    sweep = {}
    for i, l in enumerate([1.0, 0.7, 0.4]):
        rs = [pod_air(t, l) for t in Tins]
        sweep[l] = rs
        ax[0].plot(Tins, [r["cfm_servers"] / P.NODES_PER_POD for r in rs], color=COLS[i], label=f"load {l:.0%}")
        ax[1].plot(Tins, [r["W_fans_pod"] / 1e3 for r in rs], color=COLS[i], label=f"load {l:.0%}")
        ax[2].plot(Tins, [r["T_out"] for r in rs], color=COLS[i], label=f"load {l:.0%}")
    for a in ax:
        a.axvspan(P.T_INLET_MIN_C, P.T_INLET_MAX_C, color="#2a78d6", alpha=0.06, lw=0)
        a.axvline(35.0, color="#a3a29d", lw=0.8, ls=":")
        a.set_xlabel("Server inlet temperature [°C]")
    ax[0].set_ylabel("Required flow per node [CFM]"); ax[1].set_ylabel("Server-fan power, pod [kW]")
    ax[2].set_ylabel("Server exhaust temperature [°C]")
    r1 = sweep[1.0]
    lim_lo = LIMIT_NAMES[int(r1[0]["limiting_component"])]
    lim_hi = LIMIT_NAMES[int(r1[-1]["limiting_component"])]
    i35 = int(np.argmin(abs(Tins - 35.0)))
    ax[0].set_title(f"Flow set by {lim_lo} at 18 °C, {LIMIT_NAMES[int(r1[i35]['limiting_component'])]} "
                    f"above ~27 °C")
    ax[1].set_title(f"Fan power ×{r1[i35]['W_fans_pod']/r1[0]['W_fans_pod']:.2f} at 35 °C, "
                    f"×{r1[-1]['W_fans_pod']/r1[0]['W_fans_pod']:.1f} at 45 °C (full load)")
    ax[2].set_title("Exhaust falls as fans speed up (ΔT shrinks)")
    ax[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "s2_fan_vs_inlet.png"), dpi=200); plt.close(fig)

    # ---------------- Fig 4: zone air & component temperatures at 24 and 32 C
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    xs = np.arange(6)
    xl = ["Inlet", "After drive bay", "After fan wall", "After DIMM/VRM", "After PCH/HCA", "Exhaust"]
    for i, t in enumerate([T_b, 32.0]):
        nbt = node_heat_breakdown(1.0, t)
        s = server_air(t, nbt["Q_air"], load=1.0)
        ax.plot(xs, [t, s["T_air_z1"], s["T_air_fan"], s["T_air_z2"], s["T_air_z3"], s["T_out"]],
                color=COLS[i], marker="o", label=f"Air, T_in = {t:.0f} °C (N = {s['fan_speed_frac']:.0%})")
        pts = [(0.15, s["T_ssd_case"], "SSD", (6, 2)), (2.75, s["T_dimm_case"], "DIMM", (-30, 4)),
               (3.0, s["T_vrm_j"], "VRM Tj", (6, 4)), (3.25, s["T_pch_case"], "PCH", (6, 2)),
               (4.0, s["T_nic_case"], "HCA", (6, 2))]
        for x, v, n, off in pts:
            ax.plot(x, v, "s", color=COLS[i], ms=7, mec="white")
            if i == 1:
                ax.annotate(n, (x, v), textcoords="offset points", xytext=off, fontsize=8, color="#52514e")
    for n, v in [("DIMM 85", 85), ("SSD 70", 70), ("HCA air 55", 55)]:
        ax.axhline(v, color="#a3a29d", lw=0.8, ls=":")
        ax.text(5.05, v, n, fontsize=7, color="#52514e", va="center")
    ax.set_xticks(xs); ax.set_xticklabels(xl, rotation=20)
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Air (lines) and component (squares): DIMM is closest to its limit")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "s2_zone_temps.png"), dpi=200); plt.close(fig)

    # ---------------- Fig 5: air/liquid split vs board thermal management
    caps = np.linspace(0.85, 0.98, 14)
    T_SPLIT = 32.0
    scen = [("Baseline", {}), ("VR η 0.95/0.93", dict(eta_vr_cpu=0.95, eta_vr_mem=0.93)),
            ("DIMM cold plates (90 %)", dict(dimm_liquid_frac=0.9)),
            ("Titanium PSU", dict(psu_table=PSU_EFF_TITANIUM))]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))
    scen_out = {}
    for i, (nm, kw) in enumerate(scen):
        rs = [node_heat_breakdown(1.0, T_b, capture=c, **kw) for c in caps]
        ss = [server_air(T_b, r["Q_air"], load=1.0, comp=_components(1.0, r["fans"], c, **kw))
              for r, c in zip(rs, caps)]
        # flow panels at a warm inlet (T_SPLIT) where component limits, not the fan floor, set the flow
        rh = [node_heat_breakdown(1.0, T_SPLIT, capture=c, **kw) for c in caps]
        sh = [server_air(T_SPLIT, r["Q_air"], load=1.0, comp=_components(1.0, r["fans"], c, **kw))
              for r, c in zip(rh, caps)]
        ax[0].plot(caps * 100, [100 * r["air_frac"] for r in rs], color=COLS[i], label=nm)
        ax[1].plot(caps * 100, [s["V_dot_cfm"] for s in sh], color=COLS[i], label=nm)
        ax[2].plot(caps * 100, [s["m_dot"] * P.CP_AIR / C_liq for s in sh], color=COLS[i], label=nm)
        j = int(np.argmin(abs(caps - CAPTURE_FRAC)))
        scen_out[nm] = dict(air_frac=rs[j]["air_frac"], Q_air=rs[j]["Q_air"], cfm=ss[j]["V_dot_cfm"],
                            cfm_32C=sh[j]["V_dot_cfm"], cfm_32C_cap085=sh[0]["V_dot_cfm"],
                            cfm_32C_cap098=sh[-1]["V_dot_cfm"],
                            W_fan=ss[j]["W_fan"], C_ratio=ss[j]["m_dot"] * P.CP_AIR / C_liq,
                            limiting=LIMIT_NAMES[int(ss[j]["limiting_component"])],
                            Q_node=rs[j]["Q_node_total"])
    for a in ax:
        a.set_xlabel("CPU cold-plate capture fraction [%]")
    ax[0].set_ylabel("Air share of node heat [%]"); ax[1].set_ylabel("Required node air flow [CFM]")
    ax[2].set_ylabel("Capacity ratio ṁc_p,air / ṁc_p,liq [–]")
    ax[0].set_title("Air share of node heat:\nDIMM cold plates cut it the most")
    ax[1].axhline(sa["V_dot_cfm"], color="#a3a29d", lw=0.8, ls=":")
    ax[1].set_title(f"{T_SPLIT:.0f} °C inlet: only DIMM plates cut flow\n(other 3 overlap; dotted = fan floor)")
    ax[2].set_title(f"Capacity ratio at {T_SPLIT:.0f} °C:\nCPU capture fraction does not move it")
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "s2_air_liquid_split.png"), dpi=200); plt.close(fig)

    # same comparison at a hot inlet (32 C) where flow is limit-driven
    scen_hot = {}
    for nm, kw in scen:
        r = node_heat_breakdown(1.0, 32.0, **kw)
        s = server_air(32.0, r["Q_air"], load=1.0, comp=_components(1.0, r["fans"], CAPTURE_FRAC, **kw))
        scen_hot[nm] = dict(cfm=s["V_dot_cfm"], W_fan=s["W_fan"], air_frac=r["air_frac"],
                            limiting=LIMIT_NAMES[int(s["limiting_component"])])
    print("  scenarios @24 C:", {k: (round(v["air_frac"], 3), round(v["cfm"], 1)) for k, v in scen_out.items()})
    print("  scenarios @32 C:", {k: (round(v["cfm"], 1), round(v["W_fan"], 1), v["limiting"])
                                 for k, v in scen_hot.items()})

    # ---------------- JSON
    sw = {f"load_{l}": dict(T_in=Tins.tolist(), cfm_node=[r["cfm_servers"] / P.NODES_PER_POD for r in rs],
                             W_fans_pod=[r["W_fans_pod"] for r in rs], T_out=[r["T_out"] for r in rs],
                             limiting=[LIMIT_NAMES[int(r["limiting_component"])] for r in rs])
          for l, rs in sweep.items()}
    out = dict(
        inputs=dict(T_in_C=T_b, load=1.0, capture_frac=CAPTURE_FRAC, cpu_idle_frac=CPU_IDLE_FRAC,
                    n_fans=N_FANS, fan_V0_m3s=FAN_V0, fan_dP0_Pa=FAN_DP0, fan_P_rated_W=FAN_P_RATED,
                    fan_floor=FAN_FLOOR, k_sys=K_SYS, limits=LIMITS, eta_vr_cpu=ETA_VR_CPU,
                    eta_vr_mem=ETA_VR_MEM, psu_rated=PSU_RATED),
        node_breakdown=nb, server_air=sa, pod=pod,
        full_speed_operating_point=dict(V_dot=V_OP_FULL, cfm=V_OP_FULL / 4.719474e-4,
                                        dP=operating_point(1)["dP"], net_shutoff_dP_Pa=FAN_DP0),
        capacity_ratio_air_liq=C_air / C_liq, C_air_W_K=C_air, C_liq_W_K=C_liq,
        air_mass_flow_g_s_per_kW_air=1e6 * sa["m_dot"] / nb["Q_air"],
        scenarios_24C=scen_out, scenarios_32C=scen_hot, inlet_sweep=sw,
        idle=node_heat_breakdown(0.0, T_b),
    )
    with open(os.path.join(RES, "s2_baseline.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    return out


if __name__ == "__main__":
    main()
