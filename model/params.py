"""
Shared global parameters for the PARAM Rudra hybrid-cooled enclosed-aisle study.
ALL modules import from here. SI units throughout; temperatures stored in degC
unless the name ends in _K. Change a value here, not inside a module.

Sources for system facts (opened 2026-10-13):
 - NSM, PARAM Rudra IIT Bombay: https://nsmindia.in/infrastructure/nsm-systems/param-rudra-iit-bombay/
   482 CPU nodes (192 GB) + 90 high-memory nodes (768 GB) + 30 GPU nodes; each node 2x Xeon Gold 6240R
   (24C, 2.4 GHz), 800 GB SSD, InfiniBand NDR.
 - C-DAC keynote (Kuvelkar, 2024): Rudra servers offered with direct-contact liquid cooling (cold plates),
   "360 W chip cooler" development.
"""
import numpy as np

# ---------------------------------------------------------------- dead state / climate (Mumbai)
T0_C      = 30.0            # dead-state (environment) dry-bulb, degC  (representative Mumbai)
T0_K      = T0_C + 273.15
P0        = 101325.0        # Pa
T_WB_C    = 25.0            # representative ambient wet bulb, degC (cooling-tower sink)
DESIGN_SUMMER = dict(T_db=34.0, T_wb=28.0)   # for sensitivity only

# ---------------------------------------------------------------- air
CP_AIR    = 1006.0          # J/kg/K (dry air; humidity effect on cp neglected, <1 %)
R_AIR     = 287.05          # J/kg/K
def rho_air(T_C, P=P0):
    return P / (R_AIR * (T_C + 273.15))

# ---------------------------------------------------------------- hardware scope
CPU_MODEL        = "Intel Xeon Gold 6240R"
CPU_TDP_W        = 165.0    # per socket (Intel ARK) -- verify in S1
SOCKETS_PER_NODE = 2
NODES_PER_RACK   = 36       # 1U Rudra nodes per 42U rack (rest: ToR/IB switch, PDU, CDU manifold)
RACKS_PER_ROW    = 8
ROWS_PER_POD     = 2        # one enclosed-aisle pod = 2 facing rows
RACKS_PER_POD    = RACKS_PER_ROW * ROWS_PER_POD        # 16
NODES_PER_POD    = NODES_PER_RACK * RACKS_PER_POD      # 576  ~ 482 CPU + 90 high-mem = 572 nodes
N_IRC            = 7        # in-row coolers serving the pod (6 needed + 1 redundant, all running, equal share). Integration found 4 insufficient for the 126 kW air load.

# ---------------------------------------------------------------- limits / envelopes
T_INLET_MAX_C    = 27.0     # ASHRAE TC9.9 recommended server inlet upper bound (A1 class)
T_INLET_MIN_C    = 18.0
T_INLET_ALLOW_C  = 32.0     # A1 allowable
LIQ_SUPPLY_RANGE = (17.0, 45.0)   # ASHRAE W17 .. W45 facility/TCS supply classes
ROOM_DEWPOINT_C  = 12.0     # assumed controlled-room dew point (for coil condensation check)

# ---------------------------------------------------------------- baseline operating point (before optimisation)
BASE = dict(
    T_in_server   = 24.0,   # degC, server inlet set-point
    psi           = 0.0,    # intentional bypass-recirculation fraction (hot -> cold aisle), of server flow
    beta          = 0.05,   # cold-air leakage/bypass fraction of cooler flow (containment seal quality)
    T_liq_in      = 35.0,   # degC, TCS (cold-plate) supply temperature
    m_liq_plate   = 1.0/60*0.998,  # kg/s per cold plate (1.0 L/min water-like)
    T_chw_in      = 14.0,   # degC chilled-water supply to in-row coolers
    config        = "CAC",  # cold-aisle containment
)

# ---------------------------------------------------------------- v2: C-DAC design data, IIT Bombay installation
# Source: C-DAC slides "IIT-B Data centre Heat Rejection Methods".
CDAC = dict(
    rack_nodes=[64, 64, 32, 27, 28, 64, 64, 64, 64, 64, 64, 20],   # racks 1..12 (3, 4 = CPU + GPU)
    n_nodes=619,
    q_dlc_node=330.0,          # W per node to DLC (2 x 165 W), C-DAC design duty 204.3 kW
    lpm_node=0.5,              # L/min per node (secondary), 309.5 L/min total
    T_sec_supply=37.0,         # degC secondary supply to servers (33-37 range)
    lpm_primary=400.0,         # L/min facility water through the CDU primary
    T_prim_supply=35.0,        # degC facility supply (33 +/- 2 spec)
    UA_cdu=69.0e3,             # W/K, CoolIT CHx750, from the C-DAC duty sheet (eps 0.83, NTU 3.23)
    header_DN_m=0.1023,        # 4" Sch40 ID
    branch_DN_m=0.03505,       # 1-1/4" Sch40 ID (valve trains)
)

def summary():
    return {k: v for k, v in globals().items() if k.isupper()}
