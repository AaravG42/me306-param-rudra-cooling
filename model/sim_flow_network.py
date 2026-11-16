"""
SIM A - Secondary (TCS) loop flow distribution across the 12 PARAM Rudra racks (C-DAC layout).

What the lumped model cannot see: it gives every node exactly 0.5 L/min. Here the 619 nodes sit on
real hydraulic paths - CDU -> 4" supply header -> 1-1/4" valve train -> rack -> two bottom-fed vertical
manifolds (U-type: supply and return both at the bottom) -> node hoses + quick disconnects -> two cold
plates in series -> back the same way. Racks hold 20..64 nodes, so branch flows differ by 3x and the
pressure left for each node differs too.

Method: steady incompressible pipe network. Every edge obeys  dp = a Q + b Q|Q|  (laminar + quadratic).
Unknowns are junction pressures; mass balance at each junction is solved by Newton with a sparse Jacobian.
The CDU is a flow source: its outlet pressure is found (secant) so the total flow equals the target.
Node thermal: S1 cold_plate() for both plates in series at 165 W each (thermal virus, full capture =
C-DAC design duty), secondary supply 37 degC.

Assumed hardware (C-DAC gives sizes only for the 4" headers and 1-1/4" valve trains):
  * header run CDU -> first rack 5 m; racks 0.6 m apart along a direct-return header;
  * valve train K = 12 for supply + return together (2 ball valves, Y-strainer, balancing valve fully
    open, flex hoses, tees) + 2 x 3 m of 1-1/4" pipe; varied 6..30;
  * rack manifold: 2 per rack ("42U dual rack manifolds"), 25 mm ID, taps evenly over 1.87 m (42U),
    each carrying half the rack's nodes; varied 20..32 mm.
Run: python3 sim_flow_network.py
"""
import json, math, os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import params as P
import s1_liquid as s1

RES = "results/"; FIG = "figures/"
C = P.CDAC
T_SUP = C["T_sec_supply"]
PR = s1.coolant_props(41.0, "PG25")           # loop-mean properties
RHO, MU = PR["rho"], PR["mu"]
LPM = 1 / 60000.0                              # m^3/s per L/min

# -------------------------------------------------------------------------------- edge laws
def pipe_coeffs(D, L, Q_nom, K=0.0):
    """dp = a Q + b Q|Q| for a pipe of diameter D, length L at nominal flow Q_nom (friction factor frozen
    at Q_nom: laminar part in a, turbulent/minor part in b)."""
    A = math.pi * D ** 2 / 4
    Re = RHO * (Q_nom / A) * D / MU
    if Re < 2300:
        a = 128 * MU * L / (math.pi * D ** 4)          # Hagen-Poiseuille
        b = K * RHO / (2 * A ** 2)
    else:
        f = 0.316 * Re ** -0.25                          # Blasius, smooth pipe
        a = 0.0
        b = (f * L / D + K) * RHO / (2 * A ** 2)
    return a, b

_FIT = None
def fit_node_curve():
    global _FIT
    if _FIT is not None:
        return _FIT
    _FIT = _fit_node_curve()
    return _FIT

def _fit_node_curve():
    """Node = 2 QDs + hose + 2 cold plates in series; fit dp(Q) = a Q + b Q^2 from S1's component models."""
    qs = np.linspace(0.1, 1.2, 25) * LPM
    dps = []
    for q in qs:
        m = q * RHO
        p1 = s1.cold_plate(165.0, m, T_SUP)
        p2 = s1.cold_plate(165.0, m, p1["T_out"])
        dqd, dh = s1._dp_hose_qd(q, PR)
        dps.append(p1["dP"] + p2["dP"] + dqd + dh)
    A = np.vstack([qs, qs ** 2]).T
    (a, b), *_ = np.linalg.lstsq(A, np.array(dps), rcond=None)
    err = np.max(np.abs(A @ [a, b] - dps) / np.array(dps))
    return float(a), float(b), float(err), dict(q_lpm=(qs / LPM).tolist(), dp_kPa=(np.array(dps) / 1e3).tolist())

# -------------------------------------------------------------------------------- network build
def build(rack_nodes, K_vt=12.0, D_man=0.025, header_first=5.0, rack_pitch=0.6, extra_K=None, node_scale=1.0):
    """Return (n_junctions, edges, node_edge_ids, meta). Junction 0 = CDU supply outlet (p fixed),
    junction 1 = CDU return inlet (p = 0)."""
    edges = []      # (i, j, a, b)
    node_edges = []  # (rack, manifold, tap, edge_index)
    J = [0, 1]
    nxt = [2]
    def new():
        nxt[0] += 1; return nxt[0] - 1
    q_tot = sum(rack_nodes) * C["lpm_node"] * LPM
    a_n, b_n, _, _ = fit_node_curve()
    a_n, b_n = a_n * node_scale, b_n * node_scale     # sensitivity: more restrictive commercial plates/QDs
    D_h, D_b = C["header_DN_m"], C["branch_DN_m"]
    prev_s, prev_r = 0, 1
    q_left = q_tot
    for r, n in enumerate(rack_nodes):
        hs, hr = new(), new()
        L = header_first if r == 0 else rack_pitch
        a, b = pipe_coeffs(D_h, L, q_left)
        edges.append((prev_s, hs, a, b))                 # supply header segment
        edges.append((hr, prev_r, a, b))                 # return header segment (direct return)
        prev_s, prev_r = hs, hr
        q_r = n * C["lpm_node"] * LPM
        Kx = K_vt + (extra_K[r] if extra_K is not None else 0.0)
        rs, rr = new(), new()
        a, b = pipe_coeffs(D_b, 3.0, q_r, K=Kx / 2)
        edges.append((hs, rs, a, b)); edges.append((rr, hr, a, b))   # valve trains (supply, return)
        halves = [n - n // 2, n // 2]
        for m_i, nm in enumerate(halves):
            if nm == 0:
                continue
            dz = 1.867 / nm
            ps, pr = rs, rr
            q_m = nm * C["lpm_node"] * LPM
            for k in range(nm):
                js, jr = new(), new()
                q_seg = q_m * (nm - k) / nm
                a, b = pipe_coeffs(D_man, dz, q_seg, K=0.2)    # manifold segment + tap tee (run)
                edges.append((ps, js, a, b))                     # supply manifold, flowing up
                edges.append((jr, pr, a, b))                     # return manifold, flowing down (U-type)
                edges.append((js, jr, a_n, b_n)); node_edges.append((r, m_i, k, len(edges) - 1))
                ps, pr = js, jr
    return nxt[0], edges, node_edges

def solve(nJ, edges, p_s, p0=None, tol=1e-6):
    """Junction pressures with p[0] = p_s, p[1] = 0 (Newton, sparse)."""
    E = np.array(edges)
    I, Jn, a, b = E[:, 0].astype(int), E[:, 1].astype(int), E[:, 2], E[:, 3]
    p = np.zeros(nJ) if p0 is None else p0.copy()
    if p0 is None:
        p[:] = 0.5 * p_s
    p[0], p[1] = p_s, 0.0
    free = np.arange(2, nJ)
    for it in range(80):
        dp = p[I] - p[Jn]
        s = np.sqrt(a ** 2 + 4 * b * np.maximum(np.abs(dp), 1e-2))   # 0.01 Pa floor keeps dQ/ddp finite
        Q = np.sign(dp) * np.where(b > 0, (s - a) / (2 * np.where(b > 0, b, 1)), np.abs(dp) / np.where(a > 0, a, 1))
        g = np.where(b > 0, 1 / s, 1 / np.where(a > 0, a, 1))     # dQ/d(dp)
        F = np.zeros(nJ)
        np.add.at(F, I, -Q); np.add.at(F, Jn, Q)                  # net inflow
        rows = np.concatenate([I, I, Jn, Jn]); cols = np.concatenate([I, Jn, I, Jn])
        vals = np.concatenate([-g, g, g, -g])
        Jm = sp.csr_matrix((vals, (rows, cols)), shape=(nJ, nJ))[free][:, free]
        step = spla.spsolve(Jm.tocsc(), -F[free])
        lim = 0.5 * max(abs(p_s), 1.0)
        p[free] += np.clip(step, -lim, lim)                          # damped Newton
        if np.max(np.abs(F[free])) < tol * 1e-3 and np.max(np.abs(step)) < tol:
            break
    dp = p[I] - p[Jn]
    s = np.sqrt(a ** 2 + 4 * b * np.abs(dp))
    Q = np.sign(dp) * np.where(b > 0, (s - a) / (2 * np.where(b > 0, b, 1)), np.abs(dp) / np.where(a > 0, a, 1))
    return p, Q

def solve_flow(nJ, edges, q_target):
    """Find CDU external dp that delivers q_target (secant on log dp)."""
    def qtot(ps, p0=None):
        p, Q = solve(nJ, edges, ps, p0)
        out = sum(Q[k] for k, e in enumerate(edges) if e[0] == 0)
        return out, p, Q
    x0, x1 = 20e3, 60e3
    f0, p, _ = qtot(x0); f1, p, Q = qtot(x1, p)
    for _ in range(40):
        x2 = x1 + (q_target - f1) * (x1 - x0) / (f1 - f0)
        x2 = max(x2, 100.0)
        x0, f0 = x1, f1
        x1 = x2
        f1, p, Q = qtot(x1, p)
        if abs(f1 - q_target) / q_target < 1e-7:
            break
    return x1, p, Q

# -------------------------------------------------------------------------------- CDU pump (C-DAC slide: CHx750 external dp)
# Single pump with filtration, 100 % speed, digitised from C-DAC's "CHx750 External Differential Pressure" chart
# (L/min, bar). Reading uncertainty ~ +/-0.1 bar.
PQ100 = np.array([[0, 4.9], [100, 4.8], [150, 4.7], [200, 4.5], [250, 4.05], [300, 3.45], [350, 2.85], [400, 2.2], [450, 1.6]])

def pump_speed_needed(q_lpm, dp_Pa):
    """Affinity laws: H(Q, s) = s^2 H100(Q/s). Bisection on s."""
    H = lambda q, s: s ** 2 * np.interp(q / s, PQ100[:, 0], PQ100[:, 1]) * 1e5
    lo, hi = 0.02, 1.0
    if H(q_lpm, 1.0) < dp_Pa:
        return float("nan")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if H(q_lpm, mid) > dp_Pa: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)

# -------------------------------------------------------------------------------- thermal per node
_tcache = {}
def node_temps(q_lpm):
    key = round(q_lpm, 4)
    if key not in _tcache:
        m = q_lpm * LPM * RHO
        p1 = s1.cold_plate(165.0, m, T_SUP)
        p2 = s1.cold_plate(165.0, m, p1["T_out"])
        _tcache[key] = (p1["T_case"], p2["T_case"], p2["T_out"])
    return _tcache[key]

def run_case(name, q_total_lpm=None, **kw):
    rack_nodes = C["rack_nodes"]
    q_t = (q_total_lpm or sum(rack_nodes) * C["lpm_node"]) * LPM
    nJ, edges, ne = build(rack_nodes, **kw)
    dp_ext, p, Q = solve_flow(nJ, edges, q_t)
    rows = []
    for (r, m, k, e) in ne:
        q = Q[e] / LPM
        tc1, tc2, tout = node_temps(q)
        rows.append(dict(rack=r + 1, manifold=m, tap=k, q_lpm=q, T_case_2=tc2, T_out=tout))
    q = np.array([x["q_lpm"] for x in rows])
    racks = []
    for r, n in enumerate(rack_nodes):
        rq = np.array([x["q_lpm"] for x in rows if x["rack"] == r + 1])
        rt = np.array([x["T_out"] for x in rows if x["rack"] == r + 1])
        rc = np.array([x["T_case_2"] for x in rows if x["rack"] == r + 1])
        racks.append(dict(rack=r + 1, nodes=n, q_rack_lpm=float(rq.sum()), q_node_mean=float(rq.mean()),
                          q_node_min=float(rq.min()), q_node_max=float(rq.max()),
                          T_return_mixed=float((rq * rt).sum() / rq.sum()), T_case_max=float(rc.max())))
    i_hot = int(np.argmax([x["T_case_2"] for x in rows]))
    W_hyd = q_t * dp_ext
    out = dict(name=name, settings=dict(q_total_lpm=q_t / LPM, **{k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in kw.items()}),
               dp_ext_kPa=dp_ext / 1e3, pump_speed_frac=pump_speed_needed(q_t / LPM, dp_ext),
               W_hyd_W=W_hyd, q_node_min=float(q.min()), q_node_max=float(q.max()), q_node_mean=float(q.mean()),
               q_node_cov=float(q.std() / q.mean()), max_over_min=float(q.max() / q.min()),
               T_case_max=rows[i_hot]["T_case_2"], hottest=dict(rack=rows[i_hot]["rack"], manifold=rows[i_hot]["manifold"],
                                                                tap=rows[i_hot]["tap"], q_lpm=rows[i_hot]["q_lpm"]),
               T_case_uniform=node_temps(q_t / LPM / sum(rack_nodes))[1], racks=racks,
               nodes=[dict(rack=x["rack"], q=round(x["q_lpm"], 4)) for x in rows])
    return out

def balancing_K(K_vt=12.0, D_man=0.025):
    """Extra valve K per rack so every rack's mean node flow = 0.5 L/min (fixed-point on the K's)."""
    extra = np.zeros(len(C["rack_nodes"]))
    for it in range(30):
        res = run_case("bal", K_vt=K_vt, D_man=D_man, extra_K=extra)
        ratio = np.array([rk["q_node_mean"] / C["lpm_node"] for rk in res["racks"]])
        if np.max(np.abs(ratio - 1)) < 0.002:
            break
        # dp across valve ~ (K) Q^2 -> scale the total valve K by ratio^2 (more flow -> more K)
        Ktot = K_vt + extra
        Ktot = Ktot * ratio ** 2
        extra = np.maximum(Ktot - K_vt, 0.0)
        extra -= extra.min()                                    # least-throttled rack keeps its valve open
    return extra, res

def main():
    os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    a, b, err, curve = fit_node_curve()
    print(f"node curve dp = {a:.3e} Q + {b:.3e} Q^2  (max fit error {err*100:.1f} %), dp at 0.5 L/min = "
          f"{(a*0.5*LPM + b*(0.5*LPM)**2)/1e3:.2f} kPa")
    cases = {}
    cases["nominal"] = run_case("Nominal: K_vt 12, manifold 25 mm")
    cases["tight_valves"] = run_case("Valve trains K 30 (strainer fouling / partly closed valves)", K_vt=30.0)
    cases["open_valves"] = run_case("Valve trains K 6", K_vt=6.0)
    cases["small_manifold"] = run_case("Manifold 20 mm ID", D_man=0.020)
    cases["large_manifold"] = run_case("Manifold 32 mm ID", D_man=0.032)
    cases["restrictive_nodes_x5"] = run_case("Node dp x5 (about 15 kPa at 0.5 L/min)", node_scale=5.0)
    cases["restrictive_nodes_x20"] = run_case("Node dp x20 (about 60 kPa at 0.5 L/min)", node_scale=20.0)
    extra, bal = balancing_K()
    bal["name"] = "Balanced: valves trimmed so every rack averages 0.5 L/min per node"
    bal["extra_K"] = extra.tolist()
    cases["balanced"] = bal
    for k, c in cases.items():
        print(f"{k:15s} dp_ext {c['dp_ext_kPa']:6.1f} kPa  pump {c['pump_speed_frac']*100:5.1f} %  node q "
              f"{c['q_node_min']:.3f}-{c['q_node_max']:.3f} L/min (max/min {c['max_over_min']:.2f}, CoV {c['q_node_cov']*100:.1f} %)"
              f"  T_case max {c['T_case_max']:.1f} C (uniform {c['T_case_uniform']:.1f}) hottest rack {c['hottest']['rack']}")
    # turndown sweep (nominal hardware): how far can total flow fall?
    sweep = []
    for qt in [120, 150, 200, 250, 309.5, 350, 400]:
        c = run_case(f"total {qt}", q_total_lpm=qt)
        sweep.append(dict(q_total=qt, dp_ext_kPa=c["dp_ext_kPa"], pump_speed=c["pump_speed_frac"], W_hyd=c["W_hyd_W"],
                          q_node_min=c["q_node_min"], T_case_max=c["T_case_max"], T_ret_max=max(r["T_return_mixed"] for r in c["racks"])))
        print(f"  turndown {qt:6.1f} L/min: dp {c['dp_ext_kPa']:6.1f} kPa, pump {c['pump_speed_frac']*100:5.1f} %, "
              f"W_hyd {c['W_hyd_W']:6.0f} W, min node {c['q_node_min']:.3f}, T_case max {c['T_case_max']:.1f}, "
              f"hottest rack return {sweep[-1]['T_ret_max']:.1f}")
    out = dict(node_curve=dict(a=a, b=b, fit_err=err, **curve), cases=cases, turndown=sweep,
               balancing_extra_K=dict(zip([f"rack{i+1}" for i in range(12)], extra.tolist())))
    json.dump(out, open(RES + "simA_flow_network.json", "w"), indent=1, default=float)
    figs(out)

def figs(out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e3"})
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for key, col in [("nominal", "#2a78d6"), ("tight_valves", "#eb6834"), ("balanced", "#1baf7a")]:
        c = out["cases"][key]
        r = c["racks"]
        x = np.arange(1, 13)
        ax[0].errorbar(x + {"nominal": -0.15, "tight_valves": 0, "balanced": 0.15}[key],
                       [k["q_node_mean"] for k in r],
                       yerr=[[k["q_node_mean"] - k["q_node_min"] for k in r], [k["q_node_max"] - k["q_node_mean"] for k in r]],
                       fmt="o", ms=4, color=col, capsize=2, label=c["name"].split(":")[0])
    ax[0].axhline(0.5, color="#52514e", lw=0.8, ls="--")
    ax[0].set_xticks(range(1, 13)); ax[0].set_xlabel("Rack (nodes: " + ",".join(map(str, C["rack_nodes"])) + ")")
    ax[0].set_ylabel("Node flow, L/min (mean, min-max)"); ax[0].legend(fontsize=7)
    t = out["turndown"]
    ax[1].plot([s["q_total"] for s in t], [s["T_case_max"] for s in t], "o-", color="#e34948", label="Hottest CPU case")
    ax[1].set_xlabel("Total secondary flow, L/min"); ax[1].set_ylabel("°C")
    ax2 = ax[1].twinx(); ax2.plot([s["q_total"] for s in t], [s["W_hyd"] for s in t], "s--", color="#6250d6", label="Hydraulic power")
    ax2.set_ylabel("W (hydraulic)"); ax[1].axvline(309.5, color="#52514e", lw=0.8, ls="--")
    ax[1].legend(loc="upper center", fontsize=7); ax2.legend(loc="upper right", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG + "simA_flow_network.png", dpi=160)

if __name__ == "__main__":
    main()
