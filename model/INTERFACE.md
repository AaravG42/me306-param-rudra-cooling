# Project contract — Hybrid-cooled enclosed-aisle pod, PARAM Rudra (IIT Bombay)

Python 3, numpy/scipy/matplotlib, CoolProp.
Import shared constants from `params.py` (never hard-code a value that lives there). SI units; temperatures in
°C in function arguments/returns unless suffixed `_K`. Exergy uses dead state `params.T0_K` (303.15 K, 101.325 kPa).

## System picture
One enclosed-aisle pod: 2 rows × 8 racks × 36 nodes = 576 1U Rudra nodes (2× Xeon Gold 6240R each).
- Liquid path: CPU → cold plate (microchannel, Cu) → rack manifold → CDU (plate HX) → facility water
  (cooling tower when warm enough, else chiller). Coolant default: PG25 (CoolProp `INCOMP::APG-25%`), alt. water.
- Air path: everything not captured by cold plates (DIMMs, VRMs, PCH, NIC/HCA, SSD, BMC, PSU loss, residual CPU
  heat) → server fans → hot side → aisle network (leakage β, intentional bypass-recirculation ψ) → in-row
  coolers (IRC, chilled-water coil + EC fans) → cold aisle.
- Chilled water from an electric chiller rejecting to a cooling tower.

## Module contract (every function returns a plain dict of floats; vectorisation optional)

### S1 `s1_liquid.py`
- `coolant_props(T_C, fluid="PG25") -> {rho, cp, k, mu, Pr, nu}`
- `cold_plate(Q_W, m_dot, T_in_C, fluid="PG25") -> {T_j, T_case, T_out, R_total, R_conv, R_cond, dP, W_pump_ideal, X_dest_total, X_dest_heat, X_dest_fric, ...}`
  Q_W = heat into ONE cold plate (one CPU); m_dot kg/s through that plate.
- `liquid_loop(Q_liq_node_W, m_dot_plate, T_in_C, n_nodes, fluid="PG25") -> {Q_liq, T_return, m_dot_total, dP_loop, W_pump (electrical, incl. manifolds+CDU secondary), X_dest_coldplates, X_dest_pumping, T_j_max, T_case_max, feasible}`

### S2 `s2_air.py`
- `node_heat_breakdown(load=1.0) -> {component: W, ..., Q_node_total, Q_cpu_package, Q_liquid, Q_air}`
  (load = CPU utilisation 0..1; must state cold-plate capture fraction).
- `server_air(T_in_C, Q_air_W, V_dot=None) -> {V_dot, m_dot, T_out, dT, W_fan, fan_speed_frac, T_component_max, limiting_component, feasible}`
  If `V_dot` None → use the fan control law (minimum flow that keeps all air-cooled components within limits,
  with a floor on fan speed). Per node.
- `pod_air(T_in_C, load=1.0) -> {m_dot_servers, Q_air_pod, T_out, W_fans_pod, ...}` (576 nodes).

### S3 `s3_aisle.py`
- `aisle_mixing(m_s, Q_air, T_c, psi, beta, config="CAC") -> {T_in, T_h, T_ret, m_c, m_r, m_b, SHI, RHI, RTI, X_dest_mix, ...}`
  m_s total server air flow, Q_air air-side heat (W), T_c cooler supply temperature.
- `solve_for_inlet(m_s, Q_air, T_in_target, psi, beta, config="CAC")` → same dict + required `T_c`.
- `flow_network(...)` pressure-based FNM mapping leakage area / damper opening → (β, ψ). Document it.

### S4 `s4_irc.py`
- `coil(m_air, T_air_in_C, m_w, T_w_in_C) -> {Q, T_air_out, T_w_out, eps, NTU, UA, Cr, dP_air, dP_w, dry, ...}` per IRC unit.
- `irc_unit(m_air, T_air_in_C, m_w, T_w_in_C) -> {... + W_fan, W_pump_chw}`
- `solve_chw(m_air, T_air_in_C, T_air_out_target_C, T_w_in_C) -> {m_w, ...}` required water flow (or infeasible).
- RDHX passive/active variant + parasitic fan power (task list).
- `X_dest_coil` (exergy destroyed in the coil, both streams).

### S5 `s5_plant.py` (plant + integration)
- `chiller(Q_evap_W, T_chw_supply_C, T_cond_water_in_C) -> {W, COP, X_dest}`; `cooling_tower(...)`; `cdu(...)`.
- `system(x) ` integrates S1–S4 for a design vector x = (T_in_server, psi, beta, T_liq_in, m_liq_plate, T_chw_in, ...)
  → full energy + exergy ledger, COP, constraint flags. Optimiser + maps.

## Deliverables per agent
1. The module file with docstrings, every equation commented with its source.
2. A self-test `if __name__ == "__main__":` that prints a baseline table and saves figures to `figures/sN_*.png`
   (matplotlib, dpi 200, clean style: white background, no chart junk, labelled axes with units, finding in title).
3. `results/sN_summary.md`: (a) governing equations in LaTeX, (b) assumptions table (value, justification, source),
   (c) key numeric results at baseline, (d) figure list with one-line takeaway each, (e) sources actually opened (URLs).
4. `results/sN_baseline.json`: key numbers.
Do not modify `params.py` or other students' files. If you believe a shared parameter is wrong, say so in your summary.
