# S2: Air-side board components and server fans (PARAM Rudra 1U node)

Module: `s2_air.py` (run `python3 s2_air.py`). Public API per INTERFACE.md: `node_heat_breakdown(load)`,
`server_air(T_in_C, Q_air_W, V_dot=None)`, `pod_air(T_in_C, load)`. Every function returns a plain dict of floats.
Baseline = `params.BASE` (server inlet 24 °C, full CPU load, 576 nodes).

## (a) Governing equations

CPU package power versus utilisation $u$ (linear model with an idle floor):
$$P_{cpu} = n_s\,\mathrm{TDP}\,[f_{idle} + (1-f_{idle})\,u],\qquad f_{idle}=0.30$$

Split between liquid and air ($\phi$ = cold-plate capture fraction, $\chi_D$ = DIMM liquid fraction, 0 at baseline):
$$Q_{liq} = \phi\,P_{cpu} + \chi_D P_{DIMM},\qquad Q_{air} = P_{wall} - Q_{liq}$$

Voltage-regulator and PSU losses:
$$Q_{VR} = P_{out}\left(\tfrac{1}{\eta_{VR}}-1\right) + n_s P_{VR,fix},\qquad
P_{wall} = \frac{P_{DC}}{\eta_{PSU}\!\left(P_{DC}/(2\cdot 750\,\mathrm{W})\right)}$$

Air-side energy balance through each serial zone $k$ (drive bay, fan wall, DIMM/VRM, PCH/HCA, PSU):
$$T_{k+1} = T_k + \frac{Q_k}{\dot m\,c_p},\qquad \Delta T_{node} = \frac{Q_{air}}{\dot m c_p}$$

Component surface temperature from Newton cooling, with $h$ from laminar flat-plate flow:
$$T_{s} = T_{a,local} + \frac{Q}{h A \eta_f},\qquad \overline{Nu}_L = 0.664\,Re_L^{1/2}Pr^{1/3},\qquad
T_j = T_s + Q_{dev}R_{jc}$$

DIMM hotspot (the trailing-edge DRAM; JEDEC $T_{case}$ applies to the hottest device). The model uses the local
uniform-heat-flux value $Nu_x = 0.453\,Re_x^{1/2}Pr^{1/3}$, so $h_L = (0.453/0.664)\,\bar h$. The DIMM's own channel
air is heated as $T_{ch,out} = T_{in,ch} + Q_{DIMM}/(\rho u A_{ch} c_p)$.

Fan wall: 8 fans in parallel at speed fraction $N$ (affinity laws $V\propto N$, $\Delta P\propto N^2$, $P\propto N^3$):
$$\Delta P_{fan} = N^2 \Delta P_0\left[1 - a x - (1-a)x^2\right],\quad x=\frac{V}{N\,n V_0},\qquad
\Delta P_{sys} = kV^2$$
$$W_{fan} = n P_{rated}\left[c_0 + (1-c_0)N^3\right]$$
Because both curves scale with $N^2$, the operating point is $V = N\,V_{op}(1)$.

Control law:
$$N^* = \min\{N\in[N_{floor},1]:\ \text{all margins}\ge 0\}\quad\text{(bisection; margins are monotonic in }N)$$

Capacity-rate ratio and specific flow:
$$C_r = \frac{\dot m_{air} c_{p,air}}{\dot m_{liq} c_{p,liq}},\qquad \frac{\dot m_{air}}{Q_{air}}=\frac{1}{c_p\Delta T}$$

Exergy, simple form (Gouy–Stodola), dead state $T_0$ = 303.15 K:
$$\dot X_{d,HT} = T_0\sum_i Q_i\left(\frac{1}{T_{a,i}}-\frac{1}{T_{s,i}}\right),\qquad \dot X_{d,fan} = W_{fan}\ \text{(fan work fully dissipated)}$$
$$\dot X_{air} = \dot m c_p\left[(T_{out}-T_{in}) - T_0\ln\frac{T_{out}}{T_{in}}\right]$$

## (b) Assumptions

| Quantity | Value | Justification | Source |
|---|---|---|---|
| CPU TDP | 165 W × 2 | from params | params / Intel ARK |
| Idle CPU fraction | 0.30 TDP | 2×6226R server active idle = 26 % of full-load wall power | SPECpower result 20220523-01177 |
| **Cold-plate capture φ** | **0.92** | Remaining 8 % leaves through the LGA pins and substrate into the PCB. Same value as S1 `CAPTURE_FRAC`. Bounded by 85–90 % of node heat (Lenovo SD650, DIMMs also on water) and up to 98 % (Aquarius). | Lenovo Press LP0636; NREL |
| DIMM power | 1.8 W idle, 3.8 W active (16 GB 1Rx4 RDIMM) | Micron IDD3N 0.72 A, IDD4R 2.556 A, IDD4W 2.214 A at 1.2 V, plus RCD/VPP/refresh. Cross-check: about 2 W per 8 GB measured at the wall. | Micron MTA18ASF2G72PZ datasheet; ServeTheHome |
| CPU VR efficiency | 0.90 (+1.5 W fixed per socket) | 12→1.8 V POL modules are 86–90 % efficient; multiphase stages exceed 95 % per stage | Bel Fuse VRP4; TI CSD95490Q5MC |
| Memory VR efficiency | 0.88 | 81–86 % for 1.2 V modules; multiphase is better | Bel Fuse VRP4 |
| PCH | 8–12 W | ARK TDP is 15 W, but that is worst case with every I/O in use | Intel ARK C621 |
| HCA | 24.9 W (80 % at idle) | ConnectX-7 single-port NDR OSFP, typical, passive cable (ConnectX-6 HDR: 19.3 W) | NVIDIA CX-7 / CX-6 manuals |
| SSD | 0.61–3.0 W | 800 GB SATA: idle 0.61 W, read 2.9 W, write 6.3 W | Intel DC S3610 spec |
| BMC / misc | 4 W / 8 W | Engineering estimate | Assumption |
| PSU | 2×750 W Platinum, current sharing | 230 V internal-redundant minimums: 88/90/94/91 % at 10/20/50/100 % load | 80 PLUS table (Wikipedia); Supermicro PWS-751P-1R |
| Fans | 8 × Delta GFB0412EHS 40×40×56 mm | Each: 31.18 CFM free air, 558 Pa shut-off, 16.2 W. A C621 2S 1U uses 8×40×56 fans. | Delta GFB datasheet; Supermicro SYS-1029U-E1CRTP |
| Fan-curve shape | a = 0.5 | The datasheet P–Q chart is not machine-readable. End points are datasheet values. | Assumption |
| Fan floor | 25 % | BMC fan tables typically floor at 20–30 % | Assumption |
| System k | 9.98×10⁴ Pa/(m³/s)² (80 Pa at 60 CFM) | A 1U with cold plates instead of tall CPU sinks sits at the low end of 0.3–0.5 inH₂O. Square law for turbulent flow. | Comair Rotron note; **estimate** |
| Flow area | 0.43×0.04 m, porosity 0.6 | 1U internal cross-section | Assumption |
| DIMM channel | 12×31 mm per DIMM (1 DPC) | 8 mm slot pitch, every other slot empty | Assumption |
| Limits | DIMM 85 °C; SSD 70 °C; VRM T_j 110 °C; PCH 100 °C; HCA air 55 °C; HCA case 105 °C; PSU inlet 50 °C | Above 85 °C, DDR4 needs 2× refresh. SSD case limit 0–70 °C. HCA rated 0–55 °C ambient and needs 250 LFM at 35 °C / 350 LFM at 55 °C. **The PCH, VRM and HCA case limits are assumptions.** | Advantech/Micron RDIMM; Intel S3610; Micron 5300; NVIDIA CX-6 |

## (c) Baseline results (inlet 24 °C, 100 % load)

Per-node heat breakdown:

| Item | W |
|---|---|
| CPU packages (2) | 330.0, of which 303.6 to liquid and 26.4 to air |
| DIMMs (12) | 45.6 |
| CPU VR loss / memory VR loss | 39.7 / 6.2 |
| PCH / HCA / SSD / BMC / misc | 12.0 / 24.9 / 3.0 / 4.0 / 8.0 |
| Fans (25 % speed) | 5.9 |
| PSU loss (η = 0.916) | 44.0 |
| **Node wall power** | **523.2** |
| **Q_liquid / Q_air** | **303.6 (58.0 %) / 219.6 (42.0 %)** |

At idle the node draws 208 W, 56 % of it to air.

Chassis air side:

- **Flow:** 31.3 CFM, 0.0175 kg/s, 21.7 Pa, 25 % speed (at the fan floor).
- **Temperatures:** ΔT = 12.4 K, giving 36.4 °C exhaust.
- **Components (24 °C inlet):** the hottest DIMM is at 82.0 °C. VRM T_j is 81.3 °C, PCH 53.1 °C, HCA case 66.3 °C (local air 34.0 °C), SSD 38.3 °C.
- **HCA airflow:** 286 LFM core velocity against the 250 LFM requirement.
- **Fan operating point at full speed:** 125 CFM at 348 Pa.
- **Air/liquid split:**
  - Capacity-rate ratio C_air/C_liq = 0.134 (17.6 W/K against 131.4 W/K for 2 × 1 L/min PG25).
  - Specific flow: 80 g/s of air per kW of air heat.
- **Exergy per node:** heat transfer destroys 10.7 W; fan work destroys 5.9 W.

Pod (576 nodes):

| Quantity | Value |
|---|---|
| Server air flow | 10.10 kg/s = 8.50 m³/s = 18,020 CFM |
| Air heat / liquid heat / IT power | 126.5 kW / 174.9 kW / 301.4 kW |
| Exhaust temperature | 36.4 °C |
| Server-fan power | 3.37 kW |
| Exergy destroyed, chassis heat transfer | 6.13 kW |
| Exergy destroyed, fans | 3.37 kW |

Server inlet sweep at full load (per-node flow, pod fan power):

| Inlet | Per-node flow | Pod fan power | What sets the flow |
|---|---|---|---|
| 18–27 °C | 31.3 CFM | 3.37 kW | fan floor |
| 30 °C | 34.7 CFM | 3.78 kW | DIMM T_case |
| 35 °C | 41.6 CFM | 4.9 kW | DIMM T_case |
| 40 °C | 50.9 CFM | 7.1 kW | DIMM T_case |
| 45 °C | 77 CFM | 19 kW | PSU inlet limit |

Board thermal-management scenarios, at 24 °C (air share) and 32 °C (required flow):

| Scenario | Air share at 24 °C | Flow at 32 °C |
|---|---|---|
| Baseline | 42.0 % | 37.2 CFM |
| VR efficiency 0.95 / 0.93 | 39.3 % | 37.2 CFM |
| Titanium PSU | 39.9 % | 37.2 CFM |
| DIMM cold plates (90 %) | 34.1 % | 31.3 CFM (at the floor) |

Raising φ from 0.85 to 0.98 lowers the air share from 46 % to 38 % but leaves the required flow unchanged. Residual CPU heat goes into the PCB, while the flow is set by the DIMM's own channel.

**Sanity check against the brief:**
- Node power of 523 W is inside the 450–550 W window.
- ΔT of 12.4 K is inside the 10–15 K window.
- The air share of 42 % is slightly above the 25–40 % window. Two reasons:
  - All conversion losses go to air: VR losses are 46 W and PSU loss is 44 W.
  - The HCA is NDR-class at 25 W.

  A Titanium PSU (39.9 %) or better VRs (39.3 %) bring it into range.

## (d) Figures

- `figures/s2_node_breakdown.png`: full-load ledger plus load sweep. The air share falls from 56 % at idle to 42 % at full load, because liquid heat scales with CPU power while board loads are nearly fixed.
- `figures/s2_fan_curve.png`: fan wall (8×GFB0412EHS) at 25–100 % speed against the chassis curve. The baseline needs only 31 CFM, which is 25 % speed and the floor.
- `figures/s2_fan_vs_inlet.png`: required flow, pod fan power and exhaust temperature for inlets of 18–45 °C at 40/70/100 % load.
  - Flow stays at the floor up to about 27 °C, then is set by DIMM T_case.
  - Fan power is ×1.45 at 35 °C and ×5.7 at 45 °C (full load). This cubic penalty is what S5's optimiser must trade against chiller savings.
- `figures/s2_zone_temps.png`: air temperature by zone and component temperatures at 24 and 32 °C inlet. The DIMMs and the VRMs are the binding components.
- `figures/s2_air_liquid_split.png`: air share, required flow and C_air/C_liq against capture fraction for four board-management scenarios. DIMM cold plates are the only measure that reduces the required air flow.

## (e) Sources opened (2026-10-13)

- Intel ARK C621 chipset: https://www.intel.com/content/www/us/en/products/sku/97338/intel-c621-chipset/specifications.html
- Delta GFB 40×40×56 mm datasheet: https://www.mouser.com/datasheet/2/632/Delta_GFB40x40x56mm-C_DS-515721.pdf
- NVIDIA ConnectX-6 VPI specifications: https://networking-docs.nvidia.com/connectx6vpihw/specifications
- NVIDIA ConnectX-7 specifications: https://networking-docs.nvidia.com/connectx7hw/specifications
- Micron 16 GB DDR4 RDIMM (MTA18ASF2G72PZ) datasheet: https://www.mouser.com/datasheet/2/671/asf18c2gx72pz-1284542.pdf
- Advantech DDR4 RDIMM datasheet (T_C limits, 2× refresh): https://advdownload.advantech.com/productfile/PIS/AQD-D4U16R32-SEW/file/AQD-D4U16R32-SEW_datasheet20230616174904.pdf
- ServeTheHome, DDR4 DIMM power: https://www.servethehome.com/?p=20509
- Intel SSD DC S3610 specification: https://www.mouser.com/datasheet/2/612/ssd-dc-s3610-spec-769417.pdf
- Micron 5300 product brief: https://www.simms.co.uk/PDFS/Products/SSD/5300_product_brief.pdf
- 80 Plus efficiency table: https://en.wikipedia.org/wiki/80_Plus
- Supermicro PWS-751P-1R: https://store.supermicro.com/nl_nl/pws-751p-1r.html
- Supermicro SYS-1029U-E1CRTP: https://www.supermicro.com/en/products/system/1u/1029/sys-1029u-e1crtp.php
- Comair Rotron, air flow vs pressure: https://comairrotron.com/article/establishing-cooling-requirements-air-flow-vs-pressure.html
- tec-science, Nusselt correlations for plates: https://www.tec-science.com/thermodynamics/heat-transfer/calculation-of-the-nusselt-numbers-for-forced-flows-over-plates-and-in-pipes/
- Bel Fuse VRP4-80A1Ax datasheet: https://www.belfuse.com/resources/datasheets/powersolutions/ds-BPS-vrp4-80a1ax-series.pdf
- TI CSD95490Q5MC: https://www.ti.com/product/CSD95490Q5MC
- Lenovo Press SD650 (85–90 % of heat to water): https://lenovopress.lenovo.com/lp0636-thinksystem-sd650-direct-water-cooled-server-xeon-sp-gen-1
- NREL Aquarius test (up to 98 % capture): https://www.nrel.gov/grid/news/program/2019/aquarius-hpc-cooling-system-yields-impressive-energy-efficiency-results-after-nrel-sandia-test
- SPECpower result, 2×Gold 6226R: https://spec.org/power_ssj2008/results/res2022q2/power_ssj2008-20220523-01177.txt

Not found or not public: the Intel C620 PCH T_case limit (100 °C assumed), the ConnectX maximum power and LFM tables for CX-7 (behind NVOnline), and the GFB P–Q curve points.

## Concerns about shared parameters and coupling

1. **S3 `DP_SHUTOFF_SERVER = 150 Pa`** is a placeholder. The real fan wall has a shut-off pressure of 558·N² Pa, which is about 35 Pa at the 25 % floor and 558 Pa at full speed. S3 should use `s2_air.fan_dp` / `operating_point` for any server-flow-versus-aisle-ΔP coupling.
2. **`BASE['m_liq_plate']` = 1 L/min per plate** gives C_liq = 131 W/K per node, so the liquid ΔT is only 2.3 K at 303.6 W. That is a very high flow for a 152 W plate and inflates pumping power. Something like 0.4–0.5 L/min (ΔT ≈ 5 K) may suit the optimiser better. This is for S1/S5 to decide.
3. **Capture fraction** stays consistent only while both modules keep 0.92 (S1 hard-codes it). S5 should pass `node_heat_breakdown()['Q_liquid']` to S1 rather than calling S1's default.
4. **NODES_PER_POD = 576** includes 90 high-memory nodes in reality (768 GB, 24 DIMMs). Their DIMM heat would be about 2× and they would hit the DIMM limit earlier. Here all nodes are modelled as 192 GB.
5. **Least certain inputs:** the system resistance k, fan-curve shape, DIMM channel geometry and VRM/HCA sink geometry are estimates. The fan-power-versus-inlet curve is most sensitive to the DIMM channel width and the 85 °C limit.
