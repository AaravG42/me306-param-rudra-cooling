# S1 — Processor liquid-cooling loop (direct-to-chip cold plates)

Module: `s1_liquid.py` (run `python3 s1_liquid.py`). Numbers below are at `params.BASE`: PG25, $T_{in}=35$ °C, $\dot m_{plate}=16.63$ g/s (0.98 L/min PG25), $Q_{plate}=0.92\times165=151.8$ W, 576 nodes. All are reproduced in `results/s1_baseline.json`.

## (a) Governing equations

**Coolant figures of merit** (higher is better at equal geometry and velocity):
$$Mo_{turb}=\frac{\rho^{0.8}k^{0.6}c_p^{0.4}}{\mu^{0.4}}\quad(\text{Dittus–Boelter}),\qquad Mo_{lam}=(\rho c_p)^{1/3}k^{2/3}\quad(\text{Lévêque, developing laminar})$$

**Microchannel geometry and flow.** There are $N=W/(w_c+w_f)$ channels.
$$D_h=\frac{2w_cH}{w_c+H},\quad u=\frac{\dot V}{N w_c H},\quad Re=\frac{\rho u D_h}{\mu},\quad Gz=\frac{Re\,Pr\,D_h}{L}$$

**Heat transfer.** The Nusselt number blends the fully developed H1 polynomial (Shah & London) with the thermally developing Lévêque limit, using Churchill–Usagi blending:
$$Nu_{fd}=8.235(1-2.0421\alpha+3.0853\alpha^2-2.4765\alpha^3+1.0578\alpha^4-0.1861\alpha^5),\quad Nu=\left[Nu_{fd}^3+(1.953\,Gz^{1/3})^3\right]^{1/3}$$

**Fins, effective area and fluid resistance:**
$$m=\sqrt{\frac{2h}{k_{Cu}w_f}},\quad \eta_f=\frac{\tanh mH}{mH},\quad A_{eff}=NL(w_c+2\eta_fH),\quad NTU=\frac{hA_{eff}}{\dot m c_p}$$
$$R_{fl}=\frac{1}{\dot m c_p\,\varepsilon},\ \ \varepsilon=1-e^{-NTU}\qquad\big(R_{fl}=R_{cal}+R_{conv,eff},\ R_{cal}=1/\dot m c_p\big)$$

**Base and spreading resistance** (Lee et al. 1995; source = 45° footprint of the die at the IHS top, $A_s$; plate footprint $A_p$):
$$\epsilon=\frac{r_1}{r_2},\ \tau=\frac{t_b}{r_2},\ Bi=\frac{h_{eff}r_2}{k},\ \lambda_c=\pi+\frac{1}{\sqrt\pi\,\epsilon},\ \Phi=\frac{\tanh\lambda_c\tau+\lambda_c/Bi}{1+(\lambda_c/Bi)\tanh\lambda_c\tau},\ R_{sp}=\frac{(1-\epsilon)^{3/2}\Phi}{2\sqrt\pi\,k\,r_1}$$

**Package chain and the junction and case temperatures.** The die-to-IHS term is 1-D with 45° spreading:
$$R_{jc}=\frac{t_{Si}}{k_{Si}A_{die}}+\frac{R''_{TIM1}}{A_{die}}+\frac{t_{IHS}}{k_{Cu}\sqrt{A_{die}A_s}},\qquad R_{TIM2}=\frac{R''_{TIM2}}{A_s}$$
$$T_{case}=T_{in}+Q\,(R_{fl}+R_{base}+R_{TIM2}),\qquad T_j=T_{case}+Q\,R_{jc},\qquad T_{out}=T_{in}+\frac{Q}{\dot m c_p}$$

**Pressure drop.** The apparent friction factor follows Shah (1978), with $K(\infty)$ and $C$ fits from Steinke & Kandlikar (2006):
$$f_{app}Re=\frac{3.44}{\sqrt{L^+}}+\frac{K(\infty)/(4L^+)+fRe-3.44/\sqrt{L^+}}{1+C/L^{+2}},\qquad L^+=\frac{L}{D_hRe}$$
$$\Delta P=\Big(4f_{app}\frac{L}{D_h}+K_c+K_e\Big)\frac{\rho u^2}{2}+K_{ports}\frac{\rho u_{port}^2}{2},\qquad \dot W_{ideal}=\dot V\Delta P$$

**Exergy destruction** (dead state $T_0=303.15$ K). $T_s=T_j$ is the default; $T_s=T_{case}$ is also reported. $\bar T$ is the log-mean bulk temperature.
$$\dot X_{heat}=T_0\Big[\dot m c_p\ln\frac{T_{out}}{T_{in}}-\frac{Q}{T_s}\Big],\qquad \dot X_{fric}=T_0\frac{\dot m\,\Delta P}{\rho\,\bar T},\qquad Be=\frac{\dot X_{heat}}{\dot X_{heat}+\dot X_{fric}}$$

**Loop.** The QD term uses $\Delta P_{QD}=SG\,(Q_{gpm}/C_v)^2$. Rack, row and CDU terms are $\Delta P_{ref}(\dot V/\dot V_{ref})^2$.
$$\Delta P_{loop}=\Delta P_{plates}+\Delta P_{QD}+\Delta P_{hose}+\Delta P_{rack}+\Delta P_{row}+\Delta P_{CDU}$$
$$\dot W_{pump}=\frac{\dot V_{tot}\Delta P_{loop}}{\eta_{w2w}},\qquad T_{ret}=T_{in}+\frac{Q_{liq}+\dot V\Delta P}{\dot m_{tot}c_p},\qquad \dot X_{pump}=\dot W_{pump}-\dot V\Delta P\Big(1-\frac{T_0}{\bar T}\Big)$$

## (b) Assumptions

| Item | Value | Justification | Source |
|---|---|---|---|
| CPU TDP | 165 W | Intel ARK spec | Intel ARK 6240R |
| $T_{case,max}$ | 90 °C | Intel ARK "T_CASE" | Intel ARK 6240R |
| $T_{j,max}$ | 100 °C | Not on ARK. Intel quotes DTS_max 102 °C for the sibling Gold 6244. 100 °C keeps 2 K margin | Intel support art. 000057788 |
| Package | 76.0 × 56.5 mm LGA3647 | ARK | Intel ARK |
| Die | XCC, 694 mm² (≈21.6 × 32.1 mm) | 24-core Cascade Lake SKUs use the XCC die (estimate) | AnandTech forum / WikiChip estimate |
| TIM1 | soldered, 0.05 K·cm²/W | Indium STIM on Xeon SP; 0.02–0.08 typical | engineering assumption |
| IHS | Cu, 3 mm; 45° spread → $A_s$ = 13.8 cm² | standard spreading-angle rule | assumption |
| TIM2 | 0.08 K·cm²/W | PTM7950 datasheet gives 0.04; doubled for production bond line and ageing | Caplinq PTM7950 page |
| Cold plate | Cu (k = 390), 100 channels $w_c$ = 0.3, $w_f$ = 0.2, H = 2.5, L = W = 50 mm, base 2 mm, 6 mm ports | Typical skived microchannel server plate. C-DAC plate geometry is unpublished | assumption (task brief) |
| Minor losses | $K_c$ = 0.5, $K_e$ = 1.0 (channels); $K_{ports}$ = 2.0 (ports) | small σ, abrupt area change | Kays & London |
| Capture fraction | 0.92 of package power | Remainder goes through the socket and PCB to air. The NREL Asetek test captured 48–64 % of *server* heat with CPU-only plates, which is consistent with ~90 %+ of CPU heat | NREL/TP-2C00-62905 (fraction is assumption) |
| Plate arrangement | 2 plates **in parallel** per node (baseline). Series is also computed | Parallel gives equal $T_{in}$ to both sockets and needs no balancing | — |
| QD | 2 × UQD04 per node, $C_v$ = 1.12 | OCP UQD standard size | FITOK UQD datasheet |
| Node hoses | 6 mm ID, 1.5 m total, ΣK = 3 | assumption | — |
| Rack manifold | 10 kPa @ 72 L/min/rack | 5–15 kPa typical | assumption |
| Row piping + valves + strainer | 20 kPa @ ref. flow | assumption | — |
| CDU secondary | 60 kPa @ ref. flow | A liquid-to-liquid CDU held an 8.7 psi differential on a PG25 TCS | NSF PAR 10435660 |
| Pump wire-to-water | 0.55 | motor × VFD × hydraulic, 0.5–0.6 | assumption (brief) |
| Dielectric | Novec 7500 correlations | CoolProp lacks it | 3M Novec 7500 product info (SLAC copy) |
| Property pressure | 2 bar abs | TCS static pressure | — |

The reference flow for the loop terms is 1 L/min per plate, i.e. 2 L/min per node.

## (c) Baseline results

**Single cold plate** (PG25, 35 °C, 0.98 L/min, 151.8 W):

| Quantity | Value |
|---|---|
| $T_j$ / $T_{case}$ / $T_{out}$ | **45.1 / 41.2 / 37.3 °C** |
| $R_{total}$ (j → inlet) | **0.066 K/W** (inside the 0.05–0.12 sanity band) |
| $R_{case \to inlet}$ (plate R incl. TIM2) | 0.041 K/W |
| R breakdown | $R_{jc}$ 0.0256 · TIM2 0.0076 · base+spread 0.0155 · $R_{fl}$ 0.0176 ($R_{cal}$ 0.0152 + conv 0.0024) |
| Flow (Re, Nu, h) | Re 79 (laminar), Nu 7.1, h = 6.4 kW/m²K |
| Fin efficiency / NTU | η_f 0.76 / NTU 2.0 |
| Hydraulics | ΔP 2.78 kPa, ideal pumping 0.045 W |
| Exergy | $\dot X_{heat}$ 4.17 W, $\dot X_{fric}$ 0.045 W, Be 0.989 |

Cross-check: the Wakefield on-chip plate is 0.027 K/W (base → inlet, 1.6 L/min, 45 °C). Our base → inlet value is 0.033 K/W at 1 L/min, which is consistent.

**Flow sweep at 35 °C:**

| L/min | R_total K/W | ΔP kPa | Be | X_heat W |
|---|---|---|---|---|
| 0.2 | 0.133 | 0.45 | 0.9998 | 6.48 |
| 0.5 | 0.083 | 1.28 | 0.998 | 4.78 |
| 1.0 | 0.066 | 2.84 | 0.989 | 4.16 |
| 2.0 | 0.058 | 6.58 | 0.947 | 3.86 |
| 3.0 | 0.055 | 11.1 | 0.873 | 3.75 |

Over the whole map (0.2–3 L/min, 17–50 °C), the maximum $T_{case}$ is 66.2 °C and the maximum $T_j$ is 70.1 °C. The plate could take at least 374 W (2.3 × TDP) before $T_{case}$ = 90 °C.

**Fluid comparison** at 1 L/min, 35 °C:

| Fluid | R_total K/W | T_case °C | ΔP kPa | Mo_lam vs water | ρc_p vs water |
|---|---|---|---|---|---|
| Water | 0.0643 | 40.9 | 1.53 | 1 | 1 |
| PG25 | 0.0661 | 41.1 | 2.84 | 0.84 | 0.97 |
| EG25 | 0.0665 | 41.2 | 2.46 | 0.84 | 0.94 |
| Novec 7500 | 0.116 | 48.8 | 2.29 | 0.17 | 0.44 |

**Pod loop, 576 nodes, 303.6 W/node to liquid** ($Q_{liq}$ = 174.9 kW):

| | parallel, 1 L/min/plate (BASE) | series, 1 L/min/node |
|---|---|---|
| Loop flow | 1130 L/min | 565 L/min |
| ΔP_loop | 101 kPa (CDU 58, row 19, rack 10, hose 8.6, QD 3.0, plates 2.8) | 29.6 kPa |
| W_pump (elec.) | **3.46 kW** | 0.51 kW |
| T_return | 37.3 °C | 39.6 °C |
| T_case,max / T_j,max | 41.2 / 45.1 °C | 43.5 / 47.4 °C |
| X_dest cold plates | 4.86 kW | 4.82 kW |
| X_dest pumping | 3.42 kW | 0.50 kW |

The fair comparison is parallel at 0.5 L/min/plate, which has the same node flow as series at 1 L/min/node. Parallel then gives 0.47 kW pump power and T_case,max 43.7 °C, against 0.55 kW and 43.4 °C for series. The two are practically equivalent, so parallel is kept.

## (d) Figures

| Figure | Takeaway |
|---|---|
| `figures/s1_coolant_props.png` | PG25 keeps 84 % of water's laminar h-merit but has ~2× the viscosity. Novec 7500 has 17 % of the merit (k ≈ 0.065 W/m·K). |
| `figures/s1_temperature_map.png` | $T_{case}$ never approaches 90 °C for 17–50 °C inlet and 0.2–3 L/min. Above ~0.5 L/min, inlet temperature sets the die temperature, not flow. The third panel shows the allowable plate heat, with the 90 °C contour mapped as 1/2/3 × TDP. |
| `figures/s1_resistance_vs_flow.png` | Above ~1 L/min the flow-independent chain ($R_{jc}$ + TIM2 + base ≈ 0.048 K/W) dominates. Below 0.5 L/min the caloric term dominates. |
| `figures/s1_pressure_pumping.png` | Plate ΔP is 2.8 kPa at 1 L/min. Plate hydraulic power is ≪ 1 W. |
| `figures/s1_exergy_map.png` | Destruction is thermal: Be ≥ 0.99 at ≤ 1 L/min and 0.83 at 3 L/min, 17 °C. It falls with flow (smaller ΔT) and with inlet T (higher temperature level). |
| `figures/s1_loop.png` | Pod pump power scales ∝ $\dot V^3$: 3.5 kW at 1 L/min/plate versus 0.47 kW at 0.5. Return T is 37–40 °C. |

**Design implication:** run ~0.5 L/min/plate (≈3.3 L/min/kW) with the warmest feasible supply (W40/W45). This keeps $T_{case}$ ≤ 54 °C at 45 °C inlet with 0.47 kW of pumping.

## (e) Sources opened

- Intel ARK, Xeon Gold 6240R specifications: https://www.intel.com/content/www/us/en/products/sku/199343/intel-xeon-gold-6240r-processor-35-75m-cache-2-40-ghz/specifications.html
- Intel support, max temperature for Gold 6244 (DTS_max 102 °C): https://www.intel.com/content/www/us/en/support/articles/000057788.html
- XCC die ~694 mm² (WikiChip/AnandTech estimates quoted): https://forums.anandtech.com/threads/cpus-physical-dimension.2561079/post-39734840
- 3M Novec 7500 product information: https://confluence.slac.stanford.edu/download/attachments/158050956/3M%2BNovec%2B7500%2BEngineered%2BFluid-1.pdf?api=v2
- PTM7950 TIM, 0.04 °C·cm²/W: https://caplinq.com/ptm7950-most-thermally-conductive-phase-change-pad-with-very-low-thermal-impedance-that-works-in-vertical-applications-ptm7950.html
- ASHRAE W17–W45/W+ classes: https://www.upsite.com/blog/major-changes-to-ashraes-fifth-edition-of-thermal-guidelines-part-3-liquid-cooling-chapter-updates/
- Wakefield on-chip cold plate, 0.0267 °C/W @ 1.6 LPM: https://www.mouser.in/new/wakefield/wakefield-thermal-on-chip-liquid-cold-plates/
- OCP Cold Plate Test Spec v1.0 (R definition; 45 °C test fluid): https://www.opencompute.org/wp-content/uploads/2026/10/OCP-Cold-Plate-Test-Specification-V1.0.0.pdf
- PG25 TCS rack test, CDU ΔP 8.7 psi: https://par.nsf.gov/servlets/purl/10435660
- Cold-plate literature summaries (impingement plate, 350 W, ≤ 23 kPa): https://par.nsf.gov/biblio/10537555 and https://par.nsf.gov/biblio/10058150-experimental-characterization-cold-plates-used-cooling-multi-chip-server-modules-mcm
- FITOK UQD Cv data: https://www.fitok.com/tmp/file/Liquid_Cooling_Universal_Quick_Disconnect_UQD_Series_EN.pdf
- JetCool manifold/QD guidance (~1.5 LPM/kW): https://jetcool.com/post/what-are-the-design-considerations-for-liquid-manifolds-and-quick-disconnects-in-data-center-cooling/
- NREL Asetek RackCDU test (48–64 % of server heat to liquid): https://www.nrel.gov:443/docs/fy15osti/62905.pdf
- Textbook correlations (not opened online): Shah & London (1978); Steinke & Kandlikar (2006); Lee et al. (1995); Churchill & Usagi (1972); Kays & London; Incropera.

## Concerns about shared parameters

1. `BASE.m_liq_plate = 1/60*0.998` uses water density. With PG25 (ρ = 1017) it is 0.98 L/min, which is harmless but inconsistent.
2. 1 L/min/plate = 6.6 L/min/kW, about 4× industry practice (~1.5 LPM/kW). It drives 3.5 kW of pumping, about 2 % of the liquid heat. I recommend that S5 optimise over 0.3–1 L/min/plate.
3. `LIQ_SUPPLY_RANGE` caps at 45 °C. The plates stay feasible to 50 °C+, so S5 may explore W+ for free cooling.
4. $T_{j,max}$ (100 °C) is an assumption because Intel does not publish it on ARK for this SKU.
