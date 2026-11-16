# S4: In-row cooler (IRC) heat-exchanger performance

Module: `s4_irc.py` (run `python3 s4_irc.py`). Outputs: `figures/s4_*.png` and `results/s4_baseline.json`.
Scope: one chilled-water in-row cooler with a fin-and-tube coil and DC/EC fans, `params.N_IRC = 4` units per pod. The module also covers the CHW flow and EWT optimisation, a comparison with a passive or active rear-door heat exchanger (RDHX), and the coil exergy destruction.

## (a) Governing equations

**Geometry.** The coil is a staggered fin-and-tube coil. $D_c = D_o + 2t$, $N_{fin} = W/F_p$.

$$A_{fin}=2N_{fin}\left(H\,D_{depth}-N_t\frac{\pi D_c^2}{4}\right),\quad A_a=A_{fin}+N_t\pi D_c(W-N_{fin}t),\quad \sigma=\frac{(P_t-D_c)(F_p-t)}{P_tF_p},\quad D_h=\frac{4A_{min}D_{depth}}{A_a}$$

**Air side.** Wang, Chi & Chang (2000), plain fin, $N\ge2$, with $Re=G_{max}D_c/\mu$.

$$j=0.086\,Re^{P_3}N^{P_4}\left(\tfrac{F_p}{D_c}\right)^{P_5}\left(\tfrac{F_p}{D_h}\right)^{P_6}\left(\tfrac{F_p}{P_t}\right)^{-0.93},\qquad f=0.0267\,Re^{F_1}\left(\tfrac{P_t}{P_l}\right)^{F_2}\left(\tfrac{F_p}{D_c}\right)^{F_3}$$

$P_3=-0.361-0.042N/\ln Re+0.158\ln[N(F_p/D_c)^{0.41}]$, $P_4=-1.224-0.076(P_l/D_h)^{1.42}/\ln Re$, $P_5=-0.083+0.058N/\ln Re$, $P_6=-5.735+1.21\ln(Re/N)$, $F_1=-0.764+0.739P_t/P_l+0.177F_p/D_c-0.00758/N$, $F_2=-15.689+64.021/\ln Re$, $F_3=1.696-15.695/\ln Re$.

$$h_a = C_{cal}\; j\,G\,c_{p,m}\,Pr^{-2/3}$$

**Fin efficiency.** Schmidt equivalent circular fin, staggered layout.

$$\frac{R_e}{r}=1.27\frac{X_M}{r}\sqrt{\frac{X_L}{X_M}-0.3},\quad \phi=\left(\frac{R_e}{r}-1\right)\left[1+0.35\ln\frac{R_e}{r}\right],\quad \eta_f=\frac{\tanh(mr\phi)}{mr\phi},\quad m=\sqrt{\frac{2h_a}{k_ft}},\quad \eta_o=1-\frac{A_{fin}}{A_a}(1-\eta_f)$$

**Water side.** Gnielinski correlation for $3000\le Re\le5\times10^6$, with the Petukhov friction factor. For laminar flow, $Nu=3.66$. Between Re 2300 and 3000 the two are linearly blended.

$$Nu=\frac{(f/8)(Re-1000)Pr}{1+12.7\sqrt{f/8}\,(Pr^{2/3}-1)},\qquad f=(0.79\ln Re-1.64)^{-2}$$

**Overall conductance.**

$$\frac{1}{UA}=\frac{1}{\eta_o h_a A_a}+\frac{\ln(D_o/D_i)}{2\pi k_t L_{tot}}+\frac{R''_f}{A_w}+\frac{1}{h_wA_w}$$

**ε-NTU.** Each tube row is treated as one crossflow pass, with air unmixed and water mixed. The $N$ passes are connected in overall counterflow (Incropera Eq. 11.31a):

$$\varepsilon=\frac{\left(\frac{1-\varepsilon_pC_r}{1-\varepsilon_p}\right)^N-1}{\left(\frac{1-\varepsilon_pC_r}{1-\varepsilon_p}\right)^N-C_r},\qquad \varepsilon_p\big|_{air=C_{min}}=\frac{1}{C_r}\left[1-e^{-C_r(1-e^{-NTU/N})}\right]$$

For $N=4$ this gives ε = 0.5805 at baseline, against 0.5823 for pure counterflow (a +0.3 % difference in Q). The common counterflow approximation is therefore justified, and both arrangements are selectable through `arrangement=`.

**Dry/wet check and wet coil.** The coldest surface temperature is found from a resistance divider at the air-out/water-in end: $T_{s,min}=T_{w,in}+(T_{a,out}-T_{w,in})\frac{R_w}{R_a+R_w}$. If $T_{s,min}<T_{dp}$, the Braun–Mitchell–Klein enthalpy method is applied:

$$c_s=\frac{h_{sat}(T_{w,o})-h_{sat}(T_{w,i})}{T_{w,o}-T_{w,i}},\quad UA^*=\frac{1}{c_{p,m}R_a+c_sR_w},\quad m^*=\frac{\dot m_ac_s}{\dot m_wc_{p,w}},\quad Q=\varepsilon^*\,C^*_{min}\,[h_{a,i}-h_{sat}(T_{w,i})]$$

For a partly wet coil the model takes $Q=\max(Q_{dry},Q_{wet})$. The leaving air state comes from the effective surface state, $T_{a,o}=T_{s,eff}+(T_{a,i}-T_{s,eff})e^{-NTU_a}$.

**Air-side pressure drop.** Kays & London core equation, with Wang's $f$:

$$\Delta P_{coil}=\frac{G^2}{2\rho_i}\left[(1+\sigma^2)\left(\frac{\rho_i}{\rho_o}-1\right)+f\frac{A_a}{A_{min}}\frac{\rho_i}{\rho_m}\right]$$

**Fans.**

$$\Delta P_{tot}=\Delta P_{coil}+\Delta P_{f,0}(V/V_0)^{1.5}+\Delta P_{c,0}(V/V_0)^2,\qquad W_{fan}=\frac{V\,\Delta P_{tot}}{\eta_{fan}}$$

The fans sit downstream of the coil, so $T_{sup}=T_{a,out}+W_{fan}/(\dot m_ac_p)$.

**Water-side pressure drop and pump.**

$$\Delta P_w=f\frac{L_{circ}}{D_i}\frac{\rho v^2}{2}+(n_{pass}-1)K_b\frac{\rho v^2}{2}+K_{ext}V_w^2,\qquad W_{pump}=\frac{V_w(\Delta P_w+\Delta P_{branch})}{\eta_p}$$

**Coil exergy destruction** (Gouy–Stodola, temperatures in K):

$$\dot X_{dest}=T_0\left[\dot m_ac_{p,m}\ln\frac{T_{a,o}}{T_{a,i}}+\dot m_wc_{p,w}\ln\frac{T_{w,o}}{T_{w,i}}\right]+T_0\left[\frac{V_a\Delta P_a}{\bar T_a}+\frac{V_w\Delta P_w}{\bar T_w}\right]$$

**RDHX.** In passive mode the server fans supply the door's pressure drop, so $W_{par}=V\Delta P_{door}/\eta_{srv}$, valid while $\Delta P_{door}\le\Delta P_{avail}$. In active mode a fan matrix supplies it: $W_{par}=V(\Delta P_{door}+\Delta P_{fp})/\eta_{door}$.

## (b) Assumptions

| Quantity | Value | Justification | Source |
|---|---|---|---|
| Reference unit | APC/Schneider InRow RC ACRC301S, 300 mm, CW | Rating tables are published for this 300 mm CW IRC | criticalpowersupplies guide spec; eanixter PD425008 |
| Rated air flow | 1.510 m³/s (3200 CFM), 8 × 200 mm DC mixed-flow fans, 30–100 % speed | Datasheet values | guide spec; Mouser and se.com product pages |
| Rated input power | 805 W at max operating condition | Datasheet value | se.com ACRC301S page |
| Calibration point | EWT 7.2 °C, EAT 37.78/20.7 °C (DB/WB), ΔT_w 6.6 K, 1.13 L/s, 30.4 kW net total, 65 kPa | Highest return-air row closest to the pod's T_ret | eanixter PD425008 |
| Tube | 3/8" (9.525 mm) Cu, wall 0.32 mm | Guide spec gives 9.5 mm Cu tube; wall thickness is typical light-wall | guide spec / assumption |
| Fin | Al, 14 FPI, 0.115 mm thick, k = 200 W/m·K. The real unit uses raised-lance fins, modelled here as plain Wang fins × C_cal | Brief specifies 12–14 FPI. A lanced-fin correlation needs louver geometry we do not have | guide spec (fin type) / assumption |
| Coil face area and rows | 1.80 × 0.25 m, 4 rows, Pt 25.4 mm, Pl 22.0 mm (staggered) | Unit is 1991 × 300 mm. 4 rows is the low end of the 4–6 range and is consistent with the calibrated UA | assumption (brief) |
| Water circuits | 14 parallel circuits, 20 passes, 5 m per circuit | Gives about 1.3 m/s tube velocity at rated flow, a usual coil design range | assumption |
| Water fouling | 4.4×10⁻⁵ m²K/W | Closed, treated CW loop | assumption (TEMA-order value) |
| Room dew point | 12 °C | Set by params | `params.ROOM_DEWPOINT_C` |
| Filter / casing ΔP at rated flow | 25 Pa (V^1.5) / 40 Pa (V²) | 1/2" washable MERV-1 filter; rear grille, fan plate and front discharge | assumption |
| Fan wire-to-air efficiency | **0.322, calibrated** so that rated V·ΔP/η + 50 W controls = 805 W | Brief suggests 0.45–0.6. ebm-papst quotes above 60 % only for large EC radials, and small 200 mm DC fans are lower. Cross-check: the datasheet gross−net difference implies 870 W of fan heat | se.com; ebm-papst mag; cross-check from eanixter rows |
| Controls power | 50 W | Controller, display and sensors | assumption |
| CHW external loss K_ext | 2.62×10¹⁰ Pa/(m³/s)² | Fitted to the 65 kPa rated unit ΔP after subtracting the computed tube ΔP (valve, headers) | eanixter PD425008 |
| CHW branch ΔP / pump η | 60 kPa at 1.13 L/s (∝V²) / 0.55 | Riser-to-unit piping, strainer and isolation. Plant-side ΔP belongs to S5 | assumption |
| Max CW flow | 1.25 × 1.3 L/s = 1.62 kg/s | Guide spec: 1.3 L/s base-load limit, peaks 25 % above | guide spec |
| Server-fan pressure margin (RDHX) | 30 Pa nominal (range 20–40 Pa) | Brief. Dell reports less than 1 % extra fan power for passive doors | Dell equipment-response-pressure white paper |
| η of server fans / door fans | 0.25 / 0.40 | Small 40–60 mm server fans vs. door EC fans | assumption |
| RDHX door coil | 1.9 × 0.55 m, 2 rows, 10 FPI, C_cal = 1. Dense variant: 4 rows, 14 FPI | Low-impedance passive door; validated against nVent passive door ΔP (9/16/28 Pa at 0.96/1.33/1.83 m³/s) | nVent RackChiller RD datasheet |
| Air cp | 1006 J/kg·K + W·1860 (moist) | Set by params; ASHRAE moist-air enthalpy | params / ASHRAE Fundamentals ch. 1 |

## (c) Baseline results

Inputs are the brief's placeholders, passed as arguments: pod air 6.0 kg/s over 4 IRCs (1.5 kg/s each), T_ret 37 °C, target supply 24 °C, EWT 14 °C, dew point 12 °C.

**Geometry (per unit):** A_face 0.45 m², A_air 39.9 m² (of which fins 37.9 m²), A_water 1.95 m² (A_a/A_w = 20.4), A_min 0.260 m² (σ = 0.577), D_h 2.29 mm, 280 tubes.

**Calibration:** C_cal = **0.916**, a single air-side h multiplier. This means plain-fin Wang 2000 with the assumed geometry slightly over-predicts h; a lanced-fin j would be higher, so a 4-row lanced coil with somewhat less area is an equally consistent reading. Across all 13 datasheet rows (EWT 5–10 °C, EAT 26.7–40.5 °C, flow 0.39–1.29 L/s), the gross capacity error averages −0.3 % with a maximum of 2.4 %. The CW ΔP error averages −4 % with a maximum of 12 %.

| Per IRC | Value | Per IRC | Value |
|---|---|---|---|
| Required CHW flow m_w | **0.754 kg/s** | ε / NTU / C_r | 0.581 / 1.05 / 0.49 |
| Coil Q (sensible, SHR = 1) | 20.5 kW | UA | 1611 W/K |
| Coil-leaving / supply air | 23.65 / 24.0 °C | h_air / h_water | 65.8 / 3994 W/m²K |
| LWT / ΔT_w | 20.5 °C / 6.5 K | η_f / η_o | 0.80 / 0.81 |
| Re_Dc / Re_w | 3014 / 7180 | Face velocity | 2.93 m/s |
| Coldest coil surface | 16.4 °C, dry (dew point 12 °C) | Fan speed | 87 % |
| Air ΔP: coil / total | 78.6 / 129 Pa | W_fan | **530 W** |
| CW ΔP (unit) | 29.4 kPa | W_pump (unit + branch) | 77 W |
| X_dest heat / air friction / water friction | 919 / 101 / 23 W | X_dest coil | **1043 W** |

**Pod (4 IRC):** coil Q 81.9 kW (this includes 2.1 kW of fan heat), CHW 3.01 kg/s, LWT 20.5 °C, fans 2.12 kW, CHW branch pumping 0.31 kW, coil X_dest 4.17 kW.

**EWT trade-off (pod):**

| EWT | CHW flow | LWT | Pump power | Coil X_dest |
|---|---|---|---|---|
| 10 °C | 1.79 kg/s | 20.9 °C | 66 W | 4.64 kW |
| 12 °C | 2.22 kg/s | 20.8 °C | 124 W | 4.37 kW |
| 14 °C | 3.01 kg/s | 20.5 °C | 308 W | 4.17 kW |
| 16 °C | 4.99 kg/s | 19.9 °C | 1375 W | 4.27 kW |

The maximum feasible EWT at baseline duty is **16.7 °C**; above it the unit would need more than 1.62 kg/s. LWT stays near 20–21 °C, so lower EWT buys a larger ΔT_w rather than a warmer return.

**Wet-coil check:** at EWT ≥ ~10 °C the coil is fully dry. At EWT 7 °C with the flow solved for the baseline duty, the coldest surface is 13.9 °C, which is still dry. At EWT ≲ 9 °C with a high flow (0.8 kg/s) the surface falls below 12 °C, but the BMK result still gives SHR ≈ 1 at EAT ≥ 37 °C because the entering dew point is low. Latent load appears only at low EAT combined with low EWT (SHR 0.95 at 30 °C EAT, 5 °C EWT).

**RDHX (per rack, 0.375 kg/s, 0.33 m³/s):** the 2-row passive door has ΔP = 1.0 Pa, leaving air at 20.2 °C with 0.25 kg/s of water at 14 °C. Its parasitic load is 1.3 W per rack (21 W per pod), against 2.1 kW for the IRC fans. The 2-row door would need more than 3 m³/s per rack to reach the 30 Pa margin. The dense 4-row/14 FPI door reaches 30 Pa at 1.65 m³/s per rack, and above that an active fan matrix is needed (≈ 120–500 W per rack at 1.6–2.5 m³/s).

## (d) Figures

| File | Takeaway |
|---|---|
| `s4_validation.png` | One multiplier (C_cal = 0.92) reproduces all 13 ACRC301S rating rows within 2.4 %; CW ΔP within 12 %. |
| `s4_capacity_vs_EAT.png` | Capacity is linear in ITD = EAT − EWT. Each +1 K of return air adds ≈ 0.9 kW per unit at 0.8 kg/s, so cold-air bypass (β), which lowers T_ret, directly erodes capacity. Dry curves for all EWTs collapse onto one Q(ITD) line. |
| `s4_chw_map.png` | Required m_w over EWT × target supply. The infeasible wedge (hatched) starts about 7–8 K below the supply target; the wet-coil boundary (orange) only matters for EWT < ~10 °C combined with supply < 22 °C. The right panel shows LWT and ΔT_w contours. |
| `s4_ewt_tradeoff.png` | Pod flow, LWT, ΔT_w, pump power and X_dest vs EWT. Flow and pump power grow steeply above ~15 °C, while LWT is nearly flat at ~20–21 °C. |
| `s4_fan.png` | Coil is 61 % of the IRC air-side ΔP. Fan power ~V^2.6, 530 W at baseline, 760 W at rated flow. |
| `s4_rdhx.png` | At Rudra's low air load a passive door is nearly free (1 Pa). A dense door needs active fans above ~1.65 m³/s per rack. The IRC fans' per-rack share is far larger at equal flow. |
| `s4_exergy_aisle.png` | Coil X_dest is dominated by finite-ΔT heat transfer (~88 %). Using the S3 aisle model, more bypass (β) raises the pod CHW flow (2.9 → 3.6 kg/s for β 0 → 0.3). Recirculation ψ = 0.1 raises it to ~4 kg/s. ψ = 0.2 is infeasible at EWT 14 °C because the required supply T_c drops to ≈ 20.6 °C. |

## API for S3/S5

- `coil(m_air, T_air_in_C, m_w, T_w_in_C, T_dp_C=None, W_in=None, geom=None, arrangement="n_pass")`
- `irc_unit(...)` returns everything above plus `W_fan`, `W_pump_chw`, `T_supply`, `X_dest_coil` and its split.
- `solve_chw(m_air, T_air_in_C, T_air_out_target_C, T_w_in_C)` returns `feasible`, `m_w`, `T_w_out`, `dT_w`, `dP_w`, `W_pump_chw`, `W_fan`, and `reason` (1 = target below EWT, 2 = flow limit).
- `pod_irc(m_air_pod, T_ret_C, T_supply_C, T_w_in_C)` returns pod totals.
- `irc_fan(m_air, T_C)` returns `W_fan` at any flow.
- `rdhx(m_air_rack, T_in, m_w, T_w_in, mode="auto"|"passive"|"active")`.

## Concerns and notes

- **Fan efficiency.** The calibrated wire-to-air η of 0.32 is below the brief's 0.45–0.6, because the 805 W datasheet input forces it. A modern EC retrofit can be tested with `FAN["eta_tot"]=0.45`, which cuts fan power by ~28 %.
- **Fin type.** The real coil uses lanced fins, modelled as plain fins with C_cal. Because the correlation was calibrated at the rating points, extrapolating to a different fin pitch is less reliable.
- **Rating basis.** Ratings were assumed to be at 100 % speed (3200 CFM). The datasheet does not state the air flow per row.
- **Airflow balance.** The baseline 6 kg/s pod flow (~1.32 m³/s per unit, 87 % speed) is fine. However, 81.9 kW of pod air-side load is close to 4 × 20.5 kW. With N_IRC = 4 there is no N+1 redundancy: losing one unit at EWT 14 °C would be infeasible.
- **Shared parameters.** I did not change `params`. A room dew point of 12 °C is reasonable for a controlled room, but in Mumbai monsoon conditions with poor envelope control the dew point can exceed 20 °C. In that case the coil runs wet for all EWT < 18 °C, and S5 should check this.
- **P3 term.** The extracted text we opened (ENCIT 2008) omits the third P3 term of Wang (2000). The original form with that term is used here, and C_cal absorbs any residual difference.

## Sources opened

- Schneider ACRC301S guide specification (coil 3/8" Cu, raised-lance Al fin, counterflow, 8 × 200 mm fans, 30–100 %, 3200 CFM, 1.3 L/s limit): https://www.criticalpowersupplies.co.uk/downloads/apc/1636405034-8405-download3.pdf
- Schneider InRow CW technical specifications (rating tables, ACRC301S): https://objects.eanixter.com/PD425008.PDF
- ACRC301S product pages: https://www.se.com/us/en/product/ACRC301S , https://www.mouser.lt/datasheet/3/187/14/ACRC301S_document.pdf , https://www.avendor.com/products/apc-by-schneider-electric-inrow-rc-300mm-chilled-water-100-240v-50-60-hz-st
- Vertiv Liebert CRV datasheet (cross-check: CRC30 32 kW at 5100 m³/h, 37 °C return, 10/15 °C CHW): https://www.vertiv.com/49e99c/globalassets/products/thermal-management/in-row-cooling/vertiv-liebert-crv-data-sheet-sl-11998.pdf ; https://www.vertiv.com/en-asia/products-catalog/thermal-management/in-row-cooling/liebert-crv-in-row-cooling-system-10-40kw
- Wang, Chi & Chang (2000) correlation as reproduced in ENCIT 2008: https://abcm.org.br/anais/encit/2008/artigos/11_5312.pdf
- Fin-tube areas and Schmidt fin efficiency (ACHP documentation): https://achp.sourceforge.net/_sources/ACHPComponents/FinTubeHX.txt
- Gnielinski correlation and validity: https://tacodoc.readthedocs.io/en/latest/Nusselt.html
- Wet-coil effectiveness and dry/wet analysis (Purdue ME 418 notes): https://www.purdue.edu/freeform/me418/wp-content/uploads/sites/30/2024/10/ME-418-Cooling-Coil-Analysis-and-Design-Annotated.pdf
- EC plug-fan affinity laws and power (Vertiv technical note): https://www.vertiv.com/globalassets/images/about-images/news-and-insights/articles/white-papers/using-ec-plug-fans-to-improve-energy-efficiency/ec-plug-fans-wp-en-na-tn-00005_101604_2.pdf
- EC fan efficiency (ebm-papst): https://mag.ebmpapst.com/en/industries/electronics/comparison-of-fans_9184/
- nVent RackChiller rear door (passive door ΔP vs flow): https://go.nvent.com/rs/760-EGW-100/images/SCHROFF-SLSH-H84154-RackChillerRD-USEN.pdf
- IBM rear-door heat exchanger (passive, server-fan driven): https://www.ibm.com/support/knowledgecenter/POWER8/p8ebe/p8ebe_exchangerspecs.htm
- Dell, server response to rack/door pressure: https://i.dell.com/sites/csdocuments/Business_solutions_whitepapers_Documents/en/us/equipment-response-pressure.pdf

Equations taken from textbooks, not opened online: Incropera & DeWitt (ε-NTU tables, N-pass relation, laminar Nu = 3.66); Kays & London (core ΔP equation); Braun, Mitchell & Klein (1989) ASHRAE Trans. 95(2); Schmidt (1949).
