# S3: Enclosed-aisle flow network and bypass-recirculation model

Module: `s3_aisle.py` (run `python3 s3_aisle.py`). API per INTERFACE.md: `aisle_mixing`, `solve_for_inlet`, `flow_network`, plus helpers `containment_surface` and `leak_area_from_fraction`. Every function returns a dict of plain floats.

## (a) Governing equations

**Control volumes.** CV1 is the cold side and CV2 is the hot side. In CAC the cold aisle is contained and the room is the hot side. In HAC the hot aisle is contained and the room is the cold side. The IRC draws air from CV2 and discharges it into CV1. The lumped balances below are identical for CAC and HAC. The configuration only changes which surface leaks in the FNM. Streams: cooler $\dot m_c$ at $T_c$, servers $\dot m_s$, leakage $\dot m_b=\beta\dot m_c$ (cold to hot, at $T_c$), recirculation $\dot m_r=\psi\dot m_s$ (hot to cold, at $T_h$).

Mass balances:
$$(1-\beta)\dot m_c+\psi\dot m_s=\dot m_s,\qquad \dot m_s-\dot m_r+\beta\dot m_c=\dot m_c\;\Rightarrow\;\boxed{\dot m_c=\dot m_s\frac{1-\psi}{1-\beta}}$$

Energy balances, with each aisle perfectly mixed (the lumped assumption in Demetriou & Khalifa) and constant $c_p$. Here $\Delta T_s=Q_{air}/(\dot m_s c_p)$:
$$T_{in}=(1-\psi)T_c+\psi T_h,\qquad T_h=T_{in}+\Delta T_s$$
$$\Rightarrow T_{in}=T_c+\frac{\psi}{1-\psi}\Delta T_s,\qquad T_h=T_c+\frac{\Delta T_s}{1-\psi},\qquad \boxed{T_c=T_{in}-\frac{\psi}{1-\psi}\Delta T_s}$$
$$T_{ret}=\frac{(\dot m_s-\dot m_r)T_h+\dot m_bT_c}{\dot m_c}=(1-\beta)T_h+\beta T_c,\qquad Q_{air}=\dot m_c c_p(T_{ret}-T_c)\;\text{(closure, verified to }10^{-11}\,\text{W)}$$

**Passive-scalar view.** Let $c$ be the mass fraction of air that has passed through a server since leaving the coil, so $c=0$ at the IRC outlet and $c=1$ at the server outlet. Mixing conserves $c$ exactly as it conserves enthalpy:
$$c_{in}=\frac{\dot m_r\cdot1+(1-\beta)\dot m_c\cdot0}{\dot m_s}=\psi,\qquad c_{ret}=1-\beta,\qquad T=T_c+c\,(T_h-T_c)$$

**Indices.** Sharma, Bash & Patel (2002) define $\delta Q=\dot m_sc_p(T_{in}-T_c)$, with the supply temperature as the reference:
$$\mathrm{SHI}=\frac{\delta Q}{Q+\delta Q}=\frac{T_{in}-T_c}{T_h-T_c}=\psi,\qquad \mathrm{RHI}=\frac{Q}{Q+\delta Q}=1-\psi,\qquad \mathrm{SHI}+\mathrm{RHI}=\frac{\delta Q+Q}{Q+\delta Q}\equiv1$$
Under perfect mixing SHI equals the exhaust concentration $c_{in}$. The proof is $T_{in}-T_c=\psi(T_h-T_c)$. SHI is independent of β because leakage never reaches a server inlet.

Herrlin's RTI:
$$\mathrm{RTI}=\frac{T_{ret}-T_c}{\Delta T_s}\times100\%=\frac{\dot m_s}{\dot m_c}\times100\%=\frac{1-\beta}{1-\psi}\times100\%$$
RTI is 100 % when ψ = β. It is below 100 % when there is net bypass and above 100 % when there is net recirculation. The cooler flow ratio is $\dot m_c/\dot m_s=(1-\psi)/(1-\beta)$.

**Mixing exergy destruction.** The mixing is adiabatic and isobaric, and air is treated as an ideal gas with constant $c_p$. $X=T_0S_{gen}$ with $T_0$ = 303.15 K:
$$X_{rec}=T_0c_p\left[\dot m_s\ln T_{in}-(1-\beta)\dot m_c\ln T_c-\dot m_r\ln T_h\right],\qquad X_{byp}=T_0c_p\left[\dot m_c\ln T_{ret}-(\dot m_s-\dot m_r)\ln T_h-\dot m_b\ln T_c\right]$$
The pressure term is $X_{thr}=T_0(\dot m_r+\dot m_b)R\ln(1+\Delta P/P_0)$. For $\Delta P$ = 2.5 Pa this is about 0.7 W, against 86 W for thermal mixing, so it is negligible. The FNM reports the same quantity as $\dot m|\Delta P|/\rho$.

**Two-node pressure FNM.** The unknown is $\Delta P=P_{cold}-P_{hot}$.

- Servers use a quadratic net fan curve: $P_h-P_c=\Delta P_{sh}[1-(\dot V/\dot V_f)^2]$, so $\dot m_s=\dot m_{s0}\sqrt{1+\Delta P/\Delta P_{sh}}$.
- Openings (leaks $A_L$ and damper $A_r$) are treated as orifices: $\dot m=C_dA\sqrt{2\rho|\Delta P|}\,\mathrm{sgn}\Delta P$.
- Cold-node balance: $\dot m_c+\dot m_{fan,byp}-\dot m_s(\Delta P)-\dot m_{open}(\Delta P)=0$, solved with Brent's method. Alternatively the solver finds the $\dot m_c$ that holds a set-point $\Delta P$.
- Mapping to the lumped model: if $\Delta P>0$, then $\beta=\dot m_{open}/\dot m_c$. If $\Delta P<0$, then $\beta=0$ and $\psi=(\dot m_{in}+\dot m_{fan})/\dot m_s$.

## (b) Assumptions

| Item | Value | Justification | Source |
|---|---|---|---|
| Aisle mixing | each side perfectly mixed (single T) | Lumped model used to derive the bypass-recirculation optimum. Demetriou & Khalifa used CFD to show that bypass tiles with low-lift fans near the cooler discharge give "nearly thermally uniform" server inlets. Erden/Ebrahimpour report that the well-mixed assumption can err by up to 22 % in cooling power for shallow plenums. | Demetriou & Khalifa 2012 abstract; Ebrahimpour et al. (ScienceDirect S2213138822001722) |
| Leak air temperature | leaves the cold side at $T_c$ | Leak paths sit near the cooler discharge. Exact for ψ = 0, error O(βψ) otherwise. | modelling choice |
| $c_p$ | 1006 J/kg·K, constant | dry air; humidity effect under 1 % | params.CP_AIR |
| Wall heat gain, condensation | none | sensible-only aisle; $T_c$ is checked against the 12 °C dew point | params.ROOM_DEWPOINT_C |
| CAC vs HAC | same balances | With IRCs and a slab floor there are no raised-floor cut-outs. Schneider WP135 notes that CAC with a raised floor leaks about 50 % more than HAC. That does not apply here, but it does favour HAC. | Schneider WP135 |
| Baseline β | 0.05 | Containment leakage is typically 3–10 % (raised-floor leakage 25–50 %). Specifications call for at most 3 % or under 5 %. | Schneider WP135; Schneider blog 2019; Appl. Therm. Eng. case study S2214157X21002471 ("typically 10 % or less") |
| Containment surface | 29.8 m² (roof 5.76 + doors 4.8 + 16 rack faces 19.2) | 8 racks of 600 mm per row, 1.2 m aisle, 2.0 m height | geometry assumption |
| Orifice $C_d$ | 0.61 | sharp-edged crack/gap | ASHRAE Fundamentals ch. 16 (effective leakage area); textbook orifice value |
| Containment ΔP | +1 to +5 Pa (baseline 2.5 Pa) | Common practice is slight positive cold-aisle pressurisation | industry practice (Reddit/operator notes: "keep positive pressure in your cold aisle"). Weak source, flagged. |
| Server net fan shut-off | 150 Pa | 1U high-static fans. **Placeholder for S2.** At a few Pa the server flow changes by under 1 %, so results are insensitive to this value. | assumption |
| Coil feasibility | $T_c\ge T_{chw,in}+3$ K = 17 °C | Placeholder approach until S4's `coil()` provides it | params.BASE T_chw_in = 14 °C |
| m_s, Q_air | 6.0 kg/s, 80 kW | **Placeholders**. S2's `pod_air()` will replace them. Both are function arguments. | task brief |

## (c) Baseline results

Inputs: m_s = 6.0 kg/s, Q_air = 80 kW, T_in = 24 °C, β = 0.05. ΔT_s = 13.25 K.

| ψ | T_c °C | T_h °C | T_ret °C | m_c kg/s | m_c/m_s | SHI | RHI | RTI % | X_rec W | X_byp W | coil OK |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 24.00 | 37.25 | 36.59 | 6.316 | 1.053 | 0.00 | 1.00 | 95.0 | 0 | 86 | yes |
| 0.1 | 22.53 | 37.25 | 36.52 | 5.684 | 0.947 | 0.10 | 0.90 | 105.6 | 197 | 96 | yes |
| 0.2 | 20.69 | 37.25 | 36.43 | 5.053 | 0.842 | 0.20 | 0.80 | 118.8 | 445 | 108 | yes |
| 0.3 | 18.32 | 37.25 | 36.31 | 4.421 | 0.737 | 0.30 | 0.70 | 135.7 | 767 | 125 | yes |
| 0.4 | 15.16 | 37.25 | 36.15 | 3.789 | 0.632 | 0.40 | 0.60 | 158.3 | 1202 | 146 | no |
| 0.5 | 10.75 | 37.25 | 35.93 | 3.158 | 0.526 | 0.50 | 0.50 | 190.0 | 1822 | 178 | no (below dew point) |

- **Limits on ψ.** ψ_max = 0.346 at the coil limit (T_c ≥ 17 °C) and 0.475 at the dew-point limit (T_c > 12 °C).
- **IRC fan power.** By the affinity laws it scales as (1−ψ)³: −27 % at ψ = 0.1 and −49 % at ψ = 0.2. The penalty is that T_c must drop by 3.3 K at ψ = 0.2, which means colder chilled water and a lower chiller COP. That trade-off is the optimum S5 has to find.
- **Coil duty** stays equal to Q_air for every ψ and β.
- **FNM.** 0.5 % open area (1488 cm², 93 cm²/rack) at +2.5 Pa gives β = 0.035. β = 0.05 needs 0.72 % open area (134 cm²/rack). 1 % gives β ≈ 0.07 and 2 % gives β ≈ 0.13. Server flow rises by +0.8 % at +2.5 Pa.
- **Passive recirculation.** ψ = 0.2 at −2 Pa needs about 0.92 m² of opening (576 cm²/rack). That is practical with dampers, but it requires running the IRC below the server flow. A pressurised aisle cannot recirculate passively, so bypass fans are then needed (Demetriou's low-lift fan tiles).

## (d) Figures

- `figures/s3_schematic.png`: CAC and HAC control volumes with all five streams. The same balances apply to both.
- `figures/s3_mixing_vs_psi.png`: required T_c depends on ψ only and hits the coil limit at ψ ≈ 0.35. m_c falls linearly with ψ and rises as 1/(1−β). Leakage dilutes T_ret, which drops about 2.7 K at β = 0.2 and ψ = 0. The fan power ratio is (1−ψ)³.
- `figures/s3_indices.png`: SHI = ψ and RHI = 1−ψ for every β, with the sum identically 1 (markers at β = 0 and 0.2 confirm this numerically). RTI = (1−β)/(1−ψ) crosses 100 % at ψ = β.
- `figures/s3_exergy_map.png`: X_dest,mix over the (ψ, β) plane. Recirculation mixing dominates: 86 W at ψ = 0 and 554 W at ψ = 0.2, which is under 1 % of Q_air. So mixing exergy is a small direct cost. The real cost is the colder T_c the chiller has to produce.
- `figures/s3_fnm.png`: seal quality maps to β for several ΔP set-points. The second axis is cm²/rack, and the 3–10 % typical band is shaded. ΔP is plotted against the IRC supply ratio. β and ψ switch over at m_c = m_s: over-supply gives leakage and under-supply gives hot-air ingress.

## Interfaces to other students

- **S2 → S3.** S2 provides `pod_air(T_in)` → `m_dot_servers` (m_s) and `Q_air_pod`. S3 then returns T_c, T_h and T_ret. T_h from S3 must match S2's `T_out` (S3 uses ΔT_s = Q_air/(m_s c_p)). S2's fan curve should replace `DP_SHUTOFF_SERVER`.
- **S3 → S4.** S4's `coil(m_air = m_c/N_IRC, T_air_in = T_ret, …)` must deliver `T_air_out = T_c_required`, and `solve_chw` should be called with that target. `psi_max_coil` uses a 3 K approach placeholder that S4 should replace with the real coil limit.
- **S3 → S5.** The design variables (ψ, β) map to m_c, T_c and T_ret. X_dest_mix goes into the exergy ledger, and β can be set from `flow_network` (seal area and ΔP set-point).

## Concerns / notes

- The well-mixed assumption gives the best case for recirculation. Real cold aisles stratify, so the hottest inlet will be above the mean T_in.
- Nothing in `params.py` needs changing. `BASE["psi"] = 0` together with β = 0.05 gives RTI = 95 %, which is consistent.
- Containment ΔP is not in `params.py`. S3 uses 2.5 Pa locally.

## (e) Sources actually opened

- Semantic Scholar page for Demetriou & Khalifa 2012, abstract read via scraper: https://www.semanticscholar.org/paper/c2b1e09de2143ba5d85b1a342610daef924ce936. Reports up to 60 % cooling-energy savings and bypass tiles with low-lift fans. The API URL api.semanticscholar.org was blocked by the proxy.
- Demetriou PhD thesis record, Syracuse: https://surface.syr.edu/mae_etd/65 (abstract only).
- Schneider Electric WP135, "Impact of Hot and Cold Aisle Containment…": https://docs.media.bitpipe.com/io_10x/io_105811/item_561497/DBOY-7EDLE8_R2_EN.pdf. Containment leakage 3–10 %, raised floor 25–50 %.
- Schneider blog, "How much air containment is enough" (2019): https://blog.se.com/datacenter/architecture/2019/03/19/getting-it-right-how-much-air-containment-is-enough-in-your-data-center/. Specifications of at most 3 % or under 5 %.
- Search abstract, CRAH bypass assessment: https://www.sciencedirect.com/science/article/abs/pii/S2213138822001722. Covers induced bypass, plenum pressure reversal and the 22 % well-mixed error.
- Search snippet only: https://www.sciencedirect.com/science/article/pii/S2214157X21002471 ("leakage typically 10 % or less").
- Erden, Koz, Yildirim & Khalifa 2017: only the bibliographic record was found (https://research.itu.edu.tr/tr/publications/optimization-of-enclosed-aisle-data-centers-with-induced-crah-byp/fingerprints/). No abstract text was opened.
- Not opened, cited from standard literature: Sharma, Bash & Patel 2002 (AIAA 2002-3091) for SHI/RHI; Herrlin 2008 (ASHRAE Trans.) for RTI; ASHRAE Handbook Fundamentals ch. 16 for $C_d$.
