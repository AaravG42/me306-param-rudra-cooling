"""S5 - stepwise path from baseline to the recommended operating point (writes results/s5_steps.json)."""
import json, numpy as np, s5_plant as S, s5_optimize as O
pts = {
 "A baseline": dict(T_in=24.0, psi=0.0, T_chw=14.0, T_liq_in=35.0, lpm=0.5),
 "B best CHW only": dict(T_in=24.0, psi=0.0, T_liq_in=35.0, lpm=0.5),
 "C + bypass recirc (T_in 24)": dict(T_in=24.0, psi=None, T_liq_in=35.0, lpm=0.5),
 "D + T_in 27": dict(T_in=27.0, psi=None, T_liq_in=35.0, lpm=0.5),
 "E + W40 liquid 0.35 L/min": dict(T_in=27.0, psi=None, T_liq_in=40.0, lpm=0.35),
}
res = {}
for k, c in pts.items():
    base = {**S.DEFAULT_X, **{a: b for a, b in c.items() if b is not None}}
    if 'T_chw' in c: r = S.system(base)
    elif c['psi'] == 0.0: r = O.best_chw(base)
    else:
        best = None
        for psi in np.arange(0, 0.301, 0.005):
            rr = O.best_chw({**base, 'psi': float(psi)})
            if rr and (best is None or rr['W_cool_total'] < best['W_cool_total']): best = rr
        r = best
    res[k] = dict(x={a: (float(b) if not isinstance(b, str) else b) for a, b in r['x'].items()}, W=r['W_cool_total'],
        Wfac=r['W_cool_facility'], COP=r['COP_cool'], COPf=r['COP_facility'], pPUE=r['pPUE'],
        Wb={a: float(b) for a, b in r['W'].items()}, X={a: float(b) for a, b in r['X'].items()}, Xtot=r['X_dest_total'],
        Xin=r['X_in'], Xcl=r['X_closure'], Ecl=r['E_closure'], Xcross=r['X_cross'], T_c=r['T_c'], T_ret=r['T_ret'],
        m_c=r['m_c'], m_w=r['m_w'], T_w_out=r['T_w_out'], COPch=r['COP_chiller'], PLR=r['PLR'], f_econ=r['f_econ'],
        T_case=r['T_case_max'], T_tcs_ret=r['T_tcs_ret'], X_tcs=r['X_tcs_return'], P_it=r['P_it_wall'],
        P_compute=r['P_compute'], SHI=r['SHI'], RTI=r['RTI'], irc_speed=r['irc_speed'], n_ch=r['n_chillers'])
    v = res[k]; print(k, 'psi', round(v['x']['psi'], 3), 'chw', round(v['x']['T_chw'], 2), 'W', round(v['W']), 'COP', round(v['COP'], 3), 'pPUE', round(v['pPUE'], 4), 'Ecl', round(v['Ecl'], 3), 'Xcl', round(v['Xcl'], 2))
json.dump(res, open('results/s5_steps.json', 'w'), indent=1, default=float)
