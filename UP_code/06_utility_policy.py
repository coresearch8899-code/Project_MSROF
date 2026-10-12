"""
MSROF-CRRU | Script 6 | PV-only export-policy and utility-perspective analysis (Utilities Policy version)
Re-uses the model of script 02 (sections 1-7) with the digester switched off and adds:
  A  PV-only capacity-NPV sweeps for export rates 0-3.0 THB/kWh and three PV capital-cost levels
  B  coincidence curve, Eq. (5) on the 250 kWp grid, decision map and load-shape transfer (PV only)
  C  test of Eq. (5) under five tariff structures
  D  benchmark against annual, monthly and average-day sizing
  E  compensation schemes: uncompensated export and zero export with curtailment (current rules), and the counterfactual
     net billing at 1.00 and 2.20 THB/kWh and monthly net metering; itemized year-1 utility account (self-consumed,
     delivered and curtailed energy, retail revenue lost, export payments, avoided purchases)
  H  cost-ordered capacity sequence;  I  near-optimal band;  J  joint institution-utility view and break-even value
  F  Monte Carlo at the fixed design (10,000 draws, seed 20261007; correlated variant, seed 20261008), one-at-a-time
     sensitivity, synthetic weather years (500, seed 7) and the observed 12-month windows
  G  capacity-value curve with marginal IRR;  K  block decisions and stopping rule against the global maximum
  L  50 kWp step test of the optima and of Eq. (5);  M  PV-model error propagation (measured 2025 profile)
  N  worked example of the two accounts (monthly net metering)
  O  Ft sensitivity;  P  block bootstrap of weather years (seed 8);  Q  typical (climatological) year
  R  propagation of short-term PV error to export credits, demand charges and the optimum (200 simulations, seed 9)
Outputs: outputs/up_*.csv, outputs/up_results.json, outputs/up_framework.npz, outputs/up_mc.npy, outputs/up_mc_inputs.npz,
         outputs/up_wy.npy, outputs/up_wy_windows.npy (script 09 then re-optimizes the capacity in each Monte Carlo draw)
"""
import json, numpy as np, pandas as pd
SRC = open("02_msrof_analysis.py", encoding="utf-8").read()
exec(SRC.split("# 8. MONTE CARLO")[0].rsplit("# ====", 1)[0])
BG0 = False
PTS = pts                                           # 0, 250..8000 step 250, then 16 points up to the full potential
XG = np.round(np.arange(0.0, 3.01, 0.1), 2)
def eco(p, th=None): return economics(*allocate(p), th or {}, bg=BG0)

# ---------------- A. sweeps
rows = []
for cx in (0.8, 1.0, 1.2):
    for x in XG:
        for p in PTS:
            e = eco(p, {"exp_thb": x, "capex_pv": cx})
            rows.append(dict(capex=cx, exp_thb=x, p_kwp=p, npv=e["npv"], irr=e["irr"], sc_kwh=e["sc_kwh"], ex_kwh=e["ex_kwh"], pv_kwh=e["pv_kwh"]))
SW = pd.DataFrame(rows)
OPT = SW.loc[SW.groupby(["capex", "exp_thb"]).npv.idxmax()].reset_index(drop=True)
def x_for(cap_kw, cx):
    o = OPT[(OPT.capex == cx) & (OPT.p_kwp >= cap_kw)]; return float(o.exp_thb.min()) if len(o) else np.nan
THRESH = {f"{cx:.1f}": dict(x_3mw=x_for(3000, cx), x_roof=x_for(T0["roof"]["kwp"] - 1, cx), p_x0=float(OPT[(OPT.capex == cx) & (OPT.exp_thb == 0)].p_kwp.iloc[0]),
                            p_x22=float(OPT[(OPT.capex == cx) & (OPT.exp_thb == 2.2)].p_kwp.iloc[0])) for cx in (0.8, 1.0, 1.2)}
P0 = THRESH["1.0"]["p_x0"]
BASE = OPT[OPT.capex == 1.0].reset_index(drop=True)

# ---------------- B. framework (PV only: load = metered load)
C_PV = eco(0)["lcoe"] if False else economics({"roof": 1000.0, "ground": 0, "ponds": 0, "reservoir": 0}, T0, bg=False)["lcoe"]
GROOF = PV_LAND * DT; LD = LOAD.copy()
PG = np.arange(0, 40001, 250.0)
def sig_curve(load):
    S = np.array([np.minimum(p * GROOF, load).sum() for p in PG]); G = PG * GROOF.sum()
    return (G[1:] + G[:-1]) / 2 / LD.sum(), np.diff(S) / np.diff(G)
RD, SIG = sig_curve(LD)
e0 = eco(0); e3 = eco(3000)
V_R = (e3["sav1_usd"] - e0["sav1_usd"]) / (e3["sc_kwh"] - e0["sc_kwh"]); R_R = V_R / C_PV
def p_star(Rx, Rr=None, sig=None):
    Rr = R_R if Rr is None else Rr; sig = SIG if sig is None else sig
    if Rx >= 1: return np.inf
    idx = np.flatnonzero(sig < (1 - Rx) / (Rr - Rx)); return float(PG[idx[0]]) if len(idx) else float(PG[-1])
YL = GROOF.sum()
rd_of = lambda p: p * YL / LD.sum() if np.isfinite(p) else np.inf
AN = pd.DataFrame([dict(exp_thb=x, Rx=x / FX / C_PV, p_rule=p_star(x / FX / C_PV), p_sim=float(BASE[BASE.exp_thb == x].p_kwp.iloc[0])) for x in XG])
AN["rd_rule"] = AN.p_rule.map(rd_of); AN["rd_sim"] = AN.p_sim.map(rd_of)
_hour = TIDX.hour.values; _wk = np.repeat((CLS.reshape(365, 96) == 0).any(axis=1), 96)
SH = {"CRRU metered": LD, "Flat (24 h)": np.full(NSTEP, LD.sum() / NSTEP)}
_o = np.where(_wk & (_hour >= 8) & (_hour < 17), 1.0, 0.15); SH["Office hours"] = _o / _o.sum() * LD.sum()
_e = np.where((_hour >= 17) & (_hour < 23), 1.0, 0.35); SH["Evening peak"] = _e / _e.sum() * LD.sum()
SHC = {k: sig_curve(v) for k, v in SH.items()}
SHOPT = {k: {f"{x:.2f}": rd_of(p_star(x / FX / C_PV, None, SHC[k][1])) for x in (0.0, 1.0, 2.2)} for k in SH}
RROPT = {f"{r:.2f}": rd_of(p_star(0.0, r)) for r in (1.15, R_R, 1.6, 2.0)}
MAP_RX = np.linspace(0, 1.4, 141); MAP_RD = RD[RD <= 5.0]
MAP_M = np.array([[R_R * s + rx * (1 - s) - 1 for rx in MAP_RX] for s in SIG[RD <= 5.0]])

# ---------------- C. tariff structures
GRID = np.arange(0, 6001, 250.0)
base_t = dict(e_on=P["e_on"], e_off=P["e_off"], dem=P["dem"])
w_on = LOAD[CLS == 0].sum() / LOAD.sum(); e_avg = w_on * base_t["e_on"] + (1 - w_on) * base_t["e_off"]
VAR = {"PEA TOU with demand charge, base": {}, "TOU, no demand charge": dict(dem=0.0), "TOU, demand charge ×2": dict(dem=2 * base_t["dem"]),
       "Flat energy price with demand charge": dict(e_on=e_avg, e_off=e_avg),
       "TOU spread ×1.5 with demand charge": dict(e_on=e_avg + 1.5 * (base_t["e_on"] - e_avg), e_off=e_avg + 1.5 * (base_t["e_off"] - e_avg))}
RV = []
for name, mod in VAR.items():
    P.update(base_t); P.update(mod)
    a0 = eco(0); a3 = eco(3000); vr = (a3["sav1_usd"] - a0["sav1_usd"]) / (a3["sc_kwh"] - a0["sc_kwh"]); rr = vr / C_PV
    for x in (0.0, 1.0, 2.0, 2.5):
        npv = np.array([eco(p, {"exp_thb": x})["npv"] for p in GRID]); ps = float(GRID[npv.argmax()]); pr = p_star(x / FX / C_PV, rr)
        RV.append(dict(variant=name, R_r=rr, exp_thb=x, p_rule=pr, p_sim=ps, dp_kwp=pr - ps, npv_forgone=max(0.0, npv.max() - float(np.interp(pr, GRID, npv))), npv_opt=npv.max()))
P.update(base_t); RV = pd.DataFrame(RV)

# ---------------- D. benchmark
_ml = np.array([LD[MONTH == m].sum() for m in range(12)]); _my = np.array([GROOF[MONTH == m].sum() for m in range(12)])
P_ANN = LD.sum() / YIELD_LAND; P_MON = float(np.min(_ml / _my))
_h = TIDX.hour.values
_Lmh = np.array([[LD[(MONTH == m) & (_h == h)].mean() * 4 for h in range(24)] for m in range(12)])
_Gmh = np.array([[GROOF[(MONTH == m) & (_h == h)].mean() * 4 for h in range(24)] for m in range(12)])
_onf = np.array([[(CLS[(MONTH == m) & (_h == h)] == 0).mean() for h in range(24)] for m in range(12)])
_pr = (_onf * P["e_on"] + (1 - _onf) * P["e_off"] + P["ft"]) / FX
_days = np.array([np.sum(MONTH == m) / 96 for m in range(12)])
AF = float((((1 - P["deg"]) ** np.arange(1, P["N"] + 1)) / (1 + P["r"]) ** np.arange(1, P["N"] + 1)).sum())
def adb(p, x):
    G = p * _Gmh; S = np.minimum(G, _Lmh); X = G - S; return float((_days[:, None] * (S * _pr + X * x / FX)).sum())
BP = np.arange(250, 12001, 250.0)
def bench(x):
    full = {p: eco(p, {"exp_thb": x}) for p in BP}; pm = max(full, key=lambda p: full[p]["npv"])
    est = {p: full[p]["npv"] + (adb(p, x) - full[p]["sav1_usd"]) * AF for p in BP}; pa = max(est, key=lambda p: est[p])
    out = {}
    for lab, p in (("Annual energy balance", P_ANN), ("Monthly energy balance", P_MON), ("Average-day hourly profile", pa), ("MSROF (15-min, full tariff)", pm)):
        e = eco(p, {"exp_thb": x})
        out[lab] = dict(p_kwp=float(p), npv=e["npv"], capex=e["capex"], loss=e["npv"] - full[pm]["npv"], extra_capex=e["capex"] - full[pm]["capex"], sc_share=(e["pv_kwh"] - e["ex_kwh"]) / e["pv_kwh"])
    return out
BENCH = {f"{x:.2f}": bench(x) for x in (0.0, 1.0, 2.2)}

# ---------------- E. compensation schemes and utility perspective
_bill0 = bill
def bill_nem(imp, ex, th=None):
    """monthly net metering: exports credited at the TOU energy rate (plus Ft) of their interval, up to the month's energy charge"""
    r = _bill0(imp, ex, dict(th or {}, exp_thb=0.0))
    rate = np.where(CLS == 0, P["e_on"], P["e_off"]) + P["ft"]
    cr_m = np.add.reduceat(ex * rate, MSTART)
    on = CLS == 0
    e_m = np.add.reduceat(np.where(on, imp * P["e_on"], imp * P["e_off"]), MSTART) + np.add.reduceat(imp, MSTART) * P["ft"]
    cr = np.minimum(cr_m, e_m).sum()
    r["credit"] = cr; r["total"] = r["total"] - cr
    return r
SCHEMES = {"Uncompensated export (status quo)": ("nb", 0.0), "Zero export with curtailment": ("curt", 0.0),
           "Net billing, 1.00 THB/kWh (2021 pilot, up to 200 kW)": ("nb", 1.0),
           "Net billing, 2.20 THB/kWh (counterfactual)": ("nb", 2.2), "Monthly net metering (counterfactual)": ("nem", None)}
W_LIST = (2.0, 2.2, 2.7)                                  # avoided wholesale purchase cost, THB/kWh (sensitivity range)
def run_scheme(kind, x, p):
    global bill
    bill = bill_nem if kind == "nem" else _bill0           # "curt": surplus is curtailed; the institution's bill is that of zero remuneration
    try: e = eco(p, {"exp_thb": x or 0.0})
    finally: bill = _bill0
    return e
def account(kind, e):
    """year-1 utility account; exported energy that is curtailed is neither resold nor counted as an avoided purchase"""
    S, X = e["sc_kwh"], e["ex_kwh"]; Xd = 0.0 if kind == "curt" else X; Xc = X - Xd
    pay = e["credit_usd"]; lost = e["sav1_usd"] - pay
    out = dict(gen_mwh=e["pv_kwh"] / 1e3, sc_mwh=S / 1e3, ex_mwh=Xd / 1e3, curt_mwh=Xc / 1e3, bill_reduction=e["sav1_usd"],
               retail_loss=lost, retail_loss_energy=e["sav_energy_usd"], retail_loss_demand=e["sav_demand_usd"], export_payment=pay)
    for w in W_LIST:
        out[f"avoided_sc_{w:.1f}"] = (w / FX) * S; out[f"avoided_ex_{w:.1f}"] = (w / FX) * Xd
        out[f"comp_sc_{w:.1f}"] = (w / FX) * S - lost; out[f"comp_ex_{w:.1f}"] = (w / FX) * Xd - pay
    out["avoided"] = {f"{w:.1f}": (w / FX) * (S + Xd) for w in W_LIST}
    out["dM"] = {f"{w:.1f}": (w / FX) * (S + Xd) - e["sav1_usd"] for w in W_LIST}
    return out
SCH, SCH_CURVE = {}, []
for lab, (kind, x) in SCHEMES.items():
    res = {p: run_scheme(kind, x, p) for p in PTS}
    popt = max(res, key=lambda p: res[p]["npv"])
    for p, e in res.items():
        if p <= 12000:
            SCH_CURVE.append(dict(scheme=lab, p_kwp=p, npv=e["npv"], **{f"dM_w{w:.1f}": v for w, v in zip(W_LIST, account(kind, e)["dM"].values())}))
    e = res[popt]
    SCH[lab] = dict(kind=kind, p_kwp=popt, npv=e["npv"], irr=e["irr"], pv_mwh=e["pv_kwh"] / 1e3, base_bill=e["base_bill_usd"], **account(kind, e))
SCH_CURVE = pd.DataFrame(SCH_CURVE)

# ---------------- H. cost-ordered capacity sequence (sensitivity): ground, small ponds, reservoir, rooftop
def allocate_cost(p_kwp):
    T = tiers(); a = {}; left = p_kwp
    for k in ("ground", "ponds", "reservoir", "roof"):
        a[k] = min(left, T[k]["kwp"]); left -= a[k]
    return {k: a[k] for k in ("roof", "ground", "ponds", "reservoir")}, T
def run_scheme_c(kind, x, p):
    global bill
    bill = bill_nem if kind == "nem" else _bill0
    try: e = economics(*allocate_cost(p), {"exp_thb": x or 0.0}, bg=False)
    finally: bill = _bill0
    return e
PTS_C = np.arange(0, 12001, 250.0)
SCH_C = {}
for lab, (kind, x) in SCHEMES.items():
    res = {p: run_scheme_c(kind, x, p) for p in PTS_C}
    popt = max(res, key=lambda p: res[p]["npv"]); e = res[popt]
    SCH_C[lab] = dict(kind=kind, p_kwp=popt, npv=e["npv"], irr=e["irr"], pv_mwh=e["pv_kwh"] / 1e3, **account(kind, e))

# ---------------- I. near-optimal band: capacities whose NPV is within 1% of the maximum (base cost, net billing)
BAND = {}
for x in (0.0, 1.0, 2.2):
    d = SW[(SW.capex == 1.0) & (SW.exp_thb == x) & (SW.p_kwp <= 8000)]
    ok = d[d.npv >= 0.99 * d.npv.max()].p_kwp
    BAND[f"{x:.2f}"] = [float(ok.min()), float(ok.max())]

# ---------------- J. joint institution-utility view (total resource cost relative to avoided purchases)
AF_DEG = float((((1 - P["deg"]) ** np.arange(1, P["N"] + 1)) / (1 + P["r"]) ** np.arange(1, P["N"] + 1)).sum())
for lab, d in SCH_C.items():
    gen_pv = (d["sc_mwh"] + d["ex_mwh"]) * 1e3 * AF_DEG        # delivered PV energy (curtailed energy carries no external value)
    d["joint"] = {w: d["npv"] + AF_DEG * d["dM"][w] for w in d["dM"]}
    d["b_ext_thb_kwh"] = {w: max(0.0, -d["joint"][w]) / gen_pv * FX for w in d["dM"]}
for lab, d in SCH.items():
    gen_pv = (d["sc_mwh"] + d["ex_mwh"]) * 1e3 * AF_DEG
    d["joint"] = {w: d["npv"] + AF_DEG * d["dM"][w] for w in d["dM"]}
    d["b_ext_usd_kwh"] = {w: max(0.0, -d["joint"][w]) / gen_pv for w in d["dM"]}
    d["b_ext_thb_kwh"] = {w: v * FX for w, v in d["b_ext_usd_kwh"].items()}
    d["co2_price_usd_t"] = {w: v / (P["ef_grid"] / 1000) for w, v in d["b_ext_usd_kwh"].items()}

# ---------------- F. uncertainty at the PV-only optimum (x = 0)
rng = np.random.default_rng(20261007)
def tn(mu, sd, n, lo=None, hi=None): return np.clip(rng.normal(mu, sd, n), lo, hi)
NMC = 10000
MC_IN = dict(capex_pv=tn(1, .2 / 1.645, NMC, .4), om=tn(1, .3 / 1.645, NMC, .2), r=tn(.07, .02 / 1.645, NMC, .02),
             eta_scale=tn(1, 0.07 / 1.645, NMC, .7, 1.3), solar=tn(1, .10 / 1.645, NMC, .6),
             tariff_energy=tn(1, .1 / 1.645, NMC, .5), tariff_demand=tn(1, .1 / 1.645, NMC, .5), load=tn(1, .10 / 1.645, NMC, .6), salvage=rng.uniform(0, .1, NMC))
np.savez("outputs/up_mc_inputs.npz", **MC_IN)
aP, TP = allocate(P0)
MCR = np.array([[(e := economics(aP, TP, {k: float(v[i]) for k, v in MC_IN.items()}, bg=False))["npv"], e["lcoe"], e["irr"]] for i in range(NMC)])
TOR = {"PV capital cost (±20%)": ("capex_pv", .8, 1.2), "Discount rate (5% / 9%)": ("r", .05, .09), "Energy tariff (±10%)": ("tariff_energy", .9, 1.1),
       "Demand charge (±10%)": ("tariff_demand", .9, 1.1), "PV efficiency (±7%)": ("eta_scale", .93, 1.07), "Solar resource (±10%)": ("solar", .9, 1.1),
       "Campus load (±10%)": ("load", .9, 1.1), "O&M (±30%)": ("om", .7, 1.3), "Salvage value (0% / 10%)": ("salvage", 0, .1),
       "Export remuneration (0 / 1.00 THB/kWh)": ("exp_thb", 0.0, 1.0)}
TORN = {k: dict(low=economics(aP, TP, {a: lo}, bg=False)["npv"], high=economics(aP, TP, {a: hi}, bg=False)["npv"]) for k, (a, lo, hi) in TOR.items()}
for v in TORN.values(): v["swing"] = abs(v["high"] - v["low"])
# correlated variant: energy and demand tariffs move together (rho = 1, one tariff revision); PV capital cost and O&M rho = 0.5
_z = np.random.default_rng(20261008).standard_normal((NMC, 2))
MC_CORR = dict(MC_IN); MC_CORR["tariff_demand"] = MC_IN["tariff_energy"].copy()
MC_CORR["om"] = np.clip(1 + .3 / 1.645 * (0.5 * (MC_IN["capex_pv"] - 1) / (.2 / 1.645) + np.sqrt(1 - .25) * _z[:, 0]), .2, None)
MCR_C = np.array([economics(aP, TP, {k: float(v[i]) for k, v in MC_CORR.items()}, bg=False)["npv"] for i in range(NMC)])
YBY = PREP["yield_by_year"]
# serial dependence of monthly yield anomalies (ratio to the mean of the same calendar month), Mar 2023 to Aug 2026
_keys = sorted(YBY); _mm = {m: np.mean([YBY[k] for k in _keys if int(k[5:]) == m]) for m in range(1, 13)}
_an = pd.Series({pd.Timestamp(k + "-01"): YBY[k] / _mm[int(k[5:])] - 1 for k in _keys}).asfreq("MS")
_a0, _a1 = _an.values[:-1], _an.values[1:]; _ok = np.isfinite(_a0) & np.isfinite(_a1)
AC1 = dict(r=float(np.corrcoef(_a0[_ok], _a1[_ok])[0, 1]), n=int(_ok.sum()), n_months=int(np.isfinite(_an.values).sum()))
from scipy.stats import pearsonr as _pr_
AC1["p"] = float(_pr_(_a0[_ok], _a1[_ok])[1])
_fac = {m: [YBY[f"{y}-{m + 1:02d}"] / YBY[f"2025-{m + 1:02d}"] for y in (2023, 2024, 2025, 2026) if f"{y}-{m + 1:02d}" in YBY] for m in range(12)}
rngw = np.random.default_rng(7); NW = 500
WS = np.array([[rngw.choice(_fac[m]) for m in range(12)] for _ in range(NW)])
WS[0] = [YBY[f"2024-{m + 1:02d}"] / YBY[f"2025-{m + 1:02d}"] for m in range(12)]
PW = np.arange(250, 4001, 250.0)
def wy(ms):
    n = economics(aP, TP, {"mscale": ms}, bg=False)["npv"]; o = []
    for x in (0.0, 2.2):
        v = [eco(p, {"mscale": ms, "exp_thb": x})["npv"] for p in PW]; o.append(float(PW[int(np.argmax(v))]))
    return [n, o[0], o[1], float((ms[MONTH] * GROOF).sum() / GROOF.sum())]
WY = np.array([wy(ms) for ms in WS])
# alternative that keeps the observed month-to-month sequence: every contiguous 12-month window of the record
_mon = pd.period_range("2023-03", "2026-08", freq="M")
WIN = [[YBY[str(p)] / YBY[f"2025-{p.month:02d}"] for p in _mon[i:i + 12]] for i in range(len(_mon) - 11)]
WIN_MS = [np.array([w[[p.month for p in _mon[i:i + 12]].index(m + 1)] for m in range(12)]) for i, w in enumerate(WIN)]
WYW = np.array([wy(ms) for ms in WIN_MS])
pc = lambda a: dict(p10=float(np.percentile(a, 10)), p50=float(np.percentile(a, 50)), p90=float(np.percentile(a, 90)))

# ---------------- G. PV-only capacity-value curve with marginal IRR (x = 0)
rows, prev = [], None
for p in PTS:
    e = economics(*allocate(p), full=True, bg=False)
    r_ = dict(p_kwp=p, npv=e["npv"], lcoe=e["lcoe"], irr=e["irr"], capex=e["capex"], sc_share=e["sc_kwh"] / max(e["pv_kwh"], 1e-9), load_cover=e["sc_kwh"] / e["load_kwh"])
    if prev is not None: r_["marg_irr"] = inc_irr(prev, e)
    rows.append(r_); prev = e
pd.DataFrame(rows).to_csv("outputs/up_capacity_value.csv", index=False)

# ---------------- K. block-by-block decision and stopping rule against the global NPV maximum
def block_table(kind, x, pmax=4000.0, step=250.0, alloc_fn=None):
    rows, prev = [], None
    for p in np.arange(0, pmax + 1, step):
        global bill
        bill = bill_nem if kind == "nem" else _bill0
        try: e = economics(*(alloc_fn or allocate)(p), {"exp_thb": x or 0.0}, bg=False, full=True)
        finally: bill = _bill0
        r_ = dict(p_kwp=p, npv=e["npv"])
        if prev is not None:
            r_.update(dnpv=e["npv"] - prev["npv"], marg_irr=inc_irr(prev, e), dcap=e["capex"] - prev["capex"])
        rows.append(r_); prev = e
    d = pd.DataFrame(rows); d["accept"] = d.marg_irr >= P["r"]
    return d
BLK = {lab: block_table(k, x) for lab, (k, x) in SCHEMES.items() if k != "curt"}
pd.concat([v.assign(scheme=k) for k, v in BLK.items()]).to_csv("outputs/up_block_decisions.csv", index=False)
STOP = []
for x in XG:
    d = block_table("nb", x, pmax=8000.0)
    rej = d[(d.p_kwp > 0) & (d.marg_irr < P["r"])]
    p_stop = float(rej.p_kwp.iloc[0] - 250) if len(rej) else 8000.0
    p_glob = float(BASE[BASE.exp_thb == x].p_kwp.iloc[0])
    sign_ok = bool(((d.dnpv[1:] >= 0) == (d.marg_irr[1:] >= P["r"])).all())
    STOP.append(dict(exp_thb=x, p_stop=p_stop, p_maxnpv=p_glob, agree=p_stop == p_glob, sign_consistent=sign_ok,
                     npv_gap=float(BASE[BASE.exp_thb == x].npv.iloc[0] - d[d.p_kwp == p_stop].npv.iloc[0]) if p_stop <= 8000 else np.nan))
STOP = pd.DataFrame(STOP); STOP.to_csv("outputs/up_stopping_rule.csv", index=False)

# ---------------- L. capacity-step test: optima on a 50 kWp grid (and Eq. 5 evaluated on the same grid)
def fine_opt(kind, x, lo=0.0, hi=6000.0, step=50.0, th=None):
    g = np.arange(lo, hi + 1, step); v = np.array([run_scheme(kind, x, p)["npv"] if th is None else eco(p, dict(th, exp_thb=x))["npv"] for p in g])
    i = int(v.argmax()); band = g[v >= v.max() - 0.01 * abs(v.max())]
    return dict(p=float(g[i]), npv=float(v[i]), band=[float(band.min()), float(band.max())])
FINE = {}
for lab, (k, x) in SCHEMES.items():
    if k == "curt": continue
    f_ = fine_opt(k, x); c_ = SCH[lab]
    FINE[lab] = dict(p250=c_["p_kwp"], npv250=c_["npv"], p50=f_["p"], npv50=f_["npv"], gain=f_["npv"] - c_["npv"], band50=f_["band"])
PGF = np.arange(0, 40001, 50.0)
def sig_curve_g(load, pg):
    S_ = np.array([np.minimum(p * GROOF, load).sum() for p in pg]); G_ = pg * GROOF.sum()
    return np.diff(S_) / np.diff(G_)
SIGF = sig_curve_g(LD, PGF)
def p_star_f(Rx, Rr):
    if Rx >= 1: return np.inf
    idx = np.flatnonzero(SIGF < (1 - Rx) / (Rr - Rx)); return float(PGF[idx[0]]) if len(idx) else float(PGF[-1])
GF = np.arange(0, 6001, 50.0); RVF = []
for name, mod in VAR.items():
    P.update(base_t); P.update(mod)
    a0 = eco(0); a3 = eco(3000); rr = ((a3["sav1_usd"] - a0["sav1_usd"]) / (a3["sc_kwh"] - a0["sc_kwh"])) / C_PV
    xs = [x for x in XG if x <= 2.5] if name.startswith("PEA") else (0.0, 1.0, 2.0, 2.5)
    for x in xs:
        if name.startswith("PEA") and x in (1.0, 2.0) and False: continue
        npv = np.array([eco(p, {"exp_thb": x})["npv"] for p in GF]); ps = float(GF[npv.argmax()]); pr = p_star_f(x / FX / C_PV, rr)
        RVF.append(dict(variant=name, exp_thb=float(x), p_rule=pr, p_sim=ps, dp_kwp=pr - ps, npv_forgone=max(0.0, npv.max() - float(np.interp(pr, GF, npv)))))
P.update(base_t); RVF = pd.DataFrame(RVF).drop_duplicates(["variant", "exp_thb"])
RVF.to_csv("outputs/up_rule_validity_50kWp.csv", index=False)

# ---------------- M. propagation of the PV-model error: measured 2025 output of the reference array as the PV profile
MEAS = np.load("prepared/pv_interval_2025.npz")["pv_meas_per_kwp"]
_PVL, _PVF = PV_LAND, PV_FPV
PVERR = {}
for case, scale in (("Model (calibrated)", None), ("Measured profile", 1.0), ("Measured shape, model annual energy", PV_LAND.sum() / MEAS.sum())):
    if scale is not None:
        PV_LAND = MEAS * scale; PV_FPV = PV_LAND * (YIELD_FPV / YIELD_LAND)
    try:
        o = {}
        for lab, (k, x) in SCHEMES.items():
            if k == "curt": continue
            f_ = fine_opt(k, x, 0, 6000, 50.0); e125 = run_scheme(k, x, P0); ac = account(k, run_scheme(k, x, f_["p"]))
            o[lab] = dict(p50=f_["p"], npv_opt=f_["npv"], npv_at_P0=e125["npv"], irr_at_P0=e125["irr"], dM22=ac["dM"]["2.2"], yield_kwh_kwp=float(PV_LAND.sum() * DT))
        PVERR[case] = o
    finally:
        PV_LAND, PV_FPV = _PVL, _PVF

# ---------------- N. worked example of the accounts (year 1): monthly net metering at its optimum
def worked(kind, x, p):
    global bill
    a, T = allocate(p); pv = gen_profile(a, T) * (1 - P["deg"]); imp, ex, _ = dispatch(pv, 0.0, LOAD)
    b0 = _bill0(LOAD, np.zeros(NSTEP)); fn = bill_nem if kind == "nem" else _bill0
    b1 = fn(imp, ex, {"exp_thb": x or 0.0})
    rate = np.where(CLS == 0, P["e_on"], P["e_off"]) + P["ft"]
    cr_m = np.add.reduceat(ex * rate, MSTART)
    e_m = np.add.reduceat(np.where(CLS == 0, imp * P["e_on"], imp * P["e_off"]), MSTART) + np.add.reduceat(imp, MSTART) * P["ft"]
    e = run_scheme(kind, x, p); ac = account(kind, e)
    return dict(p_kwp=p, S_mwh=float((LOAD - imp).sum() / 1e3), X_mwh=float(ex.sum() / 1e3), base_energy=b0["energy"] / FX, base_demand=b0["demand"] / FX,
                new_energy=b1["energy"] / FX, new_demand=b1["demand"] / FX, credit=b1["credit"] / FX, bill_base=b0["total"] / FX, bill_new=b1["total"] / FX,
                bill_reduction=(b0["total"] - b1["total"]) / FX, cap_binding_months=int((cr_m > e_m).sum()) if kind == "nem" else 0,
                retail_lost=(b0["total"] - b1["total"] - b1["credit"]) / FX, check_economics=e["sav1_usd"], **{k_: ac[k_] for k_ in ("comp_sc_2.2", "comp_ex_2.2", "avoided_sc_2.2", "avoided_ex_2.2")},
                dM22=ac["dM"]["2.2"])
_nem = [l for l, (k, _) in SCHEMES.items() if k == "nem"][0]
WORKED = worked("nem", None, SCH[_nem]["p_kwp"])

# ---------------- O. sensitivity to the fuel adjustment charge (Ft): 2025 period values instead of the Jan-Apr 2026 value
FT_CASES = {"Jan-Apr 2026 value applied to all months (base)": P["ft"],
            "2025 period values (0.3672 / 0.1972 / 0.1572 THB/kWh)": np.array([0.3672] * 4 + [0.1972] * 4 + [0.1572] * 4),
            "Highest 2025 value (0.3672 THB/kWh) in all months": 0.3672}
FTS = {}
_ft0 = P["ft"]
for lab, ft in FT_CASES.items():
    P["ft"] = ft
    try:
        o = {}
        for xr, pmax in ((0.0, 5000.0), (2.2, 6000.0)):
            g = np.arange(0, pmax + 1, 250.0); v = [eco(p, {"exp_thb": xr}) for p in g]; i = int(np.argmax([e["npv"] for e in v])); e = v[i]
            o[f"{xr:.1f}"] = dict(p=float(g[i]), npv=e["npv"], base_bill=e["base_bill_usd"], dM22=account("nb", e)["dM"]["2.2"])
        e125 = eco(P0); o["npv_P0"] = e125["npv"]; o["irr_P0"] = e125["irr"]
        FTS[lab] = o
    finally:
        P["ft"] = _ft0

# ---------------- P. block bootstrap of weather years (3-month blocks keep the month-to-month dependence)
rngb = np.random.default_rng(8); NB = 500
_avail = set(YBY)
def _block_src(m0):
    return [y for y in (2023, 2024, 2025, 2026) if all(f"{y}-{m0 + k + 1:02d}" in _avail for k in range(3))]
WSB = np.ones((NB, 12))
for i in range(NB):
    for m0 in (0, 3, 6, 9):
        y = rngb.choice(_block_src(m0))
        for k in range(3): WSB[i, m0 + k] = YBY[f"{y}-{m0 + k + 1:02d}"] / YBY[f"2025-{m0 + k + 1:02d}"]
WYB = np.array([wy(ms) for ms in WSB])
np.save("outputs/up_wy_blocks.npy", WYB)

# ---------------- Q. study year (2025) against a typical year (2025 profile rescaled month by month to the NASA POWER 2003-2023 climatology)
TYP = {}
for xr, pmax in ((0.0, 5000.0), (2.2, 6000.0)):
    g = np.arange(0, pmax + 1, 250.0); v = [economics(*allocate(p), {"exp_thb": xr}, bg=False, lt=True)["npv"] for p in g]; i = int(np.argmax(v))
    TYP[f"{xr:.1f}"] = dict(p=float(g[i]), npv=float(v[i]))
_et = economics(*allocate(P0), bg=False, lt=True); TYP["npv_P0"] = _et["npv"]; TYP["irr_P0"] = _et["irr"]; TYP["yield"] = float(PV_LAND_LT.sum() * DT)

# ---------------- R. propagation of short-term PV error to export credits, demand charges and the optimum
# additive 30-min error of the model (measured minus modelled, kW per kWp) on each day of 2025; in each of 200
# simulations every day of the year receives the error sequence of a randomly drawn valid day of the same month,
# applied to both 15-min halves (A) or with the 30-min energy split randomly between the halves (B, sub-30-min stress test)
IV = np.load("prepared/pv_interval_2025.npz")
_e30 = np.where(IV["valid30"], (IV["meas30"] - IV["mod30"]) / 2256.24 * 2, np.nan).reshape(365, 48)
_dmon = (pd.date_range("2025-01-01", periods=365, freq="D").month.values - 1)
_okday = np.isfinite(_e30).mean(axis=1) >= 0.9
_pool = {m: np.flatnonzero(_okday & (_dmon == m)) for m in range(12)}
_e30 = np.nan_to_num(_e30)
rnge = np.random.default_rng(9); NE = 200
def _sim_profile(variant):
    src = np.array([rnge.choice(_pool[_dmon[d]]) for d in range(365)])
    e15 = np.repeat(_e30[src].reshape(-1), 2)                         # kW/kWp at 15-min interval starts
    pv = np.clip(_PVL + e15, 0, None)
    if variant == "B":
        pr = pv.reshape(-1, 2); tot = pr.sum(axis=1); sh = rnge.uniform(0.3, 0.7, len(tot))
        both = (pr > 0).all(axis=1); pr = np.where(both[:, None], np.stack([tot * sh, tot * (1 - sh)], axis=1), pr); pv = pr.reshape(-1)
    return pv
def _metrics():
    e0 = run_scheme("nb", 0.0, P0); e22 = run_scheme("nb", 2.2, 3000.0)
    g0 = np.arange(0, 5001, 250.0); g22 = np.arange(0, 6001, 250.0)
    p0 = float(g0[int(np.argmax([run_scheme("nb", 0.0, p)["npv"] for p in g0]))]); p22 = float(g22[int(np.argmax([run_scheme("nb", 2.2, p)["npv"] for p in g22]))])
    return [e0["npv"], e0["sav_demand_usd"], e0["sav_energy_usd"], e0["ex_kwh"], e22["npv"], e22["credit_usd"], e22["sav_demand_usd"], e22["ex_kwh"], p0, p22]
ERRB = _metrics()
ERRS = {}
for variant in ("A", "B"):
    rows_ = []
    for _ in range(NE):
        PV_LAND = _sim_profile(variant); PV_FPV = PV_LAND * (YIELD_FPV / YIELD_LAND)
        try: rows_.append(_metrics())
        finally: PV_LAND, PV_FPV = _PVL, _PVF
    ERRS[variant] = np.array(rows_)
np.savez("outputs/up_pv_error_sim.npz", base=np.array(ERRB), A=ERRS["A"], B=ERRS["B"])
_names = ["npv0", "dem0", "en0", "ex0", "npv22", "credit22", "dem22", "ex22", "p0", "p22"]
PVSIM = dict(base=dict(zip(_names, ERRB)), n=NE)
for v_, a_ in ERRS.items():
    PVSIM[v_] = {k: dict(p5=float(np.percentile(a_[:, j], 5)), p50=float(np.median(a_[:, j])), p95=float(np.percentile(a_[:, j], 95)), mean=float(a_[:, j].mean())) for j, k in enumerate(_names)}
    PVSIM[v_]["p0_share_base"] = float((a_[:, 8] == P0).mean() * 100); PVSIM[v_]["p22_share_base"] = float((a_[:, 9] == 3000.0).mean() * 100)
    PVSIM[v_]["p0_values"] = sorted(set(float(x) for x in a_[:, 8])); PVSIM[v_]["p22_values"] = sorted(set(float(x) for x in a_[:, 9]))

ePV = eco(P0)
RES = dict(P0=P0, c_pv=C_PV, v_r=V_R, R_r=R_R, thresh=THRESH, opt_base=BASE[["exp_thb", "p_kwp", "npv", "irr"]].to_dict("records"),
           pv_opt=dict(npv=ePV["npv"], irr=ePV["irr"], lcoe=ePV["lcoe"], capex=ePV["capex"], pv_mwh=ePV["pv_kwh"] / 1e3, sc_share=ePV["sc_kwh"] / ePV["pv_kwh"],
                       ex_share=ePV["ex_kwh"] / ePV["pv_kwh"], sav1=ePV["sav1_usd"], sav_energy=ePV["sav_energy_usd"], sav_demand=ePV["sav_demand_usd"], base_bill=ePV["base_bill_usd"],
                       cover=ePV["sc_kwh"] / ePV["load_kwh"], payback=ePV["payback"]),
           p_ann=P_ANN, p_mon=P_MON, bench=BENCH, schemes=SCH, schemes_cost_order=SCH_C, band=BAND, af_deg=AF_DEG, shape_opt=SHOPT, rr_opt=RROPT,
           rule=dict(n=int(len(RV)), eq=int((RV.dp_kwp == 0).sum()), one=int((RV.dp_kwp == -250).sum()), more=int((RV.dp_kwp < -250).sum()), over=int((RV.dp_kwp > 0).sum()), max_forgone=float(RV.npv_forgone.max())),
           an=dict(eq=int((AN[AN.exp_thb <= 2.5].p_rule == AN[AN.exp_thb <= 2.5].p_sim).sum()), n=int((AN.exp_thb <= 2.5).sum()), over=int((AN[AN.exp_thb <= 2.5].p_rule > AN[AN.exp_thb <= 2.5].p_sim).sum())),
           mc=dict(npv_mean=float(MCR[:, 0].mean()), p5=float(np.percentile(MCR[:, 0], 5)), p50=float(np.median(MCR[:, 0])), p95=float(np.percentile(MCR[:, 0], 95)),
                   p_pos=float((MCR[:, 0] > 0).mean() * 100), se_pos=float(np.sqrt((MCR[:, 0] > 0).mean() * (1 - (MCR[:, 0] > 0).mean()) / NMC) * 100),
                   half1=float((MCR[:5000, 0] > 0).mean() * 100), half2=float((MCR[5000:, 0] > 0).mean() * 100), irr_med=float(np.nanmedian(MCR[:, 2])),
                   lcoe_mean=float(MCR[:, 1].mean()), lcoe_p5=float(np.percentile(MCR[:, 1], 5)), lcoe_p95=float(np.percentile(MCR[:, 1], 95)), p_irr_r=float((MCR[:, 2] > 0.07).mean() * 100)),
           blocks={k: v.to_dict("records") for k, v in BLK.items()}, stopping=STOP.to_dict("records"), fine=FINE,
           rule50=dict(n=int(len(RVF)), eq=int((RVF.dp_kwp == 0).sum()), under=int((RVF.dp_kwp < 0).sum()), over=int((RVF.dp_kwp > 0).sum()),
                       max_under=float(-RVF.dp_kwp.min()), max_forgone=float(RVF.npv_forgone.max())),
           pv_error=PVERR, worked=WORKED, ac1=AC1, ft_sens=FTS, typical_year=TYP, pv_err_sim=PVSIM,
           wy_blocks=dict(n=NB, npv=pc(WYB[:, 0]), p_pos=float((WYB[:, 0] > 0).mean() * 100), popt0=pc(WYB[:, 1]), popt22=pc(WYB[:, 2]), yf=pc(WYB[:, 3]),
                          popt0_range=[float(WYB[:, 1].min()), float(WYB[:, 1].max())], popt22_range=[float(WYB[:, 2].min()), float(WYB[:, 2].max())],
                          npv_range=[float(WYB[:, 0].min()), float(WYB[:, 0].max())], yf_range=[float(WYB[:, 3].min()), float(WYB[:, 3].max())]),
           mc_corr=dict(p_pos=float((MCR_C > 0).mean() * 100), p5=float(np.percentile(MCR_C, 5)), p95=float(np.percentile(MCR_C, 95)), mean=float(MCR_C.mean())),
           wy_windows=dict(n=int(len(WYW)), npv=pc(WYW[:, 0]), npv_min=float(WYW[:, 0].min()), npv_max=float(WYW[:, 0].max()), p_pos=float((WYW[:, 0] > 0).mean() * 100),
                           popt0=pc(WYW[:, 1]), popt0_range=[float(WYW[:, 1].min()), float(WYW[:, 1].max())], popt22=pc(WYW[:, 2]),
                           popt22_range=[float(WYW[:, 2].min()), float(WYW[:, 2].max())], yf=pc(WYW[:, 3]), yf_range=[float(WYW[:, 3].min()), float(WYW[:, 3].max())]),
           tornado=TORN, wy=dict(npv=pc(WY[:, 0]), p_pos=float((WY[:, 0] > 0).mean() * 100), popt0=pc(WY[:, 1]), popt22=pc(WY[:, 2]), yf=pc(WY[:, 3]),
                                 actual_2024=dict(npv=float(WY[0, 0]), popt0=float(WY[0, 1]), popt22=float(WY[0, 2]), yf=float(WY[0, 3]))))
SW.to_csv("outputs/up_sweeps.csv", index=False); OPT.to_csv("outputs/up_optima.csv", index=False); AN.to_csv("outputs/up_rule_grid.csv", index=False)
RV.to_csv("outputs/up_rule_validity.csv", index=False); SCH_CURVE.to_csv("outputs/up_scheme_curves.csv", index=False)
np.save("outputs/up_mc.npy", MCR); np.save("outputs/up_wy.npy", WY); np.save("outputs/up_wy_windows.npy", WYW)
np.savez("outputs/up_framework.npz", rd=RD, sig=SIG, map_rx=MAP_RX, map_rd=MAP_RD, map_m=MAP_M, **{"shape_" + str(i): np.vstack(v) for i, v in enumerate(SHC.values())})
json.dump(RES, open("outputs/up_results.json", "w"), indent=1, default=float)
print(json.dumps({k: RES[k] for k in ("P0", "R_r", "thresh", "pv_opt", "rule", "an", "mc", "wy")}, indent=1, default=float))
print(json.dumps(SCH, indent=1, default=float)); print(json.dumps(BENCH, indent=0, default=float)); print(RV.to_string())
