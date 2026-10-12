"""
MSROF-CRRU | Script 2 of 3 | Techno-economic analysis on measured data (15-min resolution)
===========================================================================================
Inputs  prepared/measured_2025.npz  (script 0: PEA 15-min campus load 2025, TOU class of every
                                     interval, on-site irradiance and temperature 2025)
        prepared/prep_summary.json  (script 0: PV calibration statistics)
        outputs/landuse_raster_check.json (script 1: reservoir / small-pond split)
Everything is computed in-house with NumPy / SciPy / pandas. Tariff, cost and biogas inputs are
embedded below (Section 1). Monetary results are reported in USD (fixed 35.5 THB per USD).

Model chain
 A  Land-use areas -> eligible surfaces (Eq. 1) -> four capacity tiers
 B  15-min PV output from measured irradiance/temperature, lumped efficiency calibrated on the
    CRH Solar Rooftop arrays (Eqs. 2-3)
 C  Biogas: dry-basis VS, CSTR first-order conversion, CHP efficiencies, parasitic loads (Eqs. 5-6)
 D  15-min dispatch against the metered load; optional battery (TOU energy shifting) (Eqs. 7-8)
 E  PEA TOU bill (on/off-peak energy, Ft, on-peak demand charge, service charge, 70 % minimum-charge
    clause) with and without the project; export remuneration as a policy scenario (Eqs. 9-10)
 F  NPC, LCOE (cost only), NPV, IRR, payback; capacity sweep and marginal IRR (Eqs. 11-14)
 G  Monte Carlo (10,000 draws) and one-at-a-time sensitivity
 H  Co-benefits (Eq. 15) and secondary composite indicators
"""
import json, os
import numpy as np, pandas as pd
from scipy.optimize import brentq

os.makedirs("outputs", exist_ok=True)
rng = np.random.default_rng(20261007)

# =====================================================================================
# 1. INPUTS
# =====================================================================================
M = np.load("prepared/measured_2025.npz")
LOAD = M["load_kwh"].astype(float)              # kWh per 15 min, PEA meter 2025 (grid import, existing PV behind meter)
CLS = M["tou_class"].astype(int)                 # 0 on-peak, 1 off-peak, 2 holiday (from the meter's RATE A/B/C)
GHI = M["ghi_wm2"].astype(float); TAMB = M["tamb"].astype(float)
TIDX = pd.to_datetime(M["t_start"].astype("datetime64[m]"))
NSTEP = len(LOAD); DT = 0.25
MONTH = TIDX.month.values - 1
MSTART = np.r_[0, np.flatnonzero(np.diff(MONTH)) + 1]
PREP = json.load(open("prepared/prep_summary.json"))

# NASA POWER 2003-2023 monthly climatology (free access) for the long-term-resource case
GHI_NASA = np.array([4.72, 5.16, 5.68, 5.72, 5.21, 4.75, 4.41, 4.18, 4.32, 4.51, 4.63, 4.48])
DIM = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# satellite land-use inventory (m2) and raster-measured reservoir
LAND = {"buildings": 213580.61, "open_fields": 48682.98, "ponds_total": 364785.01, "roads": 185392.01, "vegetation": 1204935.32}
A_CAMPUS = sum(LAND.values()); A_RES, A_SMALL = 321755.2, 364785.01 - 321755.2
A_ROOF_ELIG = 164765.9          # roofs retained in the rooftop PV zone (script 1, cross-tabulation of the two maps)
if os.path.exists("outputs/landuse_raster_check.json"):
    _r = json.load(open("outputs/landuse_raster_check.json")); A_RES, A_SMALL = _r["reservoir_m2_used"], _r["small_ponds_m2_used"]; A_ROOF_ELIG = _r["roof_eligible_m2"]
WASTE = dict(rate=(3.12, 0.41), vs=(0.82, 0.05), moist=(0.65, 0.08), bmp=(0.300, 0.025))

FX = 35.5
# lumped efficiency energy-matched on 2024 and Jan-Aug 2026 (script 0); 2025 is held out for validation
ETA_CAL = PREP["calibration"]["eta_cal"]
P = dict(
    eta=ETA_CAL, gamma=-0.0041, noct=45.0, dT_fpv=8.0, deg=0.005,
    dens_roof=65.0, dens_ground=55.0, dens_fpv=140.0, f_roof=0.70, f_ground=0.85, cov=0.30,
    # PEA Schedule 3/4/5 TOU, 22-33 kV (THB); Ft Jan-Apr 2026
    e_on=4.1839, e_off=2.6037, ft=0.0972, dem=132.93, svc=312.24, min_frac=0.70,
    # export remuneration (THB/kWh): base case 0 = no purchase agreement applicable to a MW-scale institutional
    # self-consumption system (surplus not remunerated / curtailed); policy scenarios below
    exp=0.0,
    capex_roof=28.0 / FX, capex_ground=22.0 / FX, capex_fpv=27.0 / FX, capex_bg=12.0e6 / FX,
    capex_li=14000 / FX, capex_pb=8500 / FX,
    om_pv=0.0125, om_bg=0.05, om_batt=0.025, inv_repl=0.15, decom=0.05, salvage=0.05, r=0.07, N=25,
    kd=0.12, hrt=30.0, lhv=9.97, eta_e=0.33, eta_t=0.45, par_e=0.08, par_t=0.15, avail_h=8000.0, leak=0.02, ch4_kg_m3=0.668,
    batt=dict(li=dict(dod=0.90, rt=0.92, cyc=5000, cal=15.0), pb=dict(dod=0.50, rt=0.85, cyc=1200, cal=7.0)), c_rate=0.5, fade=0.90,
    ef_grid=0.55, gwp_ch4=27.0, doc=0.16, docf=0.5, mcf=0.6, f_ch4=0.5, evap=0.5,
)

# =====================================================================================
# 2. PV MODEL (per kWp, AC, 15-min)
# =====================================================================================
def pv_per_kwp(ghi=GHI, tamb=TAMB, eta=None, fpv=False):
    eta = P["eta"] if eta is None else eta
    tc = tamb + (P["noct"] - 20.0) / 800.0 * ghi
    if fpv: tc = tc - P["dT_fpv"] * np.minimum(ghi / 800.0, 1.0)
    return np.clip(ghi / 1000.0 * (1 + P["gamma"] * (tc - 25.0)) * eta, 0, None)        # kW per kWp

PV_LAND, PV_FPV = pv_per_kwp(), pv_per_kwp(fpv=True)
YIELD_LAND, YIELD_FPV = PV_LAND.sum() * DT, PV_FPV.sum() * DT
# long-term resource case: rescale each month of 2025 to the NASA POWER climatology
_mon_meas = np.array([GHI[MONTH == m].sum() * DT / 1000 / DIM[m] for m in range(12)])
GHI_LT = GHI * (GHI_NASA / _mon_meas)[MONTH]
PV_LAND_LT, PV_FPV_LT = pv_per_kwp(GHI_LT), pv_per_kwp(GHI_LT, fpv=True)

# =====================================================================================
# 3. BIOGAS
# =====================================================================================
def biogas(waste=3.12, vs=0.82, moist=0.65, bmp=0.300, kd=None, mode="cstr"):
    kd = P["kd"] if kd is None else kd
    vs_d = waste * 1000 * (1 - moist) * vs
    x = kd * P["hrt"] / (1 + kd * P["hrt"]) if mode == "cstr" else 1 - np.exp(-kd * P["hrt"])
    ch4_y = bmp * vs_d * x * 365; prim = ch4_y * P["lhv"]
    eg = prim * P["eta_e"]; hg = prim * P["eta_t"]
    return dict(vs_d=vs_d, ts=1 - moist, conv=x, ch4_d=ch4_y / 365, ch4_y=ch4_y, prim_kwh=prim, e_gross=eg, e_net=eg * (1 - P["par_e"]),
                h_gross=hg, h_net=hg * (1 - P["par_t"]), chp_kw=eg / P["avail_h"], waste_t_y=waste * 365, digestate_t_y=waste * 365 * 0.95)
BG = biogas(); BG_BATCH = biogas(mode="batch")

# =====================================================================================
# 4. CAPACITY TIERS
# =====================================================================================
def tiers(cov=None):
    cov = P["cov"] if cov is None else cov
    A_roof = A_ROOF_ELIG * P["f_roof"]; A_gr = LAND["open_fields"] * P["f_ground"]
    return {"roof": dict(kwp=A_roof * P["dens_roof"] / 1000, area=A_roof, capex=P["capex_roof"], fpv=False, water=0.0),
            "ground": dict(kwp=A_gr * P["dens_ground"] / 1000, area=A_gr, capex=P["capex_ground"], fpv=False, water=0.0),
            "ponds": dict(kwp=cov * A_SMALL * P["dens_fpv"] / 1000, area=cov * A_SMALL, capex=P["capex_fpv"], fpv=True, water=A_SMALL),
            "reservoir": dict(kwp=cov * A_RES * P["dens_fpv"] / 1000, area=cov * A_RES, capex=P["capex_fpv"], fpv=True, water=A_RES)}
def allocate(p_kwp, cov=None, use_res=True):
    T = tiers(cov); a = {}; left = p_kwp
    for k, t in T.items():
        if k == "reservoir" and not use_res: a[k] = 0.0; continue
        a[k] = min(left, t["kwp"]); left -= a[k]
    return a, T
T0 = tiers()

# =====================================================================================
# 5. BILL MODEL (PEA TOU) AND DISPATCH
# =====================================================================================
def bill(imp, ex, th=None):
    """Annual electricity cost (THB) for a 15-min import series and export series."""
    th = th or {}
    fe = th.get("tariff_energy", 1.0); fd = th.get("tariff_demand", 1.0); xp = th.get("exp_thb", P["exp"])
    on = CLS == 0
    e_m = (np.add.reduceat(np.where(on, imp * P["e_on"], imp * P["e_off"]), MSTART) + np.add.reduceat(imp, MSTART) * P["ft"]) * fe
    dmax = np.maximum.reduceat(np.where(on, imp, 0.0), MSTART) * 4.0
    d_m = dmax * P["dem"] * fd
    # PEA minimum-charge clause: the monthly charge may not fall below 70 % of the highest demand charge of the last
    # 12 months (steady-state year: the highest monthly demand charge of the simulated year)
    floor = P["min_frac"] * d_m.max(); m_tot = e_m + d_m + P["svc"]; topup = np.clip(floor - m_tot, 0, None)
    energy, demand = e_m.sum(), d_m.sum()
    credit = ex.sum() * xp
    return dict(energy=energy, demand=demand, service=12 * P["svc"], minimum_topup=topup.sum(), n_min_binding=int((topup > 0).sum()), credit=credit,
                total=energy + demand + 12 * P["svc"] + topup.sum() - credit, dmax=dmax)

def gen_profile(alloc, T, scale=1.0, lt=False):
    pl, pf = (PV_LAND_LT, PV_FPV_LT) if lt else (PV_LAND, PV_FPV)
    kl = sum(v for k, v in alloc.items() if not T[k]["fpv"]); kf = sum(v for k, v in alloc.items() if T[k]["fpv"])
    return (kl * pl + kf * pf) * DT * scale                                  # kWh per 15 min

def dispatch(pv, bgk, load, batt=None):
    gen = pv + bgk
    if batt is None:
        imp = np.clip(load - gen, 0, None); ex = np.clip(gen - load, 0, None)
        return imp, ex, dict(dis=0.0, efc=0.0)
    b = P["batt"][batt["kind"]]; cap = batt["cap"] * P["fade"]; eta = np.sqrt(b["rt"])
    pmax = P["c_rate"] * cap * DT; emin, emax = cap * (1 - b["dod"]), cap
    soc = 0.5 * (emin + emax); imp = np.zeros(NSTEP); ex = np.zeros(NSTEP); dis_tot = 0.0
    net = gen - load
    for k in range(NSTEP):
        n = net[k]
        if n > 0:
            ch = min(n, pmax, (emax - soc) / eta); soc += ch * eta; ex[k] = n - ch
        else:
            need = -n
            if CLS[k] == 0:                                                  # discharge during on-peak only
                dis = min(need, pmax, (soc - emin) * eta); soc -= dis / eta; dis_tot += dis; need -= dis
            imp[k] = need
    return imp, ex, dict(dis=dis_tot, efc=dis_tot / (cap * b["dod"]))

def battery_size(pv, bgk, kind):
    """Eq. (7): C = E_shift N / (DoD eta_rt); E_shift = mean daily min(surplus energy, on-peak deficit energy)."""
    b = P["batt"][kind]; net = pv + bgk - LOAD
    day = TIDX.normalize()
    sur = pd.Series(np.clip(net, 0, None)).groupby(day).sum(); defi = pd.Series(np.where(CLS == 0, np.clip(-net, 0, None), 0)).groupby(day).sum()
    e_shift = float(np.minimum(sur.values, defi.values).mean())
    return e_shift / (b["dod"] * b["rt"]), e_shift

BASE_BILL = bill(LOAD, np.zeros(NSTEP))

# =====================================================================================
# 6. ECONOMICS
# =====================================================================================
def economics(alloc, T, th=None, battery=None, bg=True, lt=False, full=False):
    th = th or {}; r = th.get("r", P["r"]); N = P["N"]
    capex_pv = sum(alloc[k] * 1000 * T[k]["capex"] for k in alloc) * th.get("capex_pv", 1.0)
    bgm = biogas(th.get("waste", 3.12), th.get("vs", 0.82), th.get("moist", 0.65), th.get("bmp", 0.300), th.get("kd")) if bg else None
    e_bg = bgm["e_net"] if bg else 0.0; capex_bg = P["capex_bg"] * th.get("capex_bg", 1.0) if bg else 0.0
    bgk = e_bg / NSTEP
    load = LOAD * th.get("load", 1.0)
    base = bill(load, np.zeros(NSTEP), th)
    sc = th.get("eta_scale", 1.0) * th.get("solar", 1.0)
    pv0 = gen_profile(alloc, T, sc, lt)
    if th.get("mscale") is not None: pv0 = pv0 * np.asarray(th["mscale"])[MONTH]
    capex_b = 0.0; life = None; kind = None
    if battery:
        kind = battery["kind"]; capex_b = battery["cap"] * (P["capex_li"] if kind == "li" else P["capex_pb"]) * th.get("capex_batt", 1.0)
    res_y = []
    for yr in (1, N):
        pv = pv0 * (1 - P["deg"]) ** yr
        imp, ex, st = dispatch(pv, bgk, load, battery)
        bl = bill(imp, ex, th)
        res_y.append(dict(pv=pv.sum(), imp=imp.sum(), ex=ex.sum(), sav=base["total"] - bl["total"],
                          sav_energy=base["energy"] - bl["energy"], sav_demand=base["demand"] - bl["demand"], credit=bl["credit"], st=st, nmin=bl["n_min_binding"]))
    if battery:
        b = P["batt"][kind]; life = min(b["cyc"] / max(res_y[0]["st"]["efc"], 1e-9), b["cal"] * th.get("batt_life", 1.0))
    t = np.arange(1, N + 1); w = (t - 1) / (N - 1)
    lin = lambda k: res_y[0][k] + (res_y[1][k] - res_y[0][k]) * w
    benefit = lin("sav") / FX                                              # USD per year (bill saving incl. export credit)
    pvgen = lin("pv")
    om = (capex_pv * P["om_pv"] + capex_bg * P["om_bg"] + capex_b * P["om_batt"]) * th.get("om", 1.0) * np.ones(N)
    repl = np.zeros(N)
    if capex_pv > 0: repl[[9, 19]] += capex_pv * P["inv_repl"]
    if battery:
        k = 1
        while life * k < N - 1e-9: repl[int(np.ceil(life * k)) - 1] += capex_b; k += 1
    capex = capex_pv + capex_bg + capex_b
    repl[N - 1] += capex * (P["decom"] - th.get("salvage", P["salvage"]))
    disc = (1 + r) ** -t
    npc = capex + float(((om + repl) * disc).sum())
    energy = pvgen + e_bg
    lcoe = npc / float((energy * disc).sum()) if energy.sum() > 0 else np.nan
    cf = benefit - om - repl
    npv = -capex + float((cf * disc).sum())
    try: irr = brentq(lambda x: -capex + (cf / (1 + x) ** t).sum(), -0.9, 2.0)
    except ValueError: irr = np.nan
    payback = capex / (benefit[0] - om[0]) if benefit[0] > om[0] else np.nan
    y1 = res_y[0]
    out = dict(capex=capex, capex_pv=capex_pv, capex_bg=capex_bg, capex_batt=capex_b, om1=float(om[0]), npc=npc, lcoe=lcoe, npv=npv,
               irr=irr, payback=payback, pv_kwh=float(y1["pv"]), bg_kwh=e_bg, imp_kwh=float(y1["imp"]), ex_kwh=float(y1["ex"]),
               sc_kwh=float(load.sum() - y1["imp"]), load_kwh=float(load.sum()), sav1_usd=float(benefit[0]),
               sav_energy_usd=float(y1["sav_energy"] / FX), sav_demand_usd=float(y1["sav_demand"] / FX), credit_usd=float(y1["credit"] / FX),
               base_bill_usd=float(base["total"] / FX), batt_life=life, efc=float(y1["st"]["efc"]), heat_kwh=bgm["h_net"] if bg else 0.0, min_binding=int(y1["nmin"] + res_y[1]["nmin"]))
    if full: out.update(cf=cf, benefit1=float(benefit[0]))
    return out

# =====================================================================================
# 7. CAPACITY-VALUE CURVE AND DESIGNS
# =====================================================================================
P_TECH_NO_RES = sum(T0[k]["kwp"] for k in ("roof", "ground", "ponds")); P_TECH_ALL = P_TECH_NO_RES + T0["reservoir"]["kwp"]
ANNUAL_LOAD = LOAD.sum()

def inc_irr(e0, e1):
    dcap = e1["capex"] - e0["capex"]; dcf = e1["cf"] - e0["cf"]; t = np.arange(1, len(dcf) + 1)
    try: return brentq(lambda x: -dcap + (dcf / (1 + x) ** t).sum(), -0.9, 3.0)
    except ValueError: return np.nan

pts = np.concatenate([[0.0], np.arange(250, 8001, 250), np.geomspace(8500, P_TECH_ALL, 16)])
rows, prev = [], None
for p in pts:
    a, T = allocate(p); e = economics(a, T, full=True)
    row = dict(p_kwp=p, npv=e["npv"], lcoe=e["lcoe"], irr=e["irr"], capex=e["capex"], sc_share=(e["sc_kwh"] - 0) / max(e["pv_kwh"] + e["bg_kwh"], 1e-9),
               load_cover=e["sc_kwh"] / e["load_kwh"], gen_to_load=(e["pv_kwh"] + e["bg_kwh"]) / e["load_kwh"], sav_demand=e["sav_demand_usd"])
    if prev is not None: row["marg_irr"] = inc_irr(prev, e)
    rows.append(row); prev = e
SWEEP = pd.DataFrame(rows)
def p_cross(h):
    d = SWEEP[SWEEP.p_kwp > 0]; b = d[d.marg_irr < h]
    return float(b.p_kwp.iloc[0]) if len(b) else np.nan
P_NPVMAX = float(SWEEP.p_kwp[SWEEP.npv.idxmax()])
P_K10, P_K12, P_K08 = p_cross(0.10), p_cross(0.12), p_cross(0.08)
P_RULE = (1.15 * ANNUAL_LOAD - BG["e_net"]) / YIELD_LAND                   # annual-energy (115 %) rule

def scenario(name, p_kwp, kind=None, use_res=True):
    a, T = allocate(p_kwp, None, use_res); battery = None
    if kind:
        cap, es = battery_size(gen_profile(a, T) * (1 - P["deg"]), BG["e_net"] / NSTEP, kind); battery = dict(kind=kind, cap=cap, e_shift=es)
    e = economics(a, T, battery=battery, full=True); e.update(name=name, p_kwp=p_kwp, alloc=a, T=T, battery=battery)
    return e
SC = {"K": scenario("K hurdle-rate design (marginal IRR 10 %)", P_K10),
      "N": scenario("N NPV-maximising design", P_NPVMAX),
      "A": scenario("A annual-energy rule (115 %)", P_RULE),
      "X": scenario("X all surfaces except reservoir", P_TECH_NO_RES, use_res=False),
      "F": scenario("F full potential", P_TECH_ALL),
      "N2": scenario("N + Li-ion", P_NPVMAX, "li"),
      "N3": scenario("N + lead-acid", P_NPVMAX, "pb")}
zero = economics(*allocate(0.0), full=True)
BLOCKS = {}
for lab, (a_, b_) in {"0 to K": ("0", "K"), "K to N": ("K", "N"), "N to A": ("N", "A"), "A to X": ("A", "X"), "X to F": ("X", "F")}.items():
    e0 = zero if a_ == "0" else SC[a_]; e1 = SC[b_]
    BLOCKS[lab] = dict(dkwp=(e1["p_kwp"] - (0 if a_ == "0" else e0["p_kwp"])) / 1000, dcap=(e1["capex"] - e0["capex"]) / 1e6,
                       dnpv=(e1["npv"] - e0["npv"]) / 1e6, irr=inc_irr(e0, e1))
ANN = float(((1 + P["r"]) ** -np.arange(1, P["N"] + 1)).sum())
WASTE_CREDIT_BE = -zero["npv"] / (ANN * BG["waste_t_y"])
PV_ONLY = {k: economics(SC[k]["alloc"], SC[k]["T"], bg=False) for k in ("K", "N", "A", "X", "F")}
LT = {k: economics(SC[k]["alloc"], SC[k]["T"], lt=True) for k in ("K", "N", "A", "X", "F")}
EXP_100 = {k: economics(SC[k]["alloc"], SC[k]["T"], {"exp_thb": 1.00}) for k in ("K", "N", "A", "X", "F")}
EXP_220 = {k: economics(SC[k]["alloc"], SC[k]["T"], {"exp_thb": 2.20}) for k in ("K", "N", "A", "X", "F")}
ETA_LOW = {k: economics(SC[k]["alloc"], SC[k]["T"], {"eta_scale": PREP["calibration"]["eta_median"] / P["eta"]}) for k in ("K", "N", "A", "X", "F")}

# =====================================================================================
# 8. MONTE CARLO AND SENSITIVITY
# =====================================================================================
def tn(mu, sd, n, lo=None, hi=None): return np.clip(rng.normal(mu, sd, n), lo, hi)
NMC = 10000
MC_IN = dict(capex_pv=tn(1, .2 / 1.645, NMC, .4), om=tn(1, .3 / 1.645, NMC, .2), r=tn(.07, .02 / 1.645, NMC, .02),
             eta_scale=tn(1, 0.07 / 1.645, NMC, .7, 1.3), solar=tn(1, .10 / 1.645, NMC, .6),
             tariff_energy=tn(1, .1 / 1.645, NMC, .5), tariff_demand=tn(1, .1 / 1.645, NMC, .5),
             load=tn(1, .10 / 1.645, NMC, .6), capex_bg=tn(1, .3 / 1.645, NMC, .3), waste=tn(3.12, .41, NMC, 1), vs=tn(.82, .05, NMC, .5, .95),
             moist=tn(.65, .08, NMC, .3, .9), bmp=tn(.30, .025, NMC, .15), kd=tn(.12, .03, NMC, .04), salvage=rng.uniform(0, .1, NMC))
def run_mc(key):
    sc = SC[key]; out = np.zeros((NMC, 3))
    for i in range(NMC):
        e = economics(sc["alloc"], sc["T"], {k: float(v[i]) for k, v in MC_IN.items()})
        out[i] = [e["npv"], e["lcoe"], e["irr"]]
    return out
MC = {k: run_mc(k) for k in ("K", "N", "A", "X", "F")}
def run_mc_pv(key):
    sc = SC[key]; out = np.zeros((NMC, 3))
    for i in range(NMC):
        e = economics(sc["alloc"], sc["T"], {k: float(v[i]) for k, v in MC_IN.items()}, bg=False)
        out[i] = [e["npv"], e["lcoe"], e["irr"]]
    return out
MC["N_pv"] = run_mc_pv("N")
MCS = {k: dict(npv_mean=float(o[:, 0].mean()), npv_p5=float(np.percentile(o[:, 0], 5)), npv_p50=float(np.median(o[:, 0])),
               npv_p95=float(np.percentile(o[:, 0], 95)), p_npv_pos=float((o[:, 0] > 0).mean() * 100), lcoe_mean=float(o[:, 1].mean()),
               lcoe_p5=float(np.percentile(o[:, 1], 5)), lcoe_p95=float(np.percentile(o[:, 1], 95)), irr_med=float(np.nanmedian(o[:, 2])),
               p_irr_r=float((o[:, 2] > 0.07).mean() * 100)) for k, o in MC.items()}

TOR = {"PV capital cost (±20 %)": ("capex_pv", .8, 1.2), "Discount rate (5 % / 9 %)": ("r", .05, .09),
       "Energy tariff (±10 %)": ("tariff_energy", .9, 1.1), "Demand charge (±10 %)": ("tariff_demand", .9, 1.1),
       "Export remuneration (0 / 1.00 THB kWh⁻¹)": ("exp_thb", 0.0, 1.0), "PV efficiency (±7 %)": ("eta_scale", .93, 1.07),
       "Solar resource (±10 %)": ("solar", .9, 1.1), "Campus load (±10 %)": ("load", .9, 1.1), "O&M (±30 %)": ("om", .7, 1.3),
       "Biogas capital cost (±30 %)": ("capex_bg", .7, 1.3), "Methane potential (275 / 325 m³ t⁻¹ VS)": ("bmp", .275, .325),
       "Waste generation (2.71 / 3.53 t d⁻¹)": ("waste", 2.71, 3.53), "Moisture (73 / 57 %)": ("moist", .73, .57),
       "Volatile solids (77 / 87 % of TS)": ("vs", .77, .87), "Net salvage (0 / 10 %)": ("salvage", 0, .1)}
TORNADO = {}
for lab, (k, lo, hi) in TOR.items():
    a = economics(SC["N"]["alloc"], SC["N"]["T"], {k: lo})["npv"]; b = economics(SC["N"]["alloc"], SC["N"]["T"], {k: hi})["npv"]
    TORNADO[lab] = dict(low=a, high=b, swing=abs(b - a))
k2 = SC["N2"]
BATT_CURVE = pd.DataFrame([dict(mult=m, price=m * P["capex_li"], npv=economics(k2["alloc"], k2["T"], {"capex_batt": m}, battery=k2["battery"])["npv"] / 1e6)
                           for m in np.linspace(0.05, 1.4, 20)])
try: BE_MULT = brentq(lambda m: economics(k2["alloc"], k2["T"], {"capex_batt": m}, battery=k2["battery"])["npv"] - SC["N"]["npv"], 0.01, 1.5)
except ValueError: BE_MULT = np.nan
FX_ROWS = []
for fx in (32.0, 35.5, 39.0, 43.0):
    f = FX / fx
    e_usd = economics(SC["N"]["alloc"], SC["N"]["T"], {"tariff_energy": f, "tariff_demand": f, "exp_thb": P["exp"] * f})
    e_thb = economics(SC["N"]["alloc"], SC["N"]["T"], {"tariff_energy": f, "tariff_demand": f, "exp_thb": P["exp"] * f, "capex_pv": f, "capex_bg": f})
    FX_ROWS.append(dict(fx=fx, npv_usd_equip=e_usd["npv"] / 1e6, irr_usd_equip=e_usd["irr"], npv_all_thb=e_thb["npv"] / 1e6, irr_all_thb=e_thb["irr"]))
def _be(key, lo, hi, sc="N"):
    try: return brentq(lambda m: economics(SC[sc]["alloc"], SC[sc]["T"], {key: m})["npv"], lo, hi)
    except ValueError: return np.nan
BREAKEVEN = dict(capex_pv=_be("capex_pv", 0.5, 2.0), tariff_energy=_be("tariff_energy", 0.3, 2.0), eta_scale=_be("eta_scale", 0.5, 1.5))
def _bepv(key, lo, hi):
    try: return brentq(lambda m: economics(SC["N"]["alloc"], SC["N"]["T"], {key: m}, bg=False)["npv"], lo, hi)
    except ValueError: return np.nan
BREAKEVEN_PV = dict(capex_pv=_bepv("capex_pv", 0.5, 3.0), tariff_energy=_bepv("tariff_energy", 0.2, 2.0), eta_scale=_bepv("eta_scale", 0.3, 1.5), r=_bepv("r", 0.01, 0.3))

# ---------- export-remuneration policy scenarios: optimal capacity against export credit
def sweep_x(x, pts_=None):
    pts_ = pts if pts_ is None else pts_
    out = []
    for p in pts_:
        a, T = allocate(p); e = economics(a, T, {"exp_thb": x})
        gen = e["pv_kwh"] + e["bg_kwh"]
        out.append(dict(exp_thb=x, p_kwp=p, npv=e["npv"], irr=e["irr"], sc_share=(gen - e["ex_kwh"]) / max(gen, 1e-9), ex_share=e["ex_kwh"] / max(gen, 1e-9),
                        cover=e["sc_kwh"] / e["load_kwh"]))
    return pd.DataFrame(out)
XGRID = np.round(np.arange(0.0, 3.01, 0.1), 2)
POLICY_SWEEPS = pd.concat([sweep_x(x) for x in XGRID], ignore_index=True)
POLICY = POLICY_SWEEPS.loc[POLICY_SWEEPS.groupby("exp_thb").npv.idxmax()].reset_index(drop=True).rename(columns={"p_kwp": "p_opt", "npv": "npv_opt", "irr": "irr_opt"})
def npv_at(p, x, bg=True):
    a, T = allocate(p); return economics(a, T, {"exp_thb": x}, bg=bg)["npv"]
P_ROOF = T0["roof"]["kwp"]
def _bx(f, lo=0.0, hi=6.0):
    try: return brentq(f, lo, hi)
    except ValueError: return np.nan
POLICY_SUM = dict(
    table={str(x): POLICY[POLICY.exp_thb == x].iloc[0].to_dict() for x in (0.0, 0.5, 1.0, 1.5, 2.0, 2.2, 2.5, 3.0)},
    x_opt_ge_3mw=float(POLICY[POLICY.p_opt >= 3000].exp_thb.min()) if (POLICY.p_opt >= 3000).any() else np.nan,
    x_opt_ge_roof=float(POLICY[POLICY.p_opt >= P_ROOF - 1].exp_thb.min()) if (POLICY.p_opt >= P_ROOF - 1).any() else np.nan,
    x_roof_npv0=_bx(lambda x: npv_at(P_ROOF, x)), x_roof_npv0_pvonly=_bx(lambda x: npv_at(P_ROOF, x, False)),
    x_rule_npv0=_bx(lambda x: npv_at(P_RULE, x)), x_full_npv0=_bx(lambda x: npv_at(P_TECH_ALL, x), 0, 12),
    x_noRes_npv0=_bx(lambda x: economics(*allocate(P_TECH_NO_RES, None, False), {"exp_thb": x})["npv"], 0, 12),
    p_roof=P_ROOF)

# ---------- biogas sensitivity: waste-management credit needed for NPV = 0 of the digester alone
def bg_breakeven(mods=None, extra_rev=0.0, extra_cost=0.0):
    mods = mods or {}; old = {k: P[k] for k in mods}; P.update(mods)
    try:
        e = economics(*allocate(0.0)); bgm = biogas()
        return dict(be=(-e["npv"] / ANN - extra_rev + extra_cost * bgm["waste_t_y"]) / bgm["waste_t_y"], e_net=bgm["e_net"], ch4=bgm["ch4_y"])
    finally: P.update(old)
_bg0 = biogas()
BG_SENS = {"Base case": bg_breakeven(),
           "HRT 20 d": bg_breakeven({"hrt": 20.0}), "HRT 40 d": bg_breakeven({"hrt": 40.0}),
           "k_d 0.09 d⁻¹": bg_breakeven({"kd": 0.09}), "k_d 0.15 d⁻¹": bg_breakeven({"kd": 0.15}),
           "η_e 0.28": bg_breakeven({"eta_e": 0.28}), "η_e 0.38": bg_breakeven({"eta_e": 0.38}),
           "Electrical parasitic 15 %": bg_breakeven({"par_e": 0.15}),
           "Heat valued at 0.03 USD kWh⁻¹": bg_breakeven(extra_rev=_bg0["h_net"] * 0.03),
           "Digestate sold at 5 USD t⁻¹": bg_breakeven(extra_rev=_bg0["digestate_t_y"] * 5.0),
           "Collection and pre-treatment 5 USD t⁻¹": bg_breakeven(extra_cost=5.0),
           "Digester CAPEX −30 %": bg_breakeven({"capex_bg": P["capex_bg"] * 0.7}), "Digester CAPEX +30 %": bg_breakeven({"capex_bg": P["capex_bg"] * 1.3})}
for lk in (0.0, 0.02, 0.05):
    BG_SENS[f"CH₄ leakage {lk*100:.0f} % (GHG only)"] = dict(be=BG_SENS["Base case"]["be"], e_net=_bg0["e_net"], ch4=_bg0["ch4_y"],
        net_ghg=BG["waste_t_y"] * P["doc"] * P["docf"] * P["mcf"] * P["f_ch4"] * 16 / 12 * P["gwp_ch4"] - _bg0["ch4_y"] * P["ch4_kg_m3"] / 1000 * lk * P["gwp_ch4"] + _bg0["e_net"] * P["ef_grid"] / 1000)

COV = pd.DataFrame([dict(ceiling=c, ponds=tiers(c)["ponds"]["kwp"] / 1000, res=tiers(c)["reservoir"]["kwp"] / 1000,
                         fpv_mwh=(tiers(c)["ponds"]["kwp"] + tiers(c)["reservoir"]["kwp"]) * YIELD_FPV / 1000,
                         total=sum(t["kwp"] for t in tiers(c).values()) / 1000) for c in (.15, .30, .45, .60)])


# =====================================================================================
# 8b. FRAMEWORK ANALYTICS: marginal self-consumption rule, decision map, benchmarks, multi-year yield
# =====================================================================================
C_PV = TIER_LCOE_ROOF = None
def _tier_lcoe_roof():
    return economics({"roof": 1000.0, "ground": 0, "ponds": 0, "reservoir": 0}, T0, bg=False)["lcoe"]
C_PV = _tier_lcoe_roof()
GROOF = PV_LAND * DT                                         # kWh per kWp per interval (roof/ground)
L_NET = LOAD - BG["e_net"] / NSTEP                           # demand left after the constant biogas output
def self_consumed(p_kwp, load=None, g=GROOF):
    load = L_NET if load is None else load
    G = p_kwp * g
    return np.minimum(G, load).sum(), G.sum()
# coincidence curve sigma(R_d): marginal share of PV energy used on campus (year 1, 15-min)
PGRID = np.arange(0, 40001, 250.0)
def sigma_curve(load=None, g=GROOF):
    load = L_NET if load is None else load
    S, G = np.array([self_consumed(p, load, g) for p in PGRID]).T
    sig = np.diff(S) / np.diff(G); rd = (G[1:] + G[:-1]) / 2 / LOAD.sum()
    return rd, sig, S, G
RD, SIG, S_CURVE, G_CURVE = sigma_curve()
# value of a self-consumed PV kWh at x = 0 (energy + demand + Ft), from the full bill model over the first 3 MWp
_e0 = economics(*allocate(0.0), bg=False); _e3 = economics(*allocate(3000.0), bg=False)
V_R = (_e3["sav1_usd"] - _e0["sav1_usd"]) / (_e3["sc_kwh"] - _e0["sc_kwh"])
R_R = V_R / C_PV
def p_star(Rx, Rr=None, sig=SIG):
    """analytic optimum on the same 250 kWp grid as the simulation: the last capacity before the first block whose
    marginal self-consumption share falls below (1 - R_x)/(R_r - R_x)  (Eq. 5); valid for R_x < 1 and one cost tier"""
    Rr = R_R if Rr is None else Rr
    if Rx >= 1: return np.inf
    s_star = (1 - Rx) / (Rr - Rx)
    idx = np.flatnonzero(sig < s_star)
    return float(PGRID[idx[0]]) if len(idx) else float(PGRID[-1])
def rd_star(Rx, Rr=None, rd=None, sig=SIG):
    p = p_star(Rx, Rr, sig)
    return p * YIELD_LAND / LOAD.sum() if np.isfinite(p) else np.inf
ANALYTIC = []
for x in XGRID:
    Rx = x / FX / C_PV; p_an = p_star(Rx); r = rd_star(Rx)
    p_sim = float(POLICY[POLICY.exp_thb == x].p_opt.iloc[0])
    ANALYTIC.append(dict(exp_thb=x, Rx=Rx, rd_analytic=r, p_analytic=p_an, p_sim=p_sim, rd_sim=p_sim * YIELD_LAND / LOAD.sum()))
ANALYTIC = pd.DataFrame(ANALYTIC)
# load-shape transfer: same annual energy, different temporal shapes
_hour = TIDX.hour.values; _wk = np.repeat((CLS.reshape(365, 96) == 0).any(axis=1), 96)
_flat = np.full(NSTEP, L_NET.sum() / NSTEP)
_office = np.where(_wk & (_hour >= 8) & (_hour < 17), 1.0, 0.15); _office = _office / _office.sum() * L_NET.sum()
_evening = np.where((_hour >= 17) & (_hour < 23), 1.0, 0.35); _evening = _evening / _evening.sum() * L_NET.sum()
LOAD_SHAPES = {"CRRU metered": L_NET, "Flat (24 h)": _flat, "Office hours": _office, "Evening peak": _evening}
SHAPE_CURVES = {k: sigma_curve(v)[:2] for k, v in LOAD_SHAPES.items()}
SHAPE_OPT = {k: {f"{x:.2f}": rd_star(x / FX / C_PV, None, None, SHAPE_CURVES[k][1]) for x in (0.0, 1.0, 2.2)} for k in LOAD_SHAPES}
RR_OPT = {f"{r:.2f}": rd_star(0.0, r) for r in (1.15, R_R, 1.6, 2.0)}
# decision map: marginal value index M = R_r sigma + R_x (1 - sigma) - 1 on a (R_x, R_d) grid
MAP_RX = np.linspace(0, 1.4, 141); MAP_RD = RD[RD <= 5.0]
MAP_M = np.array([[R_R * s + rx * (1 - s) - 1 for rx in MAP_RX] for s in SIG[RD <= 5.0]])

# ---------- benchmark against simplified sizing methods
_mon_load = np.array([L_NET[MONTH == m].sum() for m in range(12)]); _mon_y = np.array([GROOF[MONTH == m].sum() for m in range(12)])
P_ANNUAL = L_NET.sum() / YIELD_LAND                                   # annual net-zero sizing
P_MONTHLY = float(np.min(_mon_load / _mon_y))                         # no monthly surplus in any month
# average-day method: monthly mean diurnal profiles (hourly), energy rates only
_h = TIDX.hour.values
_Lmh = np.array([[L_NET[(MONTH == m) & (_h == h)].mean() * 4 for h in range(24)] for m in range(12)])      # kW
_Gmh = np.array([[GROOF[(MONTH == m) & (_h == h)].mean() * 4 for h in range(24)] for m in range(12)])     # kW per kWp
_on_frac = np.array([[(CLS[(MONTH == m) & (_h == h)] == 0).mean() for h in range(24)] for m in range(12)])
_price = (_on_frac * P["e_on"] + (1 - _on_frac) * P["e_off"] + P["ft"]) / FX                               # USD/kWh
_days = np.array([np.sum(MONTH == m) / 96 for m in range(12)])
AF = float((((1 - P["deg"]) ** np.arange(1, P["N"] + 1)) / (1 + P["r"]) ** np.arange(1, P["N"] + 1)).sum())
def avgday_benefit(p, x):
    G = p * _Gmh; S = np.minimum(G, _Lmh); X = G - S
    return float((_days[:, None] * (S * _price + X * x / FX)).sum())
BENCH_PTS = np.arange(250, 12001, 250.0)
def bench(x):
    full = {p: economics(*allocate(p), {"exp_thb": x}) for p in BENCH_PTS}
    p_ms = max(full, key=lambda p: full[p]["npv"])
    est = {p: full[p]["npv"] + (avgday_benefit(p, x) - full[p]["sav1_usd"] + _e0["sav1_usd"] * 0) * AF for p in BENCH_PTS}
    p_ad = max(est, key=lambda p: est[p])
    out = {}
    for lab, p in (("Annual energy balance", P_ANNUAL), ("Monthly energy balance", P_MONTHLY), ("Average-day hourly profile", p_ad), ("MSROF (15-min, full tariff)", p_ms)):
        e = economics(*allocate(p), {"exp_thb": x})
        out[lab] = dict(p_kwp=p, npv=e["npv"], capex=e["capex"], loss=e["npv"] - full[p_ms]["npv"], extra_capex=e["capex"] - full[p_ms]["capex"],
                        sc_share=(e["pv_kwh"] + e["bg_kwh"] - e["ex_kwh"]) / (e["pv_kwh"] + e["bg_kwh"]))
    return out
BENCH = {f"{x:.2f}": bench(x) for x in (0.0, 1.0, 2.2)}

# ---------- multi-year yield: bootstrap of measured monthly specific yield 2023-2026 (reference array)
YBY = PREP["yield_by_year"]
_fac = {m: [YBY[f"{y}-{m + 1:02d}"] / YBY[f"2025-{m + 1:02d}"] for y in (2023, 2024, 2025, 2026) if f"{y}-{m + 1:02d}" in YBY] for m in range(12)}
rngw = np.random.default_rng(7); NW = 500
W_SCALES = np.array([[rngw.choice(_fac[m]) for m in range(12)] for _ in range(NW)])
W_SCALES[0] = [YBY[f"2024-{m + 1:02d}"] / YBY[f"2025-{m + 1:02d}"] for m in range(12)]           # actual 2024 as first draw
_pts_w = np.arange(250, 4001, 250.0)
def wyear(ms):
    th = {"mscale": ms}
    nN = economics(SC["N"]["alloc"], SC["N"]["T"], th)["npv"]; nNpv = economics(SC["N"]["alloc"], SC["N"]["T"], th, bg=False)["npv"]
    opt = {}
    for x in (0.0, 2.2):
        v = [economics(*allocate(p), {"mscale": ms, "exp_thb": x})["npv"] for p in _pts_w]; opt[x] = float(_pts_w[int(np.argmax(v))])
    yf = float((ms[MONTH] * GROOF).sum() / GROOF.sum())
    return [nN, nNpv, opt[0.0], opt[2.2], yf]
WY = np.array([wyear(ms) for ms in W_SCALES])
pc = lambda a: dict(p10=float(np.percentile(a, 10)), p50=float(np.percentile(a, 50)), p90=float(np.percentile(a, 90)))
MULTIYEAR = dict(n=NW, months_years={m + 1: len(v) for m, v in _fac.items()}, npv_N=pc(WY[:, 0]), npv_N_pv=pc(WY[:, 1]), popt_x0=pc(WY[:, 2]),
                 popt_x22=pc(WY[:, 3]), yield_factor=pc(WY[:, 4]), p_npvN_pos=float((WY[:, 0] > 0).mean() * 100), p_npvNpv_pos=float((WY[:, 1] > 0).mean() * 100),
                 actual_2024=dict(npv_N=float(WY[0, 0]), npv_N_pv=float(WY[0, 1]), popt_x0=float(WY[0, 2]), popt_x22=float(WY[0, 3]), yield_factor=float(WY[0, 4])))

# =====================================================================================
# 9. CO-BENEFITS AND COMPOSITE INDICATORS
# =====================================================================================
CH4_PER_T = P["doc"] * P["docf"] * P["mcf"] * P["f_ch4"] * 16 / 12
def cobenefits(s):
    pv_t = s["pv_kwh"] * P["ef_grid"] / 1000; bg_t = s["bg_kwh"] * P["ef_grid"] / 1000
    lf = BG["waste_t_y"] * CH4_PER_T * P["gwp_ch4"]; leak = BG["ch4_y"] * P["ch4_kg_m3"] / 1000 * P["leak"] * P["gwp_ch4"]
    fpv_area = sum(s["T"][k]["area"] * s["alloc"][k] / s["T"][k]["kwp"] for k in ("ponds", "reservoir") if s["alloc"][k] > 0)
    return dict(pv_t=pv_t, bg_t=bg_t, landfill_t=lf, leak_t=-leak, ghg=pv_t + bg_t + lf - leak, evap_m3=fpv_area * P["evap"],
                dig_m3=BG["digestate_t_y"], fpv_area=fpv_area)
def metrics(s, co):
    a, T = s["alloc"], s["T"]
    used = sum(T[k]["area"] * a[k] / T[k]["kwp"] for k in a if a[k] > 0)
    water = sum(T[k]["water"] for k in ("ponds", "reservoir") if a[k] > 0) / 1e4
    e_fpv = sum(a[k] for k in ("ponds", "reservoir")) * YIELD_FPV / 1000
    return dict(used_m2=used, SEI=s["p_kwp"] * 1000 / A_CAMPUS * (1 - used / A_CAMPUS),
                CNED=(s["pv_kwh"] + s["bg_kwh"]) / (s["npc"] * used), WENI=(e_fpv + 0.0036 * co["evap_m3"]) / water if water > 0 else np.nan,
                CAI=co["ghg"] / ((A_CAMPUS - used) / 1e4))
CO = {k: cobenefits(v) for k, v in SC.items()}; MET = {k: metrics(v, CO[k]) for k, v in SC.items()}
TIER_LCOE = {k: economics({kk: (1000.0 if kk == k else 0.0) for kk in T0}, T0, bg=False)["lcoe"] for k in T0}

# =====================================================================================
# 10. EXPORT
# =====================================================================================
strip = lambda e: {k: v for k, v in e.items() if k not in ("T", "cf")}
R = dict(a_roof_elig=A_ROOF_ELIG, a_res=A_RES, a_small=A_SMALL, fx=FX, eta=P["eta"], yield_land=YIELD_LAND, yield_fpv=YIELD_FPV, yield_land_lt=PV_LAND_LT.sum() * DT, ghi_2025=GHI.sum() * DT / 1000,
         ghi_lt=GHI_LT.sum() * DT / 1000, tiers={k: dict(kwp=v["kwp"], area=v["area"]) for k, v in T0.items()}, tier_lcoe=TIER_LCOE,
         p_tech_no_res=P_TECH_NO_RES, p_tech_all=P_TECH_ALL, p_npvmax=P_NPVMAX, p_k08=P_K08, p_k10=P_K10, p_k12=P_K12, p_rule=P_RULE,
         base_bill=dict((k, float(v / FX)) for k, v in BASE_BILL.items() if k not in ("dmax", "n_min_binding")), base_dmax=[float(x) for x in BASE_BILL["dmax"]],
         biogas=BG, biogas_batch=BG_BATCH, bg_only=dict(npv=zero["npv"], lcoe=zero["lcoe"], capex=zero["capex"], waste_credit_be=WASTE_CREDIT_BE),
         pv_only={k: strip(v) for k, v in PV_ONLY.items()}, scenarios={k: strip(v) for k, v in SC.items()}, blocks=BLOCKS, lt={k: strip(v) for k, v in LT.items()},
         exp_100={k: strip(v) for k, v in EXP_100.items()}, exp_220={k: strip(v) for k, v in EXP_220.items()}, policy=POLICY_SUM, bg_sens=BG_SENS, framework=dict(c_pv=C_PV, v_r=V_R, R_r=R_R, p_annual=P_ANNUAL, p_monthly=P_MONTHLY, shape_opt=SHAPE_OPT, rr_opt=RR_OPT, af=AF), bench=BENCH, multiyear=MULTIYEAR, calib=PREP["calibration"], eta_low={k: strip(v) for k, v in ETA_LOW.items()}, mc=MCS,
         tornado=TORNADO, be_mult=BE_MULT, breakeven=BREAKEVEN, breakeven_pv=BREAKEVEN_PV, cobenefits=CO, metrics=MET, ch4_per_t=CH4_PER_T)
json.dump(R, open("outputs/results.json", "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
SWEEP.to_csv("outputs/sweep.csv", index=False); POLICY.to_csv("outputs/policy.csv", index=False); ANALYTIC.to_csv("outputs/analytic_optimum.csv", index=False); np.savez("outputs/framework.npz", rd=RD, sig=SIG, map_rx=MAP_RX, map_rd=MAP_RD, map_m=MAP_M, wy=WY, **{"shape_" + str(i): np.vstack(v) for i, v in enumerate(SHAPE_CURVES.values())}); POLICY_SWEEPS.to_csv("outputs/policy_sweeps.csv", index=False); BATT_CURVE.to_csv("outputs/batt_curve.csv", index=False)
pd.DataFrame(FX_ROWS).to_csv("outputs/fx.csv", index=False); COV.to_csv("outputs/fpv_ceiling.csv", index=False);
for k, v in MC.items(): np.save(f"outputs/mc_{k}.npy", v)
np.savez_compressed("outputs/profiles.npz", load=LOAD, pv_land=PV_LAND, cls=CLS, ghi=GHI, t=M["t_start"])

if __name__ == "__main__":
    print(f"eta {P['eta']:.3f} | yield land {YIELD_LAND:,.0f} fpv {YIELD_FPV:,.0f} (LT {R['yield_land_lt']:,.0f}) kWh/kWp | GHI 2025 {R['ghi_2025']:.0f} LT {R['ghi_lt']:.0f}")
    print("tiers MWp", {k: round(v['kwp'] / 1000, 2) for k, v in T0.items()}, "| base bill USD", {k: round(v) for k, v in R['base_bill'].items()})
    print(f"knee 12/10/8 % : {P_K12/1000:.2f}/{P_K10/1000:.2f}/{P_K08/1000:.2f} MWp | NPV-max {P_NPVMAX/1000:.2f} | rule {P_RULE/1000:.2f}")
    for k, v in SC.items():
        print(f"{k:3s} {v['p_kwp']/1000:6.2f} MWp CAPEX {v['capex']/1e6:6.2f} LCOE {v['lcoe']:.4f} NPV {v['npv']/1e6:7.2f} IRR {v['irr']*100:5.1f} pay {v['payback']:.1f} "
              f"sav {v['sav1_usd']/1e3:,.0f}k (E {v['sav_energy_usd']/1e3:,.0f} D {v['sav_demand_usd']/1e3:,.0f} X {v['credit_usd']/1e3:,.0f}) imp {v['imp_kwh']/1e6:.2f} ex {v['ex_kwh']/1e6:.2f} GWh "
              f"cover {v['sc_kwh']/v['load_kwh']*100:.1f}% batt {v['capex_batt']/1e6:.2f}")
    print("blocks", {k: {kk: round(vv, 3) for kk, vv in b.items()} for k, b in BLOCKS.items()})
    for k, v in MCS.items(): print("MC", k, {kk: round(vv, 3) for kk, vv in v.items()})
    print("biogas", {k: round(v, 1) for k, v in BG.items()}, "| waste credit BE", round(WASTE_CREDIT_BE, 1), "| Li BE mult", BE_MULT)
    print("policy", POLICY[["exp_thb","p_opt","npv_opt","irr_opt","sc_share","ex_share"]].round(3).to_string()); print({k: v for k, v in POLICY_SUM.items() if k != "table"})
    print("bg sens", {k: round(v["be"], 1) for k, v in BG_SENS.items()}); print("min charge topups base", BASE_BILL["n_min_binding"])
    print("framework", R["framework"]); print(ANALYTIC.round(3).to_string()); print(json.dumps(BENCH, indent=0, default=float)[:3000]); print(MULTIYEAR)
