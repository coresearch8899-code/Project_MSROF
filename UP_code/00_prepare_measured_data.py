"""
MSROF-CRRU | Script 0 of 3 | Preparation of the measured datasets
=================================================================
Reads the raw measured files (no licensed data, no external service) and writes the
15-minute arrays used by the analysis, plus the PV-model calibration statistics.

Inputs (folder given as first argument, default "Multi_surface"):
  Data_load/*.xls   PEA 15-min interval-meter exports (HTML tables), account 020005835092,
                    meter 6201224158, TOU tariff, 22 kV, January to December 2025.
                    Columns: interval end (dd/mm/yyyy HH.MM), RATE A (on-peak), RATE B (off-peak),
                    RATE C (holiday/weekend), total kWh per 15 min.
  Data_PV/CRH_merged_30min_zone_level.csv
                    CRH Solar Rooftop monitoring (Chiang Rai, ~5 km from campus), zones of 2,256.24,
                    207.06 and 49.90 kWp, 30-min logs: daily energy counter, instantaneous horizontal
                    irradiance, daily irradiation, ambient temperature (March 2023 to September 2026).
  Data_PV/monthly-YYYY-MM.csv
                    Daily logs of the existing CRRU rooftop PV fleet (10 buildings) and campus weather
                    station, 2023 to 2024.

Outputs: prepared/measured_2025.npz, prepared/prep_summary.json
Usage:   python 00_prepare_measured_data.py <path-to-Multi_surface>
"""
import sys, os, io, glob, json
import numpy as np, pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "Multi_surface"
os.makedirs("prepared", exist_ok=True)
S = {}

# ------------------------------------------------------------------ 1. PEA interval meter, 2025
frames = []
for f in sorted(glob.glob(os.path.join(ROOT, "Data_load", "*.xls"))):
    tabs = pd.read_html(io.StringIO(open(f, encoding="utf-8").read()))
    d = tabs[1].iloc[1:].copy(); d.columns = ["ts", "A", "B", "C", "tot"]
    d = d[~d.ts.astype(str).str.contains("ผลรวม")]
    for c in "A B C tot".split(): d[c] = pd.to_numeric(d[c], errors="coerce")
    dd = d.ts.str.slice(0, 10); hm = d.ts.str.slice(11).str.replace(".", ":", regex=False)
    end = pd.to_datetime(dd, format="%d/%m/%Y") + pd.to_timedelta(hm + ":00")      # "24:00" -> next day 00:00
    d["start"] = end - pd.Timedelta(minutes=15)
    frames.append(d)
L = pd.concat(frames).sort_values("start").set_index("start")
L = L[~L.index.duplicated()]
idx = pd.date_range("2025-01-01 00:00", "2025-12-31 23:45", freq="15min")
L = L.reindex(idx)
assert L.tot.isna().sum() == 0, "missing load intervals"
cls = np.where(L.A.notna() & (L.A > 0), 0, np.where(L.B.notna() & (L.B > 0), 1, 2))   # 0 on-peak, 1 off-peak, 2 holiday
# intervals with zero energy keep the class of their position in the TOU calendar
onpk_cal = (L.index.dayofweek < 5) & (((L.index + pd.Timedelta(minutes=15)).hour * 60 + (L.index + pd.Timedelta(minutes=15)).minute) > 9 * 60) & \
           (((L.index + pd.Timedelta(minutes=15)).hour * 60 + (L.index + pd.Timedelta(minutes=15)).minute) <= 22 * 60)
load_kwh = L.tot.values.astype(float)
monthly = pd.Series(load_kwh, idx).resample("MS").sum()
S["load"] = dict(annual_kwh=float(load_kwh.sum()), monthly_kwh=[float(x) for x in monthly],
                 peak_kw=float(load_kwh.max() * 4), peak_time=str(idx[load_kwh.argmax()]),
                 onpeak_share=float(load_kwh[cls == 0].sum() / load_kwh.sum()),
                 holiday_intervals=int((cls == 2).sum()), mean_kw=float(load_kwh.mean() * 4),
                 load_factor=float(load_kwh.mean() / load_kwh.max()),
                 monthly_onpeak_max_kw=[float(pd.Series(np.where(cls == 0, load_kwh, 0), idx).resample("MS").max().iloc[m] * 4) for m in range(12)],
                 daytime_share_08_17=float(load_kwh[(idx.hour >= 8) & (idx.hour < 17)].sum() / load_kwh.sum()),
                 tou_calendar_agreement=float(((cls == 0) == onpk_cal).mean()))

# ------------------------------------------------------------------ 2. CRH monitoring: weather 2025 + PV validation
c = pd.read_csv(os.path.join(ROOT, "Data_PV", "CRH_merged_30min_zone_level.csv"), encoding="utf-8-sig", low_memory=False)
cols = list(c.columns)
c = c.rename(columns={cols[1]: "plant", cols[2]: "t", cols[3]: "kwp", cols[8]: "e_day", cols[18]: "g", cols[19]: "h_day",
                      cols[20]: "curt", cols[21]: "tamb"})
c["t"] = pd.to_datetime(c.t, errors="coerce")
for k in ("kwp", "e_day", "g", "h_day", "curt", "tamb"): c[k] = pd.to_numeric(c[k], errors="coerce")

z1 = c[c.plant.str.contains("Zone1")].set_index("t").sort_index(); z1 = z1[~z1.index.duplicated()]

def build_weather(t0, t1):
    """15-min irradiance and temperature at interval midpoints between t0 and t1 (inclusive),
    each day rescaled to the logged daily irradiation, gaps filled with the same-month mean diurnal profile."""
    ix = pd.date_range(t0, t1, freq="15min")
    w = z1.loc[ix[0] - pd.Timedelta(hours=1): ix[-1] + pd.Timedelta(hours=1), ["g", "tamb", "h_day"]]
    mid = ix + pd.Timedelta(minutes=7.5)
    g15 = np.interp(mid.values.astype("int64"), w.index.values.astype("int64"), w.g.interpolate(limit=4).values, left=np.nan, right=np.nan)
    t15 = np.interp(mid.values.astype("int64"), w.index.values.astype("int64"), w.tamb.interpolate(limit=8).values, left=np.nan, right=np.nan)
    G_ = pd.Series(np.clip(g15, 0, None), ix); T_ = pd.Series(t15, ix)
    ww = w.loc[t0:t1]
    hd = ww.h_day.groupby(ww.index.date).max() / 1000.0
    gd = G_.groupby(G_.index.date).sum() * 0.25 / 1000.0
    fac = (hd / gd).replace([np.inf, -np.inf], np.nan); good = fac.between(0.5, 1.6) & (hd > 0.3)
    G_ = G_ * pd.Series(pd.Index(G_.index.date).map(fac.where(good)), ix).fillna(1.0).values
    filled = np.isnan(G_.values)
    key = pd.MultiIndex.from_arrays([G_.index.month, G_.index.hour * 4 + G_.index.minute // 15])
    for s_ in (G_, T_):
        prof = s_.groupby(key).transform("mean"); s_.fillna(prof, inplace=True)
    return G_, T_, dict(days_rescaled=int(good.sum()), days_total=int(len(gd)), filled=int(filled.sum()),
                        filled_share=float(G_.values[filled].sum() / G_.values.sum()) if filled.any() else 0.0,
                        not_rescaled_share=float(G_[~pd.Index(G_.index.date).isin(fac[good].index)].sum() / G_.sum()))

G, T, wq = build_weather("2025-01-01 00:00", "2025-12-31 23:45")
S["weather_2025"] = dict(annual_ghi_kwh_m2=float(G.sum() * 0.25 / 1000), monthly_ghi_kwh_m2_day=[float(x) for x in (G.resample("MS").sum() * 0.25 / 1000 / G.resample("MS").count() * 96)],
                         mean_tamb=float(T.mean()), days_rescaled=wq["days_rescaled"], filled_g=wq["filled"],
                         filled_energy_share=wq["filled_share"], not_rescaled_energy_share=wq["not_rescaled_share"])

# PV-model calibration: measured 30-min energy vs modelled output from measured G and T.
# Calibration period: 2024 and January-August 2026; validation period: 2025 (held out).
def calib(zone, kwp, period):
    s = c[c.plant.str.contains(zone)].set_index("t").sort_index(); s = s[~s.index.duplicated()].loc["2024-01-01":"2026-08-31"].copy()
    s["date"] = s.index.date
    s["P"] = s.groupby("date").e_day.diff().clip(lower=0) * 2
    s["Gm"] = (s.g + s.g.shift(1)) / 2; s["Tm"] = (s.tamb + s.tamb.shift(1)) / 2
    v = s.dropna(subset=["P", "Gm", "Tm"]); v = v[v.Gm > 300]
    v = v[v.index.year != 2025] if period == "cal" else (v[v.index.year == 2025] if period == "val" else v)
    tc = v.Tm + (45 - 20) / 800 * v.Gm
    r = v.P / (kwp * v.Gm / 1000 * (1 - 0.0041 * (tc - 25)))
    return dict(kwp=kwp, n=int(len(r)), median=float(r.median()), q25=float(r.quantile(.25)), q75=float(r.quantile(.75)), r_values=r.clip(0, 1.5).values)
ZONES = (("Zone1", 2256.24), ("Zone2", 207.06), ("Zone3", 49.90))
cal = {z: calib(z, k, "cal") for z, k in ZONES}; val = {z: calib(z, k, "val") for z, k in ZONES}; allp = {z: calib(z, k, "all") for z, k in ZONES}
pooled = np.concatenate([v.pop("r_values") for v in cal.values()])
pooled_val = np.concatenate([v.pop("r_values") for v in val.values()])
pooled_all = np.concatenate([v.pop("r_values") for v in allp.values()])
ETA = float(np.median(pooled))
np.save("prepared/calibration_ratios.npy", pooled_all); np.save("prepared/calibration_ratios_cal.npy", pooled); np.save("prepared/calibration_ratios_val.npy", pooled_val)

def zone1_energy(t0, t1):
    """measured daily-counter energy and model energy (eta = 1) of zone 1 over full days with valid counters."""
    Gp, Tp, _ = build_weather(t0, t1)
    em = z1.loc[t0:t1].e_day.groupby(z1.loc[t0:t1].index.date).max(); em.index = pd.to_datetime(em.index)
    pm1 = pd.Series((2256.24 * Gp / 1000 * (1 - 0.0041 * (Tp + (45 - 20) / 800 * Gp - 25))).values * 0.25, Gp.index).resample("D").sum()
    ok = em.reindex(pm1.index).notna() & (em.reindex(pm1.index) > 0)
    return em.reindex(pm1.index)[ok], pm1[ok]
# calibration: energy-match on 2024 + Jan-Aug 2026 (days with valid counters)
em24, pm24 = zone1_energy("2024-01-01 00:00", "2024-12-31 23:45")
em26, pm26 = zone1_energy("2026-01-01 00:00", "2026-08-31 23:45")
ETA_CAL = float((em24.sum() + em26.sum()) / (pm24.sum() + pm26.sum()))
# validation: 2025 held out
em25, pm25 = zone1_energy("2025-01-01 00:00", "2025-12-31 23:45")
mm = em25.resample("MS").sum(); mp = (pm25 * ETA_CAL).resample("MS").sum()
S["calibration"] = dict(zones_cal=cal, zones_val=val, zones_all=allp, pooled_n=int(len(pooled)), pooled_n_val=int(len(pooled_val)), pooled_n_all=int(len(pooled_all)),
                        eta_median=ETA, eta_q25=float(np.quantile(pooled, .25)), eta_q75=float(np.quantile(pooled, .75)),
                        eta_median_val=float(np.median(pooled_val)), eta_median_all=float(np.median(pooled_all)),
                        eta_cal=ETA_CAL, cal_days_2024=int(len(em24)), cal_days_2026=int(len(em26)),
                        cal_kwh_2024=float(em24.sum()), cal_kwh_2026=float(em26.sum()),
                        eta_2024_only=float(em24.sum() / pm24.sum()), eta_2026_only=float(em26.sum() / pm26.sum()),
                        eta_2025_energy_match=float(em25.sum() / pm25.sum()),
                        val_days_2025=int(len(em25)), val_measured_kwh=float(em25.sum()), val_model_kwh=float(pm25.sum() * ETA_CAL),
                        val_bias_pct=float((pm25.sum() * ETA_CAL / em25.sum() - 1) * 100),
                        val_monthly_nrmse_pct=float(np.sqrt(((mp - mm) ** 2).mean()) / mm.mean() * 100),
                        val_daily_r2=float(np.corrcoef(em25.values, pm25.values)[0, 1] ** 2),
                        zone1_2025_monthly_measured=[float(x) for x in mm], zone1_2025_monthly_model=[float(x) for x in mp])
# measured monthly specific yield of the reference array by year (months with >= 20 valid days), used for the
# multi-year resource analysis; independent of the pyranometer because it is based on the energy counter only
_d = z1.e_day.groupby(z1.index.date).max(); _d.index = pd.to_datetime(_d.index); _d = _d[_d > 0]
_m = _d.groupby([_d.index.year, _d.index.month]).agg(["mean", "count"])
S["yield_by_year"] = {f"{y}-{mo:02d}": float(v / 2256.24) for (y, mo), v, n in zip(_m.index, _m["mean"], _m["count"]) if n >= 20}
_g = z1.h_day.groupby(z1.index.date).max() / 1000; _g.index = pd.to_datetime(_g.index); _g = _g[(_g > 0.3) & (_g < 9)]
_gm = _g.groupby([_g.index.year, _g.index.month]).agg(["mean", "count"])
S["ghi_by_year"] = {f"{y}-{mo:02d}": float(v) for (y, mo), v, n in zip(_gm.index, _gm["mean"], _gm["count"]) if n >= 20}
# annual zone-1 2025 yield on the full-year weather series (benchmark)
e25 = z1.loc["2025"].e_day.groupby(z1.loc["2025"].index.date).max().sum()
S["calibration"]["zone1_2025_measured_kwh"] = float(e25); S["calibration"]["zone1_2025_measured_yield"] = float(e25 / 2256.24)
S["calibration"]["zone1_2025_PR_measured"] = float(e25 / 2256.24 / (G.sum() * 0.25 / 1000))

# ------------------------------------------------------------------ 3. Existing CRRU rooftop fleet (behind the meter)
out = []
for f in sorted(glob.glob(os.path.join(ROOT, "Data_PV", "monthly-*.csv"))):
    Ls = open(f, encoding="utf-8-sig").read().splitlines()
    grp, col = Ls[2].split(";"), Ls[3].split(";"); names = []; g_ = ""
    for a, b in zip(grp, col):
        if a.strip(): g_ = a.strip()
        names.append(f"{g_}|{b.strip()}")
    d = pd.read_csv(io.StringIO("\n".join(Ls[5:])), sep=";", header=None).iloc[:, :len(names)]
    d.columns = names[:d.shape[1]]; d = d.rename(columns={names[0]: "date"}); out.append(d)
F = pd.concat(out, ignore_index=True); F["date"] = pd.to_datetime(F.date, errors="coerce"); F = F.dropna(subset=["date"]).set_index("date")
pvm = pd.to_numeric(F["Pv Meter|Energy Today"], errors="coerce")
bld = {c_.split("|")[0]: float(pd.to_numeric(F[c_], errors="coerce").loc["2024"].sum()) for c_ in F.columns if c_.endswith("|PV Energy Today")}
S["existing_fleet"] = dict(kwh_2023=float(pvm.loc["2023"].sum()), days_2023=int(pvm.loc["2023"].count()),
                           kwh_2024=float(pvm.loc["2024"].sum()), days_2024=int(pvm.loc["2024"].count()),
                           mean_daily_2024=float(pvm.loc["2024"].mean()), by_building_2024=bld,
                           buildings_zero_share_2024={k: float((pd.to_numeric(F[k + "|PV Energy Today"], errors="coerce").loc["2024"] <= 0).mean()) for k in bld})

np.savez_compressed("prepared/measured_2025.npz", load_kwh=load_kwh, tou_class=cls, ghi_wm2=G.values, tamb=T.values,
                    t_start=idx.values.astype("datetime64[m]").astype("int64"))
json.dump(S, open("prepared/prep_summary.json", "w"), indent=1)
print(json.dumps({k: (v if k != "calibration" else {kk: vv for kk, vv in v.items() if "monthly" not in kk}) for k, v in S.items()}, indent=1)[:4000])
