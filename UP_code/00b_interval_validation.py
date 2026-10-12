"""
MSROF-CRRU | Script 0b | Interval-level checks of the measured inputs (run after script 00, needs the raw files)
==============================================================================================================
1  Quality screening of the PEA 15-min meter series (completeness, duplicates, zero and outlying intervals).
2  Interval-level validation of the calibrated PV model on the held-out year 2025: measured 30-min energy of the
   2,256 kWp reference array (CRH Solar Rooftop, Zone 1; the monitoring logs are 30-min, so 30 min is the finest
   resolution at which the model can be tested) against the model driven by the same 15-min weather series that
   the analysis uses, aggregated to 30 min. Errors are reported for daylight intervals by season, hour band,
   on-peak window and output level.
3  A measured 15-min specific-output series for 2025 (each 30-min measured energy split into its two 15-min halves
   in proportion to the modelled shape; invalid intervals filled with the model), used by script 06 to test how
   the PV-model error propagates to the optimum and the returns.
Outputs: prepared/interval_validation.json, prepared/pv_interval_2025.npz (15-min measured series and the 30-min validation arrays)
Usage:   python 00b_interval_validation.py <path-to-Multi_surface>
"""
import sys, os, json
import numpy as np, pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "Multi_surface"
SRC = open("00_prepare_measured_data.py", encoding="utf-8").read()
# re-use sections 1 and 2 of script 00 (meter parsing, weather builder, calibration) without writing any file
_save = np.save; np.save = lambda *a, **k: None
exec(SRC.split("# ------------------------------------------------------------------ 3.")[0].split('ROOT = sys.argv')[0])
exec("ROOT = %r\n" % ROOT + SRC.split("# ------------------------------------------------------------------ 3.")[0].split("os.makedirs(\"prepared\", exist_ok=True)")[1])
np.save = _save
PREP = json.load(open("prepared/prep_summary.json")); ETA_CAL = PREP["calibration"]["eta_cal"]
OUT = {}

# ------------------------------------------------------------------ 1. meter screening
raw = pd.concat(frames)
n_raw = int(len(raw)); n_dup = int(raw.set_index("start").index.duplicated().sum())
x = L.tot.values.astype(float)
key = pd.MultiIndex.from_arrays([L.index.month, L.index.hour * 4 + L.index.minute // 15, L.index.dayofweek >= 5])
med = pd.Series(x, L.index).groupby(key).transform("median").values
mad = pd.Series(np.abs(x - med), L.index).groupby(key).transform("median").values
rz = np.abs(x - med) / (1.4826 * np.maximum(mad, 1e-9))
abc = L[["A", "B", "C"]].sum(axis=1, min_count=1).values
OUT["meter"] = dict(rows_read=n_raw, duplicates_removed=n_dup, intervals_expected=int(len(L)), intervals_missing=int(np.isnan(x).sum()),
                    negative=int((x < 0).sum()), zero=int((x == 0).sum()),
                    robust_z_gt6=int((rz > 6).sum()), robust_z_gt6_energy_share=float(x[rz > 6].sum() / x.sum()),
                    max_kw=float(x.max() * 4), tou_columns_sum_mismatch=int((np.abs(abc - x) > 0.05).sum()),
                    tou_calendar_agreement=float(PREP["load"]["tou_calendar_agreement"]))

# ------------------------------------------------------------------ 2. interval validation of the PV model (2025, Zone 1)
KWP = 2256.24
s = z1.loc["2024-12-31 23:00":"2026-01-01 01:00"].copy()
s["date"] = s.index.date
s["d"] = s.groupby("date").e_day.diff()                                   # kWh in the 30 min ending at t
s["dl"] = pd.to_numeric(s[cols[10]], errors="coerce").groupby(s["date"]).diff()
s = s.loc["2025-01-01 00:30":"2026-01-01 00:00"]
idx30 = pd.date_range("2025-01-01 00:30", "2026-01-01 00:00", freq="30min")
s = s.reindex(idx30)
meas = s.d.values
valid = np.isfinite(meas) & (meas >= 0) & (meas <= KWP * 0.5 * 1.05)
# 15-min model on the analysis weather (interval starts), aggregated to 30-min periods ending at t
G15, T15 = G.values, T.values
pm15 = KWP * G15 / 1000 * (1 - 0.0041 * (T15 + (45 - 20) / 800 * G15 - 25)) * ETA_CAL * 0.25
pm15 = np.clip(pm15, 0, None)
mod = pm15.reshape(-1, 2).sum(axis=1)                                      # pairs (00:00,00:15) -> ending 00:30, ...
# timestamp convention check: correlation at lags of -1, 0, +1 half-hours
lag_r = {}
for lag in (-1, 0, 1):
    m2 = np.roll(mod, lag); ok = valid & (m2 + np.nan_to_num(meas) > 1)
    lag_r[lag * 30] = float(np.corrcoef(meas[ok], m2[ok])[0, 1])
OUT["timestamp_lag_corr"] = lag_r
t = idx30 - pd.Timedelta(minutes=15)                                       # mid-point of each 30-min period
day = valid & ((mod > 0.01 * KWP * 0.5) | (meas > 0.01 * KWP * 0.5))
season = np.select([np.isin(t.month, [11, 12, 1, 2]), np.isin(t.month, [3, 4, 5])], ["Cool (Nov-Feb)", "Hot (Mar-May)"], "Rainy (Jun-Oct)")
hb = np.select([t.hour < 9, t.hour < 12, t.hour < 15], ["06:00-09:00", "09:00-12:00", "12:00-15:00"], "15:00-19:00")
onpk = (t.dayofweek < 5) & (t.hour >= 9) & (t.hour < 22)
def stats(mask):
    m, p = meas[mask], mod[mask]
    if len(m) < 10: return None
    mu = m.mean(); e = p - m
    return dict(n=int(len(m)), mbe_pct=float(e.mean() / mu * 100), nmae_pct=float(np.abs(e).mean() / mu * 100),
                nrmse_pct=float(np.sqrt((e ** 2).mean()) / mu * 100), r2=float(np.corrcoef(m, p)[0, 1] ** 2),
                mean_meas_kw=float(mu * 2), p10_err_kw=float(np.percentile(e, 10) * 2), p90_err_kw=float(np.percentile(e, 90) * 2))
V = {"All daylight intervals": stats(day)}
for sn in ("Cool (Nov-Feb)", "Hot (Mar-May)", "Rainy (Jun-Oct)"): V[sn] = stats(day & (season == sn))
for h in ("06:00-09:00", "09:00-12:00", "12:00-15:00", "15:00-19:00"): V[h] = stats(day & (hb == h))
V["On-peak window (weekdays 09:00-22:00)"] = stats(day & onpk)
V["Measured output above 50% of rating"] = stats(day & (meas > 0.5 * KWP * 0.5))
# daily monthly-peak relevance: model error in the interval of each month's highest measured output
pk = []
for mo in range(1, 13):
    mm = day & (t.month == mo)
    if mm.any():
        i = np.flatnonzero(mm)[np.argmax(meas[mm])]; pk.append((mod[i] - meas[i]) / meas[i] * 100)
V["Monthly maximum-output interval (error, %)"] = dict(mean=float(np.mean(pk)), min=float(np.min(pk)), max=float(np.max(pk)))
OUT["pv_interval"] = V
OUT["pv_interval_meta"] = dict(resolution_min=30, valid_share=float(valid.mean()), n_daylight=int(day.sum()), eta=ETA_CAL,
                               note="measured = daily-counter differences of Zone 1 (2,256.24 kWp), 2025 held out from calibration")
# site load always exceeded PV output at the reference site (no export limitation of the measured output)
ld = s.dl.values
okl = valid & np.isfinite(ld) & (meas > 0)
OUT["pv_interval_meta"]["share_intervals_load_below_pv"] = float((ld[okl] < meas[okl]).mean()) if okl.any() else None
OUT["pv_interval_meta"]["feed_in_share_2025"] = float(pd.to_numeric(z1.loc["2025"][cols[14]], errors="coerce").groupby(z1.loc["2025"].index.date).max().sum() / z1.loc["2025"].e_day.groupby(z1.loc["2025"].index.date).max().sum())

# ------------------------------------------------------------------ 3. measured 15-min specific output for 2025
pm2 = pm15.reshape(-1, 2); tot = pm2.sum(axis=1)
share = np.where(tot[:, None] > 0, pm2 / np.where(tot[:, None] > 0, tot[:, None], 1), 0.5)
m15 = np.where(valid[:, None], np.nan_to_num(meas)[:, None] * share, pm2)  # kWh per 15 min
pv_meas = (m15.reshape(-1) / 0.25 / KWP)                                    # kW per kWp at 15-min interval starts
np.savez_compressed("prepared/pv_interval_2025.npz", pv_meas_per_kwp=pv_meas, filled=np.repeat(~valid, 2), meas30=np.nan_to_num(meas), mod30=mod, valid30=valid, day30=day,
                    t30=idx30.values.astype("datetime64[m]").astype("int64"))
OUT["measured_series"] = dict(filled_share_intervals=float((~valid).mean()), annual_kwh_per_kwp=float(pv_meas.sum() * 0.25),
                              model_kwh_per_kwp=float(pm15.sum() / KWP))
json.dump(OUT, open("prepared/interval_validation.json", "w"), indent=1)
print(json.dumps(OUT, indent=1))
