"""
MSROF-CRRU | Script 7 | Figures of the Utilities Policy version (PV only, utility perspective)
No titles; centred (a), (b), (c) labels; 900 dpi. Reads outputs/up_* (script 6), outputs/results.json, prepared/.
"""
import json, os, shutil
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 900, "savefig.bbox": "tight"})
os.makedirs("figures_up", exist_ok=True)
R = json.load(open("outputs/results.json")); U = json.load(open("outputs/up_results.json")); FX = R["fx"]
C = dict(roof="#D98E04", ground="#7A9E3B", fpv="#2E86AB", res="#1B5E7B", grid="#B0413E", gray="#6B6B6B", pur="#8E5572")
def tag(ax, s, y=1.04): ax.text(0.5, y, s, transform=ax.transAxes, ha="center", va="bottom", fontsize=11, fontweight="bold")

# ---------- Fig. 1 framework
fig, ax = plt.subplots(figsize=(13, 6.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")
def box(x, y, w, h, txt, fc, fs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec="#333333", lw=0.9)); ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)
def arr(x0, y0, x1, y1): ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2))
stages = [("1  Spatial\nresource", "Eligible surfaces $A_s$\nand tiers $s$; land-sparing\nsupply curve, $P_{max}$\n(Eqs. 1, 6, 9)"),
          ("2  Temporal\ngeneration", "Calibrated and\nvalidated output\nper kWp, $g_{s,k}$\n(Eqs. 7, 8)"),
          ("3  Load\ncoincidence", "Self-consumed $S_k$\nand exported $X_k$\nenergy per interval\n(Eq. 2)"),
          ("4  Energy\nmonetization", "TOU energy, demand\nand minimum charges;\nexport remuneration\n(Eqs. 3, 10)"),
          ("5  Marginal\ncapacity value", "Marginal IRR $\\rho$ and\nmarginal self-\nconsumption $\\sigma$\n(Eqs. 5, 16)"),
          ("6  Policy-dependent\noptimum", "$P^*(\\pi_x)$ and the\ndecision map in\n($R_x$, $R_d$)\n(Eqs. 4, 5)")]
xs = [0.25 + i * 2.13 for i in range(6)]; cols = ["#DCE9F2", "#E3EFD6", "#F6E7CC", "#F3DCDC", "#E6DFF0", "#D9E6E3"]
for (t, d), x, c in zip(stages, xs, cols):
    box(x, 2.45, 1.85, 1.95, "", c); ax.text(x + 0.925, 3.95, t, ha="center", va="center", fontsize=8.4, fontweight="bold"); ax.text(x + 0.925, 3.0, d, ha="center", va="center", fontsize=7.4)
for x in xs[:-1]: arr(x + 1.88, 3.42, x + 2.11, 3.42)
inputs = ["Satellite land-use and\nPV-zoning maps", "Near-campus irradiance and\nPV monitoring (calibration\n2024, 2026; validation 2025)", "15-min utility metering\n(single campus account)",
          "Tariff schedule, Ft, minimum\ncharge; compensation scheme", "Capital, O&M, discount\nrate, degradation", "Export remuneration\nsweep 0 to 3 THB/kWh"]
for x, t in zip(xs, inputs): box(x, 5.0, 1.85, 0.95, t, "#FFFFFF", fs=7.4); arr(x + 0.925, 5.0, x + 0.925, 4.32)
box(0.25, 0.35, 6.2, 1.25, "Institution: capacity-value curve, NPV-maximizing capacity, export rate needed\nfor each capacity, benchmark against simplified sizing; uncertainty (Monte Carlo,\nmulti-year measured yield) and test of Eq. (5) under five tariff structures", "#F2F2F2", fs=7.8)
box(6.65, 0.35, 6.1, 1.25, "Utility: retail revenue lost, export payments and avoided purchases\nunder uncompensated export, curtailment, net billing\nand net metering (Eq. 17);\nnet margin, joint NPV and break-even external value (Eq. 18)", "#F2F2F2", fs=7.8)
arr(3.35, 2.45, 3.35, 1.62); arr(9.7, 2.45, 9.7, 1.62)
plt.savefig("figures_up/fig1_framework.png"); plt.close()

# ---------- Fig. 4 capacity-value curve (PV only, no export remuneration)
CV = pd.read_csv("outputs/up_capacity_value.csv"); tr = R["tiers"]
fig, axs = plt.subplots(1, 3, figsize=(14, 4.4)); cum = 0
for k, c_, l in zip(["roof", "ground", "ponds", "reservoir"], [C["roof"], C["ground"], C["fpv"], C["res"]], ["Rooftop", "Ground-mounted", "Small ponds", "Reservoir"]):
    axs[0].bar(0, tr[k]["kwp"] / 1000, bottom=cum, color=c_, width=0.5)
    axs[0].text(0.32, cum + tr[k]["kwp"] / 1000 * (0.66 if k == "roof" else 0.5), f"{l}: {tr[k]['kwp'] / 1000:.2f} MWp", va="center", fontsize=8.5); cum += tr[k]["kwp"] / 1000
axs[0].axhline(U["P0"] / 1000, color="k", ls="--", lw=1); axs[0].text(0.32, U["P0"] / 1000 + 0.45, f"NPV-maximizing capacity ({U['P0'] / 1000:.2f} MWp)", fontsize=8.5)
axs[0].set_xlim(-0.35, 1.5); axs[0].set_xticks([]); axs[0].set_ylabel("Cumulative PV capacity (MWp)"); tag(axs[0], "(a)")
sw = CV[CV.p_kwp > 0]; ax = axs[1]
ax.plot(sw.p_kwp / 1000, sw.npv / 1e6, color=C["fpv"], lw=2); ax.axhline(0, color="k", lw=0.6); ax.set_xscale("log")
ax.set_xlabel("Installed PV capacity (MWp, log scale)"); ax.set_ylabel("25-year NPV (M USD)", color=C["fpv"])
ax2 = ax.twinx(); ax2.spines["right"].set_visible(True); ax2.plot(sw.p_kwp / 1000, sw.marg_irr * 100, color=C["grid"], lw=1.8); ax2.axhline(7, color=C["gray"], ls=":", lw=1)
ax2.text(0.27, 7.6, "discount rate 7 %", fontsize=8, color=C["gray"]); ax2.set_ylabel("Marginal IRR of the last 250 kWp (%)", color=C["grid"]); ax2.set_ylim(-10, 16)
o = CV.loc[CV.npv.idxmax()]; ax.plot(o.p_kwp / 1000, o.npv / 1e6, "o", color="k", ms=6.5); tag(ax, "(b)")
ins = ax.inset_axes([0.10, 0.08, 0.36, 0.34]); zz = sw[sw.p_kwp <= 3000]
ins.plot(zz.p_kwp / 1000, zz.npv / 1e6, color=C["fpv"], lw=1.6); ins.plot(o.p_kwp / 1000, o.npv / 1e6, "o", color="k", ms=4); ins.axhline(0, color="k", lw=0.5)
ins.set_xlim(0, 3); ins.tick_params(labelsize=6.5); ins.set_title("NPV up to 3 MWp (M USD)", fontsize=7)
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
ax = axs[2]; ax.plot(sw.p_kwp / 1000, sw.sc_share * 100, color=C["roof"], lw=2); ax.set_xscale("log")
ax.set_xlabel("Installed PV capacity (MWp, log scale)"); ax.set_ylabel("Share of generation used on campus (%)", color=C["roof"]); ax.set_ylim(0, 105)
ax3 = ax.twinx(); ax3.spines["right"].set_visible(True); ax3.plot(sw.p_kwp / 1000, sw.load_cover * 100, color=C["gray"], lw=1.8, ls="--")
ax3.set_ylabel("Share of campus grid energy displaced (%)", color=C["gray"]); ax3.set_ylim(0, 105); tag(ax, "(c)")
plt.tight_layout(); plt.savefig("figures_up/fig4_capacity_value.png"); plt.close()

# ---------- Fig. 5 export remuneration and PV cost
OP = pd.read_csv("outputs/up_optima.csv"); SWU = pd.read_csv("outputs/up_sweeps.csv")
fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
ax = axs[0]
for cx, col, ls in ((0.8, C["ground"], "--"), (1.0, C["fpv"], "-"), (1.2, C["grid"], ":")):
    d = OP[OP.capex == cx]; ax.step(d.exp_thb, d.p_kwp / 1000, where="post", color=col, ls=ls, lw=2.2 if cx == 1.0 else 1.6, label={0.8: "PV capital cost −20 %", 1.0: "base PV capital cost", 1.2: "PV capital cost +20 %"}[cx])
ax.set_yscale("log"); ax.set_yticks([0.5, 1, 2, 3, 5, 10, 25]); ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter()); ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
for xv, lab in ((1.0, "institutional\npilot (≤200 kW)"), (2.2, "residential\nscheme (≤5 kW)")):
    ax.axvline(xv, color=C["gray"], ls=":", lw=1); ax.text(xv - 0.04, 21, lab, fontsize=7, color=C["gray"], va="top", ha="right")
ax.axhline(3, color=C["gray"], lw=0.6, ls="--")
ax.set_xlabel("Export remuneration (THB kWh$^{-1}$)"); ax.set_ylabel("NPV-maximizing PV capacity (MWp, log scale)"); ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.80))
sec = ax.secondary_xaxis("top", functions=(lambda v: v / FX, lambda v: v * FX)); sec.set_xlabel("USD kWh$^{-1}$", fontsize=8); sec.tick_params(labelsize=7)
ax.text(0.5, 1.20, "(a)", transform=ax.transAxes, ha="center", va="bottom", fontsize=11, fontweight="bold")
ax = axs[1]; d = OP[OP.capex == 1.0]
ax.plot(d.exp_thb, d.npv / 1e6, color=C["fpv"], lw=2, label="NPV at the optimum")
ax.plot(d.exp_thb, (d.pv_kwh - d.ex_kwh) / d.pv_kwh, color="none")
axb = ax.twinx(); axb.spines["right"].set_visible(True)
axb.plot(d.exp_thb, d.sc_kwh / (d.pv_kwh) * 100, color=C["roof"], lw=1.8, ls="--", label="generation used on campus")
axb.plot(d.exp_thb, d.ex_kwh / d.pv_kwh * 100, color=C["res"], lw=1.8, ls=":", label="generation exported")
axb.set_ylabel("Share of PV generation at the optimum (%)"); axb.set_ylim(0, 105)
ax.set_xlabel("Export remuneration (THB kWh$^{-1}$)"); ax.set_ylabel("25-year NPV at the optimum (M USD)", color=C["fpv"]); ax.set_ylim(0, 3.0)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axb.get_legend_handles_labels(); ax.legend(h1[:1] + h2, l1[:1] + l2, fontsize=7.5, frameon=True, framealpha=0.9, loc="center left", bbox_to_anchor=(0.0, 0.62)); tag(ax, "(b)")
ax = axs[2]
for xv, col in ((0.0, C["fpv"]), (1.0, C["ground"]), (2.2, C["roof"]), (2.6, C["grid"])):
    dd = SWU[(SWU.capex == 1.0) & ((SWU.exp_thb - xv).abs() < 1e-9) & (SWU.p_kwp > 0)]
    ax.plot(dd.p_kwp / 1000, dd.npv / 1e6, color=col, lw=1.8, label=f"{xv:.2f} THB kWh$^{{-1}}$")
ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("Installed PV capacity (MWp)"); ax.set_ylabel("25-year NPV, PV only (M USD)"); ax.set_xlim(0, 12); ax.set_ylim(-3, 1.2)
ax.legend(fontsize=7.5, frameon=False, loc="lower left", title="Export remuneration", title_fontsize=7.5); tag(ax, "(c)")
plt.tight_layout(); plt.savefig("figures_up/fig5_export_policy.png"); plt.close()

# ---------- Fig. 6 utility perspective
SCH = U["schemes"]; CU = pd.read_csv("outputs/up_scheme_curves.csv")
labs = list(SCH.keys()); short = ["Uncompensated\nexport", "Zero export,\ncurtailed", "Net billing\n1.00 (pilot)", "Net billing\n2.20 (CF)", "Net metering\n(CF)"]
cols = [C["gray"], "#3D3D3D", C["ground"], C["roof"], C["grid"]]; n_ = len(labs)
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw={"width_ratios": [1.0, 1.2, 1.05]})
ax = axs[0]; xb = np.arange(n_)
ax.bar(xb, [SCH[l]["p_kwp"] / 1000 for l in labs], color=cols, width=0.62)
for i, l in enumerate(labs): ax.text(i, SCH[l]["p_kwp"] / 1000 + 0.06, f"NPV {SCH[l]['npv'] / 1e6:.2f}", ha="center", fontsize=7.6)
ax.set_xticks(xb); ax.set_xticklabels(short, fontsize=7.4); ax.set_ylabel("Institution's NPV-maximizing PV capacity (MWp)"); ax.set_ylim(0, 4.2)
ax.text((n_ - 1) / 2, 3.95, "labels: institution's 25-year NPV (M USD); CF, counterfactual", ha="center", fontsize=7.0, color=C["gray"]); tag(ax, "(a)")
ax = axs[1]
for l, s_, c_, ls_ in zip(labs, short, cols, ["-", "--", "-", "-", "-"]):
    d = CU[(CU.scheme == l) & (CU.p_kwp <= 6000)]
    ax.plot(d.p_kwp / 1000, d["dM_w2.2"] / 1e3, color=c_, lw=1.8, ls=ls_, label=s_.replace("\n", " "))
    ax.fill_between(d.p_kwp / 1000, d["dM_w2.0"] / 1e3, d["dM_w2.7"] / 1e3, color=c_, alpha=0.08)
    p = SCH[l]["p_kwp"]; ax.plot(p / 1000, SCH[l]["dM"]["2.2"] / 1e3, "o", color=c_, ms=6.5, mec="k")
ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("Installed PV capacity (MWp)"); ax.set_ylabel("Change in utility net margin (k USD yr$^{-1}$)")
ax.set_xlim(0, 6); ax.legend(fontsize=7.3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, title="Line: avoided cost 2.20 THB kWh$^{-1}$; band: 2.00 to 2.70; dots: institution's optimum", title_fontsize=7.3); tag(ax, "(b)")
ax = axs[2]
scc = np.array([SCH[l]["comp_sc_2.2"] for l in labs]); exc = np.array([SCH[l]["comp_ex_2.2"] for l in labs])
ax.bar(xb, scc / 1e3, color=C["fpv"], width=0.62, label="Self-consumption: avoided purchases\nminus retail revenue lost")
ax.bar(xb, exc / 1e3, bottom=np.where(exc < 0, scc, 0) / 1e3, color=C["roof"], width=0.62, label="Delivered exports: avoided purchases\nminus export payments (0 if curtailed)")
net = scc + exc; lo = np.array([SCH[l]["dM"]["2.0"] for l in labs]); hi = np.array([SCH[l]["dM"]["2.7"] for l in labs])
ax.errorbar(xb, net / 1e3, yerr=[(net - lo) / 1e3, (hi - net) / 1e3], fmt="D", color="k", ms=5, capsize=4, label="Net change (bars: avoided cost 2.00 to 2.70)")
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(xb); ax.set_xticklabels(short, fontsize=7.4); ax.set_ylabel("Utility net margin at the optimum (k USD yr$^{-1}$)")
ax.legend(fontsize=6.9, frameon=False, loc="lower left"); ax.set_ylim(-270, 60); tag(ax, "(c)")
plt.tight_layout(); plt.savefig("figures_up/fig6_utility.png"); plt.close()

# ---------- Fig. 7 decision map and benchmark (PV only)
FW = np.load("outputs/up_framework.npz"); AN = pd.read_csv("outputs/up_rule_grid.csv")
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.25, 1]})
ax = axs[0]
for i, (lab, col) in enumerate(zip(["CRRU metered", "Flat (24 h)", "Office hours", "Evening peak"], [C["fpv"], C["gray"], C["roof"], C["grid"]])):
    rd, sg = FW[f"shape_{i}"]; m = rd <= 2.0; ax.plot(rd[m], sg[m], color=col, lw=2.2 if i == 0 else 1.5, ls="-" if i == 0 else "--", label=lab)
ax.set_xlabel("$R_d$ = annual PV generation / annual demand"); ax.set_ylabel("Marginal self-consumption share $\\sigma$"); ax.set_ylim(0, 1.03); ax.set_xlim(0, 2.0)
ax.legend(fontsize=8, frameon=False, title="Load shape (same annual energy)", title_fontsize=8); tag(ax, "(a)")
ax = axs[1]; rr = U["R_r"]; M = FW["map_m"]
ax.contourf(FW["map_rx"], FW["map_rd"], (M > 0).astype(float), levels=[-0.5, 0.5, 1.5], colors=["#F6E7CC", "#DCE9F2"], alpha=0.9)
sig = FW["sig"]; PG = np.arange(0, 40001, 250.0); YL = R["yield_land"]; LT = float(np.sum(np.load("outputs/profiles.npz")["load"]))
def rdstar(rx, r):
    if rx >= 1: return np.nan
    idx = np.flatnonzero(sig < (1 - rx) / (r - rx)); return PG[idx[0]] * YL / LT if len(idx) else np.nan
rxs = np.linspace(0, 0.995, 200)
for r, ls, lab in ((1.15, ":", "$R_r$ = 1.15"), (rr, "-", f"$R_r$ = {rr:.2f} (CRRU)"), (1.6, "--", "$R_r$ = 1.60"), (2.0, "-.", "$R_r$ = 2.00")):
    ax.step(rxs, [rdstar(v, r) for v in rxs], where="post", color="k", ls=ls, lw=1.6 if ls == "-" else 1.1, label=lab)
sim = AN[AN.Rx < 1.0]; ax.plot(sim.Rx, sim.rd_sim, "o", ms=4, color=C["grid"], label="Full simulation optimum")
ax.axvline(1.0, color=C["gray"], lw=1); ax.axhline(4.78, color=C["gray"], lw=0.8, ls=":")
ax.text(0.03, 4.87, "technical potential of the campus ($R_d$ = 4.78)", fontsize=7, color=C["gray"], va="bottom")
ax.text(0.98, 0.08, "undersized: marginal kWp adds value", fontsize=7, ha="right"); ax.text(0.45, 2.6, "oversized:\nmarginal kWp destroys value", fontsize=7.5, ha="center")
ax.text(1.2, 2.6, "export-driven\nexpansion", fontsize=7.5, ha="center")
ax.set_xlim(0, 1.4); ax.set_ylim(0, 5.3); ax.set_xlabel("$R_x$ = export remuneration / PV generation cost"); ax.set_ylabel("$R_d$ = annual PV generation / annual demand")
ax.legend(fontsize=7, frameon=True, framealpha=0.9, loc="upper left", bbox_to_anchor=(0.0, 0.92)); tag(ax, "(b)")
ax = axs[2]; B0 = U["bench"]["0.00"]; bl = list(B0.keys()); sh = ["Annual\nbalance", "Monthly\nbalance", "Average-day\nhourly", "Value-based\n(15-min)"]
xb = np.arange(4); ax.bar(xb, [B0[l]["p_kwp"] / 1000 for l in bl], color=[C["grid"], C["roof"], C["ground"], C["fpv"]])
for i, l in enumerate(bl): ax.text(i, B0[l]["p_kwp"] / 1000 + 0.08, f"{B0[l]['loss'] / 1e6:+.2f}", ha="center", fontsize=8)
ax.set_xticks(xb); ax.set_xticklabels(sh, fontsize=8); ax.set_ylabel("Recommended PV capacity (MWp)"); ax.set_ylim(0, 7.0)
ax.text(1.5, 6.6, "labels: NPV loss vs optimum (M USD), zero export", ha="center", fontsize=7.2, color=C["gray"]); tag(ax, "(c)")
plt.tight_layout(); plt.savefig("figures_up/fig7_decision_map.png"); plt.close()

# ---------- Fig. 8 uncertainty (PV only)
MCR = np.load("outputs/up_mc.npy"); WY = np.load("outputs/up_wy.npy"); WYW = np.load("outputs/up_wy_windows.npy")
RO = np.load("outputs/up_mc_reopt.npy"); MO = json.load(open("outputs/up_mc_reopt.json"))
fig, axs = plt.subplots(1, 4, figsize=(19, 4.8), gridspec_kw={"width_ratios": [1, 1.15, 0.9, 0.8]}); a0, a1, a2, a3 = axs
bins = np.linspace(-0.6, 1.6, 71)
a0.hist(MCR[:, 0] / 1e6, bins=bins, color=C["fpv"], alpha=0.65, label="design fixed at 1.25 MWp")
a0.hist(RO[:, 1] / 1e6, bins=bins, histtype="step", color=C["grid"], lw=1.5, label="capacity re-optimized per draw")
a0.axvline(0, color="k", lw=1)
a0.text(0.97, 0.80, f"fixed: P(NPV > 0) = {U['mc']['p_pos']:.1f} %\nre-optimized: P(NPV > 0) = {MO['x0']['p_npv_opt_pos']:.1f} %", transform=a0.transAxes, ha="right", va="top", fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.85))
a0.set_xlabel("25-year NPV (M USD), no export remuneration"); a0.set_ylabel("Frequency (10,000 simulations)"); a0.legend(fontsize=7.5, frameon=False, loc="upper right"); tag(a0, "(a)")
tor = sorted(U["tornado"].items(), key=lambda kv: kv[1]["swing"]); base = U["pv_opt"]["npv"] / 1e6
for i, (k, v) in enumerate(tor):
    lo, hi = sorted([v["low"] / 1e6, v["high"] / 1e6]); a1.barh(i, hi - lo, left=lo, color=C["fpv"], height=0.6)
a1.axvline(base, color="k", ls="--", lw=1.2); a1.axvline(0, color=C["gray"], lw=0.8)
a1.set_yticks(range(len(tor))); a1.set_yticklabels([k for k, _ in tor], fontsize=8); a1.set_xlabel("25-year NPV at 1.25 MWp (M USD)"); tag(a1, "(b)")
g = np.arange(0, 8001, 250.0)
for col, data, lab, off in ((C["fpv"], RO[:, 0], "no export remuneration", -60), (C["roof"], RO[:, 3], "2.20 THB kWh$^{-1}$ (CF)", 60)):
    sh = np.array([(data == p).mean() * 100 for p in g]); a2.bar(g / 1000 + off / 1000, sh, width=0.11, color=col, label=lab)
    a2.bar(9.0 + off / 1000, (data > 8000).mean() * 100, width=0.11, color=col)
a2.set_xticks([0, 2, 4, 6, 9]); a2.set_xticklabels(["0", "2", "4", "6", "> 8"])
_hi = RO[RO[:, 3] > 8000, 3] / 1000
if len(_hi): a2.text(9.0, (RO[:, 3] > 8000).mean() * 100 + 0.6, f"{_hi.min():.1f} to\n{_hi.max():.1f} MWp", ha="center", fontsize=7)
a2.set_xlim(-0.2, 9.6); a2.set_xlabel("NPV-maximizing capacity in each draw (MWp)"); a2.set_ylabel("Share of draws (%)"); a2.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.55, 1.0)); tag(a2, "(c)")
bp = a3.boxplot([WY[:, 1] / 1000, WY[:, 2] / 1000], positions=[0.85, 2.85], widths=0.45, patch_artist=True, whis=(10, 90), showfliers=False)
for p_, c_ in zip(bp["boxes"], [C["fpv"], C["roof"]]): p_.set_facecolor(c_); p_.set_alpha(0.6)
for xpos, col_ in ((1.35, 1), (3.35, 2)):
    a3.vlines(xpos, WYW[:, col_].min() / 1000, WYW[:, col_].max() / 1000, color="k", lw=2); a3.plot(xpos, np.median(WYW[:, col_]) / 1000, "ko", ms=5)
a3.plot([0.85, 2.85], [WY[0, 1] / 1000, WY[0, 2] / 1000], "k*", ms=9, label="actual 2024 yield")
a3.plot([], [], "k-", lw=2, label="range over observed\n12-month windows")
a3.set_xticks([1.1, 3.1]); a3.set_xticklabels(["no export\nremuneration", "2.20 THB kWh$^{-1}$"], fontsize=8); a3.set_ylabel("NPV-maximizing capacity (MWp)"); a3.set_ylim(0, 4.5); a3.set_xlim(0.3, 3.8)
a3.legend(fontsize=7.2, frameon=False, loc="upper left"); tag(a3, "(d)")
plt.tight_layout(); plt.savefig("figures_up/fig8_uncertainty.png"); plt.close()

# ---------- Fig. 3 (metered load), Fig. S2 (PV calibration and interval validation) and Fig. S3 (floating ceiling)
PS = json.load(open("prepared/prep_summary.json")); PR = np.load("outputs/profiles.npz")
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
GHI_NASA = np.array([4.72, 5.16, 5.68, 5.72, 5.21, 4.75, 4.41, 4.18, 4.32, 4.51, 4.63, 4.48])
# ---------- Fig. 2 measured campus load 2025
load = PR["load"]; cls = PR["cls"]; t = pd.to_datetime(PR["t"].astype("datetime64[m]"))
kw = load * 4
fig = plt.figure(figsize=(13, 4.4)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, 1])
a0 = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1]); a2 = fig.add_subplot(gs[2])
s = pd.DataFrame({"e": load, "c": cls}, index=t)
mo = s.groupby([s.index.month, "c"]).e.sum().unstack(fill_value=0) / 1000
a0.bar(MON, mo[0], color=C["grid"], label="On-peak"); a0.bar(MON, mo[1], bottom=mo[0], color=C["roof"], label="Off-peak")
a0.bar(MON, mo[2], bottom=mo[0] + mo[1], color="#C9C9C9", label="Weekend / holiday")
a0.set_ylabel("Grid energy (MWh)"); a0.tick_params(axis="x", labelsize=8, rotation=90)
b = a0.twinx(); b.spines["right"].set_visible(True); b.plot(MON, np.array(R["base_dmax"]), "k-o", ms=3.5, lw=1.4, label="On-peak maximum demand")
b.set_ylabel("Maximum on-peak demand (kW)"); b.set_ylim(0, 3200)
h1, l1 = a0.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels(); a0.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="upper left")
tag(a0, "(a)")
hm = kw.reshape(365, 96).T
im = a1.imshow(hm, aspect="auto", cmap="viridis", origin="lower", extent=[0, 365, 0, 24])
a1.set_xlabel("Day of 2025"); a1.set_ylabel("Hour of day"); a1.set_yticks([0, 6, 12, 18, 24])
cb = plt.colorbar(im, ax=a1, pad=0.01); cb.set_label("Campus demand (kW)"); tag(a1, "(b)")
wk = np.repeat((cls.reshape(365, 96) == 0).any(axis=1), 96)          # working day = any on-peak interval
pvN = PR["pv_land"] * U["P0"]
hrs = np.arange(96) / 4
for months, lab, col in (([1, 2, 12], "Cool season (Dec to Feb)", C["fpv"]), ([6, 7, 8, 9], "Rainy season (Jun to Sep)", C["grid"])):
    sel = np.isin(t.month, months)
    a2.plot(hrs, kw[sel & wk].reshape(-1, 96).mean(0), color=col, lw=2, label=f"Weekday load, {lab}")
    a2.plot(hrs, kw[sel & ~wk].reshape(-1, 96).mean(0), color=col, lw=1.2, ls="--", label=f"Weekend load, {lab}")
    a2.fill_between(hrs, 0, pvN[sel].reshape(-1, 96).mean(0), color=col, alpha=0.15)
a2.set_xlabel("Hour of day"); a2.set_ylabel("Power (kW)"); a2.set_xlim(0, 24); a2.set_ylim(0, 2700); a2.legend(fontsize=6.8, frameon=False, loc="upper left", ncol=1)
a2.text(12, 120, "mean output of the 1.25 MWp\noptimum (shaded)", ha="center", fontsize=7.5, color=C["gray"]); tag(a2, "(c)")
plt.tight_layout(); plt.savefig("figures_up/fig3_load.png"); plt.close()


# ---------- Fig. 3 PV-model calibration (2024 + Jan-Aug 2026) and validation (2025 held out)
cal = PS["calibration"]; rc = np.load("prepared/calibration_ratios_cal.npy"); rv = np.load("prepared/calibration_ratios_val.npy")
fig, AX = plt.subplots(2, 3, figsize=(15, 9.2)); axs = AX[0]
axs[0].hist(rc, bins=60, range=(0, 1.5), color=C["fpv"], alpha=0.75, density=True, label=f"Calibration 2024, 2026 (n = {len(rc):,})")
axs[0].hist(rv, bins=60, range=(0, 1.5), histtype="step", color=C["roof"], lw=1.5, density=True, label=f"Validation 2025 (n = {len(rv):,})")
axs[0].axvline(cal["eta_median"], color="k", ls="--", lw=1.3, label=f"Interval median {cal['eta_median']:.2f}")
axs[0].axvline(R["eta"], color=C["grid"], lw=1.6, label=f"Energy-matched {R['eta']:.2f}")
axs[0].axvline(0.903, color=C["gray"], ls=":", lw=1.3, label="Design loss chain 0.90")
axs[0].set_xlabel("Measured / modelled output (-)"); axs[0].set_ylabel("Probability density (G > 300 W m$^{-2}$)"); axs[0].legend(fontsize=7, frameon=False, loc="upper left"); axs[0].set_ylim(0, 3.2); tag(axs[0], "(a)")
mm = np.array(cal["zone1_2025_monthly_measured"]) / 1000; mp = np.array(cal["zone1_2025_monthly_model"]) / 1000
x = np.arange(12); axs[1].bar(x - 0.2, mm, 0.4, color=C["roof"], label="Measured 2025"); axs[1].bar(x + 0.2, mp, 0.4, color=C["fpv"], label="Model, calibrated on 2024 and 2026")
axs[1].set_xticks(x); axs[1].set_xticklabels(MON, rotation=90, fontsize=8); axs[1].set_ylabel("Monthly energy, 2,256 kWp array (MWh)"); axs[1].set_ylim(0, max(mm.max(), mp.max()) * 1.25); axs[1].legend(fontsize=7.5, frameon=False, loc="upper left"); tag(axs[1], "(b)")
g25 = np.array(PS["weather_2025"]["monthly_ghi_kwh_m2_day"])
axs[2].plot(MON, g25, "o-", color=C["roof"], lw=2, label=f"Measured 2025 ({R['ghi_2025']:.0f} kWh m$^{{-2}}$ yr$^{{-1}}$)")
axs[2].plot(MON, GHI_NASA, "s--", color=C["gray"], lw=1.6, label=f"NASA POWER 2003-2023 ({R['ghi_lt']:.0f})")
axs[2].set_ylabel("GHI (kWh m$^{-2}$ day$^{-1}$)"); axs[2].tick_params(axis="x", rotation=90, labelsize=8); axs[2].legend(fontsize=7.5, frameon=False); tag(axs[2], "(c)")
# ---------- Fig. S4 interval-level validation of the PV model (2025, 30-min, reference array)
IV = np.load("prepared/pv_interval_2025.npz"); IVJ = json.load(open("prepared/interval_validation.json"))["pv_interval"]
m30, p30, dm = IV["meas30"] * 2, IV["mod30"] * 2, IV["day30"]; t30 = pd.to_datetime(IV["t30"].astype("datetime64[m]")) - pd.Timedelta(minutes=15)
axs = AX[1]
hb_ = axs[0].hexbin(m30[dm], p30[dm], gridsize=45, cmap="Blues", mincnt=1, bins="log"); axs[0].plot([0, 1800], [0, 1800], "k--", lw=1)
axs[0].set_xlabel("Measured 30-min mean power (kW)"); axs[0].set_ylabel("Modelled 30-min mean power (kW)"); axs[0].set_xlim(0, 1800); axs[0].set_ylim(0, 1800)
cb = plt.colorbar(hb_, ax=axs[0], pad=0.01); cb.set_label("Intervals (log scale)")
axs[0].text(0.04, 0.95, f"daylight n = {IVJ['All daylight intervals']['n']:,}\nMBE {IVJ['All daylight intervals']['mbe_pct']:.1f} %\nnRMSE {IVJ['All daylight intervals']['nrmse_pct']:.1f} %\nR$^2$ = {IVJ['All daylight intervals']['r2']:.2f}", transform=axs[0].transAxes, va="top", fontsize=8); tag(axs[0], "(d)")
grp = ["Cool (Nov-Feb)", "Hot (Mar-May)", "Rainy (Jun-Oct)", "06:00-09:00", "09:00-12:00", "12:00-15:00", "15:00-19:00", "On-peak window (weekdays 09:00-22:00)", "Measured output above 50% of rating"]
lab_ = ["Cool", "Hot", "Rainy", "06-09 h", "09-12 h", "12-15 h", "15-19 h", "On-peak\nwindow", "Output\n> 50%"]
xg = np.arange(len(grp))
axs[1].bar(xg - 0.2, [IVJ[k]["mbe_pct"] for k in grp], 0.4, color=C["roof"], label="Mean bias error")
axs[1].bar(xg + 0.2, [IVJ[k]["nrmse_pct"] for k in grp], 0.4, color=C["fpv"], label="Normalized RMSE")
axs[1].axhline(0, color="k", lw=0.6); axs[1].set_xticks(xg); axs[1].set_xticklabels(lab_, fontsize=7.4, rotation=90); axs[1].set_ylabel("Error, % of mean measured power")
axs[1].legend(fontsize=7.5, frameon=False, loc="upper left"); tag(axs[1], "(e)")
hh = t30.hour + t30.minute / 60
for months, lab, col in (([11, 12, 1, 2], "Cool", C["fpv"]), ([3, 4, 5], "Hot", C["roof"]), ([6, 7, 8, 9, 10], "Rainy", C["grid"])):
    sel = np.isin(t30.month, months) & IV["valid30"]
    prof_m = pd.Series(m30[sel]).groupby(hh[sel]).mean(); prof_p = pd.Series(p30[sel]).groupby(hh[sel]).mean()
    axs[2].plot(prof_m.index, prof_m.values, color=col, lw=2, label=f"{lab}, measured"); axs[2].plot(prof_p.index, prof_p.values, color=col, lw=1.3, ls="--", label=f"{lab}, model")
axs[2].set_xlim(5, 20); axs[2].set_xlabel("Hour of day"); axs[2].set_ylabel("Mean power, 2,256 kWp array (kW)"); axs[2].legend(fontsize=7.2, frameon=False, ncol=2, loc="upper left"); axs[2].set_ylim(0, 1400); tag(axs[2], "(f)")

plt.tight_layout(); plt.savefig("figures_up/figS2_pv_calibration.png"); plt.close()


# ---------- Fig. 4 floating ceiling
cv = pd.read_csv("outputs/fpv_ceiling.csv")
fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.5, 4.3)); x = np.arange(len(cv))
a0.bar(x, cv.ponds, 0.55, color=C["fpv"], label="Small ponds"); a0.bar(x, cv.res, 0.55, bottom=cv.ponds, color=C["res"], label="Reservoir")
a0.set_xticks(x); a0.set_xticklabels([f"{int(c * 100)} %" for c in cv.ceiling]); a0.set_xlabel("Coverage ceiling (share of water surface)")
a0.set_ylabel("Floating PV capacity (MWp)"); a0.legend(frameon=False, fontsize=8.5); tag(a0, "(a)")
a1.plot(cv.ceiling * 100, cv.fpv_mwh / 1000, "o-", color=C["fpv"], lw=2); a1.set_xlabel("Coverage ceiling (%)"); a1.set_ylabel("Annual floating PV yield (GWh)")
a1.axvspan(0, 30, color="#D9EAF2", alpha=0.6); a1.axvspan(45, 60, color="#F2EAD9", alpha=0.6); ymax = cv.fpv_mwh.max() / 1000
a1.text(15, ymax * 0.92, "precautionary\nrange", ha="center", fontsize=8.5); a1.text(52.5, ymax * 0.92, "stratification\nrisk", ha="center", fontsize=8.5); tag(a1, "(b)")
plt.tight_layout(); plt.savefig("figures_up/figS3_fpv_ceiling.png"); plt.close()


# ---------- reused figures (drawn by script 01)
for a, b in (("fig2_maps.png", "fig2_maps.png"), ("figS1_landuse_shares.png", "figS1_landuse_shares.png")):
    shutil.copy(f"figures/{a}", f"figures_up/{b}")
print("figures written")
