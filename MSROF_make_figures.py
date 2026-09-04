"""
MSROF-CRRU Recalculation - Figure Generation
==============================================
Generates all manuscript figures from outputs/results.json and the
saved arrays produced by analysis.py. Uses matplotlib only (no
proprietary plotting software), per journal reproducibility guidance.

Figures are produced without titles or subtitles (captions are placed
in the manuscript text instead). Multi-panel figures are labelled
(a), (b), (c) ... centered beneath each panel. All figures are saved
at 900 dpi.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 900, "savefig.bbox": "tight",
})

with open("outputs/results.json") as f:
    R = json.load(f)

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
COLORS = dict(roof="#D98E04", ground="#7A9E3B", floating="#2E86AB", biogas="#8E5572",
              battery="#556270", grid="#B0413E")


def panel_label(ax, label, y_offset=1.06):
    """Place a centered (a)/(b)/(c) panel label above a subplot axis."""
    ax.text(0.5, y_offset, label, transform=ax.transAxes,
             ha="center", va="bottom", fontsize=11, fontweight="bold")


# ------------------------------------------------------------------
# Figure 1. Satellite/GIS-derived land-use composition (Table 3)
# ------------------------------------------------------------------
land_use = {"Academic\nbuildings": 213_580.61, "Open\nfields": 48_682.98,
            "Ponds /\nreservoirs": 364_785.01, "Roads /\nparking": 185_392.01,
            "Vegetation /\nconservation": 1_204_935.32}
fig, ax = plt.subplots(figsize=(6.2, 5.2))
colors = ["#D98E04", "#7A9E3B", "#2E86AB", "#9C9C9C", "#3B6E3B"]
wedges, texts, autotexts = ax.pie(land_use.values(), labels=land_use.keys(),
    autopct=lambda p: f"{p:.1f}%\n({p*sum(land_use.values())/100/1e4:.1f} ha)",
    colors=colors, startangle=90, pctdistance=0.72,
    wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.5))
for t in autotexts: t.set_fontsize(8.3)
plt.savefig("figures/fig1_landuse.png"); plt.close()

# ------------------------------------------------------------------
# Figure 2. Climate & demand profile (Table 1 & 2)
# ------------------------------------------------------------------
GHI  = np.array([4.72,5.16,5.68,5.72,5.21,4.75,4.41,4.18,4.32,4.51,4.63,4.48])
Tamb = np.array([24.8,26.7,29.4,32.4,31.2,29.8,28.6,28.1,28.3,27.9,26.4,24.1])
demand = pd.read_csv("outputs/table2_demand.csv", index_col=0)["Demand_MWh"].values

fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
ax1.bar(months, demand, color="#B0413E", alpha=0.75, label="Campus electricity demand (MWh)")
ax1.set_ylabel("Electricity demand (MWh)", color="#B0413E")
ax1.tick_params(axis="y", labelcolor="#B0413E")
ax2 = ax1.twinx()
ax2.plot(months, GHI, "o-", color="#D98E04", lw=2, label="GHI (kWh/m²/day)")
ax2.plot(months, Tamb, "s--", color="#2E86AB", lw=1.6, label="T_amb (°C)")
ax2.set_ylabel("GHI (kWh/m²/day)  /  T_amb (°C)")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc="upper center", bbox_to_anchor=(0.5,-0.15), ncol=3, fontsize=8.5, frameon=False)
plt.tight_layout(); plt.savefig("figures/fig2_climate_demand.png"); plt.close()

# ------------------------------------------------------------------
# Figure 3. Recalculated monthly PV generation by surface type
# ------------------------------------------------------------------
roof = np.load("outputs/monthly_roof.npy")
ground = np.load("outputs/monthly_ground.npy")
floating = np.load("outputs/monthly_float.npy")

fig, ax = plt.subplots(figsize=(7.5, 4.3))
ax.bar(months, roof, label=f"Rooftop ({R['capacities_kWp']['rooftop']/1000:.2f} MWp)", color=COLORS["roof"])
ax.bar(months, ground, bottom=roof, label=f"Ground-mounted ({R['capacities_kWp']['ground']/1000:.2f} MWp)", color=COLORS["ground"])
ax.bar(months, floating, bottom=roof+ground, label=f"Floating PV ({R['capacities_kWp']['floating']/1000:.2f} MWp)", color=COLORS["floating"])
ax.plot(months, demand, "k--o", lw=1.6, ms=4, label="Campus demand")
ax.set_ylabel("Monthly energy (MWh)")
ax.legend(fontsize=8.5, loc="upper left", ncol=2)
plt.tight_layout(); plt.savefig("figures/fig3_pv_yield_by_surface.png"); plt.close()

# ------------------------------------------------------------------
# Figure 4. Floating-PV coverage optimization (Eq. 3-4)
# ------------------------------------------------------------------
C_range = np.linspace(0.001, 0.6, 200)
A_pond = 364_785.01
floating_density = 140.0
PR, L_total, gamma, dT_FPV, T_STC = 0.80, 0.14, -0.0041, 8.0, 25.0
NOCT = 45.0
def E_of_C(C):
    cap_kWp = C*A_pond*floating_density/1000
    T_cell = Tamb + (NOCT-20.0) - dT_FPV
    daily = (GHI/1.0)*(1+gamma*(T_cell-T_STC))*PR*(1-L_total)
    days = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    return cap_kWp*(daily*days).sum()/1000  # MWh
def dDO_of_C(C): return (7.5-4.0)*(1-np.exp(-4.5*C))
E_vals = np.array([E_of_C(c) for c in C_range])
dDO_vals = dDO_of_C(C_range)
E_norm = E_vals/E_of_C(0.6); dDO_norm = dDO_vals/dDO_of_C(0.6)
F = E_norm - 0.5*dDO_norm
C_star = R["C_star"]

fig, ax1 = plt.subplots(figsize=(7, 4.3))
ax1.plot(C_range*100, E_norm, color=COLORS["floating"], lw=2, label="Normalized FPV yield")
ax1.plot(C_range*100, dDO_norm, color="#B0413E", lw=2, label="Normalized DO depletion")
ax1.plot(C_range*100, F, "k--", lw=2, label="Composite objective F(C)")
ax1.axvline(C_star*100, color="grey", ls=":", lw=1.5)
ax1.annotate(f"C* = {C_star*100:.1f}%", (C_star*100, F.max()), xytext=(C_star*100+3, F.max()-0.15),
             fontsize=9.5, arrowprops=dict(arrowstyle="->", lw=1))
ax1.set_xlabel("Pond coverage fraction, C (%)")
ax1.set_ylabel("Normalized value (dimensionless)")
ax1.legend(fontsize=8.5)
plt.tight_layout(); plt.savefig("figures/fig4_fpv_optimization.png"); plt.close()

# ------------------------------------------------------------------
# Figure 5. Biogas mass/energy balance
# ------------------------------------------------------------------
t6 = R["table6"]
fig, ax = plt.subplots(figsize=(6.8, 4.2))
cats = ["Organic waste\ninput\n(t/yr)", "Methane\nproduced\n(1000 m³/yr)",
        "Electricity\noutput\n(MWh/yr)", "Thermal energy\nrecovered\n(MWh/yr)"]
vals = [3.12*365, t6["Annual methane production (m3/yr)"]/1000,
        t6["Annual electricity output (kWh/yr)"]/1000, t6["Annual thermal energy recovered (kWh/yr)"]/1000]
bars = ax.bar(cats, vals, color=[COLORS["biogas"], "#B08968", "#D98E04", "#B0413E"])
for b,v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout(); plt.savefig("figures/fig5_biogas_balance.png"); plt.close()

# ------------------------------------------------------------------
# Figure 6. Techno-economic comparison across scenarios (Table 7)
# ------------------------------------------------------------------
t7 = pd.read_csv("outputs/table7.csv", index_col=0)
scenarios = t7.columns.tolist()
lcoe_vals = t7.loc["LCOE (USD/kWh)"].values.astype(float)
capex_vals = t7.loc["CAPEX (M USD)"].values.astype(float)
npv_vals = t7.loc["NPV at 25yr (M USD)"].values.astype(float)

fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.8))
grid_tariff = 3.94/35.5  # USD/kWh (3.94 THB/kWh), ERC Ft-adjusted blended retail tariff (see analysis.py)
bar_colors = ["#2E86AB", "#7A9E3B", "#D98E04", "#B0413E"]
axes[0].bar(scenarios, lcoe_vals, color=bar_colors)
axes[0].axhline(grid_tariff, color="k", ls="--", lw=1.3, label=f"Grid tariff ({grid_tariff:.4f} USD/kWh)")
axes[0].set_ylabel("LCOE (USD/kWh)")
axes[0].legend(fontsize=7.5); axes[0].tick_params(axis='x', rotation=30, labelsize=7.5)
panel_label(axes[0], "(a)")
axes[1].bar(scenarios, capex_vals, color=bar_colors)
axes[1].set_ylabel("CAPEX (M USD)")
axes[1].tick_params(axis='x', rotation=30, labelsize=7.5)
panel_label(axes[1], "(b)")
axes[2].bar(scenarios, npv_vals, color=[c if v>=0 else "#888888" for c,v in zip(bar_colors,npv_vals)])
axes[2].axhline(0, color="k", lw=0.8)
axes[2].set_ylabel("25-yr NPV (M USD)")
axes[2].tick_params(axis='x', rotation=30, labelsize=7.5)
panel_label(axes[2], "(c)")
plt.tight_layout(); plt.subplots_adjust(top=0.90); plt.savefig("figures/fig6_technoeconomic.png"); plt.close()

# ------------------------------------------------------------------
# Figure 7. Battery lifecycle comparison (Table 8, Eq. 7 & 10)
# ------------------------------------------------------------------
t8 = R["table8"]
fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
techs = list(t8.keys())
axes[0].bar(techs, [t8[k]["capacity_kWh"] for k in techs], color=["#2E86AB","#8E5572"])
axes[0].set_ylabel("Battery capacity (kWh)")
panel_label(axes[0], "(a)")
axes[1].bar(techs, [t8[k]["cost_per_kWh_USD"] for k in techs], color=["#2E86AB","#8E5572"])
axes[1].set_ylabel("Levelized storage cost (USD/kWh delivered)")
panel_label(axes[1], "(b)")
plt.tight_layout(); plt.savefig("figures/fig7_battery.png"); plt.close()

# ------------------------------------------------------------------
# Figure 8. Novel metrics bar chart (Eq. 11-14)
# ------------------------------------------------------------------
m = R["metrics"]
labels = ["SEI\n(W/m²)", "CNED\n(×10⁻⁵)", "WENI\n(MWh+m³/ha)", "CAI\n(tCO₂/ha)"]
raw = [m["SEI"], m["CNED"]*1e5, m["WENI"], m["CAI"]]
fig, ax = plt.subplots(figsize=(6.5,4.2))
bars = ax.bar(labels, raw, color=["#2E86AB","#7A9E3B","#D98E04","#B0413E"])
for b,v in zip(bars, raw):
    ax.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}" if v<100 else f"{v:.0f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout(); plt.savefig("figures/fig8_metrics.png"); plt.close()

# ------------------------------------------------------------------
# Figure 9. Monte Carlo LCOE distribution + tornado sensitivity
# ------------------------------------------------------------------
lcoe_mc = np.load("outputs/lcoe_mc.npy")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
axes[0].hist(lcoe_mc, bins=60, color="#2E86AB", alpha=0.85)
axes[0].axvline(grid_tariff, color="k", ls="--", lw=1.4, label="Grid tariff")
axes[0].axvline(np.median(lcoe_mc), color="#B0413E", ls="-", lw=1.6, label="Median LCOE")
axes[0].set_xlabel("LCOE (USD/kWh)"); axes[0].set_ylabel("Frequency (of 10,000 draws)")
axes[0].legend(fontsize=8.5)
panel_label(axes[0], "(a)")

tor = R["tornado"]
labels = list(tor.keys())
lows = [tor[k]["low"] for k in labels]; highs = [tor[k]["high"] for k in labels]
base = tor[labels[0]]["base"]
order = np.argsort([abs(h-l) for l,h in zip(lows,highs)])
labels = [labels[i] for i in order]; lows=[lows[i] for i in order]; highs=[highs[i] for i in order]
y = np.arange(len(labels))
for i,(lo,hi) in enumerate(zip(lows,highs)):
    axes[1].plot([min(lo,hi), max(lo,hi)], [i,i], lw=8, color="#2E86AB", solid_capstyle="butt")
axes[1].axvline(base, color="k", ls="--", lw=1.2, label="Base case")
axes[1].set_yticks(y); axes[1].set_yticklabels(labels, fontsize=8.5)
axes[1].set_xlabel("LCOE (USD/kWh)")
axes[1].legend(fontsize=8)
panel_label(axes[1], "(b)")
plt.tight_layout(); plt.savefig("figures/fig9_montecarlo.png"); plt.close()

# ------------------------------------------------------------------
# Figure 10. Environmental & resource co-benefits (Table 10)
# ------------------------------------------------------------------
t10 = R["table10"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
ghg_cats = ["PV grid\ndisplacement", "Biogas grid\ndisplacement", "Landfill CH₄\navoidance"]
ghg_vals = [t10["PV grid displacement (tCO2/yr)"], t10["Biogas grid displacement (tCO2/yr)"],
            t10["Landfill CH4 avoidance (tCO2-eq/yr)"]]
axes[0].bar(ghg_cats, ghg_vals, color=["#D98E04","#8E5572","#7A9E3B"])
axes[0].set_ylabel("tCO₂-eq / yr")
panel_label(axes[0], "(a)")
water_cats = ["FPV evaporation\nsuppression", "Digestate irrigation\nsubstitution"]
water_vals = [t10["Water savings - FPV evaporation (m3/yr)"], t10["Water savings - digestate irrigation (m3/yr)"]]
axes[1].bar(water_cats, water_vals, color=["#2E86AB","#556270"])
axes[1].set_ylabel("m³ / yr")
panel_label(axes[1], "(b)")
plt.tight_layout(); plt.savefig("figures/fig10_cobenefits.png"); plt.close()

# ------------------------------------------------------------------
# Figure 11. Regional transferability across SE Asian cities (Eq. 15)
# ------------------------------------------------------------------
cities = pd.read_csv("outputs/table11.csv")
cities = cities.sort_values("Cost Saving (%)", ascending=True)
fig, ax = plt.subplots(figsize=(7.5, 5.2))
colors_c = ["#7A9E3B" if c else "#B0413E" for c in cities["Competitive"]]
ax.barh(cities["City"], cities["Cost Saving (%)"], color=colors_c)
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("Estimated cost saving vs local grid tariff (%)")
plt.tight_layout(); plt.savefig("figures/fig11_transferability.png"); plt.close()

print("All 11 figures generated in figures/ at 900 dpi, no titles, centered panel labels")
import os
for f in sorted(os.listdir("figures")):
    print(" -", f)
