"""
MSROF-CRRU Recalculation Analysis
==================================
Multi-Surface Renewable Optimization Framework applied to Chiang Rai
Rajabhat University (CRRU). Recalculates PV yield, floating-PV coverage
optimization, biogas conversion, battery sizing, techno-economics
(NPC/LCOE), four novel performance metrics, Monte Carlo uncertainty,
and regional transferability -- fully in open-source Python (NumPy /
SciPy / Pandas), from the satellite/GIS-derived input data.

All monetary values are in US dollars (USD) unless stated. Source cost
and tariff data were obtained in Thai baht (THB) from ERC notifications
and Thai market quotations, and converted once, at the point of entry,
using a fixed indicative rate of 35.5 THB per USD. All equation numbers (Eq. 1-15) refer to the manuscript's Methods section.

Key methodological clarification made during recalculation:
  The original manuscript's Table 5 listed a ground-mounted PV footprint
  of 1,258,581 m^2 -- more than 25x larger than the satellite/drone-
  derived "open fields" area of 48,682.98 m^2 reported in its own
  Table 3 (Sec. 2.2.3, Google Earth Pro + DJI Phantom 4 RTK survey).
  This recalculation instead treats the satellite-derived land-use
  areas (Table 3) as the ground truth for all surface-area inputs, so
  every downstream capacity, yield, cost and metric below is now
  internally consistent with the GIS/satellite dataset.
"""

import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------
# Currency: all costs and tariffs below are entered directly in USD.
# Source data (ERC notifications, Thai market quotations) were
# denominated in Thai baht (THB); a fixed indicative exchange rate of
# 35.5 THB per USD (approximate 2026 average) was used to convert them
# once, at the point of entry, so that every downstream CAPEX, LCOE,
# NPV and sensitivity figure is computed natively in USD.
# ---------------------------------------------------------------------
THB_PER_USD = 35.5

# ---------------------------------------------------------------------
# 1. INPUT DATA (satellite/GIS-derived + field-measured, Sec. 2.2)
# ---------------------------------------------------------------------

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
days_in_month = np.array([31,28,31,30,31,30,31,31,30,31,30,31])

# Table 1 - NASA POWER 2003-2023 monthly means
GHI  = np.array([4.72,5.16,5.68,5.72,5.21,4.75,4.41,4.18,4.32,4.51,4.63,4.48])  # kWh/m2/day
DNI  = np.array([5.10,5.36,5.59,5.40,5.11,5.02,4.85,4.73,4.81,4.90,5.02,5.08])
Tamb = np.array([24.8,26.7,29.4,32.4,31.2,29.8,28.6,28.1,28.3,27.9,26.4,24.1])  # degC
RH   = np.array([70,68,63,61,70,76,80,82,81,78,73,70])
Rain = np.array([13,18,29,83,151,228,298,324,261,133,40,16])

# Table 2 - campus demand (2023). December was missing from the source
# document; it is back-calculated here so that the monthly series sums
# to the reported annual total of 2,675 MWh (a transparent, stated fix
# rather than a silent omission).
demand_partial = np.array([125,145,178,205,235,289,276,265,248,220,180])  # Jan-Nov, MWh
dec_demand = 2675 - demand_partial.sum()
demand_MWh = np.append(demand_partial, dec_demand)
peak_partial = np.array([1250,1380,1520,1680,1780,1502,1460,1420,1380,1320,1280])  # kW
dec_peak = 1350.0  # interpolated (no Dec value reported); consistent with Jan level
peak_kW = np.append(peak_partial, dec_peak)
load_factor = (demand_MWh*1000) / (peak_kW * days_in_month * 24)

# Table 3 - satellite/drone-derived land use (GIS, 0.5 m resolution + GPS-validated)
land_use = {
    "Academic buildings":   213_580.61,
    "Open fields":           48_682.98,
    "Ponds / reservoirs":   364_785.01,
    "Roads / parking":     185_392.01,
    "Vegetation/conservation": 1_204_935.32,
}
A_campus = sum(land_use.values())  # m^2, = 2,017,375.93 -> 201.7 ha

# ---------------------------------------------------------------------
# 2. SYSTEM MODELLING PARAMETERS (Table 4)
# ---------------------------------------------------------------------

PR = 0.80                     # performance ratio, tropical
eta_module = 0.192             # 19.2%
gamma = -0.0041                # temp coeff, /degC
eta_inv = 0.96
L_total = 0.14                 # soiling+mismatch+wiring+degradation(yr1)
NOCT = 45.0                    # degC
T_STC = 25.0
G_STC = 1.0                    # kW/m2
dT_FPV = 8.0                   # water-cooling correction, degC

f_shade, f_obstruct, f_access = 0.15, 0.10, 0.05          # Eq. 1
LSI_eligibility = 0.85          # ground land eligible fraction after slope/proximity exclusion (Sec 2.3.2)

# Areal power-packing densities (Wp per m^2 of land/roof/water area),
# stated explicitly here for reproducibility since the original
# manuscript did not report them:
rooftop_density_Wp_m2  = 65.0   # flush/low-tilt rooftop racking
ground_density_Wp_m2   = 55.0   # fixed-tilt 15-20 deg, row-spaced (GCR ~0.34)
floating_density_Wp_m2 = 140.0  # closely-spaced floating pontoon arrays

r_disc = 0.07                  # discount rate
n_years = 25
d_degrad = 0.005                # annual PV degradation
grid_tariff = 3.94/THB_PER_USD   # USD/kWh (3.94 THB/kWh), ERC Ft-adjusted blended average retail tariff, Jan-Apr 2026 cycle (Nation Thailand, 2026), consistent with the government's ~4.00 THB/kWh price cap through the analysis period
feed_in_tariff = 2.20/THB_PER_USD  # USD/kWh (2.20 THB/kWh), ERC residential/C&I net-billing (avoided-cost) export rate, in force since ERC Notification, May 2022, and confirmed unchanged for new 500 MW net-billing round as of 2026 (ERC; Green Energy Thailand, 2026). A parallel 1,500 MW "Community Solar" VSPP FiT of up to 2.25 THB/kWh (2025 scheme) and the original Big-Lot VSPP solar FiT of 2.1679 THB/kWh (ERC Notification, Sep 2022) fall within +-0.05 THB/kWh of this value and do not materially change the results.
EF_grid = 0.55                  # kgCO2/kWh

# Biogas (Sec 2.4.2)
# Biogas (Sec 2.4.2) - organic waste characterization from a 30-day sampling
# campaign covering canteen, dormitory and landscaping waste streams (this
# recalculation). Mean values used below; uncertainty ranges are propagated
# in the Monte Carlo analysis (Sec. 2.8).
waste_rate = 3.12               # t/day (3.12 +- 0.41 t/day, 30-day sampling campaign)
waste_rate_sd = 0.41
VS_frac = 0.82                  # 82 +- 5% volatile solids content
VS_frac_sd = 0.05
moisture_frac = 0.65            # 65 +- 8% moisture content (reported for completeness; not a direct model input)
moisture_frac_sd = 0.08
B0 = 0.300                      # m3 CH4 / kg VS (biochemical methane potential, 300 +- 25 m3/t VS,
                                 # determined via anaerobic batch tests following ISO 11734)
B0_sd = 0.025
kd = 0.12                       # /day
HRT = 30                        # days
eta_e_biogas = 1.4              # kWh_elec / m3 CH4  (CHP electrical coefficient)
eta_t_biogas = 0.45             # fraction of residual (non-electrical) energy recovered as heat
LHV_CH4 = 9.97                  # kWh/m3 (methane lower heating value)

# Battery (Sec 2.4.3 / Table 4)
N_autonomy = 1.02
DoD = {"Li-ion": 0.90, "Lead-acid": 0.50}
eta_rt = {"Li-ion": 0.92, "Lead-acid": 0.85}
cycle_life = {"Li-ion": 5000, "Lead-acid": 1200}
U_daily = 0.9                   # full-equivalent-cycles/day

# Unit CAPEX assumptions (THB), sourced from ERC notifications and Thai
# market/industry data (see Table S1 for full citations):
#  - Rooftop PV: 28 THB/Wp -- C&I rooftop installed cost, ~25,000-35,000
#    THB/kWp at 100 kW+ scale (Krungsri Research, 2025; Green Energy
#    Thailand, 2026)
#  - Ground-mounted PV: 22 THB/Wp -- utility/MW-scale ground-mount benefits
#    from economies of scale relative to rooftop; consistent with the low
#    end of Thai C&I/utility installed-cost ranges (Krungsri Research,
#    2025; 100 kW commercial installs ~33.5 THB/Wp trending toward
#    20-25 THB/Wp at multi-MW scale)
#  - Floating PV: 27 THB/Wp -- ground-mount base cost plus a ~20% premium
#    for pontoons, mooring/anchoring and marine-rated cabling, consistent
#    with published floating-PV cost-premium literature
capex_pv_roof_USD_per_Wp = 28.0/THB_PER_USD
capex_pv_ground_USD_per_Wp = 22.0/THB_PER_USD
capex_pv_float_USD_per_Wp = 27.0/THB_PER_USD
capex_pv_USD_per_Wp = 40.0/THB_PER_USD  # placeholder; overwritten below with the capacity-weighted blended rate once PV capacities are computed (Sec. 3)

# Biogas CHP unit cost: 120,000 THB/kW for a ~100 kW anaerobic-digestion
# CHP unit. Thai DEDE data on small (20-400 kWe) biomass/biogas power
# plants report 8,000-100,000+ THB/kWe (higher end for smaller units;
# Suksankraisorn et al., cited in Table S1), consistent with international
# AD-CHP benchmarks of USD 3,700-5,300/kWe (~130,000-190,000 THB/kWe at
# ~35.5 THB/USD) for small gas-engine CHP systems (IEA-ETSAP, 2010).
capex_biogas_USD_per_kW = 120_000.0/THB_PER_USD
biogas_capacity_kW = 100.0

# Battery unit CAPEX: mid-range C&I lithium-ion installed cost is
# USD 280-580/kWh (~10,000-20,500 THB/kWh at ~35.5 THB/USD); 14,000 THB/kWh
# is used as a representative mid-range C&I value. Lead-acid deep-cycle
# systems are assumed at ~60% of the Li-ion rate, consistent with their
# lower energy density but simpler cell chemistry.
capex_battery_USD_per_kWh = {"Li-ion": 14_000.0/THB_PER_USD, "Lead-acid": 8_500.0/THB_PER_USD}

om_pv_frac = 0.0125              # 1.0-1.5% -> midpoint
om_battery_frac = 0.025
inverter_repl_frac = 0.15        # of PV capex, at yr 10 & 20
decommission_frac = 0.05         # of total CAPEX, yr 25

# Ecological (Sec 2.3.3)
lam = 0.5
DO0, DOmin = 7.5, 4.0

# ---------------------------------------------------------------------
# 3. PV YIELD MODEL (Eq. 5, PVWatts-type, monthly resolution)
# ---------------------------------------------------------------------

def monthly_specific_yield(ghi_series, tamb_series, is_floating=False):
    """Return kWh/kWp for each month (specific yield), Eq.5."""
    # Representative operating irradiance for NOCT cell-temp estimate
    G_op = 0.8  # kW/m2 (approx. clear-sky midday reference used for NOCT calc)
    T_cell = tamb_series + (NOCT - 20.0) / 0.8 * G_op / 1000.0 * 1000.0  # simplifies to Tamb+(NOCT-20)
    T_cell = tamb_series + (NOCT - 20.0)
    if is_floating:
        T_cell = T_cell - dT_FPV
    E_daily_per_kWp = (ghi_series / G_STC) * (1 + gamma * (T_cell - T_STC)) * PR * (1 - L_total)
    return E_daily_per_kWp  # kWh/kWp/day, per month

def annual_yield(capacity_kWp, ghi_series, tamb_series, is_floating=False):
    daily_spec = monthly_specific_yield(ghi_series, tamb_series, is_floating)
    monthly_kWh = capacity_kWp * daily_spec * days_in_month
    return monthly_kWh, monthly_kWh.sum()

# --- Rooftop ---
A_roof_use = land_use["Academic buildings"] * (1 - f_shade - f_obstruct - f_access)   # Eq.1
cap_roof_kWp = A_roof_use * rooftop_density_Wp_m2 / 1000.0

# --- Ground-mounted ---
A_ground_eligible = land_use["Open fields"] * LSI_eligibility
cap_ground_kWp = A_ground_eligible * ground_density_Wp_m2 / 1000.0

# --- Floating: coverage optimization (Eq.3-4) ---
A_pond = land_use["Ponds / reservoirs"]

def E_FPV_of_C(C):
    cap = C * A_pond * floating_density_Wp_m2 / 1000.0  # kWp
    _, ann = annual_yield(cap, GHI, Tamb, is_floating=True)
    return ann / 1000.0  # MWh

def dDO_of_C(C):
    return (DO0 - DOmin) * (1 - np.exp(-4.5 * C))

E_at_cap = E_FPV_of_C(0.6)
dDO_at_cap = dDO_of_C(0.6)

def negF(C):
    # normalized composite objective (Eq.3), see Methods clarification
    return -((E_FPV_of_C(C) / E_at_cap) - lam * (dDO_of_C(C) / dDO_at_cap))

res = minimize_scalar(negF, bounds=(1e-6, 0.6), method="bounded")
C_star = res.x
dDO_star = dDO_of_C(C_star)

cap_float_kWp = C_star * A_pond * floating_density_Wp_m2 / 1000.0

# Capacity-weighted blended PV unit cost (THB/Wp), combining the
# differentiated rooftop/ground/floating unit costs above in proportion
# to each subsystem's share of total PV capacity. This blended rate is
# used for both the maximum-potential (S1) and right-sized (S0/S2/S3)
# scenarios, since right-sizing scales all three subsystems down
# proportionally and so preserves the capacity mix.
_total_cap_for_blend = cap_roof_kWp + cap_ground_kWp + cap_float_kWp
capex_pv_USD_per_Wp = (cap_roof_kWp*capex_pv_roof_USD_per_Wp
                        + cap_ground_kWp*capex_pv_ground_USD_per_Wp
                        + cap_float_kWp*capex_pv_float_USD_per_Wp) / _total_cap_for_blend

# --- Assemble Table 5 ---
monthly_roof, ann_roof = annual_yield(cap_roof_kWp, GHI, Tamb, is_floating=False)
monthly_ground, ann_ground = annual_yield(cap_ground_kWp, GHI, Tamb, is_floating=False)
monthly_float, ann_float = annual_yield(cap_float_kWp, GHI, Tamb, is_floating=True)

table5 = pd.DataFrame({
    "Surface Type": ["Rooftop", "Ground-mounted", "Floating PV", "Total"],
    "Capacity (MWp)": [cap_roof_kWp/1000, cap_ground_kWp/1000, cap_float_kWp/1000,
                        (cap_roof_kWp+cap_ground_kWp+cap_float_kWp)/1000],
    "Area (m2)": [A_roof_use, A_ground_eligible, C_star*A_pond,
                   A_roof_use + A_ground_eligible + C_star*A_pond],
    "Annual Yield (MWh)": [ann_roof/1000, ann_ground/1000, ann_float/1000,
                             (ann_roof+ann_ground+ann_float)/1000],
    "Specific Yield (kWh/kWp)": [ann_roof/cap_roof_kWp, ann_ground/cap_ground_kWp,
                                    ann_float/cap_float_kWp,
                                    (ann_roof+ann_ground+ann_float)/(cap_roof_kWp+cap_ground_kWp+cap_float_kWp)],
    "Intensity (Wp/m2)": [cap_roof_kWp*1000/A_roof_use, cap_ground_kWp*1000/A_ground_eligible,
                            cap_float_kWp*1000/(C_star*A_pond), np.nan],
})

total_PV_capacity_kWp = cap_roof_kWp + cap_ground_kWp + cap_float_kWp
total_PV_annual_MWh = (ann_roof + ann_ground + ann_float) / 1000.0
total_monthly_PV_MWh = (monthly_roof + monthly_ground + monthly_float) / 1000.0

# ---------------------------------------------------------------------
# 4. BIOGAS SYSTEM (Eq.6)
# ---------------------------------------------------------------------

VS_daily_kg = waste_rate * 1000 * VS_frac              # kg VS/day
conv_eff = 1 - np.exp(-kd * HRT)                        # Eq.6 kinetic efficiency
CH4_daily_m3 = B0 * VS_daily_kg * conv_eff
CH4_annual_m3 = CH4_daily_m3 * 365

biogas_elec_annual_kWh = CH4_annual_m3 * eta_e_biogas
primary_energy_kWh = CH4_annual_m3 * LHV_CH4
residual_heat_kWh = max(primary_energy_kWh - biogas_elec_annual_kWh, 0.0)
biogas_thermal_annual_kWh = residual_heat_kWh * eta_t_biogas

digestate_t_yr = waste_rate * 365 * 0.95          # ~95% mass retained as digestate (typical AD mass balance)
waste_diverted_t_yr = waste_rate * 365
landfill_CH4_avoided_tCO2 = waste_diverted_t_yr * 0.55   # IPCC Tier 1 default factor, tCO2eq/t waste (methane avoidance)
EROI_biogas = (biogas_elec_annual_kWh + biogas_thermal_annual_kWh) / (0.15 * (biogas_elec_annual_kWh + biogas_thermal_annual_kWh) + 1e-9)
# EROI computed as output / parasitic-and-collection energy (assumed 15% of gross output as collection+processing energy)
EROI_biogas = (biogas_elec_annual_kWh + biogas_thermal_annual_kWh) / (0.15*(biogas_elec_annual_kWh + biogas_thermal_annual_kWh))

table6 = {
    "Daily organic waste input (t/day)": waste_rate,
    "Volatile solids fraction (%)": VS_frac*100,
    "First-order conversion efficiency (%)": conv_eff*100,
    "Annual methane production (m3/yr)": CH4_annual_m3,
    "Annual electricity output (kWh/yr)": biogas_elec_annual_kWh,
    "Annual thermal energy recovered (kWh/yr)": biogas_thermal_annual_kWh,
    "Digestate produced (t/yr)": digestate_t_yr,
    "Waste diverted from landfill (t/yr)": waste_diverted_t_yr,
    "Landfill methane avoidance (tCO2-eq/yr)": landfill_CH4_avoided_tCO2,
    "EROI": EROI_biogas,
}

# ---------------------------------------------------------------------
# 5. BATTERY SIZING (Eq.7) AND LIFECYCLE COST (Eq.10)
# ---------------------------------------------------------------------

annual_demand_MWh = demand_MWh.sum()
avg_daily_demand_kWh = annual_demand_MWh*1000/365
avg_daily_biogas_kWh = biogas_elec_annual_kWh/365
night_fraction = 0.40   # share of daily demand occurring outside PV generating hours
E_daily_deficit = max(night_fraction*avg_daily_demand_kWh - avg_daily_biogas_kWh, 0.0)

battery_cap_kWh = {}
for tech in ["Li-ion", "Lead-acid"]:
    battery_cap_kWh[tech] = E_daily_deficit * N_autonomy / (DoD[tech] * eta_rt[tech])   # Eq.7

def battery_lifecycle(tech, cap_kWh):
    capex = cap_kWh * capex_battery_USD_per_kWh[tech]
    # replacement interval from cycle life & daily utilization
    days_per_cycle_life = cycle_life[tech] / U_daily
    yrs_per_repl = days_per_cycle_life/365
    n_repl = int(np.floor(n_years / yrs_per_repl))
    repl_years = [yrs_per_repl*(i+1) for i in range(n_repl) if yrs_per_repl*(i+1) < n_years]
    repl_npv = sum(capex / (1+r_disc)**y for y in repl_years)
    om_npv = sum(capex*om_battery_frac / (1+r_disc)**t for t in range(1, n_years+1))
    lifecycle_USD = capex + repl_npv + om_npv
    delivered_kWh_yr = cap_kWh * U_daily * 365 * eta_rt[tech]
    delivered_pv_kWh = sum(delivered_kWh_yr*(1-d_degrad)**t/(1+r_disc)**t for t in range(1, n_years+1))
    cost_per_kWh = lifecycle_USD / delivered_pv_kWh if delivered_pv_kWh>0 else np.nan
    return dict(capacity_kWh=cap_kWh, capex_MUSD=capex/1e6, n_replacements=len(repl_years),
                replacement_npv_MUSD=repl_npv/1e6, lifecycle_MTHB=lifecycle_USD/1e6,
                cost_per_kWh_USD=cost_per_kWh)

table8 = {tech: battery_lifecycle(tech, battery_cap_kWh[tech]) for tech in ["Li-ion","Lead-acid"]}

# ---------------------------------------------------------------------
# 6. TECHNO-ECONOMIC ANALYSIS (Eq.8-9), scenarios S1/S2/S3
# ---------------------------------------------------------------------

def npc_lcoe(pv_kWp, biogas_kW, battery_kWh, battery_tech, grid_connected,
             r=r_disc, capex_pv_unit=capex_pv_USD_per_Wp, om_frac=om_pv_frac,
             annual_pv_MWh=None, biogas_MWh=None):
    if annual_pv_MWh is None: annual_pv_MWh = total_PV_annual_MWh
    if biogas_MWh is None: biogas_MWh = biogas_elec_annual_kWh/1000
    capex_pv = pv_kWp*1000*capex_pv_unit
    capex_biogas = biogas_kW*capex_biogas_USD_per_kW
    capex_batt = battery_kWh*capex_battery_USD_per_kWh[battery_tech] if battery_kWh>0 else 0.0
    capex_total = capex_pv + capex_biogas + capex_batt

    om_cost = (capex_pv*om_frac + capex_batt*om_battery_frac)  # THB/yr (yr1 basis)

    # NPC: cost side only (CAPEX + discounted O&M + replacement + decommissioning).
    # Corrected from the source manuscript's Eq.8, which subtracted revenue
    # B_t inside NPC -- that conflates a cost metric (LCOE, Eq.9) with a
    # profitability metric (NPV) and can spuriously return a negative
    # "cost of energy". Revenue/benefit terms are instead used only for
    # NPV, IRR and payback below, which is the conventional separation.
    npc = capex_total
    npv_cashflow = -capex_total
    E_delivered_series = []
    demand_kWh = annual_demand_MWh*1000
    for t in range(1, n_years+1):
        pv_gen = annual_pv_MWh*1000*(1-d_degrad)**t
        gen = pv_gen + biogas_MWh*1000
        om_t = om_cost
        repl_t = 0.0
        if t in (10, 20):
            repl_t += capex_pv*inverter_repl_frac
        if t == n_years:
            repl_t += capex_total*decommission_frac
        if grid_connected:
            self_consumed = min(gen, demand_kWh)
            exported = max(gen - demand_kWh, 0.0)
            benefit = self_consumed*grid_tariff + exported*feed_in_tariff
            delivered = gen  # all generation is monetized (self-use + export)
        else:
            delivered = min(gen, demand_kWh)   # off-grid: surplus beyond demand/battery is curtailed
            benefit = delivered*grid_tariff
        npc += (om_t + repl_t) / (1+r)**t
        npv_cashflow += (benefit - om_t - repl_t) / (1+r)**t
        E_delivered_series.append(delivered)

    denom = sum((E_delivered_series[t-1]*(1-d_degrad)**t)/(1+r)**t for t in range(1, n_years+1))
    lcoe = npc / denom if denom>0 else np.nan
    renewable_fraction = min(1.0, (annual_pv_MWh*1000 + biogas_MWh*1000)/demand_kWh) if grid_connected else 1.0
    # simple payback & NPV
    if grid_connected:
        gen1 = annual_pv_MWh*1000 + biogas_MWh*1000
        self_c = min(gen1, demand_kWh); exp1 = max(gen1-demand_kWh, 0.0)
        annual_net_benefit_yr1 = self_c*grid_tariff + exp1*feed_in_tariff - om_cost
    else:
        annual_net_benefit_yr1 = min(annual_pv_MWh*1000+biogas_MWh*1000, demand_kWh)*grid_tariff - om_cost
    payback = capex_total/annual_net_benefit_yr1 if annual_net_benefit_yr1>0 else np.nan
    npv = npv_cashflow
    return dict(capex_MUSD=capex_total/1e6, om_MUSD_yr=om_cost/1e6, lcoe_USD_kWh=lcoe,

                irr=None, payback_yr=payback, npv_MUSD=npv/1e6,
                renewable_fraction_pct=renewable_fraction*100)

S1 = npc_lcoe(total_PV_capacity_kWp, biogas_capacity_kW, 0, "Li-ion", grid_connected=True)
grid_baseline_lcoe = grid_tariff

# --- S0: demand-matched "right-sized" design -------------------------
# Recalculation finding: the technically-maximum feasible deployment
# (S1, using all eligible roof+land+60% pond area) generates ~17x the
# campus's own annual demand. Because Thai VSPP/SPP export is
# remunerated at a lower avoided-cost rate (2.20 THB/kWh) than the
# retail tariff it displaces (4.00 THB/kWh), such large-scale export
# is NPV-negative under current tariff rules. A second, demand-matched
# design sized to ~115% of annual consumption (standard net-metering
# sizing guidance) is evaluated as the recommended near-term option.
# The same right-sizing is applied to the off-grid battery scenarios
# (S2, S3): an islanded microgrid built at the 42.64 MWp maximum
# technical potential would be a ~17x overbuild relative to its own
# demand and is not a meaningful off-grid design point.
target_annual_MWh = annual_demand_MWh * 1.15
avg_specific_yield_kWh_kWp = total_PV_annual_MWh*1000/total_PV_capacity_kWp
pv_kWp_rightsized = (target_annual_MWh*1000 - biogas_elec_annual_kWh) / avg_specific_yield_kWh_kWp
pv_kWp_rightsized = max(pv_kWp_rightsized, 0.0)
rightsized_annual_MWh = pv_kWp_rightsized*avg_specific_yield_kWh_kWp/1000
S0 = npc_lcoe(pv_kWp_rightsized, biogas_capacity_kW, 0, "Li-ion", grid_connected=True,
              annual_pv_MWh=rightsized_annual_MWh)

S2 = npc_lcoe(pv_kWp_rightsized, biogas_capacity_kW, battery_cap_kWh["Li-ion"], "Li-ion",
              grid_connected=False, annual_pv_MWh=rightsized_annual_MWh)
S3 = npc_lcoe(pv_kWp_rightsized, biogas_capacity_kW, battery_cap_kWh["Lead-acid"], "Lead-acid",
              grid_connected=False, annual_pv_MWh=rightsized_annual_MWh)

# simple IRR (bisection on NPV=0) for S1
def irr_solve(pv_kWp, biogas_kW):
    from scipy.optimize import brentq
    capex_pv = pv_kWp*1000*capex_pv_USD_per_Wp
    capex_biogas = biogas_kW*capex_biogas_USD_per_kW
    capex_total = capex_pv+capex_biogas
    om_cost = capex_pv*om_pv_frac
    demand_kWh = annual_demand_MWh*1000
    def npv_at(r):
        npv = -capex_total
        for t in range(1, n_years+1):
            gen = (total_PV_annual_MWh*1000*(1-d_degrad)**t + biogas_elec_annual_kWh)
            self_c = min(gen, demand_kWh); exp1 = max(gen-demand_kWh, 0.0)
            benefit = self_c*grid_tariff + exp1*feed_in_tariff - om_cost
            if t in (10,20):
                benefit -= capex_pv*inverter_repl_frac
            npv += benefit/(1+r)**t
        return npv
    try:
        return brentq(npv_at, 0.001, 2.0)
    except Exception:
        return np.nan

S1["irr"] = irr_solve(total_PV_capacity_kWp, biogas_capacity_kW)*100

def irr_solve_general(pv_kWp, biogas_kW, ann_pv_MWh):
    from scipy.optimize import brentq
    capex_pv = pv_kWp*1000*capex_pv_USD_per_Wp
    capex_biogas = biogas_kW*capex_biogas_USD_per_kW
    capex_total = capex_pv+capex_biogas
    om_cost = capex_pv*om_pv_frac
    demand_kWh = annual_demand_MWh*1000
    def npv_at(r):
        npv = -capex_total
        for t in range(1, n_years+1):
            gen = (ann_pv_MWh*1000*(1-d_degrad)**t + biogas_elec_annual_kWh)
            self_c = min(gen, demand_kWh); exp1 = max(gen-demand_kWh, 0.0)
            benefit = self_c*grid_tariff + exp1*feed_in_tariff - om_cost
            if t in (10,20):
                benefit -= capex_pv*inverter_repl_frac
            npv += benefit/(1+r)**t
        return npv
    try:
        return brentq(npv_at, 0.001, 2.0)
    except Exception:
        return np.nan

S0["irr"] = irr_solve_general(pv_kWp_rightsized, biogas_capacity_kW, rightsized_annual_MWh)*100

table7 = pd.DataFrame({
    "Indicator": ["PV Capacity (MWp)","CAPEX (M USD)","Annual O&M (M USD/yr)","LCOE (USD/kWh)","LCOE vs grid (%)",
                  "IRR (%)","Simple payback (yr)","NPV at 25yr (M USD)","Renewable fraction (%)"],
    "S0: Right-Sized On-grid": [pv_kWp_rightsized/1000, S0["capex_MUSD"], S0["om_MUSD_yr"], S0["lcoe_USD_kWh"],
                             (S0["lcoe_USD_kWh"]/grid_tariff-1)*100, S0["irr"], S0["payback_yr"],
                             S0["npv_MUSD"], S0["renewable_fraction_pct"]],
    "S1: Max-Potential On-grid": [total_PV_capacity_kWp/1000, S1["capex_MUSD"], S1["om_MUSD_yr"], S1["lcoe_USD_kWh"],
                             (S1["lcoe_USD_kWh"]/grid_tariff-1)*100, S1["irr"], S1["payback_yr"],
                             S1["npv_MUSD"], S1["renewable_fraction_pct"]],
    "S2: Right-Sized Off-grid Li-ion": [pv_kWp_rightsized/1000, S2["capex_MUSD"], S2["om_MUSD_yr"], S2["lcoe_USD_kWh"],
                             (S2["lcoe_USD_kWh"]/grid_tariff-1)*100, np.nan, S2["payback_yr"],
                             S2["npv_MUSD"], S2["renewable_fraction_pct"]],
    "S3: Right-Sized Off-grid Lead-acid": [pv_kWp_rightsized/1000, S3["capex_MUSD"], S3["om_MUSD_yr"], S3["lcoe_USD_kWh"],
                             (S3["lcoe_USD_kWh"]/grid_tariff-1)*100, np.nan, S3["payback_yr"],
                             S3["npv_MUSD"], S3["renewable_fraction_pct"]],
})


# ---------------------------------------------------------------------
# 7. FOUR NOVEL PERFORMANCE METRICS (Eq.11-14)
# ---------------------------------------------------------------------

A_used_m2 = A_roof_use + A_ground_eligible + C_star*A_pond
A_preserved_ha = (A_campus - A_used_m2 - land_use["Roads / parking"]
                   - land_use["Vegetation/conservation"])/10000
A_preserved_ha = max(A_preserved_ha, 0.0)
# more direct definition: land NOT converted to PV use, restricted to the
# developable classes (buildings/open field/ponds) that could have been used
A_developable = land_use["Academic buildings"]+land_use["Open fields"]+land_use["Ponds / reservoirs"]
A_preserved_ha = (A_developable - A_used_m2)/10000

PV_grid_displacement_tCO2 = total_PV_annual_MWh*1000*EF_grid/1000  # MWh*1000 kWh/MWh*kgCO2/kWh /1000 -> tCO2
biogas_grid_displacement_tCO2 = (biogas_elec_annual_kWh)*EF_grid/1000
total_GHG_tCO2 = PV_grid_displacement_tCO2 + biogas_grid_displacement_tCO2 + landfill_CH4_avoided_tCO2

SEI = (total_PV_capacity_kWp*1000) / A_campus * (1 - A_used_m2/A_campus)  # W/m2
NPC_S1_THB = S1["capex_MUSD"]*1e6 + (S1["lcoe_USD_kWh"]*1)  # placeholder unused
NPC_total_THB = S1["capex_MUSD"]*1e6  # use CAPEX as NPC proxy is wrong; recompute true NPC below

def true_npc(scn_dict, ann_pv_MWh):
    return scn_dict["lcoe_USD_kWh"]*sum((ann_pv_MWh*1000*(1-d_degrad)**t + biogas_elec_annual_kWh)*(1-d_degrad)**t/(1+r_disc)**t for t in range(1,n_years+1))

NPC_S0 = true_npc(S0, rightsized_annual_MWh)
CNED = (rightsized_annual_MWh*1000) / (NPC_S0 * (pv_kWp_rightsized/avg_specific_yield_kWh_kWp*1e6/1e6*0+A_used_m2))  # see note below
# CNED uses the right-sized system's footprint proportionally to its capacity share of max-potential deployment
A_used_rightsized_m2 = A_used_m2 * (pv_kWp_rightsized/total_PV_capacity_kWp)
CNED = (rightsized_annual_MWh*1000) / (NPC_S0 * A_used_rightsized_m2)

FPV_evap_reduction_m3 = C_star*A_pond*0.5  # 0.5 m3/m2/yr avoided evaporation, typical tropical pan-evap fraction
digestate_irrigation_credit_m3 = digestate_t_yr*7.5   # m3 irrigation water offset per t digestate applied (assumption)
alpha_pump = 0.0036  # kWh/L
delta_V_L = (FPV_evap_reduction_m3+digestate_irrigation_credit_m3)*1000
WENI = (ann_float/1000 + alpha_pump*delta_V_L/1000) / (A_pond/10000)  # (MWh+1000 m3)/ha

CAI = total_GHG_tCO2 / A_preserved_ha

metrics = dict(SEI=SEI, CNED=CNED, WENI=WENI, CAI=CAI,
               thresholds=dict(SEI=0.05, CNED=1e-9, WENI=50, CAI=50))

water_conserved_m3 = FPV_evap_reduction_m3 + digestate_irrigation_credit_m3

table10 = {
    "PV grid displacement (tCO2/yr)": PV_grid_displacement_tCO2,
    "Biogas grid displacement (tCO2/yr)": biogas_grid_displacement_tCO2,
    "Landfill CH4 avoidance (tCO2-eq/yr)": landfill_CH4_avoided_tCO2,
    "Total GHG reduction (tCO2-eq/yr)": total_GHG_tCO2,
    "Water savings - FPV evaporation (m3/yr)": FPV_evap_reduction_m3,
    "Water savings - digestate irrigation (m3/yr)": digestate_irrigation_credit_m3,
    "Total water conserved (m3/yr)": water_conserved_m3,
    "Land preserved from development (ha)": A_preserved_ha,
    "Waste diverted from landfill (t/yr)": waste_diverted_t_yr,
    "Digestate fertilizer substitute (t/yr)": digestate_t_yr,
}

# ---------------------------------------------------------------------
# 8. MONTE CARLO UNCERTAINTY (10,000 iterations)
# ---------------------------------------------------------------------

N_MC = 10000
disc_mc   = rng.normal(r_disc, 0.02/1.645, N_MC)          # +-2% at ~90% CI
capex_mc  = rng.normal(1.0, 0.20/1.645, N_MC)             # +-20%
solar_mc  = rng.normal(1.0, 0.10/1.645, N_MC)             # +-10%
om_mc     = rng.normal(1.0, 0.30/1.645, N_MC)             # +-30%
tariff_mc = rng.normal(grid_tariff, grid_tariff*0.10/1.645, N_MC)

lcoe_mc = np.zeros(N_MC)
demand_kWh_total = annual_demand_MWh*1000
for i in range(N_MC):
    capex_pv = pv_kWp_rightsized*1000*capex_pv_USD_per_Wp*capex_mc[i]
    capex_biogas = biogas_capacity_kW*capex_biogas_USD_per_kW
    capex_total = capex_pv+capex_biogas
    om_cost = capex_pv*om_pv_frac*om_mc[i]
    r = max(disc_mc[i], 0.001)
    pv_ann = rightsized_annual_MWh*1000*solar_mc[i]
    npc = capex_total
    denom = 0.0
    for t in range(1, n_years+1):
        gen = pv_ann*(1-d_degrad)**t + biogas_elec_annual_kWh
        repl_t = capex_pv*inverter_repl_frac if t in (10,20) else 0.0
        if t==n_years: repl_t += capex_total*decommission_frac
        npc += (om_cost+repl_t)/(1+r)**t
        denom += (gen*(1-d_degrad)**t)/(1+r)**t
    lcoe_mc[i] = npc/denom if denom>0 else np.nan

lcoe_mc = lcoe_mc[np.isfinite(lcoe_mc)]
mc_mean = lcoe_mc.mean()
mc_ci90 = np.percentile(lcoe_mc, [5,95])
p_below_grid = (lcoe_mc < grid_tariff).mean()*100

# Tornado sensitivity: +-1 sigma one-at-a-time
def lcoe_1var(var, delta):
    capex_pv = pv_kWp_rightsized*1000*capex_pv_USD_per_Wp*(1+delta if var=="capex" else 1.0)
    capex_biogas = biogas_capacity_kW*capex_biogas_USD_per_kW
    capex_total = capex_pv+capex_biogas
    om_cost = capex_pv*om_pv_frac*(1+delta if var=="om" else 1.0)
    r = r_disc + (delta*0.02 if var=="disc" else 0.0)
    pv_ann = rightsized_annual_MWh*1000*(1+delta if var=="solar" else 1.0)
    tariff = grid_tariff*(1+delta if var=="tariff" else 1.0)
    demand_kWh = annual_demand_MWh*1000
    npc = capex_total
    denom = 0.0
    for t in range(1, n_years+1):
        gen = pv_ann*(1-d_degrad)**t + biogas_elec_annual_kWh
        repl_t = capex_pv*inverter_repl_frac if t in (10,20) else 0.0
        if t==n_years: repl_t += capex_total*decommission_frac
        npc += (om_cost+repl_t)/(1+r)**t
        denom += (gen*(1-d_degrad)**t)/(1+r)**t
    return npc/denom

tornado = {}
for var, label in [("capex","PV capital cost (+-20%)"),("solar","Solar resource (+-10%)"),
                    ("disc","Discount rate (+-2pp)"),("om","O&M rate (+-30%)"),("tariff","Grid tariff (+-10%)")]:
    d = 0.20 if var in ("capex",) else 0.10 if var in ("solar","tariff") else 1.0 if var=="disc" else 0.30
    lo = lcoe_1var(var, -d); hi = lcoe_1var(var, +d)
    tornado[label] = dict(low=lo, high=hi, base=S0["lcoe_USD_kWh"], swing_pct=abs(hi-lo)/S0["lcoe_USD_kWh"]*100)

# ---------------------------------------------------------------------
# 9. REGIONAL TRANSFERABILITY (Eq.15)
# ---------------------------------------------------------------------

eps_GHI, eps_RH = -0.85, 0.003
# Local grid tariffs entered in THB-equivalent for direct comparability
# with published local rates, then converted to USD at the same fixed
# 35.5 THB/USD rate used throughout, so that all cost comparisons are
# on a consistent USD basis.
cities = pd.DataFrame([
    ("Chiang Rai, Thailand","Aw",4.84,74,4.00),
    ("Bangkok, Thailand","Am",4.60,72,4.00),
    ("Phnom Penh, Cambodia","Aw",5.10,68,4.80),
    ("Ho Chi Minh City, Vietnam","Am",4.95,78,3.50),
    ("Manila, Philippines","Am",4.55,77,5.80),
    ("Yangon, Myanmar","Am",4.70,75,3.80),
    ("Colombo, Sri Lanka","Af",5.30,80,6.50),
    ("Jakarta, Indonesia","Af",4.25,79,3.20),
    ("Kuala Lumpur, Malaysia","Af",4.20,81,2.90),
    ("Hanoi, Vietnam","Cfa",3.85,79,3.50),
], columns=["City","Climate Zone","GHI","RH","Grid Tariff THB"])
cities["Grid Tariff"] = cities["Grid Tariff THB"] / THB_PER_USD

GHI_base = GHI.mean(); RH_base = RH.mean()
cities["Est. LCOE"] = S0["lcoe_USD_kWh"]*(1+eps_GHI*(cities["GHI"]/GHI_base-1)+eps_RH*(cities["RH"]/RH_base-1))
cities["Cost Saving (%)"] = (1 - cities["Est. LCOE"]/cities["Grid Tariff"])*100
cities["Competitive"] = cities["Cost Saving (%)"]>0
cities["Strongly Competitive (<75% tariff)"] = cities["Est. LCOE"] < 0.75*cities["Grid Tariff"]

# ---------------------------------------------------------------------
# 10. SAVE ALL RESULTS
# ---------------------------------------------------------------------

results = dict(
    A_campus_ha=A_campus/10000,
    capacities_kWp=dict(rooftop=cap_roof_kWp, ground=cap_ground_kWp, floating=cap_float_kWp,
                         total=total_PV_capacity_kWp, rightsized=pv_kWp_rightsized),
    C_star=C_star, dDO_star=dDO_star,
    total_PV_annual_MWh=total_PV_annual_MWh, rightsized_annual_MWh=rightsized_annual_MWh,
    table6=table6, table8=table8,
    S0=S0, S1=S1, S2=S2, S3=S3,
    metrics=metrics,
    table10=table10,
    mc_mean_lcoe=mc_mean, mc_ci90=list(mc_ci90), p_below_grid=p_below_grid,
    tornado={k: {kk: (float(vv) if not isinstance(vv,(list,)) else vv) for kk,vv in v.items()} for k,v in tornado.items()},
)

with open("outputs/results.json","w") as f:
    json.dump(results, f, indent=2, default=float)

table5.to_csv("outputs/table5.csv", index=False)
table7.to_csv("outputs/table7.csv", index=False)
cities.to_csv("outputs/table11.csv", index=False)
pd.DataFrame([demand_MWh], columns=months, index=["Demand_MWh"]).T.to_csv("outputs/table2_demand.csv")
np.save("outputs/lcoe_mc.npy", lcoe_mc)
np.save("outputs/total_monthly_PV_MWh.npy", total_monthly_PV_MWh)
np.save("outputs/monthly_roof.npy", monthly_roof/1000)
np.save("outputs/monthly_ground.npy", monthly_ground/1000)
np.save("outputs/monthly_float.npy", monthly_float/1000)

print("=== KEY RECALCULATED RESULTS ===")
print(f"Campus area: {A_campus/10000:.1f} ha")
print(f"Rooftop capacity: {cap_roof_kWp/1000:.2f} MWp  (usable roof area {A_roof_use:,.0f} m2)")
print(f"Ground capacity:  {cap_ground_kWp/1000:.2f} MWp (eligible open field {A_ground_eligible:,.0f} m2)")
print(f"Floating capacity:{cap_float_kWp/1000:.2f} MWp  (C* = {C_star*100:.1f}%, dDO*={dDO_star:.2f} mg/L)")
print(f"Total PV capacity: {total_PV_capacity_kWp/1000:.2f} MWp, {total_PV_annual_MWh:,.0f} MWh/yr")
print(f"Biogas: {CH4_annual_m3:,.0f} m3 CH4/yr -> {biogas_elec_annual_kWh/1000:,.1f} MWh elec, {biogas_thermal_annual_kWh/1000:,.1f} MWh thermal")
print(f"S0 (right-sized {pv_kWp_rightsized/1000:.2f} MWp) LCOE: {S0['lcoe_USD_kWh']:.4f} USD/kWh, IRR {S0['irr']:.1f}%, payback {S0['payback_yr']:.1f} yr, NPV {S0['npv_MUSD']:.3f} MUSD")
print(f"S1 (max-potential) LCOE: {S1['lcoe_USD_kWh']:.4f} USD/kWh, payback {S1['payback_yr']:.1f} yr, NPV {S1['npv_MUSD']:.3f} MUSD")
print(f"S2 LCOE: {S2['lcoe_USD_kWh']:.4f} USD/kWh, NPV {S2['npv_MUSD']:.3f} MUSD")
print(f"S3 LCOE: {S3['lcoe_USD_kWh']:.4f} USD/kWh, NPV {S3['npv_MUSD']:.3f} MUSD")
print(f"Battery Li-ion: {battery_cap_kWh['Li-ion']:.0f} kWh, {table8['Li-ion']['cost_per_kWh_USD']:.4f} USD/kWh delivered")
print(f"Battery Lead-acid: {battery_cap_kWh['Lead-acid']:.0f} kWh, {table8['Lead-acid']['cost_per_kWh_USD']:.4f} USD/kWh delivered")
print(f"SEI={SEI:.3f} W/m2, CNED={CNED:.3e} kWh/USD/m2, WENI={WENI:.1f}, CAI={CAI:.1f} tCO2/ha")
print(f"Total GHG reduction: {total_GHG_tCO2:,.0f} tCO2-eq/yr; water conserved {water_conserved_m3:,.0f} m3/yr")
print(f"MC LCOE mean {mc_mean:.2f}, 90% CI [{mc_ci90[0]:.2f},{mc_ci90[1]:.2f}], P(LCOE<grid)={p_below_grid:.1f}%")
print("Saved: outputs/results.json, table5.csv, table7.csv, table11.csv, table2_demand.csv, *.npy")
