"""Graphical abstract of the Utilities Policy version (Elsevier: >= 1328 x 531 px)."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
R = json.load(open("outputs/results.json")); U = json.load(open("outputs/up_results.json")); OP = pd.read_csv("outputs/up_optima.csv")
T = R["tiers"]
fig, ax = plt.subplots(1, 3, figsize=(10.0, 4.0), gridspec_kw={"width_ratios": [1.0, 1.1, 1.15]})
names = ["Rooftop", "Ground", "Ponds", "Reservoir"]; keys = ["roof", "ground", "ponds", "reservoir"]; cols = ["#D98E04", "#7A9E3B", "#2E86AB", "#1B5E7B"]; left = 0
for n, k, c in zip(names, keys, cols):
    v = T[k]["kwp"] / 1000; ax[0].barh(0, v, left=left, color=c, height=0.5, label=f"{n} {v:.1f}"); left += v
ax[0].barh(-0.7, U["P0"] / 1000, color="#d1495b", height=0.5)
ax[0].text(1.6, -0.7, "Optimum, no export\nremuneration: 1.25 MWp (5%)", va="center", fontsize=8.5)
ax[0].text(left, 0.42, f"{left:.1f} MWp technical potential", ha="right", fontsize=8.5)
ax[0].set_yticks([]); ax[0].set_xlabel("PV capacity (MWp)"); ax[0].set_ylim(-1.2, 1.9); ax[0].set_xlim(0, 26.5)
ax[0].legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2, title="Campus surfaces (MWp)", title_fontsize=8)
d = OP[OP.capex == 1.0]; ax[1].step(d.exp_thb, d.p_kwp / 1000, where="post", color="#00798c", lw=2)
ax[1].set_yscale("log"); ax[1].set_yticks([1, 2, 3, 5, 10, 25]); ax[1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
for xv, lab in ((1.0, "2021 pilot\nrate"), (2.2, "residential rate\n(counterfactual)")):
    ax[1].axvline(xv, color="0.5", ls=":", lw=1); ax[1].text(xv - 0.05, 18, lab, ha="right", va="top", fontsize=7.5, color="0.4")
ax[1].set_xlabel("Export remuneration (THB/kWh)"); ax[1].set_ylabel("Institution's optimal PV (MWp)")
S = U["schemes"]; lab = ["Uncomp.\nexport", "Curtailed", "Net bill.\n1.00", "Net bill.\n2.20*", "Net\nmetering*"]; L_ = list(S); x = np.arange(len(L_))
sc = np.array([S[l]["comp_sc_2.2"] / 1e3 for l in L_]); ex = np.array([S[l]["comp_ex_2.2"] / 1e3 for l in L_])
ax[2].bar(x, sc, color="#2E86AB", width=0.62, label="from self-consumption")
ax[2].bar(x, ex, bottom=np.where(ex < 0, sc, 0), color="#D98E04", width=0.62, label="from delivered exports")
for i, l in enumerate(L_): ax[2].text(i, -212, f"{S[l]['p_kwp'] / 1000:.2f}\nMWp", ha="center", fontsize=7)
ax[2].text(4.4, 62, "* counterfactual", ha="right", fontsize=7, color="0.4")
ax[2].axhline(0, color="k", lw=0.6); ax[2].set_xticks(x); ax[2].set_xticklabels(lab, fontsize=6.8); ax[2].set_ylim(-235, 80)
ax[2].set_ylabel("Utility net margin (k USD/yr)"); ax[2].legend(fontsize=7, frameon=False, loc="upper left")
for a, l in zip(ax, ["(a)", "(b)", "(c)"]): a.set_title(l, fontsize=10, loc="center"); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures_up/graphical_abstract.png", dpi=300); fig.savefig("figures_up/graphical_abstract.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
print("written")
