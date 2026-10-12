"""
MSROF-CRRU | Script 9 | Monte Carlo with re-optimization of the capacity in every draw (Utilities Policy version)
Uses the 10,000 input draws written by script 06 (outputs/up_mc_inputs.npz, seed 20261007). For each draw the
NPV-maximizing capacity is searched on the 250 kWp grid (0 to 5 MWp without export remuneration; 0 to 8 MWp at
2.20 THB/kWh, then 16 larger steps up to the full surface potential of 25.1 MWp, so that no draw is censored by the search range). This gives the distribution of the optimum itself, and of the NPV at that optimum, is obtained,
together with the regret of keeping the base-case design. Runs on all available cores (about 15 min on two cores).
Outputs: outputs/up_mc_reopt.npy, outputs/up_mc_reopt.json
"""
import json, numpy as np
from multiprocessing import Pool
SRC = open("02_msrof_analysis.py", encoding="utf-8").read()
exec(SRC.split("# 7. CAPACITY-VALUE CURVE")[0].rsplit("# ====", 1)[0])
D = dict(np.load("outputs/up_mc_inputs.npz")); NMC = len(D["capex_pv"])
U = json.load(open("outputs/up_results.json")); P0 = U["P0"]; P22 = [r["p_kwp"] for r in U["opt_base"] if abs(r["exp_thb"] - 2.2) < 1e-9][0]
T_ = tiers(); P_ALL = sum(t["kwp"] for t in T_.values())
G0 = np.arange(0, 5001, 250.0); G22 = np.concatenate([np.arange(0, 8001, 250.0), np.geomspace(8500, P_ALL, 16)])   # up to the full surface potential
AL = {p: allocate(p) for p in np.union1d(G0, G22)}

def one(i):
    th = {k: float(v[i]) for k, v in D.items()}
    v0 = np.array([economics(*AL[p], th, bg=False)["npv"] for p in G0])
    th22 = dict(th, exp_thb=2.2); v22 = np.array([economics(*AL[p], th22, bg=False)["npv"] for p in G22])
    i0, i22 = int(v0.argmax()), int(v22.argmax())
    return [G0[i0], v0[i0], v0[G0 == P0][0], G22[i22], v22[i22], v22[G22 == P22][0]]

if __name__ == "__main__":
    with Pool() as pool: R = np.array(pool.map(one, range(NMC), chunksize=50))
    np.save("outputs/up_mc_reopt.npy", R)
    pc = lambda a: {f"p{q}": float(np.percentile(a, q)) for q in (5, 10, 25, 50, 75, 90, 95)}
    share = lambda a, g: {f"{p / 1000:.2f}": float((a == p).mean() * 100) for p in g if (a == p).mean() > 0}
    OUT = dict(n=NMC, x0=dict(p_opt=pc(R[:, 0]), p_opt_share=share(R[:, 0], G0), share_base=float((R[:, 0] == P0).mean() * 100),
                              share_within_one_step=float((np.abs(R[:, 0] - P0) <= 250).mean() * 100), npv_opt=pc(R[:, 1]),
                              p_npv_opt_pos=float((R[:, 1] > 0).mean() * 100), regret=pc(R[:, 1] - R[:, 2]), regret_mean=float((R[:, 1] - R[:, 2]).mean()),
                              at_zero=float((R[:, 0] == 0).mean() * 100), at_grid_max=float((R[:, 0] == G0[-1]).mean() * 100)),
               x22=dict(p_opt=pc(R[:, 3]), p_opt_share=share(R[:, 3], G22), share_base=float((R[:, 3] == P22).mean() * 100),
                        share_within_one_step=float((np.abs(R[:, 3] - P22) <= 250).mean() * 100), npv_opt=pc(R[:, 4]),
                        p_npv_opt_pos=float((R[:, 4] > 0).mean() * 100), regret=pc(R[:, 4] - R[:, 5]), regret_mean=float((R[:, 4] - R[:, 5]).mean()),
                        at_zero=float((R[:, 3] == 0).mean() * 100), at_grid_max=float((R[:, 3] == G22[-1]).mean() * 100), above_8mwp=float((R[:, 3] > 8000).mean() * 100),
                        above_6mwp=float((R[:, 3] > 6000).mean() * 100), p_opt_below8=pc(R[R[:, 3] <= 8000, 3]),
                        p_pos_fixed=float((R[:, 5] > 0).mean() * 100)),
               P0=P0, P22=P22)
    json.dump(OUT, open("outputs/up_mc_reopt.json", "w"), indent=1)
    print(json.dumps(OUT, indent=1))
