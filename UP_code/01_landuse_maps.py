"""
MSROF-CRRU | Script 1 of 3 | Land-use verification and PV-zone map
=====================================================================
Inputs (the two GIS map exports of the campus, same raster, 500 dpi):
  "zones used_edit.png"  classified land use (buildings, open fields, ponds/reservoirs, roads, vegetation)
  "PV zones_edit.tif"    PV zoning drawn on the same base map (none, floating, ground-mounted, rooftop)
Method (in-house, no GIS licence needed):
  1. Locate the neatline of each map; both frames are 3,520 x 4,290 px and differ by a translation only.
  2. Scale from the UTM tick marks: 1,690 px per 1,000 m on both axes (0.5917 m/px).
  3. Nearest-reference-colour classification of the flat map colours; area restricted to the campus body
     (largest connected region of the non-road classes, closed and hole-filled), province inset excluded.
  4. Connected components separate the large reservoir from the small ponds.
  5. Cross-tabulation of the PV-zone map against the land-use map, pixel by pixel.
  6. Corrected PV-zone map: pixels drawn as "ground-mounted" on land classified as vegetation/conservation
     are recoloured as "no PV (vegetation and conservation)", so that the ground-mounted zone equals the
     open-field class used in the analysis. The cartography of the original export is otherwise kept.
Outputs: outputs/landuse_raster_check.json, figures/fig2_maps.png, figures/figS1_landuse_shares.png,
         prepared/pvzones_corrected.png
Usage: python 01_landuse_maps.py "zones used_edit.png" "PV zones_edit.tif"
"""
import sys, os, json
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import matplotlib as mpl, matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None
os.makedirs("outputs", exist_ok=True); os.makedirs("figures", exist_ok=True); os.makedirs("prepared", exist_ok=True)
F_LU = sys.argv[1] if len(sys.argv) > 1 else "zones used_edit.png"
F_PV = sys.argv[2] if len(sys.argv) > 2 else "PV zones_edit.tif"
TABLE1 = {"Buildings": 213580.61, "Open fields": 48682.98, "Ponds": 364785.01, "Roads": 185392.01, "Vegetation": 1204935.32}
REF_LU = {"Buildings": (181, 67, 63), "Open fields": (222, 158, 102), "Ponds": (151, 219, 242), "Roads": (225, 225, 225), "Vegetation": (122, 209, 82)}
REF_PV = {"none": (224, 224, 224), "Floating": (144, 216, 240), "Ground": (208, 136, 72), "Rooftop": (152, 112, 208)}
VEG_RGB = (122, 209, 82)

def frame(im):
    dark = im.sum(2) < 120; H, W = dark.shape
    c = np.where(dark.sum(0) > 0.6 * H)[0]; r = np.where(dark.sum(1) > 0.6 * W)[0]
    return int(c.min()), int(r.min()), int(c.max()), int(r.max())

def ticks(im, fx, fy):
    dark = im.sum(2) < 200
    def grp(idx):
        g = []; s = p = idx[0]
        for v in idx[1:]:
            if v - p > 5: g.append((s + p) / 2); s = v
            p = v
        g.append((s + p) / 2); return g
    top = grp(np.where(dark[fy - 30:fy - 2, :].sum(0) > 15)[0]); left = grp(np.where(dark[:, fx - 30:fx - 2].sum(1) > 15)[0])
    return np.diff(top).mean(), np.diff(left).mean()

LU = np.array(Image.open(F_LU).convert("RGB")).astype(int); PVi = np.array(Image.open(F_PV).convert("RGB")).astype(int)
f1, f2 = frame(LU), frame(PVi)
assert (f1[2] - f1[0], f1[3] - f1[1]) == (f2[2] - f2[0], f2[3] - f2[1]), "map frames differ in size"
px_e, px_n = ticks(LU, f1[0], f1[1]); MPX = 1000.0 / ((px_e + px_n) / 2); PA = MPX ** 2
m = 12
sub = LU[f1[1] + m:f1[3] - m, f1[0] + m:f1[2] - m]; subv = PVi[f2[1] + m:f2[3] - m, f2[0] + m:f2[2] - m]

def classify(img, ref, valid):
    names = list(ref); R = {}
    for k, c in ref.items():
        d = np.abs(img - np.array(c)).sum(2); R[k] = np.median(img[(d < 40) & valid], 0)
    d = np.stack([np.abs(img - R[k]).sum(2) for k in names], -1)
    return names, d.argmin(-1), (d.min(-1) < 45) & valid

valid = np.ones(sub.shape[:2], bool)
names, cz, okz = classify(sub, REF_LU, valid)
body = ndi.binary_closing(okz & (cz != names.index("Roads")), iterations=25)
lab, _ = ndi.label(body); campus = ndi.binary_dilation(ndi.binary_fill_holes(lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1), iterations=6)
okz &= campus
namesv, cv, okv = classify(subv, REF_PV, campus)

res = dict(m_per_px=MPX, px_per_km_e=px_e, px_per_km_n=px_n, classes={})
for i, k in enumerate(names):
    a = float(((cz == i) & okz).sum() * PA)
    res["classes"][k] = dict(raster_m2=a, table1_m2=TABLE1[k], diff_pct=100 * (a / TABLE1[k] - 1))
res["raster_total_m2"] = sum(v["raster_m2"] for v in res["classes"].values()); res["table1_total_m2"] = sum(TABLE1.values())
def comps(mask, it=2):
    lab, _ = ndi.label(ndi.binary_opening(mask, iterations=it)); a = np.sort(np.bincount(lab.ravel())[1:] * PA)[::-1]; return a[a > 200]
ponds = comps((cz == names.index("Ponds")) & okz)
res["reservoir_m2_used"] = float(ponds[0]); res["small_ponds_m2_used"] = TABLE1["Ponds"] - float(ponds[0]); res["n_small_ponds"] = int(len(ponds) - 1)
res["small_pond_areas_m2"] = [float(x) for x in ponds[1:]]
res["open_field_patches_m2"] = [float(x) for x in comps((cz == names.index("Open fields")) & okz)]
res["pvmap_zones_m2"] = {k: float(((cv == i) & okv).sum() * PA) for i, k in enumerate(namesv)}
ct = pd.DataFrame([[float(((cv == i) & (cz == j) & okv & okz).sum() * PA) for j in range(len(names))] for i in range(len(namesv))], index=namesv, columns=names)
res["crosstab_m2"] = ct.to_dict()
res["roof_eligible_m2"] = float(ct.loc["Rooftop", "Buildings"])                       # roofs retained by the PV zoning
res["roof_screened_out_m2"] = float(ct.loc["none", "Buildings"])
json.dump(res, open("outputs/landuse_raster_check.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("crosstab_m2", "small_pond_areas_m2")}, indent=1)); print((ct / 1e4).round(2))

# ---------------- corrected PV-zone map on the original cartography
fix = (cv == namesv.index("Ground")) & (cz == names.index("Vegetation")) & okv & okz
fix = ndi.binary_dilation(fix, iterations=1) & ~((cv != namesv.index("Ground")) & okv)
out = PVi.copy().astype(np.uint8)
win = out[f2[1] + m:f2[3] - m, f2[0] + m:f2[2] - m]
win[fix] = VEG_RGB
fix2 = (cv == namesv.index("Ground")) & (cz == names.index("Buildings")) & okv & okz      # ground zone drawn on a building footprint
fix2 = ndi.binary_dilation(fix2, iterations=1) & ~((cv != namesv.index("Ground")) & okv)
win[fix2] = (224, 224, 224)
res["ground_on_buildings_recoloured_m2"] = float(fix2.sum() * PA); res["ground_on_vegetation_recoloured_m2"] = float(fix.sum() * PA)
json.dump(res, open("outputs/landuse_raster_check.json", "w"), indent=1)
img = Image.fromarray(out); dr = ImageDraw.Draw(img)
# redraw legend area (below the neatline, right of the scale bar)
H, W = out.shape[:2]; y0 = f2[3] + 150; x0 = int(W * 0.33)
dr.rectangle([x0, y0, W - 1, H - 1], fill=(255, 255, 255))
try:
    fb = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf", 78)
    fr = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 62)
except OSError:
    fb = ImageFont.truetype("DejaVuSerif-Bold.ttf", 70); fr = ImageFont.truetype("DejaVuSerif.ttf", 56)
dr.text((x0 + 60, y0 + 40), "PV zones", font=fb, fill=(0, 0, 0))
items = [((152, 112, 208), "Rooftop (screened roofs)"), ((208, 136, 72), "Ground-mounted (open fields)"), ((144, 216, 240), "Floating (ponds and reservoir)"),
         ((224, 224, 224), "No PV (roads, parking, excluded roofs)"), (VEG_RGB, "No PV (vegetation and conservation)")]
cx, cy = x0 + 60, y0 + 170
for i, (c, t) in enumerate(items):
    xx = cx + (i % 2) * 1250; yy = cy + (i // 2) * 105
    dr.rectangle([xx, yy, xx + 150, yy + 70], fill=c, outline=(90, 90, 90)); dr.text((xx + 180, yy + 2), t, font=fr, fill=(0, 0, 0))
img.save("prepared/pvzones_corrected.png")

# ---------------- Fig. 1: (a) land use, (b) corrected PV zones, labels centred above
mpl.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 900, "savefig.bbox": "tight"})
A = Image.open(F_LU).convert("RGB"); B = img
fig, ax = plt.subplots(1, 2, figsize=(12, 8.2))
for a, im_, t in ((ax[0], A, "(a)"), (ax[1], B, "(b)")):
    a.imshow(im_); a.set_axis_off(); a.text(0.5, 1.01, t, transform=a.transAxes, ha="center", va="bottom", fontsize=12, fontweight="bold")
plt.subplots_adjust(wspace=0.03); plt.savefig("figures/fig2_maps.png"); plt.close()

# ---------------- Fig. S1: area shares measured on the land-use map
vals = [res["classes"]["Buildings"]["raster_m2"], res["classes"]["Open fields"]["raster_m2"], res["classes"]["Ponds"]["raster_m2"] - res["reservoir_m2_used"],
        res["reservoir_m2_used"], res["classes"]["Roads"]["raster_m2"], res["classes"]["Vegetation"]["raster_m2"]]
labs = ["Academic buildings", "", "", "Reservoir", "Roads / parking", "Vegetation / conservation"]
cols = ["#B5433F", "#DE9E66", "#6CC3E0", "#97DBF2", "#C9C9C9", "#7AD152"]
fig, a1 = plt.subplots(figsize=(6.5, 5.4))
wd, _, _ = a1.pie(vals, labels=labs, colors=cols, startangle=90, counterclock=False, pctdistance=0.76, labeldistance=1.12,
                  autopct=lambda p: f"{p:.1f}%\n({p * sum(vals) / 1e6:.1f} ha)" if p > 5 else "",
                  wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.2), textprops=dict(fontsize=8.5))
for i, txt, yy in [(1, f"Open fields, {vals[1] / sum(vals) * 100:.1f} % ({vals[1] / 1e4:.1f} ha)", 1.02), (2, f"Small ponds, {vals[2] / sum(vals) * 100:.1f} % ({vals[2] / 1e4:.1f} ha)", 0.80)]:
    ang = np.deg2rad((wd[i].theta1 + wd[i].theta2) / 2)
    a1.annotate(txt, xy=(0.97 * np.cos(ang), 0.97 * np.sin(ang)), xytext=(1.30, yy), fontsize=8.5, va="center", arrowprops=dict(arrowstyle="-", color="#555", lw=0.8))
plt.savefig("figures/figS1_landuse_shares.png"); plt.close()
print("written outputs/landuse_raster_check.json, figures/fig2_maps.png, figures/figS1_landuse_shares.png")
