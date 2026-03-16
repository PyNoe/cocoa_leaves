import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import laspy
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


OUT_DIR  = "/Users/noedaniel/Documents/Stage IRD/Scans_coco/separation"
RUN_NAME = "run_01"

MIN_PTS_STAT = 30   # ignorer les clusters trop petits pour le calcul SVD





run_dir     = os.path.join(OUT_DIR, RUN_NAME)
laz_path    = os.path.join(run_dir, "clustered.laz")
params_path = os.path.join(run_dir, "params.json")

if not os.path.exists(laz_path):
    raise FileNotFoundError(f"Fichier introuvable : {laz_path}\n"
                            f"Lance d'abord run_hdbscan.py avec RUN_NAME='{RUN_NAME}'")

# Chargement des paramètres du run pour affichage
if os.path.exists(params_path):
    with open(params_path) as f:
        params = json.load(f)
    hdbscan_params = params.get("hdbscan", {})
    param_str = (f"min_cluster_size={hdbscan_params.get('min_cluster_size')}  "
                 f"min_samples={hdbscan_params.get('min_samples')}  "
                 f"method={hdbscan_params.get('cluster_selection_method')}")
    run_date = params.get("date", "")
else:
    param_str = "paramètres inconnus"
    run_date  = ""

print(f"Chargement de {laz_path}...")
laz    = laspy.read(laz_path)
labels = np.array(laz.cluster_id, dtype=np.int32)
pts    = np.column_stack([
    np.array(laz.x, dtype=np.float64),
    np.array(laz.y, dtype=np.float64),
    np.array(laz.z, dtype=np.float64),
])

n_total    = len(pts)
n_clusters = int(len(np.unique(labels[labels >= 0])))
n_noise    = int((labels == -1).sum())
print(f"  {n_total:,} points  |  {n_clusters} clusters  |  {n_noise:,} bruit")
print(f"  Paramètres : {param_str}")




def cluster_geometry(cluster_pts):
    """
    SVD sur un cluster centré → indicateurs de forme.
    s[0] ≥ s[1] ≥ s[2] (valeurs singulières décroissantes)
      - grand axe   : s[0]  (direction principale, longueur)
      - axe médian  : s[1]  (dans le plan de la feuille)
      - hors-plan   : s[2]  (épaisseur, bruit de scan)
    """
    centered = cluster_pts - cluster_pts.mean(axis=0)
    _, s, _  = np.linalg.svd(centered, full_matrices=False)
    s0 = s[0] if s[0] > 1e-10 else 1e-10

    return {
        "n_pts":        len(cluster_pts),
        "extent_m":     float(2 * s[0] / np.sqrt(len(cluster_pts))),  # longueur approx.
        "aspect_ratio": float(s[1] / s0),           # 0 = bâtonnet, 1 = rond
        "planarity":    float((s[1] - s[2]) / s0),  # 0 = non plan, 1 = très plan
        "thickness":    float(s[2] / s0),            # épaisseur relative hors-plan
    }

print("\nCalcul des indicateurs géométriques...")
cluster_ids = np.unique(labels[labels >= 0])
records = []
for cid in cluster_ids:
    sub = pts[labels == cid]
    if len(sub) < MIN_PTS_STAT:
        # Cluster trop petit pour une SVD fiable : on garde juste n_pts
        records.append({"cluster_id": int(cid), "n_pts": len(sub),
                        "extent_m": None, "aspect_ratio": None,
                        "planarity": None, "thickness": None})
    else:
        geo = cluster_geometry(sub)
        geo["cluster_id"] = int(cid)
        records.append(geo)

df = pd.DataFrame(records).set_index("cluster_id")
df_valid = df.dropna()   # clusters avec SVD calculé

print(df_valid.describe().round(3))




Q1, Q3 = df["n_pts"].quantile(0.25), df["n_pts"].quantile(0.75)
IQR     = Q3 - Q1
low_thr  = Q1 - 1.5 * IQR
high_thr = Q3 + 1.5 * IQR

def flag_cluster(n):
    if n < low_thr:  return "trop_petit"
    if n > high_thr: return "trop_grand"
    return "normal"

df["flag"] = df["n_pts"].apply(flag_cluster)

n_small  = (df["flag"] == "trop_petit").sum()
n_large  = (df["flag"] == "trop_grand").sum()
n_normal = (df["flag"] == "normal").sum()

print(f"\nDétection IQR sur n_pts  [seuils : <{low_thr:.0f}  |  >{high_thr:.0f}]")
print(f"  normaux     : {n_normal}")
print(f"  trop petits : {n_small}  (fragments / pointes)")
print(f"  trop grands : {n_large}  (feuilles fusionnées ?)")

FLAG_COLORS = {"normal": "tab:blue", "trop_petit": "tab:orange", "trop_grand": "tab:red"}




csv_path = os.path.join(run_dir, "cluster_stats.csv")
df.reset_index().to_csv(csv_path, index=False, float_format="%.4f")
print(f"\nCSV sauvegardé : {csv_path}")




fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    f"Run '{RUN_NAME}'  |  {param_str}\n"
    f"{n_clusters} clusters  ({n_normal} normaux · {n_small} petits · {n_large} grands)"
    + (f"  —  {run_date}" if run_date else ""),
    fontsize=10,
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Histogramme n_pts avec zones flaggées
ax_npts = fig.add_subplot(gs[0, 0])
for flag, color in FLAG_COLORS.items():
    subset = df.loc[df["flag"] == flag, "n_pts"]
    if len(subset):
        ax_npts.hist(subset, bins=25, color=color, alpha=0.8,
                     edgecolor="white", linewidth=0.4, label=flag)
ax_npts.axvline(low_thr,  color="orange", linestyle="--", linewidth=1.2)
ax_npts.axvline(high_thr, color="red",    linestyle="--", linewidth=1.2)
ax_npts.axvline(df["n_pts"].median(), color="black", linestyle="--", linewidth=1.2,
                label=f"médiane = {df['n_pts'].median():.0f}")
ax_npts.set_title("Nb de points par cluster", fontsize=9)
ax_npts.set_xlabel("n_pts", fontsize=8)
ax_npts.set_ylabel("nb clusters", fontsize=8)
ax_npts.legend(fontsize=7)
ax_npts.tick_params(labelsize=7)

# Histogrammes des indicateurs SVD
hist_specs = [
    ("aspect_ratio", "Ratio d'aspect  (0=bâtonnet, 1=rond)", "tab:orange", gs[0, 1]),
    ("planarity",    "Planarité  (proche 1 = très plan)",     "tab:green",  gs[0, 2]),
    ("thickness",    "Épaisseur hors-plan  (proche 0 = plan)","tab:purple", gs[1, 0]),
]
for col, title, color, pos in hist_specs:
    ax = fig.add_subplot(pos)
    data = df_valid[col].dropna()
    ax.hist(data, bins=25, color=color, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(data.median(), color="black", linestyle="--", linewidth=1.2,
               label=f"médiane = {data.median():.3f}")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(col, fontsize=8)
    ax.set_ylabel("nb clusters", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

# Scatter : taille vs aspect_ratio, coloré par flag
ax_sc = fig.add_subplot(gs[1, 1])
for flag, color in FLAG_COLORS.items():
    sub = df[df["flag"] == flag].dropna(subset=["aspect_ratio"])
    ax_sc.scatter(sub["n_pts"], sub["aspect_ratio"],
                  c=color, s=20 if flag == "normal" else 40,
                  alpha=0.7, label=flag, zorder=3 if flag != "normal" else 2)
ax_sc.axvline(low_thr,  color="orange", linestyle="--", linewidth=0.8)
ax_sc.axvline(high_thr, color="red",    linestyle="--", linewidth=0.8)
ax_sc.set_xlabel("n_pts", fontsize=8)
ax_sc.set_ylabel("aspect_ratio", fontsize=8)
ax_sc.set_title("Taille vs forme", fontsize=9)
ax_sc.legend(fontsize=7)
ax_sc.tick_params(labelsize=7)

# Boxplot n_pts avec outliers colorés
ax_box = fig.add_subplot(gs[1, 2])
bp = ax_box.boxplot(df["n_pts"], patch_artist=True, vert=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
bp["boxes"][0].set_facecolor("lightsteelblue")
bp["boxes"][0].set_alpha(0.7)
for flag, color in [("trop_petit", "tab:orange"), ("trop_grand", "tab:red")]:
    vals = df.loc[df["flag"] == flag, "n_pts"]
    ax_box.scatter([1] * len(vals), vals, c=color, s=30, zorder=5,
                   label=f"{flag} ({len(vals)})")
ax_box.set_title("Distribution n_pts", fontsize=9)
ax_box.set_ylabel("n_pts", fontsize=8)
ax_box.legend(fontsize=7)
ax_box.tick_params(labelsize=7)

# Sauvegarde + affichage
fig_path = os.path.join(run_dir, "cluster_stats.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure sauvegardée : {fig_path}")
plt.show()
