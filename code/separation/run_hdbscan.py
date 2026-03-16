"""
Segmentation HDBSCAN — clustering des feuilles sur un nuage de points
======================================================================
Lit un fichier .laz, applique HDBSCAN et exporte :
  - un .laz coloré par cluster avec le champ extra cluster_id
  - params.json  : tous les paramètres utilisés (pour reproductibilité)
  - summary.txt  : rapport lisible

Paramètres
----------
    LAZ_FILE                  — fichier .laz d'entrée
    OUT_DIR                   — répertoire de sortie racine
    RUN_NAME                  — nom du sous-dossier  (OUT_DIR/RUN_NAME/)
    MIN_CLUSTER_SIZE          — taille minimale d'un cluster HDBSCAN
    MIN_SAMPLES               — min_samples HDBSCAN (régularisation du bruit)
    CLUSTER_SELECTION_METHOD  — 'leaf' (sépare plus) ou 'eom' (fusionne)
    CLUSTER_SELECTION_EPS     — distance min entre clusters (0 = désactivé)

Structure de sortie
-------------------
    OUT_DIR/RUN_NAME/
        clustered.laz   ← nuage coloré par cluster + champ cluster_id
        params.json     ← paramètres complets du run
        summary.txt     ← rapport statistique

    conda activate ird_tls
    python3 separation/run_hdbscan.py
"""

import sys, os, json, time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import laspy
import hdbscan
import matplotlib.cm as cm

LAZ_FILE = "/Users/noedaniel/Documents/Stage IRD/Scans_coco/Cocoa_4_Noe_region_small.laz"
OUT_DIR  = "/Users/noedaniel/Documents/Stage IRD/Scans_coco/separation"
RUN_NAME = "run_01"

MIN_CLUSTER_SIZE         = 200    # taille minimale d'un cluster (il faut modifier selon la densité de points du nuage)
MIN_SAMPLES              = 5      # plus grand = plus conservateur (moins de bruit classé cluster)
CLUSTER_SELECTION_METHOD = "leaf" # 'leaf' sépare davantage ; 'eom' fusionne les sous-clusters
CLUSTER_SELECTION_EPS    = 0.0    # distance min entre clusters (m) — 0 = désactivé



run_dir = os.path.join(OUT_DIR, RUN_NAME)
os.makedirs(run_dir, exist_ok=True)



print("Chargement du nuage de points...")
laz = laspy.read(LAZ_FILE)
pts = np.column_stack([
    np.array(laz.x, dtype=np.float64),
    np.array(laz.y, dtype=np.float64),
    np.array(laz.z, dtype=np.float64),
])
n_total = len(pts)
print(f"  {n_total:,} points")



print("\nHDBSCAN en cours...")
t0 = time.time()
labels = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    cluster_selection_method=CLUSTER_SELECTION_METHOD,
    cluster_selection_epsilon=CLUSTER_SELECTION_EPS,
    core_dist_n_jobs=-1,
).fit_predict(pts)
duration = time.time() - t0

n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
n_noise    = int((labels == -1).sum())
noise_pct  = 100 * n_noise / n_total

print(f"  terminé en {duration:.1f} s")
print(f"  {n_clusters} clusters  |  {n_noise:,} pts bruit ({noise_pct:.1f}%)")

# Statistiques sur les tailles de clusters
cluster_sizes = np.array([int((labels == c).sum()) for c in range(n_clusters)])
if len(cluster_sizes) > 0:
    print(f"  taille clusters — min {cluster_sizes.min()}  médiane {int(np.median(cluster_sizes))}"
          f"  max {cluster_sizes.max()}  moyenne {cluster_sizes.mean():.0f}")




# Palette tab20 cyclique, bruit en gris moyen
cmap   = cm.get_cmap("tab20")
_rgb   = np.array([cmap(i / 20)[:3] for i in range(20)])
colors = np.full((n_total, 3), 32768, dtype=np.uint16)   # gris = bruit
valid  = labels >= 0
colors[valid] = (_rgb[labels[valid] % 20] * 65535).astype(np.uint16)

header = laspy.LasHeader(point_format=2, version="1.2")
header.offsets = laz.header.offsets
header.scales  = laz.header.scales
header.add_extra_dim(laspy.ExtraBytesParams(name="cluster_id", type=np.int32))

out            = laspy.LasData(header=header)
out.x          = laz.x
out.y          = laz.y
out.z          = laz.z
out.red        = colors[:, 0]
out.green      = colors[:, 1]
out.blue       = colors[:, 2]
out.cluster_id = labels.astype(np.int32)

out_laz = os.path.join(run_dir, "clustered.laz")
out.write(out_laz)
print(f"\nLAZ sauvegardé : {out_laz}")





params = {
    "run_name":                 RUN_NAME,
    "date":                     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "laz_file":                 LAZ_FILE,
    "hdbscan": {
        "min_cluster_size":         MIN_CLUSTER_SIZE,
        "min_samples":              MIN_SAMPLES,
        "cluster_selection_method": CLUSTER_SELECTION_METHOD,
        "cluster_selection_eps":    CLUSTER_SELECTION_EPS,
    },
    "results": {
        "n_points":    n_total,
        "n_clusters":  n_clusters,
        "n_noise":     n_noise,
        "noise_pct":   round(noise_pct, 2),
        "duration_s":  round(duration, 1),
        "cluster_size_min":    int(cluster_sizes.min())    if len(cluster_sizes) else None,
        "cluster_size_median": int(np.median(cluster_sizes)) if len(cluster_sizes) else None,
        "cluster_size_max":    int(cluster_sizes.max())    if len(cluster_sizes) else None,
        "cluster_size_mean":   round(float(cluster_sizes.mean()), 1) if len(cluster_sizes) else None,
    }
}

params_path = os.path.join(run_dir, "params.json")
with open(params_path, "w") as f:
    json.dump(params, f, indent=2, ensure_ascii=False)
print(f"Paramètres sauvegardés : {params_path}")




x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
z_min, z_max = float(pts[:, 2].min()), float(pts[:, 2].max())

lines = [
    f"RUN : {RUN_NAME}",
    f"Date : {params['date']}",
    f"Entrée : {LAZ_FILE}",
    f"",
    f"=== PARAMÈTRES HDBSCAN ===",
    f"min_cluster_size         : {MIN_CLUSTER_SIZE}",
    f"min_samples              : {MIN_SAMPLES}",
    f"cluster_selection_method : {CLUSTER_SELECTION_METHOD}",
    f"cluster_selection_eps    : {CLUSTER_SELECTION_EPS}",
    f"",
    f"=== BOÎTE ENGLOBANTE ===",
    f"X : [{x_min:.3f}, {x_max:.3f}]  ({x_max - x_min:.3f} m)",
    f"Y : [{y_min:.3f}, {y_max:.3f}]  ({y_max - y_min:.3f} m)",
    f"Z : [{z_min:.3f}, {z_max:.3f}]  ({z_max - z_min:.3f} m)",
    f"",
    f"=== RÉSULTATS ===",
    f"Points total    : {n_total:,}",
    f"Bruit           : {n_noise:,}  ({noise_pct:.1f}%)",
    f"Clusters        : {n_clusters}",
    f"Durée           : {duration:.1f} s",
]
if len(cluster_sizes) > 0:
    lines += [
        f"",
        f"=== TAILLE DES CLUSTERS ===",
        f"Min             : {cluster_sizes.min()}",
        f"Médiane         : {int(np.median(cluster_sizes))}",
        f"Moyenne         : {cluster_sizes.mean():.1f}",
        f"Max             : {cluster_sizes.max()}",
    ]

summary_path = os.path.join(run_dir, "summary.txt")
with open(summary_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Résumé sauvegardé : {summary_path}")
