"""
BPA en batch
==============================================================================================
Lit un fichier .laz avec un champ cluster_id, exécute l'algorithme BPA
sur chaque cluster et écrit un .laz par cluster ainsi qu'un résumé.

Paramètres
------------------------------------------------
    LAZ_FILE     — fichier .laz d'entrée avec le champ cluster_id
    OUT_DIR      — répertoire de sortie racine
    RUN_NAME     — nom du sous-dossier de sortie  (OUT_DIR/RUN_NAME/)
    IDS_FILE     — fichier .txt optionnel avec un identifiant de cluster par ligne ;
                   laisser None pour traiter tous les clusters >= MIN_PTS
    MIN_PTS      — ignorer les clusters plus petits (ignoré si IDS_FILE est défini)
    N_JOBS       — parallélisme  (-1 = tous les cœurs)

Structure de sortie
-------------------
    OUT_DIR/RUN_NAME/
        original/           ← copie du fichier .laz d'entrée
        clusters/
            cluster_0001.laz
            cluster_0002.laz
            ...
        summary.csv         ← cluster_id, n_pts, surface_m2, surface_cm2
        summary.txt         ← rapport lisible par l'humain
"""

import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import laspy
from joblib import Parallel, delayed
import bpa_core as BPA

LAZ_FILE = "/Users/noedaniel/Documents/Stage IRD/Scans_coco/hdbscan_coco.laz"
OUT_DIR  = "/Users/noedaniel/Documents/Stage IRD/Scans_coco/surface"
RUN_NAME = "run_01"

IDS_FILE = None   # "/chemin/vers/ids.txt", ou None pour tous les clusters >= MIN_PTS
MIN_PTS  = 500    # taille minimale d'un cluster (ignoré si IDS_FILE est défini)
N_JOBS   = -1     # nombre de workers parallèles (-1 = tous les cœurs CPU)

COLOR_USED     = (65535, 27000,     0)   # orange  — points utilisés dans le maillage
COLOR_UNUSED   = (65535, 65535, 65535)   # blanc   — points hors maillage
COLOR_BOUNDARY = (    0, 65535,     0)   # vert    — points du contour frontière

run_dir      = os.path.join(OUT_DIR, RUN_NAME)
dir_original = os.path.join(run_dir, "original")
dir_clusters = os.path.join(run_dir, "clusters")

for d in [dir_original, dir_clusters]:
    if os.path.exists(d):
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
    else:
        os.makedirs(d)

# ─────────────────────────────────────────────────────────────

print("Chargement du nuage de points...")
laz    = laspy.read(LAZ_FILE)
labels = np.array(laz.cluster_id, dtype=np.int32)
pts    = np.column_stack([
    np.array(laz.x, dtype=np.float64),
    np.array(laz.y, dtype=np.float64),
    np.array(laz.z, dtype=np.float64),
])

n_total    = len(pts)
n_noise    = (labels == -1).sum()
n_clusters = len(np.unique(labels[labels >= 0]))
print(f"  {n_total:,} points  |  {n_clusters} clusters  |  "
      f"{n_noise:,} bruit ({100*n_noise/n_total:.1f}%)")

# Copie du fichier original pour référence
shutil.copy2(LAZ_FILE, os.path.join(dir_original, os.path.basename(LAZ_FILE)))

# ─────────────────────────────────────────────────────────────
# SÉLECTION DES CLUSTERS
# ─────────────────────────────────────────────────────────────

if IDS_FILE is not None:
    with open(IDS_FILE) as f:
        selected = [int(line.strip()) for line in f if line.strip()]
    print(f"\n{len(selected)} clusters depuis {IDS_FILE}")
else:
    cluster_ids = np.unique(labels[labels >= 0])
    selected    = [int(c) for c in cluster_ids if (labels == c).sum() >= MIN_PTS]
    print(f"\n{len(selected)} clusters >= {MIN_PTS} pts  (sur {n_clusters} au total)\n")



def process_cluster(rank, cid):
    mask    = labels == cid
    sub_pts = pts[mask]
    sub_idx = np.where(mask)[0]
    n       = len(sub_pts)

    mesh = BPA.bpa_mesh(sub_pts)

    if mesh is None:
        used_local   = np.array([], dtype=int)
        boundary_pts = None
        surface      = None
        status       = "BPA échoué"
    else:
        used_local   = np.unique(np.asarray(mesh.triangles))
        boundary_pts = BPA.mesh_boundary(mesh)
        surface      = mesh.get_surface_area()
        radius       = BPA.median_nn_distance(sub_pts)
        n_bd         = len(boundary_pts) if boundary_pts is not None else 0
        status       = (f"rayon {radius*1000:.2f} mm  |  "
                        f"{len(used_local)}/{n} pts utilisés ({100*len(used_local)/n:.1f}%)  |  "
                        f"frontière {n_bd} pts  |  surface {surface*1e4:.2f} cm²")

    # Attribution des couleurs
    r = np.full(n, COLOR_UNUSED[0], dtype=np.uint16)
    g = np.full(n, COLOR_UNUSED[1], dtype=np.uint16)
    b = np.full(n, COLOR_UNUSED[2], dtype=np.uint16)
    r[used_local] = COLOR_USED[0]
    g[used_local] = COLOR_USED[1]
    b[used_local] = COLOR_USED[2]

    # Ajout des points du contour frontière
    centroid = sub_pts.mean(axis=0)
    x_out = list(laz.x[sub_idx]);  y_out = list(laz.y[sub_idx]);  z_out = list(laz.z[sub_idx])
    r_out = list(r);               g_out = list(g);               b_out = list(b)

    if boundary_pts is not None and len(boundary_pts) > 0:
        bd_abs = boundary_pts + centroid   # retour en coordonnées absolues
        x_out += list(bd_abs[:, 0]);  y_out += list(bd_abs[:, 1]);  z_out += list(bd_abs[:, 2])
        r_out += [COLOR_BOUNDARY[0]] * len(bd_abs)
        g_out += [COLOR_BOUNDARY[1]] * len(bd_abs)
        b_out += [COLOR_BOUNDARY[2]] * len(bd_abs)

    # Écriture du .laz
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.offsets = laz.header.offsets
    header.scales  = laz.header.scales

    out       = laspy.LasData(header=header)
    out.x     = np.array(x_out)
    out.y     = np.array(y_out)
    out.z     = np.array(z_out)
    out.red   = np.array(r_out, dtype=np.uint16)
    out.green = np.array(g_out, dtype=np.uint16)
    out.blue  = np.array(b_out, dtype=np.uint16)
    out.write(os.path.join(dir_clusters, f"cluster_{cid:04d}.laz"))

    log = f"[{rank+1:3d}/{len(selected)}] cluster {cid:4d}  ({n:>6,} pts)  —  {status}"
    return cid, n, surface, log


results = Parallel(n_jobs=N_JOBS, prefer="threads")(
    delayed(process_cluster)(rank, cid)
    for rank, cid in enumerate(selected)
)

for _, _, _, log in results:
    print(log)

# ─────────────────────────────────────────────────────────────
# résumé

x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
z_min, z_max = float(pts[:, 2].min()), float(pts[:, 2].max())

surfaces = [s for _, _, s, _ in results if s is not None]
n_ok     = len(surfaces)
n_fail   = len(results) - n_ok
total_m2 = sum(surfaces)

# — summary.csv —
csv_path = os.path.join(run_dir, "summary.csv")
with open(csv_path, "w") as f:
    f.write("cluster_id,n_pts,surface_m2,surface_cm2\n")
    for cid, n_pts, surface, _ in sorted(results, key=lambda r: r[0]):
        if surface is not None:
            f.write(f"{cid},{n_pts},{surface:.8f},{surface*1e4:.4f}\n")
        else:
            f.write(f"{cid},{n_pts},,\n")

# — summary.txt —
txt_lines = [
    f"RUN: {RUN_NAME}",
    f"Entrée: {LAZ_FILE}",
    f"",
    f"=== BOÎTE ENGLOBANTE ===",
    f"X : [{x_min:.3f}, {x_max:.3f}]  ({x_max - x_min:.3f} m)",
    f"Y : [{y_min:.3f}, {y_max:.3f}]  ({y_max - y_min:.3f} m)",
    f"Z : [{z_min:.3f}, {z_max:.3f}]  ({z_max - z_min:.3f} m)",
    f"",
    f"=== POINTS ===",
    f"Total          : {n_total:,}",
    f"Bruit          : {n_noise:,} ({100 * n_noise / n_total:.1f}%)",
    f"Clusters total : {n_clusters}",
    f"Analysés       : {len(selected)}",
    f"BPA réussi     : {n_ok}",
    f"BPA échoué     : {n_fail}",
    f"",
    f"=== SURFACE FOLIAIRE ===",
    f"Total  : {total_m2 * 1e4:.2f} cm²  ({total_m2:.6f} m²)",
]
if n_ok > 0:
    txt_lines.append(f"Moyenne : {total_m2 / n_ok * 1e4:.2f} cm² par cluster")
txt_lines += [
    f"",
    f"=== DÉTAILS PAR CLUSTER ===",
    f"{'cluster_id':>12}  {'n_pts':>8}  {'surface_cm2':>12}",
    f"{'-' * 38}",
]
for cid, n_pts, surface, _ in sorted(results, key=lambda r: r[0]):
    surf_str = f"{surface * 1e4:.2f}" if surface is not None else "ÉCHOUÉ"
    txt_lines.append(f"{cid:>12}  {n_pts:>8,}  {surf_str:>12}")

txt_path = os.path.join(run_dir, "summary.txt")
with open(txt_path, "w") as f:
    f.write("\n".join(txt_lines) + "\n")

print(f"\nTerminé.")
print(f"  {n_ok} feuilles  |  surface totale {total_m2 * 1e4:.2f} cm²")
print(f"  Résumé   : {txt_path}")
print(f"  CSV      : {csv_path}")
print(f"  Clusters : {dir_clusters}/")
