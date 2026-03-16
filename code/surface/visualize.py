"""
Inspection visuelle d'un cluster — BPA 3D + MIP face-on

J'ai mis les trois méthodes :
- BPA (celle qui me semble la plus sure)
- Hull convex
- MIP & Delaunay (Guo et al.)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import laspy
import open3d as o3d
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, Delaunay, cKDTree
import bpa_core as BPA

LAZ_FILE   = "/Users/noedaniel/Documents/Stage IRD/Scans_coco/hdbscan_coco.laz"
CLUSTER_ID = 200

MIP_LAMBDA   = 0.01   # rayon pour l'estimation des normales MIP (m)
NORMALS_STEP = 20     # afficher 1 normale sur N points (Figure 3)

laz    = laspy.read(LAZ_FILE)
labels = np.array(laz.cluster_id, dtype=np.int32)
pts    = np.column_stack([laz.x, laz.y, laz.z])

mask  = labels == CLUSTER_ID
pts_c = pts[mask]

if len(pts_c) == 0:
    raise ValueError(f"Cluster {CLUSTER_ID} not found. "
                     f"Available: {np.unique(labels[labels >= 0])[:20]}")

centroid  = pts_c.mean(axis=0)
pts_local = pts_c - centroid

print(f"Cluster {CLUSTER_ID}  —  {len(pts_c):,} points")

# ─────────────────────────────────────────────────────────────
# BPA
# ─────────────────────────────────────────────────────────────

print("\n--- BPA ---")
mesh_bpa = BPA.bpa_mesh(pts_c)

if mesh_bpa is None:
    print("  BPA failed.")
    bpa_tris  = None
    bpa_verts = None
    bpa_area  = None
    used_mask = np.zeros(len(pts_c), dtype=bool)
else:
    bpa_tris  = np.asarray(mesh_bpa.triangles)
    bpa_verts = np.asarray(mesh_bpa.vertices)
    bpa_area  = mesh_bpa.get_surface_area()
    used_idx  = np.unique(bpa_tris)
    used_mask = np.zeros(len(pts_c), dtype=bool)
    used_mask[used_idx] = True
    print(f"  {len(bpa_tris):,} triangles  |  "
          f"{used_mask.sum():,}/{len(pts_c):,} pts used ({100*used_mask.mean():.1f}%)")
    print(f"  Surface : {bpa_area * 1e4:.2f} cm²")

# ─────────────────────────────────────────────────────────────
# 3. MIP  (sections E–F)
#
# E.1  normale par point via SVD sur le voisinage de rayon MIP_LAMBDA
# E.2  rayon d'inflation r_i = distance au plus proche voisin
# E.3  aire Delaunay filtrée sur XY  → S_projection
# F    normale moyenne pondérée → zénith θ̄ → S_feuille = S_proj / cos(θ̄)
# ─────────────────────────────────────────────────────────────

print("\n--- MIP ---")

tree        = cKDTree(pts_local)
normals_mip = np.zeros((len(pts_local), 3))
r_inflate   = np.zeros(len(pts_local))

for i, p in enumerate(pts_local):
    idx = tree.query_ball_point(p, MIP_LAMBDA)
    if len(idx) < 3:
        _, idx = tree.query(p, k=min(6, len(pts_local)))

    neighbors = pts_local[idx]
    centered  = neighbors - neighbors.mean(axis=0)
    _, _, Vt  = np.linalg.svd(centered, full_matrices=False)
    n         = Vt[-1]
    if n[2] < 0:
        n = -n
    normals_mip[i] = n

    dists, _ = tree.query(p, k=2)
    r_inflate[i] = dists[1]

# ─────────────────────────────────────────────────────────────
# Aire Delaunay avec filtrage des longues arêtes
# ─────────────────────────────────────────────────────────────

def delaunay_area(proj2d, max_edge):
    """Aire Delaunay avec filtrage des arêtes longues pour ignorer les zones vides."""
    tri   = Delaunay(proj2d)
    valid = []
    area  = 0.0
    for i, s in enumerate(tri.simplices):
        p0, p1, p2 = proj2d[s]
        if (np.linalg.norm(p0 - p1) > max_edge or
            np.linalg.norm(p1 - p2) > max_edge or
            np.linalg.norm(p0 - p2) > max_edge):
            continue
        area += 0.5 * abs(np.cross(p1 - p0, p2 - p0))
        valid.append(i)
    return area, tri, valid

max_edge = np.median(r_inflate) * 3

# S_projection : Delaunay filtré sur la projection XY
s_proj, tri_xy, valid_xy = delaunay_area(pts_local[:, :2], max_edge)

# Normale moyenne pondérée (poids = r_i²)
weights     = r_inflate ** 2
normal_mean = (normals_mip * weights[:, None]).sum(axis=0) / weights.sum()
normal_mean /= np.linalg.norm(normal_mean)

cos_theta         = abs(normal_mean[2])
theta_deg         = np.degrees(np.arccos(cos_theta))
cos_theta_clamped = max(cos_theta, 0.1)
mip_area          = s_proj / cos_theta_clamped
cos_per_pt        = np.abs(normals_mip[:, 2])

print(f"  Mean normal : ({normal_mean[0]:+.3f}, {normal_mean[1]:+.3f}, {normal_mean[2]:+.3f})")
print(f"  Zenith θ̄   : {theta_deg:.1f}°  →  cos(θ̄) = {cos_theta:.3f}")
print(f"  S_projection: {s_proj * 1e4:.2f} cm²")
print(f"  MIP surface : {mip_area * 1e4:.2f} cm²")

# ─────────────────────────────────────────────────────────────
# ENVELOPPE CONVEXE DANS LE PLAN PCA (référence)
# ─────────────────────────────────────────────────────────────

_, _, Vt_pca  = np.linalg.svd(pts_local, full_matrices=False)
proj_pca      = pts_local @ Vt_pca[:2].T
hull_pca      = ConvexHull(proj_pca)
hull_pca_area = hull_pca.volume

hull_pca_pts = np.vstack([proj_pca[hull_pca.vertices], proj_pca[hull_pca.vertices[0]]])
hull_pca_3d  = hull_pca_pts @ Vt_pca[:2]   # coords 3D centrées, pour la Figure 1

# ─────────────────────────────────────────────────────────────
# PROJECTION FACE-ON  (plan perp. à normale moyenne)
# ─────────────────────────────────────────────────────────────

# Base orthonormée tangente au plan de la feuille
arbitrary = np.array([0., 0., 1.]) if abs(normal_mean[0]) < 0.9 else np.array([0., 1., 0.])
u = np.cross(normal_mean, arbitrary); u /= np.linalg.norm(u)
v = np.cross(normal_mean, u);         v /= np.linalg.norm(v)

proj_face = pts_local @ np.column_stack([u, v])
hull_face_area, tri_face, valid_face = delaunay_area(proj_face, max_edge)

# Ajustement ellipse à partir du rapport d'aspect de l'enveloppe convexe
hull_face_cv  = ConvexHull(proj_face)
hull_verts_2d = proj_face[hull_face_cv.vertices]
eigvals, eigvecs = np.linalg.eigh(np.cov(proj_face.T))
order    = np.argsort(eigvals)[::-1]
eigvecs  = eigvecs[:, order]
# Demi-axes = moitié de l'étendue de l'enveloppe le long de chaque axe principal
proj_hull_on_axes = hull_verts_2d @ eigvecs
a = float(proj_hull_on_axes[:, 0].max() - proj_hull_on_axes[:, 0].min()) / 2
b = float(proj_hull_on_axes[:, 1].max() - proj_hull_on_axes[:, 1].min()) / 2
ellipse_area = float(np.pi * a * b)

center_face = proj_face.mean(axis=0)
t           = np.linspace(0, 2 * np.pi, 300)
ellipse_pts = (center_face
               + np.outer(a * np.cos(t), eigvecs[:, 0])
               + np.outer(b * np.sin(t), eigvecs[:, 1]))

print(f"\n  Face-on Delaunay : {hull_face_area * 1e4:.2f} cm²")
print(f"  Ellipse fit      : {ellipse_area * 1e4:.2f} cm²  "
      f"(a={a*100:.1f} cm, b={b*100:.1f} cm, ratio={a/b:.2f})")

print(f"\n--- Summary ---")
print(f"  {'Method':<26}  {'Surface (cm²)':>14}  {'/ hull PCA':>10}")
print(f"  {'-'*54}")
for name, area in [
    ("BPA",                   bpa_area),
    ("MIP",                   mip_area),
    ("Delaunay XY (MIP)",     s_proj),
    ("Delaunay face-on",      hull_face_area),
    ("Ellipse fit",           ellipse_area),
    ("Convex hull PCA",       hull_pca_area),
]:
    if area is not None:
        print(f"  {name:<26}  {area*1e4:>14.2f}  {area/hull_pca_area:>10.3f}")
    else:
        print(f"  {name:<26}  {'failed':>14}")

# Pour les figures...

def draw_wireframe(ax, tris, verts, xi, yi, color, alpha=0.25, lw=0.3):
    edges = []
    for tri in tris:
        for k in range(3):
            ia, ib = tri[k], tri[(k + 1) % 3]
            edges.append([verts[ia, [xi, yi]], verts[ib, [xi, yi]]])
    ax.add_collection(LineCollection(edges, linewidths=lw, colors=color, alpha=alpha))

projections = [
    (0, 1, "X (m)", "Y (m)", "XY"),
    (0, 2, "X (m)", "Z (m)", "XZ"),
    (1, 2, "Y (m)", "Z (m)", "YZ"),
]

# ─────────────────────────────────────────────────────────────
# FIGURE 1 — BPA (3 projections 2D)
# ─────────────────────────────────────────────────────────────

fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
for ax, (xi, yi, xl, yl, title) in zip(axes1, projections):
    ax.scatter(pts_local[~used_mask, xi], pts_local[~used_mask, yi],
               s=1, c="#cccccc", alpha=0.4, label=f"non utilisés ({(~used_mask).sum():,})")
    ax.scatter(pts_local[used_mask, xi], pts_local[used_mask, yi],
               s=1, c="darkorange", alpha=0.6, label=f"utilisés ({used_mask.sum():,})")
    if bpa_tris is not None:
        draw_wireframe(ax, bpa_tris, bpa_verts, xi, yi, color="darkorange")
    ax.plot(hull_pca_3d[:, xi], hull_pca_3d[:, yi],
            color="tomato", lw=1.2, ls="--", label="enveloppe convexe (PCA)")
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
    ax.set_aspect("equal"); ax.legend(fontsize=7, markerscale=4)

t1 = f"BPA  —  cluster {CLUSTER_ID}  ({len(pts_c):,} pts)"
if bpa_area:
    t1 += f"  |  {bpa_area*1e4:.1f} cm²  (hull PCA {hull_pca_area*1e4:.1f} cm²)"
fig1.suptitle(t1, fontsize=11)
plt.tight_layout()

# ─────────────────────────────────────────────────────────────
# FIGURE 2 — MIP  (XY + face-on)
# ─────────────────────────────────────────────────────────────

fig2, (ax_xy, ax_face) = plt.subplots(1, 2, figsize=(12, 5))

# Gauche : projection XY
sc = ax_xy.scatter(pts_local[:, 0], pts_local[:, 1],
                   s=2, c=cos_per_pt, cmap="RdYlGn", vmin=0, vmax=1, alpha=0.7)
edges_xy = []
for i in valid_xy:
    s = tri_xy.simplices[i]
    for k in range(3):
        p0, p1 = pts_local[s[k], :2], pts_local[s[(k+1)%3], :2]
        edges_xy.append([p0, p1])
ax_xy.add_collection(LineCollection(edges_xy, linewidths=0.5, colors="black", alpha=0.4,
                                    label=f"Delaunay  {s_proj*1e4:.1f} cm²"))
scale_arrow = max(pts_local[:, 0].max() - pts_local[:, 0].min(),
                  pts_local[:, 1].max() - pts_local[:, 1].min()) * 0.3
ax_xy.annotate("", xy=(normal_mean[0]*scale_arrow, normal_mean[1]*scale_arrow),
               xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="navy", lw=2))
ax_xy.scatter([0], [0], s=30, c="navy", zorder=5)
ax_xy.text(normal_mean[0]*scale_arrow*1.15, normal_mean[1]*scale_arrow*1.15,
           f"N̄  θ̄={theta_deg:.0f}°", fontsize=8, color="navy", ha="center")
plt.colorbar(sc, ax=ax_xy, label="cos(θᵢ)  [1=horiz, 0=vert]", shrink=0.8)
ax_xy.set_xlabel("X (m)"); ax_xy.set_ylabel("Y (m)")
ax_xy.set_title(f"Projection XY  —  Delaunay {s_proj*1e4:.1f} cm²  (seuil {max_edge*1e3:.1f} mm)\n"
                f"→ MIP corrigé : {mip_area*1e4:.1f} cm²  (÷ cos {cos_theta:.2f})")
ax_xy.set_aspect("equal"); ax_xy.legend(fontsize=8)

# Droite : projection face-on (plan de la feuille)
ax_face.scatter(proj_face[:, 0], proj_face[:, 1],
                s=2, c=cos_per_pt, cmap="RdYlGn", vmin=0, vmax=1, alpha=0.7)
edges_face = []
for i in valid_face:
    s = tri_face.simplices[i]
    for k in range(3):
        p0, p1 = proj_face[s[k]], proj_face[s[(k+1)%3]]
        edges_face.append([p0, p1])
ax_face.add_collection(LineCollection(edges_face, linewidths=0.5, colors="steelblue", alpha=0.4,
                                      label=f"Delaunay  {hull_face_area*1e4:.1f} cm²"))
ax_face.plot(ellipse_pts[:, 0], ellipse_pts[:, 1],
             color="crimson", lw=1.5,
             label=f"ellipse  {ellipse_area*1e4:.1f} cm²  (a={a*100:.1f} b={b*100:.1f} cm)")
ax_face.set_xlabel("u (m)"); ax_face.set_ylabel("v (m)")
ax_face.set_title(f"Projection face à la feuille  (plan ⊥ N̄)\n"
                  f"Delaunay = {hull_face_area*1e4:.1f} cm²  |  "
                  f"ellipse = {ellipse_area*1e4:.1f} cm²  (ratio {a/b:.2f})")
ax_face.set_aspect("equal"); ax_face.legend(fontsize=8)

fig2.suptitle(
    (f"MIP  —  cluster {CLUSTER_ID}  ({len(pts_c):,} pts)  |  "
     f"S_MIP = {mip_area*1e4:.1f} cm²   S_face = {hull_face_area*1e4:.1f} cm²"
     + (f"   BPA = {bpa_area*1e4:.1f} cm²" if bpa_area else "")),
    fontsize=10
)
plt.tight_layout()

# ─────────────────────────────────────────────────────────────
# FIGURE 3 — BPA 3D  (triangles + normales)
#
# Vert clair  = triangles du maillage BPA
# Vert foncé  = points utilisés dans le maillage
# Rouge       = points non utilisés (hors maillage)
# Bleu        = normales estimées (sous-échantillonnées toutes les NORMALS_STEP)
# ─────────────────────────────────────────────────────────────

fig3 = plt.figure(figsize=(9, 8))
ax3d = fig3.add_subplot(111, projection="3d")

if mesh_bpa is not None:
    # Faces du maillage BPA
    tri_faces = [bpa_verts[t] for t in bpa_tris]
    poly = Poly3DCollection(tri_faces, alpha=0.25,
                            facecolor="limegreen", edgecolor="darkgreen", linewidth=0.2)
    ax3d.add_collection3d(poly)

# Points non utilisés → rouge
ax3d.scatter(pts_local[~used_mask, 0], pts_local[~used_mask, 1], pts_local[~used_mask, 2],
             s=2, c="tomato", alpha=0.5, label=f"non utilisés ({(~used_mask).sum():,})")

# Points utilisés → vert (petit marqueur pour ne pas surcharger)
ax3d.scatter(pts_local[used_mask, 0], pts_local[used_mask, 1], pts_local[used_mask, 2],
             s=1, c="limegreen", alpha=0.4, label=f"utilisés ({used_mask.sum():,})")

# Normales estimées via Open3D (sous-échantillonnées)
pcd_n = o3d.geometry.PointCloud()
pcd_n.points = o3d.utility.Vector3dVector(pts_local)
pcd_n.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=BPA.NORMAL_RADIUS, max_nn=BPA.NORMAL_MAX_NN
    )
)
normals = np.asarray(pcd_n.normals)[::NORMALS_STEP]
origins = pts_local[::NORMALS_STEP]
scale_n = float(BPA.median_nn_distance(pts_c)) * 3

ax3d.quiver(origins[:, 0], origins[:, 1], origins[:, 2],
            normals[:, 0] * scale_n, normals[:, 1] * scale_n, normals[:, 2] * scale_n,
            color="steelblue", linewidth=0.5, alpha=0.6, label="normales")

ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
ax3d.legend(fontsize=8, markerscale=4)

title3 = f"BPA 3D  —  cluster {CLUSTER_ID}  ({len(pts_c):,} pts)"
if bpa_area:
    title3 += (f"\n{len(bpa_tris):,} triangles  |  {bpa_area*1e4:.1f} cm²  |  "
               f"{100*used_mask.mean():.1f}% pts utilisés")
fig3.suptitle(title3, fontsize=10)
plt.tight_layout()

plt.show()
