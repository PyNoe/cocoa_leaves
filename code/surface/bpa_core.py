"""
Algorithme contenant la structure de base pour l'algo BPA.

Fonctions mises en place :
----------
    mesh    = bpa_mesh(pts_cluster)       → open3d TriangleMesh or None
    contour = mesh_boundary(mesh)         → (M, 3) outer boundary or None
    area    = bpa_surface(pts_cluster)    → surface area (m²) or None
    r       = median_nn_distance(pts)     → local density estimate
"""

import numpy as np
import open3d as o3d
from collections import defaultdict
from scipy.spatial import cKDTree

NORMAL_RADIUS      = 0.05              # rayon pour l'estimation normale (m)
NORMAL_MAX_NN      = 30               # noombre max de voisins pour l'estimation normale 
BPA_RADIUS_FACTORS = [1.0, 2.0, 4.0]  
MIN_PTS_BPA        = 20               # nombre de points minimum par composante pour lancer une BPA

def median_nn_distance(pts_cluster: np.ndarray, k: int = 5) -> float:
    """Mediane du mean k-NN distances → densité locale de points"""
    pts_local = pts_cluster - pts_cluster.mean(axis=0)
    dists, _ = cKDTree(pts_local).query(pts_local, k=k + 1, workers=-1)
    return float(np.median(dists[:, 1:].mean(axis=1)))


def bpa_mesh(pts_cluster: np.ndarray):
    """
    Pour créer le maillage sur un cluster de points.

    Stratégie multi-échelle (BPA_RADIUS_FACTORS) :
      - petit rayon : triangulation des régions denses (principalement le centre)
      - grand rayon : comble les trous et zones moins denses (principalement les bords)
    Open3D applique les rayons en séquence sans re-trianguler les triangles déjà couverts,
    ce qui permet d'obtenir un maillage plus complet qu'avec un seul rayon.

    Retourne le TriangleMesh open3d, ou None en cas d'échec.
    """
    if len(pts_cluster) < MIN_PTS_BPA:
        return None

    # Centrage local --> évite les erreurs de précision sur les coordonnées absolues
    pts_local = pts_cluster - pts_cluster.mean(axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_local)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN
        )
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(10)
    except RuntimeError:
        return None

    r0    = median_nn_distance(pts_cluster)
    radii = [r0 * f for f in BPA_RADIUS_FACTORS]

    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    except RuntimeError:
        return None

    return mesh if len(mesh.triangles) > 0 else None


def mesh_boundary(mesh) -> np.ndarray | None:
    """
    Extraire la frontière extérieure d'un maillage BPA.

    Une arête de frontière appartient à exactement un triangle.
    L'ensemble des arêtes de frontière forme le(s) contour(s) externe(s).
    Retourne la chaîne la plus longue sous forme d'un tableau fermé (M, 3), ou None.
    """
    triangles = np.asarray(mesh.triangles)
    vertices  = np.asarray(mesh.vertices)

    # On compte les occurrences de chaque arête
    edge_count = defaultdict(int)
    for tri in triangles:
        for i in range(3):
            e = (int(min(tri[i], tri[(i+1) % 3])),
                 int(max(tri[i], tri[(i+1) % 3])))
            edge_count[e] += 1

    # Les arêtes de frontière apparaissent une seule fois
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    if not boundary_edges:
        return None

    # Graphe d'adjacence sur les sommets de frontière
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    # Walk closed chains
    visited = set()
    chains  = []

    for start in list(adj.keys()):
        if start in visited:
            continue
        chain   = [start]
        visited.add(start)
        current = start

        while True:
            nxt = [n for n in adj[current] if n not in visited]
            if not nxt:
                break
            current = nxt[0]
            visited.add(current)
            chain.append(current)

        if len(chain) >= 3:
            chains.append(chain)

    if not chains:
        return None

    # On retient la chaîne la plus longue (en cas de contours multiples, on suppose que le plus grand est le contour externe)
    longest = max(chains, key=len)
    pts_bd  = vertices[longest]
    return np.vstack([pts_bd, pts_bd[0]])


def bpa_surface(pts_cluster: np.ndarray) -> float | None:
    """Executer le BPA et retourner l'aire de surface (m²), ou None en cas d'échec."""
    mesh = bpa_mesh(pts_cluster)
    return mesh.get_surface_area() if mesh is not None else None
