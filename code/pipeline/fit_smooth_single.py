"""
Test mono-cluster du pipeline fit_smooth_all.py


    1. charge un seul cluster depuis clustered.laz
    2. applique la meme pipeline que fit_smooth_all.py :
       - Poisson
       - elagage par distance
       - comblage optionnel des petits trous interieurs
       - subdivision loop
       - lissage Laplacien + Taubin
       - restauration stricte des bords
    3. exporte les sorties pour comparer :
       - le nuage de points brut du cluster
       - la mesh Poisson
       - la mesh smooth
    4. peut aussi refaire exactement la meme chaine sur un nuage voxel-subsample
       pour comparer l'effet d'une densite plus regulière
"""

import contextlib
import os
from collections import defaultdict

import laspy
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

CLUSTERED_LAZ = (
    "/Users/noedaniel/Documents/Stage IRD/resultats/"
    "box_cx790614p704_cy487882p981_cz571p105_w1p000_h1p000_d1p000/"
    "runs/cluster_area_run_01/clustered.laz"
)
CLUSTER_ID = 24  #choisir l'ID du cluster.
OUTPUT_DIR = (
    "/Users/noedaniel/Documents/Stage IRD/resultats/"
    "box_cx790614p704_cy487882p981_cz571p105_w1p000_h1p000_d1p000/"
    "runs/cluster_area_run_01/single_cluster_fill"
)

MIN_PTS_CLUSTER = 20

# Paramètres pour Poisson
POISSON_DEPTH = 6  # le mieux, ne pas bassier sinon très mauvais rendu...
MAX_DIST_FACTOR = 2.0  # ce qui me semble le plus opti.
# Estimation des normales pour Poisson.
NORMAL_RADIUS = 0.02
NORMAL_MAX_NN = 100
ORIENT_K = 10

# Subdivision Loop (pour avoir une meilleure résolution dans Poisson)
SUBDIV_ITER = 2

# Laplacien : nb d'itérations de lissage (plus on lisse, mieux c'est mais plus on réduit la surface...)
LAPLACIAN_ITER = 100

# Taubin (avec extérieur préféré)
TAUBIN_ITER = 40
TAUBIN_LAMBDA = 0.5
TAUBIN_MU = -0.55

#-------------------

# Lissage leger du contour apres restauration du bord
# Rebouchage des trous interieurs Poisson
FILL_POISSON_HOLES = True
MAX_HOLE_AREA_CM2 = 15.0

#-------------------
# La c'est si on veut downsampler pour essayer de moins avoir de trous mais dcp risque de perte de précision sur les bords.

# Subsampling spatial optionnel pour comparer une densite regularisee
COMPARE_VOXEL_DOWNSAMPLE = True
VOXEL_SIZE_POISSON = None          # si None -> VOXEL_SIZE_FACTOR * median_nn
VOXEL_SIZE_FACTOR = 1.25

COLOR_INTERIOR = np.array([0.20, 0.50, 1.00], dtype=np.float64)
COLOR_ALL_BOUNDARIES = np.array([1.00, 0.20, 0.20], dtype=np.float64)


def _configure_open3d_logging():
    try:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    except Exception:
        pass


@contextlib.contextmanager
def _suppress_open3d_output():
    sys_stdout = os.dup(1)
    sys_stderr = os.dup(2)
    devnull = open(os.devnull, "w")
    os.dup2(devnull.fileno(), 1)
    os.dup2(devnull.fileno(), 2)
    try:
        yield
    finally:
        os.dup2(sys_stdout, 1)
        os.dup2(sys_stderr, 2)
        os.close(sys_stdout)
        os.close(sys_stderr)
        devnull.close()


# ------------------------------------------------------------
# HELPERS repris de fit_smooth_all.py
# ------------------------------------------------------------

def _boundary_loops(mesh):
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        return []
    edge_count = defaultdict(int)
    for tri in triangles:
        for i in range(3):
            edge = (int(min(tri[i], tri[(i + 1) % 3])), int(max(tri[i], tri[(i + 1) % 3])))
            edge_count[edge] += 1
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    if not boundary_edges:
        return []
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)
    visited, loops = set(), []
    for start in list(adj.keys()):
        if start in visited:
            continue
        loop, current = [start], start
        visited.add(start)
        while True:
            nxt = [n for n in adj[current] if n not in visited]
            if not nxt:
                break
            current = nxt[0]
            visited.add(current)
            loop.append(current)
        if len(loop) >= 3:
            loops.append(loop)
    loops.sort(key=len, reverse=True)
    return loops


def _loop_area(loop, verts):
    pts = verts[loop]
    center = pts.mean(axis=0)
    area = 0.0
    for i in range(len(loop)):
        v1 = pts[i] - center
        v2 = pts[(i + 1) % len(loop)] - center
        area += float(np.linalg.norm(np.cross(v1, v2))) / 2.0
    return area


def _fill_interior_holes(mesh, max_area_m2):
    loops = _boundary_loops(mesh)
    if len(loops) <= 1:
        return mesh

    interior_loops = loops[1:]
    verts = list(np.asarray(mesh.vertices, dtype=np.float64))
    tris = list(np.asarray(mesh.triangles, dtype=np.int32))
    n_filled = 0

    for loop in interior_loops:
        area = _loop_area(loop, np.asarray(verts, dtype=np.float64))
        if area > max_area_m2:
            continue
        centroid = np.asarray(verts, dtype=np.float64)[loop].mean(axis=0)
        c_idx = len(verts)
        verts.append(centroid)
        for i in range(len(loop)):
            tris.append([loop[i], loop[(i + 1) % len(loop)], c_idx])
        n_filled += 1

    if n_filled == 0:
        return mesh

    mesh_out = o3d.geometry.TriangleMesh()
    mesh_out.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64))
    mesh_out.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
    mesh_out.compute_vertex_normals()
    return mesh_out


def poisson_envelope(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS,
            max_nn=NORMAL_MAX_NN,
        )
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(ORIENT_K)
    except RuntimeError:
        pass

    with _suppress_open3d_output():
        mesh_env, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=POISSON_DEPTH,
        )

    if len(np.asarray(mesh_env.triangles)) == 0:
        raise ValueError("Poisson a produit une mesh vide")

    dists_nn, _ = cKDTree(pts).query(pts, k=2)
    median_nn = float(np.median(dists_nn[:, 1]))
    max_dist = MAX_DIST_FACTOR * median_nn

    env_verts = np.asarray(mesh_env.vertices, dtype=np.float64)
    nn_dists, _ = cKDTree(pts).query(env_verts, k=1)
    mesh_env.remove_vertices_by_mask(nn_dists > max_dist)

    if len(np.asarray(mesh_env.triangles)) == 0:
        raise ValueError("Mesh vide apres elagage")

    if FILL_POISSON_HOLES and MAX_HOLE_AREA_CM2 > 0:
        mesh_env = _fill_interior_holes(mesh_env, MAX_HOLE_AREA_CM2 * 1e-4)

    return mesh_env


def smooth(mesh_env, boundary_color=COLOR_ALL_BOUNDARIES):
    with _suppress_open3d_output():
        mesh_sub = mesh_env.subdivide_loop(number_of_iterations=SUBDIV_ITER)

    loops = _boundary_loops(mesh_sub)
    boundary_verts = np.unique(np.concatenate(loops)) if loops else np.array([], dtype=np.int32)
    verts_sub = np.asarray(mesh_sub.vertices).copy()

    with _suppress_open3d_output():
        mesh_lap = mesh_sub.filter_smooth_simple(number_of_iterations=LAPLACIAN_ITER)
        mesh_out = mesh_lap.filter_smooth_taubin(
            number_of_iterations=TAUBIN_ITER,
            lambda_filter=TAUBIN_LAMBDA,
            mu=TAUBIN_MU,
        )

    verts_out = np.asarray(mesh_out.vertices).copy()
    if len(boundary_verts):
        verts_out[boundary_verts] = verts_sub[boundary_verts]
    mesh_out.vertices = o3d.utility.Vector3dVector(verts_out)

    colors = np.tile(COLOR_INTERIOR, (len(verts_out), 1))
    if len(boundary_verts):
        colors[boundary_verts] = np.asarray(boundary_color, dtype=np.float64)
    mesh_out.compute_vertex_normals()
    mesh_out.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh_out


def voxel_downsample_points(points, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_ds = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd_ds.points, dtype=np.float64)


def run_pipeline(points, label, output_dir):
    mesh_env = poisson_envelope(points)
    mesh_smooth = smooth(mesh_env, boundary_color=COLOR_ALL_BOUNDARIES)
    loops = _boundary_loops(mesh_smooth)

    prefix = (
        f"cluster_{CLUSTER_ID:04d}_{label}"
        f"_d{POISSON_DEPTH}_sub{SUBDIV_ITER}_lap{LAPLACIAN_ITER}_taub{TAUBIN_ITER}"
    )
    points_path = os.path.join(output_dir, f"cluster_{CLUSTER_ID:04d}_{label}_points.ply")
    env_path = os.path.join(output_dir, f"{prefix}_poisson.ply")
    smooth_path = os.path.join(output_dir, f"{prefix}_smooth.ply")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(COLOR_INTERIOR, (len(points), 1)))
    o3d.io.write_point_cloud(points_path, pcd)
    o3d.io.write_triangle_mesh(env_path, mesh_env)
    o3d.io.write_triangle_mesh(smooth_path, mesh_smooth)

    outer_loop_size = len(loops[0]) if loops else 0
    outer_loop_area_cm2 = (
        _loop_area(loops[0], np.asarray(mesh_smooth.vertices, dtype=np.float64)) * 1e4 if loops else 0.0
    )
    return {
        "label": label,
        "n_points": len(points),
        "points_path": points_path,
        "poisson_path": env_path,
        "smooth_path": smooth_path,
        "poisson_area_cm2": mesh_env.get_surface_area() * 1e4,
        "smooth_area_cm2": mesh_smooth.get_surface_area() * 1e4,
        "n_boundary_loops": len(loops),
        "outer_loop_vertices": outer_loop_size,
        "outer_loop_area_cm2": outer_loop_area_cm2,
    }


# ------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------

_configure_open3d_logging()
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Chargement : {CLUSTERED_LAZ}")
laz = laspy.read(CLUSTERED_LAZ)
all_pts = np.column_stack([laz.x, laz.y, laz.z]).astype(np.float64)
all_ids = np.asarray(laz.cluster_id, dtype=np.int32)

cluster_ids = set(int(v) for v in np.unique(all_ids) if v >= 0)
if CLUSTER_ID not in cluster_ids:
    raise ValueError(f"CLUSTER_ID={CLUSTER_ID} introuvable dans clustered.laz")

pts = all_pts[all_ids == CLUSTER_ID]
if len(pts) < MIN_PTS_CLUSTER:
    raise ValueError(f"Cluster {CLUSTER_ID} trop petit: {len(pts)} pts")

print(f"Cluster {CLUSTER_ID:04d}  {len(pts)} points")
median_nn = float(np.median(cKDTree(pts).query(pts, k=2)[0][:, 1]))
voxel_size = VOXEL_SIZE_POISSON if VOXEL_SIZE_POISSON is not None else VOXEL_SIZE_FACTOR * median_nn
print(f"  median NN         = {median_nn:.6f} m")
if COMPARE_VOXEL_DOWNSAMPLE:
    print(f"  voxel size test   = {voxel_size:.6f} m")

results = [run_pipeline(pts, "raw", OUTPUT_DIR)]

if COMPARE_VOXEL_DOWNSAMPLE:
    pts_voxel = voxel_downsample_points(pts, voxel_size)
    if len(pts_voxel) < MIN_PTS_CLUSTER:
        raise ValueError(
            f"Voxel downsample trop agressif: {len(pts_voxel)} pts < {MIN_PTS_CLUSTER}"
        )
    results.append(run_pipeline(pts_voxel, "voxel", OUTPUT_DIR))

report_path = os.path.join(
    OUTPUT_DIR,
    f"cluster_{CLUSTER_ID:04d}_d{POISSON_DEPTH}_sub{SUBDIV_ITER}_lap{LAPLACIAN_ITER}_taub{TAUBIN_ITER}_report.txt",
)

report_lines = [
    f"cluster_id: {CLUSTER_ID}",
    f"n_points: {len(pts)}",
    f"poisson_depth: {POISSON_DEPTH}",
    f"max_dist_factor: {MAX_DIST_FACTOR}",
    f"median_nn_m: {median_nn:.8f}",
    f"compare_voxel_downsample: {COMPARE_VOXEL_DOWNSAMPLE}",
    f"voxel_size_poisson_m: {voxel_size:.8f}" if COMPARE_VOXEL_DOWNSAMPLE else "voxel_size_poisson_m: disabled",
]
for result in results:
    report_lines.extend([
        "",
        f"[{result['label']}]",
        f"n_points: {result['n_points']}",
        f"poisson_area_cm2: {result['poisson_area_cm2']:.3f}",
        f"smooth_area_cm2: {result['smooth_area_cm2']:.3f}",
        f"n_boundary_loops: {result['n_boundary_loops']}",
        f"outer_loop_vertices: {result['outer_loop_vertices']}",
        f"outer_loop_area_cm2: {result['outer_loop_area_cm2']:.3f}",
        f"output_points: {result['points_path']}",
        f"output_poisson: {result['poisson_path']}",
        f"output_smooth: {result['smooth_path']}",
    ])
with open(report_path, "w") as f:
    f.write("\n".join(report_lines) + "\n")

for result in results:
    print(f"[{result['label']}]")
    print(f"  points             = {result['n_points']}")
    print(f"  Poisson            = {result['poisson_area_cm2']:.3f} cm2")
    print(f"  Smooth bord fixe   = {result['smooth_area_cm2']:.3f} cm2")
    print(f"  Boundary loops     = {result['n_boundary_loops']}")
    print(f"  Outer loop verts   = {result['outer_loop_vertices']}")
    print("")
print("")
print(f"Sorties : {OUTPUT_DIR}")
print(f"Report  : {report_path}")
