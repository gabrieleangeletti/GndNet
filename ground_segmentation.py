import logging

import numpy as np
import open3d as o3d

from larki_pc.io import e57

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_segmented_pcd(pcd: o3d.geometry.PointCloud, out: np.ndarray) -> None:
    colors = np.asarray(pcd.colors)
    colors[out] = np.array([0., 1., 0.])

    mask = np.ones(colors.shape[0], np.bool)
    mask[out] = 0
    colors[mask] = np.array([1., 0., 1.])

    pcd.colors = o3d.utility.Vector3dVector(colors)
    e57.write_e57([pcd], "ground-segmentation.e57")


def generate_toposurface(pcd: o3d.geometry.PointCloud, out: np.ndarray) -> None:
    points, colors = np.asarray(pcd.points), np.asarray(pcd.colors)

    mask = np.zeros(points.shape[0], np.bool)
    mask[out] = 1

    toposurface = o3d.geometry.PointCloud()
    toposurface.points = o3d.utility.Vector3dVector(points[mask])
    toposurface.colors = o3d.utility.Vector3dVector(colors[mask])

    # o3d.visualization.draw_geometries([toposurface])

    logger.info("computing normals.")
    toposurface.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.9, max_nn=100),
        fast_normal_computation=False,
    )
    logger.info("orienting normals.")
    # toposurface.orient_normals_consistent_tangent_plane(30)

    logger.info("generating mesh surface.")
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(toposurface, depth=18)
    logger.info(mesh)
    # o3d.visualization.draw_geometries([mesh])
    o3d.io.write_point_cloud()


def main() -> None:
    task = "generate_segmented_pcd"

    pcd = e57.load_point_clouds_open3d("../data/larki/outdoor/Vic-Geelong-Yarra_St-217-Pointcloud-Streetscape-LARKI-210304.e57")[0]
    out = np.load("../GndNet/data/larki/out.npy")

    if task == "generate_segmented_pcd":
        generate_segmented_pcd(pcd, out)
    elif task == "generate_toposurface":
        generate_toposurface(pcd, out)


if __name__ == "__main__":
    main()

