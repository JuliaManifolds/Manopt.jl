import bz2
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import time
from pathlib import Path


def read_bal_bz2(file_name):
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty((n_cameras, 9))
        for i in range(n_cameras):
            camera_params[i] = [float(file.readline()) for _ in range(9)]

        points_3d = np.empty((n_points, 3))
        for i in range(n_points):
            points_3d[i] = [float(file.readline()) for _ in range(3)]

        return camera_params, points_3d, camera_indices, point_indices, points_2d


def subsample_bal(camera_params, points_3d, camera_indices, point_indices, points_2d, num_cameras):
    selected_cams = np.arange(num_cameras)
    mask = np.isin(camera_indices, selected_cams)

    sub_camera_indices = camera_indices[mask]
    sub_point_indices = point_indices[mask]
    sub_points_2d = points_2d[mask]

    unique_points, new_point_indices = np.unique(sub_point_indices, return_inverse=True)

    sub_camera_params = camera_params[selected_cams]
    sub_points_3d = points_3d[unique_points]

    return sub_camera_params, sub_points_3d, sub_camera_indices, new_point_indices, sub_points_2d


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + (1 - cos_theta) * dot * v


def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj ** 2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals."""
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A


def run_bundle_adjustment(camera_params, points_3d, camera_indices, point_indices, points_2d):
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    # Initialize identically to Julia's initialization
    x0_camera_params = np.zeros((n_cameras, 9))
    x0_camera_params[:, 3:6] = 1.0  # Translations to 1.0 (ones(3, data.num_cameras))
    x0_camera_params[:, 6] = 400.0  # Focal length
    x0_points_3d = np.zeros((n_points, 3))  # points initialized to 0

    x0 = np.hstack((x0_camera_params.ravel(), x0_points_3d.ravel()))

    # Matching the Julia bounds: points in [-1.0, 1.0], f in [350.0, 450.0], k1, k2 can vary
    lower_bounds_cam = np.full((n_cameras, 9), -np.inf)
    lower_bounds_cam[:, 6] = 350.0

    upper_bounds_cam = np.full((n_cameras, 9), np.inf)
    upper_bounds_cam[:, 6] = 450.0

    lower_bounds_pts = np.full((n_points, 3), -1.0)
    upper_bounds_pts = np.full((n_points, 3), 1.0)

    bounds = (
        np.hstack((lower_bounds_cam.ravel(), lower_bounds_pts.ravel())),
        np.hstack((upper_bounds_cam.ravel(), upper_bounds_pts.ravel()))
    )

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    print("Starting optimization...")
    t0 = time.time()
    max_nfev = 100
    # max_nfev = 10000

    res = least_squares(
        fun, x0, jac_sparsity=A, verbose=2, x_scale='jac',
        ftol=1e-12, xtol=1e-11, gtol=1e-12, max_nfev=max_nfev,
        loss='huber', f_scale=1.0,
        method='trf', bounds=bounds,
        args=(n_cameras, n_points, camera_indices, point_indices, points_2d)
    )

    t1 = time.time()
    print("Optimization finished in {:.2f} seconds".format(t1 - t0))

    return res


def save_solution_to_csv(camera_params, points_3d, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cameras_csv = output_path / "python_opt_camera_params.csv"
    points_csv = output_path / "python_opt_points_3d.csv"

    np.savetxt(cameras_csv, camera_params, delimiter=",")
    np.savetxt(points_csv, points_3d, delimiter=",")

    return cameras_csv, points_csv


if __name__ == "__main__":
    file_path = "/home/mateusz/data/bal/ladybug/problem-49-7776-pre.txt.bz2"
    camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_bz2(file_path)

    num_cameras = 20

    sub_camera_params, sub_points_3d, sub_camera_indices, sub_point_indices, sub_points_2d = subsample_bal(
        camera_params, points_3d, camera_indices, point_indices, points_2d, num_cameras=num_cameras
    )

    print(
        f"Cameras: {sub_camera_params.shape[0]}, Points: {sub_points_3d.shape[0]}, Observations: {sub_points_2d.shape[0]}")
    res = run_bundle_adjustment(sub_camera_params, sub_points_3d, sub_camera_indices, sub_point_indices, sub_points_2d)

    print(f"Final cost: {res.cost}")
    print(f"Optimality: {res.optimality}")

    n_cameras = sub_camera_params.shape[0]
    n_points = sub_points_3d.shape[0]
    opt_camera_params = res.x[:n_cameras * 9].reshape((n_cameras, 9))
    opt_points_3d = res.x[n_cameras * 9:].reshape((n_points, 3))

    print("\nFirst camera optimized parameters (rot_vec, t, f, k1, k2):")
    print(opt_camera_params[0])
    print("\nFirst 3D point optimized coordinates:")
    print(opt_points_3d[0])

    output_dir = Path(__file__).resolve().parent / "bal_csv_solution"
    cameras_csv, points_csv = save_solution_to_csv(opt_camera_params, opt_points_3d, output_dir)
    print(f"\nSaved optimized camera params CSV: {cameras_csv}")
    print(f"Saved optimized 3D points CSV: {points_csv}")
