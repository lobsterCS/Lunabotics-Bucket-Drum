import argparse
import os
import tempfile
import time
import numpy as np

from src.simulation.drum_preprocessor import DrumMeshPreprocessor


def create_synthetic_bucket_drum_obj(
    filepath: str,
    outer_radius_m: float = 0.15,
    width_m: float = 0.25,
    header_line: str = "# 1000.0 units = meter (Synthetic Onshape Bucket Drum)\n",
    scale_mult: float = 1000.0,
) -> None:
    """
    Creates a synthetic bucket drum .obj file that intentionally exercises:
      1. Header unit parsing (e.g., '# 1000.0 units = meter')
      2. Off-center CAD coordinates (to test midpoint centering)
      3. Standard triangles (3 vertices)
      4. Quads (4 vertices) with v/vt/vn slash syntax
      5. Concave 5-gon end-cap / scoop plate (tests mapbox_earcut)
      6. Negative relative indices (e.g., f -4 -3 -2 -1)
    """
    R = outer_radius_m * scale_mult
    W = width_m * scale_mult
    offset = np.array([120.0, -45.0, 80.0])  # Intentional off-center shift

    num_segments = 16
    angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header_line)
        f.write("# Synthetic Bucket Drum Mesh for Validation\n")

        # 1. Outer cylinder ring vertices (front & back)
        for z in (-W / 2.0, W / 2.0):
            for theta in angles:
                vx = R * np.cos(theta) + offset[0]
                vy = R * np.sin(theta) + offset[1]
                vz = z + offset[2]
                f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")

        # Dummy texture/normal lines to test v/vt/vn parsing
        f.write("vt 0.5 0.5\n")
        f.write("vn 0.0 0.0 1.0\n")

        # 2. Drum shell quads (using 1-based v/vt/vn format)
        for i in range(num_segments):
            nxt = (i + 1) % num_segments
            v1 = i + 1
            v2 = nxt + 1
            v3 = num_segments + nxt + 1
            v4 = num_segments + i + 1
            f.write(f"f {v1}/1/1 {v2}/1/1 {v3}/1/1 {v4}/1/1\n")

        # 3. Add a concave 5-vertex scoop baffle on the front face (tests earcut)
        scoop_verts = [
            [0.2 * R, 0.2 * R, -W / 2.0],
            [0.9 * R, 0.1 * R, -W / 2.0],
            [0.5 * R, 0.5 * R, -W / 2.0],  # Indented concave vertex
            [0.9 * R, 0.9 * R, -W / 2.0],
            [0.2 * R, 0.9 * R, -W / 2.0],
        ]
        for sv in scoop_verts:
            pt = np.array(sv) + offset
            f.write(f"v {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")

        # Reference the 5 scoop vertices using NEGATIVE relative indices
        f.write("f -5 -4 -3 -2 -1\n")

        # 4. Close the drum caps with triangle fans so surface area is realistic
        center_front = np.array([0.0, 0.0, -W / 2.0]) + offset
        center_back = np.array([0.0, 0.0, W / 2.0]) + offset
        f.write(f"v {center_front[0]:.6f} {center_front[1]:.6f} {center_front[2]:.6f}\n")
        f.write(f"v {center_back[0]:.6f} {center_back[1]:.6f} {center_back[2]:.6f}\n")

        c_front_idx = 2 * num_segments + 5 + 1
        c_back_idx = c_front_idx + 1
        for i in range(num_segments):
            nxt = (i + 1) % num_segments
            f.write(f"f {c_front_idx} {nxt + 1} {i + 1}\n")
            f.write(f"f {c_back_idx} {num_segments + i + 1} {num_segments + nxt + 1}\n")


def validate_drum_file(obj_path: str, target_radius: float, d0: float, show_plot: bool = False) -> None:
    print("=" * 72)
    print(f"VALIDATING DRUM PREPROCESSOR ON: {os.path.basename(obj_path)}")
    print(f"Target Outer Radius: {target_radius:.4f} m | SPH Spacing (d0): {d0:.4f} m")
    print("=" * 72)

    prep = DrumMeshPreprocessor(raw_obj_path=obj_path, target_outer_radius=target_radius, d0=d0)

    # --- Run 1: Fresh computation (force_recompute=True) ---
    t0 = time.perf_counter()
    V, F, bce_pts = prep.process(force_recompute=True)
    t_fresh = time.perf_counter() - t0

    # --- Assertion 1: Shapes & Data Types ---
    assert isinstance(V, np.ndarray) and V.ndim == 2 and V.shape[1] == 3, "V must be shape (N, 3)"
    assert V.dtype == np.float64, f"Expected V dtype float64, got {V.dtype}"
    assert isinstance(F, np.ndarray) and F.ndim == 2 and F.shape[1] == 3, "F must be triangulated (M, 3)"
    assert F.dtype == np.int32, f"Expected F dtype int32, got {F.dtype}"
    assert isinstance(bce_pts, np.ndarray) and bce_pts.ndim == 2 and bce_pts.shape[1] == 3, "bce_pts must be (K, 3)"
    assert not np.isnan(V).any() and not np.isnan(bce_pts).any(), "NaN values detected in output arrays!"
    print("[PASS] 1. Output array shapes, dtypes, and NaN checks passed.")

    # --- Assertion 2: Valid 0-based Triangle Indexing ---
    assert F.min() >= 0, f"Negative index found in F: {F.min()}"
    assert F.max() < len(V), f"Face index {F.max()} out of bounds for {len(V)} vertices"
    print(f"[PASS] 2. Face indices strictly 0-based and within bounds [0, {len(V) - 1}].")

    # --- Assertion 3: Bounding-Box Centering at (0, 0, 0) ---
    bbox_midpoint = 0.5 * (V.min(axis=0) + V.max(axis=0))
    assert np.allclose(bbox_midpoint, 0.0, atol=1e-7), f"Mesh not centered! Midpoint = {bbox_midpoint}"
    print(f"[PASS] 3. Geometry centered at origin (midpoint error = {np.linalg.norm(bbox_midpoint):.2e} m).")

    # --- Assertion 4: Scaled Meter Extents ---
    extents = V.max(axis=0) - V.min(axis=0)
    max_span = float(np.max(extents))
    expected_diam = 2.0 * target_radius
    # Allow reasonable margin in case drum width > diameter or scoops protrude
    assert 0.2 * expected_diam <= max_span <= 5.0 * expected_diam, (
        f"Scaled max dimension ({max_span:.4f} m) looks out of scale compared to target diameter ({expected_diam:.4f} m)"
    )
    print(f"[PASS] 4. Physical meter scale verified (extents = {extents[0]:.4f} x {extents[1]:.4f} x {extents[2]:.4f} m).")

    # --- Assertion 5: Clean OBJ File Exported & Valid ---
    assert os.path.isfile(prep.clean_obj_path), "Clean OBJ file was not created!"
    V_clean, F_clean = prep._parse_obj(prep.clean_obj_path)
    assert len(V_clean) == len(V) and len(F_clean) == len(F), "Clean OBJ vertex/face count mismatch!"
    assert np.allclose(V_clean, V, atol=1e-5), "Clean OBJ coordinates do not match in-memory V!"
    print(f"[PASS] 5. Clean meter OBJ verified on disk ({os.path.basename(prep.clean_obj_path)}).")

    # --- Assertion 6: 2-Layer BCE Marker Count & Voxel Spacing ---
    spacing = 0.75 * d0
    voxels = np.floor(bce_pts / spacing).astype(np.int64)
    unique_voxels = np.unique(voxels, axis=0)
    assert len(unique_voxels) == len(bce_pts), "Duplicate BCE points found in the same voxel cell!"
    assert len(bce_pts) > 100, f"Too few BCE points generated ({len(bce_pts)})!"
    print(f"[PASS] 6. 2-Layer BCE markers verified ({len(bce_pts):,} unique markers at ~{spacing*1000:.2f} mm grid).")

    # --- Assertion 7: Cache Reload Speed & Exact Match ---
    t1 = time.perf_counter()
    V_cached, F_cached, bce_cached = prep.process(force_recompute=False)
    t_cached = time.perf_counter() - t1
    assert np.array_equal(bce_pts, bce_cached), "Cached BCE points do not match freshly generated BCE points!"
    print(f"[PASS] 7. Cache reload verified (Fresh: {t_fresh*1000:.1f} ms -> Cached: {t_cached*1000:.1f} ms).")
    print("=" * 72)
    print("ALL VALIDATION CHECKS PASSED SUCCESSFULLY!\n")

    # Optional 3D Visualization for Visual Inspection of the Bucket Drum & BCE Layers
    if show_plot:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        # Subsample BCE markers if very dense so matplotlib stays responsive
        step = max(1, len(bce_pts) // 5000)
        pts_plot = bce_pts[::step]
        ax.scatter(pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2], s=2, c=pts_plot[:, 2], cmap="viridis", alpha=0.7)
        ax.set_title(f"Bucket Drum BCE Point Cloud ({len(bce_pts):,} markers, d0={d0}m)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_box_aspect(extents)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate DrumMeshPreprocessor with a bucket drum .obj file.")
    parser.add_argument("obj_path", nargs="?", default=None, help="Path to your bucket drum .obj file (optional)")
    parser.add_argument("--radius", type=float, default=0.15, help="Target outer radius of the drum in meters (default: 0.15)")
    parser.add_argument("--d0", type=float, default=0.003, help="SPH particle spacing d0 in meters (default: 0.003)")
    parser.add_argument("--plot", action="store_true", help="Display a 3D scatter plot of the generated BCE markers")
    args = parser.parse_args()

    if args.obj_path:
        # Validate the user's actual bucket drum .obj file
        validate_drum_file(args.obj_path, target_radius=args.radius, d0=args.d0, show_plot=args.plot)
    else:
        # Run automated self-test suite on synthetic bucket drum files (Onshape header + Heuristic fallback)
        print("No .obj path provided -> Running automated test suite with synthetic bucket drum .obj files...\n")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test Case A: Onshape header ("# 1.222 units = meter")
            onshape_obj = os.path.join(tmpdir, "bucket_drum_onshape.obj")
            create_synthetic_bucket_drum_obj(
                onshape_obj,
                outer_radius_m=args.radius,
                width_m=0.25,
                header_line="# 1.222 units = meter\n",
                scale_mult=1.222,
            )
            validate_drum_file(onshape_obj, target_radius=args.radius, d0=args.d0, show_plot=args.plot)

            # Test Case B: No header, exported in Inches (39.3701x) -> tests heuristic fallback
            inches_obj = os.path.join(tmpdir, "bucket_drum_inches.obj")
            create_synthetic_bucket_drum_obj(
                inches_obj,
                outer_radius_m=args.radius,
                width_m=0.25,
                header_line="# Exported without unit metadata\n",
                scale_mult=39.3700787,
            )
            validate_drum_file(inches_obj, target_radius=args.radius, d0=args.d0, show_plot=False)
