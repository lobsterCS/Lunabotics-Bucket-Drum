import os
import re
import numpy as np
import mapbox_earcut as earcut


class DrumMeshPreprocessor:

    def __init__(self, raw_obj_path: str, target_outer_radius: float, d0: float):
        if not os.path.exists(raw_obj_path):
            raise FileNotFoundError(f"File not found: {raw_obj_path}")

        self.raw_obj_path = raw_obj_path
        self.target_outer_radius = target_outer_radius
        self.d0 = float(d0)

        base, ext = os.path.splitext(raw_obj_path)
        self.clean_obj_path = f"{base}_meters_clean{ext}"
        self.bce_cache_path = f"{base}_bce_d0_{self.d0:.4f}m.npy"

    def process(self, force_recompute: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes the full mesh preprocessing pipeline for Chrono CRM.

        Args:
            force_recompute (bool): If True, ignores existing .obj/.npy caches and regenerates.

        Returns:
            V (np.ndarray): Scaled and centered vertices in meters, shape (N, 3).
            F (np.ndarray): Triangulated 0-based face indices, shape (M, 3).
            bce_pts (np.ndarray): 2-layer BCE markers sampled at ~0.75*d0 spacing, shape (K, 3).
        """
        # Optional: Invalidate caches if force_recompute is requested
        if force_recompute:
            for cache_file in (self.clean_obj_path, self.bce_cache_path):
                if os.path.isfile(cache_file):
                    os.remove(cache_file)

        # 1. Parse raw OBJ (handles negative indices and concave n-gon triangulation)
        V, F = self._parse_obj(self.raw_obj_path)

        # 2. Center geometry at its bounding-box midpoint (in raw units)
        bbox_min = V.min(axis=0)
        bbox_max = V.max(axis=0)
        raw_extents = bbox_max - bbox_min
        V -= 0.5 * (bbox_min + bbox_max)

        # 3. Detect CAD units (via header first, then heuristics) and scale to meters
        scale_factor = self._detect_scale_factor(raw_extents, path=self.raw_obj_path)
        V *= scale_factor

        scaled_extents = raw_extents * scale_factor
        print(
            f"  [Geometry] Final centered dimensions: "
            f"{scaled_extents[0]:.4f} x {scaled_extents[1]:.4f} x {scaled_extents[2]:.4f} m "
            f"({len(V):,} vertices, {len(F):,} triangles)"
        )

        # 4. Save clean, meter-scaled, triangulated OBJ for Chrono visualizers / C++ loaders
        self._export_clean_obj(V, F)

        # 5. Load or generate 2-layer BCE markers
        bce_pts = self._generate_or_load_bce(V, F)

        return V, F, bce_pts

    def _parse_obj(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        """

        :param path:
        :return: vertices and faces as numpy arrays for chrono
        """
        vertices = []
        faces = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("v "):
                    # split with argument None will get rid of all white spaces in the
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    parts = line.split()[1:]
                    # Fan-triangulate quads / n-gons
                    # Extract only the vertex index from each "vertex/texture/normal" token
                    vertex_indices = []
                    for part in parts:
                        index = int(part.split("/")[0])  # Get the part before the first slash

                        # Handle standard 1-based indexing vs. negative relative indexing
                        if index < 0:
                            actual_idx = len(vertices) + index  # e.g., -1 becomes len(vertices) - 1
                        else:
                            actual_idx = index - 1  # Convert 1-based to 0-based

                        vertex_indices.append(actual_idx)

                    num_vertices_idx = len(vertex_indices)

                    if num_vertices_idx == 3:
                        # Simple triangle so no triangulation needed
                        faces.append(vertex_indices)
                    elif num_vertices_idx > 3:
                        # Robust triangulation for concave n-gons using mapbox-earcut
                        poly_vertices = np.array([vertices[i] for i in vertex_indices], dtype=np.float64)

                        # Simple 2D projection (drop the coordinate axis with the least variance)
                        variances = np.var(poly_vertices, axis=0)
                        drop_axis = np.argmin(variances)
                        poly_2d = np.ascontiguousarray(np.delete(poly_vertices, drop_axis, axis=1), dtype=np.float64)

                        # Run ear-clipping triangulation (requires shape (N, 2) float64 and (K,) uint32)
                        rings = np.array([len(poly_2d)], dtype=np.uint32)
                        triangles = earcut.triangulate_float64(poly_2d, rings)

                        # Map local triangle indices back to global vertex indices
                        for i in range(0, len(triangles), 3):
                            faces.append([
                                vertex_indices[int(triangles[i])],
                                vertex_indices[int(triangles[i + 1])],
                                vertex_indices[int(triangles[i + 2])]
                            ])

        if not vertices or not faces:
            raise ValueError(f"No valid vertices or faces found in {path}")

        return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)

    def _read_header_scale(self, path: str) -> float | None:
        """
        Helper function for reading in stated unit conversion in .obj file.
        Check the top comments of the OBJ file for an explicit unit definition.
        """
        with open(path, "r", encoding="utf-8") as f:
            for _ in range(50):  # Check the first 50 lines (header comments)
                line = f.readline()
                if not line:
                    break
                line_lower = line.strip().lower()

                # Match patterns like: "# 1.222 units = meters" or "units = 1.222 meters"
                match = re.search(r"([\d\.]+)\s*units?\s*=\s*(?:1\.0\s*)?meter", line_lower)
                if match:
                    # If N units = 1 meter, then 1 unit = (1 / N) meters
                    units_per_meter = float(match.group(1))
                    return 1.0 / units_per_meter

                # Match patterns like: "# 1 unit = 1.222 meters"
                match_rev = re.search(r"(?:1\.0\s*)?unit\s*=\s*([\d\.]+)\s*meter", line_lower)
                if match_rev:
                    return float(match_rev.group(1))

        return None

    def _detect_scale_factor(self, extents: np.ndarray, path: str | None = None) -> float:
        """
        :param extents: A 3-element NumPy array [width, height, depth]
                        representing the raw bounding-box dimensions of the .obj mesh before any scaling
        :param path: optional path to .obj file to check if header has unit conversion data
                     assumes that no path is passed, so it falls back to guess and check unit conversion
        :return: scale factor that multiplies with the dimensions of the vertices
        """
        # 1. Try reading the exact scale directly from the OBJ header
        if path:
            header_scale = self._read_header_scale(path)
            if header_scale is not None:
                print(f"  [Scale] Found explicit unit in header: scale factor = {header_scale:.6f}")
                return header_scale

        # 2. Heuristic fallback based on expected wheel/drum dimensions
        max_dim = float(np.max(extents))
        expected_diameter = 2.0 * self.target_outer_radius
        ratio = max_dim / expected_diameter

        # Millimeters (~1000x)
        if 500.0 <= ratio <= 2000.0:
            print(f"  [Auto-Scale] Detected Millimeters (ratio ~{ratio:.1f}). Scaling x0.001.")
            return 0.001

        # Centimeters (~100x)
        elif 50.0 <= ratio < 500.0:
            print(f"  [Auto-Scale] Detected Centimeters (ratio ~{ratio:.1f}). Scaling x0.01.")
            return 0.01

        # Inches (~39.37x)
        elif 20.0 <= ratio < 50.0:
            print(f"  [Auto-Scale] Detected Inches (ratio ~{ratio:.1f}). Scaling x0.0254.")
            return 0.0254

        # Already meters (~1x)
        elif 0.5 <= ratio <= 2.0:
            print(f"  [Auto-Scale] Mesh is already in meters ({extents[0]:.3f} x {extents[1]:.3f} x {extents[2]:.3f} m).")
            return 1.0

        # Catch-all: exact fit based on target radius to prevent simulation failure
        print(f"  [Auto-Scale Warning] Unrecognized unit (ratio={ratio:.2f}). Scaling directly to target diameter.")
        return expected_diameter / max_dim

    def _export_clean_obj(self, V: np.ndarray, F: np.ndarray):
        """
        Program creates a scaled and cleaned .obj file if no such file exists or if the original raw .obj file has been updated
        :param V: Numpy array of vertices that are scaled to meters and indexed at 0
        :param F: Numpy array of 0 indexed faces
        :return: None
        """
        if not os.path.isfile(self.clean_obj_path) or os.path.getmtime(self.raw_obj_path) > os.path.getmtime(self.clean_obj_path):
            with open(self.clean_obj_path, "w", encoding="utf-8") as f_out:
                f_out.write("# Auto-scaled meter geometry OBJ\n")
                for vx, vy, vz in V:
                    f_out.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
                for i0, i1, i2 in (F + 1):
                    f_out.write(f"f {i0} {i1} {i2}\n")

    def _generate_or_load_bce(self, V: np.ndarray, F: np.ndarray) -> np.ndarray:
        """

        :param V: Numpy array of vertices that are scaled to meters and indexed at 0
        :param F: Numpy array of 0 indexed faces
        :return:
        """
        if os.path.isfile(self.bce_cache_path) and os.path.getmtime(self.bce_cache_path) >= os.path.getmtime(
                self.raw_obj_path):
            pts = np.load(self.bce_cache_path)
            print(f"  [BCE Cache] Loaded {len(pts):,} markers from {os.path.basename(self.bce_cache_path)}")
            return pts

        spacing = 0.75 * self.d0
        print(f"  [CPU] Fast-sampling 2-layer surface BCE markers (target spacing ~{spacing:.4f} m)...")

        v0 = V[F[:, 0]]
        v1 = V[F[:, 1]]
        v2 = V[F[:, 2]]

        # 1. Compute cross products, triangle areas, and unit normal vectors
        cross_vecs = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(cross_vecs, axis=1)
        areas = 0.5 * norms
        total_area = float(np.sum(areas))

        # Unit outward normals for each triangle (safe division for degenerate triangles)
        unit_normals = cross_vecs / np.maximum(norms[:, None], 1e-12)

        # Ensure normals point outward relative to mesh centroid (fallback check for CAD winding)
        tri_centers = (v0 + v1 + v2) / 3.0
        mesh_center = np.mean(V, axis=0)
        if np.sum(np.sum((tri_centers - mesh_center) * unit_normals, axis=1) * areas) < 0:
            unit_normals = -unit_normals

        # 2. Oversample by 4x before voxel deduplication to avoid random Monte Carlo gaps
        target_per_layer = max(2000, int(total_area / (spacing ** 2)))
        num_samples = 4 * target_per_layer
        probs = areas / max(total_area, 1e-12)

        rng = np.random.default_rng(42)
        tri_idx = rng.choice(len(F), size=num_samples, p=probs)

        r1 = np.sqrt(rng.random(num_samples))[:, None]
        r2 = rng.random(num_samples)[:, None]

        # Outer surface layer (Layer 0)
        pts_layer1 = (1.0 - r1) * v0[tri_idx] + (r1 * (1.0 - r2)) * v1[tri_idx] + (r1 * r2) * v2[tri_idx]

        # 3. Extrude Second Layer inward along the triangle normals (Layer 1)
        sampled_normals = unit_normals[tri_idx]
        pts_layer2 = pts_layer1 - (spacing * sampled_normals)

        # Combine both layers and deduplicate via 3D voxel grid
        pts_all = np.vstack([pts_layer1, pts_layer2])
        voxel_coords = np.floor(pts_all / spacing).astype(np.int64)
        _, unique_idx = np.unique(voxel_coords, axis=0, return_index=True)
        bce_pts = pts_all[np.sort(unique_idx)].astype(np.float64)

        np.save(self.bce_cache_path, bce_pts)
        print(f"  [CPU] Generated {len(bce_pts):,} 2-layer BCE markers in <0.1s.")
        return bce_pts
