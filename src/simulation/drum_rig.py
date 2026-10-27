"""
drum_rig.py
===========
1. Selects and aligns the bucket drum CAD orientation to match the rig rotation axis.
2. Determines and executes 3-stage kinematic motion (Hover/Settle -> Plunge -> Traverse).
3. Builds the Multibody System (SMC) gantry and registers FSI BCE markers with Chrono::CRM.
"""

import math
import numpy as np
import pychrono as chrono

from .drum_preprocessor import DrumMeshPreprocessor

class DrumOrientationAligner:
    """
    Detects the cylinder revolution axis from bounding box / CAD dimensions and
    computes the transformation required to align the drum cylinder axis with
    the rig's drive spindle (the Chrono Y-axis).
    """

    @staticmethod
    def detect_cylinder_axis(extents: np.ndarray, drum_length: float, drum_radius: float) -> str:
        """
        Auto-detects whether the drum cylinder is along 'X', 'Y', or 'Z'.
        Cylinder axis extent is closest to `drum_length` while the other two are ~ 2*drum_radius.
        """
        diameter = 2.0 * drum_radius
        diffs = [
            abs(extents[0] - drum_length) + abs(extents[1] - diameter) + abs(extents[2] - diameter),  # X
            abs(extents[1] - drum_length) + abs(extents[0] - diameter) + abs(extents[2] - diameter),  # Y
            abs(extents[2] - drum_length) + abs(extents[0] - diameter) + abs(extents[1] - diameter),  # Z
        ]
        best_axis = ["X", "Y", "Z"][int(np.argmin(diffs))]
        return best_axis

    @classmethod
    def align_to_y_axis(
        cls,
        V: np.ndarray,
        bce_pts: np.ndarray,
        axis: str = "auto",
        drum_length: float = 0.5,
        drum_radius: float = 0.15,
    ) -> tuple[np.ndarray, np.ndarray, chrono.ChQuaterniond]:
        """
        Rotates vertices and BCE points so the drum cylinder revolution axis points along +Y.
        """
        extents = V.max(axis=0) - V.min(axis=0)
        selected_axis = cls.detect_cylinder_axis(extents, drum_length, drum_radius) if axis.lower() == "auto" else axis.upper()

        # Rotation matrix to align detected axis to +Y
        if selected_axis == "X":
            # Rotate +X into +Y (around +Z by +90 deg)
            R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
            q_align = chrono.QuatFromAngleZ(chrono.CH_PI_2)
            print(f"  [Orientation] Drum axis detected along X. Aligned to Y spindle via +90° Z rotation.")
        elif selected_axis == "Z":
            # Rotate +Z into +Y (around +X by -90 deg)
            R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
            q_align = chrono.QuatFromAngleX(-chrono.CH_PI_2)
            print(f"  [Orientation] Drum axis detected along Z. Aligned to Y spindle via -90° X rotation.")
        else:
            # Already aligned along Y
            R = np.eye(3, dtype=np.float64)
            q_align = chrono.QUNIT
            print(f"  [Orientation] Drum axis is already aligned along Y spindle.")

        V_aligned = V @ R
        bce_aligned = bce_pts @ R
        return V_aligned, bce_aligned, q_align

class DrumKinematicsPlanner:
    """
    Computes time-resolved kinematic targets for Settle, Plunge, and Traverse phases.
    """

    def __init__(self, op_cfg: dict, sim_cfg: dict, hover_clearance: float = 0.01):
        self.t_settle = float(sim_cfg["settling_time"])
        self.t_plunge = float(op_cfg["plunge_duration"])
        self.cut_depth = float(op_cfg["cut_depth"])
        self.hover_clearance = hover_clearance

        self.vx_traverse = float(op_cfg["forward_velocity"])
        total_plunge_dist = self.hover_clearance + self.cut_depth
        self.vz_plunge = -total_plunge_dist / max(self.t_plunge, 1e-4)
        self.omega_target = float(op_cfg["rpm"]) * (2.0 * math.pi / 60.0)

    def evaluate(self, sim_time: float) -> tuple[int, float, float, float]:
        """
        Determines current active phase and motor speeds at time `sim_time`.

        Returns:
            phase_idx (0=Settle, 1=Plunge, 2=Traverse), vx, vz, omega
        """
        if sim_time < self.t_settle:
            return 0, 0.0, 0.0, 0.0
        elif sim_time < (self.t_settle + self.t_plunge):
            return 1, 0.0, self.vz_plunge, self.omega_target
        else:
            return 2, self.vx_traverse, 0.0, self.omega_target

class BucketDrumRig:
    """
    Constructs the 3-axis motorized multibody gantry (X traverse, Z plunge, Y spin)
    and registers the aligned BCE markers with Chrono::CRM.
    """

    def __init__(self, env, drum_cfg: dict, op_cfg: dict, y_offset: float = 0.0):
        self.env = env
        self.drum_cfg = drum_cfg
        self.op_cfg = op_cfg
        self.y_offset = float(y_offset)

        self.outer_radius = float(drum_cfg["outer_radius"])
        self.inner_radius = float(drum_cfg["inner_cavity_radius"])
        self.drum_length = float(drum_cfg["drum_length"])
        self.hover_clearance = 0.01

        # 1. Optimize, scale, and sample CAD mesh via DrumMeshPreprocessor
        preprocessor = DrumMeshPreprocessor(
            raw_obj_path=drum_cfg["obj_path"],
            target_outer_radius=self.outer_radius,
            d0=self.env.d0,
        )
        V, F, bce_pts = preprocessor.process()
        self.clean_obj_path = preprocessor.clean_obj_path

        # 2. Select orientation and align cylinder axis with rig spindle (Y-axis)
        requested_axis = drum_cfg.get("cylinder_axis", "auto")
        _, bce_aligned, self.q_mesh_align = DrumOrientationAligner.align_to_y_axis(
            V, bce_pts,
            axis=requested_axis,
            drum_length=self.drum_length,
            drum_radius=self.outer_radius,
        )

        # Convert aligned BCE points to Chrono vector
        self.bce_markers = chrono.vector_ChVector3d()
        for px, py, pz in bce_aligned:
            self.bce_markers.append(chrono.ChVector3d(float(px), float(py), float(pz)))

        # 3. Kinematic motion trajectory planner
        self.planner = DrumKinematicsPlanner(
            op_cfg=self.op_cfg,
            sim_cfg=self.env.cfg["simulation"],
            hover_clearance=self.hover_clearance,
        )
        self._current_phase = -1

        # 4. Starting 3D coordinates
        self.start_x = float(op_cfg["start_x"])
        self.start_z = self.outer_radius + self.hover_clearance

        # Multi-body elements
        self.ground = None
        self.x_stage = None
        self.z_carrier = None
        self.drum_body = None
        self.x_motor = None
        self.z_motor = None
        self.rot_motor = None
        self.fn_x = None
        self.fn_z = None
        self.fn_rot = None
        self.fsi_body_index = None

        self._build_multibody_rig()
        self._register_fsi_body()

    def _build_multibody_rig(self):
        sysMBS = self.env.sysMBS
        struct_cfg = self.env.cfg["drum_structural"]
        init_pos = chrono.ChVector3d(self.start_x, self.y_offset, self.start_z)

        # Stage 1: Fixed Ground Anchor
        self.ground = chrono.ChBody()
        self.ground.SetFixed(True)
        self.ground.SetPos(init_pos)
        sysMBS.AddBody(self.ground)

        # Stage 2: Horizontal Traverse Carriage (Moves along X)
        self.x_stage = chrono.ChBody()
        self.x_stage.SetMass(10.0)
        self.x_stage.SetInertiaXX(chrono.ChVector3d(1.0, 1.0, 1.0))
        self.x_stage.SetPos(init_pos)
        sysMBS.AddBody(self.x_stage)

        self.fn_x = chrono.ChFunctionConst(0.0)
        self.x_motor = chrono.ChLinkMotorLinearSpeed()
        q_x_dir = chrono.QuatFromAngleY(chrono.CH_PI_2)
        self.x_motor.Initialize(self.x_stage, self.ground, chrono.ChFramed(init_pos, q_x_dir))
        self.x_motor.SetSpeedFunction(self.fn_x)
        sysMBS.AddLink(self.x_motor)

        # Stage 3: Vertical Plunge Carrier (Moves along Z)
        self.z_carrier = chrono.ChBody()
        self.z_carrier.SetMass(10.0)
        self.z_carrier.SetInertiaXX(chrono.ChVector3d(1.0, 1.0, 1.0))
        self.z_carrier.SetPos(init_pos)
        sysMBS.AddBody(self.z_carrier)

        self.fn_z = chrono.ChFunctionConst(0.0)
        self.z_motor = chrono.ChLinkMotorLinearSpeed()
        self.z_motor.Initialize(self.z_carrier, self.x_stage, chrono.ChFramed(init_pos, chrono.QUNIT))
        self.z_motor.SetSpeedFunction(self.fn_z)
        sysMBS.AddLink(self.z_motor)

        # Stage 4: PLA Bucket Drum Body (Rotates around Y)
        self.drum_body = chrono.ChBody()
        self.drum_body.SetPos(init_pos)

        rho_pla = float(struct_cfg["material_density"])
        vol_approx = math.pi * (self.outer_radius ** 2 - self.inner_radius ** 2) * self.drum_length * 0.35
        mass_pla = max(0.5, rho_pla * vol_approx)
        iyy = 0.5 * mass_pla * (self.outer_radius ** 2 + self.inner_radius ** 2)
        ixx_izz = (1.0 / 12.0) * mass_pla * (
            3.0 * (self.outer_radius ** 2 + self.inner_radius ** 2) + self.drum_length ** 2
        )

        self.drum_body.SetMass(mass_pla)
        self.drum_body.SetInertiaXX(chrono.ChVector3d(ixx_izz, iyy, ixx_izz))
        sysMBS.AddBody(self.drum_body)

        # Stage 5: Motor driving drum around Y
        self.fn_rot = chrono.ChFunctionConst(0.0)
        self.rot_motor = chrono.ChLinkMotorRotationSpeed()
        q_y_rot = chrono.QuatFromAngleX(-chrono.CH_PI_2)
        self.rot_motor.Initialize(self.drum_body, self.z_carrier, chrono.ChFramed(init_pos, q_y_rot))
        self.rot_motor.SetSpeedFunction(self.fn_rot)
        sysMBS.AddLink(self.rot_motor)

    def update_motor_phase(self, sim_time: float):
        """Evaluates kinematic planner and updates motor speed constants only upon phase changes."""
        phase, vx, vz, omega = self.planner.evaluate(sim_time)
        if phase != self._current_phase:
            self._current_phase = phase
            self.fn_x.SetConstant(vx)
            self.fn_z.SetConstant(vz)
            self.fn_rot.SetConstant(omega)

    def _register_fsi_body(self):
        """Registers the aligned BCE markers directly with Chrono::CRM."""
        self.fsi_body_index = self.env.problem.AddRigidBody(
            self.drum_body,
            self.bce_markers,
            chrono.ChFramed(chrono.VNULL, chrono.QUNIT),
            False,
        )

        aabb_dims = chrono.ChVector3d(*self.env.cfg["soil_bed"]["active_domain_aabb"])
        if hasattr(self.env.sysFSI, "SetActiveDomain"):
            self.env.sysFSI.SetActiveDomain(aabb_dims)