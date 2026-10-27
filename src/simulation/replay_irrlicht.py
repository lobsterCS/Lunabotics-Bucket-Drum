import os
import csv
import glob
import argparse
import numpy as np
import pychrono as chrono
import pychrono.irrlicht as chronoirr

def load_trajectory(traj_csv_path: str) -> list:
    frames = []
    with open(traj_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append({
                "frame": int(row["frame"]),
                "time": float(row["time_s"]),
                "pos": chrono.ChVector3d(float(row["px"]), float(row["py"]), float(row["pz"])),
                "rot": chrono.ChQuaterniond(
                    float(row["e0"]), float(row["e1"]), float(row["e2"]), float(row["e3"])
                ),
            })
    return frames

def replay_simulation_irrlicht(
    run_output_dir: str,
    obj_path: str,
    pit_dimensions=(1.60, 0.85, 0.25),
    pit_center=(0.35, 0.0, -0.125),
    particle_stride: int = 2,
):
    """
    Replays a headless CRM bucket drum simulation in Irrlicht using exported
    6-DOF trajectory CSV and compressed NumPy SPH particle clouds.
    """
    traj_path = os.path.join(run_output_dir, "drum_trajectory.csv")
    particles_dir = os.path.join(run_output_dir, "particles")

    if not os.path.isfile(traj_path):
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    traj_frames = load_trajectory(traj_path)
    npz_files = sorted(glob.glob(os.path.join(particles_dir, "frame_*.npz")))

    if not traj_frames:
        raise RuntimeError("No trajectory frames found to replay.")

    # 1. Initialize Lightweight Visualization System
    sys_vis = chrono.ChSystemNSC()
    sys_vis.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))

    # 2. Add Wireframe Regolith Pit Boundary Box
    pit_body = chrono.ChBody()
    pit_body.SetFixed(True)
    pit_body.SetPos(chrono.ChVector3d(*pit_center))
    pit_shape = chrono.ChVisualShapeBox(*pit_dimensions)
    pit_shape.SetColor(chrono.ChColor(0.35, 0.30, 0.25))
    pit_body.AddVisualShape(pit_shape)
    sys_vis.AddBody(pit_body)

    # 3. Load Bucket Drum Visual Mesh (.obj)
    drum_body = chrono.ChBody()
    drum_body.SetFixed(True)
    trimesh = chrono.ChTriangleMeshConnected()
    trimesh.LoadWavefrontMesh(obj_path, True, True)
    mesh_shape = chrono.ChVisualShapeTriangleMesh()
    mesh_shape.SetMesh(trimesh)
    mesh_shape.SetColor(chrono.ChColor(0.75, 0.78, 0.82))
    drum_body.AddVisualShape(mesh_shape)
    sys_vis.AddBody(drum_body)

    # 4. Initialize ChParticleCloud for Regolith Particles
    p_cloud = chrono.ChParticleCloud()
    p_cloud.SetFixed(True)
    sphere_vis = chrono.ChVisualShapeSphere(0.0035)
    sphere_vis.SetColor(chrono.ChColor(0.55, 0.52, 0.48))
    p_cloud.AddVisualShape(sphere_vis)

    # Pre-allocate particles based on first frame
    if npz_files:
        first_data = np.load(npz_files[0])
        pts0 = first_data["pos"][::particle_stride]
        p_cloud.ResizeNparticles(len(pts0))
        for i, pt in enumerate(pts0):
            p_cloud.Particle(i).SetPos(chrono.ChVector3d(float(pt[0]), float(pt), float(pt)))
    sys_vis.Add(p_cloud)

    # 5. Configure Irrlicht Window
    vis = chronoirr.ChVisualSystemIrrlicht()
    vis.AttachSystem(sys_vis)
    vis.SetWindowSize(1280, 720)
    vis.SetWindowTitle(f"Chrono::CRM Bucket Drum Replay - {os.path.basename(run_output_dir)}")
    vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
    vis.Initialize()
    vis.AddSkyBox()
    vis.AddCamera(chrono.ChVector3d(0.25, -1.15, 0.35), chrono.ChVector3d(0.25, 0.0, -0.05))
    vis.AddTypicalLights()

    # 6. Playback Loop
    idx = 0
    n_frames = len(traj_frames)
    print(f"Replaying {n_frames} frames in Irrlicht (stride={particle_stride})...")

    while vis.Run():
        frame_data = traj_frames[idx]
        drum_body.SetPos(frame_data["pos"])
        drum_body.SetRot(frame_data["rot"])

        if idx < len(npz_files):
            p_data = np.load(npz_files[idx])
            pts = p_data["pos"][::particle_stride]
            n_pts = min(len(pts), p_cloud.GetNumParticles())
            for i in range(n_pts):
                p_cloud.Particle(i).SetPos(
                    chrono.ChVector3d(float(pts[i, 0]), float(pts), float(pts))
                )

        vis.BeginScene()
        vis.Render()
        vis.EndScene()

        idx = (idx + 1) % n_frames

if __name__ == "__main__":
    # Fix working directory to project root when run directly from PyCharm
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(str(__file__)), ".."))
    os.chdir(PROJECT_ROOT)

    # Set default run to inspect when clicking Run in PyCharm
    DEFAULT_RUN_DIR = "output/test_run_01_open_section_drum"
    DEFAULT_OBJ_PATH = "assets/bucket drums/RASSOR_SPIKEY.obj"

    parser = argparse.ArgumentParser(description="Replay exported CRM Bucket Drum run in Irrlicht")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, help="Path to output/<run_id> directory")
    parser.add_argument("--obj", default=DEFAULT_OBJ_PATH, help="Path to bucket drum .obj mesh")
    parser.add_argument("--stride", type=int, default=2, help="Downsample stride for SPH particles")
    args = parser.parse_args()

    replay_simulation_irrlicht(args.run_dir, args.obj, particle_stride=args.stride)
