"""
DEM (SMC) simulation of a RASSOR-style bucket drum collecting Black Point 1 (BP-1)
lunar regolith simulant under Earth gravity, visualized with Irrlicht.

Targets PyChrono 9.x (ChVector3d / ChContactMaterialSMC / ChCollisionShape* API).

Scenario (world frame, Z up):
  1. A bed of BP-1-like particles is poured into a bin and allowed to settle (headless).
  2. The drum is created above the bed and driven by three motors:
       plunge (down)  ->  drag (forward + rotating)  ->  lift (up, rotation stopped)
  3. Mass of regolith captured inside the drum is printed once per simulated second.
"""
import chrono_setup  # noqa: F401  (must be imported before pychrono)

import math
import os
import random

import pychrono as chrono
import pychrono.irrlicht as chronoirr


# =============================================================================
# User parameters
# =============================================================================
MESH_FILE = "../simulation/RASSOR_SPIKEY.obj"
MESH_SCALE = 1.0          # set to 0.001 if the OBJ is in millimetres
MESH_AXIS = "y"           # drum's axis of revolution *in the OBJ* ('x', 'y' or 'z')
DRUM_MASS = 200.0         # kg; large so contact forces barely perturb the prescribed motion

GRAVITY = 9.81            # m/s^2  (Earth; use 1.62 for the Moon)

# --- BP-1 surrogate (coarse-grained) -----------------------------------------
# Real BP-1 has d50 ~ 70 micron, which is far too fine for a DEM run in Python.
# We simulate larger spheres with the right density and an approximate
# friction / cohesion level; calibrate against angle of repose for real work.
PARTICLE_RADIUS = 0.0075   # m (mean); radii vary +/-20 % to avoid crystallization
PARTICLE_DENSITY = 2900.0 # kg/m^3  (BP-1 particle specific gravity ~2.9)
BED_DEPTH = 0.06          # m, settled depth target (sets particle count)
YOUNG = 1.0e7             # Pa, softened for a larger time step
POISSON = 0.25
FRICTION = 0.6            # sliding friction (BP-1 internal friction angle ~35 deg)
ROLL_FRICTION = 0.05      # mimics angular grains
RESTITUTION = 0.2
ADHESION = 0.02           # N per contact (constant model) ~ weak lunar-dust cohesion

# --- Time stepping -----------------------------------------------------------
DT = 5.0e-5               # s
SETTLE_TIME = 1.0         # s of headless settling before the drum is added
RENDER_DT = 0.04          # s of simulated time between rendered frames

# --- Drum trajectory ---------------------------------------------------------
CLEARANCE = 0.03          # m between drum bottom and bed surface at start
DIG_DEPTH = 0.03          # m the drum bottom plunges below the bed surface
DRAG_DISTANCE = 0.6       # m
LIFT_HEIGHT = 0.30        # m
V_PLUNGE = 0.03           # m/s
V_DRAG = 0.08             # m/s
V_LIFT = 0.05             # m/s
OMEGA = -1.0              # rad/s about +Y. Negative = drum bottom moves forward
                          # (counter-rotating dig). Flip sign if your scoops face the other way.
HOLD_TIME = 2.0           # s to keep running after lift


# =============================================================================
# Helpers
# =============================================================================
def find_mesh(name):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (name, os.path.join(here, name)):
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Could not find {name} in the working directory or next to this script")


def mesh_bbox(mesh):
    vs = list(mesh.GetCoordsVertices())
    xs = [v.x for v in vs]
    ys = [v.y for v in vs]
    zs = [v.z for v in vs]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def scale_matrix(s):
    """3x3 uniform-scale matrix (the Transform(displ, double) overload isn't exposed in all builds)."""
    if s == 1.0:
        return chrono.ChMatrix33d(chrono.QUNIT)  # identity
    for make in (lambda: chrono.ChMatrix33d(chrono.ChVector3d(s, s, s)),
                 lambda: chrono.ChMatrix33d(s)):
        try:
            return make()
        except Exception:
            pass
    raise RuntimeError("Could not build a scale matrix; rescale the OBJ externally and set MESH_SCALE = 1.0")


def load_drum_mesh():
    """Load the OBJ, orient so the drum axis is world Y, scale, and centre on its bbox."""
    mesh = chrono.ChTriangleMeshConnected.CreateFromWavefrontFile(find_mesh(MESH_FILE), True, False)
    origin = chrono.ChVector3d(0, 0, 0)

    if MESH_AXIS == "z":
        mesh.Transform(origin, chrono.ChMatrix33d(chrono.QuatFromAngleX(-math.pi / 2)))
    elif MESH_AXIS == "x":
        mesh.Transform(origin, chrono.ChMatrix33d(chrono.QuatFromAngleZ(math.pi / 2)))

    if MESH_SCALE != 1.0:
        mesh.Transform(origin, scale_matrix(MESH_SCALE))

    lo, hi = mesh_bbox(mesh)
    c = [(lo[i] + hi[i]) / 2 for i in range(3)]
    mesh.Transform(chrono.ChVector3d(-c[0], -c[1], -c[2]), scale_matrix(1.0))

    lo, hi = mesh_bbox(mesh)
    ext = [hi[i] - lo[i] for i in range(3)]
    radius = 0.5 * max(ext[0], ext[2])
    width = ext[1]
    return mesh, radius, width


def add_box(body, mat, center, size, color=None):
    """Add a box collision shape (and optional visual) to a body."""
    frame = chrono.ChFramed(chrono.ChVector3d(*center), chrono.QUNIT)
    body.AddCollisionShape(chrono.ChCollisionShapeBox(mat, *size), frame)
    if color is not None:
        vis = chrono.ChVisualShapeBox(*size)
        vis.SetColor(chrono.ChColor(*color))
        body.AddVisualShape(vis, frame)


# =============================================================================
# Build the system
# =============================================================================
random.seed(1)

system = chrono.ChSystemSMC()
system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -GRAVITY))
system.SetContactForceModel(chrono.ChSystemSMC.Hertz)
system.SetAdhesionForceModel(chrono.ChSystemSMC.Constant)
system.SetTangentialDisplacementModel(chrono.ChSystemSMC.MultiStep)

chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.002)
chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.001)

# Regolith / wall / drum contact material (shared for simplicity)
mat = chrono.ChContactMaterialSMC()
mat.SetYoungModulus(YOUNG)
mat.SetPoissonRatio(POISSON)
mat.SetFriction(FRICTION)
mat.SetRollingFriction(ROLL_FRICTION)
mat.SetRestitution(RESTITUTION)
mat.SetAdhesion(ADHESION)

# --- Drum mesh and bin sizing -------------------------------------------------
drum_mesh, DRUM_R, DRUM_W = load_drum_mesh()
print(f"Drum: radius ~ {DRUM_R:.3f} m, width ~ {DRUM_W:.3f} m (after scale/orientation)")

X_START = DRUM_R + 0.15
LX = X_START + DRAG_DISTANCE + DRUM_R + 0.15
LY = DRUM_W + 0.20
WALL_H = 0.25
T = 0.05  # wall thickness

# --- Static bin ---------------------------------------------------------------
ground = chrono.ChBody()
ground.SetFixed(True)
ground.EnableCollision(True)
dark = (0.25, 0.25, 0.28)
add_box(ground, mat, (LX / 2, 0, -T / 2), (LX + 2 * T, LY + 2 * T, T), color=dark)                 # floor
add_box(ground, mat, (-T / 2, 0, WALL_H / 2), (T, LY + 2 * T, WALL_H), color=(0.4, 0.4, 0.45))     # -x wall
add_box(ground, mat, (LX + T / 2, 0, WALL_H / 2), (T, LY + 2 * T, WALL_H), color=(0.4, 0.4, 0.45)) # +x wall
add_box(ground, mat, (LX / 2, LY / 2 + T / 2, WALL_H / 2), (LX, T, WALL_H), color=(0.4, 0.4, 0.45))  # +y wall
add_box(ground, mat, (LX / 2, -LY / 2 - T / 2, WALL_H / 2), (LX, T, WALL_H))                       # -y wall (invisible: camera side)
system.AddBody(ground)

# --- Regolith particles -------------------------------------------------------
particles = []  # (body, mass)


def add_particle(pos, r):
    m = PARTICLE_DENSITY * (4.0 / 3.0) * math.pi * r ** 3
    b = chrono.ChBody()
    b.SetMass(m)
    i = 0.4 * m * r * r
    b.SetInertiaXX(chrono.ChVector3d(i, i, i))
    b.SetPos(chrono.ChVector3d(*pos))
    b.AddCollisionShape(chrono.ChCollisionShapeSphere(mat, r))
    b.EnableCollision(True)
    vs = chrono.ChVisualShapeSphere(r)
    g = random.uniform(0.18, 0.40)  # BP-1 is dark grey
    vs.SetColor(chrono.ChColor(g, g, g * 1.03))
    b.AddVisualShape(vs)
    system.AddBody(b)
    particles.append((b, m))


r_max = 1.2 * PARTICLE_RADIUS
spacing = 2.4 * r_max
n_layers = max(1, math.ceil(2.0 * BED_DEPTH / spacing))  # loose lattice ~half as dense as settled bed
nx = int((LX - 0.02) / spacing)
ny = int((LY - 0.02) / spacing)
for k in range(n_layers):
    for j in range(ny):
        for i in range(nx):
            jit = 0.15 * spacing
            x = 0.01 + spacing * (i + 0.5) + random.uniform(-jit, jit)
            y = -LY / 2 + 0.01 + spacing * (j + 0.5) + random.uniform(-jit, jit)
            z = r_max + spacing * k + random.uniform(0, jit)
            add_particle((x, y, z), PARTICLE_RADIUS * random.uniform(0.8, 1.2))
print(f"Created {len(particles)} particles ({sum(m for _, m in particles):.1f} kg)")

# --- Settle the bed headless (fast), then add the drum ------------------------
print(f"Settling bed for {SETTLE_TIME:.1f} s of simulated time ...")
for step in range(int(SETTLE_TIME / DT)):
    system.DoStepDynamics(DT)

zs = sorted(b.GetPos().z for b, _ in particles)
bed_top = zs[int(0.98 * len(zs))] + PARTICLE_RADIUS
print(f"Settled bed surface at z ~ {float(bed_top):.3f} m")

# --- Drum (kinematically driven) ----------------------------------------------
# The drum's pose and velocity are prescribed from the stage schedule before every
# step. This avoids chains of motor links, whose axis/sign conventions are easy to get wrong.
start = chrono.ChVector3d(X_START, 0, bed_top + CLEARANCE + DRUM_R)

drum = chrono.ChBody()
drum.SetMass(DRUM_MASS)
i_ax = 0.5 * DRUM_MASS * DRUM_R ** 2
i_tr = DRUM_MASS * (3 * DRUM_R ** 2 + DRUM_W ** 2) / 12.0
drum.SetInertiaXX(chrono.ChVector3d(i_tr, i_ax, i_tr))  # axis of revolution = Y
drum.SetPos(start)
drum.AddCollisionShape(chrono.ChCollisionShapeTriangleMesh(mat, drum_mesh, False, False, 0.0))
drum.EnableCollision(True)
drum_vis = chrono.ChVisualShapeTriangleMesh()
drum_vis.SetMesh(drum_mesh)
drum_vis.SetMutable(False)
drum_vis.SetColor(chrono.ChColor(0.85, 0.55, 0.15))
drum.AddVisualShape(drum_vis)
system.AddBody(drum)

# (name, duration, vx, vz, omega)
T_PLUNGE = (CLEARANCE + DIG_DEPTH) / V_PLUNGE
T_DRAG = DRAG_DISTANCE / V_DRAG
T_LIFT = LIFT_HEIGHT / V_LIFT
STAGES = [
    ("plunge", T_PLUNGE, 0.0, -V_PLUNGE, OMEGA),
    ("drag", T_DRAG, V_DRAG, 0.0, OMEGA),
    ("lift", T_LIFT, 0.0, V_LIFT, 0.0),
    ("hold", HOLD_TIME, 0.0, 0.0, 0.0),
]
T_TOTAL = sum(s[1] for s in STAGES)
current_stage = None

_set_vel = getattr(drum, "SetPosDt", None) or getattr(drum, "SetPos_dt")


def drum_state(tau):
    """Analytic pose/velocity of the drum at time tau (integrates the piecewise-constant stages)."""
    x, z, theta = start.x, start.z, 0.0
    vx = vz = w = 0.0
    name = "done"
    t = 0.0
    for nm, dur, svx, svz, sw in STAGES:
        dt_in = min(max(tau - t, 0.0), dur)
        x += svx * dt_in
        z += svz * dt_in
        theta += sw * dt_in
        if t <= tau < t + dur:
            vx, vz, w, name = svx, svz, sw, nm
        t += dur
    return x, z, theta, vx, vz, w, name


def drive_drum(tau):
    global current_stage
    x, z, theta, vx, vz, w, name = drum_state(tau)
    drum.SetPos(chrono.ChVector3d(x, start.y, z))
    drum.SetRot(chrono.QuatFromAngleY(theta))  # rotation about world +Y
    _set_vel(chrono.ChVector3d(vx, 0, vz))
    drum.SetAngVelParent(chrono.ChVector3d(0, w, 0))
    if name != current_stage:
        current_stage = name
        print(f"[t = {float(tau):5.2f} s] stage: {name}")


def captured_mass():
    c = drum.GetPos()
    total, n = 0.0, 0
    for b, m in particles:
        p = b.GetPos()
        dx, dy, dz = p.x - c.x, p.y - c.y, p.z - c.z
        if abs(dy) < 0.5 * DRUM_W and dx * dx + dz * dz < (0.9 * DRUM_R) ** 2:
            total += m
            n += 1
    return total, n


# =============================================================================
# Visualization and main loop
# =============================================================================
print(f"Drum bottom starts at z = {float(start.z - DRUM_R):.3f} m, bed surface at z = {float(bed_top):.3f} m, "
      f"plunge ends with drum bottom at z = {float(bed_top - DIG_DEPTH):.3f} m")

vis = chronoirr.ChVisualSystemIrrlicht()
vis.AttachSystem(system)
vis.SetWindowSize(1280, 720)
vis.SetWindowTitle("RASSOR bucket drum - BP-1 regolith DEM (Earth gravity)")
vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
vis.Initialize()
try:
    vis.AddSkyBox()  # needs the Chrono data path (normally configured by chrono_setup)
except Exception:
    pass
vis.AddCamera(chrono.ChVector3d(LX * 0.55, -1.8, 0.75), chrono.ChVector3d(LX * 0.45, 0, 0.12))
vis.AddTypicalLights()

steps_per_frame = max(1, int(round(RENDER_DT / DT)))
t_start = system.GetChTime()
next_report = 1.0

while vis.Run():
    vis.BeginScene()
    vis.Render()
    vis.EndScene()

    for _ in range(steps_per_frame):
        drive_drum(system.GetChTime() - t_start)
        system.DoStepDynamics(DT)

    tau = system.GetChTime() - t_start
    if tau >= next_report:
        mass, n = captured_mass()
        d = drum.GetPos()
        print(f"t = {tau:5.2f} s | drum x = {d.x:.3f}, z = {d.z:.3f} m | in drum: {n} particles, {mass:.2f} kg")
        next_report += 1.0