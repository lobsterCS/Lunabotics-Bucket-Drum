#import chrono_setup
#!/usr/bin/env python3
"""
GPU performance + Irrlicht test for Chrono::FSI-SPH in CRM mode, from PyChrono.

Works with both API generations:
    SetCrmSPH(SoilProperties)   or   SetElasticSPH(ElasticMaterialProperties)

Phases (each case runs in its own subprocess so a crash or leaked GPU state
cannot affect the next case):
  1. Resolution sweep (headless): a box of granular material at several SPH
     spacings. Reports ms/step, real-time factor (sim s per wall s),
     marker-steps/s, and GPU utilization / memory / power / clocks / temperature.
  2. Sphere drop (headless): a rigid sphere dropped on the CRM bed through FSI
     coupling. Checks the bed actually stops the sphere (physics sanity check).
  3. Sphere drop with Irrlicht: same case, rendered. Compares speed with the
     headless run to show the visualization overhead.

NOTE: Irrlicht draws the multibody system (the sphere and a translucent box
marking the bed). It does not draw the SPH markers themselves.

Usage examples
  python test_fsi_crm_gpu_perf.py
  python test_fsi_crm_gpu_perf.py --no-vis
  python test_fsi_crm_gpu_perf.py --spacings 0.04,0.02,0.01 --steps 500
"""
import argparse
import faulthandler
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
faulthandler.enable()

try:
    from test import chrono_setup
except ImportError:
    pass

G = 9.81
RESULT_TAG = "RESULT_JSON:"


# ----------------------------------------------------------------------------
# GPU monitoring
# ----------------------------------------------------------------------------
def nvsmi(query):
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, *query, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def gpu_snapshot():
    """Single-GPU snapshot (first GPU listed)."""
    out = nvsmi(["--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu"])
    if not out:
        return None
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) < 5:
        return None
    return dict(zip(("util", "mem", "power", "clk", "temp"), map(_f, parts)))


def own_pid_on_gpu():
    out = nvsmi(["--query-compute-apps=pid,used_memory"])
    if out is None:
        return None
    return str(os.getpid()) in {l.split(",")[0].strip() for l in out.splitlines() if l.strip()}


class GpuSampler(threading.Thread):
    def __init__(self, period=0.5):
        super().__init__(daemon=True)
        self.period = period
        self.samples = []
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            s = gpu_snapshot()
            if s:
                self.samples.append(s)
            self._stop_evt.wait(self.period)

    def finish(self):
        self._stop_evt.set()
        self.join(timeout=5)
        out = {"n_samples": len(self.samples)}
        for k in ("util", "mem", "power", "clk", "temp"):
            v = [s[k] for s in self.samples if s.get(k) is not None]
            out[k + "_avg"] = sum(v) / len(v) if v else None
            out[k + "_max"] = max(v) if v else None
        return out


# ----------------------------------------------------------------------------
# Chrono setup helpers (used only inside workers)
# ----------------------------------------------------------------------------
def load_api():
    import pychrono as chrono
    import pychrono.fsi as fsi
    F = fsi.ChFsiFluidSystemSPH
    setter = next((n for n in ("SetCrmSPH", "SetElasticSPH") if hasattr(F, n)), None)
    if setter is None:
        raise RuntimeError("no CRM setter (SetCrmSPH / SetElasticSPH) on ChFsiFluidSystemSPH")
    mat_cls = next((getattr(fsi, n) for n in ("SoilProperties", "ElasticMaterialProperties")
                    if isinstance(getattr(fsi, n, None), type)), None)
    if mat_cls is None:
        raise RuntimeError("no CRM material struct (SoilProperties / ElasticMaterialProperties)")
    return chrono, fsi, setter, mat_cls


def make_material(mat_cls):
    mat = mat_cls()
    wanted = {"density": 1700.0, "Young_modulus": 1e6, "Poisson_ratio": 0.3, "mu_I0": 0.04,
              "mu_fric_s": 0.7, "mu_fric_2": 0.7, "average_diam": 0.005, "cohesion_coeff": 0.0}
    for k, v in wanted.items():
        try:
            setattr(mat, k, v)
        except Exception:
            pass
    return mat


def box_side_flags(fsi, open_top):
    """Combine BoxSide enum values; returns (flags, how_found)."""
    found = {}
    holders = [fsi] + [getattr(fsi, n) for n in ("ChFsiProblemSPH", "ChFsiProblemCartesian") if hasattr(fsi, n)]
    for h in holders:
        for n in dir(h):
            low = n.lower()
            if "boxside" in low:
                try:
                    found[low] = int(getattr(h, n))
                except Exception:
                    pass
    all_v = next((v for k, v in found.items() if k.endswith("_all")), None)
    zpos = next((v for k, v in found.items() if k.endswith("z_pos")), None)
    if all_v is None:
        return (0xFF & ~0x10 if open_top else 0xFF), "fallback bitmask (BoxSide enums not found)"
    if open_top and zpos is not None:
        return all_v & ~zpos, "BoxSide enums (top open)"
    return all_v, "BoxSide enums" + ("" if not open_top else " (Z_POS not found; top closed)")


def try_variants(variants):
    last = None
    for fn in variants:
        try:
            return fn()
        except Exception as e:
            last = e
    raise last


def build_problem(a, spacing, with_sphere):
    chrono, fsi, setter, mat_cls = load_api()
    V = chrono.ChVector3d
    L, W, D = a.box

    msys = chrono.ChSystemNSC()
    try:
        msys.SetGravitationalAcceleration(V(0, 0, -G))
    except Exception:
        pass

    problem = try_variants([
        lambda: fsi.ChFsiProblemCartesian(spacing, msys),
        lambda: fsi.ChFsiProblemCartesian(msys, spacing),
        lambda: fsi.ChFsiProblemCartesian(spacing),
    ])
    fluid = problem.GetFluidSystemSPH()
    getattr(fluid, setter)(make_material(mat_cls))
    fluid.SetGravitationalAcceleration(V(0, 0, -G))

    dt = min(1e-3, 0.02 * spacing)
    problem.SetStepSizeCFD(dt)
    try:
        problem.SetStepSizeMBD(dt)
    except Exception:
        pass

    flags, how = box_side_flags(fsi, open_top=with_sphere)
    problem.Construct(V(L, W, D), V(0, 0, 0), flags)

    info = {"dt": dt, "flags_how": how, "setter": setter, "material": mat_cls.__name__}

    body = None
    z0 = None
    if with_sphere:
        r = a.radius
        top_est = D / 2.0  # assumes Construct 'pos' is the box center
        z0 = top_est + r + a.drop_height
        body = try_variants([
            lambda: chrono.ChBodyEasySphere(r, a.sphere_density, True, False),
            lambda: chrono.ChBodyEasySphere(r, a.sphere_density, True),
        ])
        body.SetPos(V(0, 0, z0))
        body.SetFixed(False)
        try:
            body.GetVisualShape(0).SetColor(chrono.ChColor(0.85, 0.25, 0.2))
        except Exception:
            pass
        msys.AddBody(body)
        problem.AddRigidBodySphere(body, V(0, 0, 0), r)

        # translucent marker for the bed (visual only, no collision, not part of FSI)
        marker = chrono.ChBody()
        marker.SetFixed(True)
        marker.SetPos(V(0, 0, 0))
        try:
            shape = try_variants([lambda: chrono.ChVisualShapeBox(L, W, D),
                                  lambda: chrono.ChVisualShapeBox(V(L, W, D))])
            shape.SetColor(chrono.ChColor(0.8, 0.7, 0.45))
            try:
                shape.SetOpacity(0.25)
            except Exception:
                pass
            marker.AddVisualShape(shape)
            msys.AddBody(marker)
        except Exception:
            pass
        info["top_est"] = top_est

    problem.Initialize()
    return dict(chrono=chrono, msys=msys, problem=problem, fluid=fluid, body=body, z0=z0, info=info, dt=dt)


def make_stepper(problem, dt):
    for acc in ("GetSystemFSI", "GetFsiSystemSPH", "GetSystemFsi", "GetFsiSystem"):
        if hasattr(problem, acc):
            try:
                sysf = getattr(problem, acc)()
                sysf.DoStepDynamics(dt)  # probe (counts as one warmup step)
                return (lambda s=sysf: s.DoStepDynamics(dt)), acc
            except Exception:
                continue
    for name in ("DoStepDynamics", "Advance"):
        if hasattr(problem, name):
            try:
                getattr(problem, name)(dt)
                return (lambda n=name: getattr(problem, n)(dt)), name
            except Exception:
                continue
    raise RuntimeError("no working FSI stepping call found; problem methods: "
                       + ", ".join(n for n in dir(problem) if any(t in n.lower() for t in ("step", "advance", "system"))))


def count_markers(obj, a):
    for fn in (lambda: obj["fluid"].GetNumFluidMarkers(), lambda: obj["fluid"].GetNumMarkers()):
        try:
            return int(fn()), "reported"
        except Exception:
            continue
    return None, "unavailable"


def estimate_markers(a, spacing):
    L, W, D = a.box
    return int(round(L / spacing) * round(W / spacing) * round(D / spacing))


# ----------------------------------------------------------------------------
# Workers
# ----------------------------------------------------------------------------
def emit(result):
    print(RESULT_TAG + json.dumps(result), flush=True)


def worker_bench(a):
    base = gpu_snapshot()
    t0 = time.perf_counter()
    obj = build_problem(a, a.spacing, with_sphere=False)
    step, via = make_stepper(obj["problem"], obj["dt"])
    t_init = time.perf_counter() - t0
    after_init = gpu_snapshot()
    on_gpu_init = own_pid_on_gpu()

    for _ in range(a.warmup):
        step()

    sampler = GpuSampler()
    sampler.start()
    times = []
    wall0 = time.perf_counter()
    for _ in range(a.steps):
        t = time.perf_counter()
        step()
        times.append(time.perf_counter() - t)
    wall = time.perf_counter() - wall0
    gpu = sampler.finish()
    on_gpu_run = own_pid_on_gpu()

    n, how = count_markers(obj, a)
    if n is None:
        n, how = estimate_markers(a, a.spacing), "estimated from box/spacing (interior markers only)"
    times.sort()
    emit({
        "spacing": a.spacing, "dt": obj["dt"], "markers": n, "markers_how": how,
        "init_s": t_init, "steps": a.steps, "wall_s": wall,
        "ms_median": 1e3 * statistics.median(times),
        "ms_p95": 1e3 * times[int(0.95 * (len(times) - 1))],
        "rtf": a.steps * obj["dt"] / wall,
        "marker_steps_per_s": n * a.steps / wall,
        "mem_base": base["mem"] if base else None,
        "mem_after_init": after_init["mem"] if after_init else None,
        "pid_listed": bool(on_gpu_init or on_gpu_run) if on_gpu_init is not None else None,
        "gpu": gpu, "stepper": via, "info": obj["info"],
    })


def setup_irrlicht(chrono, msys, title, a):
    import pychrono.irrlicht as irr
    V = chrono.ChVector3d
    vis = irr.ChVisualSystemIrrlicht()
    vis.AttachSystem(msys)
    vis.SetWindowSize(1280, 720)
    vis.SetWindowTitle(title)
    try:
        vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
    except Exception:
        pass
    vis.Initialize()
    for fn in (lambda: vis.AddSkyBox(), lambda: vis.AddTypicalLights()):
        try:
            fn()
        except Exception:
            pass
    L, W, D = a.box
    vis.AddCamera(V(1.4 * L, -1.6 * L, 0.9 * L), V(0, 0, 0.25 * L))
    return vis


def worker_sphere(a, with_vis):
    obj = build_problem(a, a.drop_spacing, with_sphere=True)
    step, via = make_stepper(obj["problem"], obj["dt"])
    dt, body, z0 = obj["dt"], obj["body"], obj["z0"]

    vis = None
    if with_vis:
        vis = setup_irrlicht(obj["chrono"], obj["msys"],
                             f"FSI-SPH CRM sphere drop (spacing {a.drop_spacing})", a)

    n_steps = int(round(a.drop_time / dt))
    zs = [body.GetPos().z]
    sampler = GpuSampler()
    sampler.start()
    closed_early = False
    wall0 = time.perf_counter()
    k = 1  # the probe step in make_stepper already advanced once
    while k < n_steps:
        if vis is not None:
            if not vis.Run():
                closed_early = True
                break
            vis.BeginScene()
            vis.Render()
            vis.EndScene()
        for _ in range(a.render_every if vis is not None else 1):
            step()
            zs.append(body.GetPos().z)
            k += 1
            if k >= n_steps:
                break
    wall = time.perf_counter() - wall0
    gpu = sampler.finish()

    t_sim = (len(zs) - 1) * dt
    tail = min(20, len(zs) - 1)
    vz = (zs[-1] - zs[-1 - tail]) / (tail * dt) if tail > 0 else 0.0
    emit({
        "with_vis": with_vis, "closed_early": closed_early, "spacing": a.drop_spacing, "dt": dt,
        "markers_est": estimate_markers(a, a.drop_spacing),
        "sim_s": t_sim, "wall_s": wall, "rtf": t_sim / wall if wall > 0 else None,
        "z0": zs[0], "z_min": min(zs), "z_final": zs[-1], "vz_final": vz,
        "z_freefall": zs[0] - 0.5 * G * t_sim ** 2, "radius": a.radius,
        "top_est": obj["info"].get("top_est"), "box_depth": a.box[2],
        "gpu": gpu, "stepper": via, "info": obj["info"],
    })


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------
def common_args(a):
    return ["--box", ",".join(map(str, a.box)), "--steps", str(a.steps), "--warmup", str(a.warmup),
            "--radius", str(a.radius), "--sphere-density", str(a.sphere_density),
            "--drop-height", str(a.drop_height), "--drop-time", str(a.drop_time),
            "--drop-spacing", str(a.drop_spacing), "--render-every", str(a.render_every)]


def run_worker(a, mode, extra=(), timeout=900):
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", mode, *common_args(a), *extra]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout}s"
    for line in p.stdout.splitlines():
        if line.startswith(RESULT_TAG):
            return json.loads(line[len(RESULT_TAG):]), ""
    tail = "\n".join((p.stderr or p.stdout).strip().splitlines()[-12:])
    return None, f"exit code {p.returncode}\n{tail}"


def fmt(x, nd=1, none="-"):
    return none if x is None else f"{x:.{nd}f}"


def main_orchestrator(a):
    checks = []

    def check(status, name, detail=""):
        checks.append((status, name, detail))
        print(f"[{status:4}] {name}" + (f"  --  {detail}" if detail else ""), flush=True)

    print("=== Environment ===")
    snap = nvsmi(["--query-gpu=name,driver_version,memory.total,power.limit"])
    print("GPU:", (snap or "nvidia-smi unavailable").replace("\n", " | "))
    print("Python:", sys.version.split()[0])
    try:
        import pychrono as chrono
        import pychrono.fsi as fsi
        F = fsi.ChFsiFluidSystemSPH
        setter = next((n for n in ("SetCrmSPH", "SetElasticSPH") if hasattr(F, n)), None)
        print("pychrono:", getattr(chrono, "__version__", "?"), "| CRM setter:", setter)
        check("PASS" if setter else "FAIL", "CRM setter available", setter or "none")
        irr_ok = True
        try:
            import pychrono.irrlicht  # noqa: F401
        except Exception:
            irr_ok = False
        print("pychrono.irrlicht:", "available" if irr_ok else "NOT available")
    except Exception as e:
        check("FAIL", "import pychrono.fsi", repr(e))
        return 1
    if not setter:
        return 1

    # ---- Phase 1: sweep
    print("\n=== Phase 1: resolution sweep (headless) ===")
    print(f"box {a.box} m, {a.warmup} warmup + {a.steps} timed steps per case\n")
    rows = []
    for s in a.spacings:
        print(f"  running spacing {s} ...", flush=True)
        res, err = run_worker(a, "bench", ["--spacing", str(s)])
        if res is None:
            check("FAIL", f"sweep case spacing={s}", err)
            continue
        rows.append(res)

    if rows:
        hdr = (f"{'spacing':>8} {'markers':>9} {'dt(s)':>8} {'init(s)':>8} {'ms/step':>8} {'p95':>7} "
               f"{'RTF':>7} {'Mmk-st/s':>9} {'util%':>6} {'mem+MiB':>8} {'W':>6} {'MHz':>6} {'Tmax':>5}")
        print("\n" + hdr)
        for r in rows:
            g = r["gpu"]
            mem_d = (g["mem_max"] - r["mem_base"]) if (g.get("mem_max") is not None and r["mem_base"] is not None) else None
            print(f"{r['spacing']:>8.3f} {r['markers']:>9d} {r['dt']:>8.1e} {r['init_s']:>8.2f} "
                  f"{r['ms_median']:>8.2f} {r['ms_p95']:>7.2f} {r['rtf']:>7.3f} "
                  f"{r['marker_steps_per_s'] / 1e6:>9.2f} {fmt(g.get('util_avg'), 0):>6} "
                  f"{fmt(mem_d, 0):>8} {fmt(g.get('power_avg'), 0):>6} {fmt(g.get('clk_avg'), 0):>6} "
                  f"{fmt(g.get('temp_max'), 0):>5}")
        print("\nMarker counts:", "; ".join(f"{r['spacing']}: {r['markers']} ({r['markers_how']})" for r in rows))
        print("RTF = simulated seconds per wall second (higher is faster). "
              "Mmk-st/s = millions of marker-steps per second, assuming one CFD step per outer step.")
        print(f"GPU sampled every 0.5 s via nvidia-smi ({rows[-1]['gpu']['n_samples']} samples in the last case).")

        if len(rows) == len(a.spacings):
            check("PASS", "all sweep cases completed")
        # GPU engaged evidence
        big = rows[-1]
        g = big["gpu"]
        mem_d = (big["mem_after_init"] - big["mem_base"]) if (big["mem_after_init"] and big["mem_base"]) else None
        engaged = (mem_d is not None and mem_d > 50) and (g.get("util_max") or 0) >= 20
        pid = big["pid_listed"]
        check("PASS" if engaged else "WARN", "GPU engaged in largest case",
              f"mem +{fmt(mem_d, 0)} MiB, util avg/max {fmt(g.get('util_avg'), 0)}/{fmt(g.get('util_max'), 0)}%, "
              f"pid listed: {pid}")
        # scaling
        if len(rows) >= 2:
            print("\nScaling (consecutive cases):")
            import math
            mono = True
            for r0, r1 in zip(rows, rows[1:]):
                dm = r1["markers"] / r0["markers"]
                dt_ratio = r1["ms_median"] / r0["ms_median"]
                if dm > 1.05 and dt_ratio > 0:
                    expo = math.log(dt_ratio) / math.log(dm)
                    print(f"  {r0['markers']} -> {r1['markers']} markers: time x{dt_ratio:.2f} "
                          f"(exponent {expo:.2f}; ~1.0 = GPU saturated, <1 = not saturated yet/launch-bound)")
                if r1["ms_median"] < r0["ms_median"] * 0.9:
                    mono = False
            check("PASS" if mono else "WARN", "ms/step grows with marker count",
                  "" if mono else "non-monotonic: check for thermal/power throttling or background GPU load")
            if big["gpu"].get("temp_max") and big["gpu"]["temp_max"] >= 85:
                check("WARN", "GPU temperature high", f"{big['gpu']['temp_max']:.0f} C; laptop GPUs throttle")

    # ---- Phase 2 / 3: sphere drop
    def report_sphere(tag, r):
        print(f"\n  [{tag}] sim {r['sim_s']:.2f} s in {r['wall_s']:.2f} s wall (RTF {fmt(r['rtf'], 3)}), "
              f"~{r['markers_est']} markers, dt {r['dt']:.1e}")
        print(f"      sphere z: start {r['z0']:.3f}, min {r['z_min']:.3f}, final {r['z_final']:.3f} m, "
              f"final vz {r['vz_final']:.3f} m/s; pure free fall would end at {r['z_freefall']:.3f} m")
        g = r["gpu"]
        print(f"      GPU util avg/max {fmt(g.get('util_avg'), 0)}/{fmt(g.get('util_max'), 0)}%, "
              f"power avg {fmt(g.get('power_avg'), 0)} W, SM clock avg {fmt(g.get('clk_avg'), 0)} MHz")

    def judge_sphere(tag, r):
        supported = (r["z_final"] - r["z_freefall"]) > 0.5
        stopped = abs(r["vz_final"]) < 0.5
        above_floor = r["z_final"] > -r["box_depth"]
        if supported and above_floor:
            check("PASS", f"sphere supported by CRM bed ({tag})",
                  f"z_final {r['z_final']:.3f} m vs free-fall {r['z_freefall']:.3f} m; "
                  f"{'at rest' if stopped else 'still moving (increase --drop-time)'}")
        else:
            check("FAIL", f"sphere NOT supported by CRM bed ({tag})",
                  f"z_final {r['z_final']:.3f} m, free-fall {r['z_freefall']:.3f} m: "
                  "FSI coupling may be inactive or the sphere fell through")

    headless = vis_res = None
    if not a.skip_sphere:
        print("\n=== Phase 2: sphere drop on CRM bed (headless) ===")
        headless, err = run_worker(a, "sphere", timeout=1800)
        if headless is None:
            check("FAIL", "sphere drop (headless)", err)
            print("  (AddRigidBodySphere may be missing or have a different signature in this build.)")
        else:
            report_sphere("headless", headless)
            judge_sphere("headless", headless)

    if not a.skip_sphere and not a.no_vis:
        print("\n=== Phase 3: sphere drop with Irrlicht ===")
        if not irr_ok:
            check("WARN", "Irrlicht phase skipped", "pychrono.irrlicht not available in this build")
        else:
            print("  A window opens; close it to end early. Irrlicht shows the sphere and a translucent bed box,\n"
                  "  not the SPH markers. If the box and the sphere's resting height disagree, Construct()'s\n"
                  "  position reference is not the box center in this build; the physics check is unaffected.")
            vis_res, err = run_worker(a, "sphere-vis", timeout=3600)
            if vis_res is None:
                check("FAIL", "sphere drop (Irrlicht)", err)
            else:
                report_sphere("Irrlicht", vis_res)
                if vis_res["closed_early"]:
                    check("INFO", "Irrlicht window closed before the run finished", "results are partial")
                else:
                    judge_sphere("Irrlicht", vis_res)
                    if headless and headless.get("rtf") and vis_res.get("rtf"):
                        ov = 100.0 * (1.0 - vis_res["rtf"] / headless["rtf"])
                        check("INFO", "visualization overhead", f"RTF {headless['rtf']:.3f} -> {vis_res['rtf']:.3f} ({ov:+.0f}% slower)")

    print("\n=== Summary ===")
    counts = {}
    for s, _, _ in checks:
        counts[s] = counts.get(s, 0) + 1
    print("  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    fails = [c for c in checks if c[0] == "FAIL"]
    warns = [c for c in checks if c[0] == "WARN"]
    if fails:
        print("\nVerdict: problems found:")
        for _, n, d in fails:
            print(f"  - {n}: {d}")
    elif warns:
        print("\nVerdict: ran, with items to review (see WARN lines).")
    else:
        print("\nVerdict: CRM ran on the GPU; performance numbers above.")
    return 1 if fails else 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", choices=["bench", "sphere", "sphere-vis"], help=argparse.SUPPRESS)
    p.add_argument("--spacing", type=float, help=argparse.SUPPRESS)
    p.add_argument("--spacings", default="0.05,0.03,0.02,0.015,0.01",
                   help="comma-separated SPH spacings (m) for the sweep")
    p.add_argument("--box", default="1.0,0.5,0.3", help="bed size L,W,D in m")
    p.add_argument("--steps", type=int, default=300, help="timed steps per sweep case")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--radius", type=float, default=0.06, help="sphere radius (m)")
    p.add_argument("--sphere-density", type=float, default=2500.0)
    p.add_argument("--drop-height", type=float, default=0.2, help="gap between sphere bottom and bed top (m)")
    p.add_argument("--drop-time", type=float, default=0.8, help="simulated seconds for the drop")
    p.add_argument("--drop-spacing", type=float, default=0.02)
    p.add_argument("--render-every", type=int, default=10, help="steps between Irrlicht frames")
    p.add_argument("--no-vis", action="store_true", help="skip the Irrlicht phase")
    p.add_argument("--skip-sphere", action="store_true", help="sweep only")
    a = p.parse_args()
    a.box = tuple(float(x) for x in str(a.box).split(","))
    a.spacings = [float(x) for x in str(a.spacings).split(",")]
    return a


if __name__ == "__main__":
    args = parse_args()
    if args.worker == "bench":
        worker_bench(args)
    elif args.worker == "sphere":
        worker_sphere(args, with_vis=False)
    elif args.worker == "sphere-vis":
        worker_sphere(args, with_vis=True)
    else:
        sys.exit(main_orchestrator(args))