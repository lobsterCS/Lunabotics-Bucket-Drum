#!/usr/bin/env python3
"""
v2: test CRM + GPU support in pychrono.fsi without needing pychrono.vehicle.

Handles both API generations:
  CRM setter:   SetCrmSPH(SoilProperties)  or  SetElasticSPH(ElasticMaterialProperties)
Runs a small CRM box with ChFsiProblemCartesian, steps it, and checks GPU use.

Status: PASS works | WARN could not confirm | FAIL broken/missing | INFO note
Exit code 0 only if there are no FAILs.
"""
import faulthandler
import os
import shutil
import subprocess
import sys
import time

faulthandler.enable()

try:
    import chrono_setup  # noqa: F401  (optional personal env helper)
except ImportError:
    pass

results = []


def record(status, name, detail=""):
    results.append((status, name, detail))
    print(f"[{status:4}] {name}" + (f"  --  {detail}" if detail else ""), flush=True)


def find(obj, *terms):
    return sorted(n for n in dir(obj) if any(t in n.lower() for t in terms))


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


def gpu_mem_used_mib():
    out = nvsmi(["--query-gpu=memory.used"])
    try:
        return sum(int(x) for x in out.splitlines()) if out else None
    except ValueError:
        return None


def own_pid_on_gpu():
    out = nvsmi(["--query-compute-apps=pid,used_memory"])
    if out is None:
        return None
    return str(os.getpid()) in {l.split(",")[0].strip() for l in out.splitlines() if l.strip()}


def try_each(label, attempts):
    errors = []
    for i, fn in enumerate(attempts):
        try:
            res = fn()
            record("PASS", label, f"variant #{i + 1}")
            return True, res
        except Exception as e:
            errors.append(f"#{i + 1}: {type(e).__name__}: {str(e)[:80]}")
    record("FAIL", label, " | ".join(errors))
    return False, None


# --------------------------------------------------------------------------
print("\n=== Stage 1: imports ===")
try:
    import pychrono as chrono
    record("PASS", "import pychrono", getattr(chrono, "__version__", ""))
except Exception as e:
    record("FAIL", "import pychrono", repr(e))
    sys.exit(1)
try:
    import pychrono.fsi as fsi
    record("PASS", "import pychrono.fsi")
except Exception as e:
    record("FAIL", "import pychrono.fsi", repr(e))
    sys.exit(1)
try:
    import pychrono.vehicle  # noqa: F401
    record("INFO", "pychrono.vehicle", "available (not required by this test)")
except Exception:
    record("INFO", "pychrono.vehicle", "not built (not required by this test)")

# --------------------------------------------------------------------------
print("\n=== Stage 2: API surface ===")
for cls in ("ChFsiFluidSystemSPH", "ChFsiSystemSPH", "ChFsiProblemSPH", "ChFsiProblemCartesian"):
    record("PASS" if hasattr(fsi, cls) else "FAIL", f"fsi.{cls}")

rheo = find(fsi, "rheologycrm")
record("PASS" if rheo else "FAIL", "CRM rheology enums", ", ".join(rheo))

F = getattr(fsi, "ChFsiFluidSystemSPH", None)
setter_name = next((n for n in ("SetCrmSPH", "SetElasticSPH") if F is not None and hasattr(F, n)), None)
record("PASS" if setter_name else "FAIL", "CRM mode setter", setter_name or "neither SetCrmSPH nor SetElasticSPH found")
if setter_name:
    record("INFO", f"{setter_name} signature", (getattr(F, setter_name).__doc__ or "").strip().splitlines()[0])
record("INFO", "SetCfdSPH", "present" if F is not None and hasattr(F, "SetCfdSPH") else "missing")

MatCls = None
for n in ("SoilProperties", "ElasticMaterialProperties"):
    if isinstance(getattr(fsi, n, None), type):
        MatCls = getattr(fsi, n)
        break
if MatCls is None:
    for n in find(fsi, "soilprop", "elasticmaterial", "materialprop"):
        if isinstance(getattr(fsi, n), type):
            MatCls = getattr(fsi, n)
            break
record("PASS" if MatCls else "FAIL", "CRM material struct", MatCls.__name__ if MatCls else "not found")

mat = None
if MatCls is not None:
    try:
        mat = MatCls()
        fields = [a for a in dir(mat) if not a.startswith("_") and a not in ("this", "thisown")
                  and not callable(getattr(mat, a, None))]
        record("INFO", "material fields", ", ".join(fields))
        wanted = {"density": 1700.0, "Young_modulus": 1e6, "Poisson_ratio": 0.3, "mu_I0": 0.04,
                  "mu_fric_s": 0.7, "mu_fric_2": 0.7, "average_diam": 0.005, "cohesion_coeff": 0.0}
        ok_f, miss_f = [], []
        for k, v in wanted.items():
            try:
                if k not in fields:
                    raise AttributeError
                setattr(mat, k, v)
                ok_f.append(k)
            except Exception:
                miss_f.append(k)
        record("PASS" if ok_f else "WARN", "material fields set", ", ".join(ok_f))
        if miss_f:
            record("INFO", "fields not matched (renamed or absent)", ", ".join(miss_f))
    except Exception as e:
        record("FAIL", "instantiate material struct", repr(e))

# --------------------------------------------------------------------------
print("\n=== Stage 3: GPU environment ===")
gpu = nvsmi(["--query-gpu=name,driver_version,memory.total"])
record("PASS" if gpu else "FAIL", "nvidia-smi sees an NVIDIA GPU", (gpu or "none").replace("\n", " | "))
mem_before = gpu_mem_used_mib()
record("INFO", "GPU memory before (MiB)", str(mem_before))

# --------------------------------------------------------------------------
print("\n=== Stage 4: functional CRM run (ChFsiProblemCartesian) ===")
run_ok = bool(setter_name and mat is not None and hasattr(fsi, "ChFsiProblemCartesian"))
if not run_ok:
    record("FAIL", "functional run", "prerequisites missing (setter, material struct, or problem class)")
else:
    V = chrono.ChVector3d
    spacing = 0.05
    msys = chrono.ChSystemNSC()
    ok, problem = try_each("create ChFsiProblemCartesian", [
        lambda: fsi.ChFsiProblemCartesian(spacing, msys),
        lambda: fsi.ChFsiProblemCartesian(spacing),
    ])
    fluid = None
    if ok:
        ok, fluid = try_each("get fluid system", [lambda: problem.GetFluidSystemSPH()])
    if ok:
        ok, _ = try_each(f"fluid.{setter_name}(material)  -> CRM mode", [lambda: getattr(fluid, setter_name)(mat)])
    if ok:
        for label, fn in [
            ("fluid.SetGravitationalAcceleration", lambda: fluid.SetGravitationalAcceleration(V(0, 0, -9.81))),
            ("problem.SetStepSizeCFD", lambda: problem.SetStepSizeCFD(1e-4)),
        ]:
            try:
                fn()
                record("PASS", label)
            except Exception as e:
                record("WARN", label, f"{type(e).__name__}: {str(e)[:70]}")
    if ok:
        side_all = getattr(getattr(fsi, "ChFsiProblemSPH", None), "BoxSide_ALL", None)
        ok, _ = try_each("problem.Construct (box of granular material)", [
            lambda: problem.Construct(V(1.0, 0.5, 0.25), V(0, 0, 0), side_all if side_all is not None else 0xFF),
            lambda: problem.Construct(V(1.0, 0.5, 0.25), V(0, 0, 0)),
            lambda: problem.Construct(1.0, 0.5, 0.25, V(0, 0, 0)),
        ])
    if ok:
        ok, _ = try_each("problem.Initialize()", [lambda: problem.Initialize()])
        if not ok:
            record("INFO", "problem methods (for debugging)",
                   ", ".join(find(problem, "init", "construct", "step", "advance", "fsi", "system")))
    if ok:
        mem_init = gpu_mem_used_mib()
        on_gpu_init = own_pid_on_gpu()

        # find something that steps the FSI system
        steppers = []
        for acc in ("GetSystemFSI", "GetFsiSystemSPH", "GetSystemFsi", "GetFsiSystem"):
            if hasattr(problem, acc):
                steppers.append(lambda a=acc: getattr(problem, a)().DoStepDynamics(1e-3))
        steppers.append(lambda: problem.DoStepDynamics(1e-3))
        steppers.append(lambda: problem.Advance(1e-3))
        step_ok, _ = try_each("step FSI system once", steppers)

        if step_ok:
            # re-use the same variant for the rest of the run
            t0 = time.time()
            n = 200
            done = False
            for fn in steppers:
                try:
                    fn()
                    for _ in range(n - 1):
                        fn()
                    done = True
                    break
                except Exception:
                    continue
            wall = time.time() - t0
            record("PASS" if done else "FAIL", f"step x{n}", f"{wall:.2f} s wall")

            mem_run = gpu_mem_used_mib()
            on_gpu_run = own_pid_on_gpu()
            if on_gpu_init or on_gpu_run:
                record("PASS", "this process is listed as a GPU compute app")
            elif on_gpu_init is None:
                record("WARN", "GPU compute-app check", "nvidia-smi query unavailable")
            else:
                record("WARN", "GPU compute-app check", "pid not listed (WSL/containers); rely on memory check")
            if mem_before is not None and mem_init is not None:
                d = mem_init - mem_before
                record("PASS" if d > 50 else "WARN", "GPU memory grew after Initialize",
                       f"{mem_before} -> {mem_init} MiB (delta {d})")
            record("INFO", "GPU memory after stepping (MiB)", str(mem_run))

            for label, fn in [("SPH marker count", lambda: fluid.GetNumFluidMarkers()),
                              ("SPH marker count (alt)", lambda: fluid.GetNumMarkers())]:
                try:
                    record("INFO", label, str(fn()))
                    break
                except Exception:
                    continue
        else:
            record("INFO", "problem methods (for debugging)",
                   ", ".join(find(problem, "step", "advance", "fsi", "system", "dostep")))

# --------------------------------------------------------------------------
print("\n=== Summary ===")
counts = {}
for s, _, _ in results:
    counts[s] = counts.get(s, 0) + 1
print("  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
fails = [r for r in results if r[0] == "FAIL"]
warns = [r for r in results if r[0] == "WARN"]
if fails:
    print("\nVerdict: CRM and/or GPU support NOT confirmed. Failures:")
    for _, n, d in fails:
        print(f"  - {n}: {d}")
elif warns:
    print("\nVerdict: CRM ran; some checks unconfirmed. Review WARN lines.")
else:
    print("\nVerdict: CRM mode is exposed and ran on the GPU.")
sys.exit(1 if fails else 0)