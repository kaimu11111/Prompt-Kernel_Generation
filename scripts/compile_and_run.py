from __future__ import annotations
"""
compare_and_bench.py – single-GPU benchmark (**full compile + runtime traceback**).

Key features
------------
* Dynamically imports two PyTorch models (reference & candidate) and **captures
  every byte** printed by Python *and* child processes (ninja / nvcc).
  - On any *build* failure, raises `CompilationError(full_log)`.
* On **runtime failure** (forward, benchmark, accuracy), re-raises
  `RuntimeError(traceback.format_exc())` so callers get the *entire*
  traceback – not just `str(exc)`.
* Benchmarks on CUDA (default) or CPU (`--cpu`).
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch

# ---------------------------------------------------------------------------

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------


class CompilationError(RuntimeError):
    """Raised when dynamic import / nvcc build fails.

    The *first* argument is the full build log (Python + ninja/nvcc).
    """


# =========================== dynamic import ===============================
def _capture_import(path: Path):
    """Import *path* dynamically and capture **all** build logs.

    Returns
    -------
    (module, full_log : str)

    Raises
    ------
    FileNotFoundError
        *path* does not exist.
    CompilationError
        Any Python / ninja / nvcc error during import.  The exception's first
        argument is the concatenated log.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    mod_name = f"mod_{hashlib.md5(str(path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)                     # type: ignore[arg-type]
    sys.modules[mod_name] = module
    assert spec.loader is not None

    # ---- Python-level stdout/stderr to StringIO --------------------------
    py_buf = io.StringIO()

    # ---- OS-level FD 1/2 (stdout/stderr) to a temp file -----------------
    with tempfile.TemporaryFile(mode="w+") as fd_buf, \
         contextlib.redirect_stdout(py_buf), \
         contextlib.redirect_stderr(py_buf):

        # Save current FDs so we can restore later
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(fd_buf.fileno(), 1)     # redirect FD 1 → temp file
            os.dup2(fd_buf.fileno(), 2)     # redirect FD 2 → temp file

            # ------------ REAL IMPORT (build/compile) --------------------
            spec.loader.exec_module(module)                             # pyright: ignore[attr-defined]

            fd_buf.flush()
            fd_buf.seek(0)
            subproc_log = fd_buf.read()

        except Exception as exc:  # ← build / link / import failed
            # Combine StringIO + temp-file logs + Exception str
            fd_buf.flush(); fd_buf.seek(0)
            subproc_log = fd_buf.read()
            full_log = "".join([py_buf.getvalue(), subproc_log, str(exc)]).strip()
            raise CompilationError(full_log) from None

        finally:
            # Always restore original FDs
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

    # ---------------- SUCCESS --------------------------------------------
    return module, py_buf.getvalue() + subproc_log


# =========================== timing helpers ===============================
def _run_once(model: torch.nn.Module,
              inp: List[torch.Tensor],
              dev: torch.device) -> Tuple[torch.Tensor, float]:
    model.to(dev).eval()
    inp = [x.to(dev) for x in inp]

    if TORCH_DEVICE == "cpu":
        t0 = datetime.now()
        out = model(*inp)
        ms = (datetime.now() - t0).total_seconds() * 1_000
        return out, ms

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(dev)
    start.record()
    out = model(*inp)
    end.record()
    end.synchronize()
    return out, start.elapsed_time(end)


def _bench(model: torch.nn.Module,
           inp: List[torch.Tensor],
           dev: torch.device,
           warm: int,
           rep: int) -> List[float]:
    model.to(dev).eval()
    inp = [x.to(dev) for x in inp]

    for _ in range(warm):
        model(*inp)

    if TORCH_DEVICE == "cpu":
        res = []
        for _ in range(rep):
            t0 = datetime.now()
            model(*inp)
            res.append((datetime.now() - t0).total_seconds() * 1_000)
        return res

    torch.cuda.synchronize(dev)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    times: List[float] = []
    for _ in range(rep):
        s.record()
        model(*inp)
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return times


# =========================== public API ===================================
def compare_and_bench(
    ref_py: Path,
    test_py: Path,
    *,
    device_idx: int = 0,
    warmup: int = 5,
    repeat: int = 20,
    tol: float = 1e-4,
) -> dict:
    """Benchmark *test_py* against *ref_py*.

    Returns
    -------
    dict
        Timing/accuracy metrics.

    Raises
    ------
    CompilationError
        If either model fails to build.
    RuntimeError
        If forward / benchmark / accuracy check fails.  The exception's first
        argument is the full Python traceback.
    """
    dev = (torch.device(f"cuda:{device_idx}")
           if TORCH_DEVICE == "cuda" else torch.device("cpu"))
    if TORCH_DEVICE == "cuda":
        torch.cuda.set_device(dev)

    # ---------------- import both modules (build stage) -------------------
    ref_mod, _ = _capture_import(ref_py)
    test_mod, _ = _capture_import(test_py)

    RefModel = getattr(ref_mod, "Model", None)
    get_inputs = getattr(ref_mod, "get_inputs", None)
    ModelNew = getattr(test_mod, "ModelNew", None)

    if None in (RefModel, get_inputs):
        raise RuntimeError(f"Reference '{ref_py}' must define Model & get_inputs().")
    if ModelNew is None:
        raise RuntimeError(f"Candidate '{test_py}' must define ModelNew class.")

    # ---------------- run / benchmark ------------------------------------
    try:
        ctx = torch.cuda.device(dev) if TORCH_DEVICE == "cuda" else contextlib.nullcontext()
        with ctx:
            inp = get_inputs()
            ref_out, _ = _run_once(RefModel(), inp, dev)
            test_out, _ = _run_once(ModelNew(), inp, dev)

            diff = (test_out - ref_out).abs()
            max_err, mean_err = diff.max().item(), diff.mean().item()
            if max_err > tol:
                raise ValueError(f"max_abs_err={max_err:.3e} > tol={tol}")

            ref_t = _bench(RefModel(),  inp, dev, warmup, repeat)
            test_t = _bench(ModelNew(), inp, dev, warmup, repeat)

    except Exception:
        # Re-raise with full traceback so outer layers can log/debug
        raise RuntimeError(traceback.format_exc()) from None

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "reference_file": str(ref_py),
        "candidate_file": str(test_py),
        "tolerance": tol,
        "max_abs_err": max_err,
        "mean_abs_err": mean_err,
        "ref_latency_ms": {
            "avg": sum(ref_t) / len(ref_t),
            "min": min(ref_t),
            "max": max(ref_t),
        },
        "test_latency_ms": {
            "avg": sum(test_t) / len(test_t),
            "min": min(test_t),
            "max": max(test_t),
        },
        "num_runs": repeat,
    }


# =========================== CLI wrapper ==================================
def _cli():
    p = argparse.ArgumentParser(description="Compare & bench two model files.")
    p.add_argument("reference", type=Path, help="Path to reference .py")
    p.add_argument("candidate", type=Path, help="Path to candidate .py")
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--warmup", type=int, default=5, help="Warm-up iterations")
    p.add_argument("--repeat", type=int, default=20, help="Benchmark runs")
    p.add_argument("--tol", type=float, default=1e-4, help="Max abs error tolerance")
    p.add_argument("--dump", type=Path, help="If set, write JSON results here")
    args = p.parse_args()

    res = compare_and_bench(
        args.reference,
        args.candidate,
        device_idx=args.device,
        warmup=args.warmup,
        repeat=args.repeat,
        tol=args.tol,
    )
    print(json.dumps(res, indent=2))

    if args.dump:
        args.dump.write_text(json.dumps(res, indent=2))
        print(f"\nSaved ⇒ {args.dump}")


if __name__ == "__main__":
    _cli()
