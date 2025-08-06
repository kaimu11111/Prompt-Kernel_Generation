#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_one_task.py — Mind‑Evolution single‑task runner (v2)

Updated to work with the **history‑aware Island class** that internally
constructs its own seed prompt.  The main differences from the original
version are:

1. **No external build_prompt call** – Island takes `arch_py` and `gpu_name`
   and builds the prompt itself.
2. Island constructor signature has changed: it expects `arch_py` instead
   of `ref_py`, and no longer needs `seed_prompt`.
3. Minor cleanup of unused imports.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

from LLM_interface.query_server import query_server
from scripts.island import Island  # new Island with internal prompt builder
from utils.kernel_io import save_kernel_code
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────
# 1. CLI helper
# ──────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        "Mind‑Evolution single‑task runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("arch_py", type=Path, help="Reference model containing class Model")
    p.add_argument("--gpu", default="Quadro RTX 6000", help="GPU name in prompt spec")
    p.add_argument("--server_type", default="local", help="LLM provider type (local, openai, deepseek, etc.)")
    p.add_argument("--server_address", default="localhost", help="LLM server address (for vllm/sglang)")
    p.add_argument("--server_port", type=int, default=30000, help="LLM server port (for vllm/sglang)")
    p.add_argument("--model_name", default="deepseek-ai/deepseek-coder-6.7b-instruct", help="LLM model name or ID")
    p.add_argument("--generations", "-G", type=int, default=6, help="Number of generations")
    p.add_argument("--pop_size", "-P", type=int, default=6, help="Individuals per generation")
    p.add_argument("--work_dir", default="runs", type=Path, help="Output root directory")
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--warmup", type=int, default=5, help="Warm‑up iterations")
    p.add_argument("--repeat", type=int, default=20, help="Timed iterations per benchmark")
    p.add_argument("--tol", type=float, default=0.0001, help="Max |err| tolerated")
    p.add_argument("--survivor", type=float, default=0.5, help="Survivor ratio")
    p.add_argument("--mutate", type=float, default=1.0, help="Mutation ratio")
    p.add_argument("--crossover", type=float, default=0.4, help="Crossover ratio")
    p.add_argument("--max_tokens", type=int, default=6500, help="LLM max new tokens")
    return p

# ──────────────────────────────────────────────────
# 2. Main entry
# ──────────────────────────────────────────────────

def main(argv=None):  # noqa: D401
    args = _build_arg_parser().parse_args(argv)

    # 1. Prepare run directory
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = (args.work_dir / run_stamp).resolve()
    (work_dir / "kernels").mkdir(parents=True, exist_ok=True)
    (work_dir / "evaluation").mkdir(parents=True, exist_ok=True)

    # 2. Construct unified LLM caller
    def call_llm(prompt: str, system: str = "You are a helpful assistant") -> str:  # noqa: D401
        return query_server(
            prompt=prompt,
            system_prompt=system,
            server_type=args.server_type,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            temperature=0.9,
            top_p=0.9,
            server_address=args.server_address,
            server_port=args.server_port,
        )

    # 3. Initialise island (prompt built inside)
    print("[1/3] Initialising island …", flush=True)
    island = Island(
        llm=call_llm,
        arch_py=args.arch_py,
        gpu_name=args.gpu,
        work_dir=work_dir,
        device=args.device,
        warmup=args.warmup,
        repeat=args.repeat,
        tol=args.tol,
        survivor_ratio=args.survivor,
        mutation_ratio=args.mutate,
        crossover_ratio=args.crossover,
        pop_size=args.pop_size,
    )

    # Trackers
    best_scores: List[float] = []
    best_runtimes: List[float] = []

    # 4. Evolution loop
    for gen in range(1, args.generations + 1):
        print(f"    → Generation {gen}/{args.generations}", flush=True)
        island.evolve_one_generation()

        best = max(island.members, key=lambda ind: ind.score or float("-inf"))
        raw_score = best.score if best.score is not None else float("-inf")
        best_scores.append(0 if raw_score == float("-inf") else raw_score)
        best_runtimes.append(best.metrics.get("test_latency_ms", {}).get("avg", float("nan")))

    # 5. Save best kernel overall
    best_overall = max(island.members, key=lambda ind: ind.score or float("-inf"))
    best_path = save_kernel_code(best_overall.code, work_dir / "best")
    (work_dir / "best_metrics.json").write_text(json.dumps(best_overall.metrics, indent=2))

    print("[2/3] Evolution complete ✓", flush=True)
    print(f"Best kernel → {best_path.relative_to(Path.cwd())}")
    print(json.dumps(best_overall.metrics, indent=2))

    # 6. Plot results
    figure_dir = work_dir / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generations = list(range(1, args.generations + 1))

    plt.figure()
    plt.plot(generations, best_scores)
    plt.xlabel("Generation")
    plt.ylabel("Best Score")
    plt.title("Best Score per Generation")
    plt.tight_layout()
    plt.savefig(figure_dir / "score.png")
    plt.close()

    plt.figure()
    plt.plot(generations, best_runtimes)
    plt.xlabel("Generation")
    plt.ylabel("Best Runtime (ms)")
    plt.title("Best Runtime per Generation")
    plt.tight_layout()
    plt.savefig(figure_dir / "runtime.png")
    plt.close()


if __name__ == "__main__":
    main()
