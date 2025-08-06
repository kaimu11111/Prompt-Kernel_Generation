from __future__ import annotations
"""Island class – history‑aware seed generation.

* Stores only the **last 50 lines** of any compilation / runtime error into
  `ind.metrics["message"]`, shrinking JSON and prompt size.
* When composing repair prompts (`COMPILE_ERROR`), the **same 50‑line tail**
  is used.
* History block still contains full CUDA `source` strings from previous
  kernels (no truncation).  You may adjust if prompts grow too large.
"""

from pathlib import Path
import re
import json
from typing import List, Optional, Callable

from prompts.generate_custom_cuda import build_seed_prompt
from prompts.error import COMPILE_ERROR
from prompts.critic import CRITIC_PROMPT
from prompts.merge import build_merge_prompt
from prompts.author import build_prompt
from scripts.individual import KernelIndividual
from scripts.compile_and_run import compare_and_bench
from utils.kernel_io import extract_code_block, save_kernel_code, save_prompt_text, extract_critic_feedback
import random
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SOURCE_REGEX = re.compile(r"source\s*=\s*([\"']{3})(.*?)\1", re.DOTALL | re.I)


def _extract_full_cuda_source(code: str) -> str:
    """Return the CUDA `source` string or whole file if not found."""
    m = _SOURCE_REGEX.search(code)
    return m.group(2).strip() if m else code


def _last_n_lines(text: str, n: int = 150) -> str:
    """Return last *n* lines of *text*."""
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else text


def _build_history_block(kernel_dir: Path, keep_last: int = 5) -> str:
    """
    Collect the CUDA `source` of the most recent *keep_last* kernel files.
    """
    files: List[Path] = sorted(
        kernel_dir.glob("*.py"), key=lambda p: p.stat().st_mtime
    )[-keep_last:]

    if not files:
        return "## Existing kernels\n(None yet)\n"

    snippets: List[str] = []
    for idx, p in enumerate(files, 1):
        cuda_src = _extract_full_cuda_source(p.read_text(errors="ignore"))
        snippets.append(
            f"### Kernel {idx} · {p.name}\n```cuda\n{cuda_src}\n```"
        )

    return "## Existing kernels\n" + "\n\n".join(snippets) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# Island
# ------------------------------------------------------------------------------

class Island:
    """Evolutionary island operating on CUDA kernels."""

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        llm: Callable[[str, str], str],
        arch_py: Path,
        gpu_name: str | None,
        work_dir: Path,
        device: int,
        warmup: int,
        repeat: int,
        tol: float,
        survivor_ratio: float,
        mutation_ratio: float,
        crossover_ratio: float,
        pop_size: int,
    ) -> None:
        self.llm = llm
        self.arch_py = arch_py
        self.gpu_name = gpu_name
        self.ref_py = arch_py
        self.work_dir = work_dir
        self.device = device
        self.warmup = warmup
        self.repeat = repeat
        self.tol = tol
        self.survivor_ratio = survivor_ratio
        self.mutation_ratio = mutation_ratio
        self.crossover_ratio = crossover_ratio
        self.generation = 0
        self.pop_size = pop_size
        self.members: List[KernelIndividual] = []

        # pre-create kernel directory
        (self.work_dir / "kernels").mkdir(parents=True, exist_ok=True)

        for idx in range(pop_size):
            print(f"[Generating seed kernel {idx + 1}/{pop_size}]")
            self.members.append(self._generate_seed_kernel())

    # ------------------------------------------------------------------ #
    # 1. seed generation (no compile / evaluate here)
    # ------------------------------------------------------------------ #

    def _generate_seed_kernel(
        self, *, max_attempts: int = 3
    ) -> KernelIndividual:  # noqa: N802
        """
        Generate a seed kernel; write to kernels/; return `KernelIndividual`
        with empty metrics/score.
        """
        kernel_dir = self.work_dir / "kernels"
        history_block = _build_history_block(kernel_dir, keep_last=10)

        # Build LLM prompt
        full_prompt = build_seed_prompt(
            arch_path=self.arch_py,
            gpu_name=self.gpu_name,
            history_block=history_block,
        )

        code: Optional[str] = None
        for attempt in range(max_attempts):
            print(f"    Attempt {attempt + 1}/{max_attempts}")
            raw = self.llm(full_prompt, system="You are a CUDA kernel optimisation assistant.")
            try:
                code = extract_code_block(raw)
                break
            except Exception as exc:
                print(f"       · extract_code_block failed: {exc}")

        if code is None:
            print("       · fallback to reference kernel")
            code = self.ref_py.read_text()

        # Save only once in kernels/
        path = save_kernel_code(code, kernel_dir)
        ind = KernelIndividual(code, parents=[])
        ind.code_path = path  # type: ignore[attr-defined]
        return ind

    # ------------------------------------------------------------------ #
    # 2. evaluation & repair
    # ------------------------------------------------------------------ #

    def _repair(self, individual: KernelIndividual) -> KernelIndividual:
        """Single repair cycle via COMPILE_ERROR prompt."""
        prompt = COMPILE_ERROR.substitute(
            OLD_CODE=individual.code,
            ERROR_LOG=_last_n_lines(individual.metrics.get("message", "")),
        )
        # save_prompt_text(prompt, self.work_dir / "repair_prompts", tag=f"gen{self.generation:03d}")
        raw = self.llm(prompt, system="You are a CUDA kernel optimisation assistant.")
        fixed_code = extract_code_block(raw)

        kernel_dir = self.work_dir / "kernels"
        path = save_kernel_code(fixed_code, kernel_dir)
        child = KernelIndividual(fixed_code, parents=[individual.id])
        child.code_path = path  # type: ignore[attr-defined]
        return child

    def evaluate_all(self) -> None:
        """Compile, benchmark, (attempt to) repair every member."""
        print("[Evaluating all members…]")
        eval_dir = self.work_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        updated: List[KernelIndividual] = []

        for ind in self.members:
            metrics_path = eval_dir / f"eval_{ind.id:04d}.json"

            # skip if already has valid score
            if metrics_path.exists():
                cached = json.loads(metrics_path.read_text())
                if cached.get("score", float("-inf")) != float("-inf"):
                    ind.metrics = cached
                    ind.score = cached["score"]
                    updated.append(ind)
                    print(f"[Skipping {ind.id}, already has valid score]")
                    continue

            print(f"[Evaluating {ind.id}]")

            # locate code file
            path = getattr(ind, "code_path", None)
            if path is None or not Path(path).exists():
                path = save_kernel_code(ind.code, self.work_dir / "kernels")

            # compile / benchmark up to 3 attempts
            for attempt in range(3):
                try:
                    metrics = compare_and_bench(
                        ref_py=self.ref_py,
                        test_py=path,
                        device_idx=self.device,
                        warmup=self.warmup,
                        repeat=self.repeat,
                        tol=self.tol,
                    )
                    speedup = (
                        metrics["ref_latency_ms"]["avg"] / metrics["test_latency_ms"]["avg"]
                    )
                    metrics["score"] = speedup
                    ind.metrics = metrics
                    ind.score = speedup
                    break
                except Exception as exc:
                    print(f"[Attempt {attempt + 1}/3 failed: {exc}]")
                    ind.metrics = {
                        "error_type": exc.__class__.__name__,
                        "message": _last_n_lines(str(exc)),
                    }
                    ind.score = float("-inf")
                    if attempt < 2:
                        try:
                            ind = self._repair(ind)
                            path = ind.code_path  # type: ignore[attr-defined]
                        except Exception as rep_exc:
                            print(f"[Repair failed: {rep_exc}]")
                            continue

            # persist metrics
            updated.append(ind)
            try:
                ind.save_metrics(eval_dir)
            except Exception:
                pass

        self.members = updated

    # ------------------------------------------------------------------ #
    # 3. evolution operators
    # ------------------------------------------------------------------ #

    def select_survivors(self) -> List[KernelIndividual]:
        k = max(1, int(len(self.members) * self.survivor_ratio))
        return sorted(
            self.members,
            key=lambda x: x.score if x.score is not None else float("-inf"),
            reverse=True,
        )[:k]

    def mutate(self, parent: KernelIndividual) -> KernelIndividual:
        print(f"[Mutating {parent.id}]")
        kernel_dir = self.work_dir / "kernels"
        prompt_dir = self.work_dir / "prompts" / "mutate"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        history_block = _build_history_block(kernel_dir, keep_last=10)

        # === Critic Phase ===
        critic_prompt = history_block + CRITIC_PROMPT.format(
            code=parent.code,
            metrics=parent.metrics or {},
        )
        # critic_prompt_path = prompt_dir / f"mutate_{parent.id:04d}_critic.txt"
        # critic_prompt_path.write_text(critic_prompt)
        
        critic_reply = self.llm(critic_prompt, system="You are a CUDA kernel optimisation assistant.")

        feedback_dict = extract_critic_feedback(critic_reply)
        feedback_json = json.dumps(feedback_dict, indent=2)

        # === Author Phase ===
        author_prompt = history_block + build_prompt(
            code=parent.code,
            feedback=feedback_json,
            gpu_name=self.gpu_name or "GPU",
        )
        # author_prompt_path = prompt_dir / f"mutate_{parent.id:04d}_author.txt"
        # author_prompt_path.write_text(author_prompt)
        
        author_reply = self.llm(author_prompt, system="You are a CUDA kernel optimisation assistant.")
        code = extract_code_block(author_reply)

        path = save_kernel_code(code, kernel_dir)
        child = KernelIndividual(code, parents=[parent.id])
        child.code_path = path  # type: ignore[attr-defined]
        return child

    def crossover(self, p1: KernelIndividual, p2: KernelIndividual) -> KernelIndividual:
        print(f"[Crossover {p1.id} + {p2.id}]")
        kernel_dir = self.work_dir / "kernels"
        history_block = _build_history_block(kernel_dir, keep_last=10)

        merge_prompt = history_block + build_merge_prompt(
            parent1=p1.code,
            parent2=p2.code,
            gpu_name=self.gpu_name or "GPU",
        )
        raw = self.llm(merge_prompt, system="You are a CUDA kernel optimisation assistant.")
        merged = extract_code_block(raw)

        path = save_kernel_code(merged, kernel_dir)
        child = KernelIndividual(merged, parents=[p1.id, p2.id])
        child.code_path = path  # type: ignore[attr-defined]
        return child

    # ------------------------------------------------------------------ #
    # 4. generation loop
    # ------------------------------------------------------------------ #

    def evolve_one_generation(self) -> None:
        """Perform selection + variation to create next population."""
        print(f"[Evolving generation {self.generation + 1}]")
        print("current population size:", len(self.members))
        next_pop: List[KernelIndividual] = []
        self.evaluate_all()
        # evaluate / select
        if self.generation > 0:
            survivors = self.select_survivors()
        else:
            survivors = list(self.members)

        next_pop.extend(survivors)

        # mutation
        num_mut = int(self.mutation_ratio * len(survivors))
        if num_mut <= len(survivors):
            for parent in random.sample(survivors, num_mut):
                next_pop.append(self.mutate(parent))
        else:
            for _ in range(num_mut):
                next_pop.append(self.mutate(random.choice(survivors)))

        # crossover
        num_cross = int(self.crossover_ratio * len(survivors))
        possible_pairs = [
            (a, b) for idx, a in enumerate(survivors) for b in survivors[idx + 1 :]
        ]
        if num_cross <= len(possible_pairs):
            sampled_pairs = random.sample(possible_pairs, num_cross)
        else:
            sampled_pairs = [random.choices(survivors, k=2) for _ in range(num_cross)]

        for p1, p2 in sampled_pairs:
            next_pop.append(self.crossover(p1, p2))

        # update state
        self.members = next_pop
        self.generation += 1