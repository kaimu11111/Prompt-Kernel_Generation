from __future__ import annotations

"""Prompt builder for merging two CUDA kernels.

Generates a **single prompt** that contains:
1. Target GPU spec (from `prompts/hardware/gpu_specs.py`)
2. Two parent kernels
3. Output requirements

CLI usage
---------
```bash
python -m prompts.build_merge_prompt \
    --parent1 kernel1.cu --parent2 kernel2.cu \
    --gpu "Quadro RTX 6000" -o merge_prompt.txt
```"""
import argparse
import importlib.util
import sys
from pathlib import Path
from textwrap import dedent
from string import Template

# Constants for paths
data_dir = Path(__file__).resolve().parents[1]
HW_FILE = data_dir / "prompts" / "hardware" / "gpu_specs.py"

MERGE_PROMPT = """
You are a CUDA‑kernel optimisation specialist.
Target GPU: **NVIDIA {gpu_name} ({gpu_arch})**
{gpu_items}

Your task:
----
Below are two parent kernels that implement the same computation but use
different optimization strategies. Produce a **single** merged kernel that:

  • Retains the correct computation.
  • Combines the best optimizations from both parents.
  • Eliminates redundant or sub-optimal code.
  • Keeps the original kernel name and signature of parent #1.
  • Is compile-ready (C++/CUDA).

OUTPUT REQUIREMENTS (STRICT)
─────────────────────────────────────────────
- Respond with exactly one fenced code block labelled `python`.

Within the fenced block, follow this order:
1. Imports: `torch`, `torch.nn`, `load_inline`.
2. `source`: triple-quoted CUDA code (kernel + launcher).
3. `cpp_src`: C++ prototypes for all kernels.
4. Exactly one `load_inline` call.
5. `class ModelNew(nn.Module)`: mirrors original inputs/outputs, calls optimized kernels.

Exclude:
- Testing code
- `if __name__ == "__main__"` guard
- Additional prose or markdown

[Parent #1]
{parent1}

[Parent #2]
{parent2}

# ==========================================================
# OUTPUT FORMAT – EXACTLY:
```python
# <corrected, optimized Python script>
```
# ==========================================================
# ---------- Diversity requirement ----------
The kernel you generate **must be meaningfully different** from every existing kernel listed above. Acceptable forms of difference include, but are not limited to:

Do **not** simply adjust constants or reorder lines of an existing kernel.
"""

# Build the template
TEMPLATE = dedent(MERGE_PROMPT)

def _load_gpu_spec() -> dict:
    """Import `gpu_specs.py` and return the GPU_SPEC_INFO dict."""
    spec = importlib.util.spec_from_file_location("gpu_specs", HW_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {HW_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["gpu_specs"] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "GPU_SPEC_INFO"):
        raise AttributeError("GPU_SPEC_INFO not defined in gpu_specs.py")
    return module.GPU_SPEC_INFO  # type: ignore[attr-defined]


def build_merge_prompt(parent1: str, parent2: str, gpu_name: str) -> str:
    """
    Create a merge prompt given two parent kernel sources and GPU name.
    """
    gpu_info = _load_gpu_spec()
    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO")
    info = gpu_info[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")
    gpu_items = "\n".join(f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture")

    return TEMPLATE.format(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_items=gpu_items,
        parent1=parent1,
        parent2=parent2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a merge prompt for two CUDA kernels.")
    parser.add_argument(
        "--parent1", type=Path, required=True,
        help="Path to the first parent kernel source file."
    )
    parser.add_argument(
        "--parent2", type=Path, required=True,
        help="Path to the second parent kernel source file."
    )
    parser.add_argument(
        "--gpu", dest="gpu_name", required=True,
        help="Name of the target GPU, e.g., 'Quadro RTX 6000'."
    )
    parser.add_argument(
        "-o", "--output", dest="output_file", type=Path,
        default=None, help="Where to write the generated prompt (defaults to stdout)."
    )
    args = parser.parse_args()

    p1 = args.parent1.read_text()
    p2 = args.parent2.read_text()
    prompt = build_merge_prompt(p1, p2, args.gpu_name)

    if args.output_file:
        args.output_file.write_text(prompt)
    else:
        sys.stdout.write(prompt)


if __name__ == "__main__":
    main()
