from __future__ import annotations
import argparse
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from textwrap import dedent

# Paths
ROOT = Path(__file__).resolve().parents[1]
HW_FILE = ROOT / "prompts" / "hardware" / "gpu_specs.py"

# Strict author prompt template
AUTHOR_PROMPT = """
You are a CUDA‑kernel optimisation specialist.
Target GPU: **NVIDIA {gpu_name} ({gpu_arch})**
{gpu_items}

Task
----
Below are:
1. The original CUDA kernel.
2. Structured Critic feedback in JSON format, clearly describing issues and suggestions.

Your task: produce an improved CUDA kernel that addresses **all** issues and fully incorporates **all** suggestions.

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

[Original Kernel]
{code}

[Critic Feedback JSON]
{feedback}

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

# Build final prompt
TEMPLATE = dedent(AUTHOR_PROMPT)

def build_prompt(code: str, feedback: str, gpu_name: str) -> str:
    # Load GPU specs
    spec = importlib.util.spec_from_file_location("gpu_specs", HW_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpu_specs"] = module
    spec.loader.exec_module(module)  # type: ignore
    gpu_info = getattr(module, "GPU_SPEC_INFO")
    if gpu_name not in gpu_info:
        raise KeyError(f"GPU '{gpu_name}' not found in GPU_SPEC_INFO")
    info = gpu_info[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")
    gpu_items = "\n".join(f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture")

    # Substitute placeholders
    return TEMPLATE.format(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_items=gpu_items,
        code=code,
        feedback=feedback,
    )