# utils/kernel_io.py
"""Utility helpers for Mind‑Evolution CUDA‑kernel workflow.

This tiny module centralises two common I/O helpers that were previously
inlined in the end‑to‑end test script:

1. ``extract_code_block`` – extract first ```python ... ``` (or generic) code
   block from LLM output. Raises if none found.
2. ``save_kernel_code`` – writes extracted code to *kernels/* with a unique
   timestamped filename and returns the *Path* object.

Keeping them here avoids duplication across evolution loops / diagnostics.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Final
import json
from typing import Any, Dict
__all__: Final = [
    "extract_code_block",
    "save_kernel_code",
]

# ---------------------------------------------------------------------------
# 1. Code‑block extraction
# ---------------------------------------------------------------------------
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def extract_code_block(text: str) -> str:
    """Return the **first** triple‑back‑ticked block in *text*.

    Parameters
    ----------
    text : str
        Raw LLM output that should include a python code block.

    Returns
    -------
    str
        The code inside the back‑ticks (stripped) with a trailing newline.

    Raises
    ------
    RuntimeError
        If no code block is found.
    """
    match = _CODE_BLOCK_RE.search(text)
    if not match:
        # 保存 LLM 原始输出
        from datetime import datetime
        dump_path = f"llm_output_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(dump_path, "w") as f:
            f.write(text)
        raise RuntimeError(f"No ``` code block found in LLM output – raw output saved to {dump_path}")
    
    return match.group(1).strip() + "\n"



# ---------------------------------------------------------------------------
# 2. Persist kernel to file
# ---------------------------------------------------------------------------

def save_kernel_code(code: str, out_dir: Path | str = "kernels") -> Path:
    """Save *code* to *out_dir/kernel_YYYYmmdd_HHMMSS.py* and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"kernel_{stamp}.py"
    path.write_text(code, encoding="utf-8")

    return path


# utils/kernel_io.py





def extract_critic_feedback(raw: str) -> str:
    """
    从原始输出中提取第一个花括号包裹的块（包括内嵌换行），返回完整文本。
    """
    # 匹配第一个 { 与其对应的最外层 }
    brace_stack = []
    start_index = None
    for i, ch in enumerate(raw):
        if ch == '{':
            if not brace_stack:
                start_index = i
            brace_stack.append(ch)
        elif ch == '}':
            if brace_stack:
                brace_stack.pop()
                if not brace_stack and start_index is not None:
                    return raw[start_index:i+1]
    raise ValueError(f"未能提取到完整的 JSON 块: {raw!r}")

def save_prompt_text(text: str, out_dir: Path, *, tag: str = "repair") -> Path:
    """
    Save *text* to out_dir/{tag}_YYYYMMDD-HHMMSS.txt and return the Path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{tag}_{ts}.txt"
    path.write_text(text, encoding="utf-8")
    return path