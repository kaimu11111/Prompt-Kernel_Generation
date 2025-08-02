# prompts/error.py

"""Prompt template for automatic kernel repair.
Uses `string.Template` to avoid `{}` brace conflicts with C/CUDA code.
"""
from string import Template

COMPILE_ERROR = Template(
    """You are a senior CUDA‑extension developer.
Your job is to **FIX** the compilation or runtime errors in the Python script
shown below.

────────────────────────────────────────
OUTPUT REQUIREMENTS (STRICT)
────────────────────────────────────────
1. Respond with **exactly one** fenced block labelled `python`. **No prose**
   before or after.
2. The script must be self‑contained and runnable via

       python fixed_model.py

3. The script **must** include, in this order:
   • a `source` string containing the CUDA kernel(s)
   • a `cpp_src` string with C++ prototypes/definitions
   • a single `torch.utils.cpp_extension.load_inline` call (`with_cuda=True`)
   • `class ModelNew(nn.Module)` that wraps the custom op(s)

────────────────────────────────────────
OLD CODE (read‑only)
────────────────────────────────────────
$OLD_CODE

────────────────────────────────────────
ERROR LOG
────────────────────────────────────────
$ERROR_LOG

# ==========================================================
# ❶ OUTPUT FORMAT – Copy exactly
Return the fixed script wrapped like this – no extra text:

```python
# <your corrected code>
```
# ==========================================================
"""
)