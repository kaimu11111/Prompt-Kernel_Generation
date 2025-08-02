CRITIC_PROMPT = """
Analyze the following CUDA kernel code and its performance metrics.
Identify the primary issues and bottlenecks, and provide actionable optimization suggestions.

[Kernel Code]
{code}

[Performance Metrics]
{metrics}


The JSON must have exactly three keys:
- "issues": list of strings describing code problems.
- "bottlenecks": list of strings identifying performance bottlenecks.
- "suggestions": list of strings with optimization recommendations.

Example:
{{
  "issues": [""],
  "bottlenecks": [""],
  "suggestions": [""]
}}

# ---------- Diversity requirement ----------
The kernel you generate **must be meaningfully different** from every existing kernel listed above. Acceptable forms of difference include, but are not limited to:

Do **not** simply adjust constants or reorder lines of an existing kernel.
"""