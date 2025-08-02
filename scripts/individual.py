# evolution/individual.py
"""
KernelIndividual 类：表示一个 CUDA kernel 优化个体。

保存字段：
- id        : 全局唯一标识
- code      : Kernel 源码文本
- parents   : 父代个体 id 列表
- metrics   : 评估结果（dict），包含 latency、errors、带宽等
- score     : 打分（float）
- feedback  : 可选文本反馈，用于诊断或 RCC

方法：
- to_dict(): 序列化 id, score, parents 用于记录 summary
- save_code(dir): 将 code 写入文件并返回路径
- save_metrics(dir): 将 metrics 写入 JSON
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

class KernelIndividual:
    _next_id = 0

    def __init__(
        self,
        code: str,
        parents: Optional[List[int]] = None,
    ):
        self.id: int = KernelIndividual._next_id
        KernelIndividual._next_id += 1

        self.code: str = code
        self.parents: List[int] = parents or []
        self.metrics: Optional[Dict[str, Any]] = None
        self.score: Optional[float] = None
        self.feedback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """返回个体摘要，用于 summary JSON"""
        return {
            "id": self.id,
            "score": self.score,
            "parents": self.parents,
        }

    def save_code(self, out_dir: Path) -> Path:
        """写入 kernel 代码至指定目录，返回文件路径"""
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / f"kernel_{self.id:04d}.py"
        file_path.write_text(self.code)
        return file_path

    def save_metrics(self, out_dir: Path) -> Path:
        """将 metrics 写入 JSON 文件，返回文件路径"""
        if self.metrics is None:
            raise ValueError("metrics 未设置，无法保存")
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / f"eval_{self.id:04d}.json"
        file_path.write_text(json.dumps(self.metrics, indent=2))
        return file_path
