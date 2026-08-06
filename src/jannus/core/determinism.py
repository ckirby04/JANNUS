"""Seeding and determinism control.

Multi-site validation only means something if a site can re-run the same data
through the same code and get the same numbers. That requires more than
`torch.manual_seed`: cuDNN picks convolution algorithms by benchmarking, which
is nondeterministic across runs on the same hardware.

`configure_determinism()` is called by every CLI entry point before any tensor
work. It reports what it was actually able to enforce rather than claiming
success blindly, and that report goes into the provenance manifest.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_SEED = 20260805


@dataclass
class DeterminismReport:
    """What determinism was actually achieved, for the provenance manifest."""

    seed: int
    python_hash_seed: str | None
    numpy_seeded: bool = False
    torch_seeded: bool = False
    cuda_seeded: bool = False
    cudnn_deterministic: bool = False
    cudnn_benchmark_disabled: bool = False
    torch_deterministic_algorithms: bool = False
    #: Non-fatal problems, e.g. an op with no deterministic implementation.
    warnings: list[str] = field(default_factory=list)

    @property
    def fully_deterministic(self) -> bool:
        """True only when every knob we know about was successfully set."""
        return (
            self.numpy_seeded
            and self.torch_seeded
            and self.cudnn_deterministic
            and self.cudnn_benchmark_disabled
            and self.torch_deterministic_algorithms
            and not self.warnings
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fully_deterministic"] = self.fully_deterministic
        return d


def configure_determinism(
    seed: int = DEFAULT_SEED, *, strict: bool = False
) -> DeterminismReport:
    """Seed every RNG in play and pin cuDNN to deterministic kernels.

    Args:
        seed: master seed applied to python, numpy and torch.
        strict: when True, ask torch to *error* on any op lacking a
            deterministic implementation rather than silently using the
            nondeterministic one. Correct for a formal validation run;
            it can make some training paths unrunnable, so it is off by default.

    Returns:
        A report of what was enforced. Callers should record it rather than
        assume determinism was achieved.
    """
    report = DeterminismReport(
        seed=seed, python_hash_seed=os.environ.get("PYTHONHASHSEED")
    )

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
        report.numpy_seeded = True
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        report.warnings.append("numpy not importable; numpy RNG unseeded")

    try:
        import torch
    except ImportError:
        report.warnings.append("torch not importable; torch RNG unseeded")
        return report

    torch.manual_seed(seed)
    report.torch_seeded = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        report.cuda_seeded = True

    try:
        torch.backends.cudnn.deterministic = True
        report.cudnn_deterministic = True
        torch.backends.cudnn.benchmark = False
        report.cudnn_benchmark_disabled = True
    except AttributeError:  # pragma: no cover - non-CUDA torch build
        report.warnings.append("torch.backends.cudnn unavailable (CPU-only build)")

    # Deterministic reductions on CUDA >= 10.2 additionally require this env
    # var, and it must be set before the CUDA context is created.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    try:
        torch.use_deterministic_algorithms(True, warn_only=not strict)
        report.torch_deterministic_algorithms = True
    except Exception as exc:
        report.warnings.append(f"use_deterministic_algorithms failed: {exc}")

    return report
