"""Bootstrap repo-local GNU RISC-V binutils under `toolchains/riscv32/`."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolchains.riscv32.gnu_toolchain import bootstrap_repo_local_binutils


def main() -> None:
    """Download and unpack GNU RISC-V binutils for the local wrapper."""
    toolchain_root = bootstrap_repo_local_binutils(REPO_ROOT)
    print(toolchain_root)


if __name__ == "__main__":
    main()
