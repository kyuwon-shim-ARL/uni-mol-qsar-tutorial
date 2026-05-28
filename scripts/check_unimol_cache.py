#!/usr/bin/env python3
"""Verify the Uni-Mol weight cache is reachable BEFORE launching a GPU pod.

Saves money: if HF gates the model, fail here (free) instead of on a $1.36/hr
4-GPU pod after `create_pod_auto`.

Exit codes:
    0  weights cached and loadable
    1  unimol-tools not installed
    2  weights not cached / download failed
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch  # noqa: F401
        from unimol_tools import UniMolRepr
    except ImportError as e:
        print(f"FAIL: unimol-tools / torch not installed ({e}).", file=sys.stderr)
        print("      Install with `pip install unimol-tools torch`.", file=sys.stderr)
        return 1

    try:
        rep = UniMolRepr(
            data_type="molecule",
            remove_hs=False,
            model_name="unimolv1",
            model_size="84m",
            use_gpu=False,  # weight load only; no GPU needed for this check
        )
    except Exception as e:
        print(f"FAIL: UniMolRepr construction raised: {e}", file=sys.stderr)
        print("      Likely the HuggingFace weight download failed or is gated.", file=sys.stderr)
        return 2

    # Smoke embedding: one trivial SMILES to verify the model actually runs.
    try:
        out = rep.get_repr(["CCO"], return_atomic_reprs=False)
        if not out or len(out) != 1:
            print(f"FAIL: get_repr returned unexpected payload: {out!r}", file=sys.stderr)
            return 2
    except Exception as e:
        print(f"FAIL: smoke embedding raised: {e}", file=sys.stderr)
        return 2

    print("PASS: Uni-Mol v1 (84m) weights cached + smoke embedding succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
