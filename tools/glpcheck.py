#!/usr/bin/env python3
"""Validate a GLP v3 sidecar with the exact loader used by the TensorFold engine, without a GPU or server.

Usage: tools/glpcheck.py <control.gguf> [--layers 40] [--width 5120]
Exit 0 when the file is servable, 1 with the fail-closed refusal reason otherwise.
"""

import argparse
import hashlib
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build/tensorfold/src"))

from tensorfold.families.deepseek_v41.cuda import glp as engine_glp  # noqa: E402


def check(path: str, layers: int, width: int) -> dict:
    dirs, alpha, _ = engine_glp.load(path, layers=layers, width=width)
    meta, _ = engine_glp.read_gguf(path)
    return {"meta": meta, "layers": sorted(dirs), "alpha": alpha, "width": width}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file")
    ap.add_argument("--layers", type=int, default=40, help="decoder layers (DeepSeek-V4.1-Flash: 40)")
    ap.add_argument("--width", type=int, default=5120, help="hidden width of each stream copy (not hc_mult * hidden)")
    args = ap.parse_args()
    try:
        got = check(args.file, args.layers, args.width)
    except (ValueError, OSError, struct.error) as exc:
        print(f"glpcheck: {exc}", file=sys.stderr)
        sys.exit(1)
    meta = got["meta"]
    with open(args.file, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    print(f"{args.file}: servable")
    print(f"  mode=project hook={engine_glp.HOOK} spec_version={meta.get('glp.spec_version', 1)} "
          f"rank={meta.get('glp.rank', 1)}")
    print(f"  layers {got['layers'][0]}..{got['layers'][-1]} ({len(got['layers'])} steered), "
          f"width {got['width']} per copy, alpha_default {got['alpha']}")
    base = meta.get("general.base_model.0.name") or meta.get("glp.base_model") or "?"
    rev = str(meta.get("general.base_model.0.version") or meta.get("glp.base_revision") or "?")[:12]
    print(f"  base model: {base} @ {rev}; content sha256 {sha[:16]}...")


if __name__ == "__main__":
    main()
