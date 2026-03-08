#!/usr/bin/env python3
"""
Compare ADAM's GGUF dequantization against the reference gguf.dequantize.

For every tensor in the model:
  - dequantize with gguf.dequantize (reference)
  - compare shape and values against adam/loaders/gguf.py output
  - report max_diff, RMSE, and whether shapes match
"""
import os, sys, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gguf as gguf_ref
from adam.loaders.gguf import GGUFLoader

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"
BOLD = "\033[1m"; RST = "\033[0m"

def find_model():
    hits = glob.glob(os.path.join(ROOT, "**/*.gguf"), recursive=True)
    hits.sort(key=lambda p: os.path.basename(p).lower())
    return sys.argv[1] if len(sys.argv) > 1 else hits[0]

def main():
    path = find_model()
    print(f"\n{BOLD}Dequantization reference check: {os.path.basename(path)}{RST}")

    # --- Reference: gguf package ---
    print("Loading reference (gguf.GGUFReader)...", flush=True)
    ref_reader = gguf_ref.GGUFReader(path)
    ref = {}
    for t in ref_reader.tensors:
        ref[t.name] = (t, gguf_ref.dequantize(t.data, t.tensor_type))
    print(f"  {len(ref)} tensors loaded from reference")

    # --- Ours: adam GGUFLoader ---
    print("Loading ours (adam GGUFLoader)...", flush=True)
    loader = GGUFLoader(path)
    loader.load(verbose=False)
    print(f"  {len(loader.tensors)} tensors loaded from our loader")

    # --- Compare every tensor ---
    print(f"\n{'Tensor':<45}  {'dtype':>6}  {'shape_ok':>8}  {'max_diff':>10}  {'rmse':>10}  status")
    print("-" * 100)

    n_pass = n_fail = n_warn = 0
    worst = []

    for name in sorted(ref.keys()):
        if name not in loader.tensors:
            print(f"  {name:<45}  MISSING from our loader  {FAIL}")
            n_fail += 1
            continue

        t_ref, arr_ref = ref[name]
        arr_our = loader.tensors[name].astype(np.float32)

        # Reference shape: gguf.dequantize returns shape matching GGUF's ne[] order
        # (fastest-dim first), e.g. [in, out]. Our loader reverses to [out, in].
        # Flatten both for comparison.
        flat_ref = arr_ref.flatten()
        flat_our = arr_our.flatten()

        shape_ok = flat_ref.size == flat_our.size
        if not shape_ok:
            print(f"  {name:<45}  {t_ref.tensor_type!s:>6}  SIZE MISMATCH: ref={flat_ref.size} our={flat_our.size}  {FAIL}")
            n_fail += 1
            continue

        max_diff = float(np.max(np.abs(flat_ref - flat_our)))
        rmse     = float(np.sqrt(np.mean((flat_ref - flat_our)**2)))
        std      = float(np.std(flat_ref)) + 1e-9
        rel      = rmse / std

        if rel < 0.01:
            status = PASS; n_pass += 1
        elif rel < 0.05:
            status = WARN; n_warn += 1
        else:
            status = FAIL; n_fail += 1

        worst.append((rel, name, t_ref.tensor_type, max_diff, rmse, rel, flat_ref[:4], flat_our[:4]))
        print(f"  {name:<45}  {t_ref.tensor_type!s:>6}  {'OK':>8}  {max_diff:>10.4f}  {rmse:>10.4f}  {status}  (rel={rel:.3f})")

    print(f"\n{BOLD}Summary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL{RST}")

    # Print worst 5 mismatches
    worst.sort(reverse=True)
    if worst[:5]:
        print(f"\n{BOLD}Worst 5 mismatches:{RST}")
        for rel, name, dtype, max_diff, rmse, rel2, r4, o4 in worst[:5]:
            print(f"  {name:<45}  dtype={dtype!s}  max_diff={max_diff:.4f}  rel={rel:.3f}")
            print(f"    ref[:4] = {r4}")
            print(f"    our[:4] = {o4}")

    return 0 if n_fail == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
