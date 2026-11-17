#!/bin/bash
# Derive gfx90c (Vega APU) Tensile logic from the vega10 (gfx900) asm_full logic.
# gfx90c has the same ISA as gfx900 (plus xnack), so the tuned gfx900 assembly kernels
# can be generated for gfx90c by changing only the ArchitectureName.
# Only the GEMM types needed on this APU are converted: SGEMM (SB), HGEMM (HB) and
# half-precision with high precision accumulate (HHS_BH, HSS_BH), incl. general batched.
set -e
cd "$(dirname "$0")"
src=../asm_full/vega10
for f in "$src"/vega10_Cijk_*_{SB,SB_GB,SBIc,SBIIc,HB,HB_GB,HHS_BH,HHS_BH_GB,HSS_BH,HSS_BH_GB}.yaml; do
  [ -e "$f" ] || continue
  out=gfx90c/$(basename "$f")
  sed '3s/^- gfx900$/- gfx90c/' "$f" > "$out"
  sed -n 3p "$out" | grep -qx -- '- gfx90c' || { echo "unexpected header in $f" >&2; exit 1; }
done
ls gfx90c | wc -l
