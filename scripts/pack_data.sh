#!/bin/bash
# Pack the 2.8-degree model inputs (converted transport matrices, grid, boxes, temperature,
# light, N0/Si0) for copying to a GPU machine. Output: num_tm_2p8.tar (~420 MB; .mat files are
# already compressed, so no gzip).
set -euo pipefail
cd "$(dirname "$0")/.."
tar -cf num_tm_2p8.tar -C NUMmodel/TMs MITgcm_2.8deg
ls -lh num_tm_2p8.tar
