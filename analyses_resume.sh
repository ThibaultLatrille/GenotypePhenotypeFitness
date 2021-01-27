#!/usr/bin/env bash
CPU=6
for FOLDER in "Heatmap" "Scaling" "Relaxation"; do
    for EXPERIMENT in ${FOLDER}/Experiments/*; do
      cd "${EXPERIMENT}"
      rm -rf merge*
      rm -rf *_plot
      snakemake -j ${CPU}
      cd ../../..
    done
done

python3 ./scripts/DDG_vs_DG.py --input "Heatmap/Experiments/figure-3C"
python3 ./scripts/DDG_vs_DG.py --input "Heatmap/Experiments/figure-3D"
