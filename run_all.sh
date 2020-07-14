#!/usr/bin/env bash
CPU=2
for FOLDER in "Heatmap" "Scaling" "Relaxation"; do
    for EXPERIMENT in ${FOLDER}/*.yaml; do
      python3 ./scripts/simulated_experiment.py -f ${FOLDER} -c $(basename "${EXPERIMENT}") -j ${CPU}
    done
done

python3 ./scripts/DDG_vs_DG.py --input "Heatmap/Experiments/figure-3C"
python3 ./scripts/DDG_vs_DG.py --input "Heatmap/Experiments/figure-3D"
