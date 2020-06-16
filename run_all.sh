#!/usr/bin/env bash
CPU=2
for FOLDER in "Heatmap" "Scaling" "Relaxation"; do
    for EXPERIMENT in ${FOLDER}/*-6-4.yaml; do
      python3 ./scripts/simulated_experiment.py -f ${FOLDER} -c $(basename "${EXPERIMENT}") -j ${CPU}
    done
done
