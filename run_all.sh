#!/usr/bin/env bash
for EXPERIMENT in ./Heatmap/Experiments/*; do
  NAME=$(basename "${EXPERIMENT}")
  echo "${NAME}"
  cd ${EXPERIMENT}
  # sed -i 's#X_PARAM_MAGNITUDE: 1#X_PARAM_MAGNITUDE: 2#g' config.yaml
  snakemake --unlock
  snakemake --printshellcmds --rerun-incomplete -j 8
  cd ../../..
done