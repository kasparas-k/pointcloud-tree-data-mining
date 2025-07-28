#!/bin/sh

# In root containing raw geoportal data
IN_ESTONIA_RAW_ROOT=/data/Estonia/raw
IN_POLAND_RAW_ROOT=/data/Poland/raw

# Out root to write height-normalized geoportal point clouds
ESTONIA_NORM_ROOT=/data/Estonia/normalized
POLAND_NORM_ROOT=/data/Poland/normalized

Rscript pipeline/normalize_pointclouds.R  "${IN_ESTONIA_RAW_ROOT}" "${ESTONIA_NORM_ROOT}" "2,9" 6
Rscript pipeline/normalize_pointclouds.R  "${IN_POLAND_RAW_ROOT}" "${POLAND_NORM_ROOT}" "2,8" 12
