#!/bin/sh

# Root directory of normalized geoportal point clouds
ESTONIA_NORM_ROOT=/data/Estonia/normalized
POLAND_NORM_ROOT=/data/Poland/normalized

# Where to write initial clusters
ESTONIA_INITIAL_CLUSTERS=/data/Estonia/isolated_clusters
POLAND_INITIAL_CLUSTERS=/data/Poland/isolated_clusters

# GeoJson files containing tree coordinates
POLAND_GEOLOC=/data/tree_centers/trees_warsaw.geojson
ESTONIA_GEOLOC=/data/tree_centers/trees_tallinn.geojson

date
echo "****Estonia****"
python pipeline/extract_isolated.py \
    "${ESTONIA_NORM_ROOT}" \
    "${ESTONIA_INITIAL_CLUSTERS}" \
    --tree_center_geojson "${ESTONIA_GEOLOC}" \
    --country Estonia \
    --tree_class 5
date

date
echo "****Poland****"
python pipeline/extract_isolated.py \
    "${POLAND_NORM_ROOT}" \
    "${POLAND_INITIAL_CLUSTERS}" \
    --tree_center_geojson "${POLAND_GEOLOC}" \
    --country Poland \
    --tree_class 5
date
