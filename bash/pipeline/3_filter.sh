#!/bin/sh

# Initial clusters
ESTONIA_INITIAL_CLUSTERS=/data/Estonia/isolated_clusters
POLAND_INITIAL_CLUSTERS=/data/Poland/isolated_clusters

# GeoJson files containing tree coordinates
POLAND_GEOLOC=/data/tree_centers/trees_warsaw.geojson
ESTONIA_GEOLOC=/data/tree_centers/trees_tallinn.geojson

# Where to write feature jsons
ESTONIA_FEATURE_JSON=/data/Estonia/fec_features.json
POLAND_FEATURE_JSON=/data/Poland/fec_features.json

# Where to write filtered tree point clouds
ESTONIA_FILTERED_TREES=/data/Estonia/filtered_trees/
POLAND_FILTERED_TREES=/data/Poland/filtered_trees/


python pipeline/get_cluster_features.py \
    --country Estonia \
    --tree_center_geojson "${ESTONIA_GEOLOC}" \
    "${ESTONIA_INITIAL_CLUSTERS}" \
    "${ESTONIA_FEATURE_JSON}"
python pipeline/filter_clusters_by_features.py \
    "${ESTONIA_INITIAL_CLUSTERS}" \
    "${ESTONIA_FILTERED_TREES}" \
    "${ESTONIA_FEATURE_JSON}"

python pipeline/get_cluster_features.py \
    --country Poland \
    --tree_center_geojson "${POLAND_GEOLOC}" \
    "${POLAND_INITIAL_CLUSTERS}" \
    "${POLAND_FEATURE_JSON}"
python pipeline/filter_clusters_by_features.py  \
    "${POLAND_INITIAL_CLUSTERS}" \
    "${POLAND_FILTERED_TREES}" \
    "${POLAND_FEATURE_JSON}"
