#!/bin/bash

# filtered individual tree point clouds
TREE_ROOT='/data/Estonia/filtered_trees/'

# features of well segmented visually inspected trees
TREE_FEATURES='/data/Estonia/filtered_trees/checked_features.json'

# distribution to match
DISTRIBUTION='data/NeonTreeBbox/BART_2019.csv'

# where to write all synthetic point clouds
OUT_DIR='/data/Estonia/synthetic_scenes/BART2019'

for scale in 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2; do
    suffix=$(printf "%02d" $(echo "$scale * 10" | bc | cut -d. -f1))
    python make_synthetic_data/make_synthetic_data.py \
        "${TREE_ROOT}" "${TREE_FEATURES}" "${DISTRIBUTION}" "${OUT_DIR}"_"${suffix}" \
        --match_gauss --plot_ids 321000_4881000 \
        --bbox 321250 4881219 100 100 --bbox_absolute \
        --xy_scale "${scale}"
done


# Equivalent of above mentioned arguments for Poland
TREE_ROOT='/data/Poland/filtered_trees/'
TREE_FEATURES='/data/Poland/filtered_trees/checked_features.json'
DISTRIBUTION='/data/NeonTreeBbox/BART_2019.csv'
OUT_DIR='/data/Poland/synthetic_scenes/BART2019'

for scale in 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2; do
    suffix=$(printf "%02d" $(echo "$scale * 10" | bc | cut -d. -f1))
    python make_synthetic_data/make_synthetic_data.py \
        "${TREE_ROOT}" "${TREE_FEATURES}" "${DISTRIBUTION}" "${OUT_DIR}"_"${suffix}" \
        --match_gauss --plot_ids 321000_4881000 \
        --bbox 321250 4881219 100 100 --bbox_absolute \
        --xy_scale "${scale}"
done
