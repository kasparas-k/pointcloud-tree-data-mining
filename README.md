# Point Cloud Data Mining with HD Map Priors for making Synthetic Forest Datasets

This is the official repository for the paper *Point Cloud Data Mining with HD Map Priors for making Synthetic Forest Datasets* 

Article:  [DOI:10.1109/JSTARS.2025.3593827](https://doi.org/10.1109/JSTARS.2025.3593827)

Benchmark dataset and other associated data: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16467712.svg)](https://doi.org/10.5281/zenodo.16467712).

## Table of Contents

- [Setup](#setup)
- [Running the original experiment](#running-the-original-experiment) 
- [Creating your own synthetic forests with pre-extracted trees](#creating-your-own-synthetic-forests-with-pre-extracted-trees)
- [Citing this work](#citing-this-work)

## Setup

### For the experiment
This code has several R, C++ and Python dependencies, so it is easiest to set up using the provided Conda environment file:

```bash
conda env create -f environment.yaml
conda activate geoportal_tree_mining
```

Then install the FEC package which wraps the original [Fast Euclidean Clustering](https://github.com/YizhenLAO/FEC) by Cao et al. (2022) code in a python package:
```bash
pip install git+https://github.com/kasparas-k/FEC.git
```

### Only making synthetic data
If not reproducing the original experiment, synthetic data can be made with pre-extracted individual tree point clouds and NeonTree benchmark bounding boxes.

```
pip install -r synthetic-data-requirements.txt
```

## Running the original experiment

### 1. Data

The following data was used in the experiment:
- Inferred tree position and size data fron NeonTree benchmark [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4770593.svg)](https://doi.org/10.5281/zenodo.4770593)
- Point clouds of Tallinn from the [Estonia's geoportal](https://geoportaal.maaamet.ee/eng/spatial-data/elevation-data-p308.html) and Warsaw from the [Poland's geoportal](https://www.geoportal.gov.pl/en/data/lidar-measurements-lidar/)
- Individual tree locations from [OpenStreetMap](https://www.openstreetmap.org/), [OverpassTurbo API](https://overpass-turbo.eu/) used in this study

Links to all geoportal point clouds used in this study (with an excess of some point clouds that contain no marked trees) can also be found in this repository's [url](url) directory. In the event that the URLs no longer work, they can still be used to identify the point clouds used.

Pre-extracted individual tree point clouds, generated synthetic scenes, and individual tree location query results can also be downloaded from the study's data release [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16467712.svg)](https://doi.org/10.5281/zenodo.16467712).

### 2. Extracting candidate individual tree point clouds

Set the paths to the required data, then normalize the process the data:
```bash
# height-normalize raw geoportal data
bash bash/pipeline/1_nornalize.sh

# extract initial isolated clusters
bash bash/pipeline/2_cluster.sh

# calculate cluster features and filter out unfit clusters
bash bash/pipeline/3_filter.sh
```

### 3. Inspecting individual tree point clouds

Inspect post-filtering individual trees in your favorite point cloud viewer. **Create a filtered `features.json` for all individual trees that pass visual checks.**

In this study, a barebones point cloud viewer was developed and used: https://github.com/kasparas-k/in-context-pointcloud-instances

This tool is incomplete and has some bugs, but it is recommended due to allowing the user to view individual tree point clouds in the context of their source point cloud, and to open Google Maps in the tree's location to inspect satellite images or Street View.

Sample cofigs are as follows:

```yaml
# Estonia

data:
  background_pointcloud_root: /data/Estonia/normalized  # root directory of the source normalized point clouds
  pointcloud_root: /data/Estonia/filtered_trees/pc  # root directory of filtered individual tree point clouds
  out_json: estonia_out.json  # where to write labels for point clouds
  projection: EPSG:3301  # CRS for Estonia

viewer:
  color_mode: DEF_RGB  # foreground highlighted, background natural RGB
```

```yaml
# Poland

data:
  background_pointcloud_root: /data/Poland/normalized  # root directory of the source normalized point clouds
  pointcloud_root: /data/Poland/filtered_trees/pc  # root directory of filtered individual tree point clouds
  out_json: poland_out.json  # where to write labels for point clouds
  projection: EPSG:2180  # CRS for Poland

viewer:
  color_mode: DEF_RGB  # foreground highlighted, background natural RGB
```

The processing of visual inspection labels and filtering of feature jsons is left up to the user. Filtered features are available in the study's data release (LINK COMING SOON).

### 4. Making synthetic data

Use NeonTree benchmark's bounding boxes from the BART2019 region to construct synthetic forest scenes using the selected individual tree point clouds:

```bash
bash bash/make_synthetic_data.sh
```

### 5. Evaluating tree instance segmentation algorithms

#### 5.1. Generate predictions
For [`lidR`](https://github.com/r-lidar/lidR) tree instance segmentation algorithms, run the provided script

```
python instance_segmentation/segment_all.py <SYNTHETIC_DATA_ROOT> <SEGMENTATION_RESULT_OUTPUT_ROOT>
```

Which will produce a matching subdirectory structure where every parent folder is named after each individual synthetic forest point cloud, `gt.laz` is a copy of the original point cloud with ground truth annotations in the `treeID` field, other `*.laz` files are named after the segmentation algorithm used and contain instance predictions in the `treeID` field.

For the *SegmentAnyTree* model by Wielgosz et al. (2024), generate predictions with the run_inference.sh script provided in the [original repository](https://github.com/SmartForest-no/SegmentAnyTree/tree/main) or refer to this study's fork of the code in https://github.com/kasparas-k/SegmentAnyTree. The instance labels are saved in the `PredInstance` field. In the output point cloud, we create a new field `gt_treeID` and assign the values of `treeID` (the ground truth labels) to it, then assign `PredInstance` to `treeID`. This has to be done because the original point ordering is not preserved in the output. 

#### 5.2. Calculate metrics

To get accuracy metrics per plot for every algorithm, run

```bash
python instance_segmentation/test_all.py <SEGMENTATION_RESULT_OUTPUT_ROOT> <EVALUATION_METRIC_JSON>
```

The resulting `*.json` file can be visualized and analyzed by the user.

## Creating your own synthetic forests with pre-extracted trees

Detailed guide on using your own data to create synthetic forest scenes will be added at a lated date. If shown interest, the update may be expedited.

## Citing this work

If you found part of this work useful in your own research or projects, please cite the original paper:

```
@ARTICLE{11103575,
  author={Karlauskas, Kasparas and Gelšvartas, Julius and Treigys, Povilas},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Point Cloud Data Mining With HD Map Priors for Making Synthetic Forest Datasets}, 
  year={2025},
  volume={18},
  number={},
  pages={19606-19617},
  keywords={Vegetation;Point cloud compression;Forestry;Feature extraction;Data mining;Three-dimensional displays;Surveys;Pipelines;Manuals;Urban areas;HD map;individual tree segmentation (ITS);LiDAR;synthetic data},
  doi={10.1109/JSTARS.2025.3593827}}
```
