"""
Region lookup table definitions

Schema:
---
- region_label: pipeline label for region
- solver_label: label for region in solver
- region_type: "segmentation" if derived directly from an atlas or "derived" if calculated from other regions
- synthseg_labels: SynthSeg region labels
- nextbrain_labels: NextBrain region labels
- combine_regions: regions to combine for derived regions
- subtract_regions: regions to subtract for derived regions
"""

REGION_GROUPS = {
    "global": {
        "both": {"region_type": "derived", "combine_regions": ["wholebrain"], "subtract_regions": ["ventricles"]},
    },
    "wholebrain": {
        "both": {"region_type": "segmentation", "synthseg_labels": [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60]},
    },
    "ventricles": {
        "both": {"region_type": "segmentation", "synthseg_labels": [4, 5, 14, 15, 43, 44]},
    },
    "cerebrum": {
        "user_input": 1,
        "both": {"region_label": 1, "region_type": "segmentation", "synthseg_labels": [2, 3, 10, 11, 12, 13, 17, 18, 26, 28, 41, 42, 49, 50, 51, 52, 53, 54, 58, 60]},
        "L": {"region_label": 2, "region_type": "segmentation", "synthseg_labels": [2, 3, 10, 11, 12, 13, 17, 18, 26, 28]},
        "R": {"region_label": 3, "region_type": "segmentation", "synthseg_labels": [41, 42, 49, 50, 51, 52, 53, 54, 58, 60]},
    },
    "cerebrumGM": {
        "user_input": 2,
        "both": {"region_label": 4, "region_type": "derived", "combine_regions": ["cerebrum"], "subtract_regions": ["cerebrumWM"]},
        "L": {"region_label": 5, "region_type": "derived", "solver_label": 1, "combine_regions": ["cerebrum_L"], "subtract_regions": ["cerebrumWM_L"]},
        "R": {"region_label": 6, "region_type": "derived", "solver_label": 2, "combine_regions": ["cerebrum_R"], "subtract_regions": ["cerebrumWM_R"]},
    },
    "cerebrumWM": {
        "user_input": 3,
        "both": {"region_label": 7, "region_type": "segmentation", "synthseg_labels": [2, 41]},
        "L": {"region_label": 8, "region_type": "segmentation", "solver_label": 3, "synthseg_labels": [2]},
        "R": {"region_label": 9, "region_type": "segmentation", "solver_label": 4, "synthseg_labels": [41]},
    },
    "brainstem": {
        "user_input": 4,
        "both": {"region_label": 10, "region_type": "segmentation", "synthseg_labels": [16]},
        "L": {"region_label": 11, "region_type": "derived", "solver_label": 5},
        "R": {"region_label": 12, "region_type": "derived", "solver_label": 6},
    },
    "cerebellum": {
        "user_input": 5,
        "both": {"region_label": 13, "region_type": "segmentation", "synthseg_labels": [7, 8, 46, 47]},
        "L": {"region_label": 14, "region_type": "segmentation", "synthseg_labels": [7, 8]},
        "R": {"region_label": 15, "region_type": "segmentation", "synthseg_labels": [46, 47]},
    },
    "cerebellumGM": {
        "user_input": 6,
        "both": {"region_label": 16, "region_type": "derived", "combine_regions": ["cerebellum"], "subtract_regions": ["cerebellumWM"]},
        "L": {"region_label": 17, "region_type": "derived", "solver_label": 7, "combine_regions": ["cerebellum_L"], "subtract_regions": ["cerebellumWM_L"]},
        "R": {"region_label": 18, "region_type": "derived", "solver_label": 8, "combine_regions": ["cerebellum_R"], "subtract_regions": ["cerebellumWM_R"]},
    },
    "cerebellumWM": {
        "user_input": 7,
        "both": {"region_label": 19, "region_type": "segmentation", "synthseg_labels": [7, 46]},
        "L": {"region_label": 20, "region_type": "segmentation", "solver_label": 9, "synthseg_labels": [7]},
        "R": {"region_label": 21, "region_type": "segmentation", "solver_label": 10, "synthseg_labels": [46]},
    },
    "hippocampus": {
        "user_input": 8,
        "both": {"region_label": 22, "region_type": "segmentation", "synthseg_labels": [17, 53]},
        "L": {"region_label": 23, "region_type": "segmentation", "solver_label": 1, "synthseg_labels": [17]},
        "R": {"region_label": 24, "region_type": "segmentation", "solver_label": 2, "synthseg_labels": [53]},
    },
    "amygdala": {
        "user_input": 9,
        "both": {"region_label": 25, "region_type": "segmentation", "synthseg_labels": [18, 54]},
        "L": {"region_label": 26, "region_type": "segmentation", "solver_label": 1, "synthseg_labels": [18]},
        "R": {"region_label": 27, "region_type": "segmentation", "solver_label": 2, "synthseg_labels": [54]},
    },
    "thalamus": {
        "user_input": 10,
        "both": {"region_label": 28, "region_type": "segmentation", "synthseg_labels": [10, 49]},
        "L": {"region_label": 29, "region_type": "segmentation", "solver_label": 1, "synthseg_labels": [10]},
        "R": {"region_label": 30, "region_type": "segmentation", "solver_label": 2, "synthseg_labels": [49]},
    },
    "caudate": {
        "user_input": 11,
        "both": {"region_label": 31, "region_type": "segmentation", "synthseg_labels": [11, 50]},
        "L": {"region_label": 32, "region_type": "segmentation", "solver_label": 1, "synthseg_labels": [11]},
        "R": {"region_label": 33, "region_type": "segmentation", "solver_label": 2, "synthseg_labels": [50]},
    },
    "putamen": {
        "user_input": 12,
        "both": {"region_label": 34, "region_type": "segmentation", "synthseg_labels": [12, 51]},
        "L": {"region_label": 35, "region_type": "segmentation", "solver_label": 1, "synthseg_labels": [12]},
        "R": {"region_label": 36, "region_type": "segmentation", "solver_label": 2, "synthseg_labels": [51]},
    },
    "pallidum": {
        "user_input": 13,
        "both": {"region_label": 37, "region_type": "segmentation", "synthseg_labels": [13, 52]},
        "L": {"region_label": 38, "region_type": "segmentation", "solver_label": 1, "synthseg_labels": [13]},
        "R": {"region_label": 39, "region_type": "segmentation", "solver_label": 2, "synthseg_labels": [52]},
    },
    "entorhinal_cortex": {
        "user_input": 14,
        "both": {"region_label": 40, "region_type": "segmentation", "nextbrain_labels": [2006]},
        "L": {"region_label": 41, "region_type": "derived", "solver_label": 1},
        "R": {"region_label": 42, "region_type": "derived", "solver_label": 2},
    },
    "parahippocampal_cortex": {
        "user_input": 15,
        "both": {"region_label": 43, "region_type": "segmentation", "nextbrain_labels": [2016]},
        "L": {"region_label": 44, "region_type": "derived", "solver_label": 1},
        "R": {"region_label": 45, "region_type": "derived", "solver_label": 2},
    },
    "posterior_cingulate": {
        "user_input": 16,
        "both": {"region_label": 46, "region_type": "segmentation", "nextbrain_labels": [2023]},
        "L": {"region_label": 47, "region_type": "derived", "solver_label": 1},
        "R": {"region_label": 48, "region_type": "derived", "solver_label": 2},
    },
    "precuneus": {
        "user_input": 17,
        "both": {"region_label": 49, "region_type": "segmentation", "nextbrain_labels": [2025]},
        "L": {"region_label": 50, "region_type": "derived", "solver_label": 1},
        "R": {"region_label": 51, "region_type": "derived", "solver_label": 2},
    },
    "inferior_parietal": {
        "user_input": 18,
        "both": {"region_label": 52, "region_type": "segmentation", "nextbrain_labels": [2008]},
        "L": {"region_label": 53, "region_type": "derived", "solver_label": 1},
        "R": {"region_label": 54, "region_type": "derived", "solver_label": 2},
    },
    "inferior_temporal": {
        "user_input": 19,
        "both": {"region_label": 55, "region_type": "segmentation", "nextbrain_labels": [2009]},
        "L": {"region_label": 56, "region_type": "derived", "solver_label": 1},
        "R": {"region_label": 57, "region_type": "derived", "solver_label": 2},
    },
}
