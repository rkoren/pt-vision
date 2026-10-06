# Vendored from Pose2Sim (https://github.com/perfanalytics/pose2sim), BSD 3-Clause License.
# Copyright (c) 2022, perfanalytics / David Pagnon. See LICENSE in this directory.
# Upstream: Pose2Sim/filtering.py (hampel_filter) at commit 14d101c786e12abe24360534fb538d6ef589cfa0 (tag v0.10.49). See PROVENANCE.md for local modifications.
# ruff: noqa
# mypy: ignore-errors

import numpy as np


def hampel_filter(col, window_size=7, n_sigma=2):
    '''
    Hampel filter for outlier rejection before other filtering methods.
    Takes a sliding window of size 7, calculates its median and standard deviation, 
    replaces value by median if difference is more than 2 times the standard deviation (95% confidence interval), 
    else keeps the value.
    '''

    col_filtered = col.copy()
    half_window = window_size // 2
    
    for i in range(half_window, len(col) - half_window):
        window = col[i-half_window:i+half_window+1]
        median = np.median(window)
        mad = np.median(np.abs(window - median))  # Median Absolute Deviation
        
        if mad != 0:
            modified_z_score = 0.6745 * (col[i] - median) / mad #75% percentile from median
            if np.abs(modified_z_score) > n_sigma:
                col_filtered[i] = median
    
    return col_filtered


