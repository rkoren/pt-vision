# Vendored from Pose2Sim (https://github.com/perfanalytics/pose2sim), BSD 3-Clause License.
# Copyright (c) 2022, perfanalytics / David Pagnon. See LICENSE in this directory.
# Upstream: Pose2Sim/common.py (write_mot, write_trc) at commit 14d101c786e12abe24360534fb538d6ef589cfa0 (tag v0.10.49). See PROVENANCE.md for local modifications.
# ruff: noqa
# mypy: ignore-errors

import numpy as np
import pandas as pd


def write_mot(mot_path, mot_data, time_col, header):
    '''
    Write a .mot file (OpenSim motion file).
    
    INPUT:
    - mot_path: path to the output .mot file
    - mot_data: pandas DataFrame of filtered data columns
    - time_col: pandas Series of time values
    - header: list of header lines
    '''
    
    with open(mot_path, 'w') as f:
        for line in header:
            f.write(line)
        
        all_mot_data = pd.concat(
            [pd.Series(time_col, name='time').reset_index(drop=True),
             mot_data.reset_index(drop=True)
            ], axis=1)
        all_mot_data.to_csv(f, sep='\t', index=False, header=None, lineterminator='\n') 


def write_trc(trc_path, trc_data, frames_col, time_col, header):
    '''
    Write a .trc file (OpenSim marker trajectory file).
    
    INPUTS:
    - trc_path: path to the output .trc file
    - trc_data: pandas DataFrame of filtered data columns
    - frames_col: pandas Series of frame numbers
    - time_col: pandas Series of time values
    - header: list of header lines
    '''

    try:
        with open(trc_path, 'w') as trc_o:
            for line in header:
                trc_o.write(line)

            all_trc_data = pd.concat(
                [pd.Series(frames_col, name='Frame#').reset_index(drop=True),
                 pd.Series(time_col, name='Time').reset_index(drop=True),
                 trc_data.reset_index(drop=True)
                ], axis=1)
            all_trc_data.to_csv(trc_o, sep='\t', index=False, header=None, lineterminator='\n')

    except Exception as e:
        raise ValueError(f"Error writing TRC file at {trc_path}: {e}")


