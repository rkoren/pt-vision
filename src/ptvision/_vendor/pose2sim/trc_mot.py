# Vendored from Pose2Sim (https://github.com/perfanalytics/pose2sim), BSD 3-Clause License.
# Copyright (c) 2022, perfanalytics / David Pagnon. See LICENSE in this directory.
# Upstream: Pose2Sim/common.py (read_mot, write_mot, read_trc, write_trc) at commit 14d101c786e12abe24360534fb538d6ef589cfa0 (tag v0.10.49). See PROVENANCE.md for local modifications.
# ruff: noqa
# mypy: ignore-errors

import numpy as np
import pandas as pd


def read_mot(mot_path):
    '''
    Read a .mot file (OpenSim motion file).
    Also reads .sto files.
    
    INPUT:
    - mot_path: path to the .mot file
    
    OUTPUT:
    - data: pandas DataFrame with all data columns, excluding time
    - time_col: pandas Series of time values
    - header: list of header lines
    '''
    
    header_lines = []
    header_end_line = 0
    with open(mot_path, 'r') as f:
        lines = f.readlines()
    
    # Find the end of the header (line with "endheader")
    for i, line in enumerate(lines):
        if line.strip().lower() == 'endheader':
            header_end_line = i
            break
        
    header_lines = lines[:header_end_line + 1]
    
    # Read the data portion
    data = pd.read_csv(mot_path, sep='\t', skiprows=header_end_line + 1)
    
    # Separate time column from data columns
    time_col = data.iloc[:, 0]  # first column is time
    data_cols = data.iloc[:, 1:]  # remaining columns are data
    
    return data_cols, time_col, header_lines


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


def read_trc(trc_path):
    '''
    Read a .trc file (OpenSim marker trajectory file).

    INPUTS:
    - trc_path (str): The path to the TRC file.

    OUTPUTS:
    - Q_coords (DataFrame): A DataFrame containing the Q coordinates of the markers.
    - frames_col (Series): A Series containing the frame numbers.
    - time_col (Series): A Series containing the time values.
    - markers (list): A list of marker names.
    - header (list): A list of header lines.
    '''

    try:
        with open(trc_path, 'r') as trc_file:
            header = [next(trc_file) for _ in range(5)]
        markers = header[3].split('\t')[2::3]
        markers = [m.strip() for m in markers if m.strip()] # remove last \n character
       
        trc_df = pd.read_csv(trc_path, sep="\t", skiprows=4, encoding='utf-8')
        frames_col, time_col = trc_df.iloc[:, 0], trc_df.iloc[:, 1]
        trc_data = trc_df.drop(trc_df.columns[[0, 1]], axis=1)
        trc_data = trc_data.loc[:, ~trc_data.columns.str.startswith('Unnamed')] # remove unnamed columns
        trc_data.columns = np.array([[m,m,m] for m in markers]).ravel().tolist()

        return trc_data, frames_col, time_col, markers, header
    
    except Exception as e:
        raise ValueError(f"Error reading TRC file at {trc_path}: {e}")
    

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


