# Vendored from Pose2Sim (https://github.com/perfanalytics/pose2sim), BSD 3-Clause License.
# Copyright (c) 2022, perfanalytics / David Pagnon. See LICENSE in this directory.
# Upstream: Pose2Sim/common.py (angle_dict, points_to_angles, fixed_angles) at commit 14d101c786e12abe24360534fb538d6ef589cfa0 (tag v0.10.49). See PROVENANCE.md for local modifications.
# ruff: noqa
# mypy: ignore-errors

import numpy as np


## CONSTANTS
# 4 points joint angle: between knee and ankle, and toe and heel. Add 90° offset and multiply by -1
# 3 points joint angle: between ankle, knee, hip. -180° offset, multiply by -1
# 2 points segment angle: between horizontal and ankle and knee, 0° offset, multiply by 1 (except trunk and head: -1)
angle_dict = { # lowercase!
    # joint angles
    'right ankle': [['RKnee', 'RAnkle', 'RBigToe', 'RHeel'], 'dorsiflexion', 90, 1],
    'left ankle': [['LKnee', 'LAnkle', 'LBigToe', 'LHeel'], 'dorsiflexion', 90, 1],
    'right knee': [['RAnkle', 'RKnee', 'RHip'], 'flexion', -180, 1],
    'left knee': [['LAnkle', 'LKnee', 'LHip'], 'flexion', -180, 1],
    'right hip': [['RKnee', 'RHip', 'Hip', 'Neck'], 'flexion', 0, -1],
    'left hip': [['LKnee', 'LHip', 'Hip', 'Neck'], 'flexion', 0, -1],
    # 'lumbar': [['Neck', 'Hip', 'RHip', 'LHip'], 'flexion', -180, -1],
    # 'neck': [['Head', 'Neck', 'RShoulder', 'LShoulder'], 'flexion', -180, -1],
    'right shoulder': [['RElbow', 'RShoulder', 'Hip', 'Neck'], 'flexion', 0, -1],
    'left shoulder': [['LElbow', 'LShoulder', 'Hip', 'Neck'], 'flexion', 0, -1],
    'right elbow': [['RWrist', 'RElbow', 'RShoulder'], 'flexion', 180, -1],
    'left elbow': [['LWrist', 'LElbow', 'LShoulder'], 'flexion', 180, -1],
    'right wrist': [['RElbow', 'RWrist', 'RIndex'], 'flexion', -180, 1],
    'left wrist': [['LElbow', 'LIndex', 'LWrist'], 'flexion', -180, 1],

    # segment angles
    'right foot': [['RBigToe', 'RHeel'], 'horizontal', 0, -1],
    'left foot': [['LBigToe', 'LHeel'], 'horizontal', 0, -1],
    'right shank': [['RAnkle', 'RKnee'], 'horizontal', 0, -1],
    'left shank': [['LAnkle', 'LKnee'], 'horizontal', 0, -1],
    'right thigh': [['RKnee', 'RHip'], 'horizontal', 0, -1],
    'left thigh': [['LKnee', 'LHip'], 'horizontal', 0, -1],
    'pelvis': [['LHip', 'RHip'], 'horizontal', 0, -1],
    'trunk': [['Neck', 'Hip'], 'horizontal', 0, -1],
    'shoulders': [['LShoulder', 'RShoulder'], 'horizontal', 0, -1],
    'head': [['Head', 'Neck'], 'horizontal', 0, -1],
    'right arm': [['RElbow', 'RShoulder'], 'horizontal', 0, -1],
    'left arm': [['LElbow', 'LShoulder'], 'horizontal', 0, -1],
    'right forearm': [['RWrist', 'RElbow'], 'horizontal', 0, -1],
    'left forearm': [['LWrist', 'LElbow'], 'horizontal', 0, -1],
    'right hand': [['RIndex', 'RWrist'], 'horizontal', 0, -1],
    'left hand': [['LIndex', 'LWrist'], 'horizontal', 0, -1]
    }


def points_to_angles(points_list):
    '''
    If len(points_list)==2, computes clockwise angle of ab vector w.r.t. horizontal (e.g. RBigToe, RHeel) 
    If len(points_list)==3, computes clockwise angle from a to c around b (e.g. Neck, Hip, Knee) 
    If len(points_list)==4, computes clockwise angle between vectors ab and cd (e.g. Neck Hip, RKnee RHip)
    
    Points can be 2D or 3D.
    If parameters are float, returns a float between -180.0 and 180.0
    If parameters are arrays, returns an array of floats between -180.0 and 180.0

    INPUTS:
    - points_list: list of arrays of points

    OUTPUTS:
    - ang_deg: float or array of floats. The angle(s) in degrees.
    '''

    if len(points_list) < 2: # if not enough points, return None
        return np.nan
    
    points_array = np.array(points_list)
    dimensions = points_array.shape[-1]

    if len(points_list) == 2:
        vector_u = points_array[0] - points_array[1]
        if dimensions == 2:
            # Segment angle w.r.t. horizontal: atan2(dy, dx) directly
            ang = np.arctan2(vector_u[1], vector_u[0])
            ang_deg = np.degrees(ang)
            return ang_deg
        else:
            if len(points_array.shape)==2:
                vector_v = np.array([1, 0, 0])
            else:
                vector_v = np.array([[1, 0, 0],] * points_array.shape[1])

    elif len(points_list) == 3:
        vector_u = points_array[0] - points_array[1]
        vector_v = points_array[2] - points_array[1]

    elif len(points_list) == 4:
        vector_u = points_array[1] - points_array[0]
        vector_v = points_array[3] - points_array[2]

    else:
        return np.nan

    if dimensions == 2:
        # atan2(cross, dot) gives angle from u to v (CCW positive).
        # Negate to get angle from v to u, matching the old atan2(u)-atan2(v) convention.
        # Range: (-180, 180) instead of (-360, 360), eliminating discontinuities.
        cross = vector_u[0] * vector_v[1] - vector_u[1] * vector_v[0]
        dot = vector_u[0] * vector_v[0] + vector_u[1] * vector_v[1]
        ang = -np.arctan2(cross, dot)
    else:
        cross_product = np.cross(vector_u, vector_v)
        dot_product = np.einsum('ij,ij->i', vector_u, vector_v) # np.dot(vector_u, vector_v) does not work with time series
        ang = np.arctan2(np.linalg.norm(cross_product, axis=-1), dot_product)

    ang_deg = np.degrees(ang)
    
    return ang_deg


def fixed_angles(points_list, ang_name):
    '''
    Compute angle and apply offset and scaling factor.
    Wraps result to (-180, 180] for most angles, or (-90, 90] for pelvis/shoulders.

    INPUTS:
    - points_list: list of arrays of points
    - ang_name: str. The name of the angle to consider.

    OUTPUTS:
    - ang: float. The angle in degrees.
    '''

    ang_params = angle_dict[ang_name]
    ang = points_to_angles(points_list)
    ang = (ang + ang_params[2]) * ang_params[3]
    
    if ang_name in ['pelvis', 'shoulders']:
        ang = (ang + 90) % 180 - 90
    else:
        ang = (ang + 180) % 360 - 180

    return ang


