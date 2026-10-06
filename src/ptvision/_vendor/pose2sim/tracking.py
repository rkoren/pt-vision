# Vendored from Pose2Sim (https://github.com/perfanalytics/pose2sim), BSD 3-Clause License.
# Copyright (c) 2022, perfanalytics / David Pagnon. See LICENSE in this directory.
# Upstream: Pose2Sim/common.py (pad_shape, sort_people_sports2d) at commit 14d101c786e12abe24360534fb538d6ef589cfa0 (tag v0.10.49). See PROVENANCE.md for local modifications.
# ruff: noqa
# mypy: ignore-errors

import numpy as np
from scipy.optimize import linear_sum_assignment


def pad_shape(arr, target_len, fill_value=np.nan):
    '''
    Pads an array to the target length with specified fill values
    
    INPUTS:
    - arr: Input array to be padded.
    - target_len: The target length of the first dimension after padding.
    - fill_value: The value to use for padding (default: np.nan).
    
    OUTPUTS:
    - Padded array with shape (target_len, ...) matching the input dimensions.
    '''

    if len(arr) < target_len:
        pad_shape = (target_len - len(arr),) + arr.shape[1:]
        padding = np.full(pad_shape, fill_value)
        return np.concatenate((arr, padding))
    
    return arr




def sort_people_sports2d(keyptpre, keypt, scores=None, match_by='bbox', pred_displacement=None, max_dist=None, max_unseen_frames=None, frames_since_last_seen=None):
    '''
    Associate persons across frames, avoiding ID swaps
    A person's ID is matched to a detection in the next frame when they are close to each other
    Both the 2D and 3D cases are supported
    N.B.: Requires the pad_shape function

    Algorithm:
    - By default, sort people based on minimal frame-by-frame keypoint displacement
    - Instead of keypoints, optionally works on bounding boxes or centroids
    - Instead of minimal displacement, optionally predicts positions and minimizes their distances to detections
    - Tracks people through temporary disappearances
    - Sets a maximum number of frames beyond which a non-detected person is forgotten
    - Sets a maximum distance beyond which persons cannot be associated with each other

    Recommendations:
    - Predict displacement: 
               Initialize with np.zeros((n_persons, 2)) in 2D, np.zeros((n_persons, 3)) in 3D, or to None​ for not using it.
               Use when people move in sustained straight lines (eg sprinting, cycling, swimming) or in case of long occlusions or slow framerate. 
               Don't use if there are sudden changes of direction (eg basketball, hopping). 
               Experiment if you are unsure (eg football, ...). 
               
    - Match by Keypoints vs Centroid vs Bbox: 
      - In 2D: Usually favor keypoints, especially if people are passing closely in front of each other with distinct limb positions and limited frame-by-frame limb motion (eg yoga class, ballroom dance, and even wrestling and gymnastics if the framerate is high).
               Favor centroid in crowded scenes where people are very small in the image (eg < 40 pixels). 
               Favor bbox with `predict_displacement=true` with people of different sizes moving in a straight line with fast limb motion and large occlusions (eg sprinting with occlusions).
      - In 3D: Favor keypoints if depth is reliable (eg multiview triangulation), with relatively little frame-by-frame limb motion, and if people are "hugging" each other.
               Favor centroid if depth is unreliable (eg monocular 3D).
               Bboxes are not reliable.

    INPUTS:
    - keyptpre: (K, L, M) array of coordinates for K persons in the previous frame, L keypoints, M 2D/3D coordinates
    - keypt: idem keyptpre, for current frame
    - score: (K, L) array of confidence scores for K persons, L keypoints (optional) 
    - match_by: 'centroid' (default): Euclidean distance between keypoint centroids
                'keypoints': Mean over per-keypoint distances
    - pred_displacements: (K, M) array of M-dimensional displacement for K persons.
                  Initialize with np.zeros((n_persons, 2)) in 2D, np.zeros((n_persons, 3)) in 3D, or None to disable (default).
                  Estimated as prev_centroid_loc - prev_prev_centroid_loc (linear translation), kept if the person has momentarily disappeared.
    - max_dist: Max distance a person can jump from their previous position before being considered as a new one.
                In pixels (eg 50 pixels) or meters (eg 1 meter). Increase if predict_displacement=False. 
                If match_by='bbox', should be interpreted as `1-IoU` (eg 0.7 for a minimum of 30% overlap).
                If None (default), no threshold applied.
                If distance > max_dist, person is treated as new (no association)
    - max_unseen_frames: int or None. Max number of seconds that a person can be unseen before the next person passing by is given a new ID.
                        If None (default), staleness tracking is disabled.
    - frames_since_last_seen: array of shape (K,) tracking how many frames since each previous person
                              was last seen. Required when max_unseen_frames is not None. Updated and returned.
    
    OUTPUTS:
    - tracked_keypoints: (N, L, M) array with N reordered persons with values of previous frame in empty slots (for tracking)
                         Use this as keyptpre in the next frame.
    - sorted_keypoints: (N, L, M) array with N reordered persons with nans in empty slots (actually detected persons)
    - sorted_scores: If scores in inputs: (N, L) array with reordered scores with nans in empty slots
                     Otherwise, (N,) array mapping output index to original curr index (-1 if unmatched)
    - frames_since_last_seen: (N,) updated staleness counter array (only returned when max_unseen_frames is not None)
    - pred_displacements: (N, M) updated velocity array. Only returned when velocities is not None.
    '''

    use_staleness = max_unseen_frames is not None
    use_prediction = pred_displacement is not None

    # Handle empty frames
    n_prev = len(keyptpre)
    n_curr = len(keypt)
    if n_prev == 0 and n_curr == 0:
        if scores is not None:
            ret = (np.array([]), np.array([]), np.array([]))
        else:
            ret = (np.array([]), np.array([]), np.array([], dtype=int))
        if use_staleness: ret += (np.array([], dtype=int),)
        if use_prediction: ret += (pred_displacement,)
        return ret
    if n_prev == 0:
        new_fsls = np.zeros(n_curr, dtype=int)
        if scores is not None:
            ret = (keypt.copy(), keypt, scores)
        else:
            ret = (keypt.copy(), keypt, np.arange(n_curr))
        if use_staleness: ret += (new_fsls,)
        if use_prediction: ret += (np.zeros((n_curr, keypt.shape[2])),)
        return ret
    
    # Initialize frames_since_last_seen if staleness tracking is enabled but not yet started
    if use_staleness and frames_since_last_seen is None:
        frames_since_last_seen = np.zeros(n_prev, dtype=int)

    # Predict positions of previous frame keypoints if prediction is enabled
    keyptpre_orig = keyptpre.copy()
    if use_prediction:
        keyptpre = keyptpre + pred_displacement[:, np.newaxis, :]

    # Broadcasts the computation of distance matrix for all possible associations instead of using euclidean_distance in a loop
    if match_by == 'bbox':
        bboxes_pre = np.stack([
            np.nanmin(keyptpre[..., 0], axis=1),
            np.nanmin(keyptpre[..., 1], axis=1),
            np.nanmax(keyptpre[..., 0], axis=1),
            np.nanmax(keyptpre[..., 1], axis=1)
        ], axis=1)
        bboxes = np.stack([
            np.nanmin(keypt[..., 0], axis=1),
            np.nanmin(keypt[..., 1], axis=1),
            np.nanmax(keypt[..., 0], axis=1),
            np.nanmax(keypt[..., 1], axis=1)
        ], axis=1)
        # IoU matrix
        inter_x1 = np.maximum(bboxes_pre[:, None, 0], bboxes[None, :, 0])
        inter_y1 = np.maximum(bboxes_pre[:, None, 1], bboxes[None, :, 1])
        inter_x2 = np.minimum(bboxes_pre[:, None, 2], bboxes[None, :, 2])
        inter_y2 = np.minimum(bboxes_pre[:, None, 3], bboxes[None, :, 3])
        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        area_pre  = (bboxes_pre[:, 2] - bboxes_pre[:, 0]) * (bboxes_pre[:, 3] - bboxes_pre[:, 1])
        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        union_area = area_pre[:, None] + area[None, :] - inter_area
        iou_matrix = np.where(union_area > 0, inter_area / union_area, 0.0)
        dist_matrix = 1.0 - iou_matrix  # cost = 1 − IoU
    
    elif match_by == 'centroid':
        centpre = np.nanmean(keyptpre, axis=1)
        cent = np.nanmean(keypt, axis=1)
        diff = cent[np.newaxis, :, :] - centpre[:, np.newaxis, :]
        dist_matrix = np.sqrt(np.nansum(diff**2, axis=2))
    
    else:  # 'keypoints'
        diff = keypt[None, :, :, :] - keyptpre[:, None, :, :]
        distances_per_keypoint = np.sqrt(np.nansum(diff**2, axis=3))
        dist_matrix = np.nanmean(distances_per_keypoint, axis=2)

    # Inflate distances for stale persons so they won't attract new detections
    if use_staleness:
        stale_mask = frames_since_last_seen >= max_unseen_frames
        dist_matrix[stale_mask, :] = 1e10

    # Hungarian algorithm instead of min_with_single_indices in previous versions (faster when many people)
    # Finds the associations with smallest distances between previous and current frame, each person associated only once, with max distance threshold
    assignment_matrix = dist_matrix
    if max_dist is not None:
        assignment_matrix = np.where(dist_matrix > max_dist, 1e10, dist_matrix)
    pre_ids, curr_ids = linear_sum_assignment(assignment_matrix)
    
    # Filter associations based on max_dist threshold
    valid_associations = []
    if max_dist is not None:
        for pre_id, curr_id in zip(pre_ids, curr_ids):
            if dist_matrix[pre_id, curr_id] <= max_dist:
                valid_associations.append((pre_id, curr_id))
    else:
        valid_associations = list(zip(pre_ids, curr_ids))
    
    # Track non associated people in current frame (new people)
    associated_curr_ids = set([curr_id for _, curr_id in valid_associations])
    unassociated_curr_ids = [i for i in range(n_curr) if i not in associated_curr_ids]

    # Create sorted_keypoints filled with nans (existing slots + truly new persons)
    n_total = n_prev + len(unassociated_curr_ids)
    sorted_keypoints = np.full((n_total,) + keypt.shape[1:], np.nan)
    if scores is not None:
        sorted_scores = np.full((n_total,) + scores.shape[1:], np.nan)
    else:
        sorted_ids = np.full(n_total, -1)

    # Map valid associations to their previous indices
    for prev_idx, curr_idx in valid_associations:
        sorted_keypoints[prev_idx] = keypt[curr_idx]
        if scores is not None:
            sorted_scores[prev_idx] = scores[curr_idx]
        else:
            sorted_ids[prev_idx] = curr_idx
    
    # Add unassociated current detections as new people
    for new_idx, curr_idx in enumerate(unassociated_curr_ids):
        sorted_keypoints[n_prev + new_idx] = keypt[curr_idx]
        if scores is not None:
            sorted_scores[n_prev + new_idx] = scores[curr_idx]
        else:
            sorted_ids[n_prev + new_idx] = curr_idx

    # Keep track of previous values when missing
    keyptpre_padded = pad_shape(keyptpre_orig, n_total, fill_value=np.nan)  # Pad keyptpre to match new size
    tracked_keypoints = np.where(
        np.isnan(sorted_keypoints) & ~np.isnan(keyptpre_padded), 
        keyptpre_padded, 
        sorted_keypoints)
    
    # Update frames_since_last_seen
    if use_staleness:
        updated_fsls = np.full(n_total, 0, dtype=int)
        updated_fsls[:n_prev] = frames_since_last_seen + 1
        for prev_idx, _ in valid_associations:
            updated_fsls[prev_idx] = 0  # reset for matched (including recycled stale slots)
    
    # Update predictions
    if use_prediction:
        updated_predictions = np.zeros((n_total, keyptpre_orig.shape[2]))
        updated_predictions[:n_prev] = pred_displacement # carry forward for unmatched tracks
        if valid_associations:
            pre_va, curr_va = zip(*valid_associations)
            pre_va, curr_va = list(pre_va), list(curr_va)
            updated_predictions[pre_va] = (np.nanmean(keypt[curr_va], axis=1) -
                                        np.nanmean(keyptpre_orig[pre_va], axis=1))

    # Returns
    if use_staleness:
        if scores is not None:
            ret = (tracked_keypoints, sorted_keypoints, sorted_scores, updated_fsls)
        else:
            ret = (tracked_keypoints, sorted_keypoints, sorted_ids, updated_fsls)
    else:
        if scores is not None:
            ret = (tracked_keypoints, sorted_keypoints, sorted_scores)
        else:
            ret = (tracked_keypoints, sorted_keypoints, sorted_ids)
    if use_prediction:
        ret += (updated_predictions,)
    return ret


