"""
Основные функции калибровки камер и обработки изображений.
"""
import copy
import io
import itertools
import numpy as np
import cv2
import traceback
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import math
import os
import time
from dataclasses import dataclass
from contextlib import nullcontext, redirect_stdout
from scipy import stats

from . import utils, triangulation, camera_pose, bundle_adjustment


def _stable_sort_key(value):
    text = str(value)
    try:
        return (0, int(text))
    except (TypeError, ValueError):
        return (1, text)

def initialize_calibration():
    """
    Инициализирует данные для калибровки.
    
    Returns:
        dict: Данные калибровки
    """
    return {
        'cameras': {},         # Словарь камер {id: (R, t)}
        'points_3d': {},       # Словарь 3D точек {id: point}
        'camera_points': {},   # Словарь точек для каждой камеры {camera_id: {point_id: point_2d}}
        'raw_camera_points': {},  # Исходные ручные 2D-точки до image-based refinement
        'observation_confidences': {},  # Вес наблюдений {camera_id: {point_id: confidence}}
        'line_support_data': {},  # Стабильные line-triplets и per-observation line support
        'subpixel_refinement_enabled': True,
        'K': None,             # Матрица внутренних параметров камеры
        'dist_coeffs': None,   # Коэффициенты дисторсии
        'images': {},          # Словарь изображений {id: image_path}
        'reconstruction_ids': [], # Список ID камер, использованных в реконструкции
        'debug_logging': False,
        'precision_cleanup_enabled': True,
        'precision_target_mean_px': 0.5,
        'precision_target_p95_px': 1.0,
        'precision_target_max_px': 1.5,
        'precision_cleanup_rounds': 4,
        'strict_track_consistency': True,
        'preserve_unreconstructed_annotations': True,
        'pose_scaffold_enabled': True,
        'pose_scaffold_min_track_length': 3,
        'pose_scaffold_nonmember_confidence': 0.22,
        'two_stage_pose_scaffold_recovery': False,
        'two_stage_pose_scaffold_min_points': 12,
        'two_stage_pose_scaffold_target_ratio': 0.90,
        'two_stage_pose_scaffold_min_camera_support': 6,
        'point_drift_trace_enabled': False,
        'point_drift_trace_capture_all': True,
        'point_drift_trace_max_points': 0,
    }


def _snapshot_calibration_state(calib_data):
    individual_intrinsics = {}
    for key, value in calib_data.items():
        if isinstance(key, str) and key.startswith('K_') and value is not None:
            individual_intrinsics[key] = np.array(value, copy=True)
    return {
        'cameras': copy.deepcopy(calib_data.get('cameras', {})),
        'points_3d': copy.deepcopy(calib_data.get('points_3d', {})),
        'camera_points': copy.deepcopy(calib_data.get('camera_points', {})),
        'raw_camera_points': copy.deepcopy(calib_data.get('raw_camera_points', {})),
        'observation_confidences': copy.deepcopy(calib_data.get('observation_confidences', {})),
        'line_support_data': copy.deepcopy(calib_data.get('line_support_data', {})),
        'subpixel_refinement_enabled': bool(calib_data.get('subpixel_refinement_enabled', True)),
        'subpixel_refinement_stats': copy.deepcopy(calib_data.get('subpixel_refinement_stats', {})),
        'reconstruction_ids': list(calib_data.get('reconstruction_ids', [])),
        'K': None if calib_data.get('K') is None else np.array(calib_data['K'], copy=True),
        'dist_coeffs': None if calib_data.get('dist_coeffs') is None else np.array(calib_data['dist_coeffs'], copy=True),
        'images': copy.deepcopy(calib_data.get('images', {})),
        'image_root': calib_data.get('image_root'),
        'image_width': calib_data.get('image_width'),
        'image_height': calib_data.get('image_height'),
        'debug_logging': bool(calib_data.get('debug_logging', False)),
        'precision_cleanup_enabled': bool(calib_data.get('precision_cleanup_enabled', True)),
        'precision_target_mean_px': float(calib_data.get('precision_target_mean_px', 0.5)),
        'precision_target_p95_px': float(calib_data.get('precision_target_p95_px', 1.0)),
        'precision_target_max_px': float(calib_data.get('precision_target_max_px', 1.5)),
        'precision_cleanup_rounds': int(calib_data.get('precision_cleanup_rounds', 4)),
        'strict_track_consistency': bool(calib_data.get('strict_track_consistency', True)),
        'preserve_unreconstructed_annotations': bool(calib_data.get('preserve_unreconstructed_annotations', True)),
        'pose_scaffold_enabled': bool(calib_data.get('pose_scaffold_enabled', True)),
        'pose_scaffold_min_track_length': int(calib_data.get('pose_scaffold_min_track_length', 3)),
        'pose_scaffold_nonmember_confidence': float(calib_data.get('pose_scaffold_nonmember_confidence', 0.22)),
        'two_stage_pose_scaffold_recovery': bool(calib_data.get('two_stage_pose_scaffold_recovery', False)),
        'two_stage_pose_scaffold_min_points': int(calib_data.get('two_stage_pose_scaffold_min_points', 12)),
        'two_stage_pose_scaffold_target_ratio': float(calib_data.get('two_stage_pose_scaffold_target_ratio', 0.90)),
        'two_stage_pose_scaffold_min_camera_support': int(calib_data.get('two_stage_pose_scaffold_min_camera_support', 6)),
        'point_drift_trace_enabled': bool(calib_data.get('point_drift_trace_enabled', False)),
        'point_drift_trace_capture_all': bool(calib_data.get('point_drift_trace_capture_all', True)),
        'point_drift_trace_max_points': int(calib_data.get('point_drift_trace_max_points', 0) or 0),
        'initial_focal_estimate': copy.deepcopy(calib_data.get('initial_focal_estimate')),
        'individual_intrinsics': individual_intrinsics,
    }


def _restore_calibration_state(calib_data, snapshot):
    calib_data['cameras'] = snapshot['cameras']
    calib_data['points_3d'] = snapshot['points_3d']
    calib_data['camera_points'] = snapshot['camera_points']
    calib_data['raw_camera_points'] = snapshot.get('raw_camera_points', {})
    calib_data['observation_confidences'] = snapshot.get('observation_confidences', {})
    calib_data['line_support_data'] = snapshot.get('line_support_data', {})
    calib_data['subpixel_refinement_enabled'] = bool(snapshot.get('subpixel_refinement_enabled', True))
    calib_data['subpixel_refinement_stats'] = snapshot.get('subpixel_refinement_stats', {})
    calib_data['reconstruction_ids'] = snapshot['reconstruction_ids']
    calib_data['K'] = None if snapshot['K'] is None else np.array(snapshot['K'], copy=True)
    calib_data['dist_coeffs'] = None if snapshot['dist_coeffs'] is None else np.array(snapshot['dist_coeffs'], copy=True)
    calib_data['images'] = snapshot.get('images', {})
    calib_data['image_root'] = snapshot.get('image_root')
    calib_data['image_width'] = snapshot.get('image_width')
    calib_data['image_height'] = snapshot.get('image_height')
    calib_data['debug_logging'] = bool(snapshot.get('debug_logging', False))
    calib_data['precision_cleanup_enabled'] = bool(snapshot.get('precision_cleanup_enabled', True))
    calib_data['precision_target_mean_px'] = float(snapshot.get('precision_target_mean_px', 0.5))
    calib_data['precision_target_p95_px'] = float(snapshot.get('precision_target_p95_px', 1.0))
    calib_data['precision_target_max_px'] = float(snapshot.get('precision_target_max_px', 1.0))
    calib_data['precision_cleanup_rounds'] = int(snapshot.get('precision_cleanup_rounds', 4))
    calib_data['strict_track_consistency'] = bool(snapshot.get('strict_track_consistency', True))
    calib_data['preserve_unreconstructed_annotations'] = bool(snapshot.get('preserve_unreconstructed_annotations', True))
    calib_data['pose_scaffold_enabled'] = bool(snapshot.get('pose_scaffold_enabled', True))
    calib_data['pose_scaffold_min_track_length'] = int(snapshot.get('pose_scaffold_min_track_length', 3))
    calib_data['pose_scaffold_nonmember_confidence'] = float(snapshot.get('pose_scaffold_nonmember_confidence', 0.22))
    calib_data['two_stage_pose_scaffold_recovery'] = bool(snapshot.get('two_stage_pose_scaffold_recovery', False))
    calib_data['two_stage_pose_scaffold_min_points'] = int(snapshot.get('two_stage_pose_scaffold_min_points', 12))
    calib_data['two_stage_pose_scaffold_target_ratio'] = float(snapshot.get('two_stage_pose_scaffold_target_ratio', 0.90))
    calib_data['two_stage_pose_scaffold_min_camera_support'] = int(snapshot.get('two_stage_pose_scaffold_min_camera_support', 6))
    calib_data['point_drift_trace_enabled'] = bool(snapshot.get('point_drift_trace_enabled', False))
    calib_data['point_drift_trace_capture_all'] = bool(snapshot.get('point_drift_trace_capture_all', True))
    calib_data['point_drift_trace_max_points'] = int(snapshot.get('point_drift_trace_max_points', 0) or 0)
    if 'initial_focal_estimate' in snapshot:
        calib_data['initial_focal_estimate'] = snapshot.get('initial_focal_estimate')
    for key in [item for item in list(calib_data.keys()) if isinstance(item, str) and item.startswith('K_')]:
        del calib_data[key]
    for key, value in snapshot.get('individual_intrinsics', {}).items():
        calib_data[key] = np.array(value, copy=True)


def _replace_calibration_result(target_calib_data, source_calib_data):
    target_calib_data.clear()
    target_calib_data.update(copy.deepcopy(source_calib_data))


def _copy_individual_intrinsics(target_calib_data, source_calib_data):
    for key in [item for item in list(target_calib_data.keys()) if isinstance(item, str) and item.startswith('K_')]:
        del target_calib_data[key]
    for key, value in source_calib_data.items():
        if isinstance(key, str) and key.startswith('K_') and value is not None:
            target_calib_data[key] = np.array(value, copy=True)


def _serialize_trace_scalar(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _reset_point_drift_trace(calib_data):
    if not bool(calib_data.get('point_drift_trace_enabled', False)):
        return None

    trace_store = {
        'version': 1,
        'started_at': float(time.time()),
        'snapshots': [],
        'image_width': calib_data.get('image_width'),
        'image_height': calib_data.get('image_height'),
        'capture_all_points': bool(calib_data.get('point_drift_trace_capture_all', True)),
        'max_points': int(calib_data.get('point_drift_trace_max_points', 0) or 0),
    }
    calib_data['_point_drift_trace'] = trace_store
    return trace_store


def _select_point_drift_trace_point_ids(calib_data):
    focus_point_ids = calib_data.get('_point_drift_trace_focus_point_ids')
    if focus_point_ids:
        return sorted(
            {
                point_id
                for point_id in focus_point_ids
                if point_id is not None
            },
            key=_stable_sort_key,
        )

    point_ids = set()
    for point_map in (
        calib_data.get('points_3d', {}),
        calib_data.get('secondary_points_3d', {}),
    ):
        point_ids.update(point_map.keys())
    for camera_map in (calib_data.get('camera_points', {}), calib_data.get('raw_camera_points', {})):
        for observations in camera_map.values():
            point_ids.update(dict(observations or {}).keys())

    point_ids = sorted(point_ids, key=_stable_sort_key)
    if bool(calib_data.get('point_drift_trace_capture_all', True)):
        return point_ids

    max_points = int(calib_data.get('point_drift_trace_max_points', 0) or 0)
    if max_points > 0:
        return point_ids[:max_points]
    return point_ids


def _capture_point_drift_stage(calib_data, stage_name, info=None):
    if not bool(calib_data.get('point_drift_trace_enabled', False)):
        return None
    if bool(calib_data.get('_project_level_preview_mode', False)):
        return None

    trace_store = calib_data.get('_point_drift_trace')
    if not isinstance(trace_store, dict):
        trace_store = _reset_point_drift_trace(calib_data) or {}
        calib_data['_point_drift_trace'] = trace_store
    snapshots = trace_store.setdefault('snapshots', [])
    point_ids = _select_point_drift_trace_point_ids(calib_data)
    cameras = calib_data.get('cameras', {}) or {}
    camera_points = calib_data.get('camera_points', {}) or {}
    raw_camera_points = calib_data.get('raw_camera_points', {}) or {}
    primary_points = calib_data.get('points_3d', {}) or {}
    secondary_points = calib_data.get('secondary_points_3d', {}) or {}
    seed_points = calib_data.get('_secondary_seed_points_3d', {}) or {}

    if cameras and primary_points:
        stats = summarize_reprojection_error_distribution(calib_data, top_k=0)
    else:
        stats = {
            'count': 0,
            'mean': 0.0,
            'median': 0.0,
            'p90': 0.0,
            'p95': 0.0,
            'max': 0.0,
        }

    snapshot = {
        'stage': str(stage_name),
        'stage_index': int(len(snapshots)),
        'elapsed_sec': float(time.time() - float(trace_store.get('started_at', time.time()))),
        'camera_count': int(len(cameras)),
        'primary_point_count': int(len(primary_points)),
        'secondary_point_count': int(len(secondary_points)),
        'seed_point_count': int(len(seed_points)),
        'observation_count': int(sum(len(dict(item or {})) for item in camera_points.values())),
        'raw_observation_count': int(sum(len(dict(item or {})) for item in raw_camera_points.values())),
        'mean_error': float(stats.get('mean', 0.0)),
        'p95_error': float(stats.get('p95', 0.0)),
        'max_error': float(stats.get('max', 0.0)),
        'info': copy.deepcopy(info) if isinstance(info, dict) else ({'label': str(info)} if info is not None else {}),
        'points': {},
    }

    for point_id in point_ids:
        point_key = str(_serialize_trace_scalar(point_id))
        point_3d = None
        point_source = None
        if point_id in primary_points:
            point_3d = np.asarray(primary_points[point_id], dtype=np.float64).reshape(3)
            point_source = 'primary'
        elif point_id in secondary_points:
            point_3d = np.asarray(secondary_points[point_id], dtype=np.float64).reshape(3)
            point_source = 'secondary'
        elif point_id in seed_points:
            point_3d = np.asarray(seed_points[point_id], dtype=np.float64).reshape(3)
            point_source = 'seed'

        camera_ids = set()
        for camera_id, observations in camera_points.items():
            if point_id in dict(observations or {}):
                camera_ids.add(str(camera_id))
        for camera_id, observations in raw_camera_points.items():
            if point_id in dict(observations or {}):
                camera_ids.add(str(camera_id))

        point_entry = {
            'point_id': _serialize_trace_scalar(point_id),
            'point_source': point_source,
            'in_primary_points_3d': bool(point_id in primary_points),
            'in_secondary_points_3d': bool(point_id in secondary_points),
            'in_seed_points_3d': bool(point_id in seed_points),
            'point_3d': (
                [float(point_3d[0]), float(point_3d[1]), float(point_3d[2])]
                if point_3d is not None else None
            ),
            'observations': [],
        }

        for camera_id in sorted(camera_ids, key=_stable_sort_key):
            camera_key = str(camera_id)
            raw_obs = dict(raw_camera_points.get(camera_key, {}) or {}).get(point_id)
            kept_obs = dict(camera_points.get(camera_key, {}) or {}).get(point_id)
            observation_entry = {
                'camera_id': camera_key,
                'image_name': os.path.basename(str(calib_data.get('images', {}).get(camera_key, camera_key))),
                'status': 'kept' if kept_obs is not None else 'removed',
                'observation_confidence': float(_lookup_observation_confidence(calib_data, camera_key, point_id)),
            }
            if raw_obs is not None:
                raw_obs = np.asarray(raw_obs, dtype=np.float64).reshape(2)
                observation_entry['observed_raw'] = [float(raw_obs[0]), float(raw_obs[1])]
            if kept_obs is not None:
                kept_obs = np.asarray(kept_obs, dtype=np.float64).reshape(2)
                observation_entry['observed_kept'] = [float(kept_obs[0]), float(kept_obs[1])]

            if point_3d is not None and camera_key in cameras:
                camera_K = _get_camera_matrix(calib_data, camera_key)
                if camera_K is not None:
                    R, t = cameras[camera_key]
                    projected = utils.project_point(
                        point_3d,
                        R,
                        t,
                        camera_K,
                        calib_data.get('dist_coeffs'),
                    )
                    projected = np.asarray(projected, dtype=np.float64).reshape(2)
                    observation_entry['projected'] = [float(projected[0]), float(projected[1])]

                    if raw_obs is not None:
                        observation_entry['error_to_raw'] = float(np.linalg.norm(projected - raw_obs))
                    if kept_obs is not None:
                        observation_entry['error_to_kept'] = float(np.linalg.norm(projected - kept_obs))
                    compare_obs = raw_obs if raw_obs is not None else kept_obs
                    if compare_obs is not None:
                        observation_entry['error'] = float(np.linalg.norm(projected - compare_obs))

            point_entry['observations'].append(observation_entry)

        snapshot['points'][point_key] = point_entry

    snapshots.append(snapshot)
    return snapshot


def _filter_camera_point_map_for_point_ids(point_map, allowed_point_ids, keep_empty_cameras=True):
    allowed_point_ids = set(allowed_point_ids or [])
    filtered_map = {}
    for camera_id, observations in (point_map or {}).items():
        camera_key = str(camera_id)
        filtered_observations = {
            point_id: copy.deepcopy(value)
            for point_id, value in dict(observations or {}).items()
            if point_id in allowed_point_ids
        }
        if filtered_observations or keep_empty_cameras:
            filtered_map[camera_key] = filtered_observations
    return filtered_map


def _filter_line_support_data_for_point_ids(line_support_data, allowed_point_ids):
    if not isinstance(line_support_data, dict):
        return copy.deepcopy(line_support_data)

    allowed_point_ids = set(allowed_point_ids or [])
    allowed_point_ids_str = {str(point_id) for point_id in allowed_point_ids}

    def _point_allowed(point_id):
        return point_id in allowed_point_ids or str(point_id) in allowed_point_ids_str

    filtered_data = {}
    point_triplets = line_support_data.get('point_triplets')
    referenced_triplets = set()
    if isinstance(point_triplets, dict):
        filtered_point_triplets = {}
        for point_id, triplet_refs in point_triplets.items():
            if not _point_allowed(point_id):
                continue
            filtered_point_triplets[point_id] = copy.deepcopy(triplet_refs)
            if isinstance(triplet_refs, (list, tuple, set)):
                referenced_triplets.update(str(item) for item in triplet_refs)
            elif triplet_refs is not None:
                referenced_triplets.add(str(triplet_refs))
        if filtered_point_triplets:
            filtered_data['point_triplets'] = filtered_point_triplets

    triplets = line_support_data.get('triplets')
    if isinstance(triplets, dict):
        filtered_triplets = {}
        for triplet_id, triplet_meta in triplets.items():
            keep_triplet = str(triplet_id) in referenced_triplets
            if not keep_triplet and isinstance(triplet_meta, dict):
                candidate_point_ids = (
                    triplet_meta.get('point_ids') or
                    triplet_meta.get('points') or
                    triplet_meta.get('group_ids')
                )
                if isinstance(candidate_point_ids, (list, tuple, set)):
                    keep_triplet = all(_point_allowed(point_id) for point_id in candidate_point_ids)
            if keep_triplet:
                filtered_triplets[triplet_id] = copy.deepcopy(triplet_meta)
        if filtered_triplets:
            filtered_data['triplets'] = filtered_triplets

    for key, value in line_support_data.items():
        if key in {'point_triplets', 'triplets'}:
            continue
        filtered_data[key] = copy.deepcopy(value)

    return filtered_data


def _build_point_filtered_calibration_view(source_calib_data, allowed_point_ids):
    filtered_calib_data = copy.deepcopy(source_calib_data)
    allowed_point_ids = set(allowed_point_ids or [])

    filtered_calib_data['cameras'] = {}
    filtered_calib_data['points_3d'] = {
        point_id: copy.deepcopy(point_3d)
        for point_id, point_3d in (source_calib_data.get('points_3d') or {}).items()
        if point_id in allowed_point_ids
    }
    filtered_calib_data['secondary_points_3d'] = {}
    filtered_calib_data['_secondary_seed_points_3d'] = {}
    filtered_calib_data['reconstruction_ids'] = []
    filtered_calib_data['camera_points'] = _filter_camera_point_map_for_point_ids(
        source_calib_data.get('camera_points'),
        allowed_point_ids,
        keep_empty_cameras=True,
    )
    filtered_calib_data['raw_camera_points'] = _filter_camera_point_map_for_point_ids(
        source_calib_data.get('raw_camera_points'),
        allowed_point_ids,
        keep_empty_cameras=True,
    )
    filtered_calib_data['observation_confidences'] = _filter_camera_point_map_for_point_ids(
        source_calib_data.get('observation_confidences'),
        allowed_point_ids,
        keep_empty_cameras=False,
    )
    filtered_calib_data['line_support_data'] = _filter_line_support_data_for_point_ids(
        source_calib_data.get('line_support_data'),
        allowed_point_ids,
    )
    return filtered_calib_data


def _summarize_reconstruction_metrics(calib_data):
    mean_error, _, _ = calculate_reprojection_errors(calib_data)
    reprojection_stats = summarize_reprojection_error_distribution(calib_data, top_k=0)
    primary_points = int(len(calib_data.get('points_3d', {}) or {}))
    secondary_points = int(len(calib_data.get('secondary_points_3d', {}) or {}))
    return {
        'camera_count': int(len(calib_data.get('cameras', {}) or {})),
        'primary_points': primary_points,
        'secondary_points': secondary_points,
        'total_points': int(primary_points + secondary_points),
        'mean_error': float(mean_error),
        'p95_error': float(reprojection_stats.get('p95', mean_error)),
        'max_error': float(reprojection_stats.get('max', mean_error)),
    }


def _should_accept_two_stage_pose_scaffold_candidate(baseline_metrics, candidate_metrics):
    baseline_cameras = int(baseline_metrics.get('camera_count', 0))
    candidate_cameras = int(candidate_metrics.get('camera_count', 0))
    if candidate_cameras < baseline_cameras:
        return False, 'camera_count_regression'

    baseline_total = int(baseline_metrics.get('total_points', 0))
    candidate_total = int(candidate_metrics.get('total_points', 0))
    if candidate_total < baseline_total:
        return False, 'point_count_regression'

    baseline_mean = float(baseline_metrics.get('mean_error', 0.0))
    baseline_p95 = float(baseline_metrics.get('p95_error', baseline_mean))
    baseline_max = float(baseline_metrics.get('max_error', baseline_mean))
    candidate_mean = float(candidate_metrics.get('mean_error', 0.0))
    candidate_p95 = float(candidate_metrics.get('p95_error', candidate_mean))
    candidate_max = float(candidate_metrics.get('max_error', candidate_mean))

    mean_limit = max(baseline_mean + 0.08, baseline_mean * 1.12, 0.85)
    p95_limit = max(baseline_p95 + 0.25, baseline_p95 * 1.15, 2.40)
    max_limit = max(baseline_max + 0.80, baseline_max * 1.18, 6.00)
    if candidate_mean > mean_limit or candidate_p95 > p95_limit or candidate_max > max_limit:
        return False, 'error_regression'

    if candidate_total > baseline_total:
        return True, 'more_points'
    if candidate_cameras > baseline_cameras:
        return True, 'more_cameras'

    stable_or_better = (
        candidate_mean <= baseline_mean + 0.03 and
        candidate_p95 <= baseline_p95 + 0.08 and
        candidate_max <= baseline_max + 0.20
    )
    if stable_or_better:
        return True, 'stable_tie'
    return False, 'no_gain'


def _get_camera_matrix(calib_data, camera_id):
    camera_id = str(camera_id)
    individual_K = calib_data.get(f'K_{camera_id}')
    if individual_K is not None:
        return individual_K
    return calib_data.get('K')


def _collect_camera_intrinsics_map(calib_data):
    return {
        str(key)[2:]: np.asarray(value, dtype=np.float64)
        for key, value in calib_data.items()
        if isinstance(key, str) and key.startswith('K_') and value is not None
    }


def _lookup_observation_confidence(calib_data, camera_id, point_id, default=1.0):
    observation_confidences = calib_data.get('observation_confidences') or {}
    try:
        camera_confidences = observation_confidences.get(str(camera_id), {})
        confidence = float(camera_confidences.get(point_id, default))
    except (AttributeError, TypeError, ValueError):
        confidence = float(default)
    return float(np.clip(confidence, 0.15, 1.0))


def _set_observation_confidence(calib_data, camera_id, point_id, confidence, mode="replace"):
    observation_confidences = calib_data.setdefault('observation_confidences', {})
    camera_confidences = observation_confidences.setdefault(str(camera_id), {})
    current_value = camera_confidences.get(point_id, 1.0)
    try:
        current_value = float(current_value)
    except (TypeError, ValueError):
        current_value = 1.0
    try:
        new_value = float(confidence)
    except (TypeError, ValueError):
        new_value = current_value
    if str(mode or "replace") == "min":
        new_value = min(current_value, new_value)
    camera_confidences[point_id] = float(np.clip(new_value, 0.15, 1.0))
    return camera_confidences[point_id]


def _get_soft_gate_min_confidence(calib_data):
    base_confidence = float(calib_data.get('pose_scaffold_nonmember_confidence', 0.22))
    return float(np.clip(max(base_confidence + 0.06, 0.28), 0.15, 0.60))


def _compute_track2d_projection_candidate(
    point_id,
    target_camera_id,
    cameras,
    camera_points,
    common_K,
    target_K,
    dist_coeffs=None,
    camera_intrinsics=None,
):
    target_camera_id = str(target_camera_id)
    if common_K is None or target_camera_id not in cameras:
        return None

    point_observations = {}
    for camera_id, points_map in camera_points.items():
        camera_id_str = str(camera_id)
        if camera_id_str not in cameras or point_id not in points_map:
            continue
        point_observations[camera_id_str] = np.asarray(points_map[point_id], dtype=np.float64).reshape(2)

    support_camera_ids = sorted(
        [camera_id for camera_id in point_observations.keys() if camera_id != target_camera_id],
        key=_stable_sort_key,
    )
    if len(support_camera_ids) < 2:
        return None

    common_K = np.asarray(common_K, dtype=np.float64).reshape(3, 3)
    target_K = np.asarray(target_K, dtype=np.float64).reshape(3, 3)
    camera_intrinsics = camera_intrinsics or {}
    camera_centers = {
        camera_id: triangulation._camera_center_from_pose(*cameras[camera_id])
        for camera_id in support_camera_ids
        if camera_id in cameras
    }

    best_candidate = None
    for index_a in range(len(support_camera_ids) - 1):
        for index_b in range(index_a + 1, len(support_camera_ids)):
            camera_a_id = str(support_camera_ids[index_a])
            camera_b_id = str(support_camera_ids[index_b])
            candidate_point = triangulation._triangulate_global_point_from_pair(
                camera_a_id,
                camera_b_id,
                point_observations,
                cameras,
                common_K,
                dist_coeffs=dist_coeffs,
                debug_logging=False,
            )
            if candidate_point is None:
                continue

            support_errors = []
            for support_camera_id in support_camera_ids:
                support_camera_id = str(support_camera_id)
                support_K = np.asarray(
                    camera_intrinsics.get(support_camera_id, common_K),
                    dtype=np.float64,
                ).reshape(3, 3)
                projected_support = utils.project_point(
                    candidate_point,
                    cameras[support_camera_id][0],
                    cameras[support_camera_id][1],
                    support_K,
                    dist_coeffs,
                )
                if projected_support is None:
                    continue
                projected_support = np.asarray(projected_support, dtype=np.float64).reshape(2)
                if not np.all(np.isfinite(projected_support)):
                    continue
                support_errors.append(
                    float(np.linalg.norm(projected_support - point_observations[support_camera_id]))
                )
            if not support_errors:
                continue

            projected_target = utils.project_point(
                candidate_point,
                cameras[target_camera_id][0],
                cameras[target_camera_id][1],
                target_K,
                dist_coeffs,
            )
            if projected_target is None:
                continue
            projected_target = np.asarray(projected_target, dtype=np.float64).reshape(2)
            if not np.all(np.isfinite(projected_target)):
                continue

            baseline = None
            center_a = camera_centers.get(camera_a_id)
            center_b = camera_centers.get(camera_b_id)
            if center_a is not None and center_b is not None:
                baseline = float(np.linalg.norm(np.asarray(center_b) - np.asarray(center_a)))

            candidate = {
                'projected': projected_target,
                'support_mean_error': float(np.mean(support_errors)),
                'support_max_error': float(np.max(support_errors)),
                'support_pair': [camera_a_id, camera_b_id],
                'support_camera_ids': [str(camera_id) for camera_id in support_camera_ids],
                'pair_baseline': baseline,
            }
            candidate_score = (
                candidate['support_mean_error'],
                candidate['support_max_error'],
                0.0 if baseline is None else -baseline,
            )
            if best_candidate is None or candidate_score < best_candidate['score']:
                best_candidate = {
                    'score': candidate_score,
                    **candidate,
                }

    if best_candidate is None:
        return None

    best_candidate.pop('score', None)
    return best_candidate


def _compute_track2d_observation_modifiers(calib_data):
    cameras = calib_data.get('cameras') or {}
    points_3d = calib_data.get('points_3d') or {}
    camera_points = calib_data.get('camera_points') or {}
    common_K = calib_data.get('K')
    if common_K is None or len(cameras) < 3 or not points_3d:
        return {}, {
            'count': 0,
            'evaluated': 0,
            'downweighted': 0,
            'median_delta': 0.0,
            'p90_delta': 0.0,
            'max_delta': 0.0,
            'reason': 'insufficient_scene',
        }

    dist_coeffs = calib_data.get('dist_coeffs')
    camera_intrinsics = _collect_camera_intrinsics_map(calib_data)
    candidate_entries = []
    deltas = []

    for camera_id in sorted(cameras.keys(), key=_stable_sort_key):
        camera_id_str = str(camera_id)
        observations = camera_points.get(camera_id_str, {})
        if not observations:
            continue
        target_K = np.asarray(
            calib_data.get(f'K_{camera_id_str}', common_K),
            dtype=np.float64,
        ).reshape(3, 3)

        for point_id, _ in observations.items():
            if point_id not in points_3d:
                continue

            current_projected = utils.project_point(
                points_3d[point_id],
                cameras[camera_id_str][0],
                cameras[camera_id_str][1],
                target_K,
                dist_coeffs,
            )
            if current_projected is None:
                continue
            current_projected = np.asarray(current_projected, dtype=np.float64).reshape(2)
            if not np.all(np.isfinite(current_projected)):
                continue

            track2d_projection = _compute_track2d_projection_candidate(
                point_id=point_id,
                target_camera_id=camera_id_str,
                cameras=cameras,
                camera_points=camera_points,
                common_K=common_K,
                target_K=target_K,
                dist_coeffs=dist_coeffs,
                camera_intrinsics=camera_intrinsics,
            )
            if track2d_projection is None:
                continue

            track2d_projected = np.asarray(track2d_projection['projected'], dtype=np.float64).reshape(2)
            delta = float(np.linalg.norm(track2d_projected - current_projected))
            support_mean_error = float(track2d_projection.get('support_mean_error', 0.0))
            support_max_error = float(track2d_projection.get('support_max_error', 0.0))

            deltas.append(delta)
            candidate_entries.append(
                {
                    'camera_id': camera_id_str,
                    'point_id': point_id,
                    'delta': delta,
                    'support_mean_error': support_mean_error,
                    'support_max_error': support_max_error,
                }
            )

    if deltas:
        deltas_array = np.asarray(deltas, dtype=np.float64)
        median_delta = float(np.median(deltas_array))
        p90_delta = float(np.percentile(deltas_array, 90))
        max_delta = float(np.max(deltas_array))
    else:
        median_delta = 0.0
        p90_delta = 0.0
        max_delta = 0.0

    relaxed_scene = bool(median_delta > 1.8 or p90_delta > 6.5)
    modifiers = {}
    downweighted = 0
    for item in candidate_entries:
        delta = float(item['delta'])
        support_mean_error = float(item['support_mean_error'])
        support_max_error = float(item['support_max_error'])
        modifier = 1.0

        if relaxed_scene:
            if delta >= 5.5 and support_mean_error <= 2.5 and support_max_error <= 7.5:
                modifier = 0.52
            elif delta >= 3.5 and support_mean_error <= 1.5 and support_max_error <= 4.5:
                modifier = 0.72
        else:
            if delta >= 4.0 and support_mean_error <= 2.5 and support_max_error <= 7.5:
                modifier = 0.42
            elif delta >= 2.5 and support_mean_error <= 1.8 and support_max_error <= 5.5:
                modifier = 0.60
            elif delta >= 1.5 and support_mean_error <= 1.1 and support_max_error <= 3.0:
                modifier = 0.82

        if modifier < 0.999:
            modifiers.setdefault(item['camera_id'], {})[item['point_id']] = float(np.clip(modifier, 0.18, 1.0))
            downweighted += 1

    return modifiers, {
        'count': int(sum(len(items) for items in camera_points.values())),
        'evaluated': int(len(candidate_entries)),
        'downweighted': int(downweighted),
        'median_delta': median_delta,
        'p90_delta': p90_delta,
        'max_delta': max_delta,
        'relaxed_scene': bool(relaxed_scene),
    }


def _compute_camera_bias_observation_modifiers(calib_data):
    cameras = calib_data.get('cameras') or {}
    points_3d = calib_data.get('points_3d') or {}
    camera_points = calib_data.get('camera_points') or {}
    common_K = calib_data.get('K')
    if common_K is None or not cameras or not points_3d:
        return {}, {
            'camera_count': 0,
            'downweighted': 0,
            'max_bias': 0.0,
            'reason': 'insufficient_scene',
        }

    dist_coeffs = calib_data.get('dist_coeffs')
    modifiers = {}
    biased_cameras = 0
    downweighted = 0
    max_bias = 0.0

    for camera_id in sorted(cameras.keys(), key=_stable_sort_key):
        camera_id_str = str(camera_id)
        observations = camera_points.get(camera_id_str, {})
        if len(observations) < 8:
            continue

        K_camera = np.asarray(
            calib_data.get(f'K_{camera_id_str}', common_K),
            dtype=np.float64,
        ).reshape(3, 3)
        residual_entries = []
        for point_id, observed_xy in observations.items():
            if point_id not in points_3d:
                continue
            projected_xy = utils.project_point(
                points_3d[point_id],
                cameras[camera_id_str][0],
                cameras[camera_id_str][1],
                K_camera,
                dist_coeffs,
            )
            if projected_xy is None:
                continue
            projected_xy = np.asarray(projected_xy, dtype=np.float64).reshape(2)
            observed_xy = np.asarray(observed_xy, dtype=np.float64).reshape(2)
            if not (np.all(np.isfinite(projected_xy)) and np.all(np.isfinite(observed_xy))):
                continue
            residual_entries.append((point_id, projected_xy - observed_xy))

        if len(residual_entries) < 8:
            continue

        residual_vectors = np.asarray([entry[1] for entry in residual_entries], dtype=np.float64)
        median_bias = np.median(residual_vectors, axis=0)
        bias_norm = float(np.linalg.norm(median_bias))
        max_bias = max(max_bias, bias_norm)
        if bias_norm < 2.25:
            continue

        biased_cameras += 1
        bias_direction = median_bias / max(bias_norm, 1e-6)
        if bias_norm >= 6.0:
            camera_factor = 0.62
        elif bias_norm >= 4.0:
            camera_factor = 0.74
        else:
            camera_factor = 0.84

        for point_id, residual in residual_entries:
            residual = np.asarray(residual, dtype=np.float64).reshape(2)
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm < max(1.0, bias_norm * 0.45):
                continue
            alignment = float(np.dot(residual, bias_direction) / max(residual_norm, 1e-6))
            if alignment < 0.35:
                continue
            point_factor = camera_factor
            if residual_norm >= bias_norm * 1.25:
                point_factor *= 0.88
            modifiers.setdefault(camera_id_str, {})[point_id] = float(
                min(
                    modifiers.get(camera_id_str, {}).get(point_id, 1.0),
                    np.clip(point_factor, 0.18, 1.0),
                )
            )
            downweighted += 1

    return modifiers, {
        'camera_count': int(biased_cameras),
        'downweighted': int(downweighted),
        'max_bias': float(max_bias),
    }


def _merge_observation_confidence_maps(base_confidences, modifier_maps):
    merged = {}
    if base_confidences:
        for camera_id, camera_confidences in base_confidences.items():
            merged[str(camera_id)] = {
                point_id: float(np.clip(value, 0.15, 1.0))
                for point_id, value in dict(camera_confidences).items()
            }

    for modifier_map in modifier_maps:
        if not modifier_map:
            continue
        for camera_id, point_modifiers in modifier_map.items():
            camera_key = str(camera_id)
            merged_camera = merged.setdefault(camera_key, {})
            for point_id, modifier in point_modifiers.items():
                base_value = float(np.clip(merged_camera.get(point_id, 1.0), 0.15, 1.0))
                merged_camera[point_id] = float(np.clip(base_value * float(modifier), 0.15, 1.0))

    return merged


def _build_quality_aware_observation_confidences(calib_data, base_confidences=None, *, label=None):
    base_confidences = base_confidences if base_confidences is not None else (calib_data.get('observation_confidences') or {})
    track2d_modifiers, track2d_stats = _compute_track2d_observation_modifiers(calib_data)
    bias_modifiers, bias_stats = _compute_camera_bias_observation_modifiers(calib_data)
    merged_confidences = _merge_observation_confidence_maps(
        base_confidences,
        [track2d_modifiers, bias_modifiers],
    )
    calib_data['_observation_quality_modifiers'] = {
        'track2d': track2d_modifiers,
        'camera_bias': bias_modifiers,
    }
    calib_data['_observation_quality_stats'] = {
        'track2d': track2d_stats,
        'camera_bias': bias_stats,
    }

    if track2d_stats.get('downweighted', 0) or bias_stats.get('downweighted', 0):
        label_prefix = f"{label}: " if label else ""
        track2d_mode = "relaxed" if track2d_stats.get('relaxed_scene') else "strict"
        print(
            f"{label_prefix}quality weighting: "
            f"track2d {track2d_stats.get('downweighted', 0)}/{track2d_stats.get('evaluated', 0)} "
            f"(mode={track2d_mode}, median={track2d_stats.get('median_delta', 0.0):.2f}px, "
            f"p90={track2d_stats.get('p90_delta', 0.0):.2f}px, "
            f"max={track2d_stats.get('max_delta', 0.0):.2f}px), "
            f"camera_bias {bias_stats.get('downweighted', 0)} obs / {bias_stats.get('camera_count', 0)} cams "
            f"(max_bias={bias_stats.get('max_bias', 0.0):.2f}px)"
        )

    return merged_confidences


def _select_camera_pose_support_point_ids(
    calib_data,
    camera_id,
    common_point_ids=None,
    min_points_for_camera=4,
    min_observation_confidence=None,
):
    camera_id = str(camera_id)
    camera_points_2d = calib_data.get('camera_points', {}).get(camera_id, {})
    if not camera_points_2d:
        return [], {
            'selected_count': 0,
            'strong_count': 0,
            'reserve_count': 0,
            'min_confidence': float(min_observation_confidence or 0.0),
        }

    if common_point_ids is None:
        common_point_ids = [
            point_id
            for point_id in sorted(calib_data.get('points_3d', {}).keys(), key=_stable_sort_key)
            if point_id in camera_points_2d
        ]
    else:
        common_point_ids = [point_id for point_id in common_point_ids if point_id in camera_points_2d]

    if not common_point_ids:
        return [], {
            'selected_count': 0,
            'strong_count': 0,
            'reserve_count': 0,
            'min_confidence': float(min_observation_confidence or 0.0),
        }

    if min_observation_confidence is None:
        min_observation_confidence = _get_soft_gate_min_confidence(calib_data)
    min_observation_confidence = float(np.clip(min_observation_confidence, 0.15, 1.0))

    camera_points = calib_data.get('camera_points', {})
    reconstructed_camera_ids = [str(other_camera_id) for other_camera_id in calib_data.get('cameras', {}).keys()]
    strong_candidates = []
    reserve_candidates = []

    for point_id in common_point_ids:
        target_confidence = _lookup_observation_confidence(calib_data, camera_id, point_id)
        support_camera_count = 0
        support_confidences = []
        total_track_length = 0
        for other_camera_id, observations in camera_points.items():
            other_camera_id = str(other_camera_id)
            if point_id not in observations:
                continue
            total_track_length += 1
            if other_camera_id in reconstructed_camera_ids and other_camera_id != camera_id:
                support_camera_count += 1
                support_confidences.append(_lookup_observation_confidence(calib_data, other_camera_id, point_id))

        mean_support_confidence = float(np.mean(support_confidences)) if support_confidences else 1.0
        score = (
            support_camera_count * 3.5 +
            total_track_length * 0.9 +
            mean_support_confidence * 2.2 +
            target_confidence * 3.0
        )
        entry = {
            'point_id': point_id,
            'score': float(score),
            'target_confidence': float(target_confidence),
            'support_confidence': float(mean_support_confidence),
            'support_camera_count': int(support_camera_count),
            'track_length': int(total_track_length),
        }
        is_strong = (
            target_confidence >= min_observation_confidence and
            mean_support_confidence >= max(0.25, min_observation_confidence * 0.85) and
            support_camera_count >= max(2, min_points_for_camera - 1)
        )
        if is_strong:
            strong_candidates.append(entry)
        else:
            reserve_candidates.append(entry)

    ranked_strong = sorted(
        strong_candidates,
        key=lambda item: (
            item['score'],
            item['support_camera_count'],
            item['track_length'],
            item['target_confidence'],
        ),
        reverse=True,
    )
    ranked_reserve = sorted(
        reserve_candidates,
        key=lambda item: (
            item['target_confidence'],
            item['support_confidence'],
            item['support_camera_count'],
            item['track_length'],
            item['score'],
        ),
        reverse=True,
    )

    target_support_count = max(8, int(min_points_for_camera) + 4)
    support_cap = max(
        target_support_count,
        min(24, int(np.ceil(max(len(common_point_ids), 1) * 0.70)))
    )

    selected_point_ids = []
    for entry in ranked_strong:
        if len(selected_point_ids) >= support_cap:
            break
        selected_point_ids.append(entry['point_id'])

    if len(selected_point_ids) < target_support_count:
        for entry in ranked_reserve:
            if entry['point_id'] in selected_point_ids:
                continue
            selected_point_ids.append(entry['point_id'])
            if len(selected_point_ids) >= target_support_count:
                break

    if len(selected_point_ids) < int(min_points_for_camera):
        fallback_ranked = sorted(
            strong_candidates + reserve_candidates,
            key=lambda item: (
                item['score'],
                item['support_camera_count'],
                item['track_length'],
            ),
            reverse=True,
        )
        for entry in fallback_ranked:
            if entry['point_id'] in selected_point_ids:
                continue
            selected_point_ids.append(entry['point_id'])
            if len(selected_point_ids) >= int(min_points_for_camera):
                break

    return selected_point_ids, {
        'selected_count': int(len(selected_point_ids)),
        'strong_count': int(len(strong_candidates)),
        'reserve_count': int(len(reserve_candidates)),
        'min_confidence': float(min_observation_confidence),
    }


def _purge_missing_point_observations(calib_data):
    if bool(calib_data.get('preserve_unreconstructed_annotations', True)):
        return 0
    if not calib_data.get('camera_points'):
        return 0

    valid_point_ids = set(calib_data.get('points_3d', {}).keys())
    removed = 0
    for camera_id, observations in calib_data['camera_points'].items():
        for point_id in list(observations.keys()):
            if point_id not in valid_point_ids:
                del observations[point_id]
                removed += 1
    return removed

def add_camera_parameters(calib_data, camera_id, K, dist_coeffs=None):
    """
    Добавляет или обновляет параметры камеры.
    
    Args:
        calib_data: Данные калибровки
        camera_id: ID камеры
        K: Матрица внутренних параметров камеры
        dist_coeffs: Коэффициенты дисторсии (опционально)
    """
    camera_id = str(camera_id)

    # Сохраняем матрицу внутренних параметров
    calib_data['K'] = K.copy()
    
    # Сохраняем коэффициенты дисторсии, если они предоставлены
    if dist_coeffs is not None:
        calib_data['dist_coeffs'] = dist_coeffs.copy()
    else:
        calib_data['dist_coeffs'] = np.zeros(5, dtype=np.float32)
            
    # Создаем словарь для камеры, если его еще нет
    if camera_id not in calib_data['camera_points']:
        calib_data['camera_points'][camera_id] = {}
    if 'raw_camera_points' not in calib_data:
        calib_data['raw_camera_points'] = {}
    if camera_id not in calib_data['raw_camera_points']:
        calib_data['raw_camera_points'][camera_id] = {}

def add_image_points(calib_data, camera_id, points, image_path=None):
    """
    Добавляет точки для изображения.
    
    Args:
        calib_data: Данные калибровки
        camera_id: ID камеры
        points: Словарь точек {point_id: point_2d}
        image_path: Путь к изображению (опционально)
    """
    camera_id = str(camera_id)

    # Убеждаемся, что камера инициализирована
    if camera_id not in calib_data['camera_points']:
        calib_data['camera_points'][camera_id] = {}
    if 'raw_camera_points' not in calib_data:
        calib_data['raw_camera_points'] = {}
    if camera_id not in calib_data['raw_camera_points']:
        calib_data['raw_camera_points'][camera_id] = {}
    
    # Добавляем точки
    for point_id, point in points.items():
        point_array = np.array(point, dtype=np.float32).reshape(2)
        calib_data['camera_points'][camera_id][point_id] = point_array.copy()
        calib_data['raw_camera_points'][camera_id][point_id] = point_array.copy()
    
    # Сохраняем путь к изображению, если он предоставлен
    if image_path is not None:
        calib_data['images'][camera_id] = image_path

def _as_points_nx3(points_3d):
    points_3d = np.asarray(points_3d, dtype=np.float32)
    if points_3d.ndim == 2 and points_3d.shape[0] == 3 and points_3d.shape[1] != 3:
        points_3d = points_3d.T
    elif points_3d.ndim == 1 and points_3d.size >= 3:
        points_3d = points_3d.reshape(1, -1)
    if points_3d.ndim != 2 or points_3d.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)
    return points_3d[:, :3]


def _evaluate_initial_pair_candidate(calib_data, camera_id1, camera_id2, points1_dict, points2_dict, common_points):
    pts1 = np.array([points1_dict[point_id] for point_id in common_points], dtype=np.float32)
    pts2 = np.array([points2_dict[point_id] for point_id in common_points], dtype=np.float32)

    result = camera_pose.estimate_relative_pose(
        pts1,
        pts2,
        calib_data['K'],
        calib_data.get('dist_coeffs')
    )
    if result is None:
        return None

    R, t, mask, points_3d = result
    points_3d = _as_points_nx3(points_3d)
    if len(points_3d) == 0:
        return None

    if mask is None:
        inlier_indices = np.arange(min(len(common_points), len(points_3d)))
    else:
        inlier_indices = np.where(np.asarray(mask).ravel() > 0)[0]
    if len(inlier_indices) == 0:
        return None

    sample_count = min(len(inlier_indices), len(points_3d))
    if sample_count < 8:
        return None

    points_3d = points_3d[:sample_count]
    inlier_indices = inlier_indices[:sample_count]
    obs1 = pts1[inlier_indices]
    obs2 = pts2[inlier_indices]

    finite_mask = np.all(np.isfinite(points_3d), axis=1)
    points_3d = points_3d[finite_mask]
    obs1 = obs1[finite_mask]
    obs2 = obs2[finite_mask]
    if len(points_3d) < 8:
        return None

    rvec = cv2.Rodrigues(np.asarray(R, dtype=np.float32))[0]
    tvec = np.asarray(t, dtype=np.float32).reshape(3, 1)
    proj1, _ = cv2.projectPoints(
        points_3d,
        np.zeros((3, 1), dtype=np.float32),
        np.zeros((3, 1), dtype=np.float32),
        calib_data['K'],
        calib_data.get('dist_coeffs')
    )
    proj2, _ = cv2.projectPoints(
        points_3d,
        rvec,
        tvec,
        calib_data['K'],
        calib_data.get('dist_coeffs')
    )
    reproj1 = np.linalg.norm(proj1.reshape(-1, 2) - obs1, axis=1)
    reproj2 = np.linalg.norm(proj2.reshape(-1, 2) - obs2, axis=1)
    reproj_errors = np.concatenate([reproj1, reproj2])

    points_cam2 = (np.asarray(R, dtype=np.float32) @ points_3d.T + tvec).T
    front_ratio1 = float(np.mean(points_3d[:, 2] > 0.01))
    front_ratio2 = float(np.mean(points_cam2[:, 2] > 0.01))
    front_ratio = min(front_ratio1, front_ratio2)
    if front_ratio < 0.8:
        return None

    camera_center_2 = (-np.asarray(R, dtype=np.float32).T @ tvec).ravel()
    rays1 = points_3d
    rays2 = points_3d - camera_center_2
    ray1_norm = np.linalg.norm(rays1, axis=1)
    ray2_norm = np.linalg.norm(rays2, axis=1)
    valid_ray_mask = (ray1_norm > 1e-6) & (ray2_norm > 1e-6)
    if np.sum(valid_ray_mask) < 8:
        return None

    cos_angles = np.sum(rays1[valid_ray_mask] * rays2[valid_ray_mask], axis=1) / (
        ray1_norm[valid_ray_mask] * ray2_norm[valid_ray_mask]
    )
    parallax_deg = np.degrees(np.arccos(np.clip(cos_angles, -1.0, 1.0)))
    median_parallax = float(np.median(parallax_deg))

    centered_points = points_3d - np.mean(points_3d, axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered_points, full_matrices=False)
    planarity_ratio = float(singular_values[-1] / singular_values[0]) if singular_values[0] > 1e-6 else 0.0

    point_ids = [common_points[index] for index in inlier_indices]
    point_ids = point_ids[:sample_count]
    point_ids = [point_id for point_id, is_finite in zip(point_ids, finite_mask) if is_finite]

    track_lengths = []
    confidence_values = []
    observation_confidences = calib_data.get('observation_confidences') or {}
    for point_id in point_ids:
        track_length = 0
        point_confidences = []
        for other_camera_id, camera_points in calib_data.get('camera_points', {}).items():
            other_camera_id = str(other_camera_id)
            if point_id not in camera_points:
                continue
            track_length += 1
            point_confidence = observation_confidences.get(other_camera_id, {}).get(point_id, 1.0)
            try:
                point_confidence = float(point_confidence)
            except (TypeError, ValueError):
                point_confidence = 1.0
            point_confidences.append(float(np.clip(point_confidence, 0.15, 1.0)))
        track_lengths.append(track_length)
        if point_confidences:
            confidence_values.append(float(np.mean(point_confidences)))

    if track_lengths:
        track_lengths_array = np.asarray(track_lengths, dtype=np.float64)
        multiview_ratio = float(np.mean(track_lengths_array >= 3))
        anchor_ratio = float(np.mean(track_lengths_array >= 4))
        median_track_length = float(np.median(track_lengths_array))
        track_depth_score = min(max(median_track_length - 2.0, 0.0) / 2.0, 1.0)
    else:
        multiview_ratio = 0.0
        anchor_ratio = 0.0
        median_track_length = 2.0
        track_depth_score = 0.0

    confidence_median = float(np.median(confidence_values)) if confidence_values else 1.0
    track_support_score = (
        0.35 * multiview_ratio +
        0.30 * anchor_ratio +
        0.20 * track_depth_score +
        0.15 * confidence_median
    )

    point_id_to_point = {
        point_id: np.asarray(point_3d, dtype=np.float32).reshape(3)
        for point_id, point_3d in zip(point_ids, points_3d)
    }
    checked_support_cameras = 0
    supported_cameras = 0
    support_camera_scores = []
    support_medians = []
    support_p95s = []
    support_inliers = []
    support_error_threshold = max(3.0, float(np.median(reproj_errors)) * 2.0 + 0.75)
    for other_camera_id in sorted(calib_data.get('camera_points', {}).keys(), key=_stable_sort_key):
        other_camera_id = str(other_camera_id)
        if other_camera_id in {str(camera_id1), str(camera_id2)}:
            continue
        other_points = calib_data['camera_points'].get(other_camera_id, {})
        overlap_ids = [point_id for point_id in point_ids if point_id in other_points]
        if len(overlap_ids) < 4:
            continue

        checked_support_cameras += 1
        pts3d = np.asarray([point_id_to_point[point_id] for point_id in overlap_ids], dtype=np.float32)
        pts2d = np.asarray([other_points[point_id] for point_id in overlap_ids], dtype=np.float32)

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d,
                pts2d,
                calib_data['K'],
                calib_data.get('dist_coeffs'),
                flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=max(3.5, float(np.median(reproj_errors)) * 2.5 + 0.75),
                confidence=0.995,
                iterationsCount=300,
            )
        except cv2.error:
            success = False
            rvec = None
            tvec = None
            inliers = None

        if not success:
            try:
                success, rvec, tvec = cv2.solvePnP(
                    pts3d,
                    pts2d,
                    calib_data['K'],
                    calib_data.get('dist_coeffs'),
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                inliers = None
            except cv2.error:
                success = False

        if not success:
            continue

        rvec, tvec = _refine_pnp_solution(
            pts3d,
            pts2d,
            calib_data['K'],
            calib_data.get('dist_coeffs'),
            rvec,
            tvec,
            inliers=inliers,
        )

        projected_pts, _ = cv2.projectPoints(
            pts3d,
            rvec,
            np.asarray(tvec, dtype=np.float32).reshape(3, 1),
            calib_data['K'],
            calib_data.get('dist_coeffs'),
        )
        support_errors = np.linalg.norm(projected_pts.reshape(-1, 2) - pts2d, axis=1)
        support_median = float(np.median(support_errors))
        support_p95 = float(np.percentile(support_errors, 95))
        support_max = float(np.max(support_errors))
        if inliers is not None and len(inliers) > 0:
            inlier_ratio = float(len(inliers)) / float(len(overlap_ids))
        else:
            inlier_ratio = float(np.mean(support_errors <= support_error_threshold))

        error_score = 1.0 / (1.0 + support_median / 3.0)
        tail_score = 1.0 / (1.0 + support_p95 / 5.0)
        support_score = (
            0.40 * inlier_ratio +
            0.30 * error_score +
            0.20 * tail_score +
            0.10 * min(len(overlap_ids) / 10.0, 1.0)
        )
        if (
            support_median <= max(3.5, float(np.median(reproj_errors)) * 2.0 + 0.75) and
            support_p95 <= max(6.0, float(np.median(reproj_errors)) * 3.0 + 1.0) and
            support_max <= max(12.0, float(np.median(reproj_errors)) * 5.0 + 1.5) and
            inlier_ratio >= 0.50
        ):
            supported_cameras += 1
        support_camera_scores.append(float(np.clip(support_score, 0.0, 1.0)))
        support_medians.append(support_median)
        support_p95s.append(support_p95)
        support_inliers.append(inlier_ratio)

    if checked_support_cameras > 0 and support_camera_scores:
        support_coverage = supported_cameras / float(checked_support_cameras)
        cross_view_median = float(np.median(np.asarray(support_medians, dtype=np.float64)))
        cross_view_p95 = float(np.median(np.asarray(support_p95s, dtype=np.float64)))
        support_inlier_ratio = float(np.mean(np.asarray(support_inliers, dtype=np.float64)))
        cross_view_support_score = (
            0.45 * support_coverage +
            0.25 * support_inlier_ratio +
            0.20 * float(np.mean(np.asarray(support_camera_scores, dtype=np.float64))) +
            0.10 * (1.0 / (1.0 + cross_view_p95 / 6.0))
        )
    else:
        support_coverage = 0.0
        cross_view_median = 0.0
        cross_view_p95 = 0.0
        support_inlier_ratio = 0.0
        cross_view_support_score = 0.50

    shared_count = len(common_points)
    inlier_ratio = len(inlier_indices) / shared_count if shared_count else 0.0
    shared_score = min(shared_count / 24.0, 1.0)
    parallax_score = min(median_parallax / 6.0, 1.0)
    reprojection_score = 1.0 / (1.0 + float(np.median(reproj_errors)) / 4.0)
    planarity_score = min(planarity_ratio / 0.05, 1.0)
    triangulation_score = min(len(points_3d) / 16.0, 1.0)

    score = (
        0.12 * shared_score +
        0.15 * inlier_ratio +
        0.15 * triangulation_score +
        0.14 * reprojection_score +
        0.12 * parallax_score +
        0.04 * planarity_score +
        0.13 * track_support_score +
        0.15 * cross_view_support_score
    )

    return {
        'camera_id1': str(camera_id1),
        'camera_id2': str(camera_id2),
        'shared_points': shared_count,
        'inliers': int(len(inlier_indices)),
        'valid_points': int(len(points_3d)),
        'median_reprojection': float(np.median(reproj_errors)),
        'mean_reprojection': float(np.mean(reproj_errors)),
        'median_parallax': median_parallax,
        'front_ratio': front_ratio,
        'planarity_ratio': planarity_ratio,
        'median_track_length': median_track_length,
        'multiview_ratio': float(multiview_ratio),
        'anchor_ratio': float(anchor_ratio),
        'track_support_score': float(track_support_score),
        'checked_support_cameras': int(checked_support_cameras),
        'supported_cameras': int(supported_cameras),
        'support_coverage': float(support_coverage),
        'cross_view_median': float(cross_view_median),
        'cross_view_p95': float(cross_view_p95),
        'support_inlier_ratio': float(support_inlier_ratio),
        'cross_view_support_score': float(cross_view_support_score),
        'score': float(score),
    }


def _collect_initial_pair_candidates(calib_data):
    if 'camera_points' not in calib_data:
        print("Нет данных о точках камер")
        return []

    camera_ids = sorted(calib_data['camera_points'].keys(), key=_stable_sort_key)
    if len(camera_ids) < 2:
        print("Недостаточно камер для инициализации")
        return []

    print(f"Поиск начальной пары камер среди {len(camera_ids)} камер: {camera_ids}")

    if calib_data['K'] is None:
        print("Матрица калибровки камеры не задана")
        return []

    debug_logging = bool(calib_data.get('debug_logging', False))
    candidates = []

    for i, camera_id1 in enumerate(camera_ids[:-1]):
        for camera_id2 in camera_ids[i+1:]:
            points1_dict = calib_data['camera_points'].get(str(camera_id1), {})
            points2_dict = calib_data['camera_points'].get(str(camera_id2), {})
            common_points = sorted(set(points1_dict.keys()) & set(points2_dict.keys()), key=_stable_sort_key)

            if debug_logging:
                print(f"Пара камер {camera_id1}-{camera_id2}: {len(common_points)} общих точек")

            if len(common_points) < 8:
                continue

            metrics = _evaluate_initial_pair_candidate(
                calib_data,
                camera_id1,
                camera_id2,
                points1_dict,
                points2_dict,
                common_points,
            )
            if metrics is None:
                if debug_logging:
                    print(f"  - Пара {camera_id1}-{camera_id2} отклонена на ранней геометрической проверке")
                continue

            if debug_logging:
                print(
                    f"  - score={metrics['score']:.3f}, inliers={metrics['inliers']}/{metrics['shared_points']}, "
                    f"valid={metrics['valid_points']}, median_reproj={metrics['median_reprojection']:.2f}px, "
                    f"median_parallax={metrics['median_parallax']:.2f}°, "
                    f"track_support={metrics['track_support_score']:.2f}, "
                    f"cross_view={metrics['cross_view_support_score']:.2f} "
                    f"({metrics['supported_cameras']}/{metrics['checked_support_cameras']})"
                )

            candidates.append(metrics)

    candidates.sort(
        key=lambda item: (
            item['score'],
            item['valid_points'],
            item['shared_points'],
            -item['median_reprojection'],
        ),
        reverse=True
    )
    return candidates


def find_initial_camera_pair(calib_data):
    """
    Находит локально лучшую начальную пару камер для реконструкции.
    Для глобального выбора с коротким bootstrap-прогоном используется
    _select_initial_pair_for_reconstruction().
    """
    candidates = _collect_initial_pair_candidates(calib_data)
    if not candidates:
        print("Не удалось найти подходящую пару камер с устойчивой начальной геометрией")
        return None

    best = candidates[0]
    print(
        f"Выбрана начальная пара камер: {best['camera_id1']}-{best['camera_id2']} "
        f"(score={best['score']:.3f}, shared={best['shared_points']}, "
        f"inliers={best['inliers']}, median_reproj={best['median_reprojection']:.2f}px, "
        f"median_parallax={best['median_parallax']:.2f}°)"
    )
    return best['camera_id1'], best['camera_id2']


def _clone_calibration_for_pair_selection(calib_data):
    return {
        'cameras': {},
        'points_3d': {},
        'camera_points': copy.deepcopy(calib_data['camera_points']),
        'raw_camera_points': copy.deepcopy(calib_data.get('raw_camera_points', {})),
        'observation_confidences': copy.deepcopy(calib_data.get('observation_confidences', {})),
        'line_support_data': copy.deepcopy(calib_data.get('line_support_data', {})),
        'subpixel_refinement_enabled': bool(calib_data.get('subpixel_refinement_enabled', True)),
        'subpixel_refinement_stats': copy.deepcopy(calib_data.get('subpixel_refinement_stats', {})),
        'K': np.asarray(calib_data['K'], dtype=np.float32).copy() if calib_data.get('K') is not None else None,
        'dist_coeffs': np.asarray(calib_data['dist_coeffs'], dtype=np.float32).copy() if calib_data.get('dist_coeffs') is not None else None,
        'images': copy.deepcopy(calib_data.get('images', {})),
        'image_root': calib_data.get('image_root'),
        'image_width': calib_data.get('image_width'),
        'image_height': calib_data.get('image_height'),
        'reconstruction_ids': [],
        'debug_logging': bool(calib_data.get('debug_logging', False)),
        'precision_cleanup_enabled': bool(calib_data.get('precision_cleanup_enabled', True)),
        'precision_target_mean_px': float(calib_data.get('precision_target_mean_px', 0.5)),
        'precision_target_p95_px': float(calib_data.get('precision_target_p95_px', 1.0)),
        'precision_target_max_px': float(calib_data.get('precision_target_max_px', 1.5)),
        'precision_cleanup_rounds': int(calib_data.get('precision_cleanup_rounds', 4)),
        'strict_track_consistency': bool(calib_data.get('strict_track_consistency', True)),
        'preserve_unreconstructed_annotations': bool(calib_data.get('preserve_unreconstructed_annotations', True)),
        'initial_focal_estimate': copy.deepcopy(calib_data.get('initial_focal_estimate')),
    }


def _estimate_preview_camera_geometry(cameras):
    camera_centers = []
    for camera_id in sorted(cameras.keys(), key=_stable_sort_key):
        try:
            R, t = cameras[camera_id]
            R = np.asarray(R, dtype=np.float64)
            t = np.asarray(t, dtype=np.float64).reshape(3, 1)
            center = (-R.T @ t).reshape(3)
            if np.all(np.isfinite(center)):
                camera_centers.append(center)
        except Exception:
            continue

    if len(camera_centers) < 3:
        return {
            'orbit_span': 0.0,
            'height_range': 0.0,
            'planar_ratio': 0.0,
            'planarity_score': 1.0,
        }

    camera_centers = np.asarray(camera_centers, dtype=np.float64)
    centroid = np.mean(camera_centers, axis=0)
    centered = camera_centers - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis_u = vh[0]
        axis_v = vh[1]
        axis_w = vh[2]
    except Exception:
        return {
            'orbit_span': 0.0,
            'height_range': 0.0,
            'planar_ratio': 1.0,
            'planarity_score': 0.0,
        }

    camera_plane = np.stack(
        [
            centered @ axis_u,
            centered @ axis_v,
            centered @ axis_w,
        ],
        axis=1,
    )
    u_range = float(np.ptp(camera_plane[:, 0]))
    v_range = float(np.ptp(camera_plane[:, 1]))
    orbit_span = float(math.sqrt(u_range ** 2 + v_range ** 2))
    height_range = float(np.ptp(camera_plane[:, 2]))
    planar_ratio = float(height_range / max(orbit_span, 1e-6))
    planarity_score = float(1.0 / (1.0 + max(planar_ratio - 0.04, 0.0) * 4.0))
    return {
        'orbit_span': orbit_span,
        'height_range': height_range,
        'planar_ratio': planar_ratio,
        'planarity_score': planarity_score,
    }


def _estimate_frontier_pose_from_support(calib_data, camera_id, min_points_for_camera, ransac_threshold, confidence):
    camera_id = str(camera_id)
    camera_points_2d = calib_data.get('camera_points', {}).get(camera_id, {})
    if not camera_points_2d:
        return None

    support_point_ids, _ = _select_camera_pose_support_point_ids(
        calib_data,
        camera_id,
        min_points_for_camera=min_points_for_camera,
    )
    support_confidences = np.asarray(
        [_lookup_observation_confidence(calib_data, camera_id, point_id) for point_id in support_point_ids],
        dtype=np.float64,
    ) if support_point_ids else np.ones(0, dtype=np.float64)
    if not _should_use_confidence_guided_pose_support(
        support_confidences,
        min_confidence=_get_soft_gate_min_confidence(calib_data),
    ):
        support_point_ids = []
    if len(support_point_ids) < min_points_for_camera:
        support_point_ids = [
            point_id
            for point_id in sorted(calib_data.get('points_3d', {}).keys(), key=_stable_sort_key)
            if point_id in camera_points_2d
        ]

    points_3d = [calib_data['points_3d'][point_id] for point_id in support_point_ids]
    points_2d = [camera_points_2d[point_id] for point_id in support_point_ids]
    observation_confidences = np.asarray(
        [_lookup_observation_confidence(calib_data, camera_id, point_id) for point_id in support_point_ids],
        dtype=np.float64,
    )

    if len(points_3d) < min_points_for_camera:
        return None

    points_3d = np.asarray(points_3d, dtype=np.float32)
    points_2d = np.asarray(points_2d, dtype=np.float32)

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            calib_data['K'],
            calib_data.get('dist_coeffs'),
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=ransac_threshold,
            confidence=confidence,
            iterationsCount=500,
        )
    except cv2.error:
        success = False
        rvec = None
        tvec = None
        inliers = None

    if not success:
        try:
            success, rvec, tvec = cv2.solvePnP(
                points_3d,
                points_2d,
                calib_data['K'],
                calib_data.get('dist_coeffs'),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            inliers = None
        except cv2.error:
            success = False

    if not success:
        return None

    rvec, tvec = _refine_pnp_solution(
        points_3d,
        points_2d,
        calib_data['K'],
        calib_data.get('dist_coeffs'),
        rvec,
        tvec,
        inliers=inliers,
        observation_confidences=observation_confidences,
        support_min_confidence=_get_soft_gate_min_confidence(calib_data),
    )

    R_candidate, _ = cv2.Rodrigues(rvec)
    return {
        'R': np.asarray(R_candidate, dtype=np.float32),
        't': np.asarray(tvec, dtype=np.float32).reshape(3, 1),
        'support_count': int(len(points_3d)),
        'inlier_count': int(len(inliers)) if inliers is not None else int(len(points_3d)),
    }


def _estimate_triangulatable_growth(
    calib_data,
    camera_id,
    initialized_cameras,
    growth_candidate_ids,
    min_points_for_camera,
    ransac_threshold,
    confidence,
):
    camera_id = str(camera_id)
    if not growth_candidate_ids:
        return None

    candidate_points = calib_data.get('camera_points', {}).get(camera_id, {})
    best_metrics = None

    pose_estimate = _estimate_frontier_pose_from_support(
        calib_data,
        camera_id,
        min_points_for_camera=min_points_for_camera,
        ransac_threshold=ransac_threshold,
        confidence=confidence,
    )

    if pose_estimate is not None:
        R_candidate = pose_estimate['R']
        t_candidate = pose_estimate['t']
        for other_camera_id in sorted(initialized_cameras, key=_stable_sort_key):
            if str(other_camera_id) not in calib_data.get('cameras', {}):
                continue
            other_points = calib_data.get('camera_points', {}).get(str(other_camera_id), {})
            shared_growth_points = [
                point_id for point_id in growth_candidate_ids if point_id in other_points
            ]
            if len(shared_growth_points) < 2:
                continue

            pts_other = np.asarray(
                [other_points[point_id] for point_id in shared_growth_points],
                dtype=np.float32,
            )
            pts_candidate = np.asarray(
                [candidate_points[point_id] for point_id in shared_growth_points],
                dtype=np.float32,
            )
            R_other, t_other = calib_data['cameras'][str(other_camera_id)]
            R_other = np.asarray(R_other, dtype=np.float32)
            t_other = np.asarray(t_other, dtype=np.float32).reshape(3, 1)
            R_rel = R_candidate @ R_other.T
            t_rel = (t_candidate - R_rel @ t_other).ravel()

            with redirect_stdout(io.StringIO()):
                triangulated_points, triangulated_mask = triangulation.triangulate_points(
                    pts_other,
                    pts_candidate,
                    calib_data['K'],
                    R_rel,
                    t_rel,
                    dist_coeffs=calib_data.get('dist_coeffs'),
                    debug_logging=False,
                )

            if triangulated_mask.size == 0:
                continue

            valid_points = int(np.sum(triangulated_mask))
            if valid_points <= 0:
                continue

            shared_count = len(shared_growth_points)
            valid_ratio = valid_points / shared_count if shared_count else 0.0
            score = 0.7 * valid_ratio + 0.3 * min(valid_points / 8.0, 1.0)
            metrics = {
                'partner': str(other_camera_id),
                'shared_points': shared_count,
                'inliers': valid_points,
                'valid_points': valid_points,
                'score': float(score),
                'mode': 'pnp',
            }
            if best_metrics is None or metrics['score'] > best_metrics['score']:
                best_metrics = metrics

    if best_metrics is not None:
        return best_metrics

    for other_camera_id in sorted(initialized_cameras, key=_stable_sort_key):
        other_points = calib_data.get('camera_points', {}).get(str(other_camera_id), {})
        shared_growth_points = [
            point_id for point_id in growth_candidate_ids if point_id in other_points
        ]
        if len(shared_growth_points) < 8:
            continue

        with redirect_stdout(io.StringIO()):
            metrics = _evaluate_initial_pair_candidate(
                calib_data,
                other_camera_id,
                camera_id,
                other_points,
                candidate_points,
                shared_growth_points,
            )
        if metrics is None:
            continue

        candidate_metrics = {
            'partner': str(other_camera_id),
            'shared_points': int(metrics['shared_points']),
            'inliers': int(metrics['inliers']),
            'valid_points': int(metrics['valid_points']),
            'score': float(metrics['score']),
            'mode': '2d2d',
        }
        if best_metrics is None or candidate_metrics['score'] > best_metrics['score']:
            best_metrics = candidate_metrics

    return best_metrics


def _collect_camera_frontier_stats(
    calib_data,
    camera_id,
    initialized_cameras=None,
    min_points_for_camera=4,
    ransac_threshold=8.0,
    confidence=0.99,
    estimate_growth_geometry=False,
):
    camera_id = str(camera_id)
    initialized_cameras = {
        str(cam_id) for cam_id in (
            initialized_cameras
            if initialized_cameras is not None
            else calib_data.get('cameras', {}).keys()
        )
    }
    initialized_cameras.discard(camera_id)

    camera_observations = calib_data.get('camera_points', {}).get(camera_id, {})
    reconstructed_points = set(calib_data.get('points_3d', {}).keys())
    visible_from_reconstruction = set()
    for other_camera_id in initialized_cameras:
        visible_from_reconstruction.update(
            calib_data.get('camera_points', {}).get(str(other_camera_id), {}).keys()
        )

    support_point_ids = []
    growth_candidate_ids = []
    shared_point_ids = []

    for point_id in sorted(camera_observations.keys(), key=_stable_sort_key):
        if point_id in reconstructed_points:
            support_point_ids.append(point_id)
        if point_id in visible_from_reconstruction:
            shared_point_ids.append(point_id)
            if point_id not in reconstructed_points:
                growth_candidate_ids.append(point_id)

    best_growth_metrics = None
    if estimate_growth_geometry:
        best_growth_metrics = _estimate_triangulatable_growth(
            calib_data,
            camera_id,
            initialized_cameras,
            growth_candidate_ids,
            min_points_for_camera=min_points_for_camera,
            ransac_threshold=ransac_threshold,
            confidence=confidence,
        )

    triangulatable_growth_score = float(best_growth_metrics['score']) if best_growth_metrics is not None else 0.0
    triangulatable_growth_points = int(best_growth_metrics['valid_points']) if best_growth_metrics is not None else 0
    triangulatable_growth_inliers = int(best_growth_metrics['inliers']) if best_growth_metrics is not None else 0
    triangulatable_growth_shared = int(best_growth_metrics['shared_points']) if best_growth_metrics is not None else 0
    best_growth_partner = str(best_growth_metrics['partner']) if best_growth_metrics is not None else None
    growth_estimation_mode = str(best_growth_metrics['mode']) if best_growth_metrics is not None else None

    if triangulatable_growth_points > 0:
        role = "рост"
    elif growth_candidate_ids:
        role = "рост?"
    elif support_point_ids:
        role = "уточнение"
    else:
        role = "изолирована"

    return {
        'camera_id': camera_id,
        'support_count': len(support_point_ids),
        'growth_candidate_count': len(growth_candidate_ids),
        'shared_reconstructed_count': len(shared_point_ids),
        'observation_count': len(camera_observations),
        'support_point_ids': support_point_ids,
        'growth_candidate_ids': growth_candidate_ids,
        'shared_point_ids': shared_point_ids,
        'triangulatable_growth_score': triangulatable_growth_score,
        'triangulatable_growth_points': triangulatable_growth_points,
        'triangulatable_growth_inliers': triangulatable_growth_inliers,
        'triangulatable_growth_shared': triangulatable_growth_shared,
        'best_growth_partner': best_growth_partner,
        'growth_estimation_mode': growth_estimation_mode,
        'role': role,
    }


def _order_candidate_cameras(
    calib_data,
    remaining_cameras,
    initialized_cameras,
    min_points_for_camera,
    ransac_threshold=8.0,
    confidence=0.99,
    estimate_growth_geometry=False,
):
    camera_stats = [
        _collect_camera_frontier_stats(
            calib_data,
            camera_id,
            initialized_cameras=initialized_cameras,
            min_points_for_camera=min_points_for_camera,
            ransac_threshold=ransac_threshold,
            confidence=confidence,
            estimate_growth_geometry=estimate_growth_geometry,
        )
        for camera_id in remaining_cameras
    ]

    def _priority(stats):
        can_attach = 1 if stats['support_count'] >= min_points_for_camera else 0
        strong_growth = 1 if stats['triangulatable_growth_points'] > 0 else 0
        can_grow = 1 if stats['growth_candidate_count'] > 0 else 0
        # Камеры с высоким support_count без роста (role=уточнение)
        # должны добавляться не последними, а среди камер роста,
        # чтобы влиять на каркас пока он ещё гибкий.
        # Бонус: support >= 15 AND no growth => приравнять к weak growth.
        high_support_refinement = (
            1 if (stats['support_count'] >= 15 and
                  stats['growth_candidate_count'] == 0 and
                  can_attach)
            else 0
        )
        return (
            can_attach,
            strong_growth,
            stats['triangulatable_growth_score'],
            stats['triangulatable_growth_points'],
            max(can_grow, high_support_refinement),
            stats['growth_candidate_count'] + (stats['support_count'] if high_support_refinement else 0),
            stats['support_count'],
            stats['shared_reconstructed_count'],
            stats['observation_count'],
        )

    camera_stats.sort(key=_priority, reverse=True)
    return camera_stats


def _expand_reconstruction_frontier(
    calib_data,
    initialized_cameras,
    min_points_for_camera,
    ransac_threshold,
    confidence,
    max_attempts,
    progress_callback=None,
    progress_range=(72.0, 82.0),
):
    initialized_cameras = set(str(camera_id) for camera_id in initialized_cameras)
    total_camera_count = max(len(calib_data.get('camera_points', {})), len(initialized_cameras))
    base_initialized_count = max(len(initialized_cameras), 1)

    def _report_progress(progress_value, status_text):
        if progress_callback is None:
            return
        try:
            start_value, end_value = progress_range
            clamped_fraction = float(max(0.0, min(1.0, progress_value)))
            progress_callback(start_value + (end_value - start_value) * clamped_fraction, status_text)
        except Exception:
            pass

    remaining_cameras = [
        cam_id for cam_id in calib_data.get('camera_points', {}).keys()
        if str(cam_id) not in initialized_cameras
    ]
    if not remaining_cameras:
        return initialized_cameras

    _report_progress(0.0, "Добавление остальных камер...")
    print("Добавление остальных камер...")

    total_additional_cameras = max(total_camera_count - base_initialized_count, 1)
    for attempt in range(max_attempts):
        if not remaining_cameras:
            break

        print(f"\nПопытка добавления камер {attempt + 1}/{max_attempts}")
        progress_made = False
        next_remaining = []
        pending_cameras = list(remaining_cameras)

        while pending_cameras:
            ordered_camera_stats = _order_candidate_cameras(
                calib_data,
                pending_cameras,
                initialized_cameras,
                min_points_for_camera=min_points_for_camera,
                ransac_threshold=ransac_threshold,
                confidence=confidence,
                estimate_growth_geometry=True,
            )
            queue_summary = ", ".join(
                (
                    f"{stats['camera_id']}"
                    f"[3D={stats['support_count']}, "
                    f"new={stats['growth_candidate_count']}, "
                    f"tri={stats['triangulatable_growth_points']}, "
                    f"role={stats['role']}]"
                )
                for stats in ordered_camera_stats
            )
            if queue_summary:
                print(f"Приоритет камер: {queue_summary}")

            camera_stats = ordered_camera_stats[0]
            camera_id = camera_stats['camera_id']
            support_count = camera_stats['support_count']
            growth_count = camera_stats['growth_candidate_count']
            triangulatable_growth_count = camera_stats['triangulatable_growth_points']
            best_growth_partner = camera_stats['best_growth_partner']
            growth_estimation_mode = camera_stats.get('growth_estimation_mode')
            role = camera_stats['role']
            growth_details = ""
            if best_growth_partner is not None:
                growth_details = (
                    f", триангулируемый рост: {triangulatable_growth_count} "
                    f"через {best_growth_partner}"
                )
                if growth_estimation_mode:
                    growth_details += f" ({growth_estimation_mode})"
            print(
                f"\nДобавление камеры {camera_id} "
                f"(реконструированных опорных точек: {support_count}, "
                f"кандидатов новых точек: {growth_count}"
                f"{growth_details}, режим: {role})..."
            )

            current_added_cameras = max(len(initialized_cameras) - base_initialized_count, 0)
            _report_progress(
                current_added_cameras / max(total_additional_cameras, 1),
                f"Добавление камер: {len(initialized_cameras)}/{total_camera_count} (текущая {camera_id})"
            )
            if add_camera_to_reconstruction(
                calib_data,
                camera_id,
                min_points=min_points_for_camera,
                ransac_threshold=ransac_threshold,
                confidence=confidence
            ):
                initialized_cameras.add(str(camera_id))
                progress_made = True
                added_cameras = max(len(initialized_cameras) - base_initialized_count, 0)
                _report_progress(
                    added_cameras / max(total_additional_cameras, 1),
                    f"Камеры добавлены: {len(initialized_cameras)}/{total_camera_count}"
                )
                print(f"Камера {camera_id} успешно добавлена")

                # Промежуточный лёгкий BA для стабилизации поз и точек
                if len(calib_data.get('points_3d', {})) >= 6 and len(initialized_cameras) >= 3:
                    try:
                        refine_reconstruction(
                            calib_data,
                            max_iterations=1,
                            optimize_intrinsics=False,
                            optimize_distortion=False,
                        )
                        print(f"  Промежуточный BA после камеры {camera_id}: "
                              f"точек={len(calib_data.get('points_3d', {}))}, "
                              f"камер={len(initialized_cameras)}")
                        _capture_point_drift_stage(
                            calib_data,
                            f"camera_{camera_id}_post_intermediate_ba",
                            {
                                "camera_id": str(camera_id),
                                "camera_count": int(len(initialized_cameras)),
                            },
                        )
                    except Exception as e:
                        print(f"  Промежуточный BA пропущен: {e}")

                # --- Fix 6: K-оптимизация при достаточном количестве камер ---
                # Когда добавлено >= 4 камер, K достаточно стабилен для совместной оптимизации.
                # Уточнённый K сразу используется в следующих PnP-вызовах → меньше PnP-ошибок.
                if (
                    len(initialized_cameras) >= 4
                    and not calib_data.get('_midgrowth_k_refined', False)
                    and len(calib_data.get('points_3d', {})) >= 8
                ):
                    try:
                        focal_range = calib_data.get('_focal_range')
                        force_same_focal = bool(calib_data.get('_force_same_focal', True))
                        k_before = float(calib_data['K'][0, 0])
                        refine_reconstruction(
                            calib_data,
                            max_iterations=2,
                            optimize_intrinsics=True,
                            optimize_distortion=False,
                            focal_range=focal_range,
                            force_same_focal=force_same_focal,
                        )
                        k_after = float(calib_data['K'][0, 0])
                        calib_data['_midgrowth_k_refined'] = True
                        print(
                            f"  Fix 6: K-оптимизация при {len(initialized_cameras)} камерах: "
                            f"fx {k_before:.1f} -> {k_after:.1f} "
                            f"(Δ={abs(k_after - k_before):.1f}px)"
                        )
                        _capture_point_drift_stage(
                            calib_data,
                            f"camera_{camera_id}_post_midgrowth_k",
                            {
                                "camera_id": str(camera_id),
                                "camera_count": int(len(initialized_cameras)),
                                "fx_before": float(k_before),
                                "fx_after": float(k_after),
                            },
                        )
                    except Exception as e:
                        print(f"  Fix 6: K-оптимизация пропущена: {e}")
            else:
                next_remaining.append(camera_id)
                print(f"Не удалось добавить камеру {camera_id}")

            pending_cameras = [
                pending_camera_id
                for pending_camera_id in pending_cameras
                if str(pending_camera_id) != str(camera_id)
            ]

        remaining_cameras = next_remaining
        if not progress_made:
            print("Ни одну из оставшихся камер не удалось добавить на этой попытке")
            break

    return initialized_cameras


def _filter_camera_observations_locally(
    calib_data,
    camera_id,
    protected_point_ids=None,
    absolute_threshold=6.0,
    sigma_multiplier=2.0,
    mad_multiplier=2.5,
    min_observations_per_camera=5,
    min_track_length=3,
):
    if bool(calib_data.get('strict_track_consistency', True)):
        return {
            'removed_observations': 0,
            'downweighted_observations': 0,
            'threshold': None,
            'median_error': None,
            'max_error': None,
        }

    camera_id = str(camera_id)
    observations = calib_data.get('camera_points', {}).get(camera_id, {})
    if len(observations) <= min_observations_per_camera:
        return {
            'removed_observations': 0,
            'downweighted_observations': 0,
            'threshold': None,
            'median_error': None,
            'max_error': None,
        }

    protected_point_ids = {point_id for point_id in (protected_point_ids or [])}
    observation_errors = []
    point_error_buckets = {}
    point_track_lengths = {}

    for other_camera_id, other_observations in calib_data.get('camera_points', {}).items():
        for point_id in other_observations.keys():
            point_track_lengths[point_id] = point_track_lengths.get(point_id, 0) + 1

    R, t = calib_data['cameras'][camera_id]
    for point_id, point_2d in observations.items():
        if point_id in protected_point_ids:
            continue
        if point_id not in calib_data.get('points_3d', {}):
            continue
        projected_point = utils.project_point(
            calib_data['points_3d'][point_id],
            R,
            t,
            _get_camera_matrix(calib_data, camera_id),
            calib_data.get('dist_coeffs'),
        )
        error = float(np.linalg.norm(projected_point - point_2d))
        observation_errors.append((point_id, error))
        point_error_buckets.setdefault(point_id, []).append(error)

    for other_camera_id, (other_R, other_t) in calib_data.get('cameras', {}).items():
        other_camera_id = str(other_camera_id)
        if other_camera_id == camera_id:
            continue
        other_observations = calib_data.get('camera_points', {}).get(other_camera_id, {})
        for point_id, point_2d in other_observations.items():
            if point_id not in point_error_buckets:
                continue
            if point_id not in calib_data.get('points_3d', {}):
                continue
            projected_point = utils.project_point(
                calib_data['points_3d'][point_id],
                other_R,
                other_t,
                _get_camera_matrix(calib_data, other_camera_id),
                calib_data.get('dist_coeffs'),
            )
            error = float(np.linalg.norm(projected_point - point_2d))
            point_error_buckets[point_id].append(error)

    if len(observation_errors) < max(6, min_observations_per_camera + 1):
        return {
            'removed_observations': 0,
            'downweighted_observations': 0,
            'threshold': None,
            'median_error': None,
            'max_error': None,
        }

    errors_only = np.array([item[1] for item in observation_errors], dtype=np.float64)
    median_error = float(np.median(errors_only))
    std_error = float(np.std(errors_only))
    mad_error = float(stats.median_abs_deviation(errors_only, scale='normal'))
    sigma_threshold = median_error + sigma_multiplier * max(std_error, 1e-6)
    mad_threshold = median_error + mad_multiplier * max(mad_error, 1e-6)
    combined_threshold = max(absolute_threshold, min(sigma_threshold, mad_threshold))
    max_error = float(np.max(errors_only))

    candidate_removals = []
    for point_id, error in observation_errors:
        point_errors = np.array(point_error_buckets.get(point_id, [error]), dtype=np.float64)
        point_median = float(np.median(point_errors))
        point_mad = float(stats.median_abs_deviation(point_errors, scale='normal'))
        point_threshold = point_median + 2.5 * max(point_mad, 0.5)
        if (
            error > combined_threshold and
            error > point_threshold and
            error > max(absolute_threshold, point_median * 1.75)
        ):
            candidate_removals.append((point_id, error, point_median, point_threshold))
    if not candidate_removals:
        return {
            'removed_observations': 0,
            'downweighted_observations': 0,
            'threshold': combined_threshold,
            'median_error': median_error,
            'max_error': max_error,
        }

    camera_remaining = len(observations)
    point_remaining = dict(point_track_lengths)
    removals_to_apply = []
    for point_id, error, point_median, point_threshold in sorted(candidate_removals, key=lambda item: item[1], reverse=True):
        if camera_remaining <= min_observations_per_camera:
            break
        if point_remaining.get(point_id, 0) <= min_track_length:
            continue
        if point_id not in calib_data['camera_points'].get(camera_id, {}):
            continue
        removals_to_apply.append((point_id, error, point_median, point_threshold))
        camera_remaining -= 1
        point_remaining[point_id] -= 1

    downweighted = 0
    for point_id, _, _, _ in removals_to_apply:
        _set_observation_confidence(
            calib_data,
            camera_id,
            point_id,
            min(_lookup_observation_confidence(calib_data, camera_id, point_id) * 0.45, 0.22),
            mode="min",
        )
        downweighted += 1

    return {
        'removed_observations': 0,
        'downweighted_observations': int(downweighted),
        'threshold': combined_threshold,
        'median_error': median_error,
        'max_error': max_error,
    }


def _simulate_reconstruction_from_pair(
    calib_data,
    initial_pair,
    min_points_for_camera,
    ransac_threshold,
    confidence,
    max_attempts,
    return_trial_data=False,
):
    trial_data = _clone_calibration_for_pair_selection(calib_data)
    trial_data['_multiview_refine_mode'] = 'bootstrap_preview'
    debug_logging = bool(calib_data.get('debug_logging', False))
    output_buffer = io.StringIO()
    context = nullcontext() if debug_logging else redirect_stdout(output_buffer)

    with context:
        if not initialize_reconstruction(trial_data, initial_pair[0], initial_pair[1]):
            return None

        initialized_cameras = {str(initial_pair[0]), str(initial_pair[1])}

        remaining_cameras = [
            cam_id for cam_id in trial_data['camera_points'].keys()
            if str(cam_id) not in initialized_cameras
        ]

        for _ in range(max_attempts):
            if not remaining_cameras:
                break

            progress_made = False
            pending_cameras = list(remaining_cameras)
            next_remaining = []

            while pending_cameras:
                ordered_camera_stats = _order_candidate_cameras(
                    trial_data,
                    pending_cameras,
                    initialized_cameras,
                    min_points_for_camera=min_points_for_camera,
                    ransac_threshold=ransac_threshold,
                    confidence=confidence,
                    estimate_growth_geometry=False,
                )
                camera_stats = ordered_camera_stats[0]
                camera_id = camera_stats['camera_id']
                if add_camera_to_reconstruction(
                    trial_data,
                    camera_id,
                    min_points=min_points_for_camera,
                    ransac_threshold=ransac_threshold,
                    confidence=confidence
                ):
                    initialized_cameras.add(str(camera_id))
                    progress_made = True
                else:
                    next_remaining.append(camera_id)
                pending_cameras = [
                    pending_camera_id
                    for pending_camera_id in pending_cameras
                    if str(pending_camera_id) != str(camera_id)
                ]

            remaining_cameras = next_remaining
            if not progress_made:
                break

        mean_error, _, _ = calculate_reprojection_errors(trial_data)
        distribution = summarize_reprojection_error_distribution(trial_data, top_k=0)

    result = {
        'pair': (str(initial_pair[0]), str(initial_pair[1])),
        'cameras': len(trial_data['cameras']),
        'points': len(trial_data['points_3d']),
        'mean_error': float(mean_error),
        'median_error': float(distribution.get('median', 0.0)),
        'p95': float(distribution.get('p95', 0.0)),
        'max_error': float(distribution.get('max', 0.0)),
    }
    if return_trial_data:
        result['trial_data'] = trial_data
    return result


def _select_initial_pair_for_reconstruction(
    calib_data,
    min_points_for_camera,
    ransac_threshold,
    confidence,
    max_attempts,
    max_candidates=None,
    simulation_max_attempts=None,
    return_details=False,
    verbose=True,
):
    candidates = _collect_initial_pair_candidates(calib_data)
    if not candidates:
        return None

    debug_logging = bool(calib_data.get('debug_logging', False))
    if max_candidates is None:
        max_candidates = 4
    max_candidates = max(1, int(max_candidates))
    max_candidates = min(max_candidates, len(candidates))
    shortlisted = candidates[:max_candidates]

    if verbose:
        print(f"Короткая проверка {max_candidates} стартовых пар по ранней реконструкции...")
    best_result = None

    for candidate in shortlisted:
        pair = (candidate['camera_id1'], candidate['camera_id2'])
        simulation = _simulate_reconstruction_from_pair(
            calib_data,
            pair,
            min_points_for_camera=min_points_for_camera,
            ransac_threshold=ransac_threshold,
            confidence=confidence,
            max_attempts=min(
                max_attempts,
                2 if simulation_max_attempts is None else max(1, int(simulation_max_attempts))
            ),
        )
        if simulation is None:
            continue

        if debug_logging and verbose:
            print(
                f"  - Пара {pair[0]}-{pair[1]}: cameras={simulation['cameras']}, "
                f"points={simulation['points']}, mean_error={simulation['mean_error']:.2f}px, "
                f"local_score={candidate['score']:.3f}"
            )

        simulation_error_score = 1.0 / (1.0 + float(simulation['mean_error']) / 2.0)
        simulation_tail_score = 1.0 / (1.0 + float(simulation.get('p95', simulation['mean_error'])) / 2.5)
        simulation_max_score = 1.0 / (1.0 + float(simulation.get('max_error', simulation['mean_error'])) / 3.5)
        simulation_point_score = min(float(simulation['points']) / 24.0, 1.0)
        simulation_camera_score = min(float(simulation['cameras']) / max(2.0, float(len(calib_data.get('camera_points', {})))), 1.0)
        bootstrap_score = (
            0.34 * float(candidate['score']) +
            0.22 * simulation_camera_score +
            0.16 * simulation_error_score +
            0.14 * simulation_tail_score +
            0.08 * simulation_max_score +
            0.06 * simulation_point_score
        )

        ranking_key = (
            simulation['cameras'],
            bootstrap_score,
            -float(simulation.get('p95', simulation['mean_error'])),
            -float(simulation.get('max_error', simulation['mean_error'])),
            candidate['score'],
            simulation['points'],
            -simulation['mean_error'],
        )
        if best_result is None or ranking_key > best_result['ranking_key']:
            best_result = {
                'pair': pair,
                'simulation': simulation,
                'candidate': candidate,
                'ranking_key': ranking_key,
                'bootstrap_score': bootstrap_score,
            }

    if best_result is None:
        fallback = candidates[0]
        if verbose:
            print(
                f"Bootstrap-выбор стартовой пары не удался, используем локально лучшую пару "
                f"{fallback['camera_id1']}-{fallback['camera_id2']}"
            )
        fallback_pair = (fallback['camera_id1'], fallback['camera_id2'])
        if return_details:
            return {
                'pair': fallback_pair,
                'candidate': fallback,
                'simulation': None,
                'bootstrap_score': float(fallback['score']),
            }
        return fallback_pair

    chosen_pair = best_result['pair']
    chosen_candidate = best_result['candidate']
    chosen_simulation = best_result['simulation']
    if verbose:
        print(
            f"Выбрана стартовая пара после короткой проверки: {chosen_pair[0]}-{chosen_pair[1]} "
            f"(bootstrap_score={best_result['bootstrap_score']:.3f}, "
            f"local_score={chosen_candidate['score']:.3f}, cameras={chosen_simulation['cameras']}, "
            f"points={chosen_simulation['points']}, mean_error={chosen_simulation['mean_error']:.2f}px)"
        )
    if return_details:
        return {
            'pair': chosen_pair,
            'candidate': chosen_candidate,
            'simulation': chosen_simulation,
            'bootstrap_score': float(best_result['bootstrap_score']),
        }
    return chosen_pair


def _build_project_level_focal_hypotheses(current_focal, focal_range, hypothesis_count=5, extra_candidates=None):
    if focal_range is None:
        return [float(current_focal)]

    min_focal = float(focal_range[0])
    max_focal = float(focal_range[1])
    if not np.isfinite(min_focal) or not np.isfinite(max_focal) or max_focal <= min_focal:
        return [float(current_focal)]

    hypothesis_count = int(hypothesis_count)
    linear_samples = []
    if hypothesis_count > 0:
        hypothesis_count = max(hypothesis_count, 3)
        if hypothesis_count <= 3:
            linear_samples = np.asarray([min_focal, max_focal], dtype=np.float64)
        else:
            linear_samples = np.linspace(min_focal, max_focal, hypothesis_count, dtype=np.float64)

    def _append_stabilized_candidate(target_list, value, dense=False):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(numeric_value):
            return
        clamped = float(np.clip(numeric_value, min_focal, max_focal))
        if dense:
            target_list.append(float(np.clip(round(clamped, 2), min_focal, max_focal)))
            target_list.append(float(np.clip(round(clamped, 1), min_focal, max_focal)))
            target_list.append(float(np.clip(round(clamped), min_focal, max_focal)))
        else:
            target_list.append(float(np.clip(round(clamped), min_focal, max_focal)))

    candidates = []
    for value in linear_samples:
        _append_stabilized_candidate(candidates, value, dense=False)
    _append_stabilized_candidate(candidates, current_focal, dense=True)
    for value in extra_candidates or []:
        _append_stabilized_candidate(candidates, value, dense=True)

    unique = []
    merge_tolerance = max(50.0, (max_focal - min_focal) * 0.03)
    for value in sorted(candidates):
        numeric_value = float(value)
        if unique and abs(numeric_value - unique[-1]) < merge_tolerance:
            if abs(numeric_value - float(current_focal)) < abs(unique[-1] - float(current_focal)):
                unique[-1] = numeric_value
            continue
        unique.append(numeric_value)

    if hypothesis_count >= 3 and len(unique) < 3:
        midpoint = float((min_focal + max_focal) * 0.5)
        if not any(abs(midpoint - existing) < merge_tolerance for existing in unique):
            unique.append(midpoint)
            unique = sorted(unique)
    return unique


def _evaluate_project_level_focal_hypothesis(
    calib_data,
    focal_px,
    min_points_for_camera,
    ransac_threshold,
    confidence,
    max_attempts,
    focal_range,
    force_same_focal,
    fixed_pair=None,
    precision_preview_rounds=0,
    precision_targets=None,
    full_pipeline_preview=False,
):
    trial_data = _clone_calibration_for_pair_selection(calib_data)
    if trial_data.get('K') is None:
        return None

    trial_data['_project_level_preview_mode'] = True
    trial_data['_multiview_refine_mode'] = 'full_preview' if full_pipeline_preview else 'quick_preview'

    if not full_pipeline_preview:
        trial_data['precision_cleanup_enabled'] = False
        trial_data['precision_cleanup_rounds'] = 0

    current_K = np.asarray(trial_data['K'], dtype=np.float64)
    current_fx = float(current_K[0, 0]) if float(current_K[0, 0]) != 0.0 else 1.0
    scale = float(focal_px) / current_fx
    candidate_K = np.array(current_K, copy=True)
    candidate_K[0, 0] = current_K[0, 0] * scale
    candidate_K[1, 1] = current_K[1, 1] * scale
    trial_data['K'] = candidate_K.astype(np.float32)

    output_buffer = io.StringIO()
    context = nullcontext() if (bool(calib_data.get('debug_logging', False)) or full_pipeline_preview) else redirect_stdout(output_buffer)

    with context:
        if fixed_pair is None:
            quick_pair_bootstrap = not full_pipeline_preview
            pair_details = _select_initial_pair_for_reconstruction(
                trial_data,
                min_points_for_camera=min_points_for_camera,
                ransac_threshold=ransac_threshold,
                confidence=confidence,
                max_attempts=max_attempts,
                max_candidates=2 if quick_pair_bootstrap else 4,
                simulation_max_attempts=1 if quick_pair_bootstrap else 2,
                return_details=True,
                verbose=False,
            )
            if not pair_details:
                return None
            chosen_pair = pair_details['pair']
            bootstrap_score = float(pair_details.get('bootstrap_score', 0.0))
            local_score = float(pair_details.get('candidate', {}).get('score', 0.0))
        else:
            chosen_pair = (str(fixed_pair[0]), str(fixed_pair[1]))
            pair_details = None
            bootstrap_score = 0.0
            local_score = 0.0

        if full_pipeline_preview:
            success = perform_full_reconstruction(
                trial_data,
                initial_pair=chosen_pair,
                min_points_for_camera=min_points_for_camera,
                ransac_threshold=ransac_threshold,
                confidence=confidence,
                max_attempts=max_attempts,
                focal_range=focal_range,
                adapt_initial_focal=False,
                force_same_focal=force_same_focal,
                progress_callback=None,
            )
            if not success:
                return None
            refined_trial_data = trial_data
        else:
            simulation_attempts = max_attempts if int(precision_preview_rounds) > 0 else min(max_attempts, 2)
            simulation = _simulate_reconstruction_from_pair(
                trial_data,
                chosen_pair,
                min_points_for_camera=min_points_for_camera,
                ransac_threshold=ransac_threshold,
                confidence=confidence,
                max_attempts=simulation_attempts,
                return_trial_data=True,
            )
            if simulation is None:
                return None

            refined_trial_data = simulation.get('trial_data')
            if refined_trial_data is None:
                return None

            refine_reconstruction(
                refined_trial_data,
                max_iterations=1,
                focal_range=focal_range,
                force_same_focal=force_same_focal,
                optimize_intrinsics=False,
                optimize_distortion=False,
            )

            if int(precision_preview_rounds) > 0:
                preview_targets = precision_targets or {}
                run_precision_first_cleanup(
                    refined_trial_data,
                    target_mean=float(preview_targets.get('mean', 0.5)),
                    target_p95=float(preview_targets.get('p95', 1.0)),
                    target_max=float(preview_targets.get('max', 1.0)),
                    max_rounds=int(max(1, precision_preview_rounds)),
                    min_observations_per_camera=max(min_points_for_camera + 2, 6),
                    min_track_length=3,
                    min_points=max(12, min_points_for_camera * 3),
                    focal_range=focal_range,
                    force_same_focal=force_same_focal,
                )

        mean_error, _, _ = calculate_reprojection_errors(refined_trial_data)
        distribution_stats = summarize_reprojection_error_distribution(refined_trial_data, top_k=3)

    camera_count = len(refined_trial_data.get('cameras', {}))
    point_count = len(refined_trial_data.get('points_3d', {}))
    mean_error = float(mean_error)
    p95 = float(distribution_stats.get('p95', mean_error if np.isfinite(mean_error) else 1e6))
    max_error = float(distribution_stats.get('max', p95 if np.isfinite(p95) else 1e6))

    project_camera_points = calib_data.get('camera_points', {})
    project_point_ids = set()
    project_observation_count = 0
    for observations in project_camera_points.values():
        project_point_ids.update(observations.keys())
        project_observation_count += len(observations)
    total_project_points = max(len(project_point_ids), 1)
    total_project_cameras = max(len(project_camera_points), 1)

    reconstructed_point_ids = set(refined_trial_data.get('points_3d', {}).keys())
    reconstructed_observation_count = 0
    for camera_id, observations in refined_trial_data.get('camera_points', {}).items():
        if str(camera_id) not in refined_trial_data.get('cameras', {}):
            continue
        reconstructed_observation_count += sum(
            1 for point_id in observations.keys()
            if point_id in reconstructed_point_ids
        )

    point_coverage = float(point_count) / float(total_project_points)
    camera_coverage = float(camera_count) / float(total_project_cameras)
    observation_coverage = float(reconstructed_observation_count) / float(max(project_observation_count, 1))
    mean_track_length = float(reconstructed_observation_count) / float(max(point_count, 1))
    camera_geometry = _estimate_preview_camera_geometry(refined_trial_data.get('cameras', {}))
    planar_ratio = float(camera_geometry.get('planar_ratio', 0.0))
    planarity_score = float(camera_geometry.get('planarity_score', 1.0))
    density_floor = max(
        int(min_points_for_camera) * 4,
        min(14, total_project_points),
        min(total_project_points, max(8, int(math.ceil(float(total_project_points) * 0.24)))),
    )
    density_floor_score = min(float(point_count) / float(max(density_floor, 1)), 1.0)
    project_point_coverage_score = min(point_coverage / 0.45, 1.0)
    observation_coverage_score = min(observation_coverage / 0.30, 1.0)
    track_depth_score = min(max(mean_track_length - 2.0, 0.0) / 1.5, 1.0)

    mean_score = 1.0 / (1.0 + mean_error)
    p95_score = 1.0 / (1.0 + p95)
    max_score = 1.0 / (1.0 + max_error)
    quality_score = (
        0.16 * mean_score +
        0.10 * p95_score +
        0.06 * max_score +
        0.22 * density_floor_score +
        0.16 * project_point_coverage_score +
        0.12 * observation_coverage_score +
        0.08 * track_depth_score +
        0.06 * camera_coverage +
        0.08 * planarity_score +
        0.10 * max(bootstrap_score, local_score)
    )

    sparse_penalty = 0.0
    sparse_penalty_steps = 0
    if point_count < density_floor:
        deficit_ratio = (float(density_floor) - float(point_count)) / float(max(density_floor, 1))
        sparse_penalty += min(0.60, deficit_ratio * 0.80)
        sparse_penalty_steps += 1 if deficit_ratio < 0.15 else 2
    if point_coverage < 0.26:
        sparse_penalty += min(0.30, (0.26 - point_coverage) * 1.30)
        sparse_penalty_steps += 1
    if observation_coverage < 0.20:
        sparse_penalty += min(0.28, (0.20 - observation_coverage) * 1.20)
        sparse_penalty_steps += 1
    if mean_track_length < 2.70:
        sparse_penalty += min(0.22, (2.70 - mean_track_length) * 0.22)
        sparse_penalty_steps += 1
    if camera_count >= 5 and planar_ratio > 0.14:
        sparse_penalty += min(0.35, (planar_ratio - 0.14) * 1.80)
        sparse_penalty_steps += 1
    quality_score -= sparse_penalty

    return {
        'focal_px': float(focal_px),
        'K': np.asarray(refined_trial_data.get('K', candidate_K), dtype=np.float32),
        'pair': (str(chosen_pair[0]), str(chosen_pair[1])),
        'bootstrap_score': float(bootstrap_score),
        'local_score': float(local_score),
        'cameras': int(camera_count),
        'points': int(point_count),
        'mean_error': mean_error,
        'p95': p95,
        'max_error': max_error,
        'density_floor': int(density_floor),
        'point_coverage': float(point_coverage),
        'camera_coverage': float(camera_coverage),
        'observation_coverage': float(observation_coverage),
        'mean_track_length': float(mean_track_length),
        'track_depth_score': float(track_depth_score),
        'planar_ratio': float(planar_ratio),
        'planarity_score': float(planarity_score),
        'density_floor_score': float(density_floor_score),
        'sparse_penalty': float(sparse_penalty),
        'sparse_penalty_steps': int(sparse_penalty_steps),
        'quality_score': float(quality_score),
        'preview_calib_data': (
            copy.deepcopy(refined_trial_data)
            if (not full_pipeline_preview and int(precision_preview_rounds) > 0)
            else None
        ),
        'final_calib_data': copy.deepcopy(refined_trial_data) if full_pipeline_preview else None,
    }


def _select_project_level_focal_hypothesis(
    calib_data,
    focal_range,
    min_points_for_camera,
    ransac_threshold,
    confidence,
    max_attempts,
    force_same_focal=False,
    fixed_pair=None,
):
    if calib_data.get('K') is None:
        return None

    K = np.asarray(calib_data['K'], dtype=np.float64)
    current_focal = float((float(K[0, 0]) + float(K[1, 1])) * 0.5)
    extra_candidates = []
    focal_estimate_confidence = 1.0
    fallback_focal_px = None
    image_focal_px = None
    initial_focal_estimate = calib_data.get('initial_focal_estimate')
    if isinstance(initial_focal_estimate, dict):
        focal_estimate_confidence = float(initial_focal_estimate.get('confidence', 1.0))
        fallback_focal_px = initial_focal_estimate.get('fallback_focal_px')
        if fallback_focal_px is not None:
            extra_candidates.append(fallback_focal_px)
        image_focal_px = initial_focal_estimate.get('focal_px')
        if image_focal_px is not None:
            extra_candidates.append(image_focal_px)

    low_confidence_focal_estimate = (
        fallback_focal_px is not None and
        image_focal_px is not None and
        focal_estimate_confidence < 0.45
    )

    focal_hypotheses = _build_project_level_focal_hypotheses(
        current_focal,
        focal_range,
        hypothesis_count=0 if low_confidence_focal_estimate else 3,
        extra_candidates=extra_candidates,
    )
    if len(focal_hypotheses) <= 1:
        return None

    precision_targets = {
        'mean': float(calib_data.get('precision_target_mean_px', 0.5)),
        'p95': float(calib_data.get('precision_target_p95_px', 1.0)),
        'max': float(calib_data.get('precision_target_max_px', 1.5)),
    }
    total_project_cameras = max(1, len(calib_data.get('camera_points', {})))
    all_project_points = set()
    for observations in calib_data.get('camera_points', {}).values():
        all_project_points.update(observations.keys())
    total_project_points = max(1, len(all_project_points))

    if low_confidence_focal_estimate:
        compact_hypotheses = []
        for value in (fallback_focal_px, image_focal_px, current_focal):
            if value is None:
                continue
            numeric_value = float(value)
            if any(abs(numeric_value - existing) < 75.0 for existing in compact_hypotheses):
                continue
            compact_hypotheses.append(numeric_value)
        compact_hypotheses = sorted(compact_hypotheses)

        print(
            "Project-level focal sweep: "
            f"low-confidence quick eval для {len(compact_hypotheses)} гипотез "
            f"({', '.join(f'fx~{value:.0f}' for value in compact_hypotheses)})"
        )

        low_confidence_results = []
        for focal_px in compact_hypotheses:
            quick_result = _evaluate_project_level_focal_hypothesis(
                calib_data,
                focal_px=focal_px,
                min_points_for_camera=min_points_for_camera,
                ransac_threshold=ransac_threshold,
                confidence=confidence,
                max_attempts=max_attempts,
                focal_range=focal_range,
                force_same_focal=force_same_focal,
                fixed_pair=fixed_pair,
            )
            if quick_result is None:
                continue

            achieved_targets = int(quick_result['mean_error'] <= precision_targets['mean'])
            achieved_targets += int(quick_result['p95'] <= precision_targets['p95'])
            achieved_targets += int(quick_result['max_error'] <= precision_targets['max'])
            effective_hits = achieved_targets - int(quick_result.get('sparse_penalty_steps', 0))
            precision_penalty = (
                max(0.0, quick_result['mean_error'] - precision_targets['mean']) / max(precision_targets['mean'], 1e-6) +
                max(0.0, quick_result['p95'] - precision_targets['p95']) / max(precision_targets['p95'], 1e-6) +
                max(0.0, quick_result['max_error'] - precision_targets['max']) / max(precision_targets['max'], 1e-6)
            )
            quick_ranking_key = (
                effective_hits,
                quick_result['cameras'],
                quick_result.get('planarity_score', 1.0),
                -quick_result.get('planar_ratio', 0.0),
                quick_result.get('density_floor_score', 0.0),
                quick_result.get('point_coverage', 0.0),
                quick_result.get('observation_coverage', 0.0),
                quick_result.get('track_depth_score', 0.0),
                quick_result['quality_score'],
                -precision_penalty,
                quick_result['points'],
                -quick_result['mean_error'],
                -quick_result['p95'],
                -quick_result['max_error'],
            )
            quick_result['preview_ranking_key'] = quick_ranking_key
            low_confidence_results.append(quick_result)

            print(
                f"  - fx~{quick_result['focal_px']:.0f}: "
                f"mean={quick_result['mean_error']:.2f}px, "
                f"p95={quick_result['p95']:.2f}px, "
                f"max={quick_result['max_error']:.2f}px, "
                f"points={quick_result['points']}, hits={achieved_targets}/3, "
                f"coverage={quick_result.get('point_coverage', 0.0):.2f}"
            )

        if low_confidence_results:
            low_confidence_results.sort(key=lambda item: item['preview_ranking_key'], reverse=True)
            best_preview = low_confidence_results[0]
            if len(low_confidence_results) >= 2:
                second_preview = low_confidence_results[1]
                ambiguous_quick_eval = (
                    abs(best_preview['mean_error'] - second_preview['mean_error']) <= 0.05 and
                    abs(best_preview['p95'] - second_preview['p95']) <= 0.35 and
                    abs(best_preview['points'] - second_preview['points']) <= 2
                )
                if ambiguous_quick_eval:
                    print("Project-level focal sweep: quick eval неоднозначен, запускаем короткий preview для 2 лучших гипотез")
                    refined_results = []
                    for candidate in (best_preview, second_preview):
                        preview_result = _evaluate_project_level_focal_hypothesis(
                            calib_data,
                            focal_px=candidate['focal_px'],
                            min_points_for_camera=min_points_for_camera,
                            ransac_threshold=ransac_threshold,
                            confidence=confidence,
                            max_attempts=max_attempts,
                            focal_range=focal_range,
                            force_same_focal=force_same_focal,
                            fixed_pair=candidate['pair'],
                            precision_preview_rounds=1,
                            precision_targets=precision_targets,
                        )
                        if preview_result is None:
                            continue
                        achieved_targets = int(preview_result['mean_error'] <= precision_targets['mean'])
                        achieved_targets += int(preview_result['p95'] <= precision_targets['p95'])
                        achieved_targets += int(preview_result['max_error'] <= precision_targets['max'])
                        effective_hits = achieved_targets - int(preview_result.get('sparse_penalty_steps', 0))
                        precision_penalty = (
                            max(0.0, preview_result['mean_error'] - precision_targets['mean']) / max(precision_targets['mean'], 1e-6) +
                            max(0.0, preview_result['p95'] - precision_targets['p95']) / max(precision_targets['p95'], 1e-6) +
                            max(0.0, preview_result['max_error'] - precision_targets['max']) / max(precision_targets['max'], 1e-6)
                        )
                        preview_result['preview_ranking_key'] = (
                            effective_hits,
                            preview_result['cameras'],
                            preview_result.get('planarity_score', 1.0),
                            -preview_result.get('planar_ratio', 0.0),
                            preview_result.get('density_floor_score', 0.0),
                            preview_result.get('point_coverage', 0.0),
                            preview_result.get('observation_coverage', 0.0),
                            preview_result.get('track_depth_score', 0.0),
                            preview_result['quality_score'],
                            -precision_penalty,
                            preview_result['points'],
                            -preview_result['mean_error'],
                            -preview_result['p95'],
                            -preview_result['max_error'],
                        )
                        refined_results.append(preview_result)
                        print(
                            f"    * fx~{preview_result['focal_px']:.0f}: "
                            f"mean={preview_result['mean_error']:.2f}px, "
                            f"p95={preview_result['p95']:.2f}px, "
                            f"max={preview_result['max_error']:.2f}px, "
                            f"points={preview_result['points']}, "
                            f"coverage={preview_result.get('point_coverage', 0.0):.2f}"
                        )
                    if refined_results:
                        refined_results.sort(key=lambda item: item['preview_ranking_key'], reverse=True)
                        best_preview = refined_results[0]
                        best_preview['selection_mode'] = 'low_confidence_preview'
                    else:
                        best_preview['selection_mode'] = 'low_confidence_quick_eval'
                else:
                    best_preview['selection_mode'] = 'low_confidence_quick_eval'
            else:
                best_preview['selection_mode'] = 'low_confidence_quick_eval'
            print(
                "Project-level focal sweep: выбрана гипотеза "
                f"fx~{best_preview['focal_px']:.0f}px после {best_preview['selection_mode']}, "
                f"pair={best_preview['pair'][0]}-{best_preview['pair'][1]}, "
                f"cameras={best_preview['cameras']}, points={best_preview['points']}, "
                f"mean={best_preview['mean_error']:.2f}px, p95={best_preview['p95']:.2f}px, "
                f"max={best_preview['max_error']:.2f}px"
            )
            return best_preview

    print(
        "Project-level focal sweep: "
        f"проверяем {len(focal_hypotheses)} гипотез "
        f"в диапазоне {focal_hypotheses[0]:.0f}-{focal_hypotheses[-1]:.0f}px"
    )

    candidate_results = []
    for focal_px in focal_hypotheses:
        result = _evaluate_project_level_focal_hypothesis(
            calib_data,
            focal_px=focal_px,
            min_points_for_camera=min_points_for_camera,
            ransac_threshold=ransac_threshold,
            confidence=confidence,
            max_attempts=max_attempts,
            focal_range=focal_range,
            force_same_focal=force_same_focal,
            fixed_pair=fixed_pair,
        )
        if result is None:
            continue

        print(
            f"  - fx~{result['focal_px']:.0f}: pair={result['pair'][0]}-{result['pair'][1]}, "
            f"cameras={result['cameras']}, points={result['points']}, "
            f"mean={result['mean_error']:.2f}px, p95={result['p95']:.2f}px, "
            f"planar={result.get('planar_ratio', 0.0):.3f}, score={result['quality_score']:.3f}"
        )

        achieved_targets = int(result['mean_error'] <= precision_targets['mean'])
        achieved_targets += int(result['p95'] <= precision_targets['p95'])
        achieved_targets += int(result['max_error'] <= precision_targets['max'])
        effective_hits = achieved_targets - int(result.get('sparse_penalty_steps', 0))
        precision_penalty = (
            max(0.0, result['mean_error'] - precision_targets['mean']) / max(precision_targets['mean'], 1e-6) +
            max(0.0, result['p95'] - precision_targets['p95']) / max(precision_targets['p95'], 1e-6) +
            max(0.0, result['max_error'] - precision_targets['max']) / max(precision_targets['max'], 1e-6)
        )
        ranking_key = (
            result['cameras'],
            achieved_targets,
            -precision_penalty,
            effective_hits,
            result.get('planarity_score', 1.0),
            -result.get('planar_ratio', 0.0),
            result.get('density_floor_score', 0.0),
            result.get('point_coverage', 0.0),
            result.get('observation_coverage', 0.0),
            result.get('track_depth_score', 0.0),
            result['quality_score'],
            result['points'],
            -result['mean_error'],
            -result['p95'],
            -result['max_error'],
        )
        candidate = dict(result)
        candidate['ranking_key'] = ranking_key
        candidate['achieved_targets'] = int(achieved_targets)
        candidate['effective_hits'] = int(effective_hits)
        candidate['precision_penalty'] = float(precision_penalty)
        candidate_results.append(candidate)

    if not candidate_results:
        print("Project-level focal sweep: надежная гипотеза не найдена, сохраняем исходный K")
        return None

    candidate_results.sort(key=lambda item: item['ranking_key'], reverse=True)
    best_result = candidate_results[0]

    preview_candidates = []
    preview_candidate_keys = set()
    dominant_best_candidate = False
    if len(candidate_results) >= 2:
        first_candidate = candidate_results[0]
        second_candidate = candidate_results[1]
        dominant_best_candidate = (
            first_candidate['cameras'] > second_candidate['cameras'] or
            (
                first_candidate['cameras'] == second_candidate['cameras'] and
                first_candidate.get('achieved_targets', 0) > second_candidate.get('achieved_targets', 0)
            ) or
            (
                first_candidate['cameras'] == second_candidate['cameras'] and
                first_candidate.get('achieved_targets', 0) == second_candidate.get('achieved_targets', 0) and
                first_candidate.get('density_floor_score', 0.0) >= second_candidate.get('density_floor_score', 0.0) - 0.02 and
                first_candidate['cameras'] == second_candidate['cameras'] and
                first_candidate.get('precision_penalty', 0.0) + 0.25 < second_candidate.get('precision_penalty', 0.0)
            )
        )

    def _append_preview_candidate(candidate):
        if candidate is None:
            return
        key = round(float(candidate['focal_px']), 2)
        if key in preview_candidate_keys:
            return
        preview_candidate_keys.add(key)
        preview_candidates.append(candidate)

    anchor_focals = [current_focal]
    if isinstance(initial_focal_estimate, dict):
        for value in (initial_focal_estimate.get('focal_px'), initial_focal_estimate.get('fallback_focal_px')):
            if value is not None:
                anchor_focals.append(float(value))

    for anchor in anchor_focals:
        nearest_candidate = min(
            candidate_results,
            key=lambda item: (abs(float(item['focal_px']) - float(anchor)), -item['quality_score'])
        ) if candidate_results else None
        _append_preview_candidate(nearest_candidate)

    preview_limit = 1 if dominant_best_candidate else 2
    if dominant_best_candidate:
        print(
            "Project-level focal sweep: "
            "одна гипотеза явно доминирует по quick-eval, сокращаем precision preview до 1 кандидата"
        )

    for candidate in candidate_results:
        too_close = any(abs(float(candidate['focal_px']) - float(existing['focal_px'])) < 75.0 for existing in preview_candidates)
        if too_close:
            continue
        _append_preview_candidate(candidate)
        if len(preview_candidates) >= preview_limit:
            break

    if len(preview_candidates) < preview_limit:
        for candidate in candidate_results:
            _append_preview_candidate(candidate)
            if len(preview_candidates) >= preview_limit:
                break

    preview_candidates = preview_candidates[:min(preview_limit, len(preview_candidates))]
    if len(preview_candidates) > 1:
        print(
            "Project-level focal sweep: "
            f"precision preview для {len(preview_candidates)} лучших гипотез"
        )

        best_preview = None
        preview_results = []
        for candidate in preview_candidates:
            preview_result = _evaluate_project_level_focal_hypothesis(
                calib_data,
                focal_px=candidate['focal_px'],
                min_points_for_camera=min_points_for_camera,
                ransac_threshold=ransac_threshold,
                confidence=confidence,
                max_attempts=max_attempts,
                focal_range=focal_range,
                force_same_focal=force_same_focal,
                fixed_pair=candidate['pair'],
                precision_preview_rounds=3,
                precision_targets=precision_targets,
            )
            if preview_result is None:
                continue

            achieved_targets = int(preview_result['mean_error'] <= precision_targets['mean'])
            achieved_targets += int(preview_result['p95'] <= precision_targets['p95'])
            achieved_targets += int(preview_result['max_error'] <= precision_targets['max'])
            effective_hits = achieved_targets - int(preview_result.get('sparse_penalty_steps', 0))

            precision_penalty = (
                max(0.0, preview_result['mean_error'] - precision_targets['mean']) / max(precision_targets['mean'], 1e-6) +
                max(0.0, preview_result['p95'] - precision_targets['p95']) / max(precision_targets['p95'], 1e-6) +
                max(0.0, preview_result['max_error'] - precision_targets['max']) / max(precision_targets['max'], 1e-6)
            )
            preview_ranking_key = (
                preview_result['cameras'],
                achieved_targets,
                -precision_penalty,
                effective_hits,
                preview_result.get('planarity_score', 1.0),
                -preview_result.get('planar_ratio', 0.0),
                preview_result.get('density_floor_score', 0.0),
                preview_result.get('point_coverage', 0.0),
                preview_result.get('observation_coverage', 0.0),
                preview_result.get('track_depth_score', 0.0),
                preview_result['quality_score'],
                preview_result['points'],
                -preview_result['mean_error'],
                -preview_result['p95'],
                -preview_result['max_error'],
            )
            preview_result['preview_ranking_key'] = preview_ranking_key

            print(
                f"    - fx~{preview_result['focal_px']:.0f}: "
                f"mean={preview_result['mean_error']:.2f}px, "
                f"p95={preview_result['p95']:.2f}px, "
                f"max={preview_result['max_error']:.2f}px, "
                f"points={preview_result['points']}, hits={achieved_targets}/3, "
                f"coverage={preview_result.get('point_coverage', 0.0):.2f}"
            )

            preview_results.append(preview_result)
            if best_preview is None or preview_ranking_key > best_preview['preview_ranking_key']:
                best_preview = preview_result

        if low_confidence_focal_estimate and preview_results:
            fallback_preview = min(
                preview_results,
                key=lambda item: abs(float(item['focal_px']) - float(fallback_focal_px))
            )
            if best_preview is not None:
                fallback_close_enough = (
                    fallback_preview['cameras'] >= best_preview['cameras'] and
                    fallback_preview['points'] >= best_preview['points'] - 6 and
                    fallback_preview['mean_error'] <= best_preview['mean_error'] + 0.08 and
                    fallback_preview['p95'] <= best_preview['p95'] + 0.15 and
                    fallback_preview['max_error'] <= best_preview['max_error'] + 0.40
                )
                if fallback_close_enough:
                    print(
                        "Project-level focal sweep: "
                        f"оценка image-based слабая (confidence={focal_estimate_confidence:.2f}), "
                        f"fallback fx~{fallback_preview['focal_px']:.0f} достаточно близок по preview, "
                        "пропускаем полный preview"
                    )
                    best_result = fallback_preview
                    best_result['preview_ranking_key'] = fallback_preview.get('preview_ranking_key')
                    return best_result

        preview_results.sort(key=lambda item: item['preview_ranking_key'], reverse=True)
        if len(preview_results) >= 2:
            first_preview = preview_results[0]
            second_preview = preview_results[1]
            ambiguous_preview = (
                first_preview['preview_ranking_key'][0] == second_preview['preview_ranking_key'][0] and
                abs(first_preview['mean_error'] - second_preview['mean_error']) <= 0.12 and
                abs(first_preview['p95'] - second_preview['p95']) <= 0.20
            )
            if ambiguous_preview:
                print("Project-level focal sweep: частичный preview неоднозначен, запускаем полный preview для 2 лучших гипотез")
                best_full_preview = None
                full_preview_candidates = [first_preview, second_preview]
                primary_full_preview_anchor = current_focal
                if isinstance(initial_focal_estimate, dict):
                    focal_estimate_confidence = float(initial_focal_estimate.get('confidence', 0.0))
                    fallback_anchor = initial_focal_estimate.get('fallback_focal_px')
                    image_anchor = initial_focal_estimate.get('focal_px')
                    if focal_estimate_confidence < 0.5 and fallback_anchor is not None:
                        primary_full_preview_anchor = float(fallback_anchor)
                    elif image_anchor is not None:
                        primary_full_preview_anchor = float(image_anchor)
                    elif fallback_anchor is not None:
                        primary_full_preview_anchor = float(fallback_anchor)

                full_preview_candidates.sort(
                    key=lambda item: (
                        abs(float(item['focal_px']) - float(primary_full_preview_anchor)),
                        -float(item['quality_score']),
                        -float(item['points']),
                    )
                )

                for candidate in full_preview_candidates:
                    full_preview_result = _evaluate_project_level_focal_hypothesis(
                        calib_data,
                        focal_px=candidate['focal_px'],
                        min_points_for_camera=min_points_for_camera,
                        ransac_threshold=ransac_threshold,
                        confidence=confidence,
                        max_attempts=max_attempts,
                        focal_range=focal_range,
                        force_same_focal=force_same_focal,
                        fixed_pair=candidate['pair'],
                        full_pipeline_preview=True,
                    )
                    if full_preview_result is None:
                        continue

                    achieved_targets = int(full_preview_result['mean_error'] <= precision_targets['mean'])
                    achieved_targets += int(full_preview_result['p95'] <= precision_targets['p95'])
                    achieved_targets += int(full_preview_result['max_error'] <= precision_targets['max'])

                    precision_penalty = (
                        max(0.0, full_preview_result['mean_error'] - precision_targets['mean']) / max(precision_targets['mean'], 1e-6) +
                        max(0.0, full_preview_result['p95'] - precision_targets['p95']) / max(precision_targets['p95'], 1e-6) +
                        max(0.0, full_preview_result['max_error'] - precision_targets['max']) / max(precision_targets['max'], 1e-6)
                    )
                    full_preview_result['preview_ranking_key'] = (
                        full_preview_result['cameras'],
                        achieved_targets,
                        -precision_penalty,
                        achieved_targets - int(full_preview_result.get('sparse_penalty_steps', 0)),
                        full_preview_result.get('planarity_score', 1.0),
                        -full_preview_result.get('planar_ratio', 0.0),
                        full_preview_result.get('density_floor_score', 0.0),
                        full_preview_result.get('point_coverage', 0.0),
                        full_preview_result.get('observation_coverage', 0.0),
                        full_preview_result.get('track_depth_score', 0.0),
                        full_preview_result['quality_score'],
                        full_preview_result['points'],
                        -full_preview_result['mean_error'],
                        -full_preview_result['p95'],
                        -full_preview_result['max_error'],
                    )
                    print(
                        f"    * FULL fx~{full_preview_result['focal_px']:.0f}: "
                        f"mean={full_preview_result['mean_error']:.2f}px, "
                        f"p95={full_preview_result['p95']:.2f}px, "
                        f"max={full_preview_result['max_error']:.2f}px, "
                        f"points={full_preview_result['points']}, hits={achieved_targets}/3, "
                        f"coverage={full_preview_result.get('point_coverage', 0.0):.2f}"
                    )
                    if best_full_preview is None or full_preview_result['preview_ranking_key'] > best_full_preview['preview_ranking_key']:
                        best_full_preview = full_preview_result

                    strong_target_hit = (
                        achieved_targets == 3 and
                        full_preview_result['mean_error'] <= precision_targets['mean'] * 0.40 and
                        full_preview_result['p95'] <= precision_targets['p95'] * 0.45 and
                        full_preview_result['max_error'] <= precision_targets['max'] * 0.50
                    )
                    if strong_target_hit:
                        print(
                            "Project-level focal sweep: "
                            f"гипотеза fx~{full_preview_result['focal_px']:.0f} уверенно достигла целевой точности, "
                            "останавливаем полный preview досрочно"
                        )
                        break

                if best_full_preview is not None:
                    best_preview = best_full_preview

        if best_preview is not None:
            best_result = best_preview

    if best_result is not None and best_result.get('final_calib_data') is None:
        preview_density_floor = max(
            int(best_result.get('density_floor', 0)) + 2,
            int(math.ceil(float(total_project_points) * 0.28)),
        )
        low_density_preview = (
            best_result['cameras'] >= total_project_cameras and
            (
                best_result['points'] < preview_density_floor or
                best_result.get('point_coverage', 0.0) < 0.24 or
                best_result.get('observation_coverage', 0.0) < 0.20
            )
        )
        if low_density_preview:
            print(
                "Project-level focal sweep: "
                f"лучший quick-preview для fx~{best_result['focal_px']:.0f} все еще разрежен, "
                "проверяем полный preview этой гипотезы"
            )
            full_preview_result = _evaluate_project_level_focal_hypothesis(
                calib_data,
                focal_px=best_result['focal_px'],
                min_points_for_camera=min_points_for_camera,
                ransac_threshold=ransac_threshold,
                confidence=confidence,
                max_attempts=max_attempts,
                focal_range=focal_range,
                force_same_focal=force_same_focal,
                fixed_pair=best_result['pair'],
                full_pipeline_preview=True,
            )
            if full_preview_result is not None:
                density_gain = int(full_preview_result['points']) - int(best_result['points'])
                safe_full_preview_upgrade = (
                    full_preview_result['cameras'] >= best_result['cameras'] and
                    density_gain >= 2 and
                    full_preview_result['mean_error'] <= best_result['mean_error'] + 0.10 and
                    full_preview_result['p95'] <= best_result['p95'] + 0.25 and
                    full_preview_result['max_error'] <= best_result['max_error'] + 0.40
                )
                if safe_full_preview_upgrade:
                    full_preview_result['selection_mode'] = 'quick_then_full_density_upgrade'
                    print(
                        "Project-level focal sweep: "
                        f"полный preview для fx~{full_preview_result['focal_px']:.0f} улучшил плотность "
                        f"({best_result['points']} -> {full_preview_result['points']} точек), "
                        "используем его вместо quick-preview"
                    )
                    best_result = full_preview_result
                else:
                    print(
                        "Project-level focal sweep: "
                        f"полный preview для fx~{full_preview_result['focal_px']:.0f} "
                        "не дал безопасного выигрыша по плотности"
                    )

    print(
        "Project-level focal sweep: выбрана гипотеза "
        f"fx~{best_result['focal_px']:.0f}px, pair={best_result['pair'][0]}-{best_result['pair'][1]}, "
        f"cameras={best_result['cameras']}, points={best_result['points']}, "
        f"mean={best_result['mean_error']:.2f}px, p95={best_result['p95']:.2f}px, "
        f"max={best_result['max_error']:.2f}px, planar={best_result.get('planar_ratio', 0.0):.3f}"
    )
    return best_result


def _quick_validate_low_confidence_image_focal(
    calib_data,
    image_focal_px,
    min_points_for_camera,
    ransac_threshold,
    confidence,
    max_attempts,
    focal_range,
    force_same_focal=False,
    fixed_pair=None,
):
    if image_focal_px is None:
        return None

    result = _evaluate_project_level_focal_hypothesis(
        calib_data,
        focal_px=float(image_focal_px),
        min_points_for_camera=min_points_for_camera,
        ransac_threshold=ransac_threshold,
        confidence=confidence,
        max_attempts=max_attempts,
        focal_range=focal_range,
        force_same_focal=force_same_focal,
        fixed_pair=fixed_pair,
    )
    if result is None:
        return None

    precision_targets = {
        'mean': float(calib_data.get('precision_target_mean_px', 0.5)),
        'p95': float(calib_data.get('precision_target_p95_px', 1.0)),
        'max': float(calib_data.get('precision_target_max_px', 1.5)),
    }
    total_camera_count = len(calib_data.get('camera_points', {}))
    min_cameras = max(5, min(total_camera_count, 6))
    min_points = max(18, min_points_for_camera * 4)
    plausible = (
        result['cameras'] >= min_cameras and
        result['points'] >= min_points and
        result['mean_error'] <= max(precision_targets['mean'] * 4.0, 2.0) and
        result['p95'] <= max(precision_targets['p95'] * 5.0, 5.0) and
        result['max_error'] <= max(precision_targets['max'] * 12.0, 12.0)
    )
    result['plausible'] = bool(plausible)
    return result


def _finalize_reconstruction(
    calib_data,
    start_time,
    min_points_for_camera,
    max_bundle_iterations,
    focal_range,
    force_same_focal,
    ransac_threshold=8.0,
    confidence=0.99,
    max_attempts=2,
    progress_callback=None,
):
    def _report_progress(progress_value, status_text):
        if progress_callback is None:
            return
        try:
            progress_callback(progress_value, status_text)
        except Exception:
            pass

    preserved_secondary_seed_points = copy.deepcopy(calib_data.get('_secondary_seed_points_3d', {}))
    calib_data['secondary_points_3d'] = {}
    calib_data['_secondary_seed_points_3d'] = preserved_secondary_seed_points
    preview_mode = bool(calib_data.get('_project_level_preview_mode', False))
    _capture_point_drift_stage(
        calib_data,
        "finalize_start",
        {
            "preview_mode": bool(preview_mode),
            "precision_cleanup_enabled": bool(calib_data.get('precision_cleanup_enabled', True)),
        },
    )

    _report_progress(83.0, "Фильтрация 3D-выбросов...")
    print("\nФинальная оптимизация...")
    try:
        removed_points, _, _, _ = triangulation.filter_outliers_by_reprojection_error(
            calib_data,
            absolute_threshold=12.0,
            sigma_multiplier=2.5,
            mad_multiplier=3.0
        )
        if removed_points > 0:
            print("После удаления 3D-выбросов выполняем очистку наблюдений...")
    except Exception as e:
        print(f"Фильтрация 3D-выбросов пропущена из-за ошибки: {str(e)}")

    if not preview_mode:
        try:
            _report_progress(84.0, "Уточнение камер по конфликтным трекам...")
            refine_cameras_from_rejected_single_view_tracks(
                calib_data,
                max_cameras=2,
                max_tracks_per_camera=6,
                min_total_correspondences=max(min_points_for_camera + 3, 8),
            )
        except Exception as e:
            print(f"Уточнение камер по конфликтным трекам пропущено из-за ошибки: {str(e)}")

    try:
        _report_progress(84.5, "Локальная очистка камер...")
        refine_high_error_cameras(
            calib_data,
            max_cameras=2,
            min_observations_per_camera=max(min_points_for_camera + 2, 6),
            min_track_length=3,
        )
    except Exception as e:
        print(f"Локальная доочистка камер пропущена из-за ошибки: {str(e)}")

    try:
        _report_progress(85.0, "Фильтрация наблюдений...")
        filter_observations_by_reprojection_error(
            calib_data,
            absolute_threshold=8.0,
            sigma_multiplier=2.5,
            mad_multiplier=3.0,
            min_observations_per_camera=max(min_points_for_camera + 1, 5),
            min_track_length=2,
            force_allow=True,
        )
    except Exception as e:
        print(f"Фильтрация наблюдений пропущена из-за ошибки: {str(e)}")

    precision_cleanup_enabled = bool(calib_data.get('precision_cleanup_enabled', True))
    initial_refine_iterations = max_bundle_iterations
    if precision_cleanup_enabled:
        initial_refine_iterations = min(int(max_bundle_iterations), 1)

    _report_progress(86.0, "Оптимизация реконструкции...")
    refine_reconstruction(
        calib_data,
        max_iterations=initial_refine_iterations,
        focal_range=focal_range,
        force_same_focal=force_same_focal,
        optimize_intrinsics=not precision_cleanup_enabled,
        optimize_distortion=not precision_cleanup_enabled,
        progress_callback=progress_callback,
        progress_range=(86.0, 88.0)
    )
    _capture_point_drift_stage(calib_data, "after_initial_refine")

    try:
        existing_cameras = set(str(camera_id) for camera_id in calib_data.get('cameras', {}).keys())
        total_camera_count = len(calib_data.get('camera_points', {}))
        if len(existing_cameras) < total_camera_count:
            _report_progress(88.0, "Повторная попытка добавить оставшиеся камеры...")
            print("Повторная попытка добавить оставшиеся камеры после уточнения геометрии...")
            expanded_cameras = _expand_reconstruction_frontier(
                calib_data,
                initialized_cameras=existing_cameras,
                min_points_for_camera=min_points_for_camera,
                ransac_threshold=ransac_threshold,
                confidence=confidence,
                max_attempts=max(1, int(max_attempts)),
                progress_callback=progress_callback,
                progress_range=(88.0, 89.0),
            )
            if len(expanded_cameras) > len(existing_cameras):
                print("После позднего добора камер выполняем дополнительный шаг оптимизации...")
                refine_reconstruction(
                    calib_data,
                    max_iterations=1,
                    focal_range=focal_range,
                    force_same_focal=force_same_focal,
                    optimize_intrinsics=False,
                    optimize_distortion=False,
                    progress_callback=progress_callback,
                    progress_range=(89.0, 89.2)
                )
                _capture_point_drift_stage(
                    calib_data,
                    "after_late_camera_expand",
                    {
                        "added_cameras": int(len(expanded_cameras) - len(existing_cameras)),
                    },
                )
    except Exception as e:
        print(f"Поздний добор камер пропущен из-за ошибки: {str(e)}")

    try:
        _report_progress(89.25, "Достройка оставшихся треков...")
        backfilled_points = triangulation.triangulate_remaining_tracks(
            calib_data['points_3d'],
            calib_data['cameras'],
            calib_data['camera_points'],
            calib_data['K'],
            calib_data.get('dist_coeffs'),
            min_track_length=2,
            strict_track_consistency=bool(calib_data.get('strict_track_consistency', True)),
            debug_logging=bool(calib_data.get('debug_logging', False)),
            observation_confidences=calib_data.get('observation_confidences'),
            line_support_data=calib_data.get('line_support_data'),
        )
        if backfilled_points:
            calib_data['points_3d'].update(backfilled_points)
            print("После global track backfill выполняем дополнительный шаг оптимизации...")
            refine_reconstruction(
                calib_data,
                max_iterations=1,
                focal_range=focal_range,
                force_same_focal=force_same_focal,
                optimize_intrinsics=False,
                optimize_distortion=False,
                progress_callback=progress_callback,
                progress_range=(89.28, 89.40)
            )
            _capture_point_drift_stage(
                calib_data,
                "after_global_backfill",
                {
                    "backfilled_points": int(len(backfilled_points)),
                },
            )
    except Exception as e:
        print(f"Global track backfill пропущен из-за ошибки: {str(e)}")

    if not precision_cleanup_enabled:
        try:
            _report_progress(88.2, "Локальное уточнение поз камер...")
            refined_camera_poses = refine_high_error_camera_poses(
                calib_data,
                max_cameras=2,
                min_observations_per_camera=max(min_points_for_camera + 4, 8),
            )
            if refined_camera_poses > 0:
                print("После локального уточнения поз выполняем дополнительный шаг оптимизации...")
                refine_reconstruction(
                    calib_data,
                    max_iterations=1,
                    focal_range=focal_range,
                    force_same_focal=force_same_focal,
                    progress_callback=progress_callback,
                    progress_range=(88.3, 88.5)
                )
        except Exception as e:
            print(f"Локальное уточнение поз камер пропущено из-за ошибки: {str(e)}")
        try:
            _report_progress(88.6, "Повторная триангуляция проблемных точек...")
            retriangulated_points = triangulation.retriangulate_high_error_points(
                calib_data['points_3d'],
                calib_data['cameras'],
                calib_data['camera_points'],
                calib_data['K'],
                calib_data.get('dist_coeffs'),
                max_points=4,
                min_track_length=3,
                debug_logging=bool(calib_data.get('debug_logging', False)),
                observation_confidences=calib_data.get('observation_confidences'),
                line_support_data=calib_data.get('line_support_data'),
                camera_intrinsics=_collect_camera_intrinsics_map(calib_data),
            )
            if retriangulated_points > 0:
                print("После point retriangulation выполняем дополнительный шаг оптимизации...")
                refine_reconstruction(
                    calib_data,
                    max_iterations=1,
                    focal_range=focal_range,
                    force_same_focal=force_same_focal,
                    progress_callback=progress_callback,
                    progress_range=(88.7, 88.9)
                )
        except Exception as e:
            print(f"Point retriangulation пропущен из-за ошибки: {str(e)}")
        try:
            _report_progress(89.0, "Финальная очистка хвоста ошибок...")
            tail_removed = filter_extreme_tail_observations(
                calib_data,
                max_removals=3,
                min_observations_per_camera=max(min_points_for_camera + 2, 6),
                min_track_length=3,
                force_allow=True,
            )
            if tail_removed > 0:
                print("После post-BA tail cleanup выполняем дополнительный шаг оптимизации...")
                refine_reconstruction(
                    calib_data,
                    max_iterations=1,
                    focal_range=focal_range,
                    force_same_focal=force_same_focal,
                    progress_callback=progress_callback,
                    progress_range=(89.1, 89.3)
                )
        except Exception as e:
            print(f"Post-BA tail cleanup пропущен из-за ошибки: {str(e)}")
    try:
        if bool(calib_data.get('precision_cleanup_enabled', True)):
            merged_secondary_seed_points = copy.deepcopy(calib_data.get('_secondary_seed_points_3d', {}))
            for point_id, point_3d in (calib_data.get('points_3d') or {}).items():
                merged_secondary_seed_points[point_id] = copy.deepcopy(point_3d)
            calib_data['_secondary_seed_points_3d'] = merged_secondary_seed_points
            _report_progress(89.35, "Precision-first cleanup...")
            run_precision_first_cleanup(
                calib_data,
                target_mean=float(calib_data.get('precision_target_mean_px', 0.5)),
                target_p95=float(calib_data.get('precision_target_p95_px', 1.0)),
                target_max=float(calib_data.get('precision_target_max_px', 1.5)),
                max_rounds=int(calib_data.get('precision_cleanup_rounds', 4)),
                min_observations_per_camera=max(min_points_for_camera + 2, 6),
                min_track_length=3,
                min_points=max(12, min_points_for_camera * 3),
                focal_range=focal_range,
                force_same_focal=force_same_focal,
                progress_callback=progress_callback,
                progress_range=(89.35, 89.55),
            )
            _capture_point_drift_stage(calib_data, "after_precision_cleanup")
    except Exception as e:
        print(f"Precision-first cleanup пропущен из-за ошибки: {str(e)}")

    try:
        if bool(calib_data.get('strict_track_consistency', True)):
            _report_progress(89.40, "Финальная strict-проверка треков...")
            strict_min_points_remaining = max(
                8,
                min(
                    max(max(12, min_points_for_camera * 3) - 1, 8),
                    max(len(calib_data.get('points_3d', {})) - 1, 8)
                )
            )
            strict_removed_points = triangulation.remove_inconsistent_full_tracks(
                calib_data['points_3d'],
                calib_data['cameras'],
                calib_data['camera_points'],
                calib_data['K'],
                calib_data.get('dist_coeffs'),
                target_mean=float(calib_data.get('precision_target_mean_px', 0.5)),
                target_p95=float(calib_data.get('precision_target_p95_px', 1.0)),
                target_max=float(calib_data.get('precision_target_max_px', 1.5)),
                min_track_length=3,
                min_points_remaining=strict_min_points_remaining,
                observation_confidences=calib_data.get('observation_confidences'),
                line_support_data=calib_data.get('line_support_data'),
                camera_intrinsics=_collect_camera_intrinsics_map(calib_data),
            )
            if strict_removed_points > 0:
                print("После финальной strict-проверки треков выполняем дополнительный шаг оптимизации...")
                refine_reconstruction(
                    calib_data,
                    max_iterations=1,
                    focal_range=focal_range,
                    force_same_focal=force_same_focal,
                    optimize_intrinsics=False,
                    optimize_distortion=False,
                    progress_callback=progress_callback,
                    progress_range=(89.42, 89.45)
                )
            _capture_point_drift_stage(
                calib_data,
                "after_strict_full_track",
                {
                    "strict_removed_points": int(strict_removed_points),
                },
            )
    except Exception as e:
        print(f"Финальная strict-проверка треков пропущена из-за ошибки: {str(e)}")

    if preview_mode:
        print("Project-level preview mode: пропускаем secondary point cloud и per-camera focal")
    else:
        # --- Fix 1: Переоценка поз камер после precision cleanup ---
        try:
            _report_progress(89.39, "Переоценка поз камер после cleanup...")
            post_cleanup_pose_refined = refine_high_error_camera_poses(
                calib_data,
                max_cameras=3,
                min_observations_per_camera=max(min_points_for_camera + 2, 5),
            )
            if post_cleanup_pose_refined > 0:
                refine_reconstruction(
                    calib_data,
                    max_iterations=1,
                    focal_range=focal_range,
                    force_same_focal=force_same_focal,
                    optimize_intrinsics=False,
                    optimize_distortion=False,
                    progress_callback=progress_callback,
                    progress_range=(89.392, 89.398),
                )
        except Exception as e:
            print(f"Переоценка поз камер после cleanup пропущена: {str(e)}")

        # --- Восстановление наблюдений после уточнения поз ---
        try:
            _report_progress(89.395, "Восстановление наблюдений после уточнения поз...")
            recovered_observations = recover_observations_after_pose_refinement(
                calib_data,
                max_cameras=4,
                max_recoveries_per_camera=6,
            )
            if recovered_observations > 0:
                refine_reconstruction(
                    calib_data,
                    max_iterations=1,
                    focal_range=focal_range,
                    force_same_focal=force_same_focal,
                    optimize_intrinsics=False,
                    optimize_distortion=False,
                    progress_callback=progress_callback,
                    progress_range=(89.396, 89.399),
                )
            _capture_point_drift_stage(calib_data, "after_observation_recovery",
                {"recovered_observations": int(recovered_observations)})
        except Exception as e:
            print(f"Восстановление наблюдений пропущено: {str(e)}")

        # --- Fix 2: Повторное уточнение камер по отклонённым трекам ---
        try:
            _report_progress(89.40, "Повторное уточнение камер по отклонённым трекам...")
            post_cleanup_camera_refine = refine_cameras_from_rejected_single_view_tracks(
                calib_data,
                max_cameras=2,
                max_tracks_per_camera=6,
                min_total_correspondences=max(min_points_for_camera + 3, 8),
            )
            if post_cleanup_camera_refine > 0:
                refine_reconstruction(
                    calib_data,
                    max_iterations=1,
                    focal_range=focal_range,
                    force_same_focal=force_same_focal,
                    optimize_intrinsics=False,
                    optimize_distortion=False,
                    progress_callback=progress_callback,
                    progress_range=(89.401, 89.408),
                )
        except Exception as e:
            print(f"Повторное уточнение камер пропущено: {str(e)}")

        try:
            _report_progress(89.412, "Стабилизация поз по каркасу треков...")
            scaffold_stats = stabilize_reconstruction_from_pose_scaffold(
                calib_data,
                min_points_for_camera=min_points_for_camera,
                label="Pose scaffold stage",
                bundle_ftol=1e-7,
                bundle_max_nfev=320,
            )
            if not scaffold_stats.get('applied'):
                print(
                    "Pose scaffold stage пропущен: "
                    f"{scaffold_stats.get('reason', 'unknown')}"
                )
            else:
                retriangulated_after_scaffold = triangulation.retriangulate_high_error_points(
                    calib_data['points_3d'],
                    calib_data['cameras'],
                    calib_data['camera_points'],
                    calib_data['K'],
                    calib_data.get('dist_coeffs'),
                    max_points=max(6, min(18, len(calib_data.get('points_3d', {})) // 3)),
                    min_track_length=3,
                    debug_logging=bool(calib_data.get('debug_logging', False)),
                    observation_confidences=calib_data.get('observation_confidences'),
                    line_support_data=calib_data.get('line_support_data'),
                    camera_intrinsics=_collect_camera_intrinsics_map(calib_data),
                )
                if retriangulated_after_scaffold > 0:
                    print(
                        "Pose scaffold stage: "
                        f"дополнительно уточнено {retriangulated_after_scaffold} 3D-точек "
                        "при фиксированном каркасе камер"
                    )
                _capture_point_drift_stage(
                    calib_data,
                    "after_pose_scaffold",
                    {
                        "scaffold_applied": True,
                        "retriangulated_after_scaffold": int(retriangulated_after_scaffold),
                    },
                )
        except Exception as e:
            print(f"Pose scaffold stage пропущен из-за ошибки: {str(e)}")

        try:
            _report_progress(89.422, "Починка single-view конфликтов...")
            repaired_tracks = repair_single_view_conflict_observations(
                calib_data,
                target_mean=float(calib_data.get('precision_target_mean_px', 0.5)),
                target_p95=float(calib_data.get('precision_target_p95_px', 1.0)),
                target_max=float(calib_data.get('precision_target_max_px', 1.5)),
                max_candidates=6,
                max_shift_px=18.0,
            )
            if repaired_tracks > 0:
                print("После guided single-view repair повторно стабилизируем позы по каркасу...")
                stabilize_reconstruction_from_pose_scaffold(
                    calib_data,
                    min_points_for_camera=min_points_for_camera,
                    label="Pose scaffold after guided repair",
                    bundle_ftol=1e-7,
                    bundle_max_nfev=240,
                )
            _capture_point_drift_stage(
                calib_data,
                "after_guided_repair",
                {
                    "repaired_tracks": int(repaired_tracks),
                },
            )
        except Exception as e:
            print(f"Guided single-view repair пропущен из-за ошибки: {str(e)}")

        try:
            _report_progress(89.428, "Возврат full-track точек...")
            restored_points = reintegrate_backfillable_tracks(
                calib_data,
                target_mean=float(calib_data.get('precision_target_mean_px', 0.5)),
                target_p95=float(calib_data.get('precision_target_p95_px', 1.0)),
                target_max=float(calib_data.get('precision_target_max_px', 1.5)),
                focal_range=focal_range,
                force_same_focal=force_same_focal,
                max_candidates=12,
                progress_callback=progress_callback,
                progress_range=(89.428, 89.438),
            )
            if restored_points > 0:
                print("После возврата full-track точек повторно стабилизируем позы по каркасу...")
                stabilize_reconstruction_from_pose_scaffold(
                    calib_data,
                    min_points_for_camera=min_points_for_camera,
                    label="Pose scaffold after full-track re-entry",
                    bundle_ftol=1e-7,
                    bundle_max_nfev=240,
                )
            _capture_point_drift_stage(
                calib_data,
                "after_full_track_reentry",
                {
                    "restored_points": int(restored_points),
                },
            )
        except Exception as e:
            print(f"Возврат full-track точек пропущен из-за ошибки: {str(e)}")

        try:
            _report_progress(89.445, "Построение вторичного облака точек...")
            build_secondary_point_cloud(
                calib_data,
                target_mean=float(calib_data.get('precision_target_mean_px', 0.5)),
                target_p95=float(calib_data.get('precision_target_p95_px', 1.0)),
                target_max=float(calib_data.get('precision_target_max_px', 1.5)),
                max_candidates=32,
            )
            _capture_point_drift_stage(calib_data, "after_secondary_cloud")
        except Exception as e:
            print(f"Secondary point cloud пропущено из-за ошибки: {str(e)}")

        try:
            _report_progress(89.455, "Оценка focal по камерам...")
            per_camera_intrinsics, per_camera_intrinsics_stats = bundle_adjustment.optimize_individual_focals(
                calib_data['K'],
                calib_data['points_3d'],
                calib_data['cameras'],
                calib_data['camera_points'],
                calib_data.get('dist_coeffs'),
                force_same_focal=force_same_focal,
                min_observations=max(min_points_for_camera + 2, 6),
            )

            for key in [item for item in list(calib_data.keys()) if isinstance(item, str) and item.startswith('K_')]:
                del calib_data[key]
            for camera_id, K_i in per_camera_intrinsics.items():
                calib_data[f'K_{camera_id}'] = np.asarray(K_i, dtype=np.float32)

            if per_camera_intrinsics_stats:
                print("Per-camera focal refinement:")
                for stat in per_camera_intrinsics_stats:
                    status = "принят" if stat['accepted'] else "оставлен общий K"
                    print(
                        f"  - Камера {stat['camera_id']}: "
                        f"RMSE {stat['initial_rmse']:.4f}px -> {stat['final_rmse']:.4f}px, "
                        f"fx={stat['fx']:.2f}, fy={stat['fy']:.2f}, "
                        f"наблюдений={stat['count']}, {status}"
                    )
            _capture_point_drift_stage(calib_data, "after_per_camera_focal")
        except Exception as e:
            print(f"Per-camera focal refinement пропущен из-за ошибки: {str(e)}")

    _report_progress(89.5, "Подготовка итоговой статистики...")
    print("\nИтоговая статистика:")
    total_error, errors_by_point, errors_by_camera = calculate_reprojection_errors(calib_data)
    print(f"Средняя ошибка репроекции: {total_error:.4f} пикселей")

    # --- Per-camera anomaly detection ---
    if errors_by_camera and len(errors_by_camera) >= 3:
        cam_mean_errors = {}
        for cam_id, point_errors in errors_by_camera.items():
            if point_errors:
                cam_mean_errors[cam_id] = float(np.mean(point_errors))
        if cam_mean_errors:
            all_cam_means = np.array(list(cam_mean_errors.values()))
            median_cam_error = float(np.median(all_cam_means))
            for cam_id, cam_mean in sorted(cam_mean_errors.items(), key=lambda x: -x[1]):
                if median_cam_error > 0 and cam_mean > max(median_cam_error * 3.0, median_cam_error + 2.0):
                    image_name = calib_data.get('images', {}).get(str(cam_id), f"камера {cam_id}")
                    if isinstance(image_name, str) and os.path.sep in image_name:
                        image_name = os.path.basename(image_name)
                    print(
                        f"WARNING: Камера {cam_id} ({image_name}) имеет аномально "
                        f"высокую ошибку ({cam_mean:.2f}px vs медиана {median_cam_error:.2f}px). "
                        f"Возможные причины: неверные 2D-точки, иная камера/объектив, "
                        f"ошибка масштабирования."
                    )

    distribution_stats = summarize_reprojection_error_distribution(calib_data, top_k=5)
    if distribution_stats['count'] > 0:
        print(
            f"Распределение ошибок наблюдений: "
            f"median={distribution_stats['median']:.4f}px, "
            f"p90={distribution_stats['p90']:.4f}px, "
            f"p95={distribution_stats['p95']:.4f}px, "
            f"max={distribution_stats['max']:.4f}px"
        )
        if distribution_stats['worst_observations']:
            print("Худшие наблюдения:")
            for item in distribution_stats['worst_observations']:
                print(
                    f"  - Камера {item['camera_id']}, точка {item['point_id']}: "
                    f"{item['error']:.4f}px"
                )
        point_profiles = summarize_point_error_profiles(calib_data, top_k=3, min_track_length=3)
        if point_profiles:
            print("Нестабильные треки точек:")
            for profile in point_profiles[:5]:
                worst_summary = ", ".join(
                    f"{item['camera_id']}:{item['error']:.2f}px"
                    for item in profile['worst_observations']
                )
                print(
                    f"  - Точка {profile['point_id']}: "
                    f"median={profile['median']:.4f}px, "
                    f"max={profile['max']:.4f}px, "
                    f"asymmetry={profile['asymmetry']:.4f}px, "
                    f"трек={profile['count']}, "
                    f"худшие камеры [{worst_summary}]"
                )
        unreconstructed_diagnostics = calib_data.get('_unreconstructed_diagnostics_cache')
        if unreconstructed_diagnostics is None:
            unreconstructed_diagnostics = triangulation.diagnose_unreconstructed_tracks(
                calib_data['points_3d'],
                calib_data['cameras'],
                calib_data['camera_points'],
                calib_data['K'],
                calib_data.get('dist_coeffs'),
                min_track_length=2,
                strict_track_consistency=bool(calib_data.get('strict_track_consistency', True)),
                top_k=0,
                debug_logging=bool(calib_data.get('debug_logging', False)),
                observation_confidences=calib_data.get('observation_confidences'),
                line_support_data=calib_data.get('line_support_data'),
            )
            calib_data['_unreconstructed_diagnostics_cache'] = copy.deepcopy(unreconstructed_diagnostics)
        unreconstructed_diagnostics = list(unreconstructed_diagnostics[:5])
        if unreconstructed_diagnostics:
            print("Не вошедшие треки точек:")
            for item in unreconstructed_diagnostics:
                pair_label = (
                    f"{item['best_pair'][0]}-{item['best_pair'][1]}"
                    if item.get('best_pair') is not None else "n/a"
                )
                mean_label = (
                    f"{item['best_mean']:.2f}px"
                    if item.get('best_mean') is not None else "n/a"
                )
                max_label = (
                    f"{item['best_max']:.2f}px"
                    if item.get('best_max') is not None else "n/a"
                )
                baseline_label = (
                    f"{item['baseline']:.3f}"
                    if item.get('baseline') is not None else "n/a"
                )
                parallax_label = (
                    f"{item['min_parallax_deg']:.2f}-{item['max_parallax_deg']:.2f}deg"
                    if item.get('min_parallax_deg') is not None and item.get('max_parallax_deg') is not None else "n/a"
                )
                asymmetry_label = (
                    f"{item['error_asymmetry']:.2f}px"
                    if item.get('error_asymmetry') is not None else "n/a"
                )
                worst_summary = ", ".join(
                    f"{obs['camera_id']}:{obs['error']:.2f}px"
                    for obs in item.get('worst_observations', [])
                ) or "n/a"
                print(
                    f"  - Точка {item['point_id']}: "
                    f"трек={item['track_length']}, "
                    f"reason={item['reason']}, "
                    f"class={item.get('conflict_class') or 'n/a'}, "
                    f"pair_success={item['pair_success_count']}, "
                    f"full_accept={item['full_accept_count']}, "
                    f"subset_accept={item['subset_accept_count']}, "
                    f"strict_blocked={item['strict_blocked_count']}, "
                    f"best_pair={pair_label}, "
                    f"mode={item.get('best_mode') or 'n/a'}, "
                    f"mean={mean_label}, "
                    f"max={max_label}, "
                    f"baseline={baseline_label}, "
                    f"parallax={parallax_label}, "
                    f"asymmetry={asymmetry_label}, "
                    f"worst=[{worst_summary}]"
                )

    end_time = time.time()
    print(f"Полная реконструкция завершена за {end_time - start_time:.2f} секунд")
    print(f"Успешно добавлено {len(calib_data['cameras'])} камер")
    print(f"Реконструировано {len(calib_data['points_3d'])} 3D точек")
    if calib_data.get('secondary_points_3d'):
        print(f"Вторичное облако: {len(calib_data['secondary_points_3d'])} дополнительных 3D точек")

    _report_progress(89.8, "Реконструкция solver завершена")
    _capture_point_drift_stage(calib_data, "final")

    return True

def initialize_reconstruction(calib_data, camera_id1, camera_id2):
    """
    Инициализирует реконструкцию с начальной парой камер.
    
    Args:
        calib_data: Данные калибровки
        camera_id1: ID первой камеры
        camera_id2: ID второй камеры
    
    Returns:
        bool: True, если инициализация успешна, иначе False
    """
    print(f"\nИнициализация реконструкции с камерами {camera_id1} и {camera_id2}")
    
    try:
        # Получаем точки для обеих камер
        # Используем строковые ключи, так как в тесте они создаются как строки
        points_2d_1 = calib_data['camera_points'].get(str(camera_id1), {})
        points_2d_2 = calib_data['camera_points'].get(str(camera_id2), {})
        # Находим общие точки между камерами
        common_points = sorted(set(points_2d_1.keys()) & set(points_2d_2.keys()), key=_stable_sort_key)
        if len(common_points) < 8:
            print(f"Недостаточно общих точек между камерами {camera_id1} и {camera_id2}: {len(common_points)} (минимум 8)")
            return False
        
        pts1 = np.array([points_2d_1[point_id] for point_id in common_points])
        pts2 = np.array([points_2d_2[point_id] for point_id in common_points])
        point_ids = list(common_points)
        
        print(f"Используем {len(pts1)} общих точек для инициализации")
        print(f"Используемая матрица калибровки K: {calib_data['K']}")
        
        # Оцениваем относительную позу камер
        print(f"Оценка позы камер на основе {len(pts1)} общих точек...")
        result = camera_pose.estimate_relative_pose(pts1, pts2, calib_data['K'], calib_data.get('dist_coeffs'))
        if result is None:
            print("Не удалось оценить позу камер")
            return False
            
        R, t, mask, points_3d = result
        
        print(f"Найдено {np.sum(mask) if mask is not None else len(pts1)} инлаеров из {len(pts1)} точек")
        
        # Устанавливаем первую камеру в начало координат
        calib_data['cameras'][str(camera_id1)] = (np.eye(3), np.zeros((3, 1)))
        calib_data['cameras'][str(camera_id2)] = (R, t)
        
        # Сохраняем 3D точки. estimate_relative_pose возвращает только инлаеры,
        # поэтому нужно сопоставить их исходным point_id по маске.
        if points_3d is not None:
            points_3d = np.asarray(points_3d, dtype=np.float32)

            # Приводим к формату (N, 3), т.к. cv2.triangulatePoints обычно возвращает (3, N).
            if points_3d.ndim == 2 and points_3d.shape[0] == 3 and points_3d.shape[1] != 3:
                points_3d = points_3d.T
            elif points_3d.ndim == 1 and points_3d.size == 3:
                points_3d = points_3d.reshape(1, 3)

            if points_3d.ndim == 2 and points_3d.shape[1] >= 3 and len(points_3d) > 0:
                if mask is None:
                    inlier_indices = np.arange(min(len(point_ids), len(points_3d)))
                else:
                    inlier_indices = np.where(np.asarray(mask).ravel() > 0)[0]

                saved_points = 0
                for points_idx, original_idx in enumerate(inlier_indices):
                    if points_idx >= len(points_3d) or original_idx >= len(point_ids):
                        break
                    calib_data['points_3d'][point_ids[original_idx]] = points_3d[points_idx, :3]
                    saved_points += 1

                sanitization_stats = triangulation.sanitize_points_for_camera(
                    calib_data['points_3d'],
                    calib_data['cameras'],
                    calib_data['camera_points'],
                    calib_data['K'],
                    calib_data.get('dist_coeffs'),
                    str(camera_id2),
                    debug_logging=bool(calib_data.get('debug_logging', False)),
                    observation_confidences=calib_data.get('observation_confidences'),
                    secondary_seed_points=calib_data.setdefault('_secondary_seed_points_3d', {}),
                    line_support_data=calib_data.get('line_support_data'),
                    soft_mode=True,
                    camera_intrinsics=_collect_camera_intrinsics_map(calib_data),
                )
                print(f"Сохранено стартовых 3D точек: {saved_points}")
                if (
                    sanitization_stats['removed_points'] or
                    sanitization_stats['removed_observations'] or
                    sanitization_stats.get('downweighted_observations', 0)
                ):
                    print(
                        f"Ранняя проверка стартовых точек: удалено точек {sanitization_stats['removed_points']}, "
                        f"удалено наблюдений {sanitization_stats['removed_observations']}, "
                        f"downweight наблюдений {sanitization_stats.get('downweighted_observations', 0)}, "
                        f"уточнено точек {sanitization_stats['refined_points']}"
                    )
                local_strict_stats = triangulation.prune_focus_conflicting_tracks(
                    calib_data['points_3d'],
                    calib_data['cameras'],
                    calib_data['camera_points'],
                    calib_data['K'],
                    calib_data.get('dist_coeffs'),
                    str(camera_id2),
                    min_track_length=4,
                    strict_track_consistency=bool(calib_data.get('strict_track_consistency', True)),
                    debug_logging=bool(calib_data.get('debug_logging', False)),
                    observation_confidences=calib_data.get('observation_confidences'),
                    secondary_seed_points=calib_data.setdefault('_secondary_seed_points_3d', {}),
                    line_support_data=calib_data.get('line_support_data'),
                    soft_mode=True,
                )
                if (
                    local_strict_stats['removed_points'] or
                    local_strict_stats['removed_observations'] or
                    local_strict_stats.get('downweighted_observations', 0)
                ):
                    print(
                        f"Локальный strict-gate стартовой пары: удалено точек {local_strict_stats['removed_points']}, "
                        f"удалено наблюдений {local_strict_stats['removed_observations']}, "
                        f"downweight наблюдений {local_strict_stats.get('downweighted_observations', 0)}"
                    )
        
        # Сохраняем камеры, использованные в реконструкции
        calib_data['reconstruction_ids'] = [str(camera_id1), str(camera_id2)]
        print(f"Инициализация реконструкции успешна: {len(calib_data['points_3d'])} точек")
        _capture_point_drift_stage(
            calib_data,
            "after_initial_pair",
            {
                "camera_ids": [str(camera_id1), str(camera_id2)],
            },
        )

        # --- Fix 5: BA сразу после инициализации стартовой пары ---
        # Стабилизируем геометрию стартовой пары до добавления других камер.
        # Без этого шага все последующие PnP-решения опираются на «сырые» точки
        # от Essential matrix, что вызывает накопление дрейфа.
        if len(calib_data['points_3d']) >= 4:
            try:
                ba_observation_confidences = _build_quality_aware_observation_confidences(
                    calib_data,
                    label="Fix 5",
                )
                points_3d_opt, cameras_opt = bundle_adjustment.bundle_adjust_step(
                    calib_data['points_3d'],
                    calib_data['cameras'],
                    calib_data['camera_points'],
                    calib_data['K'],
                    calib_data.get('dist_coeffs'),
                    ftol=1e-6,
                    max_nfev=200,
                    debug_logging=False,
                    observation_confidences=ba_observation_confidences,
                    camera_intrinsics=_collect_camera_intrinsics_map(calib_data),
                )
                calib_data['points_3d'] = points_3d_opt
                calib_data['cameras'] = cameras_opt
                print(f"Fix 5: BA стартовой пары выполнен ({len(calib_data['points_3d'])} точек)")
                _capture_point_drift_stage(
                    calib_data,
                    "after_initial_pair_ba",
                    {
                        "camera_ids": [str(camera_id1), str(camera_id2)],
                    },
                )
            except Exception as e_ba:
                print(f"Fix 5: BA стартовой пары пропущен: {e_ba}")

        return True
        
    except Exception as e:
        print(f"Ошибка при инициализации реконструкции: {str(e)}")
        traceback.print_exc()
        return False

def add_camera_to_reconstruction(calib_data, camera_id, min_points=4, ransac_threshold=8.0, confidence=0.99):
    """
    Добавляет камеру к существующей реконструкции.
    
    Args:
        calib_data: Данные калибровки
        camera_id: ID камеры для добавления
        min_points: Минимальное количество общих точек для добавления камеры
        ransac_threshold: Порог для RANSAC при оценке позы камеры
        confidence: Уровень доверия для RANSAC
    
    Returns:
        bool: True, если камера успешно добавлена, иначе False
    """
    try:
        debug_logging = bool(calib_data.get('debug_logging', False))
        multiview_refine_mode = str(calib_data.get('_multiview_refine_mode', 'full') or 'full')
        fallback_result = None
        state_snapshot = _snapshot_calibration_state(calib_data)
        provisional_min_confidence = _get_soft_gate_min_confidence(calib_data)
        before_total_error, _, _ = calculate_reprojection_errors(
            calib_data,
            min_observation_confidence=provisional_min_confidence,
        )
        before_distribution = summarize_reprojection_error_distribution(
            calib_data,
            top_k=0,
            min_observation_confidence=provisional_min_confidence,
        )
        _capture_point_drift_stage(
            calib_data,
            f"camera_{camera_id}_start",
            {
                "camera_id": str(camera_id),
                "support_before": int(before_distribution.get('count', 0)),
            },
        )

        def _restore_snapshot():
            _restore_calibration_state(calib_data, state_snapshot)

        def _project_reprojection_stats(R_candidate, t_candidate, point_ids_subset, min_required=None):
            if not point_ids_subset:
                return None

            if min_required is None:
                min_required = min_points
            min_required = max(1, int(min_required))

            obj_points = []
            img_points = []
            camera_points_2d_local = calib_data['camera_points'][str(camera_id)]
            for point_id in point_ids_subset:
                if point_id not in calib_data['points_3d'] or point_id not in camera_points_2d_local:
                    continue
                obj_points.append(np.asarray(calib_data['points_3d'][point_id], dtype=np.float32).reshape(3))
                img_points.append(np.asarray(camera_points_2d_local[point_id], dtype=np.float32).reshape(2))

            if len(obj_points) < min_required:
                return None

            obj_points = np.asarray(obj_points, dtype=np.float32)
            img_points = np.asarray(img_points, dtype=np.float32)
            rvec_candidate, _ = cv2.Rodrigues(np.asarray(R_candidate, dtype=np.float32))
            projected_points, _ = cv2.projectPoints(
                obj_points,
                rvec_candidate,
                np.asarray(t_candidate, dtype=np.float32).reshape(3, 1),
                calib_data['K'],
                calib_data.get('dist_coeffs')
            )
            reprojection_errors = np.linalg.norm(
                projected_points.reshape(-1, 2) - img_points,
                axis=1
            )
            point_depths = (
                np.asarray(R_candidate, dtype=np.float32) @ obj_points.T
                + np.asarray(t_candidate, dtype=np.float32).reshape(3, 1)
            )[2]
            front_ratio = float(np.mean(point_depths > 0.01)) if len(point_depths) > 0 else 0.0

            return {
                'count': len(reprojection_errors),
                'mean': float(np.mean(reprojection_errors)),
                'median': float(np.median(reprojection_errors)),
                'front_ratio': front_ratio,
            }

        def _count_new_point_anchor_support(new_point_ids, min_points_per_anchor=2):
            support_by_camera = {}
            if not new_point_ids:
                return support_by_camera

            new_point_ids = set(new_point_ids)
            for other_id in calib_data.get('cameras', {}).keys():
                other_id = str(other_id)
                if other_id == str(camera_id):
                    continue
                other_observations = calib_data.get('camera_points', {}).get(other_id, {})
                shared_count = sum(1 for point_id in new_point_ids if point_id in other_observations)
                if shared_count >= int(min_points_per_anchor):
                    support_by_camera[other_id] = int(shared_count)
            return support_by_camera

        def _try_small_subset_pnp(all_points_3d, all_points_2d):
            point_count = len(all_points_3d)
            if point_count < min_points or point_count > 8:
                return None

            min_subset_size = max(4, min_points)
            max_subset_size = min(point_count, 6)
            if min_subset_size > max_subset_size:
                return None

            inlier_threshold = min(max(ransac_threshold * 0.75, 5.0), 10.0)
            best_candidate = None

            for subset_size in range(min_subset_size, max_subset_size + 1):
                for combo in itertools.combinations(range(point_count), subset_size):
                    subset_indices = np.array(combo, dtype=np.int32)
                    if subset_size == 4:
                        candidate_flags = [cv2.SOLVEPNP_P3P, cv2.SOLVEPNP_AP3P, cv2.SOLVEPNP_EPNP]
                    else:
                        candidate_flags = [cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE]

                    for flag in candidate_flags:
                        try:
                            success_local, rvec_local, tvec_local = cv2.solvePnP(
                                all_points_3d[subset_indices],
                                all_points_2d[subset_indices],
                                calib_data['K'],
                                calib_data.get('dist_coeffs'),
                                flags=flag
                            )
                        except cv2.error:
                            continue

                        if not success_local:
                            continue

                        try:
                            rvec_local, tvec_local = cv2.solvePnPRefineLM(
                                all_points_3d[subset_indices],
                                all_points_2d[subset_indices],
                                calib_data['K'],
                                calib_data.get('dist_coeffs'),
                                rvec_local,
                                tvec_local
                            )
                        except cv2.error:
                            pass

                        projected_points, _ = cv2.projectPoints(
                            all_points_3d,
                            rvec_local,
                            tvec_local,
                            calib_data['K'],
                            calib_data.get('dist_coeffs')
                        )
                        reprojection_errors = np.linalg.norm(
                            projected_points.reshape(-1, 2) - all_points_2d,
                            axis=1
                        )
                        inlier_indices = np.where(reprojection_errors <= inlier_threshold)[0]
                        if len(inlier_indices) < max(min_points, 4):
                            continue

                        allow_four_of_six = False
                        if point_count <= 6 and len(inlier_indices) == 4:
                            outlier_indices = np.setdiff1d(
                                np.arange(point_count, dtype=np.int32),
                                inlier_indices,
                                assume_unique=True,
                            )
                            inlier_errors_local = reprojection_errors[inlier_indices]
                            outlier_errors_local = (
                                reprojection_errors[outlier_indices]
                                if len(outlier_indices) > 0 else np.asarray([], dtype=np.float32)
                            )
                            inlier_median_local = float(np.median(inlier_errors_local))
                            outlier_median_local = (
                                float(np.median(outlier_errors_local))
                                if len(outlier_errors_local) > 0 else float("inf")
                            )
                            allow_four_of_six = (
                                inlier_median_local <= min(inlier_threshold * 0.5, 3.0) and
                                outlier_median_local >= max(inlier_threshold * 2.0, inlier_median_local * 6.0)
                            )

                        if point_count >= 7 and len(inlier_indices) < 5:
                            continue
                        if point_count == 6 and len(inlier_indices) < 5 and not allow_four_of_six:
                            continue

                        R_local, _ = cv2.Rodrigues(rvec_local)
                        point_depths = (
                            np.asarray(R_local, dtype=np.float32) @ all_points_3d.T
                            + np.asarray(tvec_local, dtype=np.float32).reshape(3, 1)
                        )[2]
                        front_ratio_local = float(np.mean(point_depths > 0.01))
                        if front_ratio_local < 0.6:
                            continue

                        try:
                            refined_rvec_local, refined_tvec_local = cv2.solvePnPRefineLM(
                                all_points_3d[inlier_indices],
                                all_points_2d[inlier_indices],
                                calib_data['K'],
                                calib_data.get('dist_coeffs'),
                                rvec_local,
                                tvec_local
                            )
                            rvec_local, tvec_local = refined_rvec_local, refined_tvec_local
                            projected_points, _ = cv2.projectPoints(
                                all_points_3d,
                                rvec_local,
                                tvec_local,
                                calib_data['K'],
                                calib_data.get('dist_coeffs')
                            )
                            reprojection_errors = np.linalg.norm(
                                projected_points.reshape(-1, 2) - all_points_2d,
                                axis=1
                            )
                            inlier_indices = np.where(reprojection_errors <= inlier_threshold)[0]
                            if len(inlier_indices) < max(min_points, 4):
                                continue
                            if point_count >= 7 and len(inlier_indices) < 5:
                                continue
                            if point_count == 6 and len(inlier_indices) < 5 and not allow_four_of_six:
                                continue
                            R_local, _ = cv2.Rodrigues(rvec_local)
                            point_depths = (
                                np.asarray(R_local, dtype=np.float32) @ all_points_3d.T
                                + np.asarray(tvec_local, dtype=np.float32).reshape(3, 1)
                            )[2]
                            front_ratio_local = float(np.mean(point_depths > 0.01))
                            if front_ratio_local < 0.6:
                                continue
                        except cv2.error:
                            pass

                        inlier_errors = reprojection_errors[inlier_indices]
                        outlier_indices = np.setdiff1d(
                            np.arange(point_count, dtype=np.int32),
                            inlier_indices,
                            assume_unique=True,
                        )
                        outlier_errors = (
                            reprojection_errors[outlier_indices]
                            if len(outlier_indices) > 0 else np.asarray([], dtype=np.float32)
                        )
                        candidate_score = (
                            -len(inlier_indices),
                            float(np.median(inlier_errors)),
                            float(np.mean(inlier_errors)),
                            -float(np.median(outlier_errors)) if len(outlier_errors) > 0 else float("-inf"),
                            float(np.median(reprojection_errors)),
                            float(np.mean(np.sort(reprojection_errors)[:len(inlier_indices)])),
                            -front_ratio_local,
                        )

                        if best_candidate is None or candidate_score < best_candidate['score']:
                            best_candidate = {
                                'score': candidate_score,
                                'R': np.asarray(R_local, dtype=np.float32),
                                't': np.asarray(tvec_local, dtype=np.float32).reshape(3, 1),
                                'inliers': inlier_indices.astype(np.int32),
                                'mean': float(np.mean(inlier_errors)),
                                'median': float(np.median(inlier_errors)),
                                'front_ratio': front_ratio_local,
                                'subset_size': subset_size,
                                'allow_four_of_six': bool(allow_four_of_six),
                            }

            return best_candidate

        def _estimate_pairwise_translation_scale(
            anchor_id,
            shared_point_ids,
            reconstructed_shared,
            R_rel,
            t_rel,
            rel_mask,
            relative_points_3d,
        ):
            if not reconstructed_shared:
                return None

            relative_points_3d = np.asarray(relative_points_3d, dtype=np.float32)
            if relative_points_3d.ndim == 2 and relative_points_3d.shape[0] == 3:
                relative_points_3d = relative_points_3d.T
            elif relative_points_3d.ndim == 1 and relative_points_3d.size >= 3:
                relative_points_3d = relative_points_3d.reshape(1, -1)

            if relative_points_3d.ndim != 2 or relative_points_3d.shape[1] < 3:
                return None

            if rel_mask is None:
                inlier_indices = np.arange(min(len(shared_point_ids), len(relative_points_3d)), dtype=np.int32)
            else:
                inlier_indices = np.where(np.asarray(rel_mask).ravel() > 0)[0].astype(np.int32)
            if len(inlier_indices) == 0:
                return None

            usable_count = min(len(inlier_indices), len(relative_points_3d))
            inlier_indices = inlier_indices[:usable_count]
            relative_points_3d = relative_points_3d[:usable_count, :3]

            R_anchor = np.asarray(calib_data['cameras'][anchor_id][0], dtype=np.float32)
            t_anchor = np.asarray(calib_data['cameras'][anchor_id][1], dtype=np.float32).reshape(3, 1)
            R_rel = np.asarray(R_rel, dtype=np.float32)
            t_rel = np.asarray(t_rel, dtype=np.float32).reshape(3, 1)
            R_candidate = R_rel @ R_anchor
            t_candidate_base = R_rel @ t_anchor

            scale_candidates = []
            for local_idx, shared_idx in enumerate(inlier_indices):
                if shared_idx >= len(shared_point_ids):
                    break

                point_id = shared_point_ids[shared_idx]
                if point_id not in calib_data['points_3d']:
                    continue

                world_point = np.asarray(calib_data['points_3d'][point_id], dtype=np.float32).reshape(3, 1)
                anchor_point = (R_anchor @ world_point + t_anchor).reshape(3)
                relative_point = np.asarray(relative_points_3d[local_idx], dtype=np.float32).reshape(3)

                if not np.all(np.isfinite(anchor_point)) or not np.all(np.isfinite(relative_point)):
                    continue
                if anchor_point[2] <= 0.01 or relative_point[2] <= 0.01:
                    continue

                depth_ratio = float(anchor_point[2] / relative_point[2])
                if np.isfinite(depth_ratio) and 1e-3 < depth_ratio < 100.0:
                    scale_candidates.append(depth_ratio)

                anchor_norm = float(np.linalg.norm(anchor_point))
                relative_norm = float(np.linalg.norm(relative_point))
                if relative_norm > 1e-6:
                    norm_ratio = anchor_norm / relative_norm
                    if np.isfinite(norm_ratio) and 1e-3 < norm_ratio < 100.0:
                        scale_candidates.append(float(norm_ratio))

            if not scale_candidates:
                return None

            initial_scale = float(np.median(scale_candidates))
            if not np.isfinite(initial_scale) or initial_scale <= 0.0 or initial_scale > 100.0:
                return None

            required_checks = min(max(1, len(reconstructed_shared)), 2)
            best_candidate = None

            def _evaluate_scale(scale_value):
                nonlocal best_candidate
                if not np.isfinite(scale_value) or scale_value <= 0.0 or scale_value > 100.0:
                    return

                t_candidate_local = t_candidate_base + scale_value * t_rel
                stats_local = _project_reprojection_stats(
                    R_candidate,
                    t_candidate_local,
                    reconstructed_shared,
                    min_required=required_checks,
                )
                if stats_local is None or stats_local['front_ratio'] < 0.6:
                    return

                candidate_score = (
                    float(stats_local['median']),
                    float(stats_local['mean']),
                    -float(stats_local['front_ratio']),
                    abs(float(np.log(scale_value / initial_scale))),
                )
                if best_candidate is None or candidate_score < best_candidate['score']:
                    best_candidate = {
                        'scale': float(scale_value),
                        'stats': stats_local,
                        't': np.asarray(t_candidate_local, dtype=np.float32).reshape(3, 1),
                        'score': candidate_score,
                    }

            for scale_value in np.geomspace(max(initial_scale * 0.25, 1e-3), min(initial_scale * 4.0, 100.0), 21):
                _evaluate_scale(float(scale_value))
            _evaluate_scale(initial_scale)

            if best_candidate is not None:
                refine_min = max(best_candidate['scale'] * 0.6, 1e-3)
                refine_max = min(best_candidate['scale'] * 1.4, 100.0)
                if refine_max > refine_min:
                    for scale_value in np.linspace(refine_min, refine_max, 17):
                        _evaluate_scale(float(scale_value))

            if best_candidate is None:
                return None

            best_candidate['R'] = R_candidate
            return best_candidate

        def _try_pairwise_pose_fallback():
            target_points = calib_data['camera_points'][str(camera_id)]
            candidate_anchors = []
            for other_id in calib_data['cameras'].keys():
                if str(other_id) == str(camera_id):
                    continue
                other_points = calib_data['camera_points'].get(str(other_id), {})
                shared_point_ids = sorted(set(target_points.keys()) & set(other_points.keys()))
                if len(shared_point_ids) >= 8:
                    reconstructed_shared = [pt_id for pt_id in shared_point_ids if pt_id in calib_data['points_3d']]
                    candidate_anchors.append((str(other_id), shared_point_ids, reconstructed_shared))

            if not candidate_anchors:
                return None

            candidate_anchors.sort(
                key=lambda item: (len(item[2]), len(item[1])),
                reverse=True
            )

            for anchor_id, shared_point_ids, reconstructed_shared in candidate_anchors:
                pts_anchor = np.array(
                    [calib_data['camera_points'][anchor_id][point_id] for point_id in shared_point_ids],
                    dtype=np.float32
                )
                pts_target = np.array(
                    [target_points[point_id] for point_id in shared_point_ids],
                    dtype=np.float32
                )

                relative_result = camera_pose.estimate_relative_pose(
                    pts_anchor,
                    pts_target,
                    calib_data['K'],
                    calib_data.get('dist_coeffs')
                )
                if relative_result is None:
                    continue

                R_rel, t_rel, rel_mask, relative_points_3d = relative_result
                inlier_count = int(np.sum(np.asarray(rel_mask).ravel() > 0)) if rel_mask is not None else len(shared_point_ids)
                if inlier_count < 8:
                    continue

                R_anchor, t_anchor = calib_data['cameras'][anchor_id]
                R_candidate = np.asarray(R_rel, dtype=np.float32) @ np.asarray(R_anchor, dtype=np.float32)
                t_candidate = (
                    np.asarray(R_rel, dtype=np.float32) @ np.asarray(t_anchor, dtype=np.float32).reshape(3, 1)
                    + np.asarray(t_rel, dtype=np.float32).reshape(3, 1)
                )

                if reconstructed_shared:
                    scale_result = _estimate_pairwise_translation_scale(
                        anchor_id,
                        shared_point_ids,
                        reconstructed_shared,
                        R_rel,
                        t_rel,
                        rel_mask,
                        relative_points_3d,
                    )
                    if scale_result is None:
                        print(
                            f"Pairwise fallback камеры {camera_id} через {anchor_id} отклонен: "
                            f"не удалось стабилизировать масштаб по {len(reconstructed_shared)} 3D опорам"
                        )
                        continue
                    t_candidate = scale_result['t']
                    stats = scale_result['stats']
                else:
                    stats = None

                if stats is None:
                    relative_points_3d = np.asarray(relative_points_3d, dtype=np.float32)
                    if relative_points_3d.ndim == 2 and relative_points_3d.shape[0] == 3:
                        relative_points_3d = relative_points_3d.T
                    elif relative_points_3d.ndim == 1 and relative_points_3d.size >= 3:
                        relative_points_3d = relative_points_3d.reshape(1, -1)

                    if relative_points_3d.ndim != 2 or relative_points_3d.shape[1] < 3:
                        continue

                    relative_points_3d = relative_points_3d[:, :3]
                    finite_mask = np.all(np.isfinite(relative_points_3d), axis=1)
                    if np.sum(finite_mask) < 8:
                        continue

                    relative_points_3d = relative_points_3d[finite_mask]
                    anchor_front_ratio = float(np.mean(relative_points_3d[:, 2] > 0.01))
                    target_points_3d = (
                        np.asarray(R_rel, dtype=np.float32) @ relative_points_3d.T
                        + np.asarray(t_rel, dtype=np.float32).reshape(3, 1)
                    ).T
                    target_front_ratio = float(np.mean(target_points_3d[:, 2] > 0.01))
                    depth_scale = float(np.median(np.linalg.norm(relative_points_3d, axis=1)))

                    if anchor_front_ratio < 0.6 or target_front_ratio < 0.6:
                        continue
                    if not np.isfinite(depth_scale) or depth_scale <= 0.0 or depth_scale > 100.0:
                        continue

                    print(
                        f"Pairwise fallback камеры {camera_id} через {anchor_id} принят: "
                        f"{inlier_count}/{len(shared_point_ids)} 2D-2D инлаеров, "
                        f"без 3D опор, front=({anchor_front_ratio:.1%}, {target_front_ratio:.1%}), "
                        f"depth~{depth_scale:.2f}"
                    )
                    return {
                        'R': R_candidate,
                        't': t_candidate,
                        'anchor_id': str(anchor_id),
                        'shared_count': int(len(shared_point_ids)),
                        'inlier_count': int(inlier_count),
                        'used_scale_anchors': False,
                        'scale_anchor_count': 0,
                    }

                mean_limit = max(ransac_threshold * 2.0, 12.0)
                median_limit = max(ransac_threshold * 1.5, 10.0)
                if stats['mean'] > mean_limit or stats['median'] > median_limit:
                    print(
                        f"Pairwise fallback камеры {camera_id} через {anchor_id} отклонен: "
                        f"{stats['count']} 3D проверок, mean={stats['mean']:.2f}px, median={stats['median']:.2f}px"
                    )
                    continue

                scale_suffix = ""
                if reconstructed_shared:
                    scale_suffix = f", scale={scale_result['scale']:.3f}"
                print(
                    f"Pairwise fallback камеры {camera_id} через {anchor_id} принят: "
                    f"{inlier_count}/{len(shared_point_ids)} 2D-2D инлаеров, "
                    f"{stats['count']} 3D проверок, mean={stats['mean']:.2f}px, median={stats['median']:.2f}px"
                    f"{scale_suffix}"
                )
                return {
                    'R': R_candidate,
                    't': t_candidate,
                    'anchor_id': str(anchor_id),
                    'shared_count': int(len(shared_point_ids)),
                    'inlier_count': int(inlier_count),
                    'used_scale_anchors': int(len(reconstructed_shared)) >= 2,
                    'scale_anchor_count': int(len(reconstructed_shared)),
                    'weak_scale_anchors': int(len(reconstructed_shared)) < 2,
                    'pairwise_mean_error': float(stats.get('mean', 999.0)),
                }

            return None

        # Проверяем, что камера еще не добавлена
        if str(camera_id) in calib_data['cameras']:
            print(f"Камера {camera_id} уже добавлена к реконструкции")
            return True
            
        # Проверяем, что у нас есть точки для этой камеры
        if str(camera_id) not in calib_data['camera_points']:
            print(f"Нет данных о точках для камеры {camera_id}")
            return False
            
        # Проверяем, что у нас есть 3D точки
        if not calib_data['points_3d']:
            print("Нет 3D точек для оценки позы камеры")
            return False
            
        # Собираем соответствия между 2D и 3D точками
        points_3d = []
        points_2d = []
        common_point_ids = []
        
        camera_points_2d = calib_data['camera_points'][str(camera_id)]
        candidate_new_point_ids = [
            point_id for point_id in camera_points_2d.keys()
            if point_id not in calib_data['points_3d']
        ]
        all_common_point_ids = [
            point_id
            for point_id in sorted(calib_data['points_3d'].keys(), key=_stable_sort_key)
            if point_id in camera_points_2d
        ]
        pose_support_point_ids, pose_support_meta = _select_camera_pose_support_point_ids(
            calib_data,
            str(camera_id),
            common_point_ids=all_common_point_ids,
            min_points_for_camera=min_points,
            min_observation_confidence=provisional_min_confidence,
        )
        selected_point_ids = pose_support_point_ids if len(pose_support_point_ids) >= min_points else all_common_point_ids
        for point_id in selected_point_ids:
            points_3d.append(calib_data['points_3d'][point_id])
            points_2d.append(camera_points_2d[point_id])
            common_point_ids.append(point_id)
        point_observation_confidences = np.asarray(
            [_lookup_observation_confidence(calib_data, str(camera_id), point_id) for point_id in common_point_ids],
            dtype=np.float64,
        )
        use_guided_pose_support = _should_use_confidence_guided_pose_support(
            point_observation_confidences,
            min_confidence=provisional_min_confidence,
        )
        if not use_guided_pose_support:
            points_3d = [calib_data['points_3d'][point_id] for point_id in all_common_point_ids]
            points_2d = [camera_points_2d[point_id] for point_id in all_common_point_ids]
            common_point_ids = list(all_common_point_ids)
            point_observation_confidences = np.asarray(
                [_lookup_observation_confidence(calib_data, str(camera_id), point_id) for point_id in common_point_ids],
                dtype=np.float64,
            )

        if use_guided_pose_support and pose_support_point_ids and pose_support_point_ids != all_common_point_ids:
            print(
                f"Опорный scaffold для PnP камеры {camera_id}: "
                f"{len(common_point_ids)}/{len(all_common_point_ids)} точек "
                f"(strong={pose_support_meta.get('strong_count', 0)}, "
                f"reserve={pose_support_meta.get('reserve_count', 0)}, "
                f"min_conf={pose_support_meta.get('min_confidence', 0.0):.2f})"
            )
        
        # Проверяем, что у нас достаточно точек
        if len(points_3d) < min_points:
            print(f"Недостаточно общих точек для добавления камеры {camera_id} (минимум {min_points})")
            print(f"  - Опорных точек после confidence-aware отбора: {len(points_3d)}")
            print(f"  - Общих точек с уже реконструированными: {len(all_common_point_ids)}")
            print(f"    - Точки: {all_common_point_ids}")
            fallback_result = _try_pairwise_pose_fallback()
            if fallback_result is None:
                return False
            R, t = fallback_result['R'], fallback_result['t']
            success = True
            pnp_inliers = None
        else:
            print(f"Решение PnP для камеры {camera_id} с {len(points_3d)} точками...")
            
            # Преобразуем точки в numpy массивы
            points_3d = np.array(points_3d, dtype=np.float32)
            points_2d = np.array(points_2d, dtype=np.float32)
            
            success = False
            rvec = None
            tvec = None
            pnp_inliers = None
            used_ransac = False
            subset_pnp_candidate = None
            pairwise_pose_candidate = None

            try:
                success, rvec, tvec, pnp_inliers = cv2.solvePnPRansac(
                    points_3d,
                    points_2d,
                    calib_data['K'],
                    calib_data.get('dist_coeffs'),
                    flags=cv2.SOLVEPNP_EPNP,
                    reprojectionError=ransac_threshold,
                    confidence=confidence,
                    iterationsCount=1000
                )
                used_ransac = bool(success)
                if success:
                    rvec, tvec = _refine_pnp_solution(
                        points_3d,
                        points_2d,
                        calib_data['K'],
                    calib_data.get('dist_coeffs'),
                    rvec,
                    tvec,
                    inliers=pnp_inliers,
                    observation_confidences=point_observation_confidences,
                    support_min_confidence=provisional_min_confidence,
                )
            except cv2.error:
                success = False

            # На малом числе точек fallback на обычный solvePnP слишком легко дает
            # внешне "успешную", но геометрически неверную позу.
            if not success and len(points_3d) <= 8:
                subset_pnp_candidate = _try_small_subset_pnp(points_3d, points_2d)
                if subset_pnp_candidate is not None:
                    R = subset_pnp_candidate['R']
                    t = subset_pnp_candidate['t']
                    pnp_inliers = subset_pnp_candidate['inliers'].reshape(-1, 1)
                    success = True
                    print(
                        f"Малый PnP fallback камеры {camera_id} принят: "
                        f"{len(subset_pnp_candidate['inliers'])}/{len(points_3d)} инлаеров, "
                        f"subset={subset_pnp_candidate['subset_size']}, "
                        f"mean={subset_pnp_candidate['mean']:.2f}px, "
                        f"median={subset_pnp_candidate['median']:.2f}px"
                    )
                else:
                    print(
                        f"RANSAC не нашел устойчивую позу для камеры {camera_id} "
                        f"на {len(points_3d)} точках, fallback отключен"
                    )
                    fallback_result = _try_pairwise_pose_fallback()
                    if fallback_result is None:
                        return False
                    R, t = fallback_result['R'], fallback_result['t']
                    success = True
            else:
                # Fallback на обычный solvePnP, если RANSAC не дал решения.
                if not success:
                    fallback_flags = [cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP]
                    for fallback_flag in fallback_flags:
                        try:
                            success, rvec, tvec = cv2.solvePnP(
                                points_3d,
                                points_2d,
                                calib_data['K'],
                                calib_data.get('dist_coeffs'),
                                flags=fallback_flag
                            )
                            if success:
                                break
                        except cv2.error:
                            success = False
                
                if not success:
                    fallback_result = _try_pairwise_pose_fallback()
                    if fallback_result is None:
                        print(f"Не удалось оценить позу камеры {camera_id}")
                        return False
                    R, t = fallback_result['R'], fallback_result['t']
                    success = True
                else:
                    # Преобразуем вектор Родригеса в матрицу поворота
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec

            if success and len(common_point_ids) >= min_points:
                support_refine_result = _stabilize_pose_candidate_with_support(
                    points_3d,
                    points_2d,
                    calib_data['K'],
                    calib_data.get('dist_coeffs'),
                    R,
                    t,
                    min_points=max(min_points, 4),
                    preferred_indices=(
                        pnp_inliers.ravel()
                        if pnp_inliers is not None and len(pnp_inliers) >= 4 else None
                    ),
                    observation_confidences=point_observation_confidences,
                    support_min_confidence=provisional_min_confidence,
                )
                if support_refine_result is not None:
                    R = support_refine_result['R']
                    t = support_refine_result['t']
                    pnp_inliers = support_refine_result['support_indices'].reshape(-1, 1)
                    print(
                        f"Ранний support-refine камеры {camera_id}: "
                        f"median {support_refine_result['initial_support_median']:.2f}px -> "
                        f"{support_refine_result['final_support_median']:.2f}px "
                        f"(support {support_refine_result['support_count']}/{len(points_3d)})"
                    )

        # Сразу проверяем, насколько кандидатная поза согласована по ошибке репроекции.
        if len(common_point_ids) >= min_points:
            preferred_indices = pnp_inliers.ravel() if pnp_inliers is not None and len(pnp_inliers) >= 4 else None
            current_pose_eval = _evaluate_pose_candidate_support(
                points_3d,
                points_2d,
                calib_data['K'],
                calib_data.get('dist_coeffs'),
                R,
                t,
                min_points=max(min_points, 4),
                preferred_indices=preferred_indices,
                observation_confidences=point_observation_confidences,
                support_min_confidence=provisional_min_confidence,
            )
            if (
                success and
                len(points_3d) >= 8 and
                (
                    current_pose_eval is None or
                    current_pose_eval['support_median'] > max(ransac_threshold * 0.75, 3.0) or
                    current_pose_eval['support_count'] <= max(min_points + 2, 8)
                )
            ):
                pairwise_pose_candidate = _try_pairwise_pose_fallback()
                if pairwise_pose_candidate is not None:
                    pairwise_pose_eval = _evaluate_pose_candidate_support(
                        points_3d,
                        points_2d,
                        calib_data['K'],
                        calib_data.get('dist_coeffs'),
                        pairwise_pose_candidate['R'],
                        pairwise_pose_candidate['t'],
                        min_points=max(min_points, 4),
                        preferred_indices=None,
                        observation_confidences=point_observation_confidences,
                        support_min_confidence=provisional_min_confidence,
                    )
                    pairwise_is_better = _is_pose_candidate_clearly_better(pairwise_pose_eval, current_pose_eval)
                    print(
                        f"Pairwise compare камеры {camera_id}: "
                        f"current[{_format_pose_candidate_eval(current_pose_eval)}] vs "
                        f"pairwise[{_format_pose_candidate_eval(pairwise_pose_eval)}], "
                        f"anchor={pairwise_pose_candidate.get('anchor_id')} -> "
                        f"{'замена' if pairwise_is_better else 'оставляем PnP'}"
                    )
                    if pairwise_is_better:
                        R = pairwise_pose_candidate['R']
                        t = pairwise_pose_candidate['t']
                        pnp_inliers = pairwise_pose_eval['support_indices'].reshape(-1, 1)
                        current_pose_eval = pairwise_pose_eval
                        print(
                            f"Ранняя замена позы камеры {camera_id} на pairwise-кандидат: "
                            f"support median {current_pose_eval['support_median']:.2f}px, "
                            f"anchor={pairwise_pose_candidate.get('anchor_id')}"
                        )

            if current_pose_eval is None:
                projected_points, _ = cv2.projectPoints(
                    points_3d,
                    cv2.Rodrigues(np.asarray(R, dtype=np.float32))[0],
                    np.asarray(t, dtype=np.float32).reshape(3, 1),
                    calib_data['K'],
                    calib_data.get('dist_coeffs')
                )
                reprojection_errors = np.linalg.norm(
                    projected_points.reshape(-1, 2) - points_2d,
                    axis=1
                )
                error_sample = reprojection_errors
                pnp_support = int(len(reprojection_errors))
                reprojection_mean = float(np.mean(error_sample))
                reprojection_median = float(np.median(error_sample))
            else:
                reprojection_errors = current_pose_eval['stats']['errors']
                error_sample = reprojection_errors[current_pose_eval['support_indices']]
                pnp_support = int(current_pose_eval['support_count'])
                reprojection_mean = float(current_pose_eval['support_mean'])
                reprojection_median = float(current_pose_eval['support_median'])

            print(
                f"Проверка PnP камеры {camera_id}: "
                f"{pnp_support} опорных точек, "
                f"средняя ошибка={reprojection_mean:.2f}px, "
                f"медиана={reprojection_median:.2f}px"
            )

            hard_median_limit = max(ransac_threshold * 2.0, 12.0)
            hard_mean_limit = max(ransac_threshold * 3.0, 20.0)
            if reprojection_median > hard_median_limit or reprojection_mean > hard_mean_limit:
                print(
                    f"Поза камеры {camera_id} отклонена по ошибке репроекции: "
                    f"mean={reprojection_mean:.2f}px, median={reprojection_median:.2f}px, "
                    f"лимиты=({hard_mean_limit:.2f}px, {hard_median_limit:.2f}px), "
                    f"RANSAC={'да' if used_ransac else 'нет'}"
                )
                subset_pnp_candidate = _try_small_subset_pnp(points_3d, points_2d)
                if subset_pnp_candidate is not None:
                    R = subset_pnp_candidate['R']
                    t = subset_pnp_candidate['t']
                    pnp_inliers = subset_pnp_candidate['inliers'].reshape(-1, 1)
                    print(
                        f"Поза камеры {camera_id} заменена малым PnP fallback: "
                        f"{len(subset_pnp_candidate['inliers'])}/{len(points_3d)} инлаеров, "
                        f"mean={subset_pnp_candidate['mean']:.2f}px, "
                        f"median={subset_pnp_candidate['median']:.2f}px"
                    )
                else:
                    fallback_result = _try_pairwise_pose_fallback()
                    if fallback_result is None:
                        return False
                    R, t = fallback_result['R'], fallback_result['t']

            weak_support = len(points_3d) <= max(min_points + 1, 5) or pnp_support <= max(min_points, 4)
            weak_reprojection_limit = min(max(ransac_threshold * 0.75, 4.0), 10.0)
            if weak_support and reprojection_median > weak_reprojection_limit:
                print(
                    f"Слабая поза камеры {camera_id} отклонена: "
                    f"медианная ошибка {reprojection_median:.2f}px > {weak_reprojection_limit:.2f}px"
                )
                subset_pnp_candidate = _try_small_subset_pnp(points_3d, points_2d)
                if subset_pnp_candidate is not None:
                    R = subset_pnp_candidate['R']
                    t = subset_pnp_candidate['t']
                    pnp_inliers = subset_pnp_candidate['inliers'].reshape(-1, 1)
                    print(
                        f"Слабая поза камеры {camera_id} заменена малым PnP fallback: "
                        f"{len(subset_pnp_candidate['inliers'])}/{len(points_3d)} инлаеров, "
                        f"mean={subset_pnp_candidate['mean']:.2f}px, "
                        f"median={subset_pnp_candidate['median']:.2f}px"
                    )
                else:
                    fallback_result = _try_pairwise_pose_fallback()
                    if fallback_result is None:
                        return False
                    R, t = fallback_result['R'], fallback_result['t']
        
        # Проверяем, что точки находятся перед камерой. Если решение слабое,
        # пробуем несколько простых поправок позы вместо немедленного отказа.
        def _front_ratio(R_candidate, t_candidate):
            if len(points_3d) == 0:
                return 1.0
            valid_points = 0
            t_vec = t_candidate.ravel()
            for point_3d in points_3d:
                point_cam = R_candidate @ point_3d + t_vec
                if point_cam[2] > 0.01:
                    valid_points += 1
            return valid_points / len(points_3d) if len(points_3d) > 0 else 0

        front_ratio = _front_ratio(R, t)
        if front_ratio < 0.5:
            best_R, best_t = R, t
            best_front_ratio = front_ratio

            candidate_poses = [
                (R, -t),
                (R.T, -R.T @ t),
                (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32) @ R,
                 np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32) @ t),
            ]

            for candidate_R, candidate_t in candidate_poses:
                candidate_ratio = _front_ratio(candidate_R, candidate_t)
                if candidate_ratio > best_front_ratio:
                    best_R, best_t = candidate_R, candidate_t
                    best_front_ratio = candidate_ratio

            R, t = best_R, best_t
            front_ratio = best_front_ratio

        if front_ratio < 0.3:
            print(f"Большинство точек позади камеры {camera_id}: {front_ratio:.1%} впереди")
            return False
        
        # Сохраняем позу камеры
        calib_data['cameras'][str(camera_id)] = (R, t)
        
        # Добавляем ID камеры к списку реконструкции
        if str(camera_id) not in calib_data['reconstruction_ids']:
            calib_data['reconstruction_ids'].append(str(camera_id))
        _capture_point_drift_stage(
            calib_data,
            f"camera_{camera_id}_after_pnp",
            {
                "camera_id": str(camera_id),
                "common_point_count": int(len(common_point_ids)),
                "front_ratio": float(front_ratio),
            },
        )

        # --- Fix 7: Итеративное уточнение позы новой камеры перед триангуляцией ---
        # Ключевое отличие от PnP-refineLM внутри PnP-блока:
        # используем ВСЕ видимые 3D точки этой камеры (не только PnP-инлайеры),
        # включая точки, триангулированные через другие пары камер.
        # Итерируем: после каждого refineLM пересчитываем инлайеры по порогу ошибки.
        if len(common_point_ids) >= min_points and isinstance(points_3d, np.ndarray) and len(points_3d) >= 4:
            try:
                camera_pts_2d_local = calib_data['camera_points'][str(camera_id)]
                # Собираем ВСЕ видимые 3D точки (больше чем PnP-инлайеры)
                all_pts3d_list, all_pts2d_list, all_point_ids_list = [], [], []
                for pt_id in sorted(calib_data['points_3d'].keys(), key=_stable_sort_key):
                    if pt_id in camera_pts_2d_local:
                        p3d = calib_data['points_3d'][pt_id]
                        if np.all(np.isfinite(p3d)):
                            all_pts3d_list.append(np.asarray(p3d, dtype=np.float32).reshape(3))
                            all_pts2d_list.append(np.asarray(camera_pts_2d_local[pt_id], dtype=np.float32).reshape(2))
                            all_point_ids_list.append(pt_id)

                if len(all_pts3d_list) >= max(min_points, 4):
                    all_pts3d = np.array(all_pts3d_list, dtype=np.float32)
                    all_pts2d = np.array(all_pts2d_list, dtype=np.float32)
                    all_confidences = np.asarray(
                        [_lookup_observation_confidence(calib_data, str(camera_id), pt_id) for pt_id in all_point_ids_list],
                        dtype=np.float64,
                    )
                    use_guided_fix7_support = _should_use_confidence_guided_pose_support(
                        all_confidences,
                        min_confidence=provisional_min_confidence,
                    )
                    camera_K_local = _get_camera_matrix(calib_data, camera_id)

                    rvec_cur, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float32))
                    t_cur = np.asarray(t, dtype=np.float32).reshape(3, 1)

                    # Начальный порог инлайеров
                    inlier_threshold = max(ransac_threshold * 0.75, 4.0)
                    proj0, _ = cv2.projectPoints(all_pts3d, rvec_cur, t_cur,
                                                 camera_K_local, calib_data.get('dist_coeffs'))
                    errs0 = np.linalg.norm(proj0.reshape(-1, 2) - all_pts2d, axis=1)
                    inliers0 = np.where(errs0 <= inlier_threshold)[0]
                    if len(inliers0) < 4:
                        # Расширяем порог если инлайеров мало
                        inlier_threshold = float(np.percentile(errs0, 70))
                        inliers0 = np.where(errs0 <= inlier_threshold)[0]

                    if use_guided_fix7_support:
                        support_idx0 = _select_pose_refine_support_indices(
                            errs0,
                            observation_confidences=all_confidences,
                            min_points=max(min_points, 4),
                            preferred_indices=inliers0,
                            min_confidence=provisional_min_confidence,
                        )
                    else:
                        support_idx0 = _select_residual_support_indices(
                            errs0,
                            min_points=max(min_points, 4),
                            preferred_indices=inliers0,
                        )
                    if len(support_idx0) < 4:
                        support_idx0 = inliers0 if len(inliers0) >= 4 else np.argsort(errs0)[:max(min_points, 4)]

                    best_median = (
                        float(np.median(errs0[support_idx0]))
                        if len(support_idx0) > 0 else float(np.median(errs0))
                    )
                    best_rvec, best_t = rvec_cur.copy(), t_cur.copy()
                    best_support_idx = np.asarray(support_idx0, dtype=np.int32)

                    rvec_iter, t_iter = rvec_cur.copy(), t_cur.copy()
                    for _iter in range(4):
                        # Проектируем и отбираем инлайеры
                        proj_iter, _ = cv2.projectPoints(all_pts3d, rvec_iter, t_iter,
                                                         camera_K_local, calib_data.get('dist_coeffs'))
                        errs_iter = np.linalg.norm(proj_iter.reshape(-1, 2) - all_pts2d, axis=1)
                        inlier_idx = np.where(errs_iter <= inlier_threshold)[0]
                        if use_guided_fix7_support:
                            support_idx = _select_pose_refine_support_indices(
                                errs_iter,
                                observation_confidences=all_confidences,
                                min_points=max(min_points, 4),
                                preferred_indices=inlier_idx,
                                min_confidence=provisional_min_confidence,
                            )
                        else:
                            support_idx = _select_residual_support_indices(
                                errs_iter,
                                min_points=max(min_points, 4),
                                preferred_indices=inlier_idx,
                            )
                        if len(support_idx) < 4:
                            break
                        # refineLM на текущих инлайерах
                        rvec_new, t_new = cv2.solvePnPRefineLM(
                            all_pts3d[support_idx],
                            all_pts2d[support_idx],
                            camera_K_local,
                            calib_data.get('dist_coeffs'),
                            rvec_iter,
                            t_iter,
                        )
                        # Пересчитываем ошибки с новой позой
                        proj_new, _ = cv2.projectPoints(all_pts3d, rvec_new, t_new,
                                                        camera_K_local, calib_data.get('dist_coeffs'))
                        errs_new = np.linalg.norm(proj_new.reshape(-1, 2) - all_pts2d, axis=1)
                        new_inliers = np.where(errs_new <= inlier_threshold)[0]
                        if use_guided_fix7_support:
                            new_support_idx = _select_pose_refine_support_indices(
                                errs_new,
                                observation_confidences=all_confidences,
                                min_points=max(min_points, 4),
                                preferred_indices=new_inliers,
                                min_confidence=provisional_min_confidence,
                            )
                        else:
                            new_support_idx = new_inliers
                        if len(new_support_idx) < 4:
                            break
                        med_new = float(np.median(errs_new[new_support_idx]))
                        if med_new < best_median:
                            best_median = med_new
                            best_rvec, best_t = rvec_new, t_new
                            best_support_idx = np.asarray(new_support_idx, dtype=np.int32)
                        # Проверяем сходимость
                        d_rvec = float(np.linalg.norm(rvec_new - rvec_iter))
                        d_t = float(np.linalg.norm(t_new - t_iter))
                        rvec_iter, t_iter = rvec_new, t_new
                        if d_rvec < 1e-6 and d_t < 1e-6:
                            break

                    # Считаем начальную медиану для лога
                    proj_init, _ = cv2.projectPoints(all_pts3d, rvec_cur, t_cur,
                                                     camera_K_local, calib_data.get('dist_coeffs'))
                    errs_init = np.linalg.norm(proj_init.reshape(-1, 2) - all_pts2d, axis=1)
                    if use_guided_fix7_support:
                        init_support_idx = _select_pose_refine_support_indices(
                            errs_init,
                            observation_confidences=all_confidences,
                            min_points=max(min_points, 4),
                            preferred_indices=np.where(errs_init <= inlier_threshold)[0],
                            min_confidence=provisional_min_confidence,
                        )
                    else:
                        init_support_idx = _select_residual_support_indices(
                            errs_init,
                            min_points=max(min_points, 4),
                            preferred_indices=np.where(errs_init <= inlier_threshold)[0],
                        )
                    med_init = float(np.median(errs_init[init_support_idx])) if len(init_support_idx) > 0 else float(np.median(errs_init))

                    proj_final, _ = cv2.projectPoints(all_pts3d, best_rvec, best_t,
                                                      camera_K_local, calib_data.get('dist_coeffs'))
                    errs_final = np.linalg.norm(proj_final.reshape(-1, 2) - all_pts2d, axis=1)
                    final_inliers = np.where(errs_final <= inlier_threshold)[0]
                    if use_guided_fix7_support:
                        final_support_idx = _select_pose_refine_support_indices(
                            errs_final,
                            observation_confidences=all_confidences,
                            min_points=max(min_points, 4),
                            preferred_indices=final_inliers,
                            min_confidence=provisional_min_confidence,
                        )
                    else:
                        final_support_idx = _select_residual_support_indices(
                            errs_final,
                            min_points=max(min_points, 4),
                            preferred_indices=final_inliers,
                        )
                    med_final = float(np.median(errs_final[final_support_idx])) if len(final_support_idx) > 0 else float(np.median(errs_final))

                    if med_final < med_init * 0.99:  # улучшение хотя бы на 1%
                        R_ref, _ = cv2.Rodrigues(best_rvec)
                        R, t = R_ref, best_t
                        calib_data['cameras'][str(camera_id)] = (R, t)
                        print(
                            f"  Fix 7: Pose refinement камеры {camera_id}: "
                            f"median {med_init:.2f}px -> {med_final:.2f}px "
                            f"(support {len(final_support_idx)}/{len(all_pts3d)}, inliers {len(final_inliers)})"
                        )
                    else:
                        print(
                            f"  Fix 7: Pose refinement камеры {camera_id}: "
                            f"без улучшения (median {med_init:.2f}px -> {med_final:.2f}px)"
                        )
            except Exception as e:
                print(f"  Fix 7: Pose refinement пропущен: {e}")
        _capture_point_drift_stage(
            calib_data,
            f"camera_{camera_id}_after_fix7",
            {
                "camera_id": str(camera_id),
                "refinement_only_camera": bool(len(candidate_new_point_ids) == 0),
            },
        )

        refinement_only_camera = len(candidate_new_point_ids) == 0
        if (
            refinement_only_camera and
            len(common_point_ids) >= max(min_points + 4, 8) and
            isinstance(points_3d, np.ndarray) and
            len(points_3d) >= 6
        ):
            current_camera_K = _get_camera_matrix(calib_data, camera_id)
            support_stats = _evaluate_camera_pose_candidate(
                points_3d,
                points_2d,
                R,
                t,
                current_camera_K,
                calib_data.get('dist_coeffs'),
            )
            if (
                support_stats['p95'] > max(ransac_threshold * 2.0, 8.0) or
                support_stats['max'] > max(ransac_threshold * 3.0, 12.0)
            ):
                robust_pose_refine = _refine_camera_pose_from_support_robust(
                    points_3d,
                    points_2d,
                    R,
                    t,
                    current_camera_K,
                    calib_data.get('dist_coeffs'),
                    observation_confidences=point_observation_confidences,
                )
                if robust_pose_refine is not None:
                    R = np.asarray(robust_pose_refine['R'], dtype=np.float32)
                    t = np.asarray(robust_pose_refine['t'], dtype=np.float32).reshape(3, 1)
                    calib_data['cameras'][str(camera_id)] = (R, t)
                    current_camera_K = _get_camera_matrix(calib_data, camera_id)
                    print(
                        f"  Fix P: Robust pose refinement камеры {camera_id}: "
                        f"mean {robust_pose_refine['initial_stats']['mean']:.2f}px -> {robust_pose_refine['final_stats']['mean']:.2f}px, "
                        f"p95 {robust_pose_refine['initial_stats']['p95']:.2f}px -> {robust_pose_refine['final_stats']['p95']:.2f}px"
                    )
                focal_refine = _refine_camera_pose_and_focal_from_support(
                    points_3d,
                    points_2d,
                    R,
                    t,
                    current_camera_K,
                    calib_data.get('dist_coeffs'),
                    force_same_focal=True,
                    observation_confidences=point_observation_confidences,
                )
                if focal_refine is not None:
                    calib_data[f'K_{camera_id}'] = np.asarray(focal_refine['K'], dtype=np.float32)
                    R = np.asarray(focal_refine['R'], dtype=np.float32)
                    t = np.asarray(focal_refine['t'], dtype=np.float32).reshape(3, 1)
                    calib_data['cameras'][str(camera_id)] = (R, t)
                    print(
                        f"  Fix F: Per-camera focal refinement камеры {camera_id}: "
                        f"mean {focal_refine['initial_stats']['mean']:.2f}px -> {focal_refine['final_stats']['mean']:.2f}px, "
                        f"p95 {focal_refine['initial_stats']['p95']:.2f}px -> {focal_refine['final_stats']['p95']:.2f}px, "
                        f"fx={float(focal_refine['K'][0, 0]):.2f}, fy={float(focal_refine['K'][1, 1]):.2f}"
                    )

        # После успешного PnP достраиваем новые 3D точки, которые эта камера видит
        # совместно с уже реконструированными камерами.
        pairwise_is_unscaled = (
            'fallback_result' in locals()
            and fallback_result is not None
            and not fallback_result.get('used_scale_anchors', True)
        )
        pairwise_has_weak_scale = (
            'fallback_result' in locals()
            and fallback_result is not None
            and fallback_result.get('weak_scale_anchors', False)
        )
        new_points = triangulation.triangulate_new_points(
            calib_data['points_3d'],
            calib_data['cameras'],
            calib_data['camera_points'],
            calib_data['K'],
            calib_data.get('dist_coeffs'),
            str(camera_id),
            strict_track_consistency=bool(calib_data.get('strict_track_consistency', True)),
            debug_logging=debug_logging,
            observation_confidences=calib_data.get('observation_confidences'),
            multiview_refine_mode=multiview_refine_mode,
            line_support_data=calib_data.get('line_support_data'),
        )
        if new_points:
            calib_data['points_3d'].update(new_points)
            print(f"Добавлено новых 3D точек после камеры {camera_id}: {len(new_points)}")
        _capture_point_drift_stage(
            calib_data,
            f"camera_{camera_id}_after_triangulation",
            {
                "camera_id": str(camera_id),
                "new_points": int(len(new_points)),
            },
        )

        if pairwise_is_unscaled:
            min_new_points_for_unscaled_pairwise = max(2, min_points - 1)
            if len(new_points) < min_new_points_for_unscaled_pairwise:
                print(
                    f"Камера {camera_id} отклонена: pairwise fallback без 3D опор "
                    f"дал только {len(new_points)} новых точек (< {min_new_points_for_unscaled_pairwise})"
                )
                _restore_snapshot()
                return False

        if pairwise_has_weak_scale:
            weak_scale_support = _count_new_point_anchor_support(new_points.keys(), min_points_per_anchor=2)
            min_new_points_for_weak_scale = max(4, min_points)
            min_support_anchors = 2
            # Relax to 1 anchor when pairwise quality is high
            if fallback_result is not None:
                _pw_inlier_ratio = fallback_result.get('inlier_count', 0) / max(fallback_result.get('shared_count', 1), 1)
                _pw_mean = fallback_result.get('pairwise_mean_error', 999.0)
                if (
                    _pw_inlier_ratio >= 0.8
                    and _pw_mean < 1.0
                    and fallback_result.get('inlier_count', 0) >= 6
                    and len(new_points) >= min_new_points_for_weak_scale
                    and len(weak_scale_support) >= 1
                ):
                    min_support_anchors = 1
                    print(
                        f"Камера {camera_id}: weak-scale guard ослаблен до {min_support_anchors} "
                        f"(inlier_ratio={_pw_inlier_ratio:.2f}, mean={_pw_mean:.2f}px, "
                        f"new_points={len(new_points)}, support={len(weak_scale_support)})"
                    )
            if (
                len(new_points) < min_new_points_for_weak_scale or
                len(weak_scale_support) < min_support_anchors
            ):
                print(
                    f"Камера {camera_id} отклонена: pairwise fallback с слабой 3D-масштабной опорой "
                    f"(anchors={fallback_result.get('scale_anchor_count', 0)}) "
                    f"дал {len(new_points)} новых точек и multi-view support "
                    f"{len(weak_scale_support)} (< {min_support_anchors})"
                )
                _restore_snapshot()
                return False

        if new_points:
            sanitization_stats = triangulation.sanitize_points_for_camera(
                calib_data['points_3d'],
                calib_data['cameras'],
                calib_data['camera_points'],
                calib_data['K'],
                calib_data.get('dist_coeffs'),
                str(camera_id),
                protected_point_ids=new_points.keys(),
                strict_track_consistency=bool(calib_data.get('strict_track_consistency', True)),
                debug_logging=debug_logging,
                observation_confidences=calib_data.get('observation_confidences'),
                secondary_seed_points=calib_data.setdefault('_secondary_seed_points_3d', {}),
                line_support_data=calib_data.get('line_support_data'),
                soft_mode=True,
                camera_intrinsics=_collect_camera_intrinsics_map(calib_data),
            )
            if (
                sanitization_stats['removed_points'] or
                sanitization_stats['removed_observations'] or
                sanitization_stats.get('downweighted_observations', 0)
            ):
                print(
                    f"Локальная проверка (soft) точек после камеры {camera_id}: "
                    f"удалено точек {sanitization_stats['removed_points']}, "
                    f"удалено наблюдений {sanitization_stats['removed_observations']}, "
                    f"downweight наблюдений {sanitization_stats.get('downweighted_observations', 0)}, "
                    f"уточнено точек {sanitization_stats['refined_points']}"
                )
        else:
            sanitization_stats = {
                'removed_points': 0,
                'removed_observations': 0,
                'refined_points': 0,
            }
            print(
                f"Локальная проверка (soft) точек после камеры {camera_id}: "
                "отложена для refinement-only камеры без новых 3D точек"
            )
        if new_points:
            local_strict_stats = triangulation.prune_focus_conflicting_tracks(
                calib_data['points_3d'],
                calib_data['cameras'],
                calib_data['camera_points'],
                calib_data['K'],
                calib_data.get('dist_coeffs'),
                str(camera_id),
                min_track_length=4,
                strict_track_consistency=bool(calib_data.get('strict_track_consistency', True)),
                debug_logging=debug_logging,
                observation_confidences=calib_data.get('observation_confidences'),
                secondary_seed_points=calib_data.setdefault('_secondary_seed_points_3d', {}),
                line_support_data=calib_data.get('line_support_data'),
                soft_mode=True,
                camera_intrinsics=_collect_camera_intrinsics_map(calib_data),
            )
            if (
                local_strict_stats['removed_points'] or
                local_strict_stats['removed_observations'] or
                local_strict_stats.get('downweighted_observations', 0)
            ):
                print(
                    f"Локальный strict-gate (soft) после камеры {camera_id}: "
                    f"удалено точек {local_strict_stats['removed_points']}, "
                    f"удалено наблюдений {local_strict_stats['removed_observations']}, "
                    f"downweight наблюдений {local_strict_stats.get('downweighted_observations', 0)}"
                )
        else:
            local_strict_stats = {
                'removed_points': 0,
                'removed_observations': 0,
                'refined_points': 0,
            }
            print(
                f"Локальный strict-gate (soft) после камеры {camera_id}: "
                "отложен для refinement-only камеры без новых 3D точек"
            )

        if new_points:
            local_filter_stats = _filter_camera_observations_locally(
                calib_data,
                str(camera_id),
                protected_point_ids=new_points.keys(),
                absolute_threshold=max(6.0, ransac_threshold * 0.75),
                sigma_multiplier=2.0,
                mad_multiplier=2.5,
                min_observations_per_camera=max(min_points + 1, 5),
                min_track_length=3,
            )
            if local_filter_stats['removed_observations'] > 0 or local_filter_stats.get('downweighted_observations', 0) > 0:
                print(
                    f"Локальная фильтрация наблюдений камеры {camera_id}: "
                    f"удалено {local_filter_stats['removed_observations']}, "
                    f"downweight {local_filter_stats.get('downweighted_observations', 0)} "
                    f"(threshold={local_filter_stats['threshold']:.2f}px, "
                    f"median={local_filter_stats['median_error']:.2f}px, "
                    f"max={local_filter_stats['max_error']:.2f}px)"
                )
        else:
            local_filter_stats = {
                'removed_observations': 0,
                'downweighted_observations': 0,
                'threshold': None,
                'median_error': None,
                'max_error': None,
            }
            print(
                f"Локальная фильтрация наблюдений камеры {camera_id}: "
                "отложена для refinement-only камеры без новых 3D точек"
            )
        _capture_point_drift_stage(
            calib_data,
            f"camera_{camera_id}_after_local_cleanup",
            {
                "camera_id": str(camera_id),
                "new_points": int(len(new_points)),
                "soft_removed_observations": int(sanitization_stats.get('removed_observations', 0)),
                "strict_removed_observations": int(local_strict_stats.get('removed_observations', 0)),
                "local_filter_removed_observations": int(local_filter_stats.get('removed_observations', 0)),
            },
        )

        used_pairwise_fallback = fallback_result is not None
        scale_anchor_count = int(fallback_result.get('scale_anchor_count', 0)) if fallback_result is not None else 0
        support_count_for_gate = int(len(common_point_ids))
        risky_addition = used_pairwise_fallback or support_count_for_gate <= max(min_points + 2, 6)
        if risky_addition and before_distribution.get('count', 0) > 0:
            after_total_error, _, _ = calculate_reprojection_errors(
                calib_data,
                min_observation_confidence=provisional_min_confidence,
            )
            after_distribution = summarize_reprojection_error_distribution(
                calib_data,
                top_k=0,
                min_observation_confidence=provisional_min_confidence,
            )
            camera_profiles = summarize_camera_error_profiles(
                calib_data,
                top_k=3,
                min_observation_confidence=provisional_min_confidence,
            )
            current_camera_profile = next(
                (profile for profile in camera_profiles if profile['camera_id'] == str(camera_id)),
                None
            )

            local_bad = False
            if current_camera_profile is not None and current_camera_profile['count'] >= max(min_points, 4):
                local_bad = (
                    current_camera_profile['p95'] > max(before_distribution['p95'] * 1.35, 5.0) or
                    current_camera_profile['max'] > max(before_distribution['max'] * 1.15, 7.5)
                )

            global_bad = (
                after_total_error > max(before_total_error * 1.12, before_total_error + 0.15) and
                after_distribution['p95'] > max(before_distribution['p95'] * 1.18, before_distribution['p95'] + 0.45)
            )
            low_growth = len(new_points) < max(3, min_points - 1)
            weak_pairwise = used_pairwise_fallback and scale_anchor_count <= 2

            if (low_growth and local_bad) or (global_bad and (low_growth or weak_pairwise)) or (weak_pairwise and local_bad):
                print(
                    f"Камера {camera_id} отклонена после provisional-проверки: "
                    f"global mean {before_total_error:.3f}px -> {after_total_error:.3f}px, "
                    f"global p95 {before_distribution['p95']:.3f}px -> {after_distribution['p95']:.3f}px, "
                    f"new_points={len(new_points)}, support={support_count_for_gate}, "
                    f"pairwise={'да' if used_pairwise_fallback else 'нет'}"
                )
                if current_camera_profile is not None:
                    print(
                        f"  - Локальный профиль камеры {camera_id}: "
                        f"median={current_camera_profile['median']:.2f}px, "
                        f"p95={current_camera_profile['p95']:.2f}px, "
                        f"max={current_camera_profile['max']:.2f}px"
                    )
                _restore_snapshot()
                return False

        print(f"Камера {camera_id} успешно добавлена к реконструкции. Всего точек: {len(calib_data['points_3d'])}")
        _capture_point_drift_stage(
            calib_data,
            f"camera_{camera_id}_accepted",
            {
                "camera_id": str(camera_id),
                "new_points": int(len(new_points)),
            },
        )
        return True
        
    except Exception as e:
        print(f"Ошибка при добавлении камеры {camera_id} к реконструкции: {str(e)}")
        traceback.print_exc()
        return False

def refine_reconstruction(
    calib_data,
    max_iterations=3,
    focal_range=None,
    force_same_focal=False,
    optimize_intrinsics=True,
    optimize_distortion=True,
    progress_callback=None,
    progress_range=None
):
    """
    Уточняет реконструкцию с помощью bundle adjustment.
    
    Args:
        calib_data: Данные калибровки
        max_iterations: Максимальное количество итераций оптимизации
    
    Returns:
        bool: True, если оптимизация была выполнена, иначе False
    """
    if not calib_data['cameras'] or not calib_data['points_3d']:
        print("Реконструкция пуста")
        return False
        
    try:
        def _report_progress(step_fraction, status_text):
            if progress_callback is None or progress_range is None:
                return
            try:
                start_value, end_value = progress_range
                clamped_fraction = float(max(0.0, min(1.0, step_fraction)))
                progress_value = start_value + (end_value - start_value) * clamped_fraction
                progress_callback(progress_value, status_text)
            except Exception:
                pass

        debug_logging = bool(calib_data.get('debug_logging', False))
        total_iterations = max(int(max_iterations), 1)
        fast_geometry_only = not (optimize_intrinsics or optimize_distortion)
        bundle_step_ftol = 1e-6 if fast_geometry_only else 1e-7
        bundle_step_max_nfev = 250 if fast_geometry_only else 400
        target_mean = float(calib_data.get('precision_target_mean_px', 0.5))
        target_p95 = float(calib_data.get('precision_target_p95_px', 1.0))
        target_max = float(calib_data.get('precision_target_max_px', 1.5))
        previous_metrics = None
        _report_progress(0.0, "Bundle adjustment...")
        for i in range(max_iterations):
            _report_progress(
                min(0.7, (i / total_iterations) * 0.7),
                f"Bundle adjustment: итерация {i + 1}/{max_iterations}"
            )
            print(f"\nИтерация оптимизации {i+1}/{max_iterations}")
            ba_observation_confidences = _build_quality_aware_observation_confidences(
                calib_data,
                label=f"BA iter {i + 1}",
            )

            # Выполняем bundle adjustment
            points_3d_optimized, cameras_optimized = bundle_adjustment.bundle_adjust_step(
                calib_data['points_3d'], 
                calib_data['cameras'], 
                calib_data['camera_points'], 
                calib_data['K'], 
                calib_data.get('dist_coeffs'),
                ftol=bundle_step_ftol,
                max_nfev=bundle_step_max_nfev,
                debug_logging=debug_logging,
                observation_confidences=ba_observation_confidences,
                camera_intrinsics=_collect_camera_intrinsics_map(calib_data),
            )
            
            # Обновляем данные
            calib_data['points_3d'] = points_3d_optimized
            calib_data['cameras'] = cameras_optimized

            try:
                current_mean, _, _ = calculate_reprojection_errors(calib_data)
                current_stats = summarize_reprojection_error_distribution(calib_data, top_k=1)
            except Exception:
                current_mean = None
                current_stats = None

            if current_mean is not None and current_stats is not None:
                current_metrics = (
                    float(current_mean),
                    float(current_stats.get('p95', current_mean)),
                    float(current_stats.get('max', current_mean)),
                )
                if previous_metrics is not None:
                    mean_gain = previous_metrics[0] - current_metrics[0]
                    p95_gain = previous_metrics[1] - current_metrics[1]
                    max_gain = previous_metrics[2] - current_metrics[2]
                    if (
                        mean_gain <= 0.01 and
                        p95_gain <= 0.03 and
                        max_gain <= 0.05
                    ):
                        print(
                            "Ранняя остановка bundle adjustment: "
                            f"улучшение стало незначительным "
                            f"(d_mean={mean_gain:.4f}px, d_p95={p95_gain:.4f}px, d_max={max_gain:.4f}px)"
                        )
                        previous_metrics = current_metrics
                        break

                previous_metrics = current_metrics
                if (
                    fast_geometry_only and
                    current_metrics[0] <= target_mean and
                    current_metrics[1] <= target_p95 and
                    current_metrics[2] <= target_max
                ):
                    print(
                        "Ранняя остановка bundle adjustment: "
                        f"геометрия уже в целевых пределах "
                        f"(mean={current_metrics[0]:.4f}px, "
                        f"p95={current_metrics[1]:.4f}px, max={current_metrics[2]:.4f}px)"
                    )
                    break

        if optimize_intrinsics:
            _report_progress(0.75, "Оптимизация матрицы камеры...")
            intrinsics_observation_confidences = _build_quality_aware_observation_confidences(
                calib_data,
                label="K optimization",
            )
            optimized_K, initial_intrinsics_rmse, final_intrinsics_rmse = bundle_adjustment.optimize_shared_intrinsics(
                calib_data['K'],
                calib_data['points_3d'],
                calib_data['cameras'],
                calib_data['camera_points'],
                calib_data.get('dist_coeffs'),
                focal_range=focal_range,
                force_same_focal=force_same_focal,
                observation_confidences=intrinsics_observation_confidences,
            )
            if optimized_K is not None:
                k_delta = np.linalg.norm(np.asarray(optimized_K, dtype=np.float64) - np.asarray(calib_data['K'], dtype=np.float64))
                if k_delta > 1e-3:
                    calib_data['K'] = optimized_K
                    print(f"Обновлена общая матрица K после BA: fx={optimized_K[0, 0]:.2f}, fy={optimized_K[1, 1]:.2f}")

                    if initial_intrinsics_rmse is not None and final_intrinsics_rmse is not None and final_intrinsics_rmse < initial_intrinsics_rmse:
                        print("Дополнительный шаг bundle adjustment с уточненной матрицей K...")
                        _report_progress(0.85, "Дополнительный bundle adjustment...")
                        followup_ba_confidences = _build_quality_aware_observation_confidences(
                            calib_data,
                            label="Post-K BA",
                        )
                        points_3d_optimized, cameras_optimized = bundle_adjustment.bundle_adjust_step(
                            calib_data['points_3d'],
                            calib_data['cameras'],
                            calib_data['camera_points'],
                            calib_data['K'],
                            calib_data.get('dist_coeffs'),
                            ftol=max(bundle_step_ftol, 1e-7),
                            max_nfev=max(bundle_step_max_nfev, 300),
                            debug_logging=debug_logging,
                            observation_confidences=followup_ba_confidences,
                            camera_intrinsics=_collect_camera_intrinsics_map(calib_data),
                        )
                        calib_data['points_3d'] = points_3d_optimized
                        calib_data['cameras'] = cameras_optimized
            
        # Оптимизируем коэффициенты дисторсии
        if optimize_distortion and calib_data['dist_coeffs'] is not None:
            _report_progress(0.92, "Оптимизация дисторсии...")
            print("\nОптимизация коэффициентов дисторсии...")
            optimized_dist = bundle_adjustment.optimize_distortion(
                calib_data['K'], 
                calib_data['points_3d'], 
                calib_data['cameras'], 
                calib_data['camera_points'], 
                calib_data['dist_coeffs']
            )
            calib_data['dist_coeffs'] = optimized_dist

        _report_progress(1.0, "Оптимизация реконструкции завершена")
        return True
        
    except Exception as e:
        print(f"Ошибка при оптимизации реконструкции: {str(e)}")
        traceback.print_exc()
        return False

def filter_observations_by_reprojection_error(
    calib_data,
    absolute_threshold=8.0,
    sigma_multiplier=2.5,
    mad_multiplier=3.0,
    min_observations_per_camera=5,
    min_track_length=2,
    min_improvement=0.25,
    min_relative_improvement=0.0,
    force_allow=False,
):
    """
    Удаляет отдельные 2D-наблюдения с большой ошибкой репроекции, не удаляя всю 3D точку целиком.

    Args:
        calib_data: Данные калибровки
        absolute_threshold: Абсолютный порог ошибки в пикселях
        sigma_multiplier: Множитель для std-based порога
        mad_multiplier: Множитель для MAD-based порога
        min_observations_per_camera: Минимум наблюдений, который нужно сохранить в камере
        min_track_length: Минимум камер, в которых должна остаться точка

    Returns:
        tuple: (removed_count, old_error, new_error)
    """
    if not calib_data.get('cameras') or not calib_data.get('points_3d'):
        return 0, 0.0, 0.0

    if bool(calib_data.get('strict_track_consistency', True)) and not force_allow:
        print("Фильтрация наблюдений пропущена: strict_track_consistency включен")
        old_error, _, _ = calculate_reprojection_errors(calib_data)
        return 0, old_error, old_error

    observation_errors = []
    point_track_lengths = {}

    for camera_id, observations in calib_data['camera_points'].items():
        for point_id in observations.keys():
            point_track_lengths[point_id] = point_track_lengths.get(point_id, 0) + 1

    for camera_id, (R, t) in calib_data['cameras'].items():
        observations = calib_data['camera_points'].get(str(camera_id), {})
        for point_id, point_2d in observations.items():
            if point_id not in calib_data['points_3d']:
                continue

            projected_point = utils.project_point(
                calib_data['points_3d'][point_id],
                R,
                t,
                _get_camera_matrix(calib_data, camera_id),
                calib_data.get('dist_coeffs')
            )
            error = float(np.linalg.norm(projected_point - point_2d))
            observation_errors.append((str(camera_id), point_id, error))

    if len(observation_errors) < 10:
        return 0, 0.0, 0.0

    old_error, _, _ = calculate_reprojection_errors(calib_data)
    errors_only = np.array([item[2] for item in observation_errors], dtype=np.float64)
    global_median = float(np.median(errors_only))
    global_std = float(np.std(errors_only))
    global_mad = float(stats.median_abs_deviation(errors_only, scale='normal'))

    candidate_removals = []
    for camera_id in calib_data['cameras'].keys():
        camera_entries = [item for item in observation_errors if item[0] == str(camera_id)]
        if len(camera_entries) <= min_observations_per_camera:
            continue

        camera_errors = np.array([item[2] for item in camera_entries], dtype=np.float64)
        camera_median = float(np.median(camera_errors))
        camera_std = float(np.std(camera_errors))
        camera_mad = float(stats.median_abs_deviation(camera_errors, scale='normal'))

        sigma_threshold = camera_median + sigma_multiplier * max(camera_std, global_std * 0.25)
        mad_threshold = camera_median + mad_multiplier * max(camera_mad, global_mad * 0.25, 1e-6)
        combined_threshold = max(absolute_threshold, min(sigma_threshold, mad_threshold))

        for _, point_id, error in camera_entries:
            if error > combined_threshold:
                candidate_removals.append((str(camera_id), point_id, error, combined_threshold))

    if not candidate_removals:
        print("Фильтрация наблюдений: выбросы не найдены")
        return 0, old_error, old_error

    camera_remaining = {
        str(camera_id): len(calib_data['camera_points'].get(str(camera_id), {}))
        for camera_id in calib_data['cameras'].keys()
    }
    point_remaining = dict(point_track_lengths)

    removals_to_apply = []
    for camera_id, point_id, error, threshold in sorted(candidate_removals, key=lambda item: item[2], reverse=True):
        if camera_remaining.get(camera_id, 0) <= min_observations_per_camera:
            continue
        if point_remaining.get(point_id, 0) <= min_track_length:
            continue
        if point_id not in calib_data['camera_points'].get(camera_id, {}):
            continue

        removals_to_apply.append((camera_id, point_id, error, threshold))
        camera_remaining[camera_id] -= 1
        point_remaining[point_id] -= 1

    if not removals_to_apply:
        print("Фильтрация наблюдений: кандидаты были найдены, но их нельзя удалить без потери устойчивости")
        return 0, old_error, old_error

    removed_by_camera = {}
    removed_records = []
    for camera_id, point_id, error, threshold in removals_to_apply:
        removed_records.append((camera_id, point_id, calib_data['camera_points'][camera_id][point_id]))
        del calib_data['camera_points'][camera_id][point_id]
        removed_by_camera[camera_id] = removed_by_camera.get(camera_id, 0) + 1

    new_error, _, _ = calculate_reprojection_errors(calib_data)
    improvement = old_error - new_error

    relative_improvement = (improvement / max(old_error, 1e-6)) if old_error > 0 else 0.0
    if (
        new_error > old_error or
        improvement < float(min_improvement) or
        relative_improvement < float(min_relative_improvement)
    ):
        for camera_id, point_id, point_2d in removed_records:
            calib_data['camera_points'][camera_id][point_id] = point_2d
        print(
            "Фильтрация наблюдений отменена: "
            f"ошибка {old_error:.4f}px -> {new_error:.4f}px не улучшилась достаточно"
        )
        return 0, old_error, old_error

    print("Фильтрация наблюдений по репроекции:")
    print(f"  - Удалено наблюдений: {len(removals_to_apply)}")
    print(f"  - Ошибка до: {old_error:.4f}px")
    print(f"  - Ошибка после: {new_error:.4f}px")
    for camera_id, count in sorted(removed_by_camera.items()):
        print(f"  - Камера {camera_id}: удалено наблюдений {count}")

    return len(removals_to_apply), old_error, new_error

def calculate_reprojection_errors(calib_data, min_observation_confidence=0.0):
    """
    Вычисляет ошибки репроекции для всех точек и камер.
    
    Args:
        calib_data: Данные калибровки
        
    Returns:
        tuple: (total_error, errors_by_point, errors_by_camera)
               - общая средняя ошибка
               - словарь {point_id: error} с ошибками репроекции для каждой точки
               - словарь {camera_id: [errors]} с ошибками репроекции для каждой камеры
    """
    try:
        if not calib_data['cameras'] or not calib_data['points_3d']:
            print("Реконструкция пуста")
            return 0.0, {}, {}
            
        errors_by_camera = {}
        errors_by_point = {}
        all_errors = []
        
        min_observation_confidence = float(min_observation_confidence or 0.0)

        for camera_id, (R, t) in calib_data['cameras'].items():
            points_2d = calib_data['camera_points'].get(str(camera_id), {})
            
            # Пропускаем камеры без точек
            if not points_2d:
                continue
                
            errors_by_camera[camera_id] = []
            
            # Вычисляем ошибки для каждой точки
            for point_id, point_2d in points_2d.items():
                if _lookup_observation_confidence(calib_data, camera_id, point_id) < min_observation_confidence:
                    continue
                if point_id in calib_data['points_3d']:
                    point_3d = calib_data['points_3d'][point_id]
                    camera_K = _get_camera_matrix(calib_data, camera_id)
                    if camera_K is None:
                        continue
                    
                    # Проецируем 3D точку на 2D
                    projected_point = utils.project_point(point_3d, R, t, camera_K, calib_data.get('dist_coeffs'))
                    
                    # Считаем ошибку
                    error = np.linalg.norm(projected_point - point_2d)
                    errors_by_camera[camera_id].append(error)
                    
                    # Сохраняем ошибку для точки
                    if point_id not in errors_by_point:
                        errors_by_point[point_id] = []
                    errors_by_point[point_id].append(error)
                    
                    all_errors.append(error)
        
        # Вычисляем среднее значение ошибки для каждой точки
        for point_id in errors_by_point:
            errors_by_point[point_id] = np.mean(errors_by_point[point_id])
        
        # Рассчитываем общую среднюю ошибку
        total_error = np.mean(all_errors) if all_errors else 0.0
        
        return total_error, errors_by_point, errors_by_camera
    
    except Exception as e:
        print(f"Ошибка при вычислении ошибок репроекции: {str(e)}")
        traceback.print_exc()
        return 0.0, {}, {}


def summarize_reprojection_error_distribution(calib_data, top_k=5, min_observation_confidence=0.0):
    """
    Возвращает расширенную статистику по ошибкам репроекции на уровне отдельных наблюдений.

    Returns:
        dict: {
            'count': int,
            'mean': float,
            'median': float,
            'p90': float,
            'p95': float,
            'max': float,
            'worst_observations': [{'camera_id': str, 'point_id': any, 'error': float}, ...]
        }
    """
    if not calib_data.get('cameras') or not calib_data.get('points_3d'):
        return {
            'count': 0,
            'mean': 0.0,
            'median': 0.0,
            'p90': 0.0,
            'p95': 0.0,
            'max': 0.0,
            'worst_observations': [],
        }

    min_observation_confidence = float(min_observation_confidence or 0.0)
    observation_errors = []
    for camera_id, (R, t) in calib_data['cameras'].items():
        observations = calib_data['camera_points'].get(str(camera_id), {})
        for point_id, point_2d in observations.items():
            if _lookup_observation_confidence(calib_data, camera_id, point_id) < min_observation_confidence:
                continue
            if point_id not in calib_data['points_3d']:
                continue

            projected_point = utils.project_point(
                calib_data['points_3d'][point_id],
                R,
                t,
                _get_camera_matrix(calib_data, camera_id),
                calib_data.get('dist_coeffs')
            )
            error = float(np.linalg.norm(projected_point - point_2d))
            observation_errors.append((error, str(camera_id), point_id))

    if not observation_errors:
        return {
            'count': 0,
            'mean': 0.0,
            'median': 0.0,
            'p90': 0.0,
            'p95': 0.0,
            'max': 0.0,
            'worst_observations': [],
        }

    errors_only = np.array([item[0] for item in observation_errors], dtype=np.float64)
    worst_observations = [
        {
            'camera_id': camera_id,
            'point_id': point_id,
            'error': float(error),
        }
        for error, camera_id, point_id in sorted(observation_errors, reverse=True)[:max(1, int(top_k))]
    ]

    return {
        'count': int(len(errors_only)),
        'mean': float(np.mean(errors_only)),
        'median': float(np.median(errors_only)),
        'p90': float(np.percentile(errors_only, 90)),
        'p95': float(np.percentile(errors_only, 95)),
        'max': float(np.max(errors_only)),
        'worst_observations': worst_observations,
    }


def summarize_camera_error_profiles(calib_data, top_k=3, min_observation_confidence=0.0):
    if not calib_data.get('cameras') or not calib_data.get('points_3d'):
        return []

    min_observation_confidence = float(min_observation_confidence or 0.0)
    profiles = []
    for camera_id, (R, t) in calib_data['cameras'].items():
        observations = calib_data['camera_points'].get(str(camera_id), {})
        camera_errors = []
        for point_id, point_2d in observations.items():
            if _lookup_observation_confidence(calib_data, camera_id, point_id) < min_observation_confidence:
                continue
            if point_id not in calib_data['points_3d']:
                continue
            projected_point = utils.project_point(
                calib_data['points_3d'][point_id],
                R,
                t,
                _get_camera_matrix(calib_data, camera_id),
                calib_data.get('dist_coeffs'),
            )
            error = float(np.linalg.norm(projected_point - point_2d))
            camera_errors.append((error, point_id))

        if not camera_errors:
            continue

        errors_only = np.array([item[0] for item in camera_errors], dtype=np.float64)
        worst_observations = [
            {
                'point_id': point_id,
                'error': float(error),
            }
            for error, point_id in sorted(camera_errors, reverse=True)[:max(1, int(top_k))]
        ]
        profiles.append({
            'camera_id': str(camera_id),
            'count': int(len(errors_only)),
            'mean': float(np.mean(errors_only)),
            'median': float(np.median(errors_only)),
            'p90': float(np.percentile(errors_only, 90)),
            'p95': float(np.percentile(errors_only, 95)),
            'max': float(np.max(errors_only)),
            'worst_observations': worst_observations,
        })

    profiles.sort(key=lambda item: (item['p95'], item['max'], item['mean']), reverse=True)
    return profiles


def summarize_point_error_profiles(calib_data, top_k=5, min_track_length=3, min_observation_confidence=0.0):
    if not calib_data.get('cameras') or not calib_data.get('points_3d'):
        return []

    min_observation_confidence = float(min_observation_confidence or 0.0)
    profiles = []
    for point_id, point_3d in calib_data['points_3d'].items():
        point_errors = []
        for camera_id, (R, t) in calib_data['cameras'].items():
            point_2d = calib_data['camera_points'].get(str(camera_id), {}).get(point_id)
            if point_2d is None:
                continue
            if _lookup_observation_confidence(calib_data, camera_id, point_id) < min_observation_confidence:
                continue
            projected_point = utils.project_point(
                point_3d,
                R,
                t,
                _get_camera_matrix(calib_data, camera_id),
                calib_data.get('dist_coeffs'),
            )
            error = float(np.linalg.norm(projected_point - point_2d))
            point_errors.append((error, str(camera_id)))

        if len(point_errors) < int(min_track_length):
            continue

        errors_only = np.array([item[0] for item in point_errors], dtype=np.float64)
        median_error = float(np.median(errors_only))
        max_error = float(np.max(errors_only))
        min_error = float(np.min(errors_only))
        asymmetry = max_error - median_error
        max_to_median = max_error / max(median_error, 1e-6)

        worst_observations = [
            {
                'camera_id': camera_id,
                'error': float(error),
            }
            for error, camera_id in sorted(point_errors, reverse=True)[:max(1, int(top_k))]
        ]

        profiles.append({
            'point_id': point_id,
            'count': int(len(errors_only)),
            'mean': float(np.mean(errors_only)),
            'median': median_error,
            'p90': float(np.percentile(errors_only, 90)),
            'p95': float(np.percentile(errors_only, 95)),
            'min': min_error,
            'max': max_error,
            'asymmetry': float(asymmetry),
            'max_to_median': float(max_to_median),
            'worst_observations': worst_observations,
        })

    profiles.sort(
        key=lambda item: (
            item['asymmetry'],
            item['max_to_median'],
            item['max'],
            item['count'],
        ),
        reverse=True,
    )
    return profiles


def _count_point_support_by_camera(calib_data, point_ids):
    point_ids = set(point_ids or [])
    support = {}
    for camera_id in sorted(calib_data.get('cameras', {}).keys(), key=_stable_sort_key):
        camera_key = str(camera_id)
        observations = calib_data.get('camera_points', {}).get(camera_key, {})
        support[camera_key] = int(sum(1 for point_id in point_ids if point_id in observations))
    return support


def _select_pose_scaffold_point_ids(
    calib_data,
    min_points_for_camera=4,
):
    if not calib_data.get('cameras') or not calib_data.get('points_3d'):
        return set(), {}

    min_track_length = max(
        2,
        int(calib_data.get('pose_scaffold_min_track_length', 3)),
        min(3, int(min_points_for_camera)),
    )
    min_total_points = max(10, int(min_points_for_camera) * 2)
    target_camera_support = max(4, int(min_points_for_camera) + 1)

    min_observation_confidence = _get_soft_gate_min_confidence(calib_data)
    global_stats = summarize_reprojection_error_distribution(
        calib_data,
        top_k=0,
        min_observation_confidence=min_observation_confidence,
    )
    point_profiles = summarize_point_error_profiles(
        calib_data,
        top_k=0,
        min_track_length=min_track_length,
        min_observation_confidence=min_observation_confidence,
    )
    if not point_profiles:
        return set(), {
            'reason': 'no_profiles',
            'min_track_length': int(min_track_length),
        }

    global_median = float(global_stats.get('median') or 0.0)
    global_p95 = float(global_stats.get('p95') or 0.0)
    strong_candidates = []
    reserve_candidates = []

    for profile in point_profiles:
        track_length = int(profile.get('count') or 0)
        mean_error = float(profile.get('mean') or float('inf'))
        p95_error = float(profile.get('p95') or float('inf'))
        max_error = float(profile.get('max') or float('inf'))
        asymmetry = float(profile.get('asymmetry') or 0.0)
        max_to_median = float(profile.get('max_to_median') or float('inf'))
        point_id = profile.get('point_id')
        if point_id is None or track_length < min_track_length:
            continue

        quality_score = (
            track_length * 2.80 -
            mean_error * 1.20 -
            max_error * 0.70 -
            asymmetry * 0.55 -
            max(0.0, max_to_median - 1.0) * 0.35
        )
        entry = {
            'point_id': point_id,
            'score': float(quality_score),
            'track_length': track_length,
            'mean': mean_error,
            'p95': p95_error,
            'max': max_error,
            'asymmetry': asymmetry,
            'max_to_median': max_to_median,
        }

        is_strong = (
            mean_error <= max(global_median * 1.35, 0.95) and
            p95_error <= max(global_p95 * 1.05, global_median + 0.90, 1.30) and
            max_error <= max(global_p95 * 1.18, 1.85) and
            asymmetry <= max(global_median * 1.25 + 0.30, 1.05) and
            max_to_median <= 2.80
        )
        is_usable = (
            mean_error <= max(global_p95 * 1.08, 1.75) and
            max_error <= max(global_p95 * 1.40, 2.75) and
            max_to_median <= 4.25
        )

        if is_strong:
            strong_candidates.append(entry)
        elif is_usable:
            reserve_candidates.append(entry)

    if not strong_candidates and not reserve_candidates:
        return set(), {
            'reason': 'no_candidates',
            'min_track_length': int(min_track_length),
        }

    ranked_candidates = sorted(
        strong_candidates + reserve_candidates,
        key=lambda item: (
            item['score'],
            item['track_length'],
            -item['mean'],
            -item['max'],
        ),
        reverse=True,
    )
    scaffold_cap = max(
        min_total_points,
        min(
            len(ranked_candidates),
            int(np.ceil(len(point_profiles) * 0.70))
        )
    )

    selected_point_ids = []
    selected_point_set = set()
    for entry in sorted(
        strong_candidates,
        key=lambda item: (
            item['score'],
            item['track_length'],
            -item['mean'],
            -item['max'],
        ),
        reverse=True,
    ):
        if len(selected_point_ids) >= scaffold_cap:
            break
        point_id = entry['point_id']
        if point_id in selected_point_set:
            continue
        selected_point_ids.append(point_id)
        selected_point_set.add(point_id)

    camera_support = _count_point_support_by_camera(calib_data, selected_point_set)
    remaining_candidates = [entry for entry in ranked_candidates if entry['point_id'] not in selected_point_set]

    for camera_id in sorted(calib_data.get('cameras', {}).keys(), key=_stable_sort_key):
        camera_key = str(camera_id)
        current_support = int(camera_support.get(camera_key, 0))
        if current_support >= target_camera_support:
            continue
        camera_observations = calib_data.get('camera_points', {}).get(camera_key, {})
        for entry in remaining_candidates:
            point_id = entry['point_id']
            if point_id in selected_point_set or point_id not in camera_observations:
                continue
            selected_point_ids.append(point_id)
            selected_point_set.add(point_id)
            for supported_camera_id in calib_data.get('cameras', {}).keys():
                supported_camera_key = str(supported_camera_id)
                if point_id in calib_data.get('camera_points', {}).get(supported_camera_key, {}):
                    camera_support[supported_camera_key] = int(camera_support.get(supported_camera_key, 0)) + 1
            current_support = int(camera_support.get(camera_key, 0))
            if current_support >= target_camera_support or len(selected_point_ids) >= max(scaffold_cap, min_total_points):
                break

    if len(selected_point_ids) < min_total_points:
        for entry in remaining_candidates:
            point_id = entry['point_id']
            if point_id in selected_point_set:
                continue
            selected_point_ids.append(point_id)
            selected_point_set.add(point_id)
            if len(selected_point_ids) >= min_total_points:
                break

    selected_point_set = set(selected_point_ids)
    camera_support = _count_point_support_by_camera(calib_data, selected_point_set)
    return selected_point_set, {
        'min_track_length': int(min_track_length),
        'selected_count': int(len(selected_point_set)),
        'strong_candidate_count': int(len(strong_candidates)),
        'reserve_candidate_count': int(len(reserve_candidates)),
        'camera_support': camera_support,
    }


def _expand_two_stage_pose_scaffold_point_ids(
    calib_data,
    scaffold_point_ids,
    min_points_for_camera=4,
):
    scaffold_point_set = set(scaffold_point_ids or [])
    min_observation_confidence = _get_soft_gate_min_confidence(calib_data)
    point_profiles = summarize_point_error_profiles(
        calib_data,
        top_k=0,
        min_track_length=max(2, min(3, int(min_points_for_camera))),
        min_observation_confidence=min_observation_confidence,
    )
    if not point_profiles:
        return scaffold_point_set, {
            'initial_count': int(len(scaffold_point_set)),
            'expanded_count': int(len(scaffold_point_set)),
            'target_count': int(len(scaffold_point_set)),
            'target_camera_support': int(max(min_points_for_camera + 2, 6)),
            'camera_support': _count_point_support_by_camera(calib_data, scaffold_point_set),
            'reason': 'no_profiles',
        }

    target_ratio = float(np.clip(calib_data.get('two_stage_pose_scaffold_target_ratio', 0.90), 0.50, 1.00))
    target_count = max(
        int(len(scaffold_point_set)),
        int(calib_data.get('two_stage_pose_scaffold_min_points', 12)),
        int(np.ceil(len(point_profiles) * target_ratio)),
    )
    target_count = min(int(len(point_profiles)), int(target_count))
    target_camera_support = max(
        int(min_points_for_camera) + 1,
        int(calib_data.get('two_stage_pose_scaffold_min_camera_support', 6)),
    )

    ranked_profiles = sorted(
        point_profiles,
        key=lambda item: (
            int(item.get('count') or 0),
            -(float(item.get('mean') or 0.0)),
            -(float(item.get('p95') or 0.0)),
            -(float(item.get('max') or 0.0)),
            -(float(item.get('asymmetry') or 0.0)),
            -(float(item.get('max_to_median') or 0.0)),
        ),
        reverse=True,
    )

    selected_point_ids = sorted(scaffold_point_set, key=_stable_sort_key)
    selected_point_set = set(selected_point_ids)
    camera_support = _count_point_support_by_camera(calib_data, selected_point_set)
    remaining_profiles = [
        profile
        for profile in ranked_profiles
        if profile.get('point_id') not in selected_point_set
    ]

    for camera_id in sorted(calib_data.get('camera_points', {}).keys(), key=_stable_sort_key):
        camera_key = str(camera_id)
        current_support = int(camera_support.get(camera_key, 0))
        if current_support >= target_camera_support:
            continue
        camera_observations = calib_data.get('camera_points', {}).get(camera_key, {})
        for profile in remaining_profiles:
            point_id = profile.get('point_id')
            if point_id is None or point_id in selected_point_set or point_id not in camera_observations:
                continue
            selected_point_ids.append(point_id)
            selected_point_set.add(point_id)
            for supported_camera_id in calib_data.get('camera_points', {}).keys():
                supported_camera_key = str(supported_camera_id)
                if point_id in calib_data.get('camera_points', {}).get(supported_camera_key, {}):
                    camera_support[supported_camera_key] = int(camera_support.get(supported_camera_key, 0)) + 1
            current_support = int(camera_support.get(camera_key, 0))
            if len(selected_point_ids) >= target_count or current_support >= target_camera_support:
                break

    if len(selected_point_ids) < target_count:
        for profile in remaining_profiles:
            point_id = profile.get('point_id')
            if point_id is None or point_id in selected_point_set:
                continue
            selected_point_ids.append(point_id)
            selected_point_set.add(point_id)
            if len(selected_point_ids) >= target_count:
                break

    selected_point_set = set(selected_point_ids)
    return selected_point_set, {
        'initial_count': int(len(scaffold_point_set)),
        'expanded_count': int(len(selected_point_set)),
        'target_count': int(target_count),
        'target_camera_support': int(target_camera_support),
        'camera_support': _count_point_support_by_camera(calib_data, selected_point_set),
    }


def _build_pose_scaffold_observation_confidences(
    calib_data,
    scaffold_point_ids,
    nonmember_confidence=0.22,
):
    scaffold_point_ids = set(scaffold_point_ids or [])
    base_confidences = calib_data.get('observation_confidences') or {}
    merged_confidences = {}

    for camera_id, observations in calib_data.get('camera_points', {}).items():
        camera_key = str(camera_id)
        camera_confidences = dict(base_confidences.get(camera_key, {}))
        merged_camera_confidences = {}
        for point_id in observations.keys():
            base_value = camera_confidences.get(point_id, 1.0)
            try:
                base_value = float(base_value)
            except (TypeError, ValueError):
                base_value = 1.0
            base_value = float(np.clip(base_value, 0.15, 1.0))
            if point_id in scaffold_point_ids:
                merged_value = max(base_value, 0.95)
            else:
                merged_value = min(base_value, float(nonmember_confidence))
            merged_camera_confidences[point_id] = float(np.clip(merged_value, 0.15, 1.0))
        if merged_camera_confidences:
            merged_confidences[camera_key] = merged_camera_confidences

    return merged_confidences


def _estimate_similarity_transform(source_points, target_points):
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    if source_points.ndim != 2 or target_points.ndim != 2:
        return None
    if source_points.shape != target_points.shape or source_points.shape[0] < 3 or source_points.shape[1] != 3:
        return None

    source_mean = np.mean(source_points, axis=0)
    target_mean = np.mean(target_points, axis=0)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean

    source_var = float(np.mean(np.sum(source_centered ** 2, axis=1)))
    if not np.isfinite(source_var) or source_var < 1e-9:
        return None

    covariance = (target_centered.T @ source_centered) / float(source_points.shape[0])
    try:
        U, singular_values, Vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None

    reflection_fix = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        reflection_fix[-1, -1] = -1.0

    rotation = U @ reflection_fix @ Vt
    scale = float(np.sum(singular_values * np.diag(reflection_fix)) / source_var)
    if not np.isfinite(scale) or scale <= 1e-6:
        return None

    translation = target_mean - scale * (rotation @ source_mean)
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        return None

    return {
        'scale': float(scale),
        'rotation': np.asarray(rotation, dtype=np.float64),
        'translation': np.asarray(translation, dtype=np.float64).reshape(3),
    }


def _apply_similarity_transform(point_3d, similarity_transform):
    if similarity_transform is None:
        return None
    point_3d = np.asarray(point_3d, dtype=np.float64).reshape(3)
    scale = float(similarity_transform['scale'])
    rotation = np.asarray(similarity_transform['rotation'], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(similarity_transform['translation'], dtype=np.float64).reshape(3)
    transformed_point = scale * (rotation @ point_3d) + translation
    if not np.all(np.isfinite(transformed_point)):
        return None
    return transformed_point


def _seed_recovery_points_from_baseline_alignment(
    baseline_data,
    scaffold_data,
    recovery_data,
    scaffold_point_ids,
    *,
    min_points_for_camera,
    label="Two-stage alignment seeding",
):
    seed_store = recovery_data.setdefault('_secondary_seed_points_3d', {})
    common_point_ids = [
        point_id
        for point_id in sorted(set(scaffold_point_ids or []), key=_stable_sort_key)
        if point_id in (baseline_data.get('points_3d') or {}) and point_id in (scaffold_data.get('points_3d') or {})
    ]
    min_required = max(6, int(min_points_for_camera) + 2)
    if len(common_point_ids) < min_required:
        return {
            'applied': False,
            'reason': 'too_few_common_points',
            'common_point_count': int(len(common_point_ids)),
            'seeded_points': 0,
        }

    source_points = np.array(
        [baseline_data['points_3d'][point_id] for point_id in common_point_ids],
        dtype=np.float64,
    ).reshape(-1, 3)
    target_points = np.array(
        [scaffold_data['points_3d'][point_id] for point_id in common_point_ids],
        dtype=np.float64,
    ).reshape(-1, 3)
    similarity_transform = _estimate_similarity_transform(source_points, target_points)
    if similarity_transform is None:
        return {
            'applied': False,
            'reason': 'similarity_fit_failed',
            'common_point_count': int(len(common_point_ids)),
            'seeded_points': 0,
        }

    target_mean = float(recovery_data.get('precision_target_mean_px', 0.5))
    target_p95 = float(recovery_data.get('precision_target_p95_px', 1.0))
    target_max = float(recovery_data.get('precision_target_max_px', 1.5))
    seeded_points = 0
    relaxed_seeded_points = 0
    candidate_count = 0
    accepted_with_subset = 0

    for point_id in sorted((baseline_data.get('points_3d') or {}).keys(), key=_stable_sort_key):
        if point_id in recovery_data.get('points_3d', {}):
            continue
        point_observations = {
            str(camera_id): recovery_data['camera_points'][str(camera_id)][point_id]
            for camera_id in sorted(recovery_data.get('cameras', {}).keys(), key=_stable_sort_key)
            if point_id in recovery_data.get('camera_points', {}).get(str(camera_id), {})
        }
        if len(point_observations) < 2:
            continue

        transformed_point = _apply_similarity_transform(
            baseline_data['points_3d'][point_id],
            similarity_transform,
        )
        if transformed_point is None:
            continue

        point_observation_confidences = triangulation._extract_point_observation_confidences(
            recovery_data.get('observation_confidences'),
            point_id,
            point_observations.keys(),
        )
        candidate_count += 1
        accepted, refined_point, metrics = triangulation.evaluate_multiview_point(
            transformed_point,
            point_observations,
            recovery_data['cameras'],
            recovery_data['K'],
            recovery_data.get('dist_coeffs'),
            allow_subset=True,
            min_subset_views=2,
            observation_confidences=point_observation_confidences,
            point_id=point_id,
            points_3d_context=recovery_data.get('points_3d'),
            line_support_data=recovery_data.get('line_support_data'),
            camera_intrinsics=_collect_camera_intrinsics_map(recovery_data),
        )
        if metrics is None:
            continue

        used_camera_ids = [
            str(camera_id)
            for camera_id in metrics.get('used_camera_ids', metrics.get('camera_ids', []))
        ]
        seeded_mean = float(metrics.get('mean_error', float('inf')))
        seeded_max = float(metrics.get('max_error', float('inf')))
        seeded_inlier_ratio = float(metrics.get('inlier_ratio', 0.0))
        strong_seed_ok = (
            bool(accepted) and
            len(used_camera_ids) >= 2 and
            seeded_inlier_ratio >= 0.60 and
            seeded_mean <= max(target_mean * 5.0, 2.75) and
            seeded_max <= max(target_max * 5.0, 5.50)
        )
        relaxed_seed_ok = (
            bool(accepted) and
            len(used_camera_ids) >= 2 and
            seeded_inlier_ratio >= 0.45 and
            seeded_mean <= max(target_p95 * 6.50, 6.5) and
            seeded_max <= max(target_max * 10.50, 15.5)
        )
        if not strong_seed_ok and not relaxed_seed_ok:
            continue

        if len(used_camera_ids) < len(point_observations):
            accepted_with_subset += 1
        seeded_point = np.asarray(refined_point, dtype=np.float32).reshape(3)
        seed_store[point_id] = np.array(seeded_point, copy=True)
        if strong_seed_ok:
            recovery_data['points_3d'][point_id] = seeded_point
            seeded_points += 1
        else:
            relaxed_seeded_points += 1

    print(
        f"{label}: common={len(common_point_ids)}, candidates={candidate_count}, "
        f"seeded={seeded_points}, relaxed={relaxed_seeded_points}, subset_accepted={accepted_with_subset}"
    )
    return {
        'applied': True,
        'reason': 'ok',
        'common_point_count': int(len(common_point_ids)),
        'seed_candidate_count': int(candidate_count),
        'seeded_points': int(seeded_points),
        'relaxed_seeded_points': int(relaxed_seeded_points),
        'subset_accepted': int(accepted_with_subset),
        'scale': float(similarity_transform['scale']),
    }


def _run_fixed_camera_recovery_pass(
    calib_data,
    *,
    min_points_for_camera,
    label="Fixed-camera recovery",
    max_rounds=2,
):
    if not calib_data.get('cameras') or not calib_data.get('points_3d'):
        return {
            'applied': False,
            'reason': 'empty_scene',
        }

    target_mean = float(calib_data.get('precision_target_mean_px', 0.5))
    target_p95 = float(calib_data.get('precision_target_p95_px', 1.0))
    target_max = float(calib_data.get('precision_target_max_px', 1.5))
    max_rounds = max(1, int(max_rounds))

    before_metrics = _summarize_reconstruction_metrics(calib_data)
    seed_store = calib_data.setdefault('_secondary_seed_points_3d', {})
    total_prebackfilled = 0
    total_relaxed_prebackfilled = 0
    total_reintegrated = 0
    total_retriangulated = 0
    rounds_completed = 0

    for round_index in range(max_rounds):
        rounds_completed += 1
        round_added = 0
        round_retriangulated = 0

        backfilled_points = triangulation.triangulate_remaining_tracks(
            calib_data['points_3d'],
            calib_data['cameras'],
            calib_data['camera_points'],
            calib_data['K'],
            calib_data.get('dist_coeffs'),
            min_track_length=2,
            strict_track_consistency=bool(calib_data.get('strict_track_consistency', True)),
            debug_logging=bool(calib_data.get('debug_logging', False)),
            observation_confidences=calib_data.get('observation_confidences'),
            line_support_data=calib_data.get('line_support_data'),
        )
        if backfilled_points:
            existing_point_ids = set(calib_data.get('points_3d', {}).keys())
            calib_data['points_3d'].update(backfilled_points)
            for point_id, point_3d in backfilled_points.items():
                seed_store[point_id] = np.asarray(point_3d, dtype=np.float32).reshape(3)
            added_count = sum(
                1
                for point_id in backfilled_points.keys()
                if point_id not in existing_point_ids
            )
            total_prebackfilled += int(added_count)
            round_added += int(added_count)
            if added_count > 0:
                print(
                    f"{label}: round {round_index + 1}, global backfill added {added_count} points"
                )

        if (
            round_added <= 0 and
            bool(calib_data.get('strict_track_consistency', True))
        ):
            relaxed_backfilled_points = triangulation.triangulate_remaining_tracks(
                calib_data['points_3d'],
                calib_data['cameras'],
                calib_data['camera_points'],
                calib_data['K'],
                calib_data.get('dist_coeffs'),
                min_track_length=2,
                strict_track_consistency=False,
                debug_logging=bool(calib_data.get('debug_logging', False)),
                observation_confidences=calib_data.get('observation_confidences'),
                line_support_data=calib_data.get('line_support_data'),
            )
            if relaxed_backfilled_points:
                existing_point_ids = set(calib_data.get('points_3d', {}).keys())
                calib_data['points_3d'].update(relaxed_backfilled_points)
                for point_id, point_3d in relaxed_backfilled_points.items():
                    seed_store[point_id] = np.asarray(point_3d, dtype=np.float32).reshape(3)
                relaxed_added_count = sum(
                    1
                    for point_id in relaxed_backfilled_points.keys()
                    if point_id not in existing_point_ids
                )
                total_relaxed_prebackfilled += int(relaxed_added_count)
                round_added += int(relaxed_added_count)
                if relaxed_added_count > 0:
                    print(
                        f"{label}: round {round_index + 1}, relaxed backfill added "
                        f"{relaxed_added_count} points"
                    )

        reintegrated_points = reintegrate_backfillable_tracks(
            calib_data,
            target_mean=target_mean,
            target_p95=target_p95,
            target_max=target_max,
            max_candidates=max(6, min(12, len(calib_data.get('camera_points', {})) * 2)),
        )
        if reintegrated_points > 0:
            total_reintegrated += int(reintegrated_points)
            round_added += int(reintegrated_points)

        retriangulated_points = triangulation.retriangulate_high_error_points(
            calib_data['points_3d'],
            calib_data['cameras'],
            calib_data['camera_points'],
            calib_data['K'],
            calib_data.get('dist_coeffs'),
            max_points=max(6, min(18, max(1, len(calib_data.get('points_3d', {})) // 3))),
            min_track_length=max(3, int(min_points_for_camera)),
            debug_logging=bool(calib_data.get('debug_logging', False)),
            observation_confidences=calib_data.get('observation_confidences'),
            line_support_data=calib_data.get('line_support_data'),
            camera_intrinsics=_collect_camera_intrinsics_map(calib_data),
        )
        if retriangulated_points > 0:
            total_retriangulated += int(retriangulated_points)
            round_retriangulated += int(retriangulated_points)

        if round_added <= 0 and round_retriangulated <= 0:
            rounds_completed = int(round_index)
            break

    after_metrics = _summarize_reconstruction_metrics(calib_data)
    scene_not_broken = (
        after_metrics['mean_error'] <= max(
            before_metrics['mean_error'] + 0.20,
            before_metrics['mean_error'] * 1.30,
            target_mean * 2.25,
            0.95,
        ) and
        after_metrics['p95_error'] <= max(
            before_metrics['p95_error'] + 0.50,
            before_metrics['p95_error'] * 1.30,
            target_p95 * 2.25,
            2.80,
        ) and
        after_metrics['max_error'] <= max(
            before_metrics['max_error'] + 1.20,
            before_metrics['max_error'] * 1.25,
            target_max * 3.25,
            8.50,
        )
    )

    summary = {
        'applied': True,
        'reason': 'ok' if scene_not_broken else 'prefill_scene_regression',
        'rounds_completed': int(max(0, rounds_completed)),
        'prebackfilled_points': int(total_prebackfilled),
        'relaxed_prebackfilled_points': int(total_relaxed_prebackfilled),
        'reintegrated_points': int(total_reintegrated),
        'retriangulated_points': int(total_retriangulated),
        'before_metrics': before_metrics,
        'after_metrics': after_metrics,
    }

    if not scene_not_broken:
        print(
            f"{label}: prefill temporarily raised error envelope "
            f"(mean {before_metrics['mean_error']:.4f}px -> {after_metrics['mean_error']:.4f}px, "
            f"p95 {before_metrics['p95_error']:.4f}px -> {after_metrics['p95_error']:.4f}px, "
            f"max {before_metrics['max_error']:.4f}px -> {after_metrics['max_error']:.4f}px); "
            "continuing to full finalize"
        )
        return summary

    print(
        f"{label}: primary points {before_metrics['primary_points']} -> {after_metrics['primary_points']}, "
        f"mean {before_metrics['mean_error']:.4f}px -> {after_metrics['mean_error']:.4f}px, "
        f"p95 {before_metrics['p95_error']:.4f}px -> {after_metrics['p95_error']:.4f}px"
    )
    return summary


def stabilize_reconstruction_from_pose_scaffold(
    calib_data,
    min_points_for_camera=4,
    *,
    label="Pose scaffold",
    bundle_ftol=1e-7,
    bundle_max_nfev=300,
):
    if not bool(calib_data.get('pose_scaffold_enabled', True)):
        return {
            'applied': False,
            'reason': 'disabled',
        }
    if not calib_data.get('cameras') or not calib_data.get('points_3d'):
        return {
            'applied': False,
            'reason': 'empty_scene',
        }

    scaffold_point_ids, scaffold_meta = _select_pose_scaffold_point_ids(
        calib_data,
        min_points_for_camera=min_points_for_camera,
    )
    min_required_points = max(8, int(min_points_for_camera) * 2)
    if len(scaffold_point_ids) < min_required_points:
        return {
            'applied': False,
            'reason': 'too_few_scaffold_points',
            'selected_count': int(len(scaffold_point_ids)),
            'meta': scaffold_meta,
        }

    scaffold_confidences = _build_pose_scaffold_observation_confidences(
        calib_data,
        scaffold_point_ids,
        nonmember_confidence=float(calib_data.get('pose_scaffold_nonmember_confidence', 0.22)),
    )
    scaffold_confidences = _build_quality_aware_observation_confidences(
        calib_data,
        base_confidences=scaffold_confidences,
        label=label,
    )

    state_snapshot = {
        'cameras': copy.deepcopy(calib_data.get('cameras', {})),
        'points_3d': copy.deepcopy(calib_data.get('points_3d', {})),
    }
    before_mean, _, _ = calculate_reprojection_errors(calib_data)
    before_stats = summarize_reprojection_error_distribution(calib_data, top_k=0)

    points_3d_opt, cameras_opt = bundle_adjustment.bundle_adjust_step(
        calib_data['points_3d'],
        calib_data['cameras'],
        calib_data['camera_points'],
        calib_data['K'],
        calib_data.get('dist_coeffs'),
        ftol=bundle_ftol,
        max_nfev=bundle_max_nfev,
        debug_logging=bool(calib_data.get('debug_logging', False)),
        observation_confidences=scaffold_confidences,
        camera_intrinsics=_collect_camera_intrinsics_map(calib_data),
    )
    calib_data['points_3d'] = points_3d_opt
    calib_data['cameras'] = cameras_opt
    calib_data['_pose_scaffold_point_ids'] = sorted(scaffold_point_ids, key=_stable_sort_key)

    after_mean, _, _ = calculate_reprojection_errors(calib_data)
    after_stats = summarize_reprojection_error_distribution(calib_data, top_k=0)
    scene_not_broken = (
        after_mean <= max(before_mean + 0.12, before_mean * 1.18, 0.90) and
        after_stats['p95'] <= max(before_stats['p95'] + 0.30, before_stats['p95'] * 1.20, 2.20) and
        after_stats['max'] <= max(before_stats['max'] + 0.80, before_stats['max'] * 1.18, 4.50)
    )
    if not scene_not_broken:
        calib_data['cameras'] = state_snapshot['cameras']
        calib_data['points_3d'] = state_snapshot['points_3d']
        return {
            'applied': False,
            'reason': 'global_regression',
            'selected_count': int(len(scaffold_point_ids)),
            'before_mean': float(before_mean),
            'after_mean': float(after_mean),
            'before_p95': float(before_stats.get('p95', before_mean)),
            'after_p95': float(after_stats.get('p95', after_mean)),
            'meta': scaffold_meta,
        }

    print(
        f"{label}: стабилизация поз по каркасу из {len(scaffold_point_ids)} треков "
        f"(global mean {before_mean:.4f}px -> {after_mean:.4f}px, "
        f"p95 {before_stats['p95']:.4f}px -> {after_stats['p95']:.4f}px)"
    )
    return {
        'applied': True,
        'selected_count': int(len(scaffold_point_ids)),
        'before_mean': float(before_mean),
        'after_mean': float(after_mean),
        'before_p95': float(before_stats.get('p95', before_mean)),
        'after_p95': float(after_stats.get('p95', after_mean)),
        'meta': scaffold_meta,
    }


def _run_two_stage_pose_scaffold_recovery(
    calib_data,
    *,
    start_time,
    initial_pair,
    min_points_for_camera,
    bundle_method,
    bundle_ftol,
    max_bundle_iterations,
    ransac_threshold,
    confidence,
    max_attempts,
    focal_range,
    adapt_initial_focal,
    check_focal_consistency,
    auto_correct_focal,
    force_same_focal,
    progress_callback,
):
    original_input_data = copy.deepcopy(calib_data)

    baseline_data = copy.deepcopy(original_input_data)
    baseline_data['two_stage_pose_scaffold_recovery'] = False
    baseline_data['_two_stage_pose_scaffold_recovery_active'] = True

    print("\nTwo-stage pose scaffold recovery: stage A / baseline full reconstruction")
    baseline_success = perform_full_reconstruction(
        baseline_data,
        initial_pair=initial_pair,
        min_points_for_camera=min_points_for_camera,
        bundle_method=bundle_method,
        bundle_ftol=bundle_ftol,
        max_bundle_iterations=max_bundle_iterations,
        ransac_threshold=ransac_threshold,
        confidence=confidence,
        max_attempts=max_attempts,
        focal_range=focal_range,
        adapt_initial_focal=adapt_initial_focal,
        check_focal_consistency=check_focal_consistency,
        auto_correct_focal=auto_correct_focal,
        force_same_focal=force_same_focal,
        progress_callback=progress_callback,
    )
    if not baseline_success:
        return False

    scaffold_point_ids, scaffold_meta = _select_pose_scaffold_point_ids(
        baseline_data,
        min_points_for_camera=min_points_for_camera,
    )
    expanded_scaffold_point_ids, scaffold_expand_meta = _expand_two_stage_pose_scaffold_point_ids(
        baseline_data,
        scaffold_point_ids,
        min_points_for_camera=min_points_for_camera,
    )
    if len(expanded_scaffold_point_ids) > len(scaffold_point_ids):
        print(
            "Two-stage pose scaffold recovery: "
            f"expanded scaffold {len(scaffold_point_ids)} -> {len(expanded_scaffold_point_ids)} tracks "
            f"(target={scaffold_expand_meta.get('target_count')}, "
            f"min camera support={scaffold_expand_meta.get('target_camera_support')})"
        )
        scaffold_point_ids = expanded_scaffold_point_ids
    min_scaffold_points = max(
        8,
        int(min_points_for_camera) * 2,
        int(calib_data.get('two_stage_pose_scaffold_min_points', 12)),
    )
    if len(scaffold_point_ids) < min_scaffold_points:
        baseline_data['_two_stage_pose_scaffold_summary'] = {
            'enabled': True,
            'accepted': False,
            'reason': 'too_few_scaffold_points',
            'selected_count': int(len(scaffold_point_ids)),
            'min_required_points': int(min_scaffold_points),
            'scaffold_meta': scaffold_meta,
            'scaffold_expand_meta': scaffold_expand_meta,
            'baseline_metrics': _summarize_reconstruction_metrics(baseline_data),
        }
        print(
            "Two-stage pose scaffold recovery: "
            f"fallback to baseline, scaffold points {len(scaffold_point_ids)} < {min_scaffold_points}"
        )
        _replace_calibration_result(calib_data, baseline_data)
        return True

    scaffold_data = _build_point_filtered_calibration_view(original_input_data, scaffold_point_ids)
    scaffold_data['two_stage_pose_scaffold_recovery'] = False
    scaffold_data['_two_stage_pose_scaffold_recovery_active'] = True
    scaffold_data['_multiview_refine_mode'] = str(
        baseline_data.get('_multiview_refine_mode', 'full') or 'full'
    )
    if baseline_data.get('K') is not None:
        scaffold_data['K'] = np.array(baseline_data['K'], copy=True)
    if baseline_data.get('dist_coeffs') is not None:
        scaffold_data['dist_coeffs'] = np.array(baseline_data['dist_coeffs'], copy=True)
    _copy_individual_intrinsics(scaffold_data, baseline_data)
    if 'project_level_focal_sweep' in baseline_data:
        scaffold_data['project_level_focal_sweep'] = copy.deepcopy(
            baseline_data.get('project_level_focal_sweep')
        )

    print(
        "Two-stage pose scaffold recovery: "
        f"stage B / scaffold-only reconstruction on {len(scaffold_point_ids)} tracks"
    )
    scaffold_success = perform_full_reconstruction(
        scaffold_data,
        initial_pair=None,
        min_points_for_camera=min_points_for_camera,
        bundle_method=bundle_method,
        bundle_ftol=bundle_ftol,
        max_bundle_iterations=max_bundle_iterations,
        ransac_threshold=ransac_threshold,
        confidence=confidence,
        max_attempts=max_attempts,
        focal_range=focal_range,
        adapt_initial_focal=False,
        check_focal_consistency=check_focal_consistency,
        auto_correct_focal=auto_correct_focal,
        force_same_focal=force_same_focal,
        progress_callback=progress_callback,
    )
    if not scaffold_success or not scaffold_data.get('cameras') or not scaffold_data.get('points_3d'):
        baseline_data['_two_stage_pose_scaffold_summary'] = {
            'enabled': True,
            'accepted': False,
            'reason': 'scaffold_reconstruction_failed',
            'selected_count': int(len(scaffold_point_ids)),
            'scaffold_meta': scaffold_meta,
            'scaffold_expand_meta': scaffold_expand_meta,
            'baseline_metrics': _summarize_reconstruction_metrics(baseline_data),
        }
        print("Two-stage pose scaffold recovery: scaffold-only stage failed, fallback to baseline")
        _replace_calibration_result(calib_data, baseline_data)
        return True

    recovery_data = copy.deepcopy(original_input_data)
    recovery_data['two_stage_pose_scaffold_recovery'] = False
    recovery_data['_two_stage_pose_scaffold_recovery_active'] = True
    recovery_data['_multiview_refine_mode'] = str(
        scaffold_data.get('_multiview_refine_mode', 'full') or 'full'
    )
    recovery_data['precision_cleanup_enabled'] = False
    recovery_data['cameras'] = copy.deepcopy(scaffold_data.get('cameras', {}))
    recovery_data['points_3d'] = copy.deepcopy(scaffold_data.get('points_3d', {}))
    recovery_data['secondary_points_3d'] = {}
    recovery_data['_secondary_seed_points_3d'] = {}
    recovery_data['reconstruction_ids'] = list(scaffold_data.get('reconstruction_ids', []))
    if scaffold_data.get('K') is not None:
        recovery_data['K'] = np.array(scaffold_data['K'], copy=True)
    if scaffold_data.get('dist_coeffs') is not None:
        recovery_data['dist_coeffs'] = np.array(scaffold_data['dist_coeffs'], copy=True)
    _copy_individual_intrinsics(recovery_data, scaffold_data)
    if 'project_level_focal_sweep' in baseline_data:
        recovery_data['project_level_focal_sweep'] = copy.deepcopy(
            baseline_data.get('project_level_focal_sweep')
        )

    alignment_seed_stats = _seed_recovery_points_from_baseline_alignment(
        baseline_data,
        scaffold_data,
        recovery_data,
        scaffold_point_ids,
        min_points_for_camera=min_points_for_camera,
    )

    try:
        prebackfilled_points = triangulation.triangulate_remaining_tracks(
            recovery_data['points_3d'],
            recovery_data['cameras'],
            recovery_data['camera_points'],
            recovery_data['K'],
            recovery_data.get('dist_coeffs'),
            min_track_length=2,
            strict_track_consistency=bool(recovery_data.get('strict_track_consistency', True)),
            debug_logging=bool(recovery_data.get('debug_logging', False)),
            observation_confidences=recovery_data.get('observation_confidences'),
            line_support_data=recovery_data.get('line_support_data'),
        )
        if prebackfilled_points:
            added_point_count = sum(
                1
                for point_id in prebackfilled_points.keys()
                if point_id not in recovery_data['points_3d']
            )
            recovery_data['points_3d'].update(prebackfilled_points)
            print(
                "Two-stage pose scaffold recovery: "
                f"pre-backfill added {added_point_count} full-track points before finalize"
            )
    except Exception as exc:
        print(f"Two-stage pose scaffold recovery: pre-backfill skipped due to error: {exc}")

    fixed_camera_recovery_stats = _run_fixed_camera_recovery_pass(
        recovery_data,
        min_points_for_camera=min_points_for_camera,
        label="Two-stage fixed-camera recovery",
        max_rounds=2,
    )
    for point_id, point_3d in (recovery_data.get('points_3d') or {}).items():
        recovery_data.setdefault('_secondary_seed_points_3d', {})[point_id] = copy.deepcopy(point_3d)

    print("Two-stage pose scaffold recovery: stage C / full-track recovery from scaffold poses")
    recovery_success = _finalize_reconstruction(
        recovery_data,
        start_time=start_time,
        min_points_for_camera=min_points_for_camera,
        max_bundle_iterations=max_bundle_iterations,
        focal_range=focal_range,
        force_same_focal=force_same_focal,
        ransac_threshold=ransac_threshold,
        confidence=confidence,
        max_attempts=max_attempts,
        progress_callback=progress_callback,
    )
    if not recovery_success:
        baseline_data['_two_stage_pose_scaffold_summary'] = {
            'enabled': True,
            'accepted': False,
            'reason': 'full_recovery_failed',
            'selected_count': int(len(scaffold_point_ids)),
            'scaffold_meta': scaffold_meta,
            'scaffold_expand_meta': scaffold_expand_meta,
            'baseline_metrics': _summarize_reconstruction_metrics(baseline_data),
            'scaffold_metrics': _summarize_reconstruction_metrics(scaffold_data),
        }
        print("Two-stage pose scaffold recovery: full-track recovery failed, fallback to baseline")
        _replace_calibration_result(calib_data, baseline_data)
        return True

    extra_secondary_points = augment_secondary_point_cloud_from_seed_points(
        recovery_data,
        target_p95=float(recovery_data.get('precision_target_p95_px', 1.0)),
        target_max=float(recovery_data.get('precision_target_max_px', 1.5)),
        max_candidates=48,
    )

    baseline_metrics = _summarize_reconstruction_metrics(baseline_data)
    scaffold_metrics = _summarize_reconstruction_metrics(scaffold_data)
    recovery_metrics = _summarize_reconstruction_metrics(recovery_data)
    accepted, accept_reason = _should_accept_two_stage_pose_scaffold_candidate(
        baseline_metrics,
        recovery_metrics,
    )
    summary = {
        'enabled': True,
        'accepted': bool(accepted),
        'reason': str(accept_reason),
        'selected_count': int(len(scaffold_point_ids)),
        'min_required_points': int(min_scaffold_points),
        'scaffold_meta': scaffold_meta,
        'scaffold_expand_meta': scaffold_expand_meta,
        'baseline_metrics': baseline_metrics,
        'scaffold_metrics': scaffold_metrics,
        'recovery_metrics': recovery_metrics,
        'alignment_seed': alignment_seed_stats,
        'fixed_camera_recovery': fixed_camera_recovery_stats,
        'extra_secondary_points': int(extra_secondary_points),
    }

    if accepted:
        recovery_data['_two_stage_pose_scaffold_summary'] = summary
        print(
            "Two-stage pose scaffold recovery: accepted "
            f"({accept_reason}, total points {baseline_metrics['total_points']} -> {recovery_metrics['total_points']}, "
            f"mean {baseline_metrics['mean_error']:.4f}px -> {recovery_metrics['mean_error']:.4f}px)"
        )
        _replace_calibration_result(calib_data, recovery_data)
        return True

    baseline_data['_two_stage_pose_scaffold_summary'] = summary
    print(
        "Two-stage pose scaffold recovery: fallback to baseline "
        f"({accept_reason}, total points {baseline_metrics['total_points']} vs {recovery_metrics['total_points']}, "
        f"mean {baseline_metrics['mean_error']:.4f}px vs {recovery_metrics['mean_error']:.4f}px)"
    )
    _replace_calibration_result(calib_data, baseline_data)
    return True


def _collect_camera_pose_correspondences(calib_data, camera_id):
    camera_id = str(camera_id)
    if camera_id not in calib_data.get('cameras', {}):
        return None

    observations = calib_data.get('camera_points', {}).get(camera_id, {})
    if not observations:
        return None

    R, t = calib_data['cameras'][camera_id]
    points_3d = []
    points_2d = []
    point_ids = []
    errors = []

    for point_id, point_2d in observations.items():
        if point_id not in calib_data.get('points_3d', {}):
            continue

        point_3d = np.asarray(calib_data['points_3d'][point_id], dtype=np.float64).reshape(3)
        point_2d = np.asarray(point_2d, dtype=np.float64).reshape(2)
        projected_point = utils.project_point(
            point_3d,
            R,
            t,
            _get_camera_matrix(calib_data, camera_id),
            calib_data.get('dist_coeffs'),
        )
        error = float(np.linalg.norm(projected_point - point_2d))

        points_3d.append(point_3d)
        points_2d.append(point_2d)
        point_ids.append(point_id)
        errors.append(error)

    if len(points_3d) < 4:
        return None

    return {
        'point_ids': point_ids,
        'points_3d': np.asarray(points_3d, dtype=np.float64),
        'points_2d': np.asarray(points_2d, dtype=np.float64),
        'errors': np.asarray(errors, dtype=np.float64),
        'confidences': np.asarray(
            [_lookup_observation_confidence(calib_data, camera_id, point_id) for point_id in point_ids],
            dtype=np.float64,
        ),
    }


def _evaluate_camera_pose_candidate(points_3d, points_2d, R, t, K, dist_coeffs=None):
    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    points_2d = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    projected_points, _ = cv2.projectPoints(
        points_3d,
        rvec,
        t,
        np.asarray(K, dtype=np.float64),
        None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1),
    )
    errors = np.linalg.norm(projected_points.reshape(-1, 2) - points_2d, axis=1)
    depths = (R @ points_3d.T + t)[2]
    return {
        'count': int(len(errors)),
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'p90': float(np.percentile(errors, 90)),
        'p95': float(np.percentile(errors, 95)),
        'max': float(np.max(errors)),
        'front_ratio': float(np.mean(depths > 0.01)) if len(depths) > 0 else 0.0,
        'errors': np.asarray(errors, dtype=np.float64),
    }


def _normalize_observation_confidences_array(observation_confidences, count, default=1.0):
    if observation_confidences is None:
        return np.full(int(count), float(default), dtype=np.float64)

    try:
        confidences = np.asarray(observation_confidences, dtype=np.float64).reshape(-1)
    except Exception:
        confidences = np.full(int(count), float(default), dtype=np.float64)

    if confidences.size != int(count):
        return np.full(int(count), float(default), dtype=np.float64)
    return np.clip(confidences, 0.15, 1.0)


def _should_use_confidence_guided_pose_support(observation_confidences, min_confidence=0.28):
    confidences = _normalize_observation_confidences_array(
        observation_confidences,
        0 if observation_confidences is None else len(np.asarray(observation_confidences).reshape(-1)),
        default=1.0,
    )
    if confidences.size == 0:
        return False

    if confidences.size < 6:
        return False

    min_confidence = float(np.clip(min_confidence, 0.15, 1.0))
    low_confidence_threshold = max(0.24, min_confidence - 0.04)
    strong_confidence_threshold = max(0.94, min_confidence + 0.66)

    low_confidence_mask = confidences <= low_confidence_threshold
    strong_confidence_mask = confidences >= strong_confidence_threshold

    low_confidence_count = int(np.count_nonzero(low_confidence_mask))
    strong_confidence_count = int(np.count_nonzero(strong_confidence_mask))
    low_confidence_fraction = float(low_confidence_count / max(confidences.size, 1))
    confidence_spread = float(np.percentile(confidences, 90) - np.percentile(confidences, 10))
    upper_quartile_confidence = float(np.percentile(confidences, 75))
    p90_confidence = float(np.percentile(confidences, 90))

    return (
        strong_confidence_count >= 4 and
        low_confidence_count >= max(2, int(np.ceil(confidences.size * 0.12))) and
        low_confidence_fraction >= 0.10 and
        confidence_spread >= 0.08 and
        upper_quartile_confidence >= max(0.94, min_confidence + 0.62) and
        p90_confidence >= 0.97
    )


def _select_pose_refine_support_indices(
    errors,
    observation_confidences=None,
    *,
    min_points=4,
    preferred_indices=None,
    min_confidence=0.28,
    max_fraction=0.75,
):
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    if errors.size == 0:
        return np.zeros(0, dtype=np.int32)

    confidences = _normalize_observation_confidences_array(observation_confidences, errors.size)
    finite_mask = np.isfinite(errors)
    finite_indices = np.where(finite_mask)[0]
    if finite_indices.size <= int(min_points):
        return finite_indices.astype(np.int32)

    preferred_mask = np.zeros(errors.size, dtype=bool)
    if preferred_indices is not None:
        try:
            preferred_indices = np.asarray(preferred_indices, dtype=np.int32).reshape(-1)
            preferred_indices = preferred_indices[(preferred_indices >= 0) & (preferred_indices < errors.size)]
            preferred_mask[preferred_indices] = True
        except Exception:
            preferred_mask[:] = False
    preferred_mask &= finite_mask

    min_confidence = float(np.clip(min_confidence, 0.15, 1.0))
    strong_mask = finite_mask & (confidences >= min_confidence)
    basis_indices = np.where(strong_mask)[0]
    if basis_indices.size < int(min_points):
        basis_indices = finite_indices

    basis_errors = errors[basis_indices]
    basis_median = float(np.median(basis_errors))
    basis_mad = float(stats.median_abs_deviation(basis_errors, scale='normal'))
    robust_scale = max(0.60, basis_mad, basis_median * 0.35)
    threshold = basis_median + 2.0 * robust_scale

    selected_mask = finite_mask & (
        preferred_mask |
        (errors <= threshold) |
        ((confidences >= min_confidence) & (errors <= basis_median + 2.6 * robust_scale))
    )
    selected_indices = np.where(selected_mask)[0]

    quality = errors / np.clip(confidences, 0.20, 1.0)
    ranked_indices = sorted(
        finite_indices.tolist(),
        key=lambda idx: (
            0 if preferred_mask[idx] else 1,
            0 if confidences[idx] >= min_confidence else 1,
            float(quality[idx]),
            float(errors[idx]),
            -float(confidences[idx]),
        ),
    )

    target_cap = max(int(min_points), int(np.ceil(finite_indices.size * float(max_fraction))))
    target_cap = max(target_cap, int(min_points) + 2)
    if selected_indices.size < int(min_points):
        selected_indices = np.asarray(ranked_indices[:max(int(min_points), min(target_cap, len(ranked_indices)))], dtype=np.int32)
    elif selected_indices.size > target_cap:
        selected_set = set(selected_indices.tolist())
        limited = [idx for idx in ranked_indices if idx in selected_set][:target_cap]
        selected_indices = np.asarray(limited, dtype=np.int32)

    return np.asarray(sorted(set(int(idx) for idx in selected_indices.tolist())), dtype=np.int32)


def _select_residual_support_indices(
    errors,
    *,
    min_points=4,
    preferred_indices=None,
    min_fraction=0.45,
    max_fraction=0.70,
):
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    if errors.size == 0:
        return np.zeros(0, dtype=np.int32)

    finite_indices = np.where(np.isfinite(errors))[0]
    if finite_indices.size <= int(min_points):
        return finite_indices.astype(np.int32)

    target_min = max(int(min_points) + 2, int(np.ceil(finite_indices.size * float(min_fraction))))
    target_cap = max(target_min, int(np.ceil(finite_indices.size * float(max_fraction))))
    target_cap = min(target_cap, finite_indices.size)

    selected = set()
    if preferred_indices is not None:
        try:
            preferred_indices = np.asarray(preferred_indices, dtype=np.int32).reshape(-1)
            for idx in preferred_indices.tolist():
                idx = int(idx)
                if 0 <= idx < errors.size and np.isfinite(errors[idx]):
                    selected.add(idx)
        except Exception:
            pass

    ranked = np.argsort(errors[finite_indices])
    ranked_indices = finite_indices[ranked].tolist()
    for idx in ranked_indices:
        selected.add(int(idx))
        if len(selected) >= target_min:
            break

    if len(selected) > target_cap:
        selected = set(ranked_indices[:target_cap])

    return np.asarray(sorted(selected), dtype=np.int32)


def _stabilize_pose_candidate_with_support(
    points_3d,
    points_2d,
    K,
    dist_coeffs,
    R,
    t,
    *,
    min_points=4,
    preferred_indices=None,
    observation_confidences=None,
    support_min_confidence=0.28,
):
    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    points_2d = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    if len(points_3d) < max(int(min_points), 4):
        return None

    observation_confidences = _normalize_observation_confidences_array(
        observation_confidences,
        len(points_3d),
    )
    base_R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    base_t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    camera_K = np.asarray(K, dtype=np.float64)
    dist_coeffs = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

    initial_stats = _evaluate_camera_pose_candidate(
        points_3d,
        points_2d,
        base_R,
        base_t,
        camera_K,
        dist_coeffs,
    )
    use_guided_support = _should_use_confidence_guided_pose_support(
        observation_confidences,
        min_confidence=support_min_confidence,
    )
    if use_guided_support:
        support_indices = _select_pose_refine_support_indices(
            initial_stats.get('errors'),
            observation_confidences=observation_confidences,
            min_points=max(int(min_points), 4),
            preferred_indices=preferred_indices,
            min_confidence=support_min_confidence,
        )
    else:
        support_indices = _select_residual_support_indices(
            initial_stats.get('errors'),
            min_points=max(int(min_points), 4),
            preferred_indices=preferred_indices,
            min_fraction=0.45,
            max_fraction=0.72,
        )
    if len(support_indices) < max(int(min_points), 4):
        return None

    rvec_seed, _ = cv2.Rodrigues(base_R)
    tvec_seed = base_t.copy()
    best_candidate = None

    for _ in range(2):
        try:
            rvec_seed, tvec_seed = cv2.solvePnPRefineLM(
                points_3d[support_indices],
                points_2d[support_indices],
                camera_K,
                dist_coeffs,
                rvec_seed,
                tvec_seed,
            )
        except cv2.error:
            break

        refined_R, _ = cv2.Rodrigues(rvec_seed)
        refined_stats = _evaluate_camera_pose_candidate(
            points_3d,
            points_2d,
            refined_R,
            tvec_seed,
            camera_K,
            dist_coeffs,
        )
        if use_guided_support:
            refined_support_indices = _select_pose_refine_support_indices(
                refined_stats.get('errors'),
                observation_confidences=observation_confidences,
                min_points=max(int(min_points), 4),
                preferred_indices=preferred_indices,
                min_confidence=support_min_confidence,
            )
        else:
            refined_support_indices = _select_residual_support_indices(
                refined_stats.get('errors'),
                min_points=max(int(min_points), 4),
                preferred_indices=preferred_indices,
                min_fraction=0.45,
                max_fraction=0.72,
            )
        if len(refined_support_indices) < max(int(min_points), 4):
            break

        support_indices = refined_support_indices
        support_errors = refined_stats['errors'][support_indices]
        candidate = {
            'R': np.asarray(refined_R, dtype=np.float32),
            't': np.asarray(tvec_seed, dtype=np.float32).reshape(3, 1),
            'stats': refined_stats,
            'support_indices': np.asarray(support_indices, dtype=np.int32),
            'support_mean': float(np.mean(support_errors)),
            'support_median': float(np.median(support_errors)),
            'support_p95': float(np.percentile(support_errors, 95)),
        }
        if (
            best_candidate is None or
            (
                candidate['support_median'],
                candidate['support_mean'],
                candidate['stats']['p95'],
                candidate['stats']['mean'],
            ) <
            (
                best_candidate['support_median'],
                best_candidate['support_mean'],
                best_candidate['stats']['p95'],
                best_candidate['stats']['mean'],
            )
        ):
            best_candidate = candidate

    if best_candidate is None:
        return None

    initial_support_errors = initial_stats['errors'][support_indices]
    initial_support_median = float(np.median(initial_support_errors))
    initial_support_mean = float(np.mean(initial_support_errors))
    final_stats = best_candidate['stats']

    support_improved = (
        best_candidate['support_median'] <= initial_support_median - 0.30 or
        best_candidate['support_median'] <= initial_support_median * 0.90 or
        best_candidate['support_mean'] <= initial_support_mean - 0.35
    )
    front_not_worse = final_stats['front_ratio'] >= max(0.80, initial_stats['front_ratio'] - 0.05)
    global_not_much_worse = (
        final_stats['mean'] <= max(initial_stats['mean'] + 0.40, initial_stats['mean'] * 1.08) and
        final_stats['p95'] <= max(initial_stats['p95'] + 0.80, initial_stats['p95'] * 1.10) and
        final_stats['max'] <= max(initial_stats['max'] + 1.20, initial_stats['max'] * 1.12)
    )
    if not support_improved or not front_not_worse or not global_not_much_worse:
        return None

    return {
        'R': best_candidate['R'],
        't': best_candidate['t'],
        'support_indices': best_candidate['support_indices'],
        'initial_support_median': float(initial_support_median),
        'final_support_median': float(best_candidate['support_median']),
        'initial_support_mean': float(initial_support_mean),
        'final_support_mean': float(best_candidate['support_mean']),
        'support_count': int(len(best_candidate['support_indices'])),
        'guided': bool(use_guided_support),
    }


def _evaluate_pose_candidate_support(
    points_3d,
    points_2d,
    K,
    dist_coeffs,
    R,
    t,
    *,
    min_points=4,
    preferred_indices=None,
    observation_confidences=None,
    support_min_confidence=0.28,
):
    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    points_2d = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    if len(points_3d) < max(int(min_points), 4):
        return None

    observation_confidences = _normalize_observation_confidences_array(
        observation_confidences,
        len(points_3d),
    )
    camera_K = np.asarray(K, dtype=np.float64)
    dist_coeffs = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    stats = _evaluate_camera_pose_candidate(
        points_3d,
        points_2d,
        np.asarray(R, dtype=np.float64).reshape(3, 3),
        np.asarray(t, dtype=np.float64).reshape(3, 1),
        camera_K,
        dist_coeffs,
    )

    if _should_use_confidence_guided_pose_support(
        observation_confidences,
        min_confidence=support_min_confidence,
    ):
        support_indices = _select_pose_refine_support_indices(
            stats['errors'],
            observation_confidences=observation_confidences,
            min_points=max(int(min_points), 4),
            preferred_indices=preferred_indices,
            min_confidence=support_min_confidence,
        )
    else:
        support_indices = _select_residual_support_indices(
            stats['errors'],
            min_points=max(int(min_points), 4),
            preferred_indices=preferred_indices,
            min_fraction=0.45,
            max_fraction=0.72,
        )

    if len(support_indices) >= 4:
        support_errors = stats['errors'][support_indices]
    else:
        support_indices = np.arange(len(stats['errors']), dtype=np.int32)
        support_errors = stats['errors']

    return {
        'stats': stats,
        'support_indices': np.asarray(support_indices, dtype=np.int32),
        'support_count': int(len(support_indices)),
        'support_mean': float(np.mean(support_errors)),
        'support_median': float(np.median(support_errors)),
        'support_p95': float(np.percentile(support_errors, 95)),
    }


def _format_pose_candidate_eval(eval_result):
    if eval_result is None:
        return "none"

    stats = eval_result.get('stats', {})
    return (
        f"support={int(eval_result.get('support_count', 0))}, "
        f"med={float(eval_result.get('support_median', np.inf)):.2f}px, "
        f"p95={float(eval_result.get('support_p95', np.inf)):.2f}px, "
        f"mean={float(stats.get('mean', np.inf)):.2f}px, "
        f"front={float(stats.get('front_ratio', 0.0)):.0%}"
    )


def _is_pose_candidate_clearly_better(candidate_eval, reference_eval):
    if candidate_eval is None:
        return False
    if reference_eval is None:
        return True

    candidate_stats = candidate_eval.get('stats', {})
    reference_stats = reference_eval.get('stats', {})
    candidate_front = float(candidate_stats.get('front_ratio', 0.0))
    reference_front = float(reference_stats.get('front_ratio', 0.0))
    candidate_support = int(candidate_eval.get('support_count', 0))
    reference_support = int(reference_eval.get('support_count', 0))
    candidate_support_median = float(candidate_eval.get('support_median', np.inf))
    reference_support_median = float(reference_eval.get('support_median', np.inf))
    candidate_support_mean = float(candidate_eval.get('support_mean', np.inf))
    reference_support_mean = float(reference_eval.get('support_mean', np.inf))
    candidate_support_p95 = float(candidate_eval.get('support_p95', np.inf))
    reference_support_p95 = float(reference_eval.get('support_p95', np.inf))
    candidate_mean = float(candidate_stats.get('mean', np.inf))
    reference_mean = float(reference_stats.get('mean', np.inf))
    candidate_p95 = float(candidate_stats.get('p95', np.inf))
    reference_p95 = float(reference_stats.get('p95', np.inf))

    front_not_worse = candidate_front >= max(0.72, reference_front - 0.06)
    support_not_too_small = candidate_support >= max(reference_support - 4, 4)
    global_not_much_worse = (
        candidate_mean <= max(reference_mean + 0.60, reference_mean * 1.12) and
        candidate_p95 <= max(reference_p95 + 1.20, reference_p95 * 1.14)
    )

    score = 0
    if candidate_support >= reference_support + 2:
        score += 1
    if candidate_support_median <= min(reference_support_median - 1.25, reference_support_median * 0.88):
        score += 2
    elif candidate_support_median <= min(reference_support_median - 0.55, reference_support_median * 0.94):
        score += 1
    if candidate_support_mean <= min(reference_support_mean - 0.75, reference_support_mean * 0.90):
        score += 1
    if candidate_support_p95 <= min(reference_support_p95 - 1.75, reference_support_p95 * 0.86):
        score += 1
    if candidate_mean <= min(reference_mean - 0.35, reference_mean * 0.94):
        score += 1
    if candidate_p95 <= min(reference_p95 - 0.90, reference_p95 * 0.92):
        score += 1
    if candidate_front >= reference_front + 0.05:
        score += 1

    return front_not_worse and support_not_too_small and global_not_much_worse and score >= 2


def _refine_pnp_solution(
    points_3d,
    points_2d,
    K,
    dist_coeffs,
    rvec,
    tvec,
    inliers=None,
    observation_confidences=None,
    support_min_confidence=0.28,
):
    points_3d = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
    points_2d = np.asarray(points_2d, dtype=np.float32).reshape(-1, 2)
    if points_3d.shape[0] < 4 or points_2d.shape[0] < 4:
        return rvec, tvec

    observation_confidences = _normalize_observation_confidences_array(
        observation_confidences,
        points_3d.shape[0],
    )
    use_guided_support = _should_use_confidence_guided_pose_support(
        observation_confidences,
        min_confidence=support_min_confidence,
    )
    seed_points_3d = points_3d
    seed_points_2d = points_2d
    preferred_indices = None
    if inliers is not None:
        inlier_indices = np.asarray(inliers).ravel()
        if inlier_indices.size >= 4:
            seed_points_3d = points_3d[inlier_indices]
            seed_points_2d = points_2d[inlier_indices]
            preferred_indices = inlier_indices

    def _refine_and_score(seed_rvec, seed_tvec):
        local_rvec = np.asarray(seed_rvec, dtype=np.float64).reshape(3, 1)
        local_tvec = np.asarray(seed_tvec, dtype=np.float64).reshape(3, 1)
        try:
            projected_points, _ = cv2.projectPoints(
                points_3d,
                local_rvec,
                local_tvec,
                K,
                dist_coeffs,
            )
            seed_errors = np.linalg.norm(projected_points.reshape(-1, 2) - points_2d, axis=1)
        except cv2.error:
            seed_errors = np.full(points_3d.shape[0], np.inf, dtype=np.float64)

        if use_guided_support:
            support_indices = _select_pose_refine_support_indices(
                seed_errors,
                observation_confidences=observation_confidences,
                min_points=4,
                preferred_indices=preferred_indices,
                min_confidence=support_min_confidence,
            )
        else:
            support_indices = np.arange(points_3d.shape[0], dtype=np.int32)
        support_points_3d = points_3d[support_indices] if support_indices.size >= 4 else points_3d
        support_points_2d = points_2d[support_indices] if support_indices.size >= 4 else points_2d
        try:
            local_rvec, local_tvec = cv2.solvePnPRefineLM(
                support_points_3d,
                support_points_2d,
                K,
                dist_coeffs,
                local_rvec,
                local_tvec,
            )
        except cv2.error:
            pass

        try:
            R_local, _ = cv2.Rodrigues(local_rvec)
            metrics = _evaluate_camera_pose_candidate(
                points_3d,
                points_2d,
                R_local,
                local_tvec,
                K,
                dist_coeffs,
            )
            support_metrics = _evaluate_camera_pose_candidate(
                support_points_3d,
                support_points_2d,
                R_local,
                local_tvec,
                K,
                dist_coeffs,
            )
        except cv2.error:
            return None

        score = (
            float(metrics.get('front_ratio', 0.0)),
            -float(support_metrics.get('p95', np.inf)),
            -float(support_metrics.get('mean', np.inf)),
            -float(metrics.get('p95', np.inf)),
            -float(metrics.get('mean', np.inf)),
            -float(metrics.get('max', np.inf)),
        )
        return {
            'rvec': local_rvec,
            'tvec': local_tvec,
            'score': score,
            'support_count': int(len(support_points_3d)),
        }

    best_candidate = _refine_and_score(rvec, tvec)
    if best_candidate is None:
        best_candidate = {
            'rvec': np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            'tvec': np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            'score': None,
        }

    if (
        hasattr(cv2, 'SOLVEPNP_SQPNP') and
        seed_points_3d.shape[0] >= 6
    ):
        try:
            sq_success, sq_rvec, sq_tvec = cv2.solvePnP(
                seed_points_3d,
                seed_points_2d,
                K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP,
            )
        except cv2.error:
            sq_success = False
            sq_rvec = None
            sq_tvec = None

        if sq_success:
            sq_candidate = _refine_and_score(sq_rvec, sq_tvec)
            if (
                sq_candidate is not None and
                (
                    best_candidate.get('score') is None or
                    sq_candidate['score'] > best_candidate['score']
                )
            ):
                best_candidate = sq_candidate

    return best_candidate['rvec'], best_candidate['tvec']


def _solve_camera_pose_from_support(
    points_3d,
    points_2d,
    K,
    dist_coeffs=None,
    reprojection_threshold=6.0,
    confidence=0.995,
    observation_confidences=None,
    support_min_confidence=0.28,
):
    points_3d = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
    points_2d = np.asarray(points_2d, dtype=np.float32).reshape(-1, 2)
    if len(points_3d) < 4:
        return None

    success = False
    rvec = None
    tvec = None
    inliers = None
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=float(reprojection_threshold),
            confidence=float(confidence),
            iterationsCount=1000,
        )
    except cv2.error:
        success = False

    if not success:
        try:
            success, rvec, tvec = cv2.solvePnP(
                points_3d,
                points_2d,
                K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            inliers = None
        except cv2.error:
            success = False

    if not success:
        return None

    rvec, tvec = _refine_pnp_solution(
        points_3d,
        points_2d,
        K,
        dist_coeffs,
        rvec,
        tvec,
        inliers=inliers,
        observation_confidences=observation_confidences,
        support_min_confidence=support_min_confidence,
    )

    R_candidate, _ = cv2.Rodrigues(rvec)
    return {
        'R': np.asarray(R_candidate, dtype=np.float32),
        't': np.asarray(tvec, dtype=np.float32).reshape(3, 1),
        'inlier_count': int(len(inliers)) if inliers is not None else int(len(points_3d)),
        'support_count': int(len(points_3d)),
    }


def _refine_camera_pose_and_focal_from_support(
    points_3d,
    points_2d,
    R,
    t,
    K,
    dist_coeffs=None,
    force_same_focal=True,
    observation_confidences=None,
):
    try:
        from scipy.optimize import least_squares
    except Exception:
        return None

    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    points_2d = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    if len(points_3d) < 6:
        return None

    observation_confidences = _normalize_observation_confidences_array(observation_confidences, len(points_3d))
    base_K = np.asarray(K, dtype=np.float64)
    dist_coeffs = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    base_R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    base_t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    base_rvec, _ = cv2.Rodrigues(base_R)

    base_fx = float(base_K[0, 0])
    base_fy = float(base_K[1, 1])
    base_cx = float(base_K[0, 2])
    base_cy = float(base_K[1, 2])
    base_focal = float((base_fx + base_fy) * 0.5)
    min_focal = max(400.0, base_focal * 0.50)
    max_focal = max(min_focal + 1.0, base_focal * 2.50)

    if force_same_focal:
        x0 = np.concatenate([base_rvec.reshape(3), base_t.reshape(3), np.array([base_focal], dtype=np.float64)])
        lower = np.concatenate([np.full(6, -np.inf), np.array([min_focal], dtype=np.float64)])
        upper = np.concatenate([np.full(6, np.inf), np.array([max_focal], dtype=np.float64)])
    else:
        x0 = np.concatenate([base_rvec.reshape(3), base_t.reshape(3), np.array([base_fx, base_fy], dtype=np.float64)])
        lower = np.concatenate([np.full(6, -np.inf), np.array([min_focal, min_focal], dtype=np.float64)])
        upper = np.concatenate([np.full(6, np.inf), np.array([max_focal, max_focal], dtype=np.float64)])

    def _build_camera_matrix(params):
        if force_same_focal:
            fx = fy = float(params[6])
        else:
            fx = float(params[6])
            fy = float(params[7])
        return np.array(
            [
                [fx, 0.0, base_cx],
                [0.0, fy, base_cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _raw_residuals(params):
        camera_K = _build_camera_matrix(params)
        projected_points, _ = cv2.projectPoints(
            points_3d,
            np.asarray(params[:3], dtype=np.float64).reshape(3, 1),
            np.asarray(params[3:6], dtype=np.float64).reshape(3, 1),
            camera_K,
            dist_coeffs,
        )
        return projected_points.reshape(-1, 2) - points_2d

    initial_K = _build_camera_matrix(x0)
    initial_stats = _evaluate_camera_pose_candidate(points_3d, points_2d, base_R, base_t, initial_K, dist_coeffs)
    use_guided_support = _should_use_confidence_guided_pose_support(
        observation_confidences,
        min_confidence=0.28,
    )
    if use_guided_support:
        support_indices = _select_pose_refine_support_indices(
            initial_stats.get('errors'),
            observation_confidences=observation_confidences,
            min_points=6,
            min_confidence=0.28,
        )
        if support_indices.size < 6:
            return None
        support_points_3d = points_3d[support_indices]
        support_points_2d = points_2d[support_indices]
        support_confidences = observation_confidences[support_indices]
    else:
        support_points_3d = points_3d
        support_points_2d = points_2d
        support_confidences = np.ones_like(observation_confidences, dtype=np.float64)

    def _objective(params):
        camera_K = _build_camera_matrix(params)
        projected_points, _ = cv2.projectPoints(
            support_points_3d,
            np.asarray(params[:3], dtype=np.float64).reshape(3, 1),
            np.asarray(params[3:6], dtype=np.float64).reshape(3, 1),
            camera_K,
            dist_coeffs,
        )
        raw = (
            (projected_points.reshape(-1, 2) - support_points_2d)
            * np.sqrt(np.clip(support_confidences.reshape(-1, 1), 0.15, 1.0))
        ).reshape(-1)
        if force_same_focal:
            focal_delta = (float(params[6]) - base_focal) / max(abs(base_focal), 1.0)
            regularization = np.array([0.06 * focal_delta], dtype=np.float64)
        else:
            fx_delta = (float(params[6]) - base_fx) / max(abs(base_fx), 1.0)
            fy_delta = (float(params[7]) - base_fy) / max(abs(base_fy), 1.0)
            regularization = np.array([0.06 * fx_delta, 0.06 * fy_delta], dtype=np.float64)
        return np.concatenate([raw, regularization])

    result = least_squares(
        _objective,
        x0,
        bounds=(lower, upper),
        method='trf',
        loss='soft_l1',
        f_scale=4.0,
        max_nfev=600,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )
    final_params = result.x if result.success and np.all(np.isfinite(result.x)) else x0
    final_K = _build_camera_matrix(final_params)
    final_R, _ = cv2.Rodrigues(np.asarray(final_params[:3], dtype=np.float64).reshape(3, 1))
    final_t = np.asarray(final_params[3:6], dtype=np.float64).reshape(3, 1)
    final_stats = _evaluate_camera_pose_candidate(points_3d, points_2d, final_R, final_t, final_K, dist_coeffs)

    focal_delta = max(
        abs(float(final_K[0, 0]) - base_fx),
        abs(float(final_K[1, 1]) - base_fy),
    )
    accepted = (
        final_stats['front_ratio'] >= max(0.90, initial_stats['front_ratio'] - 0.02) and
        focal_delta > max(base_focal * 0.05, 10.0) and
        focal_delta <= max(base_focal * 0.35, 450.0) and
        (
            final_stats['mean'] <= initial_stats['mean'] - 0.75 or
            final_stats['p95'] <= initial_stats['p95'] - 2.00 or
            (
                final_stats['mean'] <= initial_stats['mean'] * 0.60 and
                final_stats['max'] <= initial_stats['max'] * 0.60
            )
        )
    )
    if not accepted:
        return None

    return {
        'R': np.asarray(final_R, dtype=np.float32),
        't': np.asarray(final_t, dtype=np.float32).reshape(3, 1),
        'K': np.asarray(final_K, dtype=np.float32),
        'initial_stats': initial_stats,
        'final_stats': final_stats,
    }


def _refine_camera_pose_from_support_robust(
    points_3d,
    points_2d,
    R,
    t,
    K,
    dist_coeffs=None,
    observation_confidences=None,
):
    try:
        from scipy.optimize import least_squares
    except Exception:
        return None

    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    points_2d = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    if len(points_3d) < 6:
        return None

    observation_confidences = _normalize_observation_confidences_array(observation_confidences, len(points_3d))
    camera_K = np.asarray(K, dtype=np.float64)
    dist_coeffs = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    base_R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    base_t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    base_rvec, _ = cv2.Rodrigues(base_R)
    x0 = np.concatenate([base_rvec.reshape(3), base_t.reshape(3)])

    initial_stats = _evaluate_camera_pose_candidate(points_3d, points_2d, base_R, base_t, camera_K, dist_coeffs)
    use_guided_support = _should_use_confidence_guided_pose_support(
        observation_confidences,
        min_confidence=0.28,
    )
    if use_guided_support:
        support_indices = _select_pose_refine_support_indices(
            initial_stats.get('errors'),
            observation_confidences=observation_confidences,
            min_points=6,
            min_confidence=0.28,
        )
        if support_indices.size < 6:
            return None
        support_points_3d = points_3d[support_indices]
        support_points_2d = points_2d[support_indices]
        support_confidences = observation_confidences[support_indices]
    else:
        support_points_3d = points_3d
        support_points_2d = points_2d
        support_confidences = np.ones_like(observation_confidences, dtype=np.float64)

    def _objective(params):
        projected_points, _ = cv2.projectPoints(
            support_points_3d,
            np.asarray(params[:3], dtype=np.float64).reshape(3, 1),
            np.asarray(params[3:6], dtype=np.float64).reshape(3, 1),
            camera_K,
            dist_coeffs,
        )
        return (
            (projected_points.reshape(-1, 2) - support_points_2d)
            * np.sqrt(np.clip(support_confidences.reshape(-1, 1), 0.15, 1.0))
        ).reshape(-1)

    result = least_squares(
        _objective,
        x0,
        method='trf',
        loss='soft_l1',
        f_scale=6.0,
        max_nfev=500,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )
    final_params = result.x if result.success and np.all(np.isfinite(result.x)) else x0
    final_R, _ = cv2.Rodrigues(np.asarray(final_params[:3], dtype=np.float64).reshape(3, 1))
    final_t = np.asarray(final_params[3:6], dtype=np.float64).reshape(3, 1)
    final_stats = _evaluate_camera_pose_candidate(points_3d, points_2d, final_R, final_t, camera_K, dist_coeffs)

    accepted = (
        final_stats['front_ratio'] >= max(0.90, initial_stats['front_ratio'] - 0.02) and
        (
            final_stats['mean'] <= initial_stats['mean'] - 0.75 or
            final_stats['p95'] <= initial_stats['p95'] - 2.00 or
            (
                final_stats['mean'] <= initial_stats['mean'] * 0.65 and
                final_stats['max'] <= initial_stats['max'] * 0.65
            )
        )
    )
    if not accepted:
        return None

    return {
        'R': np.asarray(final_R, dtype=np.float32),
        't': np.asarray(final_t, dtype=np.float32).reshape(3, 1),
        'initial_stats': initial_stats,
        'final_stats': final_stats,
    }


def refine_high_error_camera_poses(
    calib_data,
    max_cameras=2,
    min_observations_per_camera=8,
):
    if not calib_data.get('cameras') or not calib_data.get('points_3d'):
        return 0

    global_stats = summarize_reprojection_error_distribution(calib_data, top_k=0)
    camera_profiles = summarize_camera_error_profiles(calib_data, top_k=3)
    if not camera_profiles or global_stats['count'] <= 0:
        return 0

    global_p95 = max(global_stats['p95'], 1e-6)
    candidate_profiles = []
    for profile in camera_profiles:
        if profile['count'] < min_observations_per_camera:
            continue
        precision_target_mean = float(calib_data.get('precision_target_mean_px', 0.5))
        absolute_rmse_exceeded = profile.get('mean', 0.0) > precision_target_mean * 1.6
        if (
            profile['p95'] <= max(global_p95 * 1.12, 3.5) and
            profile['max'] <= max(global_p95 * 1.55, 5.25) and
            not absolute_rmse_exceeded
        ):
            continue
        severity = max(
            profile['p95'] / global_p95,
            profile['max'] / max(global_p95 * 1.35, 1e-6),
        )
        candidate_profiles.append((severity, profile))

    if not candidate_profiles:
        print("Локальное уточнение поз камер: кандидатов не найдено")
        return 0

    refined_total = 0
    print("Локальное уточнение поз камер:")
    for _, profile in sorted(candidate_profiles, key=lambda item: item[0], reverse=True)[:max(1, int(max_cameras))]:
        camera_id = profile['camera_id']
        correspondences = _collect_camera_pose_correspondences(calib_data, camera_id)
        if correspondences is None or len(correspondences['points_3d']) < min_observations_per_camera:
            continue

        current_R, current_t = calib_data['cameras'][camera_id]
        current_stats = _evaluate_camera_pose_candidate(
            correspondences['points_3d'],
            correspondences['points_2d'],
            current_R,
            current_t,
            calib_data['K'],
            calib_data.get('dist_coeffs'),
        )

        local_mad = float(stats.median_abs_deviation(correspondences['errors'], scale='normal'))
        support_threshold = max(
            3.0,
            min(
                float(profile['p90']),
                current_stats['median'] + 1.75 * max(local_mad, 0.75),
            ),
        )
        support_confidences = correspondences.get('confidences')
        if _should_use_confidence_guided_pose_support(
            support_confidences,
            min_confidence=_get_soft_gate_min_confidence(calib_data),
        ):
            support_indices = _select_pose_refine_support_indices(
                correspondences['errors'],
                observation_confidences=support_confidences,
                min_points=6,
                min_confidence=_get_soft_gate_min_confidence(calib_data),
            )
        else:
            support_indices = np.where(correspondences['errors'] <= support_threshold)[0]
            if len(support_indices) < 6:
                subset_size = min(
                    len(correspondences['errors']),
                    max(6, int(np.ceil(len(correspondences['errors']) * 0.65))),
                )
                support_indices = np.argsort(correspondences['errors'])[:subset_size]
        if len(support_indices) < 4:
            continue

        pose_candidate = _solve_camera_pose_from_support(
            correspondences['points_3d'][support_indices],
            correspondences['points_2d'][support_indices],
            calib_data['K'],
            calib_data.get('dist_coeffs'),
            reprojection_threshold=max(3.5, min(8.0, support_threshold * 1.1)),
            confidence=0.995,
            observation_confidences=support_confidences[support_indices],
            support_min_confidence=_get_soft_gate_min_confidence(calib_data),
        )
        if pose_candidate is None:
            continue

        candidate_stats = _evaluate_camera_pose_candidate(
            correspondences['points_3d'],
            correspondences['points_2d'],
            pose_candidate['R'],
            pose_candidate['t'],
            calib_data['K'],
            calib_data.get('dist_coeffs'),
        )
        local_improved = (
            candidate_stats['p95'] <= current_stats['p95'] - 0.35 or
            candidate_stats['p95'] <= current_stats['p95'] * 0.92 or
            candidate_stats['max'] <= current_stats['max'] - 0.75
        )
        mean_not_worse = candidate_stats['mean'] <= current_stats['mean'] + 0.05
        front_not_worse = candidate_stats['front_ratio'] >= max(0.85, current_stats['front_ratio'] - 0.05)
        if not local_improved or not mean_not_worse or not front_not_worse:
            continue

        before_error, _, _ = calculate_reprojection_errors(calib_data)
        before_distribution = summarize_reprojection_error_distribution(calib_data, top_k=0)
        original_pose = calib_data['cameras'][camera_id]
        calib_data['cameras'][camera_id] = (pose_candidate['R'], pose_candidate['t'])

        after_error, _, _ = calculate_reprojection_errors(calib_data)
        after_distribution = summarize_reprojection_error_distribution(calib_data, top_k=0)
        global_not_worse = (
            after_error <= before_error + 0.03 and
            after_distribution['p95'] <= before_distribution['p95'] + 0.05
        )
        if not global_not_worse:
            calib_data['cameras'][camera_id] = original_pose
            continue

        refined_total += 1
        print(
            f"  - Камера {camera_id}: "
            f"local p95 {current_stats['p95']:.2f}px -> {candidate_stats['p95']:.2f}px, "
            f"max {current_stats['max']:.2f}px -> {candidate_stats['max']:.2f}px, "
            f"global mean {before_error:.4f}px -> {after_error:.4f}px"
        )

    if refined_total <= 0:
        print("Локальное уточнение поз камер: изменений не потребовалось")

    return refined_total


def recover_observations_after_pose_refinement(
    calib_data,
    max_cameras=4,
    max_recoveries_per_camera=6,
):
    """
    Восстанавливает удалённые наблюдения после уточнения позы камеры.
    
    После precision cleanup удаляются наблюдения с высокой ошибкой репроекции.
    Корневая причина часто — неточная поза камеры. После refine_high_error_camera_poses
    некоторые удалённые наблюдения можно вернуть, используя raw_camera_points.
    
    Args:
        calib_data: Данные калибровки
        max_cameras: Максимум камер для обработки
        max_recoveries_per_camera: Максимум восстановленных наблюдений на камеру
        
    Returns:
        Количество восстановленных наблюдений
    """
    if not calib_data.get('cameras') or not calib_data.get('points_3d'):
        return 0
    
    # Шаг 1: Найти камеры с потерянными наблюдениями
    raw_camera_points = calib_data.get('raw_camera_points', {})
    camera_points = calib_data.get('camera_points', {})
    points_3d = calib_data.get('points_3d', {})
    
    if not raw_camera_points:
        return 0
    
    # Определить потерянные наблюдения для каждой камеры
    camera_candidates = []
    for camera_id in raw_camera_points.keys():
        camera_id_str = str(camera_id)
        raw_points = raw_camera_points.get(camera_id_str, {})
        existing_points = camera_points.get(camera_id_str, {})
        
        # Найти потерянные точки, которые существуют в points_3d
        lost_point_ids = []
        for point_id in raw_points.keys():
            if point_id not in existing_points and point_id in points_3d:
                lost_point_ids.append(point_id)
        
        if lost_point_ids:
            camera_candidates.append((len(lost_point_ids), camera_id_str, lost_point_ids))
    
    if not camera_candidates:
        return 0
    
    # Сортировка: камеры с наибольшим кол-вом recoverable наблюдений первыми
    camera_candidates.sort(reverse=True)
    
    # Получить целевые пороги
    target_mean = float(calib_data.get('precision_target_mean_px', 0.5))
    target_p95 = float(calib_data.get('precision_target_p95_px', 1.0))
    target_max = float(calib_data.get('precision_target_max_px', 1.5))
    
    # Gate 1: Индивидуальный порог (строже чем cleanup threshold)
    individual_threshold = min(target_max * 0.85, target_p95)
    
    total_recovered = 0
    print("Восстановление наблюдений после уточнения поз:")
    
    # При необходимости — получить глобальную статистику до изменений
    before_global_error, _, _ = calculate_reprojection_errors(calib_data)
    before_global_dist = summarize_reprojection_error_distribution(calib_data, top_k=0)
    
    for _, camera_id_str, lost_point_ids in camera_candidates[:max_cameras]:
        if camera_id_str not in calib_data['cameras']:
            continue
        
        # Шаг 2: Оценить каждое удалённое наблюдение с текущей позой
        R, t = calib_data['cameras'][camera_id_str]
        camera_K = _get_camera_matrix(calib_data, camera_id_str)
        
        recovery_candidates = []
        for point_id in lost_point_ids:
            raw_2d = np.asarray(raw_camera_points[camera_id_str][point_id], dtype=np.float64).reshape(2)
            point_3d = np.asarray(points_3d[point_id], dtype=np.float64).reshape(1, 3)
            
            # Проецировать 3D-точку через текущую позу камеры
            rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64).reshape(3, 3))
            projected, _ = cv2.projectPoints(
                point_3d,
                rvec,
                np.asarray(t, dtype=np.float64).reshape(3, 1),
                np.asarray(camera_K, dtype=np.float64),
                None if calib_data.get('dist_coeffs') is None else np.asarray(calib_data.get('dist_coeffs'), dtype=np.float64).reshape(-1, 1),
            )
            reprojection_error = float(np.linalg.norm(projected.reshape(2) - raw_2d))
            
            # Gate 1: Индивидуальный порог
            if reprojection_error <= individual_threshold:
                recovery_candidates.append((reprojection_error, point_id))
        
        if not recovery_candidates:
            continue
        
        # Шаг 3: Сортировка кандидатов по ошибке (лучшие первые)
        recovery_candidates.sort()
        
        camera_recovered = 0
        for reprojection_error, point_id in recovery_candidates[:max_recoveries_per_camera]:
            # Запомнить состояние до добавления
            before_mean = before_global_error
            before_stats = before_global_dist
            
            # Добавить наблюдение обратно в camera_points
            raw_2d = np.asarray(raw_camera_points[camera_id_str][point_id], dtype=np.float64).reshape(2)
            camera_points.setdefault(camera_id_str, {})[point_id] = raw_2d
            
            # Вычислить статистику после добавления
            after_mean, _, _ = calculate_reprojection_errors(calib_data)
            after_stats = summarize_reprojection_error_distribution(calib_data, top_k=0)
            
            # Gate 2: Глобальная проверка
            scene_ok = (
                after_mean <= max(before_mean + 0.03, target_mean * 1.15) and
                after_stats['p95'] <= max(before_stats['p95'] + 0.08, target_p95 * 1.10) and
                after_stats['max'] <= max(before_stats['max'] + 0.15, target_max * 1.25)
            )
            
            if not scene_ok:
                # Откатить добавление
                del camera_points[camera_id_str][point_id]
                continue
            
            # Установить confidence=0.70
            _set_observation_confidence(calib_data, camera_id_str, point_id, 0.70)
            camera_recovered += 1
            total_recovered += 1
            
            # Обновить индекс before для следующей итерации
            before_global_error = after_mean
            before_global_dist = after_stats
        
        if camera_recovered > 0:
            print(
                f"  - Камера {camera_id_str}: "
                f"+{camera_recovered} наблюдения "
                f"(из {len(lost_point_ids)} удалённых), "
                f"mean {before_mean:.4f}px -> {after_mean:.4f}px"
            )
    
    if total_recovered <= 0:
        print("Восстановление наблюдений: кандидатов не найдено")
    
    return total_recovered


def _refine_cameras_from_pseudo_track_entries(
    calib_data,
    by_camera,
    *,
    label,
    max_cameras=2,
    max_tracks_per_camera=6,
    min_total_correspondences=8,
):
    if not by_camera:
        print(f"{label}: кандидатов не найдено")
        return 0

    camera_candidates = []
    for camera_id, entries in by_camera.items():
        aggregate_severity = sum(float(entry['severity']) for entry in entries)
        camera_candidates.append((aggregate_severity, camera_id, entries))

    refined_total = 0
    print(f"{label}:")
    for _, camera_id, entries in sorted(camera_candidates, key=lambda item: item[0], reverse=True)[:max(1, int(max_cameras))]:
        base_correspondences = _collect_camera_pose_correspondences(calib_data, camera_id)
        if base_correspondences is None:
            continue

        existing_points_3d = np.asarray(base_correspondences['points_3d'], dtype=np.float64).reshape(-1, 3)
        existing_points_2d = np.asarray(base_correspondences['points_2d'], dtype=np.float64).reshape(-1, 2)
        current_R, current_t = calib_data['cameras'][camera_id]
        camera_K = _get_camera_matrix(calib_data, camera_id)
        current_existing_stats = _evaluate_camera_pose_candidate(
            existing_points_3d,
            existing_points_2d,
            current_R,
            current_t,
            camera_K,
            calib_data.get('dist_coeffs'),
        )

        pseudo_entries = sorted(entries, key=lambda item: item['severity'], reverse=True)[:max(1, int(max_tracks_per_camera))]
        pseudo_points_3d = np.asarray([entry['point_3d'] for entry in pseudo_entries], dtype=np.float64).reshape(-1, 3)
        pseudo_points_2d = np.asarray([entry['point_2d'] for entry in pseudo_entries], dtype=np.float64).reshape(-1, 2)
        if len(pseudo_points_3d) < 2:
            continue

        current_pseudo_stats = _evaluate_camera_pose_candidate(
            pseudo_points_3d,
            pseudo_points_2d,
            current_R,
            current_t,
            camera_K,
            calib_data.get('dist_coeffs'),
        )

        combined_points_3d = np.vstack([existing_points_3d, pseudo_points_3d])
        combined_points_2d = np.vstack([existing_points_2d, pseudo_points_2d])
        combined_confidences = np.concatenate([
            np.asarray(base_correspondences.get('confidences', np.ones(len(existing_points_3d))), dtype=np.float64).reshape(-1),
            np.full(len(pseudo_points_3d), 0.95, dtype=np.float64),
        ])
        if len(combined_points_3d) < max(6, int(min_total_correspondences)):
            continue

        pose_hypotheses = []
        combined_threshold = max(
            2.5,
            min(
                6.0,
                current_existing_stats['median'] + 1.25,
            ),
        )
        pose_hypotheses.append({
            'label': 'combined',
            'points_3d': combined_points_3d,
            'points_2d': combined_points_2d,
            'confidences': combined_confidences,
            'threshold': combined_threshold,
        })
        if len(pseudo_points_3d) >= 4:
            pseudo_threshold = max(
                2.0,
                min(
                    8.0,
                    current_pseudo_stats['median'] * 0.75,
                ),
            )
            pose_hypotheses.append({
                'label': 'pseudo_only',
                'points_3d': pseudo_points_3d,
                'points_2d': pseudo_points_2d,
                'confidences': np.full(len(pseudo_points_3d), 0.95, dtype=np.float64),
                'threshold': pseudo_threshold,
            })

        best_pose_candidate = None
        for pose_hypothesis in pose_hypotheses:
            pose_candidate = _solve_camera_pose_from_support(
                pose_hypothesis['points_3d'],
                pose_hypothesis['points_2d'],
                camera_K,
                calib_data.get('dist_coeffs'),
                reprojection_threshold=pose_hypothesis['threshold'],
                confidence=0.995,
                observation_confidences=pose_hypothesis.get('confidences'),
                support_min_confidence=_get_soft_gate_min_confidence(calib_data),
            )
            if pose_candidate is None:
                continue

            candidate_existing_stats = _evaluate_camera_pose_candidate(
                existing_points_3d,
                existing_points_2d,
                pose_candidate['R'],
                pose_candidate['t'],
                camera_K,
                calib_data.get('dist_coeffs'),
            )
            candidate_pseudo_stats = _evaluate_camera_pose_candidate(
                pseudo_points_3d,
                pseudo_points_2d,
                pose_candidate['R'],
                pose_candidate['t'],
                camera_K,
                calib_data.get('dist_coeffs'),
            )

            pseudo_improved = (
                candidate_pseudo_stats['median'] <= current_pseudo_stats['median'] - 0.75 or
                candidate_pseudo_stats['p95'] <= current_pseudo_stats['p95'] - 1.00 or
                candidate_pseudo_stats['max'] <= current_pseudo_stats['max'] - 1.50 or
                (
                    candidate_pseudo_stats['median'] <= current_pseudo_stats['median'] * 0.82 and
                    candidate_pseudo_stats['max'] <= current_pseudo_stats['max'] * 0.78
                )
            )
            existing_not_worse = (
                candidate_existing_stats['mean'] <= current_existing_stats['mean'] + 0.06 and
                candidate_existing_stats['p95'] <= current_existing_stats['p95'] + 0.12 and
                candidate_existing_stats['max'] <= current_existing_stats['max'] + 0.20 and
                candidate_existing_stats['front_ratio'] >= max(0.90, current_existing_stats['front_ratio'] - 0.03)
            )
            if not pseudo_improved or not existing_not_worse:
                continue

            candidate_score = (
                (current_pseudo_stats['median'] - candidate_pseudo_stats['median']) * 0.40 +
                (current_pseudo_stats['p95'] - candidate_pseudo_stats['p95']) * 0.35 +
                (current_pseudo_stats['max'] - candidate_pseudo_stats['max']) * 0.25
            )
            if best_pose_candidate is None or candidate_score > best_pose_candidate['score'] + 1e-6:
                best_pose_candidate = {
                    'label': pose_hypothesis['label'],
                    'pose': pose_candidate,
                    'existing_stats': candidate_existing_stats,
                    'pseudo_stats': candidate_pseudo_stats,
                    'score': float(candidate_score),
                }

        if best_pose_candidate is None:
            continue

        before_mean, _, _ = calculate_reprojection_errors(calib_data)
        before_distribution = summarize_reprojection_error_distribution(calib_data, top_k=0)
        original_pose = calib_data['cameras'][camera_id]
        calib_data['cameras'][camera_id] = (
            best_pose_candidate['pose']['R'],
            best_pose_candidate['pose']['t'],
        )

        after_mean, _, _ = calculate_reprojection_errors(calib_data)
        after_distribution = summarize_reprojection_error_distribution(calib_data, top_k=0)
        global_not_worse = (
            after_mean <= before_mean + 0.04 and
            after_distribution['p95'] <= before_distribution['p95'] + 0.08 and
            after_distribution['max'] <= before_distribution['max'] + 0.20
        )
        if not global_not_worse:
            calib_data['cameras'][camera_id] = original_pose
            continue

        refined_total += 1
        print(
            f"  - Камера {camera_id}: "
            f"mode={best_pose_candidate['label']}, "
            f"pseudo median {current_pseudo_stats['median']:.2f}px -> {best_pose_candidate['pseudo_stats']['median']:.2f}px, "
            f"pseudo max {current_pseudo_stats['max']:.2f}px -> {best_pose_candidate['pseudo_stats']['max']:.2f}px, "
            f"accepted p95 {current_existing_stats['p95']:.2f}px -> {best_pose_candidate['existing_stats']['p95']:.2f}px, "
            f"global mean {before_mean:.4f}px -> {after_mean:.4f}px"
        )

    if refined_total <= 0:
        print(f"{label}: улучшений не найдено")

    return refined_total


def refine_cameras_from_rejected_single_view_tracks(
    calib_data,
    max_cameras=2,
    max_tracks_per_camera=6,
    min_total_correspondences=8,
):
    if not calib_data.get('cameras'):
        return 0

    diagnostics = triangulation.diagnose_unreconstructed_tracks(
        calib_data.get('points_3d', {}),
        calib_data.get('cameras', {}),
        calib_data.get('camera_points', {}),
        calib_data.get('K'),
        calib_data.get('dist_coeffs'),
        min_track_length=3,
        strict_track_consistency=bool(calib_data.get('strict_track_consistency', True)),
        top_k=0,
        debug_logging=bool(calib_data.get('debug_logging', False)),
        observation_confidences=calib_data.get('observation_confidences'),
        line_support_data=calib_data.get('line_support_data'),
    )
    if not diagnostics:
        return 0

    by_camera = {}
    for item in diagnostics:
        if item.get('conflict_class') != 'single_view_conflict':
            continue
        if item.get('reason') not in {'strict_subset_only', 'backfillable_full_track'}:
            continue
        best_point = item.get('best_point')
        worst_observations = list(item.get('worst_observations', []))
        if best_point is None or not worst_observations:
            continue

        worst_error = float(worst_observations[0].get('error', float('inf')))
        second_error = float(worst_observations[1].get('error', 0.0)) if len(worst_observations) > 1 else 0.0
        if worst_error < max(2.5, second_error + 0.75, second_error * 1.35):
            continue

        worst_camera_id = str(worst_observations[0].get('camera_id'))
        point_id = item.get('point_id')
        if (
            worst_camera_id not in calib_data.get('cameras', {}) or
            point_id not in calib_data.get('camera_points', {}).get(worst_camera_id, {})
        ):
            continue

        severity = (
            max(0.0, worst_error - second_error) * 0.55 +
            max(0.0, worst_error - 2.0) * 0.25 +
            float(item.get('track_length', 0)) * 0.10 +
            max(0.0, float(item.get('best_max') or worst_error) - 1.0) * 0.10
        )
        by_camera.setdefault(worst_camera_id, []).append({
            'point_id': point_id,
            'point_3d': np.asarray(best_point, dtype=np.float64).reshape(3),
            'point_2d': np.asarray(
                calib_data['camera_points'][worst_camera_id][point_id],
                dtype=np.float64,
            ).reshape(2),
            'worst_error': worst_error,
            'second_error': second_error,
            'severity': float(severity),
            'reason': str(item.get('reason') or ''),
        })

    return _refine_cameras_from_pseudo_track_entries(
        calib_data,
        by_camera,
        label="Refine rejected single-view cameras",
        max_cameras=max_cameras,
        max_tracks_per_camera=max_tracks_per_camera,
        min_total_correspondences=min_total_correspondences,
    )


def refine_cameras_from_asymmetric_tracks(
    calib_data,
    max_cameras=2,
    max_tracks_per_camera=8,
    min_total_correspondences=8,
    min_track_length=3,
):
    if not calib_data.get('cameras') or not calib_data.get('points_3d'):
        return 0

    required_track_length = max(3, int(min_track_length))
    by_camera = {}
    for point_id in sorted(calib_data.get('points_3d', {}).keys(), key=_stable_sort_key):
        point_observations = {
            str(camera_id): calib_data['camera_points'][str(camera_id)][point_id]
            for camera_id in sorted(calib_data['cameras'].keys(), key=_stable_sort_key)
            if point_id in calib_data.get('camera_points', {}).get(str(camera_id), {})
        }
        if len(point_observations) < required_track_length:
            continue

        point_observation_confidences = triangulation._extract_point_observation_confidences(
            calib_data.get('observation_confidences'),
            point_id,
            point_observations.keys(),
        )
        accepted, refined_point, metrics = triangulation.evaluate_multiview_point(
            calib_data['points_3d'][point_id],
            point_observations,
            calib_data['cameras'],
            calib_data['K'],
            calib_data.get('dist_coeffs'),
            allow_subset=False,
            min_subset_views=required_track_length,
            observation_confidences=point_observation_confidences,
            point_id=point_id,
            points_3d_context=calib_data.get('points_3d'),
            line_support_data=calib_data.get('line_support_data'),
        )
        if metrics is None:
            continue

        finite_entries = []
        for camera_id, error in zip(metrics.get('camera_ids', []), metrics.get('errors', [])):
            if np.isfinite(error):
                finite_entries.append((float(error), str(camera_id)))
        if len(finite_entries) < required_track_length:
            continue

        finite_entries.sort(reverse=True)
        worst_error, worst_camera_id = finite_entries[0]
        second_error = finite_entries[1][0] if len(finite_entries) > 1 else 0.0
        finite_errors = np.asarray([entry[0] for entry in finite_entries], dtype=np.float64)
        median_error = float(np.median(finite_errors))
        asymmetry = worst_error - median_error
        if (
            asymmetry < 0.75 or
            worst_error < max(2.5, second_error + 0.75, second_error * 1.35, median_error + 0.50)
        ):
            continue

        kept_camera_ids = [
            camera_id for camera_id in sorted(point_observations.keys(), key=_stable_sort_key)
            if camera_id != worst_camera_id
        ]
        if len(kept_camera_ids) < 2:
            continue

        kept_observations = {
            camera_id: point_observations[camera_id]
            for camera_id in kept_camera_ids
        }
        best_subset = None
        for camera_a_id, camera_b_id in itertools.combinations(kept_camera_ids, 2):
            candidate_point = triangulation._triangulate_global_point_from_pair(
                camera_a_id,
                camera_b_id,
                kept_observations,
                calib_data['cameras'],
                calib_data['K'],
                dist_coeffs=calib_data.get('dist_coeffs'),
                debug_logging=bool(calib_data.get('debug_logging', False)),
            )
            if candidate_point is None:
                continue

            subset_accepted, subset_refined_point, subset_metrics = triangulation.evaluate_multiview_point(
                candidate_point,
                kept_observations,
                calib_data['cameras'],
                calib_data['K'],
                calib_data.get('dist_coeffs'),
                allow_subset=False,
                min_subset_views=max(2, min(required_track_length, len(kept_observations))),
                observation_confidences=point_observation_confidences,
                point_id=point_id,
                points_3d_context=calib_data.get('points_3d'),
                line_support_data=calib_data.get('line_support_data'),
            )
            if not subset_accepted or subset_metrics is None:
                continue

            subset_max = float(subset_metrics.get('max_error', float('inf')))
            subset_mean = float(subset_metrics.get('mean_error', float('inf')))
            if worst_error < max(subset_max + 0.90, subset_mean + 1.20, 3.0):
                continue

            candidate_score = (
                (worst_error - subset_max) * 0.65 +
                (float(metrics.get('mean_error', median_error)) - subset_mean) * 0.20 +
                (asymmetry - max(0.0, subset_max - subset_mean)) * 0.15
            )
            if best_subset is None or candidate_score > best_subset['score'] + 1e-6:
                best_subset = {
                    'score': float(candidate_score),
                    'point_3d': np.asarray(subset_refined_point, dtype=np.float64).reshape(3),
                    'point_2d': np.asarray(point_observations[worst_camera_id], dtype=np.float64).reshape(2),
                    'severity': float(
                        max(0.0, worst_error - second_error) * 0.55 +
                        max(0.0, worst_error - subset_max) * 0.30 +
                        max(0.0, asymmetry) * 0.15
                    ),
                }

        if best_subset is None:
            continue

        by_camera.setdefault(worst_camera_id, []).append({
            'point_id': point_id,
            'point_3d': best_subset['point_3d'],
            'point_2d': best_subset['point_2d'],
            'worst_error': float(worst_error),
            'second_error': float(second_error),
            'severity': float(best_subset['severity']),
            'reason': 'asymmetric_track',
        })

    return _refine_cameras_from_pseudo_track_entries(
        calib_data,
        by_camera,
        label="Refine asymmetric-track cameras",
        max_cameras=max_cameras,
        max_tracks_per_camera=max_tracks_per_camera,
        min_total_correspondences=min_total_correspondences,
    )


def refine_high_error_cameras(
    calib_data,
    max_cameras=2,
    min_observations_per_camera=6,
    min_track_length=3,
):
    if not calib_data.get('cameras') or not calib_data.get('points_3d'):
        return 0

    global_stats = summarize_reprojection_error_distribution(calib_data, top_k=0)
    camera_profiles = summarize_camera_error_profiles(calib_data, top_k=3)
    if not camera_profiles or global_stats['count'] <= 0:
        return 0

    global_p95 = max(global_stats['p95'], 1e-6)
    candidate_profiles = []
    for profile in camera_profiles:
        if profile['count'] <= min_observations_per_camera:
            continue
        precision_target_mean = float(calib_data.get('precision_target_mean_px', 0.5))
        absolute_rmse_exceeded = profile.get('mean', 0.0) > precision_target_mean * 1.6
        if (
            profile['p95'] <= max(global_p95 * 1.20, 3.5) and
            profile['max'] <= max(global_p95 * 1.75, 5.5) and
            not absolute_rmse_exceeded
        ):
            continue
        severity = max(
            profile['p95'] / global_p95,
            profile['max'] / max(global_p95 * 1.5, 1e-6),
        )
        candidate_profiles.append((severity, profile))

    if not candidate_profiles:
        print("Профили камер: локальных хвостов ошибок не найдено")
        return 0

    removed_total = 0
    print("Локальная доочистка камер с плохим хвостом ошибок:")
    for _, profile in sorted(candidate_profiles, key=lambda item: item[0], reverse=True)[:max(1, int(max_cameras))]:
        camera_id = profile['camera_id']
        before_error, _, _ = calculate_reprojection_errors(calib_data)
        original_observations = dict(calib_data['camera_points'].get(camera_id, {}))
        filter_stats = _filter_camera_observations_locally(
            calib_data,
            camera_id,
            protected_point_ids=None,
            absolute_threshold=max(6.0, global_p95 * 1.35),
            sigma_multiplier=2.0,
            mad_multiplier=2.5,
            min_observations_per_camera=min_observations_per_camera,
            min_track_length=min_track_length,
        )
        if filter_stats['removed_observations'] <= 0:
            continue

        after_error, _, _ = calculate_reprojection_errors(calib_data)
        improvement = before_error - after_error
        if after_error > before_error or improvement < 0.05:
            calib_data['camera_points'][camera_id] = original_observations
            continue

        removed_total += int(filter_stats['removed_observations'])
        refreshed_profiles = {
            item['camera_id']: item
            for item in summarize_camera_error_profiles(calib_data, top_k=3)
        }
        refreshed_profile = refreshed_profiles.get(camera_id)
        if refreshed_profile is not None:
            print(
                f"  - Камера {camera_id}: удалено {filter_stats['removed_observations']} наблюдений, "
                f"p95 {profile['p95']:.2f}px -> {refreshed_profile['p95']:.2f}px, "
                f"max {profile['max']:.2f}px -> {refreshed_profile['max']:.2f}px"
            )
        else:
            print(
                f"  - Камера {camera_id}: удалено {filter_stats['removed_observations']} наблюдений, "
                f"ошибка {before_error:.4f}px -> {after_error:.4f}px"
            )

    if removed_total <= 0:
        print("Локальная доочистка камер: изменений не потребовалось")

    return removed_total


def _resolve_camera_image_path(calib_data, camera_id):
    images = calib_data.get('images', {}) or {}
    camera_key = str(camera_id)
    image_path = images.get(camera_key, images.get(camera_id))
    if image_path is None:
        return None

    image_path = str(image_path)
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    if os.path.exists(image_path):
        return image_path

    image_root = calib_data.get('image_root') or calib_data.get('images_root')
    if image_root:
        root_candidate = os.path.join(str(image_root), image_path)
        if os.path.exists(root_candidate):
            return root_candidate
        basename_candidate = os.path.join(str(image_root), os.path.basename(image_path))
        if os.path.exists(basename_candidate):
            return basename_candidate
    return None


def _sample_bilinear_single_channel(image, x, y):
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return 0.0

    x = float(np.clip(x, 0.0, max(width - 1, 0)))
    y = float(np.clip(y, 0.0, max(height - 1, 0)))
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    dx = x - x0
    dy = y - y0

    top = float(image[y0, x0]) * (1.0 - dx) + float(image[y0, x1]) * dx
    bottom = float(image[y1, x0]) * (1.0 - dx) + float(image[y1, x1]) * dx
    return float(top * (1.0 - dy) + bottom * dy)


def _suggest_guided_observation_repair(gray_u8, eigen_map, observed_xy, predicted_xy, max_shift_px=18.0):
    observed_xy = np.asarray(observed_xy, dtype=np.float64).reshape(2)
    predicted_xy = np.asarray(predicted_xy, dtype=np.float64).reshape(2)
    shift = float(np.linalg.norm(predicted_xy - observed_xy))
    if not np.isfinite(shift) or shift < 0.75 or shift > float(max_shift_px):
        return None

    height, width = gray_u8.shape[:2]
    if (
        predicted_xy[0] < 3 or predicted_xy[1] < 3 or
        predicted_xy[0] >= width - 3 or predicted_xy[1] >= height - 3
    ):
        return None

    search_radius = int(np.clip(max(6.0, shift * 1.15 + 2.0), 6.0, float(max_shift_px) + 2.0))
    x0 = max(0, int(np.floor(predicted_xy[0])) - search_radius)
    x1 = min(width, int(np.floor(predicted_xy[0])) + search_radius + 1)
    y0 = max(0, int(np.floor(predicted_xy[1])) - search_radius)
    y1 = min(height, int(np.floor(predicted_xy[1])) + search_radius + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None

    roi = gray_u8[y0:y1, x0:x1]
    if roi.size == 0 or float(np.std(roi.astype(np.float32))) < 3.0:
        return None

    roi_eigen = eigen_map[y0:y1, x0:x1]
    local_peak = float(np.max(roi_eigen)) if roi_eigen.size else 0.0
    if local_peak <= 1e-8:
        return None

    candidate_points = []
    try:
        corners = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=10,
            qualityLevel=0.01,
            minDistance=2,
            blockSize=3,
            useHarrisDetector=False,
        )
    except cv2.error:
        corners = None

    if corners is not None:
        for corner in corners.reshape(-1, 2):
            candidate_points.append(np.array([corner[0] + x0, corner[1] + y0], dtype=np.float64))
    candidate_points.append(predicted_xy.astype(np.float64))

    termination = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        0.01,
    )
    gray_f32 = gray_u8.astype(np.float32)

    best_candidate = None
    best_score = None
    for candidate_xy in candidate_points:
        if (
            candidate_xy[0] < 3 or candidate_xy[1] < 3 or
            candidate_xy[0] >= width - 3 or candidate_xy[1] >= height - 3
        ):
            continue

        refined = np.asarray(candidate_xy, dtype=np.float32).reshape(1, 1, 2)
        try:
            cv2.cornerSubPix(gray_f32, refined, (2, 2), (-1, -1), termination)
            candidate_xy = refined.reshape(2).astype(np.float64)
        except cv2.error:
            candidate_xy = np.asarray(candidate_xy, dtype=np.float64).reshape(2)

        total_shift = float(np.linalg.norm(candidate_xy - observed_xy))
        predicted_distance = float(np.linalg.norm(candidate_xy - predicted_xy))
        if total_shift > float(max_shift_px):
            continue
        if predicted_distance > max(5.0, shift * 0.95):
            continue

        response = _sample_bilinear_single_channel(eigen_map, candidate_xy[0], candidate_xy[1])
        response_score = float(np.clip(response / max(local_peak, 1e-8), 0.0, 1.0))
        if response_score < 0.22:
            continue

        score = (
            predicted_distance * 0.70 +
            total_shift * 0.20 -
            response_score * 0.60
        )
        if best_candidate is None or score < best_score - 1e-6:
            best_candidate = candidate_xy.astype(np.float32)
            best_score = float(score)

    if best_candidate is None:
        return None

    improved_distance = float(np.linalg.norm(np.asarray(best_candidate, dtype=np.float64) - predicted_xy))
    if improved_distance >= shift - 0.35:
        return None

    return np.asarray(best_candidate, dtype=np.float32).reshape(2)


def repair_single_view_conflict_observations(
    calib_data,
    target_mean=0.5,
    target_p95=1.0,
    target_max=1.0,
    max_candidates=6,
    max_shift_px=18.0,
):
    if not calib_data.get('cameras') or not calib_data.get('camera_points'):
        return 0

    diagnostics = triangulation.diagnose_unreconstructed_tracks(
        calib_data.get('points_3d', {}),
        calib_data.get('cameras', {}),
        calib_data.get('camera_points', {}),
        calib_data.get('K'),
        calib_data.get('dist_coeffs'),
        min_track_length=3,
        strict_track_consistency=bool(calib_data.get('strict_track_consistency', True)),
        top_k=0,
        debug_logging=bool(calib_data.get('debug_logging', False)),
        observation_confidences=calib_data.get('observation_confidences'),
        line_support_data=calib_data.get('line_support_data'),
    )

    candidates = []
    for item in diagnostics:
        if item.get('conflict_class') != 'single_view_conflict':
            continue
        if item.get('reason') not in {'strict_subset_only', 'backfillable_full_track'}:
            continue
        if item.get('best_point') is None:
            continue
        worst_observations = list(item.get('worst_observations', []))
        if not worst_observations:
            continue
        worst_error = float(worst_observations[0].get('error', float('inf')))
        second_error = float(worst_observations[1].get('error', 0.0)) if len(worst_observations) > 1 else 0.0
        if worst_error < max(2.0, second_error + 0.60, second_error * 1.30):
            continue
        candidates.append(item)

    if not candidates:
        print("Guided single-view observation repair: кандидатов не найдено")
        return 0

    image_cache = {}
    repaired_total = 0
    print("Guided single-view observation repair:")
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.get('track_length', 0),
            float(item.get('best_max') or 0.0),
            float(item.get('best_mean') or 0.0),
        ),
        reverse=True,
    )[:max(1, int(max_candidates))]:
        point_id = candidate['point_id']
        worst_info = list(candidate.get('worst_observations', []))[0]
        worst_camera_id = str(worst_info.get('camera_id'))
        if worst_camera_id not in calib_data.get('camera_points', {}):
            continue
        if point_id not in calib_data['camera_points'].get(worst_camera_id, {}):
            continue

        image_path = _resolve_camera_image_path(calib_data, worst_camera_id)
        if image_path is None:
            continue
        if image_path not in image_cache:
            image_u8 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image_u8 is None or image_u8.size == 0:
                image_cache[image_path] = None
            else:
                image_cache[image_path] = (
                    image_u8,
                    cv2.cornerMinEigenVal(np.ascontiguousarray(image_u8), blockSize=3, ksize=3),
                )
        cache_entry = image_cache.get(image_path)
        if cache_entry is None:
            continue

        image_u8, eigen_map = cache_entry
        camera_K = _get_camera_matrix(calib_data, worst_camera_id)
        R_worst, t_worst = calib_data['cameras'][worst_camera_id]
        try:
            rvec_worst, _ = cv2.Rodrigues(np.asarray(R_worst, dtype=np.float64))
            projected, _ = cv2.projectPoints(
                np.asarray(candidate['best_point'], dtype=np.float64).reshape(1, 3),
                rvec_worst,
                np.asarray(t_worst, dtype=np.float64).reshape(3, 1),
                np.asarray(camera_K, dtype=np.float64),
                np.asarray(
                    calib_data.get('dist_coeffs') if calib_data.get('dist_coeffs') is not None else np.zeros(5),
                    dtype=np.float64,
                ),
            )
        except cv2.error:
            continue

        predicted_xy = np.asarray(projected, dtype=np.float64).reshape(2)
        observed_xy = np.asarray(calib_data['camera_points'][worst_camera_id][point_id], dtype=np.float64).reshape(2)
        repaired_xy = _suggest_guided_observation_repair(
            image_u8,
            eigen_map,
            observed_xy,
            predicted_xy,
            max_shift_px=max_shift_px,
        )
        if repaired_xy is None:
            continue

        point_observations = {
            str(camera_id): calib_data['camera_points'][str(camera_id)][point_id]
            for camera_id in sorted(calib_data['cameras'].keys(), key=_stable_sort_key)
            if point_id in calib_data['camera_points'].get(str(camera_id), {})
        }
        if len(point_observations) < 3:
            continue

        point_observation_confidences = triangulation._extract_point_observation_confidences(
            calib_data.get('observation_confidences'),
            point_id,
            point_observations.keys(),
        )

        before_mean, _, _ = calculate_reprojection_errors(calib_data)
        before_stats = summarize_reprojection_error_distribution(calib_data, top_k=0)
        original_point_2d = calib_data['camera_points'][worst_camera_id][point_id]
        calib_data['camera_points'][worst_camera_id][point_id] = np.asarray(repaired_xy, dtype=np.float32).reshape(2)
        point_observations[worst_camera_id] = calib_data['camera_points'][worst_camera_id][point_id]

        accepted, refined_point, metrics = triangulation.evaluate_multiview_point(
            np.asarray(candidate['best_point'], dtype=np.float64).reshape(3),
            point_observations,
            calib_data['cameras'],
            calib_data['K'],
            calib_data.get('dist_coeffs'),
            allow_subset=False,
            min_subset_views=min(3, len(point_observations)),
            observation_confidences=point_observation_confidences,
            point_id=point_id,
            points_3d_context=calib_data.get('points_3d'),
            line_support_data=calib_data.get('line_support_data'),
        )
        if accepted:
            calib_data['points_3d'][point_id] = np.asarray(refined_point, dtype=np.float32).reshape(3)

        after_mean, _, _ = calculate_reprojection_errors(calib_data)
        after_stats = summarize_reprojection_error_distribution(calib_data, top_k=0)
        candidate_mean = float(metrics.get('mean_error', float('inf'))) if metrics is not None else float('inf')
        candidate_max = float(metrics.get('max_error', float('inf'))) if metrics is not None else float('inf')
        candidate_ok = (
            bool(accepted) and
            candidate_mean <= max(float(target_p95) * 0.90, 0.90) and
            candidate_max <= max(float(target_max) * 1.35, 1.50)
        )
        scene_ok = (
            after_mean <= max(before_mean + 0.05, before_mean * 1.10, float(target_mean) * 1.35) and
            after_stats['p95'] <= max(before_stats['p95'] + 0.12, before_stats['p95'] * 1.10, float(target_p95) * 1.20) and
            after_stats['max'] <= max(before_stats['max'] + 0.35, float(target_max) * 1.80)
        )
        if not (candidate_ok and scene_ok):
            calib_data['camera_points'][worst_camera_id][point_id] = original_point_2d
            if point_id in calib_data['points_3d']:
                del calib_data['points_3d'][point_id]
            continue

        repaired_total += 1
        shift_px = float(np.linalg.norm(np.asarray(repaired_xy, dtype=np.float64) - observed_xy))
        print(
            f"  - Точка {point_id}: камера {worst_camera_id}, "
            f"2D shift={shift_px:.2f}px, track mean={candidate_mean:.2f}px, max={candidate_max:.2f}px, "
            f"global mean {before_mean:.4f}px -> {after_mean:.4f}px"
        )

    if repaired_total <= 0:
        print("Guided single-view observation repair: улучшений не найдено")
    return int(repaired_total)

def filter_extreme_tail_observations(
    calib_data,
    max_removals=3,
    min_observations_per_camera=6,
    min_track_length=3,
    max_fraction_per_camera=0.15,
    force_allow=False,
):
    if not calib_data.get('cameras') or not calib_data.get('points_3d'):
        return 0

    if bool(calib_data.get('strict_track_consistency', True)) and not force_allow:
        print("Post-BA tail cleanup пропущен: strict_track_consistency включен")
        return 0

    before_stats = summarize_reprojection_error_distribution(calib_data, top_k=max(5, int(max_removals)))
    if before_stats['count'] < 16:
        return 0

    observation_errors = []
    point_track_lengths = {}
    point_error_map = {}

    for camera_id, observations in calib_data['camera_points'].items():
        for point_id in observations.keys():
            point_track_lengths[point_id] = point_track_lengths.get(point_id, 0) + 1

    for camera_id, (R, t) in calib_data['cameras'].items():
        observations = calib_data['camera_points'].get(str(camera_id), {})
        for point_id, point_2d in observations.items():
            if point_id not in calib_data['points_3d']:
                continue

            projected_point = utils.project_point(
                calib_data['points_3d'][point_id],
                R,
                t,
                _get_camera_matrix(calib_data, camera_id),
                calib_data.get('dist_coeffs')
            )
            error = float(np.linalg.norm(projected_point - point_2d))
            observation_errors.append((str(camera_id), point_id, error))
            point_error_map.setdefault(point_id, []).append(error)

    if len(observation_errors) < 16:
        return 0

    errors_only = np.array([item[2] for item in observation_errors], dtype=np.float64)
    global_median = float(np.median(errors_only))
    global_p95 = float(np.percentile(errors_only, 95))
    global_max = float(np.max(errors_only))
    global_mad = float(stats.median_abs_deviation(errors_only, scale='normal'))
    scale_floor = 0.35 if force_allow else 0.75
    global_scale = max(global_mad, scale_floor)

    tail_abs_floor = 1.5 if force_allow else 4.5
    tail_threshold = max(global_p95 * 1.18, global_median + global_scale * 3.0, tail_abs_floor)
    if global_max <= tail_threshold:
        print("Post-BA tail cleanup: экстремального хвоста не найдено")
        return 0

    candidate_removals = []
    for camera_id in calib_data['cameras'].keys():
        camera_entries = [item for item in observation_errors if item[0] == str(camera_id)]
        if len(camera_entries) <= min_observations_per_camera:
            continue

        camera_errors = np.array([item[2] for item in camera_entries], dtype=np.float64)
        camera_median = float(np.median(camera_errors))
        camera_mad = float(stats.median_abs_deviation(camera_errors, scale='normal'))
        camera_scale_floor = 0.30 if force_allow else 0.75
        camera_scale = max(camera_mad, global_scale * 0.75, camera_scale_floor)
        camera_threshold = max(global_p95, camera_median + 2.5 * camera_scale)

        for _, point_id, error in camera_entries:
            if point_track_lengths.get(point_id, 0) <= min_track_length:
                continue

            point_errors = np.array(point_error_map.get(point_id, []), dtype=np.float64)
            if point_errors.size <= min_track_length:
                continue

            point_median = float(np.median(point_errors))
            point_mad = float(stats.median_abs_deviation(point_errors, scale='normal'))
            point_scale_floor = 0.25 if force_allow else 0.6
            point_scale = max(point_mad, global_scale * 0.5, point_scale_floor)
            point_threshold = max(global_p95 * 0.95, point_median + 2.2 * point_scale)
            global_threshold_floor = 1.5 if force_allow else 4.75
            global_threshold = max(global_p95 * 1.08, global_median + 3.0 * global_scale, global_threshold_floor)

            if error <= global_threshold or error <= camera_threshold or error <= point_threshold:
                continue

            severity = max(
                error / max(global_threshold, 1e-6),
                error / max(camera_threshold, 1e-6),
                error / max(point_threshold, 1e-6),
            )
            candidate_removals.append((severity, str(camera_id), point_id, error))

    if not candidate_removals:
        print("Post-BA tail cleanup: кандидатов для удаления не найдено")
        return 0

    before_error, _, _ = calculate_reprojection_errors(calib_data)
    camera_initial_counts = {
        str(camera_id): len(calib_data['camera_points'].get(str(camera_id), {}))
        for camera_id in calib_data['cameras'].keys()
    }
    camera_remaining = {
        str(camera_id): len(calib_data['camera_points'].get(str(camera_id), {}))
        for camera_id in calib_data['cameras'].keys()
    }
    camera_limits = {
        camera_id: max(
            1,
            min(
                max(1, initial_count - min_observations_per_camera),
                int(np.ceil(initial_count * max_fraction_per_camera)),
            ),
        )
        for camera_id, initial_count in camera_initial_counts.items()
    }
    point_remaining = dict(point_track_lengths)
    removed_by_camera = {}
    removed_points = set()
    removed_records = []

    for _, camera_id, point_id, error in sorted(candidate_removals, key=lambda item: (item[0], item[3]), reverse=True):
        if len(removed_records) >= max(1, int(max_removals)):
            break
        if camera_remaining.get(camera_id, 0) <= min_observations_per_camera:
            continue
        if point_remaining.get(point_id, 0) <= min_track_length:
            continue
        if removed_by_camera.get(camera_id, 0) >= camera_limits.get(camera_id, 1):
            continue
        if point_id in removed_points:
            continue
        if point_id not in calib_data['camera_points'].get(camera_id, {}):
            continue

        removed_records.append((camera_id, point_id, calib_data['camera_points'][camera_id][point_id], error))
        del calib_data['camera_points'][camera_id][point_id]
        camera_remaining[camera_id] -= 1
        point_remaining[point_id] -= 1
        removed_by_camera[camera_id] = removed_by_camera.get(camera_id, 0) + 1
        removed_points.add(point_id)

    if not removed_records:
        print("Post-BA tail cleanup: кандидаты есть, но их нельзя удалить без потери устойчивости")
        return 0

    after_error, _, _ = calculate_reprojection_errors(calib_data)
    after_stats = summarize_reprojection_error_distribution(calib_data, top_k=max(5, int(max_removals)))

    tail_improved = (
        after_stats['p95'] <= before_stats['p95'] * 0.97 or
        after_stats['max'] <= before_stats['max'] - 0.75
    )
    mean_not_worse = after_error <= before_error + 0.05
    if not tail_improved or not mean_not_worse:
        for camera_id, point_id, point_2d, _ in removed_records:
            calib_data['camera_points'][camera_id][point_id] = point_2d
        print(
            "Post-BA tail cleanup отменен: "
            f"mean {before_error:.4f}px -> {after_error:.4f}px, "
            f"p95 {before_stats['p95']:.4f}px -> {after_stats['p95']:.4f}px, "
            f"max {before_stats['max']:.4f}px -> {after_stats['max']:.4f}px"
        )
        return 0

    print("Post-BA tail cleanup:")
    print(
        f"  - Удалено наблюдений: {len(removed_records)}"
    )
    print(
        f"  - Mean: {before_error:.4f}px -> {after_error:.4f}px"
    )
    print(
        f"  - P95: {before_stats['p95']:.4f}px -> {after_stats['p95']:.4f}px"
    )
    print(
        f"  - Max: {before_stats['max']:.4f}px -> {after_stats['max']:.4f}px"
    )
    for camera_id, count in sorted(removed_by_camera.items()):
        print(f"  - Камера {camera_id}: удалено наблюдений {count}")

    return len(removed_records)


def run_precision_first_cleanup(
    calib_data,
    target_mean=0.5,
    target_p95=1.0,
    target_max=1.0,
    max_rounds=4,
    min_observations_per_camera=6,
    min_track_length=3,
    min_points=12,
    focal_range=(800, 3000),
    force_same_focal=False,
    progress_callback=None,
    progress_range=(89.35, 89.55),
):
    if not calib_data.get('cameras') or not calib_data.get('points_3d'):
        return 0

    target_mean = float(max(target_mean, 0.05))
    target_p95 = float(max(target_p95, target_mean))
    target_max = float(max(target_max, target_p95))
    max_rounds = max(1, int(max_rounds))
    min_points = max(8, int(min_points))

    def _report_progress(progress_value, status_text):
        if progress_callback is None:
            return
        try:
            progress_callback(progress_value, status_text)
        except Exception:
            pass

    print("Precision-first cleanup:")
    preview_mode = bool(calib_data.get('_project_level_preview_mode', False))
    total_removed_observations = 0
    total_removed_points = 0
    current_mean, _, _ = calculate_reprojection_errors(calib_data)
    current_stats = summarize_reprojection_error_distribution(calib_data, top_k=5)

    if (
        current_stats['count'] > 0 and
        current_mean <= target_mean and
        current_stats['p95'] <= target_p95 and
        current_stats['max'] <= target_max
    ):
        print(
            f"  - Цель уже достигнута: mean={current_mean:.4f}px, "
            f"p95={current_stats['p95']:.4f}px, max={current_stats['max']:.4f}px"
        )
        return 0

    try:
        asymmetric_pose_refined = refine_cameras_from_asymmetric_tracks(
            calib_data,
            max_cameras=2,
            max_tracks_per_camera=8,
            min_total_correspondences=max(min_observations_per_camera + 2, 8),
            min_track_length=max(3, int(min_track_length)),
        )
        if asymmetric_pose_refined > 0:
            print("  - После asymmetric-track pose refine выполняем дополнительный шаг оптимизации...")
            refine_reconstruction(
                calib_data,
                max_iterations=1,
                focal_range=focal_range,
                force_same_focal=force_same_focal,
                optimize_intrinsics=False,
                optimize_distortion=False,
                progress_callback=progress_callback,
                progress_range=(progress_range[0], progress_range[0]),
            )
    except Exception as e:
        print(f"  - Asymmetric-track pose refine пропущен: {str(e)}")

    for round_index in range(max_rounds):
        before_mean, _, _ = calculate_reprojection_errors(calib_data)
        before_stats = summarize_reprojection_error_distribution(calib_data, top_k=5)
        if before_stats['count'] < 12:
            print("  - Precision cleanup остановлен: слишком мало наблюдений")
            break

        if (
            before_mean <= target_mean and
            before_stats['p95'] <= target_p95 and
            before_stats['max'] <= target_max
        ):
            break

        round_start, round_end = progress_range
        round_span = max(round_end - round_start, 0.0) / max(max_rounds, 1)
        _report_progress(
            round_start + round_span * round_index,
            f"Precision cleanup: раунд {round_index + 1}/{max_rounds}"
        )

        snapshot = _snapshot_calibration_state(calib_data)
        point_track_removed = triangulation.repair_asymmetric_point_tracks(
            calib_data['points_3d'],
            calib_data['cameras'],
            calib_data['camera_points'],
            calib_data['K'],
            calib_data.get('dist_coeffs'),
            max_points=max(4, min(8, len(calib_data.get('points_3d', {})) // 4)),
            min_track_length=max(3, int(min_track_length)),
            max_removed_observations=2,
                    strict_track_consistency=bool(calib_data.get('strict_track_consistency', True)),
                    debug_logging=bool(calib_data.get('debug_logging', False)),
                    observation_confidences=calib_data.get('observation_confidences'),
                    secondary_seed_points=calib_data.setdefault('_secondary_seed_points_3d', {}),
                    line_support_data=calib_data.get('line_support_data'),
                    camera_intrinsics=_collect_camera_intrinsics_map(calib_data),
                )
        repaired_track_points = int(point_track_removed.get('repaired_points', 0)) if isinstance(point_track_removed, dict) else 0
        repaired_track_observations = int(point_track_removed.get('removed_observations', 0)) if isinstance(point_track_removed, dict) else 0
        repaired_track_point_removals = int(point_track_removed.get('removed_points', 0)) if isinstance(point_track_removed, dict) else 0
        dynamic_observation_threshold = max(
            target_max * 1.05,
            min(
                before_stats['p95'] * 0.90,
                max(before_mean * 1.35, target_p95 * 1.10, 1.15)
            )
        )
        observation_removed, _, _ = filter_observations_by_reprojection_error(
            calib_data,
            absolute_threshold=dynamic_observation_threshold,
            sigma_multiplier=1.75,
            mad_multiplier=1.90,
            min_observations_per_camera=min_observations_per_camera,
            min_track_length=min_track_length,
            min_improvement=0.01,
            min_relative_improvement=0.01,
            force_allow=True,
        )
        tail_removed = filter_extreme_tail_observations(
            calib_data,
            max_removals=max(4, min(10, int(np.ceil(before_stats['count'] * 0.06)))),
            min_observations_per_camera=min_observations_per_camera,
            min_track_length=min_track_length,
            max_fraction_per_camera=0.30,
            force_allow=True,
        )

        removed_points = 0
        if (
            before_mean > target_mean * 1.8 or
            before_stats['p95'] > target_p95 * 2.0 or
            before_stats['max'] > target_max * 3.0
        ):
            removed_points, _, _, _ = triangulation.filter_outliers_by_reprojection_error(
                calib_data,
                absolute_threshold=max(target_max * 1.8, 2.0),
                sigma_multiplier=2.0,
                mad_multiplier=2.2,
            )
            if removed_points > 0:
                purged = _purge_missing_point_observations(calib_data)
                if purged > 0:
                    print(f"  - Precision cleanup: удалено dangling observations {purged}")

        # Прямое удаление наблюдений с ошибкой > target_max * 1.3
        # при сохранении min_track_length для каждой точки
        direct_removed = 0
        direct_threshold = max(target_max * 1.2, before_stats['p95'] * 0.85)
        if before_stats['max'] > direct_threshold:
            point_track_lens = {}
            for cam_id, obs in calib_data.get('camera_points', {}).items():
                for pid in obs:
                    point_track_lens[pid] = point_track_lens.get(pid, 0) + 1
            cam_obs_counts = {
                cam_id: len(obs)
                for cam_id, obs in calib_data.get('camera_points', {}).items()
            }
            to_remove = []
            for cam_id, (R, t) in calib_data.get('cameras', {}).items():
                obs = calib_data['camera_points'].get(str(cam_id), {})
                if len(obs) <= min_observations_per_camera:
                    continue
                for pid, pt2d in obs.items():
                    if pid not in calib_data.get('points_3d', {}):
                        continue
                    if point_track_lens.get(pid, 0) <= max(min_track_length, 3):
                        continue
                    proj = utils.project_point(
                        calib_data['points_3d'][pid], R, t,
                        _get_camera_matrix(calib_data, cam_id), calib_data.get('dist_coeffs'))
                    err = float(np.linalg.norm(proj - pt2d))
                    if err > direct_threshold:
                        to_remove.append((err, str(cam_id), pid))
            to_remove.sort(reverse=True)
            for err, cam_id, pid in to_remove[:max(6, int(len(to_remove) * 0.5))]:
                if point_track_lens.get(pid, 0) <= max(min_track_length, 3):
                    continue
                if cam_obs_counts.get(cam_id, 0) <= min_observations_per_camera:
                    continue
                if pid in calib_data['camera_points'].get(cam_id, {}):
                    del calib_data['camera_points'][cam_id][pid]
                    point_track_lens[pid] = point_track_lens.get(pid, 1) - 1
                    cam_obs_counts[cam_id] = cam_obs_counts.get(cam_id, 1) - 1
                    direct_removed += 1
            if direct_removed > 0:
                print(f"  - Precision direct tail removal: удалено {direct_removed} наблюдений > {direct_threshold:.2f}px")

        round_removed = (
            repaired_track_points +
            repaired_track_observations +
            repaired_track_point_removals +
            int(observation_removed) +
            int(tail_removed) +
            int(removed_points) +
            int(direct_removed)
        )
        if round_removed <= 0:
            _restore_calibration_state(calib_data, snapshot)
            print("  - Precision cleanup: новых кандидатов для ужесточения не найдено")
            break

        if len(calib_data.get('points_3d', {})) < min_points:
            _restore_calibration_state(calib_data, snapshot)
            print(
                f"  - Precision cleanup отменен: точек стало слишком мало "
                f"({len(calib_data.get('points_3d', {}))} < {min_points})"
            )
            break

        refine_reconstruction(
            calib_data,
            max_iterations=2,
            focal_range=focal_range,
            force_same_focal=force_same_focal,
            optimize_intrinsics=False,
            optimize_distortion=False,
            progress_callback=progress_callback,
            progress_range=(
                round_start + round_span * round_index + round_span * 0.25,
                round_start + round_span * round_index + round_span * 0.85,
            )
        )

        after_mean, _, _ = calculate_reprojection_errors(calib_data)
        after_stats = summarize_reprojection_error_distribution(calib_data, top_k=5)
        improved = (
            after_mean <= before_mean - 0.04 or
            after_stats['p95'] <= before_stats['p95'] - 0.12 or
            after_stats['max'] <= before_stats['max'] - 0.25
        )
        not_too_sparse = len(calib_data.get('points_3d', {})) >= min_points
        if not improved or not not_too_sparse:
            _restore_calibration_state(calib_data, snapshot)
            print(
                f"  - Раунд {round_index + 1} отменен: "
                f"mean {before_mean:.4f}px -> {after_mean:.4f}px, "
                f"p95 {before_stats['p95']:.4f}px -> {after_stats['p95']:.4f}px, "
                f"max {before_stats['max']:.4f}px -> {after_stats['max']:.4f}px"
            )
            break

        total_removed_observations += repaired_track_observations + int(observation_removed) + int(tail_removed)
        total_removed_points += repaired_track_point_removals + int(removed_points)
        print(
            f"  - Раунд {round_index + 1}: "
            f"obs={repaired_track_observations + int(observation_removed) + int(tail_removed)}, "
            f"points={repaired_track_point_removals + int(removed_points)}, "
            f"repaired_tracks={repaired_track_points}, "
            f"mean {before_mean:.4f}px -> {after_mean:.4f}px, "
            f"p95 {before_stats['p95']:.4f}px -> {after_stats['p95']:.4f}px, "
            f"max {before_stats['max']:.4f}px -> {after_stats['max']:.4f}px, "
            f"points_3d={len(calib_data.get('points_3d', {}))}"
        )

    if total_removed_observations > 0 or total_removed_points > 0:
        _report_progress(progress_range[1], "Precision cleanup: финальный refine...")
        refine_reconstruction(
            calib_data,
            max_iterations=1,
            focal_range=focal_range,
            force_same_focal=force_same_focal,
            optimize_intrinsics=not preview_mode,
            optimize_distortion=not preview_mode,
            progress_callback=progress_callback,
            progress_range=(progress_range[1], progress_range[1]),
        )

    strict_removed_points = 0
    if bool(calib_data.get('strict_track_consistency', True)):
        _report_progress(progress_range[1], "Precision cleanup: strict full-track gate...")
        strict_min_points_remaining = max(
            8,
            min(
                max(int(min_points) - 1, 8),
                max(len(calib_data.get('points_3d', {})) - 1, 8)
            )
        )
        strict_removed_points = triangulation.remove_inconsistent_full_tracks(
            calib_data['points_3d'],
            calib_data['cameras'],
            calib_data['camera_points'],
            calib_data['K'],
            calib_data.get('dist_coeffs'),
            target_mean=target_mean,
            target_p95=target_p95,
            target_max=target_max,
            min_track_length=max(3, int(min_track_length)),
            min_points_remaining=strict_min_points_remaining,
            observation_confidences=calib_data.get('observation_confidences'),
            line_support_data=calib_data.get('line_support_data'),
            camera_intrinsics=_collect_camera_intrinsics_map(calib_data),
        )
        if strict_removed_points > 0:
            total_removed_points += int(strict_removed_points)
            refine_reconstruction(
                calib_data,
                max_iterations=1,
                focal_range=focal_range,
                force_same_focal=force_same_focal,
                optimize_intrinsics=False,
                optimize_distortion=False,
                progress_callback=progress_callback,
                progress_range=(progress_range[1], progress_range[1]),
            )

    final_mean, _, _ = calculate_reprojection_errors(calib_data)
    final_stats = summarize_reprojection_error_distribution(calib_data, top_k=5)
    print(
        f"Precision-first cleanup итог: mean={final_mean:.4f}px, "
        f"p95={final_stats['p95']:.4f}px, max={final_stats['max']:.4f}px, "
        f"удалено наблюдений={total_removed_observations}, удалено точек={total_removed_points}"
    )
    if (
        final_stats['count'] > 0 and
        final_mean <= target_mean and
        final_stats['p95'] <= target_p95 and
        final_stats['max'] <= target_max
    ):
        print("Precision-first cleanup: целевой уровень точности достигнут")
    else:
        print(
            "Precision-first cleanup: целевой уровень еще не достигнут "
            f"(target mean<={target_mean:.2f}, p95<={target_p95:.2f}, max<={target_max:.2f})"
        )

    return total_removed_observations + total_removed_points


def reintegrate_backfillable_tracks(
    calib_data,
    target_mean=0.5,
    target_p95=1.0,
    target_max=1.0,
    focal_range=(800, 3000),
    force_same_focal=False,
    max_candidates=4,
    progress_callback=None,
    progress_range=None,
):
    if not calib_data.get('cameras') or not calib_data.get('points_3d'):
        return 0

    diagnostics = triangulation.diagnose_unreconstructed_tracks(
        calib_data['points_3d'],
        calib_data['cameras'],
        calib_data['camera_points'],
        calib_data['K'],
        calib_data.get('dist_coeffs'),
        min_track_length=2,
        strict_track_consistency=bool(calib_data.get('strict_track_consistency', True)),
        top_k=0,
        debug_logging=bool(calib_data.get('debug_logging', False)),
        observation_confidences=calib_data.get('observation_confidences'),
        line_support_data=calib_data.get('line_support_data'),
    )
    candidates = [
        item for item in diagnostics
        if item.get('reason') == 'backfillable_full_track' and item.get('best_point') is not None
    ]
    if not candidates:
        print("Backfillable full-track re-entry: кандидатов не найдено")
        return 0

    candidates.sort(
        key=lambda item: (
            item.get('track_length', 0),
            0.0 if item.get('baseline') is None else item.get('baseline'),
            item.get('full_accept_count', 0),
            item.get('pair_success_count', 0),
            -(0.0 if item.get('best_max') is None else item.get('best_max')),
            -(0.0 if item.get('best_mean') is None else item.get('best_mean')),
        ),
        reverse=True,
    )

    def _report_progress(progress_value, status_text):
        if progress_callback is None or progress_range is None:
            return
        try:
            start_value, end_value = progress_range
            clamped_fraction = float(max(0.0, min(1.0, progress_value)))
            progress_callback(start_value + (end_value - start_value) * clamped_fraction, status_text)
        except Exception:
            pass

    print("Backfillable full-track re-entry:")
    accepted_points = 0
    max_candidates = max(1, int(max_candidates))

    def _build_error_map(metrics):
        if not metrics:
            return {}
        camera_ids = [str(camera_id) for camera_id in metrics.get('camera_ids', [])]
        errors = list(metrics.get('errors', []))
        return {
            camera_id: float(error)
            for camera_id, error in zip(camera_ids, errors)
            if np.isfinite(error)
        }

    for index, candidate in enumerate(candidates[:max_candidates]):
        _report_progress(
            (index / max_candidates) if max_candidates > 0 else 0.0,
            f"Возврат full-track точек: {index + 1}/{max_candidates}"
        )
        point_id = candidate['point_id']
        point_observations = {
            str(camera_id): calib_data['camera_points'][str(camera_id)][point_id]
            for camera_id in sorted(calib_data['cameras'].keys(), key=_stable_sort_key)
            if point_id in calib_data['camera_points'].get(str(camera_id), {})
        }
        if len(point_observations) < 2:
            continue

        point_observation_confidences = triangulation._extract_point_observation_confidences(
            calib_data.get('observation_confidences'),
            point_id,
            point_observations.keys(),
        )

        snapshot = _snapshot_calibration_state(calib_data)
        before_mean, _, _ = calculate_reprojection_errors(calib_data)
        before_stats = summarize_reprojection_error_distribution(calib_data, top_k=0)
        accepted, refined_point, metrics = triangulation.evaluate_multiview_point(
            candidate['best_point'],
            point_observations,
            calib_data['cameras'],
            calib_data['K'],
            calib_data.get('dist_coeffs'),
            allow_subset=False,
            min_subset_views=min(3, len(point_observations)),
            observation_confidences=point_observation_confidences,
            point_id=point_id,
            points_3d_context=calib_data.get('points_3d'),
            line_support_data=calib_data.get('line_support_data'),
        )

        if (
            candidate.get('conflict_class') == 'single_view_conflict' and
            metrics is not None and
            len(point_observations) >= 3
        ):
            current_mean = float(metrics.get('mean_error', float('inf')))
            current_max = float(metrics.get('max_error', float('inf')))
            best_repair = None
            worst_candidates = [item['camera_id'] for item in candidate.get('worst_observations', [])[:2]]
            for worst_camera_id in worst_candidates:
                subset_observations = {
                    camera_id: point_2d
                    for camera_id, point_2d in point_observations.items()
                    if str(camera_id) != str(worst_camera_id)
                }
                if len(subset_observations) < 2:
                    continue

                subset_accepted, subset_point, subset_metrics = triangulation.evaluate_multiview_point(
                    candidate['best_point'],
                    subset_observations,
                    calib_data['cameras'],
                    calib_data['K'],
                    calib_data.get('dist_coeffs'),
                    allow_subset=False,
                    min_subset_views=min(3, len(subset_observations)),
                    observation_confidences=point_observation_confidences,
                    point_id=point_id,
                    points_3d_context=calib_data.get('points_3d'),
                    line_support_data=calib_data.get('line_support_data'),
                )
                if not subset_accepted or subset_metrics is None:
                    continue

                repaired_accepted, repaired_point, repaired_metrics = triangulation.evaluate_multiview_point(
                    subset_point,
                    point_observations,
                    calib_data['cameras'],
                    calib_data['K'],
                    calib_data.get('dist_coeffs'),
                    allow_subset=False,
                    min_subset_views=min(3, len(point_observations)),
                    observation_confidences=point_observation_confidences,
                    point_id=point_id,
                    points_3d_context=calib_data.get('points_3d'),
                    line_support_data=calib_data.get('line_support_data'),
                )
                if not repaired_accepted or repaired_metrics is None:
                    continue

                repaired_mean = float(repaired_metrics.get('mean_error', float('inf')))
                repaired_max = float(repaired_metrics.get('max_error', float('inf')))
                repaired_error_map = _build_error_map(repaired_metrics)
                repaired_worst_error = repaired_error_map.get(str(worst_camera_id), float('inf'))
                repair_improves = (
                    repaired_mean <= current_mean - 0.10 or
                    repaired_max <= current_max - 0.40 or
                    (
                        repaired_mean <= current_mean * 0.92 and
                        repaired_max <= current_max * 0.90 and
                        repaired_worst_error <= max(1.25, current_max * 0.70)
                    )
                )
                if not repair_improves:
                    continue

                repair_score = (
                    (current_max - repaired_max) * 0.65 +
                    (current_mean - repaired_mean) * 0.35
                )
                if best_repair is None or repair_score > best_repair['score'] + 1e-6:
                    best_repair = {
                        'score': float(repair_score),
                        'point': np.asarray(repaired_point, dtype=np.float64).reshape(3),
                        'metrics': repaired_metrics,
                        'seed_camera': str(worst_camera_id),
                    }

            if best_repair is not None:
                accepted = True
                refined_point = best_repair['point']
                metrics = best_repair['metrics']
                candidate['repair_seed_camera'] = best_repair['seed_camera']

        if accepted:
            calib_data['points_3d'][point_id] = np.asarray(refined_point, dtype=np.float32)

        after_mean, _, _ = calculate_reprojection_errors(calib_data)
        after_stats = summarize_reprojection_error_distribution(calib_data, top_k=0)
        candidate_mean = float(metrics.get('mean_error', float('inf'))) if metrics is not None else float('inf')
        candidate_max = float(metrics.get('max_error', float('inf'))) if metrics is not None else float('inf')
        candidate_inlier_ratio = float(metrics.get('inlier_ratio', 0.0)) if metrics is not None else 0.0
        candidate_ok = (
            bool(accepted) and
            candidate_inlier_ratio >= 0.999 and
            candidate_mean <= max(float(target_p95) * 1.80, 1.50) and
            candidate_max <= max(float(target_max) * 2.50, 3.00)
        )
        combined_scene_ok = (
            after_mean <= max(before_mean + 0.15, before_mean * 1.30, float(target_mean) * 1.60) and
            after_stats['p95'] <= max(before_stats['p95'] + 0.30, before_stats['p95'] * 1.25, float(target_p95) * 1.50) and
            after_stats['max'] <= max(before_stats['max'] + 0.80, float(target_max) * 2.50)
        )

        pair_label = (
            f"{candidate['best_pair'][0]}-{candidate['best_pair'][1]}"
            if candidate.get('best_pair') is not None else "n/a"
        )
        repair_suffix = (
            f", repair_seed={candidate['repair_seed_camera']}"
            if candidate.get('repair_seed_camera') else ""
        )
        worst_summary = ", ".join(
            f"{item['camera_id']}:{item['error']:.2f}px"
            for item in candidate.get('worst_observations', [])
        ) or "n/a"
        if not candidate_ok or not combined_scene_ok:
            _restore_calibration_state(calib_data, snapshot)
            print(
                f"  - Точка {point_id}: отклонена, "
                f"pair={pair_label}, "
                f"candidate mean={candidate_mean:.2f}px, max={candidate_max:.2f}px, "
                f"class={candidate.get('conflict_class') or 'n/a'}, "
                f"inlier_ratio={candidate_inlier_ratio:.2f}, "
                f"fixed-scene mean {before_mean:.4f}px -> {after_mean:.4f}px, "
                f"p95 {before_stats['p95']:.4f}px -> {after_stats['p95']:.4f}px, "
                f"worst [{worst_summary}]{repair_suffix}"
            )
            continue

        accepted_points += 1
        print(
            f"  - Точка {point_id}: принята, "
            f"pair={pair_label}, "
            f"candidate mean={candidate_mean:.2f}px, max={candidate_max:.2f}px, "
            f"class={candidate.get('conflict_class') or 'n/a'}, "
            f"inlier_ratio={candidate_inlier_ratio:.2f}, "
            f"fixed-scene mean {before_mean:.4f}px -> {after_mean:.4f}px, "
            f"p95 {before_stats['p95']:.4f}px -> {after_stats['p95']:.4f}px, "
            f"worst [{worst_summary}]{repair_suffix}"
        )

    if accepted_points <= 0:
        print("Backfillable full-track re-entry: улучшений не найдено")
    else:
        _report_progress(1.0, "Возврат full-track точек завершен")

    return int(accepted_points)


def build_secondary_point_cloud(
    calib_data,
    target_mean=0.5,
    target_p95=1.0,
    target_max=1.0,
    max_candidates=32,
):
    if not calib_data.get('cameras') or not calib_data.get('points_3d'):
        calib_data['secondary_points_3d'] = {}
        calib_data['_unreconstructed_diagnostics_cache'] = []
        return 0

    diagnostics = triangulation.diagnose_unreconstructed_tracks(
        calib_data['points_3d'],
        calib_data['cameras'],
        calib_data['camera_points'],
        calib_data['K'],
        calib_data.get('dist_coeffs'),
        min_track_length=2,
        strict_track_consistency=bool(calib_data.get('strict_track_consistency', True)),
        top_k=0,
        debug_logging=bool(calib_data.get('debug_logging', False)),
        observation_confidences=calib_data.get('observation_confidences'),
        line_support_data=calib_data.get('line_support_data'),
    )
    calib_data['_unreconstructed_diagnostics_cache'] = copy.deepcopy(diagnostics)

    secondary_points = {}
    max_candidates = max(1, int(max_candidates))
    accepted = 0
    accepted_from_removed = 0

    def _candidate_ok(item):
        if item.get('best_point') is None:
            return False
        reason = str(item.get('reason') or '')
        conflict_class = str(item.get('conflict_class') or '')
        track_length = int(item.get('track_length') or 0)
        best_mean = float(item.get('best_mean') or float('inf'))
        best_max = float(item.get('best_max') or float('inf'))
        min_parallax = item.get('min_parallax_deg')
        min_parallax = float(min_parallax) if min_parallax is not None else 0.0

        if reason == 'backfillable_full_track':
            if track_length >= 3:
                return (
                    best_mean <= max(float(target_p95) * 4.25, 4.3) and
                    best_max <= max(float(target_max) * 6.50, 9.5) and
                    conflict_class not in {'depth_instability'}
                )
            if track_length == 2:
                return (
                    min_parallax >= 8.0 and
                    best_mean <= max(float(target_p95) * 3.25, 3.3) and
                    best_max <= max(float(target_max) * 4.25, 4.3) and
                    conflict_class in {'global_tension', 'single_view_conflict'}
                )
            return False
        if reason in {'strict_subset_only', 'subset_only'}:
            return (
                track_length >= 3 and
                best_mean <= max(float(target_p95) * 6.25, 6.5) and
                best_max <= max(float(target_max) * 8.00, 12.0) and
                min_parallax >= 1.2 and
                conflict_class in {'single_view_conflict', 'multi_view_conflict', 'global_tension'}
            )
        return False

    candidates = [item for item in diagnostics if _candidate_ok(item)]
    candidates.sort(
        key=lambda item: (
            int(item.get('track_length') or 0),
            0.0 if item.get('min_parallax_deg') is None else float(item.get('min_parallax_deg')),
            -(0.0 if item.get('best_mean') is None else float(item.get('best_mean'))),
            -(0.0 if item.get('best_max') is None else float(item.get('best_max'))),
        ),
        reverse=True,
    )

    for item in candidates[:max_candidates]:
        point_id = item['point_id']
        secondary_points[point_id] = np.asarray(item['best_point'], dtype=np.float32).reshape(3)
        accepted += 1

    seed_points = calib_data.get('_secondary_seed_points_3d', {}) or {}
    current_point_ids = set(calib_data.get('points_3d', {}).keys())
    dist_coeffs = calib_data.get('dist_coeffs')

    def _seed_point_errors(point_id, point_3d):
        errors = []
        point_3d = np.asarray(point_3d, dtype=np.float64).reshape(1, 3)
        for camera_id, pose in calib_data.get('cameras', {}).items():
            camera_key = str(camera_id)
            point_2d = calib_data.get('camera_points', {}).get(camera_key, {}).get(point_id)
            if point_2d is None:
                continue
            K_camera = calib_data.get(f'K_{camera_key}', calib_data.get('K'))
            if K_camera is None:
                continue
            R, t = pose
            try:
                rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
                projected_points, _ = cv2.projectPoints(
                    point_3d,
                    rvec,
                    np.asarray(t, dtype=np.float64).reshape(3, 1),
                    np.asarray(K_camera, dtype=np.float64),
                    None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64),
                )
            except Exception:
                continue
            projected_xy = projected_points.reshape(2)
            observed_xy = np.asarray(point_2d, dtype=np.float64).reshape(2)
            errors.append(float(np.linalg.norm(projected_xy - observed_xy)))
        return errors

    removed_seed_candidates = []
    for point_id, point_3d in seed_points.items():
        if point_id in current_point_ids or point_id in secondary_points:
            continue
        errors = _seed_point_errors(point_id, point_3d)
        if len(errors) < 3:
            continue
        mean_error = float(np.mean(errors))
        p95_error = float(np.percentile(errors, 95))
        max_error = float(np.max(errors))
        if (
            mean_error <= max(float(target_p95) * 6.50, 6.5) and
            p95_error <= max(float(target_max) * 10.50, 15.5) and
            max_error <= max(float(target_max) * 11.50, 17.0)
        ):
            removed_seed_candidates.append((point_id, np.asarray(point_3d, dtype=np.float32).reshape(3), mean_error, max_error))

    removed_seed_candidates.sort(key=lambda item: (item[2], item[3], _stable_sort_key(item[0])))
    for point_id, point_3d, _, _ in removed_seed_candidates:
        if len(secondary_points) >= max_candidates:
            break
        secondary_points[point_id] = point_3d
        accepted += 1
        accepted_from_removed += 1

    calib_data['secondary_points_3d'] = secondary_points
    if accepted > 0:
        if accepted_from_removed > 0:
            print(
                "Secondary point cloud: "
                f"добавлено {accepted} точек без влияния на калибровочный каркас "
                f"(из них {accepted_from_removed} восстановлены из pre-cleanup облака)"
            )
        else:
            print(f"Secondary point cloud: добавлено {accepted} точек без влияния на калибровочный каркас")
    else:
        print("Secondary point cloud: пригодных вторичных точек не найдено")
    return int(accepted)


def augment_secondary_point_cloud_from_seed_points(
    calib_data,
    *,
    target_p95=1.0,
    target_max=1.5,
    max_candidates=32,
):
    if not calib_data.get('cameras'):
        return 0

    seed_points = calib_data.get('_secondary_seed_points_3d', {}) or {}
    if not seed_points:
        return 0

    current_point_ids = set(calib_data.get('points_3d', {}).keys())
    secondary_points = dict(calib_data.get('secondary_points_3d', {}) or {})
    if len(secondary_points) >= max_candidates:
        return 0

    dist_coeffs = calib_data.get('dist_coeffs')

    def _seed_point_errors(point_id, point_3d):
        errors = []
        point_3d = np.asarray(point_3d, dtype=np.float64).reshape(1, 3)
        for camera_id, pose in calib_data.get('cameras', {}).items():
            camera_key = str(camera_id)
            point_2d = calib_data.get('camera_points', {}).get(camera_key, {}).get(point_id)
            if point_2d is None:
                continue
            K_camera = calib_data.get(f'K_{camera_key}', calib_data.get('K'))
            if K_camera is None:
                continue
            R, t = pose
            try:
                rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
                projected_points, _ = cv2.projectPoints(
                    point_3d,
                    rvec,
                    np.asarray(t, dtype=np.float64).reshape(3, 1),
                    np.asarray(K_camera, dtype=np.float64),
                    None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64),
                )
            except Exception:
                continue
            projected_xy = projected_points.reshape(2)
            observed_xy = np.asarray(point_2d, dtype=np.float64).reshape(2)
            errors.append(float(np.linalg.norm(projected_xy - observed_xy)))
        return errors

    extra_candidates = []
    for point_id, point_3d in seed_points.items():
        if point_id in current_point_ids or point_id in secondary_points:
            continue
        errors = _seed_point_errors(point_id, point_3d)
        if len(errors) < 2:
            continue
        mean_error = float(np.mean(errors))
        p95_error = float(np.percentile(errors, 95))
        max_error = float(np.max(errors))
        if (
            mean_error <= max(float(target_p95) * 7.50, 7.5) and
            p95_error <= max(float(target_max) * 11.50, 17.5) and
            max_error <= max(float(target_max) * 12.50, 19.0)
        ):
            extra_candidates.append(
                (point_id, np.asarray(point_3d, dtype=np.float32).reshape(3), mean_error, max_error, len(errors))
            )

    extra_candidates.sort(key=lambda item: (item[4], -item[2], -item[3], _stable_sort_key(item[0])), reverse=True)
    added = 0
    for point_id, point_3d, _, _, _ in extra_candidates:
        if len(secondary_points) >= max_candidates:
            break
        secondary_points[point_id] = point_3d
        added += 1

    if added > 0:
        calib_data['secondary_points_3d'] = secondary_points
        print(f"Secondary seed recovery: добавлено {added} seed-точек во вторичное облако")
    return int(added)


def perform_full_reconstruction(calib_data, initial_pair=None, min_points_for_camera=4, 
                            bundle_method='trf', bundle_ftol=1e-8, max_bundle_iterations=3,
                            ransac_threshold=8.0, confidence=0.99, max_attempts=3,
                            focal_range=(800, 3000), adapt_initial_focal=True,
                            check_focal_consistency=True, auto_correct_focal=False,
                            force_same_focal=False, progress_callback=None):
    """
    Выполняет полную реконструкцию от инициализации до оптимизации.
    """
    try:
        import time
        start_time = time.time()
        np.random.seed(0)
        try:
            cv2.setRNGSeed(0)
        except AttributeError:
            pass

        def _report_progress(progress_value, status_text):
            if progress_callback is None:
                return
            try:
                progress_callback(progress_value, status_text)
            except Exception:
                pass

        print("\nНачало полной реконструкции")
        calib_data['_multiview_refine_mode'] = str(calib_data.get('_multiview_refine_mode', 'full') or 'full')
        _reset_point_drift_trace(calib_data)
        _capture_point_drift_stage(
            calib_data,
            "solver_start",
            {
                "initial_pair": list(initial_pair) if initial_pair is not None else None,
                "min_points_for_camera": int(min_points_for_camera),
            },
        )

        if (
            bool(calib_data.get('two_stage_pose_scaffold_recovery', False)) and
            not bool(calib_data.get('_two_stage_pose_scaffold_recovery_active', False)) and
            not bool(calib_data.get('_project_level_preview_mode', False))
        ):
            return _run_two_stage_pose_scaffold_recovery(
                calib_data,
                start_time=start_time,
                initial_pair=initial_pair,
                min_points_for_camera=min_points_for_camera,
                bundle_method=bundle_method,
                bundle_ftol=bundle_ftol,
                max_bundle_iterations=max_bundle_iterations,
                ransac_threshold=ransac_threshold,
                confidence=confidence,
                max_attempts=max_attempts,
                focal_range=focal_range,
                adapt_initial_focal=adapt_initial_focal,
                check_focal_consistency=check_focal_consistency,
                auto_correct_focal=auto_correct_focal,
                force_same_focal=force_same_focal,
                progress_callback=progress_callback,
            )

        if adapt_initial_focal and calib_data.get('K') is not None and focal_range is not None:
            initial_focal_estimate = calib_data.get('initial_focal_estimate')
            skip_focal_sweep = False
            if isinstance(initial_focal_estimate, dict):
                estimate_confidence = float(initial_focal_estimate.get('confidence', 1.0))
                image_focal_px = initial_focal_estimate.get('focal_px')
                fallback_focal_px = initial_focal_estimate.get('fallback_focal_px')
                min_confidence_for_full_sweep = float(
                    calib_data.get('min_focal_estimate_confidence_for_full_sweep', 0.35)
                )
                if estimate_confidence < min_confidence_for_full_sweep:
                    print(
                        "Project-level focal sweep: "
                        f"image-based оценка слабая (confidence={estimate_confidence:.2f} < {min_confidence_for_full_sweep:.2f}); "
                        "сравниваем гипотезы по всему фотосету"
                    )
                    quick_image_validation = _quick_validate_low_confidence_image_focal(
                        calib_data,
                        image_focal_px=image_focal_px,
                        min_points_for_camera=min_points_for_camera,
                        ransac_threshold=ransac_threshold,
                        confidence=confidence,
                        max_attempts=max_attempts,
                        focal_range=focal_range,
                        force_same_focal=force_same_focal,
                        fixed_pair=initial_pair,
                    )
                    if quick_image_validation is None or not bool(quick_image_validation.get('plausible', False)):
                        if fallback_focal_px is not None:
                            fallback_focal_px = round(float(fallback_focal_px), 2)
                            current_K = np.asarray(calib_data['K'], dtype=np.float64)
                            fallback_scale = float(fallback_focal_px) / max(float(current_K[0, 0]), 1e-6)
                            fallback_K = np.array(current_K, copy=True)
                            fallback_K[0, 0] = current_K[0, 0] * fallback_scale
                            fallback_K[1, 1] = current_K[1, 1] * fallback_scale
                            calib_data['K'] = fallback_K.astype(np.float32)
                            calib_data['project_level_focal_sweep'] = {
                                'mode': 'fallback_after_quick_reject',
                                'focal_px': float(fallback_focal_px),
                                'confidence': float(estimate_confidence),
                                'rejected_image_focal_px': float(image_focal_px) if image_focal_px is not None else None,
                                'rejected_mean_error': (
                                    float(quick_image_validation['mean_error'])
                                    if isinstance(quick_image_validation, dict) and quick_image_validation.get('mean_error') is not None
                                    else None
                                ),
                                'rejected_p95': (
                                    float(quick_image_validation['p95'])
                                    if isinstance(quick_image_validation, dict) and quick_image_validation.get('p95') is not None
                                    else None
                                ),
                            }
                            skip_focal_sweep = True
                            print(
                                "Project-level focal sweep: "
                                f"weak image-based гипотеза fx~{float(image_focal_px):.0f} отвергнута быстрым project-level check, "
                                f"сохраняем нейтральный prior fx~{float(fallback_focal_px):.0f}"
                            )

            if not skip_focal_sweep:
                _report_progress(67.0, "Project-level focal sweep...")
                focal_sweep_result = _select_project_level_focal_hypothesis(
                    calib_data,
                    focal_range=focal_range,
                    min_points_for_camera=min_points_for_camera,
                    ransac_threshold=ransac_threshold,
                    confidence=confidence,
                    max_attempts=max_attempts,
                    force_same_focal=force_same_focal,
                    fixed_pair=initial_pair,
                )
                if focal_sweep_result is not None:
                    calib_data['K'] = np.asarray(focal_sweep_result['K'], dtype=np.float32)
                    calib_data['project_level_focal_sweep'] = {
                        'mode': str(focal_sweep_result.get('selection_mode', 'project_level_preview')),
                        'focal_px': float(focal_sweep_result['focal_px']),
                        'pair': tuple(focal_sweep_result['pair']),
                        'cameras': int(focal_sweep_result['cameras']),
                        'points': int(focal_sweep_result['points']),
                        'mean_error': float(focal_sweep_result['mean_error']),
                        'p95': float(focal_sweep_result['p95']),
                        'quality_score': float(focal_sweep_result['quality_score']),
                    }
                    if initial_pair is None:
                        initial_pair = focal_sweep_result['pair']
                    cached_full_preview = focal_sweep_result.get('final_calib_data')
                    if isinstance(cached_full_preview, dict):
                        restored_state = copy.deepcopy(cached_full_preview)
                        restored_state['project_level_focal_sweep'] = copy.deepcopy(calib_data['project_level_focal_sweep'])
                        restored_state['_project_level_preview_mode'] = False
                        restored_state['_multiview_refine_mode'] = 'full'
                        calib_data.clear()
                        calib_data.update(restored_state)
                        _report_progress(100.0, "Калибровка завершена (reuse full preview)")
                        print(
                            "Project-level focal sweep: "
                            f"используем готовый результат полного preview для fx~{focal_sweep_result['focal_px']:.0f}"
                        )
                        return True
                    cached_preview = focal_sweep_result.get('preview_calib_data')
                    if isinstance(cached_preview, dict):
                        restored_state = copy.deepcopy(cached_preview)
                        restored_state['project_level_focal_sweep'] = copy.deepcopy(calib_data['project_level_focal_sweep'])
                        restored_state['_project_level_preview_mode'] = False
                        restored_state['_multiview_refine_mode'] = 'full'
                        calib_data.clear()
                        calib_data.update(restored_state)
                        preview_initialized_cameras = set(str(camera_id) for camera_id in calib_data.get('cameras', {}).keys())
                        print(
                            "Project-level focal sweep: "
                            f"используем быстрый preview как старт для финальной оптимизации "
                            f"(fx~{focal_sweep_result['focal_px']:.0f})"
                        )
                        if len(preview_initialized_cameras) < len(calib_data.get('camera_points', {})):
                            print("Project-level focal sweep: продолжаем добор камер после preview...")
                            _expand_reconstruction_frontier(
                                calib_data,
                                initialized_cameras=preview_initialized_cameras,
                                min_points_for_camera=min_points_for_camera,
                                ransac_threshold=ransac_threshold,
                                confidence=confidence,
                                max_attempts=max_attempts,
                                progress_callback=progress_callback,
                                progress_range=(72.0, 82.0),
                            )
                        return _finalize_reconstruction(
                            calib_data,
                            start_time=start_time,
                            min_points_for_camera=min_points_for_camera,
                            max_bundle_iterations=max_bundle_iterations,
                            focal_range=focal_range,
                            force_same_focal=force_same_focal,
                            ransac_threshold=ransac_threshold,
                            confidence=confidence,
                            max_attempts=max_attempts,
                            progress_callback=progress_callback,
                        )
        
        # Находим начальную пару камер
        _report_progress(68.0, "Поиск стартовой пары...")
        if not initial_pair:
            print("Поиск лучшей начальной пары камер...")
            initial_pair = _select_initial_pair_for_reconstruction(
                calib_data,
                min_points_for_camera=min_points_for_camera,
                ransac_threshold=ransac_threshold,
                confidence=confidence,
                max_attempts=max_attempts,
            )
            if not initial_pair:
                print("Не удалось найти подходящую начальную пару камер")
                return False
        
        camera_id1, camera_id2 = initial_pair
        print(f"Начальная пара камер: {camera_id1}-{camera_id2}")
        
        # Инициализируем реконструкцию с начальной парой
        _report_progress(70.0, f"Инициализация стартовой пары {camera_id1}-{camera_id2}...")
        print(f"Инициализация начальной пары камер: {camera_id1}-{camera_id2}")
        if not initialize_reconstruction(calib_data, camera_id1, camera_id2):
            print("Не удалось инициализировать реконструкцию с начальной парой камер")
            return False
            
        # Список камер в реконструкции
        initialized_cameras = set([str(camera_id1), str(camera_id2)])

        initialized_cameras = _expand_reconstruction_frontier(
            calib_data,
            initialized_cameras=initialized_cameras,
            min_points_for_camera=min_points_for_camera,
            ransac_threshold=ransac_threshold,
            confidence=confidence,
            max_attempts=max_attempts,
            progress_callback=progress_callback,
            progress_range=(72.0, 82.0),
        )
        _capture_point_drift_stage(
            calib_data,
            "after_frontier_expansion",
            {
                "camera_count": int(len(initialized_cameras)),
            },
        )

        return _finalize_reconstruction(
            calib_data,
            start_time=start_time,
            min_points_for_camera=min_points_for_camera,
            max_bundle_iterations=max_bundle_iterations,
            focal_range=focal_range,
            force_same_focal=force_same_focal,
            ransac_threshold=ransac_threshold,
            confidence=confidence,
            max_attempts=max_attempts,
            progress_callback=progress_callback,
        )
        
    except Exception as e:
        print(f"Ошибка при реконструкции: {str(e)}")
        traceback.print_exc()
        return False
