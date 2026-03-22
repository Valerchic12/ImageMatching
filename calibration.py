"""
Основной модуль калибровки камер для Blender аддона.
Использует модули из calibration_modules для выполнения всех операций.
"""
import os
import sys
import time
import traceback
import json
import logging
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Union, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from centralized import system
from core_imports import (
    bpy,
    BLENDER_AVAILABLE,
    initialize,
    get_numpy,
    get_cv2,
    check_core_dependencies,
    get_calibration_modules,
    get_calibration_utils,
    get_main_utils
)

# Initialize import system
if not initialize():
    logger.warning("Import system initialization failed. Some features may not work.")

# Import dependencies
np = get_numpy()
cv2 = get_cv2()

# Check if core dependencies are available
DEPENDENCIES_INSTALLED = check_core_dependencies()

if DEPENDENCIES_INSTALLED:
    # Import calibration modules
    calibration_core, camera_pose, triangulation, bundle_adjustment = get_calibration_modules()
    
    if None in (calibration_core, camera_pose, triangulation, bundle_adjustment):
        logger.error("Some calibration modules failed to import")
        DEPENDENCIES_INSTALLED = False
    else:
        logger.info("All calibration modules imported successfully")
        
        # Import utility modules
        calibration_utils = get_calibration_utils()
        utils = get_main_utils()
        
        if calibration_utils is None:
            logger.warning("calibration_modules.utils import failed")
        if utils is None:
            logger.warning("main utils import failed")
else:
    logger.error("Core dependencies (numpy, cv2) not available")
    calibration_core = None
    camera_pose = None
    triangulation = None
    bundle_adjustment = None
    calibration_utils = None
    utils = None

# Глобальные данные калибровки
calibration_data = None


def compute_default_focal_prior_px(image_width, image_height):
    """
    Нейтральный bootstrap-prior для fx/fy в пикселях.

    Важно: это не "истинный" фокус камеры, а только устойчивая стартовая
    гипотеза в пиксельном пространстве. Она не зависит от sensor_width/mm.
    """
    width = max(float(image_width or 0), 1.0)
    height = max(float(image_height or 0), 1.0)
    diagonal = float(np.hypot(width, height))

    # Умеренный prior по диагонали кадра. Для 1600x1200 дает ~2222px,
    # что уже показало устойчивую геометрию на текущем solver path.
    return float(round(diagonal * 1.1111111111111112, 2))


def estimate_initial_focal_length(
    image_path,
    image_width,
    image_height,
    sensor_width_mm=36.0,
    provided_focal_mm=None,
):
    """
    Оценивает стартовое фокусное расстояние в пикселях.

    EXIF используется только если доступен. Основной упор делается на
    image-based методы и pixel-domain prior. sensor_width_mm/provided_focal_mm
    сохранены в сигнатуре только для обратной совместимости и больше не
    участвуют в solver path.
    """
    width = max(float(image_width or 0), 1.0)
    height = max(float(image_height or 0), 1.0)
    neutral_prior_focal_px = compute_default_focal_prior_px(width, height)

    if not DEPENDENCIES_INSTALLED or triangulation is None:
        return neutral_prior_focal_px, neutral_prior_focal_px, 0.0

    _ = sensor_width_mm
    _ = provided_focal_mm
    prior_focal_px = neutral_prior_focal_px

    cx = width * 0.5
    cy = height * 0.5
    K_initial = np.array([
        [prior_focal_px, 0.0, cx],
        [0.0, prior_focal_px, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    def _weighted_median(values, weights):
        values = np.asarray(values, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if values.size == 0:
            return None
        order = np.argsort(values)
        sorted_values = values[order]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        cutoff = sorted_weights.sum() * 0.5
        idx = int(np.searchsorted(cumulative, cutoff, side='left'))
        idx = min(max(idx, 0), sorted_values.size - 1)
        return float(sorted_values[idx])

    estimates = []

    def _append_estimate(fx, fy, confidence, method):
        if fx is None or fy is None:
            return
        fx = float(fx)
        fy = float(fy)
        if not np.isfinite(fx) or not np.isfinite(fy):
            return
        estimates.append({
            'fx': fx,
            'fy': fy,
            'confidence': float(confidence),
            'method': method,
        })

    if image_path and os.path.exists(image_path):
        try:
            exif_focal = triangulation.compute_focal_length_from_exif(image_path, width, height)
            if exif_focal is not None and np.isfinite(exif_focal):
                _append_estimate(exif_focal, exif_focal, 0.95, "exif")
        except Exception as e:
            logger.warning(f"EXIF focal estimate failed: {e}")

        try:
            vp_fx, vp_fy = triangulation.estimate_focal_length_from_vanishing_points_advanced(
                image_path,
                K_initial
            )
            _append_estimate(vp_fx, vp_fy, 0.55, "vanishing_points")
        except Exception as e:
            logger.warning(f"Vanishing-point focal estimate failed: {e}")

        try:
            geo_fx, geo_fy = triangulation.estimate_focal_length_from_geometric_constraints(
                image_path,
                K_initial
            )
            _append_estimate(geo_fx, geo_fy, 0.25, "geometric_constraints")
        except Exception as e:
            logger.warning(f"Geometric focal estimate failed: {e}")

    image_based_estimates = [item for item in estimates if item['method'] != 'prior']

    # Слабый prior держим только как страховку, если image-based методы молчат.
    prior_confidence = 0.08 if image_based_estimates else 0.20
    _append_estimate(prior_focal_px, prior_focal_px, prior_confidence, "prior")

    fx_values = np.asarray([item['fx'] for item in estimates], dtype=np.float64)
    fy_values = np.asarray([item['fy'] for item in estimates], dtype=np.float64)
    weights = np.asarray([item['confidence'] for item in estimates], dtype=np.float64)

    if image_based_estimates:
        image_fx = np.asarray([item['fx'] for item in image_based_estimates], dtype=np.float64)
        image_fy = np.asarray([item['fy'] for item in image_based_estimates], dtype=np.float64)
        image_weights = np.asarray([item['confidence'] for item in image_based_estimates], dtype=np.float64)

        fx_weighted = float(np.average(image_fx, weights=image_weights))
        fy_weighted = float(np.average(image_fy, weights=image_weights))
        fx_median = _weighted_median(image_fx, image_weights)
        fy_median = _weighted_median(image_fy, image_weights)

        fx_spread = float(np.std(image_fx)) / max(abs(fx_weighted), 1e-6)
        fy_spread = float(np.std(image_fy)) / max(abs(fy_weighted), 1e-6)
        spread = max(fx_spread, fy_spread)

        if spread > 0.18 and fx_median is not None and fy_median is not None:
            final_fx = fx_median
            final_fy = fy_median
        else:
            final_fx = fx_weighted
            final_fy = fy_weighted

        base_confidence = min(0.95, float(np.mean(image_weights)))
        consistency_bonus = max(0.0, 1.0 - min(spread / 0.35, 1.0))
        confidence = min(0.95, 0.55 * base_confidence + 0.40 * consistency_bonus)

        # Один слабый image-based метод не должен полностью перебивать prior.
        if len(image_based_estimates) == 1 and image_based_estimates[0]['method'] != 'exif':
            only_method = image_based_estimates[0]['method']
            if only_method == 'geometric_constraints':
                image_weight = 0.28
                confidence = min(confidence, 0.28)
            else:
                image_weight = 0.45
                confidence = min(confidence, 0.42)
            prior_weight = 1.0 - image_weight
            final_fx = prior_focal_px * prior_weight + final_fx * image_weight
            final_fy = prior_focal_px * prior_weight + final_fy * image_weight
    else:
        final_fx = float(np.average(fx_values, weights=weights))
        final_fy = float(np.average(fy_values, weights=weights))
        confidence = 0.0

    min_focal = max(min(width, height) * 0.45, 100.0)
    max_focal = max(width, height) * 4.0
    final_fx = float(np.clip(final_fx, min_focal, max_focal))
    final_fy = float(np.clip(final_fy, min_focal, max_focal))

    logger.info(
        "Initial focal estimate: "
        f"fx={final_fx:.2f}, fy={final_fy:.2f}, confidence={confidence:.2f}, "
        f"methods={[item['method'] for item in estimates if item['method'] != 'prior'] or ['fallback_prior']}"
    )
    return final_fx, final_fy, float(confidence)


def resolve_bootstrap_focal_estimate(initial_focal_estimate, min_confidence_for_direct_use=0.35):
    """
    Выбирает стартовое фокусное расстояние для bootstrap.

    Низкодостоверная image-based оценка не должна сразу менять стартовую K.
    В таком случае используем fallback_focal_px, а саму оценку оставляем только
    как метаданные для последующей логики внутри core.
    """
    if not isinstance(initial_focal_estimate, dict):
        return None

    focal_px = initial_focal_estimate.get('focal_px')
    fallback_focal_px = initial_focal_estimate.get('fallback_focal_px')
    confidence = float(initial_focal_estimate.get('confidence', 0.0))
    threshold = float(min_confidence_for_direct_use)

    chosen_focal_px = None
    source = 'missing'

    if focal_px is not None and confidence >= threshold:
        chosen_focal_px = float(focal_px)
        source = 'image_based'
    elif fallback_focal_px is not None:
        chosen_focal_px = float(fallback_focal_px)
        source = 'fallback_low_confidence'
    elif focal_px is not None:
        chosen_focal_px = float(focal_px)
        source = 'image_based_untrusted'

    if chosen_focal_px is None:
        return None

    normalized_focal_px = round(float(chosen_focal_px), 2)

    return {
        'focal_px': float(normalized_focal_px),
        'confidence': confidence,
        'source': source,
        'threshold': threshold,
    }


def _estimate_image_observation_confidences(image_path, points_data, patch_radius=6):
    if not points_data:
        return {}

    default_confidences = {point_id: 1.0 for point_id in points_data.keys()}
    if not image_path or not os.path.exists(image_path) or cv2 is None:
        return default_confidences

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return default_confidences

    gray = image.astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    height, width = gray.shape[:2]

    raw_entries = []
    for point_id, point in points_data.items():
        point_xy = np.asarray(point, dtype=np.float32).reshape(2)
        x = int(round(float(point_xy[0])))
        y = int(round(float(point_xy[1])))
        x0 = max(0, x - patch_radius)
        x1 = min(width, x + patch_radius + 1)
        y0 = max(0, y - patch_radius)
        y1 = min(height, y + patch_radius + 1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            raw_entries.append((point_id, 0.55, 0.0, 0.0))
            continue

        gx_patch = grad_x[y0:y1, x0:x1]
        gy_patch = grad_y[y0:y1, x0:x1]
        sxx = float(np.sum(gx_patch * gx_patch))
        syy = float(np.sum(gy_patch * gy_patch))
        sxy = float(np.sum(gx_patch * gy_patch))
        tensor = np.array([[sxx, sxy], [sxy, syy]], dtype=np.float64)

        try:
            eigenvalues = np.linalg.eigvalsh(tensor)
            lambda_1 = float(max(eigenvalues))
            lambda_2 = float(min(eigenvalues))
        except np.linalg.LinAlgError:
            lambda_1 = 0.0
            lambda_2 = 0.0

        anisotropy = float(np.clip(lambda_2 / max(lambda_1, 1e-8), 0.0, 1.0))
        strength = float(np.sqrt(max(lambda_1 + lambda_2, 0.0)))
        harris = float(max((lambda_1 * lambda_2) - 0.04 * (lambda_1 + lambda_2) ** 2, 0.0))
        raw_entries.append((point_id, anisotropy, strength, harris))

    strengths = np.asarray([entry[2] for entry in raw_entries], dtype=np.float64)
    harrises = np.asarray([entry[3] for entry in raw_entries], dtype=np.float64)
    strength_hi = float(np.percentile(strengths, 80)) if strengths.size else 1.0
    harris_hi = float(np.percentile(harrises, 80)) if harrises.size else 1.0
    strength_hi = max(strength_hi, 1e-6)
    harris_hi = max(harris_hi, 1e-6)

    confidences = {}
    for point_id, anisotropy, strength, harris in raw_entries:
        strength_score = float(np.clip(strength / strength_hi, 0.0, 1.0))
        corner_score = float(np.clip(harris / harris_hi, 0.0, 1.0))
        confidence = (
            0.30 +
            0.45 * anisotropy +
            0.15 * strength_score +
            0.10 * corner_score
        )
        confidences[point_id] = float(np.clip(confidence, 0.25, 1.0))

    return confidences


def _ensure_raw_camera_points(calib_data):
    if calib_data is None:
        return {}

    camera_points = calib_data.get('camera_points', {})
    raw_camera_points = calib_data.get('raw_camera_points')
    if not isinstance(raw_camera_points, dict):
        raw_camera_points = {}
        calib_data['raw_camera_points'] = raw_camera_points

    for camera_id, points_data in camera_points.items():
        camera_id = str(camera_id)
        current_ids = set(points_data.keys())
        raw_points = raw_camera_points.get(camera_id)
        raw_ids = set(raw_points.keys()) if isinstance(raw_points, dict) else set()
        if raw_points is None or current_ids != raw_ids:
            raw_camera_points[camera_id] = {
                point_id: np.asarray(point, dtype=np.float32).reshape(2).copy()
                for point_id, point in points_data.items()
            }

    for camera_id in list(raw_camera_points.keys()):
        if camera_id not in camera_points:
            del raw_camera_points[camera_id]

    return raw_camera_points


def _sample_bilinear(image, x, y):
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


def _refine_image_points_subpixel(
    image_path,
    points_data,
    patch_radius=8,
    max_shift_px=0.85,
    corner_win_radius=5,
):
    refined_points = {
        point_id: np.asarray(point, dtype=np.float32).reshape(2).copy()
        for point_id, point in points_data.items()
    }
    stats = {
        'point_count': len(points_data),
        'refined_count': 0,
        'median_shift_px': 0.0,
        'max_shift_px': 0.0,
    }
    if not points_data or not image_path or not os.path.exists(image_path) or cv2 is None:
        return refined_points, stats

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return refined_points, stats

    gray_u8 = np.ascontiguousarray(image)
    gray_f32 = gray_u8.astype(np.float32)
    eigen_map = cv2.cornerMinEigenVal(gray_u8, blockSize=3, ksize=3)
    height, width = gray_u8.shape[:2]
    termination = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        0.01,
    )
    accepted_shifts = []

    for point_id, point in points_data.items():
        original = np.asarray(point, dtype=np.float32).reshape(2)
        x = float(original[0])
        y = float(original[1])
        if (
            x < patch_radius + 1 or
            y < patch_radius + 1 or
            x >= width - patch_radius - 1 or
            y >= height - patch_radius - 1
        ):
            continue

        x0 = max(0, int(np.floor(x)) - patch_radius)
        x1 = min(width, int(np.floor(x)) + patch_radius + 1)
        y0 = max(0, int(np.floor(y)) - patch_radius)
        y1 = min(height, int(np.floor(y)) + patch_radius + 1)
        patch = gray_u8[y0:y1, x0:x1]
        if patch.size == 0 or patch.shape[0] < 5 or patch.shape[1] < 5:
            continue
        if float(np.std(patch.astype(np.float32))) < 2.0:
            continue

        local_eigen_patch = eigen_map[y0:y1, x0:x1]
        local_peak = float(np.max(local_eigen_patch)) if local_eigen_patch.size else 0.0
        if local_peak <= 1e-8:
            continue

        original_response = _sample_bilinear(eigen_map, x, y)
        original_response_score = float(np.clip(original_response / max(local_peak, 1e-8), 0.0, 1.0))
        if original_response_score < 0.10:
            continue

        refined = np.asarray(original, dtype=np.float32).reshape(1, 1, 2)
        try:
            cv2.cornerSubPix(
                gray_f32,
                refined,
                (int(corner_win_radius), int(corner_win_radius)),
                (-1, -1),
                termination,
            )
        except cv2.error:
            continue

        refined_xy = refined.reshape(2)
        total_shift = float(np.linalg.norm(refined_xy - original))
        if total_shift < 0.01 or total_shift > float(max_shift_px):
            continue

        response = _sample_bilinear(eigen_map, refined_xy[0], refined_xy[1])
        response_score = float(np.clip(response / max(local_peak, 1e-8), 0.0, 1.0))
        if response_score < 0.15:
            continue
        if response_score + 0.10 < original_response_score:
            continue

        refined_points[point_id] = refined_xy.astype(np.float32)
        accepted_shifts.append(total_shift)

    if accepted_shifts:
        shift_values = np.asarray(accepted_shifts, dtype=np.float64)
        stats['refined_count'] = int(len(accepted_shifts))
        stats['median_shift_px'] = float(np.median(shift_values))
        stats['max_shift_px'] = float(np.max(shift_values))

    return refined_points, stats


def _resolve_calibration_image_path(calib_data, camera_id, image_path):
    if not image_path:
        return None

    image_path = str(image_path)
    if os.path.exists(image_path):
        return os.path.abspath(image_path)

    search_roots = []
    if calib_data:
        for key in ('image_root', 'images_root'):
            root = calib_data.get(key)
            if root:
                search_roots.append(str(root))

    for root in search_roots:
        candidate = os.path.join(root, image_path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    return None


def _stable_sort_key(value):
    text = str(value)
    try:
        return (0, int(text))
    except (TypeError, ValueError):
        return (1, text)


def _distance_point_to_line_2d(point, line_start, line_end):
    point = np.asarray(point, dtype=np.float64).reshape(2)
    line_start = np.asarray(line_start, dtype=np.float64).reshape(2)
    line_end = np.asarray(line_end, dtype=np.float64).reshape(2)
    direction = line_end - line_start
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-8:
        return float('inf')
    point_offset = point - line_start
    cross_value = direction[0] * point_offset[1] - direction[1] * point_offset[0]
    return float(abs(cross_value) / direction_norm)


def _build_camera_line_triplets(points_data):
    point_ids = sorted(points_data.keys(), key=_stable_sort_key)
    if len(point_ids) < 3:
        return []

    triplets = []
    for point_id_a, point_id_b, point_id_c in combinations(point_ids, 3):
        point_a = np.asarray(points_data[point_id_a], dtype=np.float64).reshape(2)
        point_b = np.asarray(points_data[point_id_b], dtype=np.float64).reshape(2)
        point_c = np.asarray(points_data[point_id_c], dtype=np.float64).reshape(2)

        pair_candidates = [
            ((point_id_a, point_id_b), point_a, point_b, point_c),
            ((point_id_a, point_id_c), point_a, point_c, point_b),
            ((point_id_b, point_id_c), point_b, point_c, point_a),
        ]
        endpoints = None
        max_span = 0.0
        max_distance = None
        for _, line_start, line_end, other_point in pair_candidates:
            span = float(np.linalg.norm(line_end - line_start))
            if span <= max_span:
                continue
            endpoints = (line_start, line_end, other_point)
            max_span = span

        if endpoints is None or max_span < 48.0:
            continue

        line_start, line_end, _ = endpoints
        distances = [
            _distance_point_to_line_2d(point_a, line_start, line_end),
            _distance_point_to_line_2d(point_b, line_start, line_end),
            _distance_point_to_line_2d(point_c, line_start, line_end),
        ]
        max_distance = float(max(distances))
        tolerance_px = max(1.35, max_span * 0.0038)
        if max_distance > tolerance_px:
            continue

        support_score = float(np.clip(1.0 - max_distance / max(tolerance_px, 1e-6), 0.0, 1.0))
        triplets.append({
            'triplet': tuple(sorted((str(point_id_a), str(point_id_b), str(point_id_c)), key=_stable_sort_key)),
            'max_distance_px': max_distance,
            'span_px': max_span,
            'support_score': support_score,
        })

    return triplets


def _build_line_support_data(camera_points):
    line_support_data = {
        'triplets': {},
        'point_triplets': {},
        'observation_supports': {},
    }
    if not camera_points:
        return line_support_data

    triplet_cameras = {}
    for camera_id in sorted(camera_points.keys(), key=_stable_sort_key):
        points_data = camera_points.get(camera_id, {})
        triplets = _build_camera_line_triplets(points_data)
        for triplet_data in triplets:
            triplet = triplet_data['triplet']
            entry = triplet_cameras.setdefault(triplet, [])
            entry.append({
                'camera_id': str(camera_id),
                'max_distance_px': float(triplet_data['max_distance_px']),
                'span_px': float(triplet_data['span_px']),
                'support_score': float(triplet_data['support_score']),
            })

    stable_triplets = {
        triplet: entries
        for triplet, entries in triplet_cameras.items()
        if (
            len(entries) >= 2 and
            float(np.mean([item['support_score'] for item in entries])) >= 0.72 and
            float(np.mean([item['span_px'] for item in entries])) >= 80.0
        )
    }
    if not stable_triplets:
        return line_support_data

    for triplet, entries in stable_triplets.items():
        support_scores = np.asarray([item['support_score'] for item in entries], dtype=np.float64)
        span_values = np.asarray([item['span_px'] for item in entries], dtype=np.float64)
        line_support_data['triplets'][triplet] = {
            'support_count': int(len(entries)),
            'camera_ids': [item['camera_id'] for item in entries],
            'mean_support': float(np.mean(support_scores)) if support_scores.size else 0.0,
            'mean_span_px': float(np.mean(span_values)) if span_values.size else 0.0,
        }
        for point_id in triplet:
            line_support_data['point_triplets'].setdefault(str(point_id), []).append(triplet)

    observation_supports = {}
    for camera_id in sorted(camera_points.keys(), key=_stable_sort_key):
        points_data = camera_points.get(camera_id, {})
        points_by_text = {
            str(point_id): np.asarray(point, dtype=np.float64).reshape(2)
            for point_id, point in points_data.items()
        }
        camera_support = {}
        for triplet in stable_triplets.keys():
            if not all(point_id in points_by_text for point_id in triplet):
                continue
            triplet_points = {
                point_id: points_by_text[point_id]
                for point_id in triplet
            }
            for point_id in triplet:
                other_ids = [other_id for other_id in triplet if other_id != point_id]
                line_start = triplet_points[other_ids[0]]
                line_end = triplet_points[other_ids[1]]
                distance_px = _distance_point_to_line_2d(triplet_points[point_id], line_start, line_end)
                span_px = float(np.linalg.norm(line_end - line_start))
                tolerance_px = max(1.20, span_px * 0.0032)
                support = float(np.clip(1.0 - distance_px / max(tolerance_px, 1e-6), 0.0, 1.0))
                camera_support[point_id] = max(float(camera_support.get(point_id, 0.0)), support)
        observation_supports[str(camera_id)] = camera_support

    line_support_data['observation_supports'] = observation_supports
    return line_support_data


def populate_observation_confidences(calib_data):
    if calib_data is None:
        return {}

    camera_points = calib_data.get('camera_points', {})
    image_map = calib_data.get('images', {})
    if not camera_points:
        calib_data['observation_confidences'] = {}
        return {}

    _ensure_raw_camera_points(calib_data)
    subpixel_enabled = bool(calib_data.get('subpixel_refinement_enabled', True))

    observation_confidences = {}
    subpixel_stats = {}
    summary = []
    subpixel_summary = []
    for camera_id in sorted(camera_points.keys(), key=lambda value: (0, int(str(value))) if str(value).isdigit() else (1, str(value))):
        raw_points_data = calib_data.get('raw_camera_points', {}).get(str(camera_id), camera_points.get(str(camera_id), {}))
        image_path = _resolve_calibration_image_path(calib_data, camera_id, image_map.get(str(camera_id)))
        if image_path:
            calib_data['images'][str(camera_id)] = image_path
        if subpixel_enabled:
            refined_points, camera_refine_stats = _refine_image_points_subpixel(image_path, raw_points_data)
        else:
            refined_points = {
                point_id: np.asarray(point, dtype=np.float32).reshape(2).copy()
                for point_id, point in raw_points_data.items()
            }
            camera_refine_stats = {
                'point_count': len(refined_points),
                'refined_count': 0,
                'median_shift_px': 0.0,
                'max_shift_px': 0.0,
            }
        calib_data['camera_points'][str(camera_id)] = {
            point_id: np.asarray(point, dtype=np.float32).reshape(2)
            for point_id, point in refined_points.items()
        }
        subpixel_stats[str(camera_id)] = camera_refine_stats
        if camera_refine_stats.get('refined_count', 0) > 0:
            subpixel_summary.append(
                f"{camera_id}: refined={camera_refine_stats['refined_count']}/{camera_refine_stats['point_count']}, "
                f"median_shift={camera_refine_stats['median_shift_px']:.2f}px, "
                f"max_shift={camera_refine_stats['max_shift_px']:.2f}px"
            )
        camera_confidences = _estimate_image_observation_confidences(image_path, calib_data['camera_points'][str(camera_id)])
        observation_confidences[str(camera_id)] = camera_confidences
        if camera_confidences:
            values = np.asarray(list(camera_confidences.values()), dtype=np.float64)
            summary.append(
                f"{camera_id}: median={float(np.median(values)):.2f}, "
                f"min={float(np.min(values)):.2f}, max={float(np.max(values)):.2f}, n={len(values)}"
            )

    line_support_data = _build_line_support_data(calib_data.get('camera_points', {}))
    line_triplet_count = len(line_support_data.get('triplets', {}))
    line_supported_observations = 0
    if line_triplet_count > 0:
        for camera_id, camera_support in line_support_data.get('observation_supports', {}).items():
            if camera_id not in observation_confidences:
                continue
            for point_id, support in camera_support.items():
                camera_confidence_map = observation_confidences[camera_id]
                if point_id in camera_confidence_map:
                    confidence_key = point_id
                else:
                    confidence_key = next(
                        (existing_key for existing_key in camera_confidence_map.keys() if str(existing_key) == str(point_id)),
                        point_id,
                    )
                base_confidence = float(camera_confidence_map.get(confidence_key, 1.0))
                support = float(np.clip(support, 0.0, 1.0))
                support_gain = 1.0 + 0.16 * max(support - 0.60, 0.0) / 0.40
                observation_confidences[camera_id][confidence_key] = float(np.clip(base_confidence * support_gain, 0.15, 1.0))
                line_supported_observations += 1

    calib_data['observation_confidences'] = observation_confidences
    calib_data['subpixel_refinement_stats'] = subpixel_stats
    calib_data['line_support_data'] = line_support_data
    if subpixel_enabled and subpixel_summary:
        logger.info("Subpixel refinement: " + " | ".join(subpixel_summary))
    elif not subpixel_enabled:
        logger.info("Subpixel refinement: disabled")
    if summary:
        logger.info("Observation confidence profiles: " + " | ".join(summary))
    if line_triplet_count > 0:
        logger.info(
            f"Line support: stable_triplets={line_triplet_count}, "
            f"supported_observations={line_supported_observations}"
        )
    else:
        logger.info("Line support: no stable triplets detected")
    return observation_confidences

def init_calibration():
    """
    Инициализирует данные калибровки.
    
    Returns:
        dict: Данные калибровки
    """
    if not DEPENDENCIES_INSTALLED:
        logger.error("Невозможно инициализировать калибровку: отсутствуют зависимости")
        return None
        
    global calibration_data
    try:
        # Используем правильный путь к функции инициализации
        if 'calibration_core' in globals() and hasattr(calibration_core, 'initialize_calibration'):
            calibration_data = calibration_core.initialize_calibration()
        else:
            # Если модуль calibration_core не доступен, создаем базовую структуру данных
            calibration_data = {
                'cameras': {},
                'points_3d': {},
                'camera_points': {},
                'raw_camera_points': {},
                'observation_confidences': {},
                'line_support_data': {},
                'subpixel_refinement_enabled': True,
                'subpixel_refinement_stats': {},
                'K': np.array([[1000, 0, 800], [0, 1000, 600], [0, 0, 1]], dtype=np.float32),  # Стандартная матрица калибровки
                'dist_coeffs': np.zeros(5, dtype=np.float32),
                'images': {},
                'reconstruction_ids': [],
                'debug_logging': False,
                'precision_cleanup_enabled': True,
                'precision_target_mean_px': 0.5,
                'precision_target_p95_px': 1.0,
                'precision_target_max_px': 1.0,
                'precision_cleanup_rounds': 4,
            }
        logger.info("Калибровка инициализирована")
        return calibration_data
    except Exception as e:
        logger.error(f"Ошибка при инициализации калибровки: {str(e)}")
        import traceback
        traceback.print_exc()
        # Возвращаем базовую структуру данных даже при ошибке
        calibration_data = {
            'cameras': {},
            'points_3d': {},
            'camera_points': {},
            'raw_camera_points': {},
            'observation_confidences': {},
            'line_support_data': {},
            'subpixel_refinement_enabled': True,
            'subpixel_refinement_stats': {},
            'K': np.array([[1000, 0, 800], [0, 1000, 600], [0, 0, 1]], dtype=np.float32),
            'dist_coeffs': np.zeros(5, dtype=np.float32),
            'images': {},
            'reconstruction_ids': [],
            'debug_logging': False,
            'precision_cleanup_enabled': True,
            'precision_target_mean_px': 0.5,
            'precision_target_p95_px': 1.0,
            'precision_target_max_px': 1.0,
            'precision_cleanup_rounds': 4,
        }
        return calibration_data

def set_debug_logging(enabled):
    """
    Включает или выключает подробный низкоуровневый вывод калибровки.

    Args:
        enabled: True для включения, False для отключения
    """
    global calibration_data
    if calibration_data is None:
        calibration_data = init_calibration()
    if calibration_data is not None:
        calibration_data['debug_logging'] = bool(enabled)

def set_camera_parameters(camera_id, K, dist_coeffs=None):
    """
    Устанавливает параметры камеры.
    
    Args:
        camera_id: ID камеры
        K: Матрица внутренних параметров камеры
        dist_coeffs: Коэффициенты дисторсии (опционально)
    """
    if not DEPENDENCIES_INSTALLED:
        logger.error("Невозможно установить параметры камеры: отсутствуют зависимости")
        return
        
    global calibration_data
    if calibration_data is None:
        calibration_data = init_calibration()
    
    # Проверяем, что calibration_core доступен
    if 'calibration_core' in globals() and hasattr(calibration_core, 'add_camera_parameters'):
        try:
            calibration_core.add_camera_parameters(calibration_data, camera_id, K, dist_coeffs)
        except Exception as e:
            logger.error(f"Ошибка при добавлении параметров камеры {camera_id}: {str(e)}")
            # Если основной метод не работает, сохраняем параметры напрямую в calibration_data
            if 'cameras' not in calibration_data:
                calibration_data['cameras'] = {}
            if 'K' not in calibration_data:
                calibration_data['K'] = K
            if 'dist_coeffs' not in calibration_data:
                calibration_data['dist_coeffs'] = dist_coeffs if dist_coeffs is not None else np.zeros(5, dtype=np.float32)
    else:
        # Если calibration_core не доступен, сохраняем параметры напрямую в calibration_data
        if 'cameras' not in calibration_data:
            calibration_data['cameras'] = {}
        if 'K' not in calibration_data:
            calibration_data['K'] = K
        if 'dist_coeffs' not in calibration_data:
            calibration_data['dist_coeffs'] = dist_coeffs if dist_coeffs is not None else np.zeros(5, dtype=np.float32)
    
    logger.info(f"Установлены параметры для камеры {camera_id}")

def set_points_from_blender(camera_id, points_data, image_path=None):
    """
    Добавляет точки для конкретной камеры из Blender.
    
    Args:
        camera_id (int): Идентификатор камеры
        points_data (dict): Словарь точек в формате {point_id: point_2d}, где point_2d - это координаты точки [x, y]
        image_path (str, optional): Путь к изображению
    """
    global calibration_data
    if calibration_data is None:
        calibration_data = init_calibration()
    
    # Преобразуем точки в формат OpenCV
    opencv_points = {}
    for point_id, point in points_data.items():
        # Точки уже 2D, поэтому просто добавляем их
        opencv_points[point_id] = point.copy()
    
    # Добавляем точки
    if 'calibration_core' in globals() and hasattr(calibration_core, 'add_image_points'):
        try:
            calibration_core.add_image_points(calibration_data, str(camera_id), opencv_points, image_path)
        except Exception as e:
            logger.error(f"Ошибка при добавлении точек для камеры {camera_id}: {str(e)}")
            # Если основной метод не работает, добавляем точки напрямую в calibration_data
            if 'camera_points' not in calibration_data:
                calibration_data['camera_points'] = {}
            if 'raw_camera_points' not in calibration_data:
                calibration_data['raw_camera_points'] = {}
            if str(camera_id) not in calibration_data['camera_points']:
                calibration_data['camera_points'][str(camera_id)] = {}
            if str(camera_id) not in calibration_data['raw_camera_points']:
                calibration_data['raw_camera_points'][str(camera_id)] = {}
            for point_id, point in opencv_points.items():
                point_array = np.asarray(point, dtype=np.float32).reshape(2)
                calibration_data['camera_points'][str(camera_id)][point_id] = point_array.copy()
                calibration_data['raw_camera_points'][str(camera_id)][point_id] = point_array.copy()
    else:
        # Если calibration_core не доступен, добавляем точки напрямую в данные калибровки
        if 'camera_points' not in calibration_data:
            calibration_data['camera_points'] = {}
        if 'raw_camera_points' not in calibration_data:
            calibration_data['raw_camera_points'] = {}
        if str(camera_id) not in calibration_data['camera_points']:
            calibration_data['camera_points'][str(camera_id)] = {}
        if str(camera_id) not in calibration_data['raw_camera_points']:
            calibration_data['raw_camera_points'][str(camera_id)] = {}
        for point_id, point in opencv_points.items():
            point_array = np.asarray(point, dtype=np.float32).reshape(2)
            calibration_data['camera_points'][str(camera_id)][point_id] = point_array.copy()
            calibration_data['raw_camera_points'][str(camera_id)][point_id] = point_array.copy()
    logger.info(f"Добавлено {len(points_data)} точек для камеры {camera_id}")

def run_calibration(initial_pair=None, min_points_for_camera=4, bundle_method='trf', 
                  bundle_ftol=1e-8, max_bundle_iterations=3, ransac_threshold=8.0, 
                  confidence=0.99, max_attempts=3, focal_range=(800, 3000), adapt_initial_focal=True,
                  check_focal_consistency=True, auto_correct_focal=False, force_same_focal=False,
                  debug_logging=False, progress_callback=None):
    """
    Запускает процесс калибровки.
    
    Args:
        initial_pair: Пара начальных камер (опционально)
        min_points_for_camera: Минимальное количество общих точек для добавления камеры
        bundle_method: Метод оптимизации для bundle adjustment ('trf', 'dogbox', 'lm')
        bundle_ftol: Порог сходимости по функции для bundle adjustment
        max_bundle_iterations: Максимальное количество итераций bundle adjustment
        ransac_threshold: Порог для RANSAC при оценке позы камеры
        confidence: Уровень доверия для RANSAC
        max_attempts: Максимальное количество попыток добавления камер
        focal_range: Кортеж (min_focal, max_focal) с реалистичными ограничениями для фокусного расстояния в пикселях
        adapt_initial_focal: Использовать ли механизм адаптации начального фокусного расстояния
        check_focal_consistency: Проверять согласованность фокусных расстояний между камерами
        auto_correct_focal: Автоматически корректировать аномальные фокусные расстояния
        force_same_focal: Принудительно использовать одинаковое фокусное расстояние для всех камер
    
    Returns:
        bool: True, если калибровка успешна, иначе False
    """
    try:
        logger.info("Запуск процесса калибровки...")
        
        # Проверяем, что calibration_data инициализирована
        global calibration_data
        if calibration_data is None:
            calibration_data = init_calibration()
            if calibration_data is None:
                logger.error("Не удалось инициализировать данные калибровки")
                return False

        # Проверяем, что K не равна None
        if calibration_data['K'] is None:
            logger.error("Матрица K не инициализирована")
            # Создаем стандартную матрицу калибровки
            calibration_data['K'] = np.array([
                [1000, 0, 800],  # fx, 0, cx
                [0, 1000, 600],  # 0, fy, cy
                [0, 0, 1]        # 0, 0, 1
            ], dtype=np.float32)
            logger.info("Создана стандартная матрица калибровки")

        calibration_data['debug_logging'] = bool(debug_logging)

        try:
            populate_observation_confidences(calibration_data)
        except Exception as e:
            logger.warning(f"Не удалось подготовить confidence наблюдений: {e}")
            calibration_data['observation_confidences'] = {}

        # Выполняем полную реконструкцию
        logger.info("Запуск полной реконструкции...")
        from calibration_modules import calibration_core

        result = calibration_core.perform_full_reconstruction(
            calib_data=calibration_data,  # Используем правильное имя параметра
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
        
        logger.info("Калибровка завершена")
        return result
    except Exception as e:
        logger.error(f"Ошибка при выполнении калибровки: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def get_camera_poses():
    """
    Возвращает позы камер.
    
    Returns:
        dict: Словарь поз камер {camera_id: (R, t)}
    """
    if not DEPENDENCIES_INSTALLED:
        logger.error("Невозможно получить позы камер: отсутствуют зависимости")
        return {}
        
    global calibration_data
    if calibration_data is None or 'cameras' not in calibration_data:
        logger.error("Данные о позах камер отсутствуют")
        return {}
    
    return calibration_data['cameras']

def get_3d_points():
    """
    Возвращает 3D точки.
    
    Returns:
        dict: Словарь 3D точек {point_id: point_3d}
    """
    if not DEPENDENCIES_INSTALLED:
        logger.error("Невозможно получить 3D точки: отсутствуют зависимости")
        return {}
        
    global calibration_data
    if calibration_data is None or 'points_3d' not in calibration_data:
        logger.error("Данные о 3D точках отсутствуют")
        return {}
    
    return calibration_data['points_3d']


def get_unreconstructed_diagnostics():
    """
    Возвращает диагностику для нереконструированных точек.
    
    Returns:
        list: Список диагностических записей из _unreconstructed_diagnostics_cache
    """
    global calibration_data
    if calibration_data is None:
        return []
    return list(calibration_data.get('_unreconstructed_diagnostics_cache', []))


def calculate_reprojection_errors(calibration_data, use_robust_statistics=False, return_detailed_stats=False):
    """
    Вычисляет ошибки репроекции для всех точек и камер.
    
    Args:
        calibration_data: Данные калибровки
        use_robust_statistics: Использовать ли робастные статистики (медиана, MAD) вместо среднего
        return_detailed_stats: Возвращать ли подробную статистику по ошибкам
        
    Returns:
        tuple: (total_error, errors_by_point, errors_by_camera) или
               (total_error, errors_by_point, errors_by_camera, detailed_stats) если return_detailed_stats=True
    """
    try:
        if not calibration_data or 'cameras' not in calibration_data or not calibration_data['cameras']:
            print("Реконструкция пуста")
            if return_detailed_stats:
                return 0.0, {}, {}, {}
            return 0.0, {}, {}
            
        if not calibration_data.get('points_3d'):
            print("Нет 3D точек")
            if return_detailed_stats:
                return 0.0, {}, {}, {}
            return 0.0, {}, {}
        errors_by_camera = {}
        errors_by_point = {}
        all_errors = []
        detailed_projections = {}  # Подробная информация о проекциях
        
        for camera_id, (R, t) in calibration_data['cameras'].items():
            points_2d = calibration_data['camera_points'].get(str(camera_id), {})  # Используем строковый ключ
            
            # Пропускаем камеры без точек
            if not points_2d:
                continue
                
            errors_by_camera[str(camera_id)] = []  # Сохраняем ошибки с строковым ключом
            camera_projections = {}
            
            # Вычисляем ошибки для каждой точки
            for point_id, point_2d in points_2d.items():
                    if point_id in calibration_data['points_3d']:
                        point_3d = calibration_data['points_3d'][point_id]
                        camera_K = calibration_data.get(f'K_{camera_id}', calibration_data['K'])
                        if camera_K is None:
                            continue
                        
                        # Проецируем 3D точку на 2D
                        try:
                            projected_point = utils.project_point(point_3d, R, t, camera_K, calibration_data.get('dist_coeffs'))
                            # Считаем ошибку
                            error = np.linalg.norm(projected_point - point_2d)
                            errors_by_camera[str(camera_id)].append(error)  # Используем строковый ключ
                            
                            # Сохраняем ошибку для точки
                            if point_id not in errors_by_point:
                                errors_by_point[point_id] = []
                            errors_by_point[point_id].append(error)
                            
                            all_errors.append(error)
                            
                            # Сохраняем подробную информацию о проекции
                            camera_projections[point_id] = {
                                'observed': point_2d.copy(),
                                'projected': projected_point,
                                'error': error
                            }
                        except Exception as e:
                            logger.error(f"Ошибка при проекции точки {point_id} для камеры {camera_id}: {str(e)}")
                            # Добавляем большую ошибку для точки
                            error = 1000.0
                            errors_by_camera[str(camera_id)].append(error)  # Используем строковый ключ
                            if point_id not in errors_by_point:
                                errors_by_point[point_id] = []
                            errors_by_point[point_id].append(error)
                            all_errors.append(error)
            # Сохраняем подробные проекции для камеры
            detailed_projections[camera_id] = camera_projections
        
        # Вычисляем среднее значение ошибки для каждой точки
        for point_id in errors_by_point:
            if use_robust_statistics:
                errors_by_point[point_id] = np.median(errors_by_point[point_id])
            else:
                errors_by_point[point_id] = np.mean(errors_by_point[point_id])
        
        # Рассчитываем общую среднюю ошибку
        if use_robust_statistics and all_errors:
            # Используем медиану и MAD для более устойчивой оценки
            from scipy import stats
            total_error = np.median(all_errors)
            mad = stats.median_abs_deviation(all_errors, scale='normal')
        else:
            total_error = np.mean(all_errors) if all_errors else 0.0
            mad = None
        
        if return_detailed_stats:
            # Вычисляем дополнительную статистику
            if all_errors:
                from scipy import stats
                detailed_stats = {
                    'total_points': len(all_errors),
                    'mean_error': float(np.mean(all_errors)),
                    'median_error': float(np.median(all_errors)),
                    'std_error': float(np.std(all_errors)),
                    'min_error': float(np.min(all_errors)),
                    'max_error': float(np.max(all_errors)),
                    'mad_error': float(mad) if mad is not None else float(stats.median_abs_deviation(all_errors, scale='normal')),
                    'percentiles': {
                        '25%': float(np.percentile(all_errors, 25)),
                        '50%': float(np.percentile(all_errors, 50)),
                        '75%': float(np.percentile(all_errors, 75)),
                        '90%': float(np.percentile(all_errors, 90)),
                        '95%': float(np.percentile(all_errors, 95)),
                        '99%': float(np.percentile(all_errors, 99))
                    },
                    'outliers_count': int(np.sum(np.array(all_errors) > (np.median(all_errors) + 3 * stats.median_abs_deviation(all_errors, scale='normal')))),
                    'projections': detailed_projections
                }
                
                return total_error, errors_by_point, errors_by_camera, detailed_stats
            else:
                return 0.0, {}, {}, {
                    'total_points': 0,
                    'mean_error': 0.0,
                    'median_error': 0.0,
                    'std_error': 0.0,
                    'min_error': 0.0,
                    'max_error': 0.0,
                    'mad_error': 0.0,
                    'percentiles': {str(p): 0.0 for p in [25, 50, 75, 90, 95, 99]},
                    'outliers_count': 0,
                    'projections': detailed_projections
                }
        
        return total_error, errors_by_point, errors_by_camera
    
    except Exception as e:
        print(f"Ошибка при вычислении ошибок репроекции: {str(e)}")
        import traceback
        traceback.print_exc()
        if return_detailed_stats:
            return 0.0, {}, {}, {}
        return 0.0, {}, {}

def test_module():
    """
    Тестирует функциональность модуля.
    
    Returns:
        bool: True, если тесты пройдены, иначе False
    """
    if not DEPENDENCIES_INSTALLED:
        logger.error("Невозможно запустить тесты: отсутствуют зависимости")
        return False
        
    logger.info("Запуск тестирования модуля...")
    
    # Инициализируем данные калибровки
    init_calibration()
    
    # Задаем параметры камеры
    K = np.array([
        [1000, 0, 500],
        [0, 1000, 500],
        [0, 0, 1]
    ])
    dist_coeffs = np.zeros(5)
    
    # Устанавливаем параметры камеры
    set_camera_parameters(0, K, dist_coeffs)
    
    # Создаем искусственные данные
    points_3d_gt = {}
    for i in range(20):
        points_3d_gt[i] = np.array([
            np.random.uniform(-1, 1),
            np.random.uniform(-1, 1),
            np.random.uniform(2, 4)
        ])
    
    # Создаем искусственные камеры
    cameras_gt = {}
    for i in range(3):
        # Создаем случайную позу камеры
        angle = np.random.uniform(0, 2 * np.pi)
        R = cv2.Rodrigues(np.array([0, angle, 0]))[0]
        t = np.array([[np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]]).T
        cameras_gt[i] = (R, t)
    
    # Проецируем точки на изображения
    for camera_id, (R, t) in cameras_gt.items():
        rvec, _ = cv2.Rodrigues(R)
        points_2d = {}
        
        for point_id, point_3d in points_3d_gt.items():
            # Проецируем точку
            point_3d_reshaped = point_3d.reshape(1, 3)
            projected_point, _ = cv2.projectPoints(
                point_3d_reshaped, rvec, t, K, dist_coeffs
            )
            
            # Добавляем шум
            noise = np.random.normal(0, 1, 2)
            point_2d = projected_point.flatten() + noise
            
            # Сохраняем точку
            points_2d[point_id] = point_2d
        
        # Добавляем точки для камеры
        set_points_from_blender(camera_id, points_2d)
    
    # Запускаем калибровку
    result = run_calibration(initial_pair=(0, 1))
    
    # Проверяем результаты
    if not result:
        logger.error("Калибровка не удалась")
        return False
    
    logger.info("Тестирование завершено успешно")
    return True

# Запуск тестирования при прямом вызове скрипта
if __name__ == "__main__":
    test_module()
