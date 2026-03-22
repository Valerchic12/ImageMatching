"""
Функции для пакетной оптимизации (bundle adjustment) параметров камер и 3D точек.
"""
import numpy as np
import cv2
import traceback
from typing import List, Tuple, Dict, Optional, Union
import logging
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares, minimize

def _debug_print(debug_logging, message):
    if debug_logging:
        print(message)

def _robust_scale(values, fallback=1.0):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return float(fallback)

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        upper = float(np.percentile(values, 90)) if values.size >= 4 else float(np.max(values))
        scale = max(upper - median, fallback)
    return float(max(scale, fallback))


def _lookup_observation_confidence(observation_confidences, camera_id, point_id):
    if not observation_confidences:
        return 1.0
    try:
        camera_confidences = observation_confidences.get(str(camera_id), {})
        confidence = float(camera_confidences.get(point_id, 1.0))
    except (AttributeError, TypeError, ValueError):
        confidence = 1.0
    return float(np.clip(confidence, 0.15, 1.0))


def _compute_observation_weights(observation_errors, camera_indices, point_indices, base_confidences=None):
    """
    Строит мягкие веса для наблюдений на основе глобального и локального хвоста ошибок.

    Идея: не удалять плохие наблюдения слишком рано, а уменьшать их влияние на BA.
    """
    observation_errors = np.asarray(observation_errors, dtype=np.float64).reshape(-1)
    if observation_errors.size == 0:
        return np.ones(0, dtype=np.float64), {
            'count': 0,
            'downweighted': 0,
            'median': 0.0,
            'p90': 0.0,
            'max': 0.0,
            'mean_weight': 1.0,
            'min_weight': 1.0,
        }

    camera_indices = np.asarray(camera_indices, dtype=np.int32).reshape(-1)
    point_indices = np.asarray(point_indices, dtype=np.int32).reshape(-1)
    if base_confidences is None:
        base_confidences = np.ones_like(observation_errors, dtype=np.float64)
    else:
        base_confidences = np.asarray(base_confidences, dtype=np.float64).reshape(-1)
        if base_confidences.size != observation_errors.size:
            base_confidences = np.ones_like(observation_errors, dtype=np.float64)
    base_confidences = np.clip(base_confidences, 0.15, 1.0)

    global_median = float(np.median(observation_errors))
    global_p90 = float(np.percentile(observation_errors, 90))
    global_scale = _robust_scale(observation_errors, fallback=max(global_median * 0.5, 0.75))

    camera_stats = {}
    point_stats = {}
    for camera_idx in np.unique(camera_indices):
        camera_errors = observation_errors[camera_indices == camera_idx]
        camera_stats[int(camera_idx)] = (
            float(np.median(camera_errors)),
            _robust_scale(camera_errors, fallback=max(global_scale * 0.8, 0.75)),
            int(camera_errors.size),
        )

    for point_idx in np.unique(point_indices):
        point_errors = observation_errors[point_indices == point_idx]
        point_stats[int(point_idx)] = (
            float(np.median(point_errors)),
            _robust_scale(point_errors, fallback=max(global_scale * 0.9, 0.75)),
            int(point_errors.size),
        )

    weights = np.ones_like(observation_errors, dtype=np.float64)
    for obs_idx, error in enumerate(observation_errors):
        camera_median, camera_scale, camera_count = camera_stats[int(camera_indices[obs_idx])]
        point_median, point_scale, point_count = point_stats[int(point_indices[obs_idx])]
        base_confidence = float(base_confidences[obs_idx])

        global_excess = max(0.0, (float(error) - global_median) / global_scale)
        camera_excess = max(0.0, (float(error) - camera_median) / camera_scale)
        point_excess = max(0.0, (float(error) - point_median) / point_scale)

        severity = max(global_excess, 0.9 * camera_excess, 0.85 * point_excess)
        confidence_softener = 0.70 + 0.30 * base_confidence
        effective_severity = severity / max(confidence_softener, 0.25)

        if effective_severity <= 1.0:
            weight = 1.0
        else:
            weight = 1.0 / np.sqrt(1.0 + 0.45 * (effective_severity - 1.0) ** 2)

        min_weight = 0.22
        if camera_count <= 6 or point_count <= 2:
            min_weight = 0.55
        elif camera_count <= 8 or point_count <= 3:
            min_weight = 0.40

        if float(error) > max(global_p90 * 1.75, global_median + global_scale * 4.5):
            min_weight = max(min_weight, 0.30)

        if effective_severity > 1.0:
            min_weight = max(0.12, min_weight * (0.75 + 0.25 * base_confidence))
        weights[obs_idx] = float(np.clip(weight, min_weight, 1.0))

    stats = {
        'count': int(observation_errors.size),
        'downweighted': int(np.count_nonzero(weights < 0.999)),
        'median': global_median,
        'p90': global_p90,
        'max': float(np.max(observation_errors)),
        'mean_weight': float(np.mean(weights)),
        'min_weight': float(np.min(weights)),
        'median_confidence': float(np.median(base_confidences)),
        'low_confidence': int(np.count_nonzero(base_confidences < 0.6)),
    }
    return weights, stats

def optimize_distortion(K, points_3d, cameras, camera_points, initial_dist=None):
    """
    Оптимизирует коэффициенты дисторсии.

    Args:
        K: Матрица калибровки камеры
        points_3d: Словарь 3D точек {id: point}
        cameras: Словарь камер {id: (R, t)}
        camera_points: Словарь точек для каждой камеры {camera_id: {point_id: point_2d}}
        initial_dist: Начальные коэффициенты дисторсии

    Returns:
        np.ndarray: Оптимизированные коэффициенты дисторсии
    """
    from scipy.optimize import minimize
    
    K = np.asarray(K, dtype=np.float64)
    if initial_dist is None:
        initial_dist = np.zeros(5, dtype=np.float64)
    else:
        initial_dist = np.asarray(initial_dist, dtype=np.float64).reshape(-1)
    
    bounds = [
        (-1.0, 1.0),   # k1
        (-1.0, 1.0),   # k2
        (-0.1, 0.1),   # p1
        (-0.1, 0.1),   # p2
        (-1.0, 1.0),   # k3
    ]
    regularization_scales = np.array([0.25, 0.25, 0.02, 0.02, 0.25], dtype=np.float64)
    regularization_weight = 0.25

    # Функция для оптимизации
    def compute_reprojection_error(dist_coeffs):
        total_error = 0
        point_count = 0
        
        # Для каждой камеры
        for cam_id, points in camera_points.items():
            if cam_id not in cameras:
                continue
                
            R, t = cameras[cam_id]
            R = np.asarray(R, dtype=np.float64)
            t = np.asarray(t, dtype=np.float64).reshape(3, 1)
            rvec, _ = cv2.Rodrigues(R)
            
            # Собираем 3D точки и их 2D проекции
            obj_points = []
            img_points = []
            
            for point_id, point_2d in points.items():
                if point_id not in points_3d:
                    continue
                    
                obj_points.append(np.asarray(points_3d[point_id], dtype=np.float64).reshape(3))
                img_points.append(np.asarray(point_2d, dtype=np.float64).reshape(2))
                
            if not obj_points:
                continue
                
            obj_points = np.array(obj_points, dtype=np.float64)
            img_points = np.array(img_points, dtype=np.float64)
            
            # Вычисляем проекции точек
            try:
                projected_dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
                projected_points, _ = cv2.projectPoints(
                    obj_points, rvec, t, K, projected_dist_coeffs
                )
                
                # Вычисляем ошибку репроекции
                errors = np.sqrt(np.sum((projected_points.reshape(-1, 2) - img_points) ** 2, axis=1))
                
                # Добавляем в общую ошибку
                total_error += np.sum(errors)
                point_count += len(errors)
            except Exception as e:
                print(f"Ошибка при проекции точек для камеры {cam_id}: {str(e)}")
                continue
            
        if point_count == 0:
            return 1e6  # Возвращаем большое значение ошибки

        mean_error = total_error / point_count
        regularization = regularization_weight * np.sum((np.asarray(dist_coeffs) / regularization_scales) ** 2)
        return mean_error + regularization
    
    # Проводим оптимизацию с несколькими методами
    methods = ['L-BFGS-B', 'Powell', 'TNC']
    best_result = None
    best_error = float('inf')
    
    for method in methods:
        try:
            minimize_kwargs = {
                'fun': compute_reprojection_error,
                'x0': initial_dist,
                'method': method,
            }
            if method in ('L-BFGS-B', 'Powell', 'TNC', 'SLSQP'):
                minimize_kwargs['bounds'] = bounds

            result = minimize(**minimize_kwargs)
            
            if result.success:
                final_error = compute_reprojection_error(result.x)
                print(f"Оптимизация дисторсии успешна ({method}): ошибка={final_error:.4f}, коэффициенты={result.x}")
                
                # Сохраняем лучший результат
                if final_error < best_error:
                    best_error = final_error
                    best_result = result
            else:
                print(f"Оптимизация дисторсии не сходится ({method}): {result.message}")
                
        except Exception as e:
            print(f"Ошибка при оптимизации дисторсии ({method}): {str(e)}")
            traceback.print_exc()
    
    # Возвращаем лучший результат или начальные коэффициенты
    if best_result is not None and best_result.success:
        clipped = np.array(best_result.x, dtype=np.float64)
        for idx, (low, high) in enumerate(bounds):
            clipped[idx] = np.clip(clipped[idx], low, high)
        return clipped
    else:
        print(f"Не удалось найти удовлетворительное решение, возвращаем начальные коэффициенты")
        return initial_dist

def optimize_shared_intrinsics(
    K,
    points_3d,
    cameras,
    camera_points,
    dist_coeffs=None,
    focal_range=None,
    force_same_focal=False,
    max_nfev=400,
    observation_confidences=None,
):
    """
    Уточняет общую матрицу внутренних параметров камеры при фиксированных позах и 3D точках.

    Args:
        K: Начальная матрица калибровки
        points_3d: Словарь 3D точек {id: point}
        cameras: Словарь камер {id: (R, t)}
        camera_points: Словарь наблюдений {camera_id: {point_id: point_2d}}
        dist_coeffs: Коэффициенты дисторсии
        focal_range: Допустимый диапазон фокусных расстояний (min_fx, max_fx)
        force_same_focal: Использовать один общий focal для fx/fy
        max_nfev: Лимит вызовов функции

    Returns:
        tuple: (optimized_K, initial_rmse, final_rmse)
    """
    try:
        if K is None or len(points_3d) < 4 or len(cameras) < 2:
            return K, None, None

        observations = []
        observed_points = []
        observation_camera_indices = []
        observation_point_indices = []
        observation_base_confidences = []
        camera_idx_map = {}
        point_idx_map = {}
        for cam_id, points in camera_points.items():
            if cam_id not in cameras:
                continue

            if cam_id not in camera_idx_map:
                camera_idx_map[cam_id] = len(camera_idx_map)
            R, t = cameras[cam_id]
            rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
            tvec = np.asarray(t, dtype=np.float64).reshape(3, 1)

            for point_id, point_2d in points.items():
                if point_id not in points_3d:
                    continue

                if point_id not in point_idx_map:
                    point_idx_map[point_id] = len(point_idx_map)
                point_3d = np.asarray(points_3d[point_id], dtype=np.float64).reshape(3)
                observed = np.asarray(point_2d, dtype=np.float64).reshape(2)
                observations.append((rvec, tvec, point_3d))
                observed_points.append(observed)
                observation_camera_indices.append(camera_idx_map[cam_id])
                observation_point_indices.append(point_idx_map[point_id])
                observation_base_confidences.append(
                    _lookup_observation_confidence(observation_confidences, cam_id, point_id)
                )

        if len(observations) < 12:
            print("Недостаточно наблюдений для оптимизации общей матрицы K")
            return K, None, None

        observed_points = np.asarray(observed_points, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        dist_coeffs = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)

        fx0 = float(K[0, 0])
        fy0 = float(K[1, 1])
        cx0 = float(K[0, 2])
        cy0 = float(K[1, 2])

        max_x = max(float(np.max(observed_points[:, 0])), cx0 * 2.0, 1.0)
        max_y = max(float(np.max(observed_points[:, 1])), cy0 * 2.0, 1.0)
        width_guess = max(max_x, cx0 * 2.0)
        height_guess = max(max_y, cy0 * 2.0)

        if focal_range is None:
            min_focal = max(min(width_guess, height_guess) * 0.5, 100.0)
            max_focal = max(width_guess, height_guess) * 4.0
        else:
            min_focal = float(focal_range[0])
            max_focal = float(focal_range[1])

        cx_bounds = (width_guess * 0.35, width_guess * 0.65)
        cy_bounds = (height_guess * 0.35, height_guess * 0.65)

        if force_same_focal:
            x0 = np.array([(fx0 + fy0) * 0.5, cx0, cy0], dtype=np.float64)
            lower = np.array([min_focal, cx_bounds[0], cy_bounds[0]], dtype=np.float64)
            upper = np.array([max_focal, cx_bounds[1], cy_bounds[1]], dtype=np.float64)
        else:
            x0 = np.array([fx0, fy0, cx0, cy0], dtype=np.float64)
            lower = np.array([min_focal, min_focal, cx_bounds[0], cy_bounds[0]], dtype=np.float64)
            upper = np.array([max_focal, max_focal, cx_bounds[1], cy_bounds[1]], dtype=np.float64)

        clipped_x0 = np.clip(x0, lower + 1e-6, upper - 1e-6)
        if not np.allclose(clipped_x0, x0):
            print(
                "Оптимизация общей матрицы K: начальная точка вне bounds, "
                f"clip x0 {x0.tolist()} -> {clipped_x0.tolist()}"
            )
            x0 = clipped_x0

        def build_K(params):
            if force_same_focal:
                fx = fy = float(params[0])
                cx = float(params[1])
                cy = float(params[2])
            else:
                fx = float(params[0])
                fy = float(params[1])
                cx = float(params[2])
                cy = float(params[3])

            return np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)

        observation_camera_indices = np.asarray(observation_camera_indices, dtype=np.int32)
        observation_point_indices = np.asarray(observation_point_indices, dtype=np.int32)
        observation_base_confidences = np.asarray(observation_base_confidences, dtype=np.float64)

        def raw_residuals_for_K(K_candidate):
            res = []
            for idx, (rvec, tvec, point_3d) in enumerate(observations):
                try:
                    projected, _ = cv2.projectPoints(
                        point_3d.reshape(1, 3),
                        rvec,
                        tvec,
                        K_candidate,
                        dist_coeffs
                    )
                    delta = projected.reshape(2) - observed_points[idx]
                    res.extend(delta.tolist())
                except Exception:
                    res.extend([1000.0, 1000.0])
            return np.asarray(res, dtype=np.float64)

        initial_raw_residuals = raw_residuals_for_K(build_K(x0))
        initial_observation_errors = np.linalg.norm(
            initial_raw_residuals.reshape(-1, 2),
            axis=1
        )
        observation_weights, weight_stats = _compute_observation_weights(
            initial_observation_errors,
            observation_camera_indices,
            observation_point_indices,
            base_confidences=observation_base_confidences,
        )
        observation_sqrt_weights = np.sqrt(np.clip(observation_weights, 1e-6, 1.0))

        print(
            f"K optimization reweighting: downweighted {weight_stats['downweighted']}/{weight_stats['count']} "
            f"наблюдений, median={weight_stats['median']:.2f}px, p90={weight_stats['p90']:.2f}px, "
            f"conf_median={weight_stats['median_confidence']:.2f}"
        )

        def residuals(params):
            K_candidate = build_K(params)
            raw_res = raw_residuals_for_K(K_candidate).reshape(-1, 2)
            res = (raw_res * observation_sqrt_weights[:, None]).reshape(-1).tolist()

            param_delta = (params - x0)
            param_scale = np.maximum(np.abs(x0), 1.0)
            res.extend((0.05 * (param_delta / param_scale)).tolist())
            return np.asarray(res, dtype=np.float64)

        initial_residuals = residuals(x0)
        initial_rmse = float(np.sqrt(np.mean(initial_residuals[:-len(x0)] ** 2)))

        result = least_squares(
            residuals,
            x0,
            bounds=(lower, upper),
            method='trf',
            loss='soft_l1',
            f_scale=2.0,
            max_nfev=max_nfev,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8
        )

        if not result.success and not np.isfinite(result.cost):
            print(f"Оптимизация общей матрицы K не удалась: {result.message}")
            return K, initial_rmse, initial_rmse

        optimized_K = build_K(result.x)
        final_residuals = residuals(result.x)
        final_rmse = float(np.sqrt(np.mean(final_residuals[:-len(x0)] ** 2)))

        print(
            "Оптимизация общей матрицы K: "
            f"RMSE {initial_rmse:.4f} -> {final_rmse:.4f}, "
            f"fx={optimized_K[0, 0]:.2f}, fy={optimized_K[1, 1]:.2f}, "
            f"cx={optimized_K[0, 2]:.2f}, cy={optimized_K[1, 2]:.2f}"
        )

        if final_rmse <= initial_rmse * 0.995:
            return optimized_K.astype(np.float32), initial_rmse, final_rmse

        print("Оптимизация общей матрицы K не дала значимого улучшения, сохраняем исходную K")
        return K, initial_rmse, final_rmse

    except Exception as e:
        print(f"Ошибка при оптимизации общей матрицы K: {str(e)}")
        traceback.print_exc()
        return K, None, None


def optimize_individual_focals(
    common_K,
    points_3d,
    cameras,
    camera_points,
    dist_coeffs=None,
    force_same_focal=False,
    min_observations=8,
    max_nfev=250,
):
    """
    Оценивает отдельный focal для каждой камеры при фиксированных позах и 3D-точках.

    Пока оптимизируются только fx/fy. cx/cy и коэффициенты дисторсии остаются общими,
    чтобы не делать модель слишком неустойчивой на разреженных ручных соответствиях.
    """
    try:
        if common_K is None or len(points_3d) < 4 or len(cameras) < 2:
            return {}, []

        common_K = np.asarray(common_K, dtype=np.float64)
        dist_coeffs = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)

        common_fx = float(common_K[0, 0])
        common_fy = float(common_K[1, 1])
        common_cx = float(common_K[0, 2])
        common_cy = float(common_K[1, 2])

        optimized_intrinsics = {}
        optimization_stats = []

        for camera_id in sorted(camera_points.keys(), key=lambda item: str(item)):
            if camera_id not in cameras:
                continue

            observations = camera_points.get(camera_id, {})
            object_points = []
            image_points = []
            for point_id, point_2d in observations.items():
                if point_id not in points_3d:
                    continue
                object_points.append(np.asarray(points_3d[point_id], dtype=np.float64).reshape(3))
                image_points.append(np.asarray(point_2d, dtype=np.float64).reshape(2))

            if len(object_points) < max(4, int(min_observations)):
                continue

            object_points = np.asarray(object_points, dtype=np.float64)
            image_points = np.asarray(image_points, dtype=np.float64)

            R, t = cameras[camera_id]
            rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
            tvec = np.asarray(t, dtype=np.float64).reshape(3, 1)

            image_width_guess = max(float(np.max(image_points[:, 0])), common_cx * 2.0, 1.0)
            image_height_guess = max(float(np.max(image_points[:, 1])), common_cy * 2.0, 1.0)
            min_focal = max(
                min(image_width_guess, image_height_guess) * 0.35,
                100.0,
                min(common_fx, common_fy) * 0.35,
            )
            max_focal = max(
                max(image_width_guess, image_height_guess) * 4.0,
                max(common_fx, common_fy) * 2.5,
            )

            if force_same_focal:
                x0 = np.array([(common_fx + common_fy) * 0.5], dtype=np.float64)
                lower = np.array([min_focal], dtype=np.float64)
                upper = np.array([max_focal], dtype=np.float64)
            else:
                x0 = np.array([common_fx, common_fy], dtype=np.float64)
                lower = np.array([min_focal, min_focal], dtype=np.float64)
                upper = np.array([max_focal, max_focal], dtype=np.float64)

            def build_K(params):
                if force_same_focal:
                    fx = fy = float(params[0])
                else:
                    fx = float(params[0])
                    fy = float(params[1])
                return np.array(
                    [
                        [fx, 0.0, common_cx],
                        [0.0, fy, common_cy],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )

            def raw_residuals(params):
                K_candidate = build_K(params)
                projected, _ = cv2.projectPoints(
                    object_points,
                    rvec,
                    tvec,
                    K_candidate,
                    dist_coeffs,
                )
                return projected.reshape(-1, 2) - image_points

            initial_raw = raw_residuals(x0)
            initial_errors = np.linalg.norm(initial_raw, axis=1)
            initial_rmse = float(np.sqrt(np.mean(np.sum(initial_raw ** 2, axis=1))))

            def residuals(params):
                raw = raw_residuals(params).reshape(-1)
                delta = (np.asarray(params, dtype=np.float64) - x0) / np.maximum(np.abs(x0), 1.0)
                reg_weight = 0.08 if len(object_points) >= 12 else 0.14
                return np.concatenate([raw, reg_weight * delta])

            result = least_squares(
                residuals,
                x0,
                bounds=(lower, upper),
                method="trf",
                loss="soft_l1",
                f_scale=2.0,
                max_nfev=max_nfev,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
            )

            final_params = result.x if np.all(np.isfinite(result.x)) else x0
            final_raw = raw_residuals(final_params)
            final_errors = np.linalg.norm(final_raw, axis=1)
            final_rmse = float(np.sqrt(np.mean(np.sum(final_raw ** 2, axis=1))))
            optimized_K = build_K(final_params)

            focal_delta = float(
                max(
                    abs(float(optimized_K[0, 0]) - common_fx),
                    abs(float(optimized_K[1, 1]) - common_fy),
                )
            )
            accepted = bool(final_rmse < initial_rmse * 0.995 and focal_delta > 1e-3)

            optimization_stats.append({
                "camera_id": str(camera_id),
                "count": int(len(object_points)),
                "initial_rmse": float(initial_rmse),
                "final_rmse": float(final_rmse),
                "fx": float(optimized_K[0, 0]),
                "fy": float(optimized_K[1, 1]),
                "accepted": accepted,
                "initial_median": float(np.median(initial_errors)) if initial_errors.size else 0.0,
                "final_median": float(np.median(final_errors)) if final_errors.size else 0.0,
            })

            if accepted:
                optimized_intrinsics[str(camera_id)] = optimized_K.astype(np.float32)

        return optimized_intrinsics, optimization_stats

    except Exception as e:
        print(f"Ошибка при оптимизации per-camera focal: {str(e)}")
        traceback.print_exc()
        return {}, []

def refine_calibration(points_3d, cameras, camera_points, K, dist_coeffs=None):
    """
    Уточняет параметры камер и 3D точки с помощью bundle adjustment.
    
    Args:
        points_3d: Словарь 3D точек {id: point}
        cameras: Словарь камер {id: (R, t)}
        camera_points: Словарь точек для каждой камеры {camera_id: {point_id: point_2d}}
        K: Матрица калибровки камеры
        dist_coeffs: Коэффициенты дисторсии (опционально)
        
    Returns:
        tuple: (points_3d_refined, cameras_refined)
    """
    try:
        from scipy.optimize import least_squares
        import numpy as np
        from scipy.sparse import lil_matrix
        
        print(f"\nЗапуск bundle adjustment...")
        print(f"Камер: {len(cameras)}, Точек: {len(points_3d)}")
        
        # Подготавливаем данные для оптимизации
        # Создаем сопоставления ID -> индекс
        camera_ids = list(cameras.keys())
        point_ids = list(points_3d.keys())
        
        camera_id_to_idx = {cam_id: idx for idx, cam_id in enumerate(camera_ids)}
        point_id_to_idx = {pt_id: idx for idx, pt_id in enumerate(point_ids)}
        
        # Создаем списки наблюдений
        obs_camera_idxs = []
        obs_point_idxs = []
        obs_points_2d = []
        
        for cam_id, points in camera_points.items():
            if cam_id not in camera_id_to_idx:
                continue
                
            cam_idx = camera_id_to_idx[cam_id]
            for point_id, point_2d in points.items():
                if point_id not in point_id_to_idx or point_id not in points_3d:
                    continue
                    
                obs_camera_idxs.append(cam_idx)
                obs_point_idxs.append(point_id_to_idx[point_id])
                obs_points_2d.append(point_2d)
        
        if not obs_camera_idxs:
            print("Нет наблюдений для оптимизации")
            return points_3d, cameras
            
        obs_camera_idxs = np.array(obs_camera_idxs)
        obs_point_idxs = np.array(obs_point_idxs)
        obs_points_2d = np.array(obs_points_2d)
        
        n_cameras = len(cameras)
        n_points = len(points_3d)
        n_observations = len(obs_points_2d)
        
        print(f"Подготовлено {n_observations} наблюдений для оптимизации")
        
        # Инициализируем параметры камер и точек
        camera_params = np.zeros((n_cameras, 6))  # [rot_vec(3), trans_vec(3)]
        point_params = np.zeros((n_points, 3))   # [X, Y, Z]
        
        # Заполняем начальные параметры камер
        for i, cam_id in enumerate(camera_ids):
            R, t = cameras[cam_id]
            rvec, _ = cv2.Rodrigues(R)
            camera_params[i, :3] = rvec.flatten()
            camera_params[i, 3:] = t.flatten()
        
        # Заполняем начальные параметры точек
        for i, point_id in enumerate(point_ids):
            point_params[i] = points_3d[point_id]
        
        # Определяем функцию ошибки
        def reprojection_error(params):
            """
            Функция ошибки репроекции для оптимизации.
            
            Args:
                params: Объединенные параметры [camera_params, point_params]
                
            Returns:
                np.ndarray: Вектор ошибок для всех наблюдений
            """
            # Разделяем параметры
            camera_params_flat = params[:n_cameras*6].reshape((n_cameras, 6))
            point_params_flat = params[n_cameras*6:].reshape((n_points, 3))
            
            errors = np.zeros(2 * n_observations)
            
            for i, (cam_idx, point_idx) in enumerate(zip(obs_camera_idxs, obs_point_idxs)):
                # Получаем параметры камеры
                rvec = camera_params_flat[cam_idx, :3]
                tvec = camera_params_flat[cam_idx, 3:]
                
                # Получаем 3D точку
                point_3d = point_params_flat[point_idx]
                
                # Проекция точки на изображение
                try:
                    projected_point, _ = cv2.projectPoints(
                        point_3d.reshape(1, 3),
                        rvec,
                        tvec,
                        K,
                        dist_coeffs
                    )
                    
                    projected = projected_point.flatten()
                    observed = obs_points_2d[i]
                    
                    # Вычисляем ошибку репроекции
                    errors[2*i] = projected[0] - observed[0]
                    errors[2*i+1] = projected[1] - observed[1]
                except Exception as e:
                    # Если возникла ошибка при проекции, возвращаем большую ошибку
                    errors[2*i] = 1000.0
                    errors[2*i+1] = 1000.0
            
            return errors
        
        # Подготавливаем начальный вектор параметров
        x0 = np.hstack([
            camera_params.flatten(),
            point_params.flatten()
        ])
        
        print(f"Начальный вектор параметров длины: {len(x0)}")
        print(f"Размер задачи: {n_observations*2} уравнений, {len(x0)} переменных")
        
        # Создаем разреженную матрицу Якоби
        A = lil_matrix((2 * n_observations, n_cameras * 6 + n_points * 3), dtype=np.int32)
        
        # Заполняем связи между параметрами и наблюдениями
        for i, (cam_idx, point_idx) in enumerate(zip(obs_camera_idxs, obs_point_idxs)):
            # Каждое наблюдение зависит от 6 параметров камеры и 3 параметров точки
            A[2*i:2*i+2, cam_idx*6:cam_idx*6+6] = 1  # Параметры камеры
            A[2*i:2*i+2, n_cameras*6+point_idx*3:n_cameras*6+point_idx*3+3] = 1  # Параметры точки
        
        # Преобразуем в CSR формат
        A_csr = A.tocsr()
        
        print(f"Размер матрицы Якоби: {A_csr.shape}")
        
        # Выполняем оптимизацию
        try:
            result = least_squares(
                reprojection_error,
                x0,
                jac_sparsity=A_csr,
                verbose=0,
                x_scale='jac',
                ftol=1e-10,
                method='trf',
                max_nfev=200
            )
            
            if result.success:
                print(f"Bundle adjustment завершен успешно за {result.nfev} итераций")
                print(f"Итоговая ошибка: {result.cost:.6f}")
                
                # Извлекаем оптимизированные параметры
                camera_params_opt = result.x[:n_cameras*6].reshape((n_cameras, 6))
                point_params_opt = result.x[n_cameras*6:].reshape((n_points, 3))
                
                # Обновляем камеры
                cameras_refined = {}
                for i, cam_id in enumerate(camera_ids):
                    rvec = camera_params_opt[i, :3]
                    tvec = camera_params_opt[i, 3:]
                    
                    # Преобразуем вектор поворота в матрицу
                    R, _ = cv2.Rodrigues(rvec)
                    
                    # Проверяем ортогональность матрицы поворота
                    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
                        print(f"Предупреждение: матрица поворота для камеры {cam_id} не ортогональна")
                        # Пытаемся исправить матрицу поворота
                        U, _, Vt = np.linalg.svd(R)
                        R = U @ Vt  # Убедиться, что это ортогональная матрица
                    
                    # Проверяем, что камера не слишком далеко
                    translation_norm = np.linalg.norm(tvec)
                    if translation_norm > 1000:
                        print(f"Предупреждение: камера {cam_id} имеет слишком большой вектор перемещения: {translation_norm:.2f}")
                        # Ограничиваем вектор перемещения
                        tvec = tvec / translation_norm * 1000
                    
                    cameras_refined[cam_id] = (R, tvec.reshape(3, 1))
                
                # Обновляем 3D точки
                points_3d_refined = {}
                for i, point_id in enumerate(point_ids):
                    point_3d = point_params_opt[i]
                    
                    # Проверяем, что точка не имеет аномальных значений
                    if np.any(np.abs(point_3d) > 10000):
                        print(f"Предупреждение: точка {point_id} имеет аномальные координаты: {point_3d}")
                        # Используем исходное значение точки
                        points_3d_refined[point_id] = points_3d[point_id]
                    else:
                        points_3d_refined[point_id] = point_3d
                
                print(f"Оптимизировано {len(points_3d_refined)} точек и {len(cameras_refined)} камер")
                
                return points_3d_refined, cameras_refined
            else:
                print(f"Bundle adjustment не сходится: {result.message}")
                return points_3d, cameras
                
        except Exception as e:
            print(f"Ошибка при оптимизации bundle adjustment: {str(e)}")
            traceback.print_exc()
            return points_3d, cameras
            
    except Exception as e:
        print(f"Ошибка при подготовке bundle adjustment: {str(e)}")
        traceback.print_exc()
        return points_3d, cameras

def bundle_adjustment(points_2d, points_3d, image_sizes, initial_K, camera_type='auto'):
    """
    Выполняет полную процедуру bundle adjustment для уточнения параметров камер и 3D точек.
    
    Args:
        points_2d: Словарь 2D точек {camera_id: {point_id: point_2d}}
        points_3d: Словарь 3D точек {point_id: point_3d}
        image_sizes: Размеры изображений [(width, height), ...]
        initial_K: Начальная матрица калибровки
        camera_type: Тип камеры ('auto', 'perspective', 'fisheye', 'mei', 'kannala_brandt')
        
    Returns:
        tuple: (K, dist_coeffs, cameras, reprojection_error) или None в случае ошибки
    """
    try:
        from scipy.optimize import least_squares
        import numpy as np
        from scipy.sparse import lil_matrix
        
        print(f"Начало bundle adjustment...")
        print(f"Камер: {len(points_2d)}, Точек: {len(points_3d)}")
        
        # Проверяем, что у нас есть хотя бы 2 камеры и 3D точки
        if len(points_2d) < 2 or len(points_3d) < 3:
            print("Недостаточно данных для bundle adjustment")
            return None
        
        # Создаем список ID камер и точек
        camera_ids = list(points_2d.keys())
        point_ids = list(points_3d.keys())
        
        # Создаем сопоставления ID -> индекс
        camera_id_to_idx = {cam_id: idx for idx, cam_id in enumerate(camera_ids)}
        point_id_to_idx = {pt_id: idx for idx, pt_id in enumerate(point_ids)}
        
        # Создаем списки наблюдений
        obs_camera_idxs = []
        obs_point_idxs = []
        obs_points_2d = []
        
        for cam_id, points in points_2d.items():
            cam_idx = camera_id_to_idx[cam_id]
            for point_id, point_2d in points.items():
                if point_id in point_id_to_idx:  # Проверяем, что 3D точка существует
                    obs_camera_idxs.append(cam_idx)
                    obs_point_idxs.append(point_id_to_idx[point_id])
                    obs_points_2d.append(point_2d)
        
        if len(obs_points_2d) < 10:
            print(f"Недостаточно наблюдений для bundle adjustment: {len(obs_points_2d)} (минимум 10)")
            return None
            
        obs_camera_idxs = np.array(obs_camera_idxs)
        obs_point_idxs = np.array(obs_point_idxs)
        obs_points_2d = np.array(obs_points_2d)
        
        n_cameras = len(camera_ids)
        n_points = len(point_ids)
        n_observations = len(obs_points_2d)
        
        print(f"Подготовлено {n_observations} наблюдений для {n_cameras} камер и {n_points} точек")
        
        # Инициализируем параметры
        camera_params = np.zeros((n_cameras, 6))  # [rot_vec(3), trans_vec(3)]
        point_params = np.zeros((n_points, 3))   # [X, Y, Z]
        
        # Заполняем начальные параметры точек
        for i, point_id in enumerate(point_ids):
            point_params[i] = points_3d[point_id]
        
        # Заполняем начальные параметры камер (пока только позиции)
        # Для начальной инициализации используем identity для первой камеры и оценки для остальных
        first_camera_initialized = False
        for i, cam_id in enumerate(camera_ids):
            if not first_camera_initialized:
                # Первая камера в начале координат
                camera_params[i, :3] = np.zeros(3)  # [0, 0, 0] для вектора поворота
                camera_params[i, 3:] = np.zeros(3)  # [0, 0, 0] для вектора переноса
                first_camera_initialized = True
            else:
                # Для остальных камер используем оценки из предыдущей калибровки
                if cam_id in cameras:  # Предполагаем, что переменная cameras определена в глобальной области
                    R, t = cameras[cam_id]
                    rvec, _ = cv2.Rodrigues(R)
                    camera_params[i, :3] = rvec.flatten()
                    camera_params[i, 3:] = t.flatten()
        
        # Определяем функцию ошибки
        def reprojection_error(params):
            """
            Функция ошибки репроекции для оптимизации.
            
            Args:
                params: Объединенные параметры [camera_params, point_params]
                
            Returns:
                np.ndarray: Вектор ошибок для всех наблюдений
            """
            # Разделяем параметры
            camera_params_flat = params[:n_cameras*6].reshape((n_cameras, 6))
            point_params_flat = params[n_cameras*6:].reshape((n_points, 3))
            
            errors = np.zeros(2 * n_observations)
            
            for i, (cam_idx, point_idx) in enumerate(zip(obs_camera_idxs, obs_point_idxs)):
                # Получаем параметры камеры
                rvec = camera_params_flat[cam_idx, :3]
                tvec = camera_params_flat[cam_idx, 3:]
                
                # Получаем 3D точку
                point_3d = point_params_flat[point_idx]
                
                # Проекция точки на изображение
                try:
                    projected_point, _ = cv2.projectPoints(
                        point_3d.reshape(1, 3),
                        rvec,
                        tvec,
                        initial_K,  # Используем начальную матрицу калибровки
                        None  # Пока не оптимизируем дисторсию
                    )
                    
                    projected = projected_point.flatten()
                    observed = obs_points_2d[i]
                    
                    # Вычисляем ошибку репроекции
                    errors[2*i] = projected[0] - observed[0]
                    errors[2*i+1] = projected[1] - observed[1]
                except Exception as e:
                    # Если возникла ошибка при проекции, возвращаем большую ошибку
                    errors[2*i] = 1000.0
                    errors[2*i+1] = 1000.0
            
            return errors
        
        # Подготавливаем начальный вектор параметров
        x0 = np.hstack([
            camera_params.flatten(),
            point_params.flatten()
        ])
        
        print(f"Начальный вектор параметров длины: {len(x0)}")
        print(f"Размер задачи: {n_observations*2} уравнений, {len(x0)} переменных")
        
        # Создаем разреженную матрицу Якоби
        A = lil_matrix((2 * n_observations, n_cameras * 6 + n_points * 3), dtype=np.int32)
        
        # Заполняем связи между параметрами и наблюдениями
        for i, (cam_idx, point_idx) in enumerate(zip(obs_camera_idxs, obs_point_idxs)):
            # Каждое наблюдение зависит от 6 параметров камеры и 3 параметров точки
            A[2*i:2*i+2, cam_idx*6:cam_idx*6+6] = 1  # Параметры камеры
            A[2*i:2*i+2, n_cameras*6+point_idx*3:n_cameras*6+point_idx*3+3] = 1  # Параметры точки
        
        # Преобразуем в CSR формат
        A_csr = A.tocsr()
        
        print(f"Размер матрицы Якоби: {A_csr.shape}")
        
        # Выполняем оптимизацию
        try:
            result = least_squares(
                reprojection_error,
                x0,
                jac_sparsity=A_csr,
                verbose=0,
                x_scale='jac',
                ftol=1e-10,
                method='trf',
                max_nfev=200
            )
            
            if result.success:
                print(f"Bundle adjustment завершен успешно за {result.nfev} итераций")
                print(f"Итоговая ошибка: {result.cost:.6f}")
                
                # Извлекаем оптимизированные параметры
                camera_params_opt = result.x[:n_cameras*6].reshape((n_cameras, 6))
                point_params_opt = result.x[n_cameras*6:].reshape((n_points, 3))
                
                # Обновляем камеры
                cameras_opt = {}
                for i, cam_id in enumerate(camera_ids):
                    rvec = camera_params_opt[i, :3]
                    tvec = camera_params_opt[i, 3:]
                    
                    # Преобразуем вектор поворота в матрицу
                    R, _ = cv2.Rodrigues(rvec)
                    
                    # Проверяем ортогональность матрицы поворота
                    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
                        print(f"Предупреждение: матрица поворота для камеры {cam_id} не ортогональна")
                        # Пытаемся исправить матрицу поворота
                        U, _, Vt = np.linalg.svd(R)
                        R = U @ Vt  # Убедиться, что это ортогональная матрица
                    
                    cameras_opt[cam_id] = (R, tvec.reshape(3, 1))
                
                # Обновляем 3D точки
                points_3d_opt = {}
                for i, point_id in enumerate(point_ids):
                    points_3d_opt[point_id] = point_params_opt[i]
                
                # Вычисляем среднюю ошибку репроекции
                final_errors = reprojection_error(result.x)
                reprojection_error_mean = np.sqrt(np.mean(final_errors**2))
                
                print(f"Средняя ошибка репроекции: {reprojection_error_mean:.4f} пикселей")
                
                # Возвращаем результаты (предполагаем, что K остается постоянной)
                return initial_K, None, cameras_opt, reprojection_error_mean
            else:
                print(f"Bundle adjustment не сходится: {result.message}")
                return None
                
        except Exception as e:
            print(f"Ошибка при оптимизации bundle adjustment: {str(e)}")
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"Ошибка при подготовке bundle adjustment: {str(e)}")
        traceback.print_exc()
        return None

def bundle_adjust_step(
    points_3d,
    cameras,
    camera_points,
    K,
    dist_coeffs=None,
    method='trf',
    ftol=1e-8,
    max_nfev=1000,
    debug_logging=False,
    observation_confidences=None,
    camera_intrinsics=None,
    fixed_camera_ids=None,
):
    """
    Выполняет один шаг bundle adjustment для камер и точек.

    Args:
        points_3d: Словарь 3D точек {id: point}
        cameras: Словарь камер {id: (R, t)}
        camera_points: Словарь точек для каждой камеры {camera_id: {point_id: point_2d}}
        K: Матрица калибровки камеры
        dist_coeffs: Коэффициенты дисторсии (опционально)
        method: Метод оптимизации ('trf', 'dogbox', 'lm')
        ftol: Порог сходимости по функции
        max_nfev: Максимальное количество вычислений функции

    Returns:
        tuple: (points_3d_optimized, cameras_optimized)
    """
    print(f"\nЗапуск bundle adjustment...")
    print(f"Камер: {len(cameras)}, Точек: {len(points_3d)}")
    
    try:
        # Подготавливаем данные для оптимизации
        fixed_camera_ids_set = set(str(c) for c in (fixed_camera_ids or []))
        free_camera_idx_to_param_idx = {}
        camera_idx_to_idx = {}  # Словарь {camera_id: index}
        point_id_to_idx = {}    # Словарь {point_id: index}
        
        camera_indices = []     # Массив индексов камер
        point_indices = []      # Массив индексов точек
        points_2d = []          # Массив 2D точек
        base_confidences = []   # Базовые confidence 2D наблюдений
        
        # Заполняем данные для оптимизации
        for cam_idx, camera_data in cameras.items():
            if cam_idx not in camera_points:
                continue
                
            for point_id, point_2d in camera_points[cam_idx].items():
                if point_id not in points_3d:
                    continue
                    
                # Добавляем камеру, если ее еще нет
                if cam_idx not in camera_idx_to_idx:
                    cur_idx = len(camera_idx_to_idx)
                    camera_idx_to_idx[cam_idx] = cur_idx
                    if str(cam_idx) not in fixed_camera_ids_set:
                        free_camera_idx_to_param_idx[cur_idx] = len(free_camera_idx_to_param_idx)
                    
                # Добавляем точку, если ее еще нет
                if point_id not in point_id_to_idx:
                    point_id_to_idx[point_id] = len(point_id_to_idx)
                    
                # Добавляем данные наблюдения
                camera_indices.append(camera_idx_to_idx[cam_idx])
                point_indices.append(point_id_to_idx[point_id])
                points_2d.append(point_2d)
                base_confidences.append(
                    _lookup_observation_confidence(observation_confidences, cam_idx, point_id)
                )
        
        # Проверяем, что у нас есть данные для оптимизации
        if not camera_indices or not point_indices:
            print("Нет данных для оптимизации")
            return points_3d, cameras
            
        # Преобразуем в numpy массивы
        camera_indices = np.array(camera_indices, dtype=np.int32)
        point_indices = np.array(point_indices, dtype=np.int32)
        points_2d = np.array(points_2d, dtype=np.float64)
        base_confidences = np.asarray(base_confidences, dtype=np.float64)
        
        # Создаем массивы для параметров камер и точек
        n_cameras = len(camera_idx_to_idx)
        n_points = len(point_id_to_idx)
        n_free_cameras = len(free_camera_idx_to_param_idx)
        
        print(f"Подготовлено {n_cameras} камер и {n_points} точек для оптимизации")
        
        # Параметры камер: [R1, R2, R3, t1, t2, t3] для каждой камеры
        camera_params_all = np.zeros((n_cameras, 6), dtype=np.float64)
        camera_params_free = np.zeros((n_free_cameras, 6), dtype=np.float64)
        camera_matrices = [None] * n_cameras
        
        # Параметры точек: [X, Y, Z] для каждой точки
        point_params = np.zeros((n_points, 3), dtype=np.float64)
        
        # Заполняем параметры камер
        for cam_id, cam_idx in camera_idx_to_idx.items():
            R, t = cameras[cam_id]
            
            # Преобразуем матрицу поворота в вектор Родригеса
            rvec, _ = cv2.Rodrigues(R)
            
            # Заполняем параметры камеры
            camera_params_all[cam_idx, :3] = rvec.flatten()
            camera_params_all[cam_idx, 3:] = t.flatten()
            camera_K = K
            if camera_intrinsics is not None:
                camera_K = camera_intrinsics.get(str(cam_id), camera_K)
            camera_matrices[cam_idx] = np.asarray(camera_K, dtype=np.float64)
            if cam_idx in free_camera_idx_to_param_idx:
                free_idx = free_camera_idx_to_param_idx[cam_idx]
                camera_params_free[free_idx] = camera_params_all[cam_idx]
        
        # Заполняем параметры точек
        for point_id, point_idx in point_id_to_idx.items():
            point_3d = points_3d[point_id]
            # Убедимся, что точка имеет правильный формат (3,)
            if hasattr(point_3d, 'shape') and len(point_3d.shape) > 1:
                if point_3d.shape[0] == 3 and point_3d.shape[1] == 1:
                    # Точка в формате (3, 1) - преобразуем в (3,)
                    point_3d = point_3d.ravel()
                elif point_3d.shape[0] == 1 and point_3d.shape[1] == 3:
                    # Точка в формате (1, 3) - преобразуем в (3,)
                    point_3d = point_3d.ravel()
                elif point_3d.size == 3:
                    # Точка в плоском формате с 3 элементами - используем как есть
                    point_3d = np.asarray(point_3d).flatten()
            elif hasattr(point_3d, '__len__') and len(point_3d) == 3:
                # Список или tuple из 3 элементов
                point_3d = np.array(point_3d, dtype=np.float64)
            else:
                # Преобразуем в numpy array и убедимся, что это вектор (3,)
                point_3d = np.asarray(point_3d, dtype=np.float64).flatten()
                if point_3d.size != 3:
                    print(f"Предупреждение: точка {point_id} имеет неверный размер {point_3d.size}, используем нулевую точку")
                    point_3d = np.zeros(3, dtype=np.float64)
            
            point_params[point_idx] = point_3d
        
        camera_to_obs = {c: [] for c in range(n_cameras)}
        for i, c in enumerate(camera_indices):
            camera_to_obs[c].append(i)
            
        def compute_raw_residuals(params):
            """
            Вычисляет невзвешенные ошибки репроекции.
            
            Args:
                params: Параметры камер и точек
                
            Returns:
                np.ndarray: Невзвешенные ошибки репроекции
            """
            # Разделяем параметры на камеры и точки
            camera_params_opt_free = params[:n_free_cameras * 6].reshape((n_free_cameras, 6))
            point_params_opt = params[n_free_cameras * 6:].reshape((n_points, 3))
            
            # Вычисляем ошибки репроекции
            reprojection_errors = np.zeros(len(points_2d) * 2, dtype=np.float64)
            
            # Vectorized projection per camera
            for cam_idx in range(n_cameras):
                obs_idxs = camera_to_obs[cam_idx]
                if not obs_idxs:
                    continue
                
                if cam_idx in free_camera_idx_to_param_idx:
                    param_idx = free_camera_idx_to_param_idx[cam_idx]
                    cam_params = camera_params_opt_free[param_idx]
                else:
                    cam_params = camera_params_all[cam_idx]
                    
                rvec = cam_params[:3]
                tvec = cam_params[3:]
                
                # Получаем 3D точку
                pt_idxs = point_indices[obs_idxs]
                pts_3d = point_params_opt[pt_idxs]
                
                # Проецируем точку на изображение
                try:
                    pts_2d_proj, _ = cv2.projectPoints(
                        pts_3d,
                        rvec,
                        tvec,
                        camera_matrices[cam_idx],
                        dist_coeffs
                    )
                    
                    # Вычисляем ошибку репроекции
                    pts_2d_proj = pts_2d_proj.reshape(-1, 2)
                    pts_2d_obs = points_2d[obs_idxs]
                    
                    diff = pts_2d_proj - pts_2d_obs
                    obs_idxs_np = np.array(obs_idxs)
                    reprojection_errors[2 * obs_idxs_np] = diff[:, 0]
                    reprojection_errors[2 * obs_idxs_np + 1] = diff[:, 1]
                except Exception as e:
                    print(f"Ошибка при проекции для камеры {cam_idx}: {str(e)}")
                    obs_idxs_np = np.array(obs_idxs)
                    # Устанавливаем большие ошибки для этой точки
                    reprojection_errors[2 * obs_idxs_np] = 1000.0
                    reprojection_errors[2 * obs_idxs_np + 1] = 1000.0
            
            return reprojection_errors

        # Создаем начальные параметры
        x0 = np.hstack([camera_params_free.flatten(), point_params.flatten()]).astype(np.float64)
        
        initial_raw_residuals = compute_raw_residuals(x0)
        initial_observation_errors = np.linalg.norm(
            initial_raw_residuals.reshape(-1, 2),
            axis=1
        )
        observation_weights, weight_stats = _compute_observation_weights(
            initial_observation_errors,
            camera_indices,
            point_indices,
            base_confidences=base_confidences,
        )
        observation_sqrt_weights = np.sqrt(np.clip(observation_weights, 1e-6, 1.0))

        print(
            f"BA reweighting: downweighted {weight_stats['downweighted']}/{weight_stats['count']} "
            f"наблюдений, median={weight_stats['median']:.2f}px, "
            f"p90={weight_stats['p90']:.2f}px, min_weight={weight_stats['min_weight']:.2f}, "
            f"conf_median={weight_stats['median_confidence']:.2f}"
        )
        _debug_print(
            debug_logging,
            f"BA weights: mean={weight_stats['mean_weight']:.4f}, max_error={weight_stats['max']:.4f}px"
        )

        # Создаем функцию для оптимизации
        def objective_function(params):
            raw_residuals = compute_raw_residuals(params).reshape(-1, 2)
            weighted_residuals = raw_residuals * observation_sqrt_weights[:, None]
            return weighted_residuals.reshape(-1)
        
        _debug_print(debug_logging, f"Начальный параметр вектора длины: {len(x0)}")
        
        # Создаем разреженную матрицу Якоби
        A = lil_matrix((2 * len(points_2d), n_free_cameras * 6 + n_points * 3), dtype=int)
        
        # Заполняем матрицу Якоби
        for i, (cam_idx, point_idx) in enumerate(zip(camera_indices, point_indices)):
            # Параметры камеры влияют на все точки, видимые этой камерой
            if cam_idx in free_camera_idx_to_param_idx:
                param_idx = free_camera_idx_to_param_idx[cam_idx]
                A[2*i:2*i+2, param_idx*6:param_idx*6+6] = 1
            
            # Параметры точки влияют только на проекции этой точки
            A[2*i:2*i+2, n_free_cameras*6+point_idx*3:n_free_cameras*6+point_idx*3+3] = 1
        
        # Преобразуем в CSR формат для быстрого умножения
        A = A.tocsr()
        
        _debug_print(debug_logging, f"Размер матрицы Якоби: {A.shape}")
        
        initial_residuals = objective_function(x0)
        initial_cost = 0.5 * np.sum(initial_residuals ** 2)

        # Запускаем оптимизацию
        try:
            result = least_squares(
                objective_function,
                x0,
                jac_sparsity=A,
                verbose=0,
                x_scale='jac',
                ftol=ftol,
                xtol=ftol,
                gtol=ftol,
                method=method,
                max_nfev=max_nfev,
                loss='soft_l1',
                f_scale=2.0
            )
            
            accepted_result = result.success
            if not accepted_result and np.isfinite(result.cost) and result.cost < initial_cost * 0.995:
                print(f"Bundle adjustment не достиг формального success, но улучшил cost: {initial_cost:.6f} -> {result.cost:.6f}")
                accepted_result = True

            if not accepted_result:
                print(f"Bundle adjustment не сходится: {result.message}")
                return points_3d, cameras
        except Exception as e:
            print(f"Ошибка при запуске оптимизации: {str(e)}")
            # Пробуем с другими параметрами
            try:
                result = least_squares(
                    objective_function,
                    x0,
                    verbose=0,
                    x_scale='jac',
                    ftol=ftol,
                    xtol=ftol,
                    gtol=ftol,
                    method='trf',
                    max_nfev=max_nfev,
                    loss='soft_l1',
                    f_scale=2.0
                )
            except Exception as e2:
                print(f"Ошибка при запуске оптимизации (альтернатива): {str(e2)}")
                return points_3d, cameras
        
        # Разделяем результаты на камеры и точки
        camera_params_opt_free = result.x[:n_free_cameras * 6].reshape((n_free_cameras, 6))
        point_params_opt = result.x[n_free_cameras * 6:].reshape((n_points, 3))
        
        # Создаем новые словари для оптимизированных параметров
        cameras_optimized = {}
        points_3d_optimized = {}
        
        # Заполняем оптимизированные параметры камер
        for cam_id, cam_idx in camera_idx_to_idx.items():
            if cam_idx in free_camera_idx_to_param_idx:
                free_idx = free_camera_idx_to_param_idx[cam_idx]
                rvec = camera_params_opt_free[free_idx, :3]
                tvec = camera_params_opt_free[free_idx, 3:]
            else:
                rvec = camera_params_all[cam_idx, :3]
                tvec = camera_params_all[cam_idx, 3:]
            
            # Преобразуем вектор Родригеса в матрицу поворота
            R, _ = cv2.Rodrigues(rvec)
            
            # Проверяем, что матрица поворота корректна
            if not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
                print(f"Предупреждение: матрица поворота для камеры {cam_id} не ортогональна")
            
            # Сохраняем оптимизированные параметры камеры
            cameras_optimized[cam_id] = (R, tvec)
        
        # Заполняем оптимизированные параметры точек
        for point_id, point_idx in point_id_to_idx.items():
            points_3d_optimized[point_id] = point_params_opt[point_idx]
        
        # Добавляем неоптимизированные камеры и точки
        for cam_id, camera_data in cameras.items():
            if cam_id not in cameras_optimized:
                cameras_optimized[cam_id] = camera_data
        
        for point_id, point_3d in points_3d.items():
            if point_id not in points_3d_optimized:
                points_3d_optimized[point_id] = point_3d
        
        print(f"Bundle adjustment завершен: оптимизировано {n_points} точек и {n_cameras} камер")
        print(f"Итоговая ошибка: {result.cost:.6f}")
        
        return points_3d_optimized, cameras_optimized
        
    except Exception as e:
        print(f"Ошибка при выполнении bundle adjustment: {str(e)}")
        traceback.print_exc()
        return points_3d, cameras


def huber_loss(residuals, delta=1.345):
   """
   Huber loss функция для робастной оптимизации.
   """
   abs_residuals = np.abs(residuals)
   huber_loss_values = np.zeros_like(residuals)
   
   # Для |x| <= delta используем квадратичную функцию
   mask = abs_residuals <= delta
   huber_loss_values[mask] = 0.5 * residuals[mask] ** 2
   
   # Для |x| > delta используем линейную функцию
   mask = ~mask
   huber_loss_values[mask] = delta * abs_residuals[mask] - 0.5 * delta ** 2
   
   return huber_loss_values


def cauchy_loss(residuals, c=1.0):
   """
   Cauchy (Lorentzian) loss функция для робастной оптимизации.
   """
   return c**2 * np.log(1 + (residuals / c)**2)


def tukey_loss(residuals, c=4.6851):
   """
   Tukey loss функция для робастной оптимизации.
   """
   abs_residuals = np.abs(residuals)
   tukey_loss_values = np.zeros_like(residuals)
   
   # Для |x| <= c
   mask = abs_residuals <= c
   squared = (residuals[mask] / c)**2
   tukey_loss_values[mask] = (c**2 / 6) * (1 - (1 - squared)**3)
   
   # Для |x| > c
   mask = ~mask
   tukey_loss_values[mask] = c**2 / 6
   
   return tukey_loss_values


def robust_reprojection_error(params, camera_indices, point_indices, points_2d, n_cameras, n_points, K, dist_coeffs=None, loss_function='huber', loss_params=None):
   """
   Робастная функция ошибки репроекции с использованием различных loss функций.
   """
   if loss_params is None:
       loss_params = {}
   
   # Разделяем параметры
   camera_params = params[:n_cameras*6].reshape((n_cameras, 6))
   point_params = params[n_cameras*6:].reshape((n_points, 3))
   
   # Вычисляем стандартные ошибки репроекции
   errors = np.zeros(2 * len(points_2d))
   
   for i, (cam_idx, point_idx) in enumerate(zip(camera_indices, point_indices)):
       # Получаем параметры камеры
       rvec = camera_params[cam_idx, :3]
       tvec = camera_params[cam_idx, 3:]
       
       # Получаем 3D точку
       point_3d = point_params[point_idx]
       
       # Проекция точки на изображение
       try:
           projected_point, _ = cv2.projectPoints(
               point_3d.reshape(1, 3),
               rvec,
               tvec,
               K,
               dist_coeffs
           )
           
           projected = projected_point.flatten()
           observed = points_2d[i]
           
           # Вычисляем ошибку репроекции
           errors[2*i] = projected[0] - observed[0]
           errors[2*i+1] = projected[1] - observed[1]
       except Exception as e:
           # Если возникла ошибка при проекции, возвращаем большую ошибку
           errors[2*i] = 1000.0
           errors[2*i+1] = 1000.0
   
   # Применяем робастную loss функцию
   if loss_function == 'huber':
       delta = loss_params.get('delta', 1.345)
       return np.sqrt(2 * huber_loss(errors, delta))
   elif loss_function == 'cauchy':
       c = loss_params.get('c', 1.0)
       return np.sqrt(2 * cauchy_loss(errors, c))
   elif loss_function == 'tukey':
       c = loss_params.get('c', 4.6851)
       return np.sqrt(2 * tukey_loss(errors, c))
   else:  # 'linear' (стандартный случай)
       return errors


def hierarchical_bundle_adjustment(points_3d, cameras, camera_points, K, dist_coeffs=None, initial_refinements=3, final_refinements=5):
   """
   Иерархическая оптимизация bundle adjustment с поэтапной настройкой параметров.
   """
   print(f"\nЗапуск иерархического bundle adjustment...")
   print(f"Камер: {len(cameras)}, Точек: {len(points_3d)}")
   
   # Подготовка данных для оптимизации (аналогично стандартному bundle adjustment)
   camera_ids = list(cameras.keys())
   point_ids = list(points_3d.keys())
   
   camera_id_to_idx = {cam_id: idx for idx, cam_id in enumerate(camera_ids)}
   point_id_to_idx = {pt_id: idx for idx, pt_id in enumerate(point_ids)}
   
   # Создаем списки наблюдений
   obs_camera_idxs = []
   obs_point_idxs = []
   obs_points_2d = []
   
   for cam_id, points in camera_points.items():
       if cam_id not in camera_id_to_idx:
           continue
           
       cam_idx = camera_id_to_idx[cam_id]
       for point_id, point_2d in points.items():
           if point_id not in point_id_to_idx or point_id not in points_3d:
               continue
               
           obs_camera_idxs.append(cam_idx)
           obs_point_idxs.append(point_id_to_idx[point_id])
           obs_points_2d.append(point_2d)
   
   if not obs_camera_idxs:
       print("Нет наблюдений для оптимизации")
       return points_3d, cameras
       
   obs_camera_idxs = np.array(obs_camera_idxs)
   obs_point_idxs = np.array(obs_point_idxs)
   obs_points_2d = np.array(obs_points_2d)
   
   n_cameras = len(cameras)
   n_points = len(points_3d)
   n_observations = len(obs_points_2d)
   
   print(f"Подготовлено {n_observations} наблюдений для оптимизации")
   
   # Инициализируем параметры камер и точек
   camera_params = np.zeros((n_cameras, 6))  # [rot_vec(3), trans_vec(3)]
   point_params = np.zeros((n_points, 3))   # [X, Y, Z]
   
   # Заполняем начальные параметры камер
   for i, cam_id in enumerate(camera_ids):
       R, t = cameras[cam_id]
       rvec, _ = cv2.Rodrigues(R)
       camera_params[i, :3] = rvec.flatten()
       camera_params[i, 3:] = t.flatten()
   
   # Заполняем начальные параметры точек
   for i, point_id in enumerate(point_ids):
       point_params[i] = points_3d[point_id]
   
   # Подготавливаем начальный вектор параметров
   x0 = np.hstack([
       camera_params.flatten(),
       point_params.flatten()
   ])
   
   # 1. Грубая оптимизация только с участием первых нескольких камер
   if n_cameras > 2:
       print("1. Выполняем грубую оптимизацию с подмножеством камер...")
       # Используем только первые 2-3 камеры для начальной оценки
       subset_cameras = min(3, n_cameras)
       subset_obs_mask = np.isin(obs_camera_idxs, range(subset_cameras))
       
       if np.sum(subset_obs_mask) > 10:  # Проверяем, достаточно ли наблюдений
           subset_camera_idxs = obs_camera_idxs[subset_obs_mask]
           subset_point_idxs = obs_point_idxs[subset_obs_mask]
           subset_points_2d = obs_points_2d[subset_obs_mask]
           
           def subset_objective(params):
               return robust_reprojection_error(
                   params, subset_camera_idxs, subset_point_idxs, subset_points_2d,
                   subset_cameras, n_points, K, dist_coeffs, 'huber', {'delta': 2.0}
               )
           
           try:
               # Оптимизация с грубой настройкой
               result_coarse = least_squares(
                   subset_objective,
                   x0[:subset_cameras*6 + n_points*3],  # Только параметры выбранных камер и всех точек
                   verbose=0,
                   x_scale='jac',
                   ftol=1e-6,
                   method='trf',
                   max_nfev=100
               )
               
               if result_coarse.success:
                   # Обновляем начальные параметры для полной оптимизации
                   updated_params = x0.copy()
                   updated_params[:len(result_coarse.x)] = result_coarse.x
                   x0 = updated_params
                   print(f"  - Грубая оптимизация завершена, ошибка: {result_coarse.cost:.6f}")
               else:
                   print("  - Грубая оптимизация не удалась, продолжаем с начальными параметрами")
           except Exception as e:
               print(f"  - Ошибка при грубой оптимизации: {str(e)}")
   
   # 2. Оптимизация с робастной loss функцией
   print("2. Выполняем робастную оптимизацию с Huber loss...")
   
   def robust_objective(params):
       return robust_reprojection_error(
           params, obs_camera_idxs, obs_points_2d,
           n_cameras, n_points, K, dist_coeffs, 'huber', {'delta': 1.345}
       )
   
   try:
       # Выполняем несколько итераций с разными параметрами
       current_params = x0
       for iteration in range(initial_refinements):
           print(f" - Итерация {iteration + 1}/{initial_refinements}")
           
           result = least_squares(
               robust_objective,
               current_params,
               verbose=0,
               x_scale='jac',
               ftol=1e-8,
               method='trf',
               max_nfev=10
           )
           
           if result.success:
               current_params = result.x
               print(f"    - Ошибка после итерации {iteration + 1}: {result.cost:.6f}")
           else:
               print(f"    - Итерация {iteration + 1} не удалась: {result.message}")
               break
   except Exception as e:
       print(f"Ошибка при робастной оптимизации: {str(e)}")
       current_params = x0  # Возвращаемся к начальным параметрам
   
   # 3. Финальная точная оптимизация
   print("3. Выполняем финальную точную оптимизацию...")
   
   def final_objective(params):
       return robust_reprojection_error(
           params, obs_camera_idxs, obs_point_idxs, obs_points_2d,
           n_cameras, n_points, K, dist_coeffs, 'tukey', {'c': 4.6851}
       )
   
   try:
       final_results = []
       methods = ['trf', 'dogbox']
       
       for method in methods:
           try:
               result = least_squares(
                   final_objective,
                   current_params,
                   verbose=0,
                   x_scale='jac',
                   ftol=1e-10,
                   method=method,
                   max_nfev=final_refinements * 50
               )
               
               if result.success:
                   final_results.append((result, method))
                   print(f"  - Метод {method}: ошибка={result.cost:.6f}, итераций={result.nfev}")
               else:
                   print(f"  - Метод {method} не сходится: {result.message}")
           except Exception as e:
               print(f"  - Ошибка при использовании метода {method}: {str(e)}")
       
       # Выбираем лучший результат
       if final_results:
           best_result, best_method = min(final_results, key=lambda x: x[0].cost)
           print(f"  - Лучший метод: {best_method}, ошибка: {best_result.cost:.6f}")
           final_params = best_result.x
       else:
           print("  - Все методы оптимизации не сработали, используем параметры после робастной оптимизации")
           final_params = current_params
           
   except Exception as e:
       print(f"Ошибка при финальной оптимизации: {str(e)}")
       final_params = current_params
   
   # Извлекаем оптимизированные параметры
   camera_params_opt = final_params[:n_cameras*6].reshape((n_cameras, 6))
   point_params_opt = final_params[n_cameras*6:].reshape((n_points, 3))
   
   # Обновляем камеры
   cameras_refined = {}
   for i, cam_id in enumerate(camera_ids):
       rvec = camera_params_opt[i, :3]
       tvec = camera_params_opt[i, 3:]
       
       # Преобразуем вектор поворота в матрицу
       R, _ = cv2.Rodrigues(rvec)
       
       # Проверяем ортогональность матрицы поворота
       if not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
           print(f"Предупреждение: матрица поворота для камеры {cam_id} не ортогональна")
           # Пытаемся исправить матрицу поворота
           U, _, Vt = np.linalg.svd(R)
           R = U @ Vt  # Убедиться, что это ортогональная матрица
       
       # Проверяем, что камера не слишком далеко
       translation_norm = np.linalg.norm(tvec)
       if translation_norm > 1000:
           print(f"Предупреждение: камера {cam_id} имеет слишком большой вектор перемещения: {translation_norm:.2f}")
           # Ограничиваем вектор перемещения
           tvec = tvec / translation_norm * 1000
       
       cameras_refined[cam_id] = (R, tvec.reshape(3, 1))
   
   # Обновляем 3D точки
   points_3d_refined = {}
   for i, point_id in enumerate(point_ids):
       point_3d = point_params_opt[i]
       
       # Проверяем, что точка не имеет аномальных значений
       if np.any(np.abs(point_3d) > 10000):
           print(f"Предупреждение: точка {point_id} имеет аномальные координаты: {point_3d}")
           # Используем исходное значение точки
           points_3d_refined[point_id] = points_3d[point_id]
       else:
           points_3d_refined[point_id] = point_3d
   
   print(f"Иерархический bundle adjustment завершен")
   print(f"Оптимизировано {len(points_3d_refined)} точек и {len(cameras_refined)} камер")
   
   return points_3d_refined, cameras_refined
