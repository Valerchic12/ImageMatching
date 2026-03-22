"""
Функции для оценки и оптимизации положений камер.
"""
import numpy as np
import cv2
import traceback
from typing import List, Tuple, Dict, Optional, Union
import logging
from scipy.spatial.transform import Rotation as R

try:
    from . import utils
except ImportError:  # pragma: no cover - fallback for direct module execution
    import utils


def _stable_sort_key(value):
    text = str(value)
    try:
        return (0, int(text))
    except (TypeError, ValueError):
        return (1, text)

def check_pose_consistency(R, t):
    """
    Проверяет корректность позы камеры.

    Args:
        R: Матрица поворота 3x3
        t: Вектор переноса 3x1

    Returns:
        bool: True если поза корректна, False в противном случае
    """
    try:
        # Проверяем, что R - это 3x3 матрица
        if R.shape != (3, 3):
            print(f"Некорректная форма матрицы R: {R.shape}, ожидалось (3, 3)")
            return False

        # Проверяем определитель (должен быть близок к 1 для матрицы поворота)
        det = np.linalg.det(R)
        if not 0.99 < det < 1.01:
            print(f"Некорректный определитель: {det:.3f}, ожидалось ~1.0")
            return False

        # Проверяем ортогональность (R * R.T должно быть близко к единичной матрице)
        orth = R @ R.T
        if not np.allclose(orth, np.eye(3), atol=1e-5):
            print("Матрица не ортогональная")
            return False

        # Проверяем, что t - это 3x1 или (3,) вектор
        t = np.asarray(t)
        if t.shape not in [(3,), (3, 1)]:
            print(f"Некорректная форма вектора t: {t.shape}, ожидалось (3,) или (3,1)")
            return False

        # Проверяем, что вектор переноса не слишком большой (для устойчивости)
        translation_magnitude = np.linalg.norm(t)
        if translation_magnitude > 1000:
            print(f"Слишком большой вектор переноса: {translation_magnitude:.2f}")
            return False

        # Проверяем угол поворота
        trace_val = np.clip(np.trace(R), -1.0, 3.0)  # Ограничиваем значение для arccos
        angle = np.arccos(np.clip((trace_val - 1) / 2, -1.0, 1.0))
        angle_deg = np.degrees(angle)
        if angle_deg > 180:
            print(f"Слишком большой угол поворота: {angle_deg:.2f}°")
            return False

        return True
    except Exception as e:
        print(f"Ошибка при проверке согласованности позы: {str(e)}")
        return False


def _as_points_nx3(points_3d):
    points_3d = np.asarray(points_3d, dtype=np.float64)
    if points_3d.ndim == 1 and points_3d.size >= 3:
        return points_3d[:3].reshape(1, 3)
    if points_3d.ndim == 2 and points_3d.shape[0] == 3 and points_3d.shape[1] != 3:
        points_3d = points_3d.T
    if points_3d.ndim != 2 or points_3d.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(points_3d[:, :3], dtype=np.float64)


def _compute_front_ratio(points_3d, R, t, min_z=0.01):
    points_3d_nx3 = _as_points_nx3(points_3d)
    if points_3d_nx3.size == 0:
        return 0.0
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t_col = utils.normalize_translation(t, dtype=np.float64)
    transformed_points = (R @ points_3d_nx3.T) + t_col
    front_mask = transformed_points[2, :] > float(min_z)
    return float(np.mean(front_mask)) if front_mask.size > 0 else 0.0


def _iter_essential_matrices(E):
    if E is None:
        return
    E = np.asarray(E, dtype=np.float64)
    if E.shape == (3, 3):
        yield E
        return
    if E.ndim == 2 and E.shape[1] == 3 and E.shape[0] % 3 == 0:
        for row_offset in range(0, E.shape[0], 3):
            yield E[row_offset:row_offset + 3, :]


def _evaluate_relative_pose_solution(R_candidate, t_candidate, pts1_inliers, pts2_inliers, inlier_indices, total_point_count):
    if not check_pose_consistency(R_candidate, t_candidate):
        return None

    t_col = utils.normalize_translation(t_candidate, dtype=np.float64)
    P1 = np.hstack((np.eye(3), np.zeros((3, 1), dtype=np.float64)))
    P2 = np.hstack((np.asarray(R_candidate, dtype=np.float64).reshape(3, 3), t_col))

    try:
        points_4d = cv2.triangulatePoints(P1, P2, pts1_inliers, pts2_inliers)
        points_3d = points_4d[:3, :] / points_4d[3, :]
    except cv2.error:
        return None

    if points_3d.shape[1] < 8:
        return None

    z1 = points_3d[2, :]
    z2 = (np.asarray(R_candidate, dtype=np.float64).reshape(3, 3) @ points_3d + t_col)[2, :]
    front_mask = (z1 > 0.01) & (z2 > 0.01)
    finite_mask = np.all(np.isfinite(points_3d), axis=0)
    bounded_mask = np.all(np.abs(points_3d) <= 1000.0, axis=0)
    valid_mask = front_mask & finite_mask & bounded_mask
    if np.count_nonzero(valid_mask) < 8:
        return None

    filtered_points_3d = points_3d[:, valid_mask]
    filtered_indices = inlier_indices[valid_mask]
    front_ratio1 = float(np.mean(z1[valid_mask] > 0.01))
    front_ratio2 = float(np.mean(z2[valid_mask] > 0.01))
    point_distances = np.linalg.norm(filtered_points_3d.T, axis=1)
    mean_distance = float(np.mean(point_distances)) if point_distances.size > 0 else float("inf")
    if not np.isfinite(mean_distance) or mean_distance > 1000.0:
        return None

    full_mask = np.zeros((int(total_point_count), 1), dtype=np.uint8)
    full_mask[filtered_indices] = 1
    candidate_score = (
        int(np.count_nonzero(valid_mask)),
        int(filtered_points_3d.shape[1]),
        float(front_ratio1 + front_ratio2),
    )
    return {
        'result': (
            np.asarray(R_candidate, dtype=np.float64).reshape(3, 3),
            t_col.astype(np.float64),
            full_mask,
            filtered_points_3d,
        ),
        'score': candidate_score,
    }

def check_points_in_front(points_3d, R, t):
    """
    Проверяет, находятся ли точки перед камерой (положительная Z-координата в системе координат камеры)
    
    Args:
        points_3d: 3D точки в мировой системе координат
        R: матрица поворота камеры
        t: вектор переноса камеры
    
    Returns:
        bool или ndarray: True/False для одной точки, массив bool для нескольких точек
    """
    try:
        points_3d_nx3 = _as_points_nx3(points_3d)
        if points_3d_nx3.size == 0:
            return False
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        t_col = utils.normalize_translation(t, dtype=np.float64)
        points_cam = (R @ points_3d_nx3.T).T + t_col.ravel()
        front_mask = points_cam[:, 2] > 0.01
        if points_3d_nx3.shape[0] == 1:
            return bool(front_mask[0])
        return front_mask

    except Exception as e:
        print(f"Ошибка при преобразовании точек в систему координат камеры: {str(e)}")
        traceback.print_exc()
        if np.asarray(points_3d).ndim == 1:
            return False
        else:
            return np.zeros(_as_points_nx3(points_3d).shape[0], dtype=bool)

def _recover_relative_pose_candidate(pts1_norm, pts2_norm, E, mask, total_point_count):
    if E is None or mask is None:
        return None

    mask = np.asarray(mask, dtype=np.uint8).reshape(-1, 1)
    inliers = int(np.sum(mask > 0))
    if inliers < 8:
        return None

    inlier_indices = np.where(mask.ravel() > 0)[0]
    if len(inlier_indices) == 0:
        print("Нет инлаеров для триангуляции")
        return None

    pts1_inliers = pts1_norm[inlier_indices].T
    pts2_inliers = pts2_norm[inlier_indices].T

    if pts1_inliers.shape[0] != 2 or pts2_inliers.shape[0] != 2:
        print(f"Неверный формат точек для триангуляции: {pts1_inliers.shape}, {pts2_inliers.shape}")
        if pts1_inliers.shape[0] != 2:
            pts1_inliers = pts1_inliers.T
        if pts2_inliers.shape[0] != 2:
            pts2_inliers = pts2_inliers.T

    pts1_inliers = np.ascontiguousarray(pts1_inliers, dtype=np.float32)
    pts2_inliers = np.ascontiguousarray(pts2_inliers, dtype=np.float32)

    best_candidate = None
    best_score = None
    for essential_matrix in _iter_essential_matrices(E):
        try:
            R1, R2, t = cv2.decomposeEssentialMat(np.asarray(essential_matrix, dtype=np.float64))
        except cv2.error:
            continue
        for R_candidate in (R1, R2):
            for t_candidate in (t, -t):
                candidate = _evaluate_relative_pose_solution(
                    R_candidate,
                    t_candidate,
                    pts1_inliers,
                    pts2_inliers,
                    inlier_indices,
                    total_point_count,
                )
                if candidate is None:
                    continue
                if best_candidate is None or candidate['score'] > best_score:
                    best_candidate = candidate
                    best_score = candidate['score']

    if best_candidate is None:
        print("Не удалось выбрать корректную позу из 4-кратной декомпозиции Essential")
        return None

    return best_candidate


def estimate_relative_pose(pts1, pts2, K, dist_coeffs=None):
    """
    Оценивает относительное положение камеры на основе соответствий точек

    Args:
        pts1, pts2: соответствующие точки на двух изображениях
        K: матрица внутренних параметров камеры
        dist_coeffs: коэффициенты дисторсии (опционально)

    Returns:
        tuple: (R, t, mask, points_3d) или None в случае ошибки
    """
    if len(pts1) < 8 or len(pts2) < 8:
        print(f"Недостаточно точек для оценки позы: {len(pts1)} (минимум 8)")
        return None

    try:
        if K is None:
            print("Матрица K равна None, создаем стандартную матрицу калибровки")
            K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=np.float32)
        elif K.shape != (3, 3):
            print(f"Некорректная форма матрицы K: {K.shape}, ожидалось (3, 3)")
            return None

        pts1_norm = cv2.undistortPoints(
            pts1.reshape(-1, 1, 2),
            K,
            dist_coeffs
        ).reshape(-1, 2)
        pts2_norm = cv2.undistortPoints(
            pts2.reshape(-1, 1, 2),
            K,
            dist_coeffs
        ).reshape(-1, 2)

        focal_scale = max(float(K[0, 0]), float(K[1, 1]), 1.0)
        essential_threshold = max(1e-4, min(0.01, 2.0 / focal_scale))

        essential_methods = [('RANSAC', cv2.RANSAC)]
        usac_magsac = getattr(cv2, 'USAC_MAGSAC', None)
        if usac_magsac is not None and usac_magsac != cv2.RANSAC:
            essential_methods.append(('USAC_MAGSAC', usac_magsac))

        best_candidate = None
        best_score = None

        for method_name, essential_method in essential_methods:
            try:
                E, mask = cv2.findEssentialMat(
                    pts1_norm,
                    pts2_norm,
                    np.eye(3),
                    essential_method,
                    0.999,
                    essential_threshold,
                    maxIters=5000
                )
            except cv2.error:
                continue

            candidate = _recover_relative_pose_candidate(
                pts1_norm,
                pts2_norm,
                E,
                mask,
                total_point_count=len(pts1),
            )
            if candidate is None:
                continue

            candidate_score = candidate['score']
            if best_candidate is None or candidate_score > best_score:
                best_candidate = candidate
                best_score = candidate_score
                best_candidate['method'] = method_name

        if best_candidate is None:
            print("Не удалось найти устойчивую существенную матрицу")
            return None

        if best_candidate.get('method') == 'USAC_MAGSAC':
            print("Relative pose: выбран USAC_MAGSAC для этой пары")

        return best_candidate['result']

    except Exception as e:
        print(f"Ошибка при оценке относительной позы: {str(e)}")
        traceback.print_exc()
        return None

def find_best_initial_pair(camera_points, K, dist_coeffs=None):
    """
    Находит лучшую начальную пару камер для реконструкции

    Args:
        camera_points: Словарь {camera_id: {point_id: point_2d}} с точками для каждой камеры
        K: Матрица калибровки камеры
        dist_coeffs: Коэффициенты дисторсии (может быть None)

    Returns:
        tuple: (cam_id1, cam_id2, R, t, points_3d) или None в случае ошибки
    """
    from calibration_modules.utils import estimate_point_coverage
    
    print("\nПоиск лучшей начальной пары камер...")
    print(f"Количество камер: {len(camera_points)}")
    print(f"Матрица калибровки:\n{K}")
    if dist_coeffs is not None:
        print(f"Коэффициенты дисторсии: {dist_coeffs}")
    else:
        print("Коэфициенты дисторсии не заданы")

    best_score = -1
    best_pair = None

    # Перебираем все возможные пары камер
    camera_ids = sorted(camera_points.keys(), key=_stable_sort_key)
    for i, cam_id1 in enumerate(camera_ids[:-1]):
        points1 = camera_points[cam_id1]
        print(f"\nКамера {cam_id1}: {len(points1)} точек")

        for cam_id2 in camera_ids[i + 1:]:
            points2 = camera_points[cam_id2]
            print(f"Камера {cam_id2}: {len(points2)} точек")

            # Находим общие точки
            common_points = sorted(set(points1.keys()) & set(points2.keys()), key=_stable_sort_key)
            if len(common_points) < 8:  # Минимум 8 точек для essential matrix
                print(f"  - Недостаточно общих точек: {len(common_points)} (минимум 8)")
                continue

            print(f"\nПара камер {cam_id1}-{cam_id2}:")
            print(f"  - общих точек: {len(common_points)}")

            # Собираем точки в массивы
            pts1 = []
            pts2 = []
            point_ids = []
            for point_id in common_points:
                pts1.append(points1[point_id])
                pts2.append(points2[point_id])
                point_ids.append(point_id)
            pts1 = np.array(pts1)
            pts2 = np.array(pts2)

            # Оцениваем относительную позу
            print(f"  - Оценка относительной позы...")
            result = estimate_relative_pose(pts1, pts2, K, dist_coeffs)
            if result is None:
                print(f"  - Не удалось оценить относительную позу")
                continue

            R, t, mask, points_3d = result
            
            # Считаем количество инлаеров
            inliers = np.sum(mask) if mask is not None else 0
            
            # Считаем покрытие (процент точек от максимума)
            max_points = max(len(points1), len(points2))
            coverage = len(common_points) / max_points if max_points > 0 else 0

            # Считаем угол между камерами
            angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))

            # Считаем базовую линию
            baseline = np.linalg.norm(t)

            # Считаем количество валидных точек
            valid_points = 0
            if points_3d is not None and points_3d.shape[1] > 0:
                for i in range(points_3d.shape[1]):
                    point_3d = points_3d[:, i]
                    if check_points_in_front(
                            point_3d,
                            np.eye(3, 4),
                            np.hstack((R, t.reshape(3, 1)))
                    ):
                        valid_points += 1

            print(f"  - инлаеров: {inliers}/{len(pts1)} ({inliers / len(pts1):.1%})")
            print(f"  - покрытие: {coverage:.1%}")
            print(f"  - угол: {angle:.1f}°")
            print(f"  - базовая линия: {baseline:.2f}")
            print(f"  - валидных точек: {valid_points}")

            # Считаем общий скор
            inlier_ratio = inliers / len(pts1) if len(pts1) > 0 else 0
            angle_score = np.sin(np.radians(angle)) if angle is not None else 0  # максимум при 90°
            baseline_score = min(baseline, 1.0) if baseline is not None else 0  # ограничиваем сверху

            score = (
                    0.4 * inlier_ratio +  # вес инлаеров
                    0.3 * coverage +  # вес покрытия
                    0.2 * angle_score +  # вес угла
                    0.1 * baseline_score  # вес базовой линии
            )

            print(f"  - скор: {score:.3f}")

            # Проверяем, что есть хотя бы одна валидная точка
            if valid_points > 0:
                # Проверяем ошибку репроекции для первой валидной точки
                reprojection_error_checked = False
                if points_3d is not None and points_3d.shape[1] > 0:
                    for i in range(points_3d.shape[1]):
                        point_3d = points_3d[:, i]
                        if check_points_in_front(
                                point_3d,
                                np.eye(3, 4),
                                np.hstack((R, t.reshape(3, 1)))
                        ):
                            # Находим индекс точки в исходном массиве
                            if mask is not None:
                                mask_indices = np.where(mask.ravel() > 0)[0]
                                if i < len(mask_indices):
                                    point_idx = mask_indices[i]
                                    if point_idx < len(pts1):
                                        print(f"  - Проверка ошибки репроекции для точки {i}...")
                                        if check_reprojection_error(
                                                point_3d,
                                                pts1[point_idx],
                                                pts2[point_idx],
                                                np.eye(3, 4),
                                                np.hstack((R, t.reshape(3, 1))),
                                                K,
                                                dist_coeffs
                                        ):
                                            reprojection_error_checked = True
                                            if score > best_score:
                                                print("  -> новая лучшая пара")
                                                best_score = score

                                                # Создаем словарь с 3D точками
                                                points_dict = {}
                                                for j, point_id in enumerate(point_ids):
                                                    if mask is not None and j < len(mask) and mask[j]:
                                                        # Находим индекс точки в массиве points_3d
                                                        if j < points_3d.shape[1]:
                                                            points_dict[point_id] = points_3d[:, j]

                                                best_pair = (cam_id1, cam_id2, R, t, points_dict)
                                            break
                
                if not reprojection_error_checked:
                    print(f"  - Не удалось проверить ошибку репроекции ни для одной точки")

    if best_pair is None:
        print("Не удалось найти подходящую пару камер")
    else:
        print(f"\nЛучшая пара камер: {best_pair[0]}-{best_pair[1]}")
        print(f"- Скор: {best_score:.3f}")
        print(f"- 3D точек: {len(best_pair[4])}")
        
    return best_pair

def check_point_in_front_of_cameras(point_3d, P1, P2):
    """
    Проверяет, что 3D точка находится перед обеими камерами.

    Args:
        point_3d: 3D точка (3,)
        P1, P2: Матрицы проекции камер (3x4)

    Returns:
        bool: True, если точка перед обеими камерами
    """
    # Проверяем z-координату в системе координат первой камеры
    if point_3d[2] <= 0:
        return False

    # Преобразуем точку в систему координат второй камеры
    R = P2[:, :3]
    t = P2[:, 3]
    point_cam2 = R @ point_3d + t

    # Проверяем z-координату в системе координат второй камеры
    return point_cam2[2] > 0

def check_reprojection_error(point_3d, point_2d_1, point_2d_2=None, P1=None, P2=None, K=None, dist_coeffs=None, threshold=10.0):
    """
    Проверяет ошибку репроекции для 3D точки.

    Args:
        point_3d: 3D точка
        point_2d_1: 2D точка на первом изображении
        point_2d_2: 2D точка на втором изображении (если есть)
        P1, P2: Матрицы проекции камер (если есть)
        K: Матрица калибровки камеры (если есть)
        dist_coeffs: Коэффициенты дисторсии (если есть)
        threshold: Пороговое значение ошибки репроекции в пикселях

    Returns:
        bool: True, если ошибка репроекции меньше порога
    """
    try:
        print(f"    Проверка ошибки репроекции для точки {point_3d}")
        print(f"    - 2D точка 1: {point_2d_1}")
        if point_2d_2 is not None:
            print(f"    - 2D точка 2: {point_2d_2}")
        
        if P1 is None:
            P1 = np.eye(3, 4)
            
        if K is None:
            # Если матрица калибровки не задана, используем единичную матрицу
            K = np.eye(3)
            
        # Проецируем 3D точку на первое изображение
        point_3d_homogeneous = np.append(point_3d, 1)
        point_projected_1 = K @ (P1 @ point_3d_homogeneous)
        point_projected_1 = point_projected_1[:2] / point_projected_1[2]
        
        # Вычисляем ошибку репроекции для первого изображения
        error_1 = np.linalg.norm(point_projected_1 - point_2d_1)
        print(f"    - Проекция на первое изображение: {point_projected_1}")
        print(f"    - Ошибка репроекции для первого изображения: {error_1:.2f} пикселей")
        
        # Если есть второе изображение, проверяем и его
        if point_2d_2 is not None and P2 is not None:
            # Проецируем 3D точку на второе изображение
            point_projected_2 = K @ (P2 @ point_3d_homogeneous)
            point_projected_2 = point_projected_2[:2] / point_projected_2[2]
            
            # Вычисляем ошибку репроекции для второго изображения
            error_2 = np.linalg.norm(point_projected_2 - point_2d_2)
            print(f"    - Проекция на второе изображение: {point_projected_2}")
            print(f"    - Ошибка репроекции для второго изображения: {error_2:.2f} пикселей")
            
            # Вычисляем среднюю ошибку репроекции
            error = (error_1 + error_2) / 2
        else:
            error = error_1
            
        print(f"    - Итоговая ошибка репроекции: {error:.2f} пикселей (порог: {threshold:.2f})")
        
        # Проверяем, что ошибка репроекции меньше порога
        if error < threshold:
            print(f"    - Ошибка репроекции в пределах допустимого")
            return True
        else:
            print(f"    - Ошибка репроекции превышает порог")
            return False
            
    except Exception as e:
        print(f"    - Ошибка при проверке ошибки репроекции: {str(e)}")
        traceback.print_exc()
        return False

def compute_angle_between_cameras(R1, t1, R2, t2):
    """
    Вычисляет угол между оптическими осями двух камер.

    Args:
        R1, R2: Матрицы поворота камер (3x3)
        t1, t2: Векторы переноса камер (3x1 или 3,)

    Returns:
        float: Угол в градусах
    """
    # Направление оптической оси каждой камеры (ось Z в локальных координатах)
    axis1 = R1[:, 2]  # Третий столбец матрицы поворота
    axis2 = R2[:, 2]  # Третий столбец матрицы поворота

    # Нормализуем векторы
    axis1 = axis1 / np.linalg.norm(axis1)
    axis2 = axis2 / np.linalg.norm(axis2)

    # Вычисляем угол между векторами
    cos_angle = np.clip(np.dot(axis1, axis2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def count_points_in_front(points_3d, R, t):
    """
    Подсчитывает количество точек, находящихся перед камерой.
    
    Args:
        points_3d: 3D точки в мировой системе координат
        R: матрица поворота камеры
        t: вектор переноса камеры
    
    Returns:
        int: количество точек перед камерой
    """
    try:
        points_3d_nx3 = _as_points_nx3(points_3d)
        if points_3d_nx3.size == 0:
            return 0
        return int(np.round(_compute_front_ratio(points_3d_nx3, R, t, min_z=0.0) * points_3d_nx3.shape[0]))

    except Exception as e:
        print(f"Ошибка при подсчете точек перед камерой: {str(e)}")
        traceback.print_exc()
        return 0

def improved_pose_consistency_check(R: np.ndarray, t: np.ndarray, points_3d: np.ndarray, threshold: float = 0.7) -> bool:
    """
    Проверяет, что достаточное количество точек находится перед обеими камерами
    """
    if points_3d.size == 0:
        return False
    
    front_ratio1 = _compute_front_ratio(points_3d, np.eye(3), np.zeros((3, 1)), min_z=0.0)
    front_ratio2 = _compute_front_ratio(points_3d, R, t, min_z=0.0)

    return front_ratio1 > threshold and front_ratio2 > threshold

def check_cheirality_condition(R: np.ndarray, t: np.ndarray, points_3d: np.ndarray, min_front_ratio: float = 0.6) -> bool:
    """
    Проверяет условие чиральности (cheirality condition) для оценки позы
    """
    if points_3d.size == 0:
        return False
    
    front_ratio1 = _compute_front_ratio(points_3d, np.eye(3), np.zeros((3, 1)), min_z=0.0)
    front_ratio2 = _compute_front_ratio(points_3d, R, t, min_z=0.0)
    
    return front_ratio1 >= min_front_ratio and front_ratio2 >= min_front_ratio
