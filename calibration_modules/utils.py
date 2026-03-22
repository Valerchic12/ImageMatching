"""
Вспомогательные функции и утилиты для калибровки камеры.
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Union
import logging
from dataclasses import dataclass
import sys
import os

# Logger setup
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import functions from main utils module using qualified paths to avoid circular imports
_main_utils = None
try:
    # First, try to import from parent directory explicitly by name
    import importlib.util
    utils_path = os.path.join(parent_dir, 'utils.py')
    if os.path.exists(utils_path):
        spec = importlib.util.spec_from_file_location("main_utils", utils_path)
        _main_utils = importlib.util.module_from_spec(spec)
        sys.modules['main_utils'] = _main_utils
        spec.loader.exec_module(_main_utils)
        logger.info("Successfully imported utility functions from main utils module")
    else:
        logger.error(f"Main utils.py not found at {utils_path}")
except Exception as e:
    logger.error(f"Failed to import utility functions from main utils: {e}")
    logger.debug("Some functions may not be available")

# Create fallback imports if _main_utils failed
if _main_utils is None:
    # Define dummy functions to prevent AttributeError
    def _dummy_func(*args, **kwargs):
        logger.warning("Function not available - main utils failed to import")
        return None
    
    _main_utils = type('module', (), {
        'invert_y_coordinate': _dummy_func,
        'normalize_points': _dummy_func,
        'normalize_points_for_calibration': _dummy_func,
        'project_point': _dummy_func,
        'blender_to_opencv_points': _dummy_func,
        'opencv_to_blender_points': _dummy_func,
        'check_camera_data_format': _dummy_func,
        'estimate_point_coverage': _dummy_func,
        'compute_angle_between_cameras': _dummy_func,
        'validate_camera_pose': _dummy_func,
        'check_points_in_front': _dummy_func,
        'normalize_rotation_matrix': _dummy_func,
        'validate_matrix_shape': _dummy_func,
        'ensure_valid_camera_data': _dummy_func,
        'compute_reprojection_error': _dummy_func,
        'calculate_reprojection_errors_detailed': _dummy_func,
    })()

# Re-export functions from main utils for backward compatibility
invert_y_coordinate = _main_utils.invert_y_coordinate
normalize_points = _main_utils.normalize_points
normalize_points_for_calibration = _main_utils.normalize_points_for_calibration
project_point = _main_utils.project_point
blender_to_opencv_points = _main_utils.blender_to_opencv_points
opencv_to_blender_points = _main_utils.opencv_to_blender_points
check_camera_data_format = _main_utils.check_camera_data_format
estimate_point_coverage = _main_utils.estimate_point_coverage
compute_angle_between_cameras = _main_utils.compute_angle_between_cameras
validate_camera_pose = _main_utils.validate_camera_pose
check_points_in_front = _main_utils.check_points_in_front
normalize_rotation_matrix = _main_utils.normalize_rotation_matrix
validate_matrix_shape = _main_utils.validate_matrix_shape
ensure_valid_camera_data = _main_utils.ensure_valid_camera_data
compute_reprojection_error = _main_utils.compute_reprojection_error
calculate_reprojection_errors_detailed = _main_utils.calculate_reprojection_errors_detailed


def normalize_translation(t, dtype=np.float64):
    """
    Нормализует вектор переноса в формат (3, 1).

    Функция остаётся совместимой со старыми форматами, которые уже встречаются
    в кодовой базе: (3,), (3,1), (1,3), (3,3)/(9,), (3,4)/(12,), (3,1,3).
    """
    t = np.asarray(t, dtype=dtype)

    if t.shape == (3, 1):
        return np.array(t, copy=True)
    if t.shape == (1, 3):
        return np.array(t.reshape(3, 1), copy=True)
    if t.shape == (3,):
        return np.array(t.reshape(3, 1), copy=True)

    if t.shape == (3, 1, 3):
        t = t.reshape(3, 3)[:, 2]
        return np.array(np.asarray(t, dtype=dtype).reshape(3, 1), copy=True)

    if t.shape == (3, 3) or t.size == 9:
        t = np.asarray(t, dtype=dtype).reshape(3, 3)[:, 2]
        return np.array(t.reshape(3, 1), copy=True)

    if t.shape == (3, 4) or t.size == 12:
        t = np.asarray(t, dtype=dtype).reshape(3, 4)[:, 3]
        return np.array(t.reshape(3, 1), copy=True)

    if t.size == 3:
        return np.array(np.asarray(t, dtype=dtype).reshape(3, 1), copy=True)

    if t.size == 1:
        scalar = float(np.asarray(t, dtype=dtype).reshape(-1)[0])
        return np.array([[scalar], [0.0], [0.0]], dtype=dtype)

    raise ValueError(f"Unexpected translation shape: {t.shape}, size={t.size}")

# ===== DATACLASSES AND SUPPORTING TYPES =====

def check_3d_point_collinearity(points_3d, distance_threshold=0.1, angle_threshold=0.1):
    """
    Проверяет коллинеарность 3D точек с использованием более надежного подхода.
    Коллинеарные точки лежат на одной прямой в 3D пространстве.
    
    Args:
        points_3d: Список или массив 3D точек
        distance_threshold: Порог расстояния от точки до линии (в 3D пространстве)
        angle_threshold: Порог угла между векторами (в радианах)
        
    Returns:
        bool: True, если точки коллинеарны
    """
    if len(points_3d) < 3:
        return False
    
    points_3d = np.array(points_3d, dtype=np.float32)
    
    # Проверяем, что все точки не совпадают
    if np.allclose(points_3d[0], points_3d[1]):
        # Если первые две точки совпадают, ищем первую отличающуюся точку
        base_idx = 0
        direction_idx = 1
        for i in range(2, len(points_3d)):
            if not np.allclose(points_3d[base_idx], points_3d[i]):
                direction_idx = i
                break
        else:
            # Все точки совпадают - считаем их коллинеарными
            return True
    else:
        base_idx = 0
        direction_idx = 1
    
    # Вектор направления линии
    direction_vector = points_3d[direction_idx] - points_3d[base_idx]
    direction_norm = np.linalg.norm(direction_vector)
    
    if direction_norm < 1e-6:
        # Направляющий вектор слишком мал - все точки в одной точке
        return True
    
    # Нормализуем вектор направления
    direction_unit = direction_vector / direction_norm
    
    # Проверяем, лежат ли все остальные точки близко к линии
    for i, point in enumerate(points_3d):
        if i == base_idx or i == direction_idx:
            continue
            
        # Вектор от базовой точки к текущей точке
        to_point = point - points_3d[base_idx]
        
        # Вычисляем расстояние от точки до линии
        # Формула: ||(A-P) × (A-B)|| / ||A-B||, где A и B - точки на линии, P - проверяемая точка
        cross_product = np.cross(to_point, direction_unit)
        distance_to_line = np.linalg.norm(cross_product)
        
        if distance_to_line > distance_threshold:
            return False
    
    return True

def check_3d_point_planarity(points_3d, distance_threshold=0.1, use_ransac=True, ransac_iterations=100, ransac_confidence=0.95):
    """
    Проверяет, лежат ли 3D точки в одной плоскости.
    
    Args:
        points_3d: Список или массив 3D точек
        distance_threshold: Порог расстояния от точки до плоскости (в 3D пространстве)
        use_ransac: Использовать ли RANSAC для устойчивой оценки планарности
        ransac_iterations: Количество итераций RANSAC
        ransac_confidence: Уровень доверия RANSAC
        
    Returns:
        tuple: (is_planar, inlier_count, outlier_count, inlier_ratio, plane_params)
            - is_planar: True, если точки лежат в одной плоскости
            - inlier_count: количество точек, лежащих на плоскости
            - outlier_count: количество точек, не лежащих на плоскости
            - inlier_ratio: доля точек, лежащих на плоскости
            - plane_params: параметры плоскости (a, b, c, d) для уравнения ax + by + cz + d = 0
    """
    if len(points_3d) < 4:
        return False, 0, 0, 0.0, None
    
    points_3d = np.array(points_3d, dtype=np.float32)
    
    if not use_ransac:
        # Используем метод главных компонент для проверки планарности
        centroid = np.mean(points_3d, axis=0)
        centered_points = points_3d - centroid
        cov_matrix = centered_points.T @ centered_points / len(points_3d)
        
        # Вычисляем собственные значения
        eigenvalues, _ = np.linalg.eigh(cov_matrix)
        
        # Сортируем собственные значения в порядке возрастания
        eigenvalues = np.sort(eigenvalues)
        
        # Если наименьшее собственное значение близко к нулю, точки лежат в плоскости
        smallest_eigenvalue = eigenvalues[0]
        is_planar = smallest_eigenvalue < distance_threshold**2
        
        return is_planar, len(points_3d) if is_planar else 0, 0 if is_planar else len(points_3d), 1.0 if is_planar else 0.0, None
    
    # RANSAC метод для проверки планарности
    best_inlier_count = 0
    best_plane_params = None
    
    for iteration in range(ransac_iterations):
        # Выбираем случайные 3 точки для определения плоскости
        random_indices = np.random.choice(len(points_3d), 3, replace=False)
        p1, p2, p3 = points_3d[random_indices[0]], points_3d[random_indices[1]], points_3d[random_indices[2]]
        
        # Проверяем, что точки не лежат на одной линии
        v1 = p2 - p1
        v2 = p3 - p1
        cross_product = np.cross(v1, v2)
        if np.linalg.norm(cross_product) < 1e-6:
            continue  # Точки коллинеарны, не подходят для определения плоскости
            
        # Нормализуем нормальный вектор
        normal = cross_product / np.linalg.norm(cross_product)
        
        # Вычисляем параметр d для уравнения плоскости ax + by + cz + d = 0
        d = -np.dot(normal, p1)
        
        # Считаем количество инлаеров
        inlier_count = 0
        for point in points_3d:
            # Расстояние от точки до плоскости
            distance_to_plane = abs(np.dot(normal, point) + d)
            
            if distance_to_plane <= distance_threshold:
                inlier_count += 1
        
        # Сохраняем лучшие параметры
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane_params = (normal[0], normal[1], normal[2], d)
            
            # Если достигли высокого процента инлаеров, можем остановиться раньше
            if inlier_count / len(points_3d) > ransac_confidence:
                break
    
    # Определяем, лежат ли точки в одной плоскости
    inlier_ratio = best_inlier_count / len(points_3d)
    is_planar = inlier_ratio > ransac_confidence and best_inlier_count >= 4
    
    outlier_count = len(points_3d) - best_inlier_count
    
    print(f"RANSAC результаты (планарность): {best_inlier_count}/{len(points_3d)} точек лежат на плоскости (доля: {inlier_ratio:.2%})")
    
    return is_planar, best_inlier_count, outlier_count, inlier_ratio, best_plane_params

def check_3d_point_collinearity_enhanced(points_3d, distance_threshold=0.1, angle_threshold=0.1,
                                      use_ransac=False, ransac_iterations=100, ransac_confidence=0.99):
    """
    Улучшенная функция проверки коллинеарности 3D точек с использованием RANSAC и других методов.
    
    Args:
        points_3d: Список или массив 3D точек
        distance_threshold: Порог расстояния от точки до линии (в 3D пространстве)
        angle_threshold: Порог угла между векторами (в радианах)
        use_ransac: Использовать ли RANSAC для устойчивой оценки коллинеарности
        ransac_iterations: Количество итераций RANSAC
        ransac_confidence: Уровень доверия RANSAC
        
    Returns:
        tuple: (is_collinear, inlier_count, outlier_count, inlier_ratio)
            - is_collinear: True, если точки коллинеарны
            - inlier_count: количество точек, лежащих на линии
            - outlier_count: количество точек, не лежащих на линии
            - inlier_ratio: доля точек, лежащих на линии
    """
    if len(points_3d) < 3:
        return False, 0, 0, 0.0
    
    points_3d = np.array(points_3d, dtype=np.float32)
    
    # Если точек мало, используем базовый метод
    if len(points_3d) < 6 or not use_ransac:
        is_collinear = check_3d_point_collinearity(points_3d, distance_threshold, angle_threshold)
        return is_collinear, len(points_3d) if is_collinear else 0, 0 if is_collinear else len(points_3d), 1.0 if is_collinear else 0.0
    
    # RANSAC метод для проверки коллинеарности
    best_inlier_count = 0
    best_line_params = None
    
    for iteration in range(ransac_iterations):
        # Выбираем случайные 2 точки для определения линии
        random_indices = np.random.choice(len(points_3d), 2, replace=False)
        p1, p2 = points_3d[random_indices[0]], points_3d[random_indices[1]]
        
        # Проверяем, что точки не совпадают
        if np.allclose(p1, p2):
            continue
            
        # Направляющий вектор линии
        direction = p2 - p1
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < 1e-6:
            continue
            
        direction_unit = direction / direction_norm
        
        # Считаем количество инлаеров
        inlier_count = 0
        for point in points_3d:
            # Расстояние от точки до линии
            to_point = point - p1
            cross_product = np.cross(to_point, direction_unit)
            distance_to_line = np.linalg.norm(cross_product)
            
            if distance_to_line <= distance_threshold:
                inlier_count += 1
        
        # Сохраняем лучшие параметры
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_line_params = (p1, direction_unit)
            
            # Если достигли высокого процента инлаеров, можем остановиться раньше
            if inlier_count / len(points_3d) > ransac_confidence:
                break
    
    # Определяем, коллинеарны ли точки
    inlier_ratio = best_inlier_count / len(points_3d)
    is_collinear = inlier_ratio > ransac_confidence and best_inlier_count >= 3
    
    outlier_count = len(points_3d) - best_inlier_count
    
    print(f"RANSAC результаты: {best_inlier_count}/{len(points_3d)} точек лежат на линии (доля: {inlier_ratio:.2%})")
    
def check_3d_point_planarity(points_3d, distance_threshold=0.1, use_ransac=True, ransac_iterations=100, ransac_confidence=0.95):
    """
    Проверяет, лежат ли 3D точки в одной плоскости.
    
    Args:
        points_3d: Список или массив 3D точек
        distance_threshold: Порог расстояния от точки до плоскости (в 3D пространстве)
        use_ransac: Использовать ли RANSAC для устойчивой оценки планарности
        ransac_iterations: Количество итераций RANSAC
        ransac_confidence: Уровень доверия RANSAC
        
    Returns:
        tuple: (is_planar, inlier_count, outlier_count, inlier_ratio, plane_params)
            - is_planar: True, если точки лежат в одной плоскости
            - inlier_count: количество точек, лежащих на плоскости
            - outlier_count: количество точек, не лежащих на плоскости
            - inlier_ratio: доля точек, лежащих на плоскости
            - plane_params: параметры плоскости (a, b, c, d) для уравнения ax + by + cz + d = 0
    """
    if len(points_3d) < 4:
        return False, 0, 0, 0.0, None
    
    points_3d = np.array(points_3d, dtype=np.float32)
    
    if not use_ransac:
        # Используем метод главных компонент для проверки планарности
        centroid = np.mean(points_3d, axis=0)
        centered_points = points_3d - centroid
        cov_matrix = centered_points.T @ centered_points / len(points_3d)
        
        # Вычисляем собственные значения
        eigenvalues, _ = np.linalg.eigh(cov_matrix)
        
        # Сортируем собственные значения в порядке возрастания
        eigenvalues = np.sort(eigenvalues)
        
        # Если наименьшее собственное значение близко к нулю, точки лежат в плоскости
        smallest_eigenvalue = eigenvalues[0]
        is_planar = smallest_eigenvalue < distance_threshold**2
        
        return is_planar, len(points_3d) if is_planar else 0, 0 if is_planar else len(points_3d), 1.0 if is_planar else 0.0, None
    
    # RANSAC метод для проверки планарности
    best_inlier_count = 0
    best_plane_params = None
    
    for iteration in range(ransac_iterations):
        # Выбираем случайные 3 точки для определения плоскости
        random_indices = np.random.choice(len(points_3d), 3, replace=False)
        p1, p2, p3 = points_3d[random_indices[0]], points_3d[random_indices[1]], points_3d[random_indices[2]]
        
        # Проверяем, что точки не лежат на одной линии
        v1 = p2 - p1
        v2 = p3 - p1
        cross_product = np.cross(v1, v2)
        if np.linalg.norm(cross_product) < 1e-6:
            continue  # Точки коллинеарны, не подходят для определения плоскости
            
        # Нормализуем нормальный вектор
        normal = cross_product / np.linalg.norm(cross_product)
        
        # Вычисляем параметр d для уравнения плоскости ax + by + cz + d = 0
        d = -np.dot(normal, p1)
        
        # Считаем количество инлаеров
        inlier_count = 0
        for point in points_3d:
            # Расстояние от точки до плоскости
            distance_to_plane = abs(np.dot(normal, point) + d)
            
            if distance_to_plane <= distance_threshold:
                inlier_count += 1
        
        # Сохраняем лучшие параметры
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane_params = (normal[0], normal[1], normal[2], d)
            
            # Если достигли высокого процента инлаеров, можем остановиться раньше
            if inlier_count / len(points_3d) > ransac_confidence:
                break
    
    # Определяем, лежат ли точки в одной плоскости
    inlier_ratio = best_inlier_count / len(points_3d)
    is_planar = inlier_ratio > ransac_confidence and best_inlier_count >= 4
    
    outlier_count = len(points_3d) - best_inlier_count
    
    print(f"RANSAC результаты (планарность): {best_inlier_count}/{len(points_3d)} точек лежат на плоскости (доля: {inlier_ratio:.2%})")
    
    return is_planar, best_inlier_count, outlier_count, inlier_ratio, best_plane_params

def validate_calibration_points_3d(points_3d, min_points=8, collinearity_threshold=0.1, planarity_threshold=0.1,
                                use_robust_validation=True):
    """
    Проверяет валидность 3D точек для калибровки.
    Проверяет, что точки не коллинеарны и не лежат в одной плоскости (для надежной калибровки).
    
    Args:
        points_3d: Словарь 3D точек {point_id: [x, y, z]} или список 3D точек
        min_points: Минимальное количество точек
        collinearity_threshold: Порог для проверки коллинеарности
        planarity_threshold: Порог для проверки планарности
        use_robust_validation: Использовать ли робастные методы проверки (RANSAC и т.д.)
        
    Returns:
        dict: Результаты проверки с информацией о валидности и найденных проблемах
    """
    result = {
        'valid': True,
        'issues': [],
        'collinear_groups': [],
        'planar_groups': [],
        'insufficient_points': False,
        'statistical_metrics': {}
    }
    
    # Если передан словарь точек, преобразуем в список
    if isinstance(points_3d, dict):
        points_list = list(points_3d.values())
    else:
        points_list = points_3d
    
    # Проверяем минимальное количество точек
    if len(points_list) < min_points:
        result['valid'] = False
        result['insufficient_points'] = True
        result['issues'].append(f"Недостаточно точек: {len(points_list)} (минимум {min_points})")
        return result
    
    # Проверяем, что у нас есть хотя бы 8 уникальных точек (для калибровки)
    unique_points = list(set(tuple(point) for point in points_list))
    if len(unique_points) < min_points:
        result['valid'] = False
        result['issues'].append(f"Недостаточно уникальных точек: {len(unique_points)} (минимум {min_points})")
    
    points_array = np.array(points_list, dtype=np.float32)
    
    # Вычисляем статистические метрики
    if len(points_array) > 0:
        centroid = np.mean(points_array, axis=0)
        distances_from_centroid = np.linalg.norm(points_array - centroid, axis=1)
        mean_distance = np.mean(distances_from_centroid)
        std_distance = np.std(distances_from_centroid)
        min_distance = np.min(distances_from_centroid)
        max_distance = np.max(distances_from_centroid)
        
        result['statistical_metrics'] = {
            'centroid': centroid.tolist(),
            'mean_distance_from_centroid': float(mean_distance),
            'std_distance_from_centroid': float(std_distance),
            'min_distance_from_centroid': float(min_distance),
            'max_distance_from_centroid': float(max_distance),
            'total_points': len(points_array),
            'unique_points': len(unique_points)
        }
    
    # Проверяем, что не все точки находятся в одной плоскости
    is_planar, inlier_count, outlier_count, inlier_ratio, plane_params = check_3d_point_planarity(
        points_array,
        distance_threshold=planarity_threshold,
        use_ransac=use_robust_validation
    )
    
    if is_planar:
        result['valid'] = False
        result['issues'].append(f"Все точки лежат в одной плоскости (планарность: {inlier_ratio:.2%}, инлаеров: {inlier_count}/{len(points_array)})")
    
    # Проверяем коллинеарность точек
    # Для этого разбиваем точки на потенциальные линии и проверяем каждую
    is_collinear, collinear_inliers, collinear_outliers, collinear_ratio = check_3d_point_collinearity_enhanced(
        points_array,
        distance_threshold=collinearity_threshold,
        use_ransac=use_robust_validation
    )
    
    if is_collinear:
        result['valid'] = False
        result['issues'].append(f"Все точки коллинеарны (на одной линии): {collinear_inliers}/{len(points_array)} точек лежат на линии ({collinear_ratio:.2%})")
    
    # Проверяем, что точки не сконцентрированы в одном месте (слишком близко друг к другу)
    if len(points_array) > 1:
        # Вычисляем расстояния между всеми парами точек
        distances = []
        for i in range(len(points_array)):
            for j in range(i+1, len(points_array)):
                dist = np.linalg.norm(points_array[i] - points_array[j])
                distances.append(dist)
        
        if distances:
            mean_distance = np.mean(distances)
            min_distance = np.min(distances)
            
            if mean_distance < 0.1:  # Если точки слишком близко друг к другу
                result['valid'] = False
                result['issues'].append(f"Точки слишком близко друг к другу: среднее расстояние {mean_distance:.4f}")
            elif min_distance < 1e-6:  # Если есть совпадающие точки
                result['valid'] = False
                result['issues'].append("Обнаружены совпадающие точки")
    
    return result

def check_3d_point_planarity(points_3d, distance_threshold=0.1):
    """
    Проверяет, лежат ли 3D точки в одной плоскости.
    
    Args:
        points_3d: Список или массив 3D точек
        distance_threshold: Порог расстояния от точки до плоскости (в 3D пространстве)
        
    Returns:
        bool: True, если точки лежат в одной плоскости
    """
    if len(points_3d) < 4:
        return True  # 3 точки всегда лежат в плоскости
    
    points_3d = np.array(points_3d, dtype=np.float32)
    
    # Вычисляем центр масс точек
    centroid = np.mean(points_3d, axis=0)
    
    # Вычисляем ковариационную матрицу
    centered_points = points_3d - centroid
    cov_matrix = centered_points.T @ centered_points / len(points_3d)
    
    # Вычисляем собственные значения
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    
    # Сортируем собственные значения в порядке возрастания
    eigenvalues = np.sort(eigenvalues)
    
    # Если наименьшее собственное значение близко к нулю, точки лежат в плоскости
    smallest_eigenvalue = eigenvalues[0]
    
    return smallest_eigenvalue < distance_threshold**2

def validate_calibration_points_3d(points_3d, min_points=8, collinearity_threshold=0.1, planarity_threshold=0.1,
                                use_robust_validation=True, ransac_iterations=100, ransac_confidence=0.95):
    """
    Проверяет валидность 3D точек для калибровки.
    Проверяет, что точки не коллинеарны и не лежат в одной плоскости (для надежной калибровки).
    
    Args:
        points_3d: Словарь 3D точек {point_id: [x, y, z]}
        min_points: Минимальное количество точек
        collinearity_threshold: Порог для проверки коллинеарности
        planarity_threshold: Порог для проверки планарности
        use_robust_validation: Использовать ли робастные методы проверки (RANSAC и т.д.)
        ransac_iterations: Количество итераций RANSAC
        ransac_confidence: Уровень доверия RANSAC
        
    Returns:
        dict: Результаты проверки с информацией о валидности и найденных проблемах
    """
    result = {
        'valid': True,
        'issues': [],
        'collinear_groups': [],
        'planar_groups': [],
        'insufficient_points': False,
        'statistics': {}
    }
    
    # Если передан словарь точек, преобразуем в список
    if isinstance(points_3d, dict):
        points_list = list(points_3d.values())
        point_ids = list(points_3d.keys())
    else:
        points_list = points_3d
        point_ids = list(range(len(points_3d)))
    
    # Проверяем минимальное количество точек
    if len(points_list) < min_points:
        result['valid'] = False
        result['insufficient_points'] = True
        result['issues'].append(f"Недостаточно точек: {len(points_list)} (минимум {min_points})")
        return result
    
    # Проверяем, что у нас есть хотя бы 8 уникальных точек (для калибровки)
    unique_points = list(set(tuple(point) for point in points_list))
    if len(unique_points) < min_points:
        result['valid'] = False
        result['issues'].append(f"Недостаточно уникальных точек: {len(unique_points)} (минимум {min_points})")
    
    points_array = np.array(points_list, dtype=np.float32)
    
    # Вычисляем статистику точек
    if len(points_array) > 0:
        centroid = np.mean(points_array, axis=0)
        distances_from_centroid = np.linalg.norm(points_array - centroid, axis=1)
        mean_distance = np.mean(distances_from_centroid)
        std_distance = np.std(distances_from_centroid)
        min_distance = np.min(distances_from_centroid)
        max_distance = np.max(distances_from_centroid)
        
        result['statistics'] = {
            'centroid': centroid.tolist(),
            'mean_distance_from_centroid': float(mean_distance),
            'std_distance_from_centroid': float(std_distance),
            'min_distance_from_centroid': float(min_distance),
            'max_distance_from_centroid': float(max_distance),
            'total_points': len(points_array),
            'unique_points': len(unique_points)
        }
    
    # Проверяем, что точки не лежат в одной плоскости
    is_planar, inlier_count, outlier_count, inlier_ratio, plane_params = check_3d_point_planarity(
        points_array,
        distance_threshold=planarity_threshold,
        use_ransac=use_robust_validation,
        ransac_iterations=ransac_iterations,
        ransac_confidence=ransac_confidence
    )
    
    if is_planar:
        result['valid'] = False
        result['issues'].append(f"Все точки лежат в одной плоскости (планарность: {inlier_ratio:.2%}, инлаеров: {inlier_count}/{len(points_array)})")
    
    # Проверяем коллинеарность точек
    # Для этого разбиваем точки на потенциальные линии и проверяем каждую
    is_collinear, collinear_inliers, collinear_outliers, collinear_ratio = check_3d_point_collinearity_enhanced(
        points_array,
        distance_threshold=collinearity_threshold,
        use_ransac=use_robust_validation,
        ransac_iterations=ransac_iterations,
        ransac_confidence=ransac_confidence
    )
    
    if is_collinear:
        result['valid'] = False
        result['issues'].append(f"Все точки коллинеарны (на одной линии): {collinear_inliers}/{len(points_array)} точек лежат на линии ({collinear_ratio:.2%})")
    
    # Проверяем, что точки не сконцентрированы в одном месте (слишком близко друг к другу)
    if len(points_array) > 1:
        # Вычисляем расстояния между всеми парами точек
        distances = []
        for i in range(len(points_array)):
            for j in range(i+1, len(points_array)):
                dist = np.linalg.norm(points_array[i] - points_array[j])
                distances.append(dist)
        
        if distances:
            mean_distance = np.mean(distances)
            min_distance = np.min(distances)
            
            if mean_distance < 0.1:  # Если точки слишком близко друг к другу
                result['valid'] = False
                result['issues'].append(f"Точки слишком близко друг к другу: среднее расстояние {mean_distance:.4f}")
            elif min_distance < 1e-6:  # Если есть совпадающие точки
                result['valid'] = False
                result['issues'].append("Обнаружены совпадающие точки")
    
    return result

def validate_calibration_points_3d_comprehensive(points_3d, min_points=8, collinearity_threshold=0.1, planarity_threshold=0.1,
                                              use_robust_validation=True, ransac_iterations=100, ransac_confidence=0.95,
                                              check_distribution=True, min_distribution_ratio=0.1):
    """
    Комплексная проверка валидности 3D точек для калибровки.
    Проверяет, что точки не коллинеарны и не лежат в одной плоскости (для надежной калибровки),
    а также оценивает распределение точек в 3D пространстве.
    
    Args:
        points_3d: Словарь 3D точек {point_id: [x, y, z]}
        min_points: Минимальное количество точек
        collinearity_threshold: Порог для проверки коллинеарности
        planarity_threshold: Порог для проверки планарности
        use_robust_validation: Использовать ли робастные методы проверки (RANSAC и т.д.)
        ransac_iterations: Количество итераций RANSAC
        ransac_confidence: Уровень доверия RANSAC
        check_distribution: Проверять ли равномерность распределения точек
        min_distribution_ratio: Минимальный порог отношения минимального расстояния к максимальному (для проверки равномерности)
        
    Returns:
        dict: Результаты проверки с информацией о валидности и найденных проблемах
    """
    result = {
        'valid': True,
        'issues': [],
        'collinear_groups': [],
        'planar_groups': [],
        'insufficient_points': False,
        'statistics': {},
        'distribution_analysis': {}
    }
    
    # Если передан словарь точек, преобразуем в список
    if isinstance(points_3d, dict):
        points_list = list(points_3d.values())
        point_ids = list(points_3d.keys())
    else:
        points_list = points_3d
        point_ids = list(range(len(points_3d)))
    
    # Проверяем минимальное количество точек
    if len(points_list) < min_points:
        result['valid'] = False
        result['insufficient_points'] = True
        result['issues'].append(f"Недостаточно точек: {len(points_list)} (минимум {min_points})")
        return result
    
    # Проверяем, что у нас есть хотя бы 8 уникальных точек (для калибровки)
    unique_points = list(set(tuple(point) for point in points_list))
    if len(unique_points) < min_points:
        result['valid'] = False
        result['issues'].append(f"Недостаточно уникальных точек: {len(unique_points)} (минимум {min_points})")
    
    points_array = np.array(points_list, dtype=np.float32)
    
    # Вычисляем статистику точек
    if len(points_array) > 0:
        centroid = np.mean(points_array, axis=0)
        distances_from_centroid = np.linalg.norm(points_array - centroid, axis=1)
        mean_distance = np.mean(distances_from_centroid)
        std_distance = np.std(distances_from_centroid)
        min_distance = np.min(distances_from_centroid)
        max_distance = np.max(distances_from_centroid)
        
        result['statistics'] = {
            'centroid': centroid.tolist(),
            'mean_distance_from_centroid': float(mean_distance),
            'std_distance_from_centroid': float(std_distance),
            'min_distance_from_centroid': float(min_distance),
            'max_distance_from_centroid': float(max_distance),
            'total_points': len(points_array),
            'unique_points': len(unique_points)
        }
    
    # Проверяем, что точки не лежат в одной плоскости
    is_planar, inlier_count, outlier_count, inlier_ratio, plane_params = check_3d_point_planarity(
        points_array,
        distance_threshold=planarity_threshold,
        use_ransac=use_robust_validation,
        ransac_iterations=ransac_iterations,
        ransac_confidence=ransac_confidence
    )
    
    if is_planar:
        result['valid'] = False
        result['issues'].append(f"Все точки лежат в одной плоскости (планарность: {inlier_ratio:.2%}, инлаеров: {inlier_count}/{len(points_array)})")
    
    # Проверяем коллинеарность точек
    is_collinear, collinear_inliers, collinear_outliers, collinear_ratio = check_3d_point_collinearity_enhanced(
        points_array,
        distance_threshold=collinearity_threshold,
        use_ransac=use_robust_validation,
        ransac_iterations=ransac_iterations,
        ransac_confidence=ransac_confidence
    )
    
    if is_collinear:
        result['valid'] = False
        result['issues'].append(f"Все точки коллинеарны (на одной линии): {collinear_inliers}/{len(points_array)} точек лежат на линии ({collinear_ratio:.2%})")
    
    # Проверяем распределение точек в 3D пространстве
    if check_distribution and len(points_array) > 1:
        # Вычисляем расстояния между всеми парами точек
        distances = []
        for i in range(len(points_array)):
            for j in range(i+1, len(points_array)):
                dist = np.linalg.norm(points_array[i] - points_array[j])
                distances.append(dist)
        
        if distances:
            mean_distance = np.mean(distances)
            min_distance = np.min(distances)
            max_distance = np.max(distances)
            
            # Проверяем отношение минимального расстояния к максимальному
            # Если отношение слишком маленькое, точки неравномерно распределены
            distance_ratio = min_distance / max_distance if max_distance > 0 else 0
            
            result['distribution_analysis'] = {
                'mean_distance': float(mean_distance),
                'min_distance': float(min_distance),
                'max_distance': float(max_distance),
                'distance_ratio': float(distance_ratio),
                'total_distances': len(distances)
            }
            
            if mean_distance < 0.1:  # Если точки слишком близко друг к другу
                result['valid'] = False
                result['issues'].append(f"Точки слишком близко друг к другу: среднее расстояние {mean_distance:.4f}")
            elif min_distance < 1e-6:  # Если есть совпадающие точки
                result['valid'] = False
                result['issues'].append("Обнаружены совпадающие точки")
            elif distance_ratio < min_distribution_ratio:  # Если распределение неравномерное
                result['issues'].append(f"Неравномерное распределение точек: отношение min/max расстояний {distance_ratio:.4f} < {min_distribution_ratio}")
                print(f"Предупреждение: неравномерное распределение точек, но продолжаем проверку")
    
    # Также проверяем, что точки не лежат на одной линии или в одной плоскости, используя SVD
    if len(points_array) >= 3:
        # Центрируем точки
        centered_points = points_array - np.mean(points_array, axis=0)
        
        # Вычисляем SVD
        U, S, Vt = np.linalg.svd(centered_points)
        
        # S содержит сингулярные значения - они показывают размеры в каждом измерении
        # Если одно из значений близко к 0, точки лежат в плоскости или на линии
        singular_values = S
        result['statistics']['singular_values'] = [float(val) for val in singular_values]
        
        # Проверяем, что все три сингулярных значения не равны нулю (или близки к нулю)
        # Если два из трех значений близки к нулю, точки коллинеарны
        # Если одно из трех значений близко к нулю, точки лежат в плоскости
        near_zero_count = np.sum(singular_values < 1e-6)
        if near_zero_count >= 2:
            result['valid'] = False
            result['issues'].append(f"Точки коллинеарны (два сингулярных значения близки к нулю: {singular_values[:2]})")
        elif near_zero_count >= 1:
            result['valid'] = False
            result['issues'].append(f"Точки лежат в одной плоскости (одно сингулярное значение близко к нулю: {singular_values[0]})")
    
    return result

def validate_triangulated_point(point_3d, camera1, camera2):
    """
    Проверяет, что триангулированная точка находится перед обеими камерами.
    
    Args:
        point_3d: 3D точка в глобальной системе координат
        camera1: Кортеж (R1, t1) для первой камеры
        camera2: Кортеж (R2, t2) для второй камеры
        
    Returns:
        bool: True, если точка валидна
    """
    try:
        R1, t1 = camera1
        R2, t2 = camera2
        
        # Преобразуем в numpy массивы
        point_3d = np.asarray(point_3d, dtype=np.float32)
        R1 = np.asarray(R1, dtype=np.float32)
        R2 = np.asarray(R2, dtype=np.float32)
        t1 = np.asarray(t1, dtype=np.float32)
        t2 = np.asarray(t2, dtype=np.float32)
        
        # Обработка различных форматов t1
        if t1.size == 9 and (t1.shape == (3, 3) or t1.shape == (9,)):
            if t1.shape == (9,):
                t1 = t1.reshape(3, 3)
            t1 = t1[:, 2].copy()
        elif t1.size == 12 and (t1.shape == (3, 4) or t1.shape == (12,)):
            if t1.shape == (12,):
                t1 = t1.reshape(3, 4)
            t1 = t1[:, 3].copy()
        elif t1.size == 1:  # Если передано скалярное значение
            print("Предупреждение: вектор t1 имеет скалярное значение, преобразование в вектор")
            t1 = np.array([t1[0], 0, 0])  # Преобразуем в вектор [t, 0]
        elif t1.size == 3 and len(t1.shape) > 1:  # Если t - это вектор в виде 2D массива
            t1 = t1.ravel()  # Плоский массив из многомерного
            
        # Обработка различных форматов t2
        if t2.size == 9 and (t2.shape == (3, 3) or t2.shape == (9,)):
            if t2.shape == (9,):
                t2 = t2.reshape(3, 3)
            t2 = t2[:, 2].copy()
        elif t2.size == 12 and (t2.shape == (3, 4) or t2.shape == (12,)):
            if t2.shape == (12,):
                t2 = t2.reshape(3, 4)
            t2 = t2[:, 3].copy()
        elif t2.size == 1:  # Если передано скалярное значение
            print("Предупреждение: вектор t2 имеет скалярное значение, преобразование в вектор")
            t2 = np.array([t2[0], 0, 0])  # Преобразуем в вектор [t, 0]
        elif t2.size == 3 and len(t2.shape) > 1:  # Если t - это вектор в виде 2D массива
            t2 = t2.ravel()  # Плоский массив из многомерного
        
        # Преобразуем векторы t в правильный формат (3,)
        if t1.ndim > 1 and t1.shape[1] == 1:
            t1 = t1.ravel()
        elif t1.ndim == 1 and t1.size == 3:
            pass  # Уже в нужном формате
        else:
            t1 = t1.reshape(3,)
            
        if t2.ndim > 1 and t2.shape[1] == 1:
            t2 = t2.ravel()
        elif t2.ndim == 1 and t2.size == 3:
            pass  # Уже в нужном формате
        else:
            t2 = t2.reshape(3,)
        
        # Преобразуем точку в систему координат первой камеры
        point_camera1 = R1 @ point_3d + t1
        
        # Преобразуем точку в систему координат второй камеры
        point_camera2 = R2 @ point_3d + t2
        
        # Проверяем, что точка находится перед обеими камерами (z > 0)
        # Также проверяем, что точка не слишком близко к камере (z > 0.01) и не слишком далеко (z < 1000)
        is_in_front_1 = 0.01 < point_camera1[2] < 1000
        is_in_front_2 = 0.01 < point_camera2[2] < 1000
        
        # Проверяем, что точка не имеет аномально больших координат
        reasonable_coords = np.all(np.abs(point_3d) < 10000) and np.all(np.abs(point_camera1) < 10000) and np.all(np.abs(point_camera2) < 10000)
        
        # Дополнительно проверяем, что точка не слишком близко к линии базиса между камерами
        # Это помогает избежать точек, которые находятся на неоправданно большом расстоянии
        baseline_vector = t2 - t1
        baseline_length = np.linalg.norm(baseline_vector)
        
        # Вычисляем расстояние от точки до линии базиса
        if baseline_length > 0.01:  # Проверяем, что базовая линия не слишком короткая
            point_to_camera1 = point_camera1  # В системе координат камеры 1
            point_to_camera2 = point_camera2  # В системе координат камеры 2
            baseline_normalized = baseline_vector / baseline_length
            
            # Вектор от камеры 1 к 3D точке в мировой системе координат
            point_world_offset = point_3d - t1
            projection_length = np.dot(point_world_offset, baseline_normalized)
            projection_on_baseline = projection_length * baseline_normalized
            perpendicular_vector = point_world_offset - projection_on_baseline
            distance_to_baseline = np.linalg.norm(perpendicular_vector)
            
            # Если точка слишком близко к линии базиса, это может быть неверная триангуляция
            reasonable_triangulation = distance_to_baseline > 0.01 * baseline_length
        else:
            reasonable_triangulation = True  # Если базовая линия короткая, пропускаем эту проверку
        
        return is_in_front_1 and is_in_front_2 and reasonable_coords and reasonable_triangulation
        
    except Exception as e:
        print(f"Ошибка при проверке положения точки: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
