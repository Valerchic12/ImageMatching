"""
Функции для триангуляции 3D точек.
"""
import cv2
import numpy as np
import traceback
import io
from itertools import combinations
from typing import List, Tuple, Dict, Optional, Union
import logging
from contextlib import nullcontext, redirect_stdout

try:
    from . import utils
except ImportError:  # pragma: no cover - fallback for direct module execution
    import utils

def _debug_print(debug_logging, message):
    if debug_logging:
        print(message)

def _stable_sort_key(value):
    text = str(value)
    try:
        return (0, int(text))
    except (TypeError, ValueError):
        return (1, text)


def _downweight_observation_confidence(
    observation_confidences,
    camera_id,
    point_id,
    *,
    target_confidence=0.22,
    decay=0.45,
):
    if not isinstance(observation_confidences, dict):
        return None

    camera_key = str(camera_id)
    camera_confidences = observation_confidences.setdefault(camera_key, {})
    current_value = camera_confidences.get(point_id, 1.0)
    try:
        current_value = float(current_value)
    except (TypeError, ValueError):
        current_value = 1.0

    new_value = min(current_value * float(decay), float(target_confidence))
    new_value = float(np.clip(new_value, 0.15, 1.0))
    camera_confidences[point_id] = new_value
    return new_value


def normalize_points(points, K, dist_coeffs=None):
    """
    Нормализует точки на изображении с использованием матрицы калибровки камеры.
    Преобразует координаты пикселей в нормализованные координаты камеры.
    
    Args:
        points: Массив 2D точек на изображении
        K: Матрица калибровки камеры
        
    Returns:
        Массив нормализованных точек
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    normalized = cv2.undistortPoints(
        points.reshape(-1, 1, 2), 
        K, 
        dist_coeffs
    ).reshape(-1, 2)
    return normalized

def triangulate_points(pts1, pts2, K, R=None, t=None, robust=True, reprojection_threshold=10.0,
                      numerical_stability_check=True, parallax_threshold=2.0, uncertainty_based_filtering=True,
                      use_ensemble_method=True, check_planarity=True, min_parallax_for_validation=0.5,
                      dist_coeffs=None,
                      debug_logging=False):
    """
    Триангулирует 3D точки из соответствующих 2D точек с улучшенной фильтрацией и проверкой.
    
    Args:
        pts1: Массив 2D точек первой камеры
        pts2: Массив 2D точек второй камеры
        K: Матрица калибровки камеры
        R: Матрица поворота второй камеры относительно первой (опционально)
        t: Вектор перемещения второй камеры относительно первой (опционально)
        robust: Применять ли робастные методы для отбрасывания выбросов
        reprojection_threshold: Порог ошибки репроекции для фильтрации выбросов
        numerical_stability_check: Выполнять ли проверку на численную стабильность
        parallax_threshold: Порог угла параллакса для проверки надежности триангуляции (в градусах)
        uncertainty_based_filtering: Использовать ли фильтрацию на основе неопределенности
        use_ensemble_method: Использовать ли ансамблевый метод для улучшения точности
        check_planarity: Проверять ли, что точки не лежат в одной плоскости (для улучшения точности)
        min_parallax_for_validation: Минимальный параллакс для проверки валидности точки (в градусах)
        
    Returns:
        Массив 3D точек и маска валидных точек
    """
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    
    # Проверяем, что количество точек совпадает
    if len(pts1) != len(pts2):
        print(f"Ошибка: количество точек не совпадает: {len(pts1)} vs {len(pts2)}")
        return np.array([]), np.array([])
    
    # Нормализуем координаты точек для численной стабильности
    pts1_normalized = normalize_points(pts1, K, dist_coeffs=dist_coeffs)
    pts2_normalized = normalize_points(pts2, K, dist_coeffs=dist_coeffs)
    
    if R is None or t is None:
        # Вычисляем фундаментальную матрицу с улучшенными параметрами
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.USAC_MAGSAC, 3.0, 0.999, maxIters=10000)
        
        # Получаем существенную матрицу
        E = K.T @ F @ K
        
        # Восстанавливаем позу из существенной матрицы с улучшенной стабильностью
        _, R, t, _ = cv2.recoverPose(E, pts1_normalized, pts2_normalized)
        t_col = utils.normalize_translation(t, dtype=np.float64)
    else:
        try:
            t_col = utils.normalize_translation(t, dtype=np.float64)
        except ValueError as exc:
            print(f"Ошибка: некорректный формат вектора t: {exc}")
            return np.array([]), np.array([])

    # Точки уже нормализованы через undistortPoints, поэтому матрицы проекции
    # должны быть в координатах камеры без повторного умножения на K.
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t_col))

    # Проверяем, что матрицы проекции имеют правильный формат
    if P1.shape != (3, 4) or P2.shape != (3, 4):
        print(f"Ошибка: неправильный формат матриц проекции: P1 {P1.shape}, P2 {P2.shape}")
        return np.array([]), np.array([])

    # Логируем матрицы проекции для отладки
    _debug_print(debug_logging, f"DEBUG: Матрица проекции P1: {P1.shape}")
    _debug_print(debug_logging, f"DEBUG: Матрица проекции P2: {P2.shape}")

    # Триангулируем точки с помощью OpenCV
    try:
        points_4d = cv2.triangulatePoints(P1, P2, pts1_normalized.T, pts2_normalized.T)
        
        # Преобразуем из однородных координат
        points_3d = points_4d[:3, :] / points_4d[3, :]
        
        # Проверяем точки на валидность (перед камерами)
        mask = np.ones(points_3d.shape[1], dtype=bool)
        
        if robust:
            # Проверяем, находятся ли точки перед обеими камерами
            for i in range(points_3d.shape[1]):
                point = points_3d[:, i]
                
                # Проверяем, находится ли точка перед первой камерой
                if point[2] <= 0.01:  # Увеличили порог с 0 до 0.01 для большей устойчивости
                    mask[i] = False
                    continue
                    
                # Преобразуем точку в систему координат второй камеры
                point2 = R @ point + t_col.ravel()
                
                # Проверяем, находится ли точка перед второй камерой
                if point2[2] <= 0.01:  # Увеличили порог с 0 до 0.01 для большей устойчивости
                    mask[i] = False
                    continue
                    
                # Проверяем разумность масштаба 3D точки (не слишком ли велики координаты)
                # Это может указывать на численную нестабильность
                if np.any(np.abs(point) > 10000) or np.any(np.abs(point2) > 1000):
                    _debug_print(debug_logging, f"Предупреждение: точка {i} имеет подозрительно большие координаты: {point}, {point2}")
                    mask[i] = False
                    continue
                    
                # Проверяем ошибку репроекции
                if reprojection_threshold > 0:
                    # Проецируем 3D точку на обе камеры
                    proj1 = K @ point.reshape(3, 1)
                    proj1 = proj1[:2] / proj1[2]
                    
                    proj2 = K @ point2.reshape(3, 1)
                    proj2 = proj2[:2] / proj2[2]
                    
                    # Вычисляем ошибку репроекции
                    error1 = np.linalg.norm(proj1.ravel() - pts1[i])
                    error2 = np.linalg.norm(proj2.ravel() - pts2[i])
                    
                    # Если ошибка слишком велика, отбрасываем точку
                    if error1 > reprojection_threshold or error2 > reprojection_threshold:
                        mask[i] = False
                
                # Проверяем угол параллакса между лучами от обеих камер к точке
                if parallax_threshold > 0:
                    # Вектора от центров камер к 3D точке
                    ray1 = point  # Вектор от центра первой камеры к точке
                    ray2 = point - (-R.T @ t_col.ravel())  # центр второй камеры в системе координат первой
                    
                    # Нормализуем лучи
                    ray1_norm = ray1 / (np.linalg.norm(ray1) + 1e-10)
                    ray2_norm = ray2 / (np.linalg.norm(ray2) + 1e-10)
                    
                    # Вычисляем угол между лучами (в радианах)
                    cos_angle = np.clip(np.dot(ray1_norm, ray2_norm), -1.0, 1.0)
                    angle_rad = np.arccos(cos_angle)
                    angle_deg = np.degrees(angle_rad)
                    
                    # Если угол параллакса слишком маленький, точка может быть нестабильной
                    if angle_deg < min_parallax_for_validation:  # Используем более строгий порог для проверки валидности
                        _debug_print(debug_logging, f"Предупреждение: точка {i} имеет очень малый угол параллакса: {angle_deg:.2f}° < {min_parallax_for_validation}°")
                        mask[i] = False
                        continue
                        
                    if angle_deg < parallax_threshold:
                        _debug_print(debug_logging, f"Предупреждение: точка {i} имеет малый угол параллакса: {angle_deg:.2f}° < {parallax_threshold}°")
                        # Не отбрасываем точку, но добавляем предупреждение о потенциальной нестабильности
                        # mask[i] = False  # Закомментировано, чтобы не отбрасывать точки с малым параллаксом
                        continue
                        
                    # Дополнительно проверяем, что угол параллакса не слишком близок к 180 градусам
                    # Это может указывать на нефизичное расположение точек
                    if angle_deg > 175:  # Угол близок к 180 градусам
                        _debug_print(debug_logging, f"Предупреждение: точка {i} имеет подозрительно большой угол параллакса: {angle_deg:.2f}° > 175°")
                        mask[i] = False
                        continue
                        
                    # Проверяем, что точка не слишком близко к плоскости, образованной базовой линией и оптическими осями
                    # Это помогает избежать точек, которые находятся на неоправданно большом расстоянии
                    baseline_vector = t_col.ravel()
                    baseline_length = np.linalg.norm(baseline_vector)
                    
                    if baseline_length > 0:
                        # Нормализуем вектор базовой линии
                        baseline_unit = baseline_vector / baseline_length
                        
                        # Вычисляем вектор от центра первой камеры к точке
                        point_vec = point
                        
                        # Проекция точки на линию базиса
                        projection_length = np.dot(point_vec, baseline_unit)
                        projection_on_baseline = projection_length * baseline_unit
                        
                        # Перпендикулярный вектор от линии базиса к точке
                        perpendicular_vector = point_vec - projection_on_baseline
                        distance_to_baseline = np.linalg.norm(perpendicular_vector)
                        
                        # Если точка слишком близко к линии базиса, это может указывать на нестабильную триангуляцию
                        # Но также проверяем, что точка не слишком далеко
                        point_distance = np.linalg.norm(point)
                        max_distance_ratio = 100  # Максимальное отношение расстояния до точки к базовой линии
                        
                        if distance_to_baseline < 0.01 and point_distance / baseline_length > max_distance_ratio:
                            _debug_print(debug_logging, f"Предупреждение: точка {i} близка к линии базиса и находится слишком далеко: "
                                  f"расстояние до базиса={distance_to_baseline:.4f}, отношение к базовой линии={point_distance/baseline_length:.2f}")
                            mask[i] = False
                            continue
                
                # Фильтрация на основе неопределенности триангуляции
                if uncertainty_based_filtering:
                    # Вычисляем неопределенность триангуляции на основе геометрии
                    # Это может включать в себя проверку угла между лучами
                    baseline = np.linalg.norm(t_col.ravel())
                    distance1 = np.linalg.norm(point)
                    distance2 = np.linalg.norm(point2)
                    
                    # Если точка очень далеко от обеих камер или близко к линии базиса,
                    # неопределенность будет высокой
                    if baseline > 0:
                        # Вычисляем расстояние от точки до линии базиса
                        baseline_unit = t_col.ravel() / baseline
                        point_to_camera1 = point
                        point_to_camera2 = point2
                        
                        # Вычисляем векторное произведение для нахождения перпендикулярного расстояния
                        cross_product = np.cross(baseline_unit, point_to_camera1)
                        distance_to_baseline = np.linalg.norm(cross_product) / np.linalg.norm(baseline_unit)
                        
                        # Если точка слишком близко к линии базиса, это может указывать на нестабильную триангуляцию
                        # Но также проверяем, что точка не слишком далеко
                        if distance_to_baseline < 0.01 and max(distance1, distance2) > baseline * 10:
                            _debug_print(debug_logging, f"Предупреждение: точка {i} близка к линии базиса и находится далеко: расстояние={distance_to_baseline:.4f}")
                            mask[i] = False
                            continue
                
                # Проверка на численную стабильность
                if numerical_stability_check:
                    # Проверяем, что 3D точка не является результатом численной нестабильности
                    # Это может происходить, когда точки на изображениях почти коллинеарны
                    if np.any(np.isnan(point)) or np.any(np.isinf(point)):
                        _debug_print(debug_logging, f"Предупреждение: точка {i} содержит NaN или Inf значения: {point}")
                        mask[i] = False
                        continue
                    
                    # Проверяем, что 3D точка не слишком удалена
                    # Это может указывать на численную нестабильность при малом параллаксе
                    distance_to_origin = np.linalg.norm(point)
                    max_reasonable_distance = 1000  # Максимальное разумное расстояние
                    
                    if distance_to_origin > max_reasonable_distance:
                        # Также проверяем, не является ли это результатом малого параллакса
                        baseline_length = np.linalg.norm(t_col.ravel())
                        if baseline_length > 0:
                            # Отношение расстояния до точки к базовой линии
                            distance_ratio = distance_to_origin / baseline_length
                            
                            # Если отношение слишком велико, это может указывать на нестабильность
                            if distance_ratio > 1000:  # Отношение больше 1000:1
                                _debug_print(debug_logging, f"Предупреждение: точка {i} находится слишком далеко от начала координат: {distance_to_origin:.2f}, "
                                      f"отношение к базовой линии: {distance_ratio:.2f}")
                                mask[i] = False
                                continue

        # Если включена проверка плоскости, дополнительно проверяем, что точки не лежат в одной плоскости
        if check_planarity and len(points_3d.T) > 3:
            # Проверяем, что не все точки лежат в одной плоскости
            # Используем SVD для анализа распределения точек
            centered_points = points_3d.T - np.mean(points_3d.T, axis=0)
            U, S, Vt = np.linalg.svd(centered_points)
            
            # Если наименьшее сингулярное значение близко к 0, точки лежат в плоскости
            smallest_singular = S[-1]
            if smallest_singular < 1e-6:
                print(f"Предупреждение: точки потенциально лежат в одной плоскости (наименьшее сингулярное значение: {smallest_singular})")
                
                # Вместо отбрасывания всех точек, уменьшаем уверенность в них
                # Это позволяет сохранить информацию, но с пониманием, что она может быть нестабильной
                # Проверяем, насколько значительна проблема с плоскостью
                second_smallest = S[-2] if len(S) > 1 else S[0]
                ratio = smallest_singular / second_smallest if second_smallest > 0 else 0
                
                # Если отношение слишком маленькое (меньше 1e-3), это указывает на реальную проблему
                if ratio < 1e-3:
                    print(f"  - Точки действительно лежат в одной плоскости, снижаем уверенность в результатах")
                    # Для простоты, возвращаем маску без изменений, но с предупреждением

        # Сообщаем о количестве отфильтрованных точек
        filtered_points = np.sum(~mask)
        if filtered_points > 0:
            print(f"Отфильтровано {filtered_points} из {len(mask)} точек при триангуляции")
        else:
            print(f"Все {len(mask)} точек прошли валидацию при триангуляции")
        
        # Транспонируем обратно в формат (n_points, 3)
        points_3d = points_3d.T
        
        # Если включено использование ансамблевого метода, улучшаем точность
        if use_ensemble_method:
            # Вычисляем улучшенную оценку с использованием нескольких подходов
            # 1. Усреднение с учетом надежности
            refined_points = points_3d.copy()
            
            # Проверяем, есть ли хотя бы несколько точек для усреднения
            if len(refined_points) >= 3:
                # Используем взвешенное усреднение для улучшения точности
                # Веса определяем на основе ошибки репроекции и угла параллакса
                weights = np.ones(len(refined_points))
                
                for i in range(len(refined_points)):
                    if not mask[i]:
                        continue
                        
                    point = refined_points[i]
                    
                    # Вес на основе ошибки репроекции (чем меньше ошибка, тем выше вес)
                    if reprojection_threshold > 0:
                        proj1 = K @ point.reshape(3, 1)
                        proj1 = proj1[:2] / proj1[2]
                        proj2 = K @ (R @ point + t_col.ravel()).reshape(3, 1)
                        proj2 = proj2[:2] / proj2[2]
                        
                        error1 = np.linalg.norm(proj1.ravel() - pts1[i])
                        error2 = np.linalg.norm(proj2.ravel() - pts2[i])
                        avg_error = (error1 + error2) / 2
                        
                        # Нормализуем вес ошибки (чем меньше ошибка, тем больше вес)
                        error_weight = max(0.1, 1.0 - avg_error/reprojection_threshold)  # Минимальный вес 0.1
                        weights[i] *= error_weight
                    
                    # Вес на основе угла параллакса (чем больше параллакс, тем выше вес)
                    if parallax_threshold > 0:
                        ray1 = point
                        ray2 = point - (-R.T @ t_col.ravel())
                        
                        ray1_norm = ray1 / (np.linalg.norm(ray1) + 1e-10)
                        ray2_norm = ray2 / (np.linalg.norm(ray2) + 1e-10)
                        
                        cos_angle = np.clip(np.dot(ray1_norm, ray2_norm), -1.0, 1.0)
                        angle_rad = np.arccos(cos_angle)
                        angle_deg = np.degrees(angle_rad)
                        
                        # Нормализуем вес угла (чем больше параллакс, тем больше вес)
                        # Максимальный вес при 90 градусах
                        parallax_weight = max(0.1, min(1.0, angle_deg/90.0))  # Вес 0.1-1.0
                        weights[i] *= parallax_weight
                
                # Если веса не равны 1, применяем взвешенное усреднение
                if not np.allclose(weights, 1.0):
                    # Для улучшения точности, особенно для точек с малым параллаксом,
                    # можем использовать дополнительные проверки и уточнения
                    print(f"Применение ансамблевого метода с весами точек")
                    
                    # Уточняем только точки с высокой уверенностью
                    high_confidence_mask = weights > 0.5
                    if np.sum(high_confidence_mask) >= 3:  # Достаточно точек для уточнения
                        # Используем только высокодостоверенные точки для уточнения других точек
                        # (например, с помощью локального усреднения или уточнения по соседям)
                        
                        # Для каждой точки с низкой уверенностью, ищем ближайших соседей с высокой уверенностью
                        for i in range(len(refined_points)):
                            if not mask[i] or weights[i] > 0.7:  # Пропускаем невалидные или уже высокодостоверные точки
                                continue
                                
                            point = refined_points[i]
                            
                            # Находим ближайших соседей с высокой уверенностью
                            distances = np.linalg.norm(points_3d - point, axis=1)
                            high_conf_indices = np.where(high_confidence_mask)[0]
                            
                            if len(high_conf_indices) >= 2:  # Нужно хотя бы 2 точки для уточнения
                                # Сортируем индексы по расстоянию
                                sorted_indices = high_conf_indices[np.argsort(distances[high_conf_indices])]
                                
                                # Берем только ближайшие точки (в пределах определенного радиуса)
                                neighbors = []
                                for idx in sorted_indices[:5]:  # Максимум 5 соседей
                                    if distances[idx] < 1.0:  # Радиус 1.0 для поиска соседей
                                        neighbors.append(idx)
                                
                                if len(neighbors) >= 2:  # Достаточно соседей для уточнения
                                    # Усредняем положение точки с учетом соседей
                                    neighbor_points = points_3d[neighbors]
                                    neighbor_weights = weights[neighbors]
                                    
                                    # Взвешенное усреднение соседей
                                    weighted_mean = np.average(neighbor_points, axis=0, weights=neighbor_weights)
                                    
                                    # Интерполируем между исходной точкой и усредненным значением
                                    # с учетом уверенности в исходной точке
                                    interpolation_factor = min(0.3, (1.0 - weights[i]) * 0.5)  # Максимум 30% влияния соседей
                                    refined_points[i] = (1 - interpolation_factor) * point + interpolation_factor * weighted_mean
            
            points_3d = refined_points
        
        return points_3d, mask
        
    except Exception as e:
        print(f"Ошибка при триангуляции точек: {str(e)}")
        traceback.print_exc()
        return np.array([]), np.array([])

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
        t1 = utils.normalize_translation(t1, dtype=np.float32).ravel()
        t2 = utils.normalize_translation(t2, dtype=np.float32).ravel()
        
        # Преобразуем точку в систему координат первой камеры
        point_camera1 = R1 @ point_3d + t1
        
        # Преобразуем точку в систему координат второй камеры
        point_camera2 = R2 @ point_3d + t2
        
        # Проверяем, что точка находится перед обеими камерами (z > 0)
        # Также проверяем, что точка не слишком близко к камере (z > 0.01) и не слишком далеко (z < 1000)
        is_in_front_1 = 0.01 < point_camera1[2] < 1000
        is_in_front_2 = 0.01 < point_camera2[2] < 1000
        
        # Проверяем, что точка не имеет аномально больших координат
        reasonable_coords = (
            np.all(np.abs(point_3d) < 10000) and
            np.all(np.abs(point_camera1) < 10000) and
            np.all(np.abs(point_camera2) < 10000)
        )

        # Дополнительно проверяем геометрию относительно реальной базовой линии
        # между центрами камер в мировых координатах.
        camera_center_1 = (-R1.T @ t1.reshape(3, 1)).ravel()
        camera_center_2 = (-R2.T @ t2.reshape(3, 1)).ravel()
        baseline_vector = camera_center_2 - camera_center_1
        baseline_length = np.linalg.norm(baseline_vector)

        if baseline_length > 1e-4:
            baseline_normalized = baseline_vector / baseline_length
            point_world_offset = point_3d - camera_center_1
            projection_length = np.dot(point_world_offset, baseline_normalized)
            projection_on_baseline = projection_length * baseline_normalized
            perpendicular_vector = point_world_offset - projection_on_baseline
            distance_to_baseline = np.linalg.norm(perpendicular_vector)
            point_distance = max(
                np.linalg.norm(point_3d - camera_center_1),
                np.linalg.norm(point_3d - camera_center_2),
            )

            # Отбрасываем только действительно вырожденный случай:
            # точка почти лежит на базовой линии и при этом находится очень далеко.
            reasonable_triangulation = not (
                distance_to_baseline < 0.002 * baseline_length and
                point_distance > 50.0 * baseline_length
            )
        else:
            reasonable_triangulation = True
        
        return is_in_front_1 and is_in_front_2 and reasonable_coords and reasonable_triangulation
        
    except Exception as e:
        print(f"Ошибка при проверке положения точки: {str(e)}")
        traceback.print_exc()
        return False


def _camera_center_from_pose(R, t):
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    return (-R.T @ t).ravel()


def _extract_point_observation_confidences(observation_confidences, point_id, camera_ids):
    if not observation_confidences:
        return {}
    point_id = point_id
    result = {}
    for camera_id in camera_ids:
        camera_id_str = str(camera_id)
        confidence = observation_confidences.get(camera_id_str, {}).get(point_id, 1.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 1.0
        result[camera_id_str] = float(np.clip(confidence, 0.15, 1.0))
    return result


def _prepare_multiview_observations(point_2d_observations, cameras, observation_confidences=None):
    prepared = []
    for cam_id, point_2d in point_2d_observations.items():
        cam_id_str = str(cam_id)
        if cam_id_str not in cameras:
            continue
        confidence = 1.0
        if observation_confidences is not None:
            try:
                confidence = float(observation_confidences.get(cam_id_str, 1.0))
            except (AttributeError, TypeError, ValueError):
                confidence = 1.0
        prepared.append((cam_id_str, np.asarray(point_2d, dtype=np.float64).reshape(2), float(np.clip(confidence, 0.15, 1.0))))
    return prepared


def _lookup_point_3d(points_3d_context, point_id):
    if not isinstance(points_3d_context, dict):
        return None
    if point_id in points_3d_context:
        return points_3d_context.get(point_id)
    point_id_str = str(point_id)
    if point_id_str in points_3d_context:
        return points_3d_context.get(point_id_str)
    for key in points_3d_context.keys():
        if str(key) == point_id_str:
            return points_3d_context.get(key)
    return None


def _build_line_constraints_for_point(point_id, points_3d_context, line_support_data):
    if point_id is None or not isinstance(points_3d_context, dict) or not isinstance(line_support_data, dict):
        return []

    point_triplets = line_support_data.get('point_triplets') or {}
    triplet_lookup = point_triplets.get(str(point_id), [])
    if not triplet_lookup:
        return []

    triplet_meta = line_support_data.get('triplets') or {}
    constraints = []
    for triplet in triplet_lookup:
        triplet_key = tuple(str(value) for value in triplet)
        if str(point_id) not in triplet_key:
            continue
        other_ids = [candidate_id for candidate_id in triplet_key if candidate_id != str(point_id)]
        if len(other_ids) != 2:
            continue
        point_a = _lookup_point_3d(points_3d_context, other_ids[0])
        point_b = _lookup_point_3d(points_3d_context, other_ids[1])
        if point_a is None or point_b is None:
            continue

        point_a = np.asarray(point_a, dtype=np.float64).reshape(3)
        point_b = np.asarray(point_b, dtype=np.float64).reshape(3)
        line_length = float(np.linalg.norm(point_b - point_a))
        if line_length <= 1e-4 or not np.all(np.isfinite(point_a)) or not np.all(np.isfinite(point_b)):
            continue

        meta = triplet_meta.get(triplet_key, {})
        support_count = int(meta.get('support_count') or 0)
        mean_support = float(meta.get('mean_support') or 0.0)
        mean_span = float(meta.get('mean_span_px') or 0.0)
        weight = (
            0.03 +
            0.02 * min(max(support_count - 2, 0), 3) +
            0.03 * np.clip(mean_support, 0.0, 1.0) +
            0.02 * np.clip(mean_span / 240.0, 0.0, 1.0)
        )
        constraints.append({
            'point_a': point_a,
            'point_b': point_b,
            'line_length': line_length,
            'weight': float(np.clip(weight, 0.03, 0.12)),
        })

    return constraints


def _summarize_track_candidate(
    point_3d,
    point_2d_observations,
    cameras,
    K,
    dist_coeffs=None,
    observation_confidences=None,
    point_id=None,
    points_3d_context=None,
    line_support_data=None,
):
    prepared_observations = _prepare_multiview_observations(point_2d_observations, cameras, observation_confidences)
    if len(prepared_observations) < 2:
        return {
            'conflict_class': 'insufficient_support',
            'worst_observations': [],
            'median_error': None,
            'max_error': None,
            'asymmetry': None,
            'min_parallax_deg': None,
            'max_parallax_deg': None,
            'min_baseline': None,
            'max_baseline': None,
            'high_error_count': 0,
        }

    line_constraints = _build_line_constraints_for_point(point_id, points_3d_context, line_support_data)
    refined_point = _refine_point_multiview(
        point_3d,
        prepared_observations,
        cameras,
        K,
        dist_coeffs,
        line_constraints=line_constraints,
    )
    observation_entries = []
    finite_entries = []
    camera_centers = {}

    for cam_id, point_2d, confidence in prepared_observations:
        R, t = cameras[cam_id]
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).reshape(3, 1)
        camera_center = _camera_center_from_pose(R, t)
        camera_centers[cam_id] = camera_center
        point_cam = (R @ refined_point.reshape(3, 1) + t).ravel()
        front = bool(np.all(np.isfinite(point_cam)) and point_cam[2] > 0.01)
        if front:
            projected_point = _project_point_with_pose(refined_point, R, t, K, dist_coeffs)
            error = float(np.linalg.norm(projected_point - point_2d))
            depth = float(point_cam[2])
        else:
            error = float('inf')
            depth = None

        entry = {
            'camera_id': str(cam_id),
            'error': error,
            'front': front,
            'depth': depth,
            'confidence': confidence,
        }
        observation_entries.append(entry)
        if front and np.isfinite(error):
            finite_entries.append(entry)

    finite_errors = np.asarray([entry['error'] for entry in finite_entries], dtype=np.float64)
    if finite_errors.size == 0:
        return {
            'conflict_class': 'insufficient_support',
            'worst_observations': [],
            'median_error': None,
            'max_error': None,
            'asymmetry': None,
            'min_parallax_deg': None,
            'max_parallax_deg': None,
            'min_baseline': None,
            'max_baseline': None,
            'high_error_count': 0,
        }

    median_error = float(np.median(finite_errors))
    max_error = float(np.max(finite_errors))
    asymmetry = float(max_error - median_error)
    mad = float(np.median(np.abs(finite_errors - median_error)))
    robust_scale = max(mad * 1.4826, 0.20)
    high_error_threshold = max(median_error + 2.5 * robust_scale, median_error * 2.0, 1.0)
    high_error_count = int(np.sum(finite_errors >= high_error_threshold))

    sorted_entries = sorted(finite_entries, key=lambda entry: entry['error'], reverse=True)
    worst_observations = [
        {
            'camera_id': entry['camera_id'],
            'error': float(entry['error']),
        }
        for entry in sorted_entries[:3]
    ]

    parallax_values = []
    baseline_values = []
    for camera_a_id, camera_b_id in combinations([entry['camera_id'] for entry in finite_entries], 2):
        center_a = camera_centers[camera_a_id]
        center_b = camera_centers[camera_b_id]
        baseline = float(np.linalg.norm(center_b - center_a))
        baseline_values.append(baseline)

        ray_a = refined_point - center_a
        ray_b = refined_point - center_b
        ray_a_norm = float(np.linalg.norm(ray_a))
        ray_b_norm = float(np.linalg.norm(ray_b))
        if ray_a_norm <= 1e-8 or ray_b_norm <= 1e-8:
            continue
        cos_angle = float(np.clip(np.dot(ray_a, ray_b) / (ray_a_norm * ray_b_norm), -1.0, 1.0))
        parallax_values.append(float(np.degrees(np.arccos(cos_angle))))

    min_parallax = float(min(parallax_values)) if parallax_values else None
    max_parallax = float(max(parallax_values)) if parallax_values else None
    min_baseline = float(min(baseline_values)) if baseline_values else None
    max_baseline = float(max(baseline_values)) if baseline_values else None

    if max_parallax is not None and max_parallax < 0.75:
        conflict_class = 'weak_geometry'
    elif high_error_count <= 1 and asymmetry >= max(0.50, median_error * 1.5):
        conflict_class = 'single_view_conflict'
    elif high_error_count >= max(2, int(np.ceil(len(finite_entries) * 0.4))):
        conflict_class = 'multi_view_conflict'
    elif min_parallax is not None and min_parallax < 0.35 and max_error >= max(1.5, median_error * 2.0):
        conflict_class = 'depth_instability'
    else:
        conflict_class = 'global_tension'

    return {
        'conflict_class': conflict_class,
        'worst_observations': worst_observations,
        'median_error': median_error,
        'max_error': max_error,
        'asymmetry': asymmetry,
        'min_parallax_deg': min_parallax,
        'max_parallax_deg': max_parallax,
        'min_baseline': min_baseline,
        'max_baseline': max_baseline,
        'high_error_count': high_error_count,
    }


def _evaluate_track_geometry_gate(track_summary, track_length, strict_track_consistency=True, candidate_stage="new_point"):
    if not isinstance(track_summary, dict):
        return {
            'accepted': True,
            'geometry_score': 0.5,
            'reason': None,
            'baseline_ratio': None,
        }

    conflict_class = str(track_summary.get('conflict_class') or 'unknown')
    median_error = float(track_summary.get('median_error') or 0.0)
    max_error = float(track_summary.get('max_error') or 0.0)
    asymmetry = float(track_summary.get('asymmetry') or 0.0)
    min_parallax = track_summary.get('min_parallax_deg')
    max_parallax = track_summary.get('max_parallax_deg')
    min_baseline = track_summary.get('min_baseline')
    max_baseline = track_summary.get('max_baseline')
    high_error_count = int(track_summary.get('high_error_count') or 0)
    min_parallax = None if min_parallax is None else float(min_parallax)
    max_parallax = None if max_parallax is None else float(max_parallax)
    min_baseline = None if min_baseline is None else float(min_baseline)
    max_baseline = None if max_baseline is None else float(max_baseline)

    baseline_ratio = None
    if min_baseline is not None and max_baseline is not None and max_baseline > 1e-8:
        baseline_ratio = float(min_baseline / max_baseline)

    parallax_quality = 0.5
    if max_parallax is not None:
        parallax_quality = min(max_parallax / 24.0, 1.0)
        if min_parallax is not None:
            parallax_quality = 0.65 * parallax_quality + 0.35 * min(max(min_parallax, 0.0) / 8.0, 1.0)

    baseline_quality = 0.5
    if baseline_ratio is not None:
        baseline_quality = min(max(baseline_ratio, 0.0) / 0.30, 1.0)

    error_quality = 1.0 / (1.0 + max_error / 1.50)
    asymmetry_quality = 1.0 / (1.0 + asymmetry / 1.20)
    geometry_score = float(
        0.35 * parallax_quality +
        0.20 * baseline_quality +
        0.25 * error_quality +
        0.20 * asymmetry_quality
    )

    weak_pair_geometry = (
        max_parallax is not None and max_parallax < 8.0 and max_error > 0.85
    )
    weak_track_geometry = (
        min_parallax is not None and max_parallax is not None and
        min_parallax < 2.5 and max_parallax < 16.0 and max_error > 0.80
    )
    weak_baseline_support = (
        baseline_ratio is not None and baseline_ratio < 0.08 and
        (max_parallax is None or max_parallax < 18.0) and
        max_error > 0.80
    )
    unstable_conflict = (
        strict_track_consistency and
        conflict_class in {'single_view_conflict', 'multi_view_conflict', 'global_tension'} and
        track_length >= 4 and
        max_error > 1.25 and
        asymmetry > 0.70 and
        high_error_count >= 1
    )

    reject_reasons = []
    if conflict_class == 'weak_geometry':
        reject_reasons.append('weak_geometry')
    if weak_pair_geometry:
        reject_reasons.append('low_parallax_pair')
    if weak_track_geometry:
        reject_reasons.append('weak_track_geometry')
    if weak_baseline_support:
        reject_reasons.append('weak_baseline_support')
    if unstable_conflict:
        reject_reasons.append(f'{conflict_class}_tail')

    if candidate_stage == "backfill":
        if max_error > 1.10 and asymmetry > 0.55:
            reject_reasons.append('backfill_tail')
        if max_parallax is not None and max_parallax < 10.0 and max_error > 0.70:
            reject_reasons.append('backfill_low_parallax')

    return {
        'accepted': len(reject_reasons) == 0,
        'geometry_score': geometry_score,
        'reason': ", ".join(reject_reasons) if reject_reasons else None,
        'baseline_ratio': baseline_ratio,
        'conflict_class': conflict_class,
        'min_parallax_deg': min_parallax,
        'max_parallax_deg': max_parallax,
        'min_baseline': min_baseline,
        'max_baseline': max_baseline,
    }


def _project_point_with_pose(point_3d, R, t, K, dist_coeffs=None):
    point_3d = np.asarray(point_3d, dtype=np.float64).reshape(1, 3)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    K = np.asarray(K, dtype=np.float64)
    dist_coeffs = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    rvec, _ = cv2.Rodrigues(R)
    projected_point, _ = cv2.projectPoints(point_3d, rvec, t, K, dist_coeffs)
    return projected_point.reshape(2)


def _resolve_camera_matrix(K, camera_id=None, camera_intrinsics=None):
    if camera_intrinsics is not None and camera_id is not None:
        camera_key = str(camera_id)
        if camera_key in camera_intrinsics and camera_intrinsics[camera_key] is not None:
            return np.asarray(camera_intrinsics[camera_key], dtype=np.float64)
    return np.asarray(K, dtype=np.float64)


def _get_multiview_refine_settings(refine_mode, observation_count):
    mode = str(refine_mode or "balanced")
    observation_count = max(0, int(observation_count))

    if observation_count <= 2:
        return {
            'skip_refine': True,
            'max_nfev': 0,
            'f_scale': 2.5,
        }

    if mode == "bootstrap_preview":
        if observation_count == 3:
            return {'skip_refine': False, 'max_nfev': 12, 'f_scale': 2.8}
        if observation_count == 4:
            return {'skip_refine': False, 'max_nfev': 10, 'f_scale': 2.8}
        return {'skip_refine': False, 'max_nfev': 8, 'f_scale': 3.0}

    if mode == "quick_preview":
        if observation_count == 3:
            return {'skip_refine': False, 'max_nfev': 18, 'f_scale': 2.5}
        if observation_count == 4:
            return {'skip_refine': False, 'max_nfev': 14, 'f_scale': 2.5}
        return {'skip_refine': False, 'max_nfev': 12, 'f_scale': 2.6}

    if mode in {"full_preview", "balanced_preview", "balanced"}:
        if observation_count == 3:
            return {'skip_refine': False, 'max_nfev': 26, 'f_scale': 2.2}
        if observation_count == 4:
            return {'skip_refine': False, 'max_nfev': 20, 'f_scale': 2.2}
        return {'skip_refine': False, 'max_nfev': 16, 'f_scale': 2.3}

    if mode == "full":
        if observation_count == 3:
            return {'skip_refine': False, 'max_nfev': 32, 'f_scale': 2.0}
        if observation_count == 4:
            return {'skip_refine': False, 'max_nfev': 24, 'f_scale': 2.0}
        return {'skip_refine': False, 'max_nfev': 18, 'f_scale': 2.1}

    return _get_multiview_refine_settings("balanced", observation_count)


def _refine_point_multiview(
    point_3d,
    prepared_observations,
    cameras,
    K,
    dist_coeffs=None,
    refine_mode="balanced",
    line_constraints=None,
    camera_intrinsics=None,
):
    try:
        from scipy.optimize import least_squares
    except Exception:
        return np.asarray(point_3d, dtype=np.float64).reshape(3)

    dist_coeffs = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    initial_point = np.asarray(point_3d, dtype=np.float64).reshape(3)

    refine_settings = _get_multiview_refine_settings(refine_mode, len(prepared_observations))
    if bool(refine_settings.get('skip_refine', False)):
        return initial_point

    def _compute_observation_weights(reference_point):
        errors = []
        for cam_id, point_2d, confidence in prepared_observations:
            R, t = cameras[cam_id]
            R = np.asarray(R, dtype=np.float64)
            t = np.asarray(t, dtype=np.float64).reshape(3, 1)
            point_cam = (R @ reference_point.reshape(3, 1) + t).ravel()
            if point_cam[2] <= 0.01 or not np.all(np.isfinite(point_cam)):
                errors.append(float('inf'))
                continue

            projected_point = _project_point_with_pose(
                reference_point,
                R,
                t,
                _resolve_camera_matrix(K, cam_id, camera_intrinsics),
                dist_coeffs,
            )
            errors.append(float(np.linalg.norm(projected_point - point_2d)))

        finite_errors = np.asarray([err for err in errors if np.isfinite(err)], dtype=np.float64)
        if finite_errors.size < 2:
            return np.ones(len(prepared_observations), dtype=np.float64)

        median_error = float(np.median(finite_errors))
        mad = float(np.median(np.abs(finite_errors - median_error)))
        robust_scale = max(
            0.75,
            median_error + 1.5 * mad,
            float(np.percentile(finite_errors, 70))
        )
        min_weight = 0.10 if finite_errors.size <= 3 else 0.20

        weights = []
        for obs_index, error in enumerate(errors):
            if not np.isfinite(error):
                weights.append(min_weight * 0.5)
                continue

            normalized = error / max(robust_scale, 1e-6)
            weight = 1.0 / (1.0 + normalized * normalized)
            if error <= median_error + max(mad, 0.20):
                weight = max(weight, 0.85)
            observation_confidence = float(prepared_observations[obs_index][2])
            confidence_floor = max(min_weight, observation_confidence * 0.60)
            weight *= max(observation_confidence, 0.25)
            weights.append(float(np.clip(weight, confidence_floor, 1.0)))

        return np.asarray(weights, dtype=np.float64)

    observation_weights = _compute_observation_weights(initial_point)

    line_constraints = list(line_constraints or [])

    def residuals(x):
        point = np.asarray(x, dtype=np.float64).reshape(3)
        reprojection_residuals = []
        for obs_index, (cam_id, point_2d, confidence) in enumerate(prepared_observations):
            weight = float(np.sqrt(max(observation_weights[obs_index], 1e-6)))
            R, t = cameras[cam_id]
            R = np.asarray(R, dtype=np.float64)
            t = np.asarray(t, dtype=np.float64).reshape(3, 1)
            point_cam = (R @ point.reshape(3, 1) + t).ravel()
            if point_cam[2] <= 0.01 or not np.all(np.isfinite(point_cam)):
                reprojection_residuals.extend([50.0 * weight, 50.0 * weight])
                continue

            projected_point = _project_point_with_pose(
                point,
                R,
                t,
                _resolve_camera_matrix(K, cam_id, camera_intrinsics),
                dist_coeffs,
            )
            reprojection_residuals.extend(((projected_point - point_2d) * weight).tolist())
        if line_constraints:
            for constraint in line_constraints:
                point_a = np.asarray(constraint['point_a'], dtype=np.float64).reshape(3)
                point_b = np.asarray(constraint['point_b'], dtype=np.float64).reshape(3)
                direction = point_b - point_a
                direction_norm_sq = float(np.dot(direction, direction))
                if direction_norm_sq <= 1e-10:
                    continue
                closest_factor = float(np.dot(point - point_a, direction) / direction_norm_sq)
                closest_point = point_a + closest_factor * direction
                line_length = max(float(constraint.get('line_length', np.sqrt(direction_norm_sq))), 1e-6)
                line_weight = float(np.sqrt(max(constraint.get('weight', 0.08), 1e-6)))
                normalized_offset = ((point - closest_point) / line_length) * line_weight
                reprojection_residuals.extend(normalized_offset.tolist())
        return np.asarray(reprojection_residuals, dtype=np.float64)

    try:
        initial_cost = float(np.sum(residuals(initial_point) ** 2))
        result = least_squares(
            residuals,
            initial_point,
            method='trf',
            loss='soft_l1',
            f_scale=float(refine_settings.get('f_scale', 2.0)),
            max_nfev=int(refine_settings.get('max_nfev', 20)),
            verbose=0,
        )
        refined_point = np.asarray(result.x if result.success or np.isfinite(result.cost) else initial_point, dtype=np.float64).reshape(3)
        refined_cost = float(np.sum(residuals(refined_point) ** 2))
        if refined_cost <= initial_cost:
            return refined_point
    except Exception:
        pass

    return initial_point


def _get_multiview_thresholds(track_length, acceptance_profile="default"):
    profile = str(acceptance_profile or "default")

    if profile == "bootstrap_pair":
        if track_length <= 2:
            return {
                'inlier_threshold': 9.0,
                'max_mean_limit': 8.5,
                'max_median_limit': 8.5,
                'max_single_limit': 12.0,
                'min_inlier_ratio': 1.0,
            }
        if track_length == 3:
            return {
                'inlier_threshold': 8.0,
                'max_mean_limit': 7.0,
                'max_median_limit': 6.5,
                'max_single_limit': 16.0,
                'min_inlier_ratio': 2.0 / 3.0,
            }

    if track_length <= 2:
        return {
            'inlier_threshold': 6.0,
            'max_mean_limit': 5.0,
            'max_median_limit': 5.0,
            'max_single_limit': 12.0,
            'min_inlier_ratio': 1.0,
        }
    if track_length == 3:
        return {
            'inlier_threshold': 6.0,
            'max_mean_limit': 6.0,
            'max_median_limit': 5.0,
            'max_single_limit': 18.0,
            'min_inlier_ratio': 0.75,
        }
    return {
        'inlier_threshold': 6.0,
        'max_mean_limit': 8.0,
        'max_median_limit': 6.0,
        'max_single_limit': 25.0,
        'min_inlier_ratio': 0.75,
    }


def evaluate_multiview_point(
    point_3d,
    point_2d_observations,
    cameras,
    K,
    dist_coeffs=None,
    allow_subset=False,
    min_subset_views=2,
    acceptance_profile="default",
    observation_confidences=None,
    refine_mode="balanced",
    point_id=None,
    points_3d_context=None,
    line_support_data=None,
    camera_intrinsics=None,
):
    prepared_observations = _prepare_multiview_observations(point_2d_observations, cameras, observation_confidences)
    if len(prepared_observations) < 2:
        return False, np.asarray(point_3d, dtype=np.float64).reshape(3), None

    def _evaluate_subset(entries):
        finite_entries = [entry for entry in entries if np.isfinite(entry['error'])]
        if len(finite_entries) < 2:
            return False, None

        errors = np.array([entry['error'] for entry in finite_entries], dtype=np.float64)
        camera_ids = [entry['camera_id'] for entry in entries]
        used_camera_ids = [entry['camera_id'] for entry in finite_entries]
        dropped_camera_ids = [entry['camera_id'] for entry in entries if entry['camera_id'] not in used_camera_ids]
        track_length = len(finite_entries)
        thresholds = _get_multiview_thresholds(track_length, acceptance_profile=acceptance_profile)
        inlier_threshold = thresholds['inlier_threshold']
        inlier_ratio = float(np.mean(errors <= inlier_threshold))
        median_error = float(np.median(errors))
        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))
        front_ratio = float(np.mean([entry['front'] for entry in entries])) if entries else 0.0

        accepted = (
            mean_error <= thresholds['max_mean_limit'] and
            median_error <= thresholds['max_median_limit'] and
            max_error <= thresholds['max_single_limit'] and
            inlier_ratio >= thresholds['min_inlier_ratio']
        )

        metrics = {
            'track_length': track_length,
            'mean_error': mean_error,
            'median_error': median_error,
            'max_error': max_error,
            'inlier_ratio': inlier_ratio,
            'front_ratio': front_ratio,
            'camera_ids': camera_ids,
            'used_camera_ids': used_camera_ids,
            'dropped_camera_ids': dropped_camera_ids,
            'errors': [entry['error'] for entry in entries],
            'thresholds': dict(thresholds),
        }
        return accepted, metrics

    def _build_observation_entries(candidate_point):
        entries = []
        candidate_point = np.asarray(candidate_point, dtype=np.float64).reshape(3)
        for cam_id, point_2d, confidence in prepared_observations:
            R, t = cameras[cam_id]
            R = np.asarray(R, dtype=np.float64)
            t = np.asarray(t, dtype=np.float64).reshape(3, 1)
            point_cam = (R @ candidate_point.reshape(3, 1) + t).ravel()
            front = bool(np.all(np.isfinite(point_cam)) and point_cam[2] > 0.01)

            if front:
                projected_point = _project_point_with_pose(
                    candidate_point,
                    R,
                    t,
                    _resolve_camera_matrix(K, cam_id, camera_intrinsics),
                    dist_coeffs,
                )
                error = float(np.linalg.norm(projected_point - point_2d))
            else:
                error = float('inf')

            entries.append({
                'camera_id': cam_id,
                'error': error,
                'front': front,
                'confidence': confidence,
            })
        return entries

    initial_point = np.asarray(point_3d, dtype=np.float64).reshape(3)
    raw_observation_entries = _build_observation_entries(initial_point)
    raw_accepted, raw_metrics = _evaluate_subset(raw_observation_entries)
    preview_fast_mode = str(refine_mode or "balanced") in {
        "bootstrap_preview",
        "quick_preview",
        "full_preview",
        "balanced_preview",
    }

    if raw_metrics is not None:
        raw_thresholds = raw_metrics.get('thresholds', {})
        catastrophic_reject = (
            raw_metrics['front_ratio'] < 0.34 or
            raw_metrics['mean_error'] > max(float(raw_thresholds.get('max_mean_limit', 8.0)) * 3.0, 10.0) or
            raw_metrics['max_error'] > max(float(raw_thresholds.get('max_single_limit', 25.0)) * 2.5, 18.0)
        )
        if catastrophic_reject:
            return False, initial_point, raw_metrics

        strong_preview_accept = (
            preview_fast_mode and
            raw_accepted and
            raw_metrics['track_length'] >= 3 and
            raw_metrics['mean_error'] <= max(float(raw_thresholds.get('max_mean_limit', 8.0)) * 0.45, 0.90) and
            raw_metrics['max_error'] <= max(float(raw_thresholds.get('max_single_limit', 25.0)) * 0.35, 1.50) and
            raw_metrics['front_ratio'] >= 0.99
        )
        if strong_preview_accept:
            return True, initial_point, raw_metrics

    enable_line_constraints = str(refine_mode or "balanced") not in {
        "bootstrap_preview",
        "quick_preview",
        "full_preview",
        "balanced_preview",
    }
    refined_point = _refine_point_multiview(
        point_3d,
        prepared_observations,
        cameras,
        K,
        dist_coeffs,
        refine_mode=refine_mode,
        line_constraints=(
            _build_line_constraints_for_point(point_id, points_3d_context, line_support_data)
            if enable_line_constraints else None
        ),
        camera_intrinsics=camera_intrinsics,
    )
    observation_entries = _build_observation_entries(refined_point)

    full_entries = [entry for entry in observation_entries if entry['front'] and np.isfinite(entry['error'])]
    accepted, metrics = _evaluate_subset(full_entries)
    if accepted or not allow_subset or len(full_entries) < max(3, int(min_subset_views) + 1):
        return accepted, refined_point, metrics

    best_subset_metrics = metrics
    front_sorted = sorted(full_entries, key=lambda entry: entry['error'])
    min_subset_views = max(2, int(min_subset_views))

    for subset_size in range(len(front_sorted) - 1, min_subset_views - 1, -1):
        subset_entries = front_sorted[:subset_size]
        subset_accepted, subset_metrics = _evaluate_subset(subset_entries)
        if subset_metrics is None:
            continue
        if subset_accepted:
            return True, refined_point, subset_metrics
        if best_subset_metrics is None or subset_metrics['mean_error'] < best_subset_metrics['mean_error']:
            best_subset_metrics = subset_metrics

    return False, refined_point, best_subset_metrics


def sanitize_points_for_camera(
    points_3d,
    cameras,
    camera_points,
    K,
    dist_coeffs,
    focus_camera_id,
    protected_point_ids=None,
    strict_track_consistency=True,
    debug_logging=False,
    observation_confidences=None,
    secondary_seed_points=None,
    line_support_data=None,
    soft_mode=False,
    camera_intrinsics=None,
):
    focus_camera_id = str(focus_camera_id)
    focus_observations = camera_points.get(focus_camera_id, {})
    if not focus_observations:
        return {'removed_points': 0, 'removed_observations': 0, 'refined_points': 0}

    protected_point_ids = {point_id for point_id in (protected_point_ids or [])}
    removed_points = 0
    removed_observations = 0
    downweighted_observations = 0
    refined_points = 0

    point_ids = [point_id for point_id in focus_observations.keys() if point_id in points_3d]
    for point_id in point_ids:
        if point_id in protected_point_ids:
            _debug_print(
                debug_logging,
                f"Локальная очистка: пропуск новой точки {point_id} для камеры {focus_camera_id}"
            )
            continue

        point_2d_observations = {
            str(camera_id): camera_points[str(camera_id)][point_id]
            for camera_id in cameras.keys()
            if point_id in camera_points.get(str(camera_id), {})
        }
        point_observation_confidences = _extract_point_observation_confidences(
            observation_confidences,
            point_id,
            point_2d_observations.keys(),
        )

        if len(point_2d_observations) < 2:
            continue

        accepted, refined_point, metrics = evaluate_multiview_point(
            points_3d[point_id],
            point_2d_observations,
            cameras,
            K,
            dist_coeffs,
            observation_confidences=point_observation_confidences,
            point_id=point_id,
            points_3d_context=points_3d,
            line_support_data=line_support_data,
            camera_intrinsics=camera_intrinsics,
        )
        if accepted:
            original_point = np.asarray(points_3d[point_id], dtype=np.float64).reshape(3)
            if np.linalg.norm(refined_point - original_point) > 1e-4:
                points_3d[point_id] = refined_point.astype(np.float32)
                refined_points += 1
            continue

        observations_without_focus = {
            camera_id: point_2d
            for camera_id, point_2d in point_2d_observations.items()
            if camera_id != focus_camera_id
        }
        if len(observations_without_focus) >= 2:
            accepted_without_focus, refined_without_focus, metrics_without_focus = evaluate_multiview_point(
                points_3d[point_id],
                observations_without_focus,
                cameras,
                K,
                dist_coeffs,
                observation_confidences=point_observation_confidences,
                point_id=point_id,
                points_3d_context=points_3d,
                line_support_data=line_support_data,
                camera_intrinsics=camera_intrinsics,
            )
            if accepted_without_focus:
                if strict_track_consistency and not soft_mode:
                    if secondary_seed_points is not None:
                        secondary_seed_points[point_id] = np.asarray(refined_without_focus, dtype=np.float32).reshape(3)
                    del points_3d[point_id]
                    removed_points += 1
                    if debug_logging:
                        _debug_print(
                            debug_logging,
                            f"Локальная очистка: точка {point_id} удалена целиком из реконструкции, "
                            f"потому что согласуется только без камеры {focus_camera_id}"
                        )
                else:
                    downgraded_confidence = _downweight_observation_confidence(
                        observation_confidences,
                        focus_camera_id,
                        point_id,
                    )
                    points_3d[point_id] = refined_without_focus.astype(np.float32)
                    downweighted_observations += 1
                    if debug_logging:
                        _debug_print(
                            debug_logging,
                            f"Локальная очистка: наблюдение точки {point_id} в камере {focus_camera_id} мягко downweight "
                            f"до confidence={downgraded_confidence if downgraded_confidence is not None else 1.0:.2f}, "
                            f"ошибка full={metrics['mean_error']:.2f}px -> keep={metrics_without_focus['mean_error']:.2f}px"
                        )
                continue

        if soft_mode:
            if point_id in camera_points.get(focus_camera_id, {}):
                downgraded_confidence = _downweight_observation_confidence(
                    observation_confidences,
                    focus_camera_id,
                    point_id,
                )
                downweighted_observations += 1
        else:
            del points_3d[point_id]
            removed_points += 1
        if debug_logging and metrics is not None:
            _debug_print(
                debug_logging,
                f"Локальная очистка: удалена точка {point_id}, track={metrics['track_length']}, "
                f"mean={metrics['mean_error']:.2f}px, max={metrics['max_error']:.2f}px"
            )

    return {
        'removed_points': removed_points,
        'removed_observations': removed_observations,
        'downweighted_observations': downweighted_observations,
        'refined_points': refined_points,
    }


def prune_focus_conflicting_tracks(
    points_3d,
    cameras,
    camera_points,
    K,
    dist_coeffs,
    focus_camera_id,
    min_track_length=4,
    strict_track_consistency=True,
    debug_logging=False,
    observation_confidences=None,
    secondary_seed_points=None,
    line_support_data=None,
    soft_mode=False,
    camera_intrinsics=None,
):
    focus_camera_id = str(focus_camera_id)
    focus_observations = camera_points.get(focus_camera_id, {})
    if not focus_observations:
        return {'removed_points': 0, 'removed_observations': 0, 'refined_points': 0}

    min_track_length = max(4, int(min_track_length))
    removed_points = 0
    removed_observations = 0
    downweighted_observations = 0
    refined_points = 0
    removed_details = []

    point_ids = [point_id for point_id in sorted(focus_observations.keys(), key=_stable_sort_key) if point_id in points_3d]
    for point_id in point_ids:
        point_2d_observations = {
            str(camera_id): camera_points[str(camera_id)][point_id]
            for camera_id in cameras.keys()
            if point_id in camera_points.get(str(camera_id), {})
        }
        if len(point_2d_observations) < min_track_length:
            continue

        point_observation_confidences = _extract_point_observation_confidences(
            observation_confidences,
            point_id,
            point_2d_observations.keys(),
        )

        accepted, refined_point, metrics = evaluate_multiview_point(
            points_3d[point_id],
            point_2d_observations,
            cameras,
            K,
            dist_coeffs,
            allow_subset=False,
            min_subset_views=min_track_length,
            observation_confidences=point_observation_confidences,
            point_id=point_id,
            points_3d_context=points_3d,
            line_support_data=line_support_data,
            camera_intrinsics=camera_intrinsics,
        )
        if metrics is None or not accepted:
            continue

        refined_point = np.asarray(refined_point, dtype=np.float64).reshape(3)
        original_point = np.asarray(points_3d[point_id], dtype=np.float64).reshape(3)
        if np.linalg.norm(refined_point - original_point) > 1e-4:
            points_3d[point_id] = refined_point.astype(np.float32)
            refined_points += 1

        error_by_camera = {
            str(camera_id): float(error)
            for camera_id, error in zip(metrics.get('camera_ids', []), metrics.get('errors', []))
            if np.isfinite(error)
        }
        if focus_camera_id not in error_by_camera:
            continue

        focus_error = float(error_by_camera[focus_camera_id])
        other_errors = np.asarray(
            [error for camera_id, error in error_by_camera.items() if camera_id != focus_camera_id],
            dtype=np.float64,
        )
        if other_errors.size < max(2, min_track_length - 2):
            continue

        current_mean = float(metrics.get('mean_error', float('inf')))
        current_median = float(metrics.get('median_error', float('inf')))
        current_max = float(metrics.get('max_error', float('inf')))
        other_median = float(np.median(other_errors))
        if soft_mode:
            early_skip_threshold = max(other_median + 0.90, current_median + 0.50, 2.0)
        else:
            early_skip_threshold = max(other_median + 0.45, current_median + 0.25, 1.10)
        if focus_error < early_skip_threshold:
            continue

        track_summary = _summarize_track_candidate(
            refined_point,
            point_2d_observations,
            cameras,
            K,
            dist_coeffs,
            observation_confidences=point_observation_confidences,
            point_id=point_id,
            points_3d_context=points_3d,
            line_support_data=line_support_data,
        )
        conflict_class = str(track_summary.get('conflict_class') or '')
        if conflict_class not in {'single_view_conflict', 'global_tension', 'multi_view_conflict', 'depth_instability'}:
            continue

        observations_without_focus = {
            camera_id: point_2d
            for camera_id, point_2d in point_2d_observations.items()
            if camera_id != focus_camera_id
        }
        if len(observations_without_focus) < max(2, min_track_length - 1):
            continue

        confidences_without_focus = {
            camera_id: point_observation_confidences.get(camera_id, 1.0)
            for camera_id in observations_without_focus.keys()
        }
        accepted_without_focus, refined_without_focus, metrics_without_focus = evaluate_multiview_point(
            refined_point,
            observations_without_focus,
            cameras,
            K,
            dist_coeffs,
            allow_subset=False,
            min_subset_views=max(2, min_track_length - 1),
            observation_confidences=confidences_without_focus,
            point_id=point_id,
            points_3d_context=points_3d,
            line_support_data=line_support_data,
            camera_intrinsics=camera_intrinsics,
        )
        if not accepted_without_focus or metrics_without_focus is None:
            continue

        if soft_mode:
            culprit_delta = 0.80
            culprit_abs = 2.0
            improve_mean_delta = 0.25
            improve_max_delta = 0.70
        else:
            culprit_delta = 0.40
            culprit_abs = 1.15
            improve_mean_delta = 0.12
            improve_max_delta = 0.35
        improved_mean = float(metrics_without_focus.get('mean_error', float('inf'))) <= current_mean - improve_mean_delta
        improved_max = float(metrics_without_focus.get('max_error', float('inf'))) <= current_max - improve_max_delta
        focus_is_culprit = (
            focus_error - other_median >= culprit_delta and
            focus_error >= max(current_median + culprit_delta, culprit_abs)
        )
        if not (focus_is_culprit and (improved_mean or improved_max)):
            continue

        if strict_track_consistency and not soft_mode:
            if secondary_seed_points is not None:
                secondary_seed_points[point_id] = np.asarray(refined_without_focus, dtype=np.float32).reshape(3)
            del points_3d[point_id]
            removed_points += 1
            removed_details.append(
                (
                    point_id,
                    current_mean,
                    current_max,
                    focus_error,
                    float(metrics_without_focus.get('mean_error', float('inf'))),
                    float(metrics_without_focus.get('max_error', float('inf'))),
                )
            )
        else:
            if point_id in camera_points.get(focus_camera_id, {}):
                downgraded_confidence = _downweight_observation_confidence(
                    observation_confidences,
                    focus_camera_id,
                    point_id,
                )
                points_3d[point_id] = np.asarray(refined_without_focus, dtype=np.float32).reshape(3)
                downweighted_observations += 1
                if debug_logging:
                    _debug_print(
                        debug_logging,
                        f"Локальный strict-gate: наблюдение точки {point_id} в камере {focus_camera_id} мягко downweight "
                        f"до confidence={downgraded_confidence if downgraded_confidence is not None else 1.0:.2f}"
                    )

    if removed_details:
        print(f"Локальный strict-gate камеры {focus_camera_id}:")
        for point_id, current_mean, current_max, focus_error, repaired_mean, repaired_max in removed_details[:8]:
            print(
                f"  - Точка {point_id}: удалена целиком, т.к. камера {focus_camera_id} "
                f"ломала трек (focus={focus_error:.2f}px, mean={current_mean:.2f}px, max={current_max:.2f}px -> "
                f"без камеры mean={repaired_mean:.2f}px, max={repaired_max:.2f}px)"
            )

    return {
        'removed_points': int(removed_points),
        'removed_observations': int(removed_observations),
        'downweighted_observations': int(downweighted_observations),
        'refined_points': int(refined_points),
    }


def _triangulate_global_point_from_pair(
    camera_a_id,
    camera_b_id,
    point_observations,
    cameras,
    K,
    dist_coeffs=None,
    debug_logging=False,
):
    camera_a_id = str(camera_a_id)
    camera_b_id = str(camera_b_id)
    if camera_a_id not in cameras or camera_b_id not in cameras:
        return None
    if camera_a_id not in point_observations or camera_b_id not in point_observations:
        return None

    R_a, t_a = cameras[camera_a_id]
    R_b, t_b = cameras[camera_b_id]
    R_a = np.asarray(R_a, dtype=np.float64)
    R_b = np.asarray(R_b, dtype=np.float64)
    t_a = np.asarray(t_a, dtype=np.float64).reshape(3, 1)
    t_b = np.asarray(t_b, dtype=np.float64).reshape(3, 1)

    R_rel = R_b @ R_a.T
    t_rel = (t_b - R_rel @ t_a).reshape(3)

    pts_a = np.asarray([point_observations[camera_a_id]], dtype=np.float32)
    pts_b = np.asarray([point_observations[camera_b_id]], dtype=np.float32)
    points_local, mask = triangulate_points(
        pts1=pts_a,
        pts2=pts_b,
        K=K,
        R=R_rel,
        t=t_rel,
        robust=True,
        reprojection_threshold=8.0,
        parallax_threshold=1.0,
        min_parallax_for_validation=0.35,
        dist_coeffs=dist_coeffs,
        debug_logging=debug_logging,
    )
    if points_local.size == 0 or mask.size == 0 or not bool(mask[0]):
        return None

    point_local = np.asarray(points_local[0], dtype=np.float64).reshape(3)
    point_global = R_a.T @ (point_local - t_a.ravel())
    if not np.all(np.isfinite(point_global)):
        return None

    if not validate_triangulated_point(point_global, (R_a, t_a), (R_b, t_b)):
        return None

    return point_global


def retriangulate_high_error_points(
    points_3d,
    cameras,
    camera_points,
    K,
    dist_coeffs=None,
    max_points=4,
    min_track_length=3,
    debug_logging=False,
    observation_confidences=None,
    line_support_data=None,
    camera_intrinsics=None,
):
    if not points_3d or not cameras:
        return 0

    point_entries = []
    observation_errors = []
    for point_id in sorted(points_3d.keys(), key=_stable_sort_key):
        point_observations = {
            str(camera_id): camera_points[str(camera_id)][point_id]
            for camera_id in sorted(cameras.keys(), key=_stable_sort_key)
            if point_id in camera_points.get(str(camera_id), {})
        }
        point_observation_confidences = _extract_point_observation_confidences(
            observation_confidences,
            point_id,
            point_observations.keys(),
        )
        if len(point_observations) < max(2, int(min_track_length)):
            continue

        accepted, refined_point, metrics = evaluate_multiview_point(
            points_3d[point_id],
            point_observations,
            cameras,
            K,
            dist_coeffs,
            allow_subset=False,
            min_subset_views=min_track_length,
            observation_confidences=point_observation_confidences,
            point_id=point_id,
            points_3d_context=points_3d,
            line_support_data=line_support_data,
            camera_intrinsics=camera_intrinsics,
        )
        if metrics is None:
            continue

        point_entries.append({
            'point_id': point_id,
            'observations': point_observations,
            'track_length': len(point_observations),
            'accepted': bool(accepted),
            'point': np.asarray(refined_point, dtype=np.float64).reshape(3),
            'metrics': metrics,
        })
        finite_errors = [error for error in metrics.get('errors', []) if np.isfinite(error)]
        observation_errors.extend(finite_errors)

    if len(point_entries) < 2 or len(observation_errors) < 8:
        return 0

    observation_errors = np.asarray(observation_errors, dtype=np.float64)
    global_p95 = float(np.percentile(observation_errors, 95))
    global_median = float(np.median(observation_errors))
    global_mad = float(np.median(np.abs(observation_errors - global_median))) * 1.4826
    global_scale = max(global_mad, 0.75)

    candidates = []
    for entry in point_entries:
        metrics = entry['metrics']
        max_error = float(metrics.get('max_error', 0.0))
        mean_error = float(metrics.get('mean_error', 0.0))
        median_error = float(metrics.get('median_error', 0.0))
        if (
            max_error <= max(global_p95 * 1.08, global_median + 3.0 * global_scale, 4.5) and
            mean_error <= max(global_median * 1.4, 2.0)
        ):
            continue
        severity = max(
            max_error / max(global_p95, 1e-6),
            mean_error / max(global_median + global_scale, 1e-6),
            median_error / max(global_median, 1e-6),
        )
        candidates.append((severity, entry))

    if not candidates:
        print("Point retriangulation: кандидатов для уточнения не найдено")
        return 0

    refined_count = 0
    print("Point retriangulation для плохих 3D-треков:")
    for _, entry in sorted(candidates, key=lambda item: item[0], reverse=True)[:max(1, int(max_points))]:
        point_id = entry['point_id']
        point_observations = entry['observations']
        current_metrics = entry['metrics']
        current_mean = float(current_metrics.get('mean_error', float('inf')))
        current_max = float(current_metrics.get('max_error', float('inf')))
        current_median = float(current_metrics.get('median_error', float('inf')))

        best_candidate = None
        camera_ids = sorted(point_observations.keys(), key=_stable_sort_key)
        for camera_a_id, camera_b_id in combinations(camera_ids, 2):
            candidate_point = _triangulate_global_point_from_pair(
                camera_a_id,
                camera_b_id,
                point_observations,
                cameras,
                K,
                dist_coeffs=dist_coeffs,
                debug_logging=debug_logging,
            )
            if candidate_point is None:
                continue

            candidate_accepted, candidate_refined_point, candidate_metrics = evaluate_multiview_point(
                candidate_point,
                point_observations,
                cameras,
                K,
                dist_coeffs,
                allow_subset=False,
                min_subset_views=min_track_length,
                observation_confidences=point_observation_confidences,
                point_id=point_id,
                points_3d_context=points_3d,
                line_support_data=line_support_data,
                camera_intrinsics=camera_intrinsics,
            )
            if not candidate_accepted or candidate_metrics is None:
                continue

            candidate_mean = float(candidate_metrics.get('mean_error', float('inf')))
            candidate_max = float(candidate_metrics.get('max_error', float('inf')))
            candidate_median = float(candidate_metrics.get('median_error', float('inf')))
            improvement_score = (
                (current_max - candidate_max) * 0.60 +
                (current_mean - candidate_mean) * 0.30 +
                (current_median - candidate_median) * 0.10
            )

            if best_candidate is None or (
                improvement_score > best_candidate['improvement_score'] + 1e-6 or
                (
                    abs(improvement_score - best_candidate['improvement_score']) <= 1e-6 and
                    candidate_max < best_candidate['metrics']['max_error']
                )
            ):
                best_candidate = {
                    'point': np.asarray(candidate_refined_point, dtype=np.float64).reshape(3),
                    'metrics': candidate_metrics,
                    'pair': (camera_a_id, camera_b_id),
                    'improvement_score': float(improvement_score),
                }

        if best_candidate is None:
            continue

        best_mean = float(best_candidate['metrics'].get('mean_error', float('inf')))
        best_max = float(best_candidate['metrics'].get('max_error', float('inf')))
        best_median = float(best_candidate['metrics'].get('median_error', float('inf')))
        significant_improvement = (
            best_mean <= current_mean - 0.08 or
            best_max <= current_max - 0.50 or
            (
                best_mean <= current_mean * 0.93 and
                best_max <= current_max + 0.10 and
                best_median <= current_median
            )
        )
        if not significant_improvement:
            continue

        points_3d[point_id] = best_candidate['point'].astype(np.float32)
        refined_count += 1
        print(
            f"  - Точка {point_id}: "
            f"mean {current_mean:.2f}px -> {best_mean:.2f}px, "
            f"max {current_max:.2f}px -> {best_max:.2f}px, "
            f"pair {best_candidate['pair'][0]}-{best_candidate['pair'][1]}"
        )

    if refined_count <= 0:
        print("Point retriangulation: улучшений не найдено")

    return refined_count


def repair_asymmetric_point_tracks(
    points_3d,
    cameras,
    camera_points,
    K,
    dist_coeffs=None,
    max_points=6,
    min_track_length=3,
    max_removed_observations=2,
    strict_track_consistency=True,
    debug_logging=False,
    observation_confidences=None,
    secondary_seed_points=None,
    line_support_data=None,
    camera_intrinsics=None,
):
    if not points_3d or not cameras:
        return {'repaired_points': 0, 'removed_observations': 0, 'removed_points': 0}

    required_track_length = max(3, int(min_track_length))
    candidates = []
    for point_id in sorted(points_3d.keys(), key=_stable_sort_key):
        point_observations = {
            str(camera_id): camera_points[str(camera_id)][point_id]
            for camera_id in sorted(cameras.keys(), key=_stable_sort_key)
            if point_id in camera_points.get(str(camera_id), {})
        }
        point_observation_confidences = _extract_point_observation_confidences(
            observation_confidences,
            point_id,
            point_observations.keys(),
        )
        if len(point_observations) < required_track_length:
            continue

        accepted, refined_point, metrics = evaluate_multiview_point(
            points_3d[point_id],
            point_observations,
            cameras,
            K,
            dist_coeffs,
            allow_subset=False,
            min_subset_views=required_track_length,
            observation_confidences=point_observation_confidences,
            point_id=point_id,
            points_3d_context=points_3d,
            line_support_data=line_support_data,
            camera_intrinsics=camera_intrinsics,
        )
        if metrics is None:
            continue

        finite_entries = []
        for camera_id, error in zip(metrics.get('camera_ids', []), metrics.get('errors', [])):
            if np.isfinite(error):
                finite_entries.append((float(error), str(camera_id)))
        if len(finite_entries) < required_track_length:
            continue

        finite_errors = np.asarray([entry[0] for entry in finite_entries], dtype=np.float64)
        median_error = float(np.median(finite_errors))
        max_error = float(np.max(finite_errors))
        asymmetry = max_error - median_error
        max_to_median = max_error / max(median_error, 1e-6)

        if asymmetry < 0.75 and max_to_median < 2.25:
            continue

        severity = (
            asymmetry * 0.65 +
            max_to_median * 0.25 +
            float(metrics.get('mean_error', median_error)) * 0.10
        )
        candidates.append({
            'point_id': point_id,
            'observations': point_observations,
            'metrics': metrics,
            'entries': sorted(finite_entries, reverse=True),
            'severity': float(severity),
            'accepted': bool(accepted),
            'refined_point': np.asarray(refined_point, dtype=np.float64).reshape(3),
        })

    if not candidates:
        print("Point-track repair: асимметричных треков не найдено")
        return {'repaired_points': 0, 'removed_observations': 0, 'removed_points': 0}

    repaired_points = 0
    removed_observations = 0
    removed_points = 0
    removed_by_camera = {}
    print("Point-track repair для асимметричных треков:")

    for candidate in sorted(candidates, key=lambda item: item['severity'], reverse=True)[:max(1, int(max_points))]:
        point_id = candidate['point_id']
        point_observations = candidate['observations']
        current_metrics = candidate['metrics']
        current_mean = float(current_metrics.get('mean_error', float('inf')))
        current_median = float(current_metrics.get('median_error', float('inf')))
        current_max = float(current_metrics.get('max_error', float('inf')))
        sorted_entries = list(candidate['entries'])
        camera_error_map = {camera_id: error for error, camera_id in sorted_entries}
        minimum_kept_views = 2 if len(sorted_entries) <= 3 else required_track_length

        best_fix = None
        max_drop = min(int(max_removed_observations), len(sorted_entries) - minimum_kept_views)
        if max_drop <= 0:
            continue

        for drop_count in range(1, max_drop + 1):
            dropped_camera_ids = [camera_id for _, camera_id in sorted_entries[:drop_count]]
            kept_camera_ids = [
                camera_id for camera_id in sorted(point_observations.keys(), key=_stable_sort_key)
                if camera_id not in dropped_camera_ids
            ]
            if len(kept_camera_ids) < minimum_kept_views:
                continue

            kept_observations = {camera_id: point_observations[camera_id] for camera_id in kept_camera_ids}
            kept_current_errors = np.asarray(
                [camera_error_map[camera_id] for camera_id in kept_camera_ids if camera_id in camera_error_map],
                dtype=np.float64,
            )
            dropped_errors = np.asarray(
                [camera_error_map[camera_id] for camera_id in dropped_camera_ids if camera_id in camera_error_map],
                dtype=np.float64,
            )
            if kept_current_errors.size < minimum_kept_views or dropped_errors.size < 1:
                continue

            kept_median = float(np.median(kept_current_errors))
            dropped_min = float(np.min(dropped_errors))
            dropped_median = float(np.median(dropped_errors))
            clearly_worse = (
                dropped_min >= kept_median + 0.35 or
                dropped_median >= max(kept_median * 1.9, kept_median + 0.50)
            )
            if not clearly_worse:
                continue

            subset_min_views = 2 if len(kept_camera_ids) <= 2 else required_track_length
            subset_profile = "bootstrap_pair" if len(kept_camera_ids) <= 2 else "default"
            for camera_a_id, camera_b_id in combinations(kept_camera_ids, 2):
                candidate_point = _triangulate_global_point_from_pair(
                    camera_a_id,
                    camera_b_id,
                    kept_observations,
                    cameras,
                    K,
                    dist_coeffs=dist_coeffs,
                    debug_logging=debug_logging,
                )
                if candidate_point is None:
                    continue

                subset_accepted, subset_refined_point, subset_metrics = evaluate_multiview_point(
                    candidate_point,
                    kept_observations,
                    cameras,
                    K,
                    dist_coeffs,
                    allow_subset=False,
                    min_subset_views=subset_min_views,
                    acceptance_profile=subset_profile,
                    observation_confidences=point_observation_confidences,
                    point_id=point_id,
                    points_3d_context=points_3d,
                    line_support_data=line_support_data,
                    camera_intrinsics=camera_intrinsics,
                )
                if not subset_accepted or subset_metrics is None:
                    continue

                subset_mean = float(subset_metrics.get('mean_error', float('inf')))
                subset_median = float(subset_metrics.get('median_error', float('inf')))
                subset_max = float(subset_metrics.get('max_error', float('inf')))
                if len(kept_camera_ids) <= 2:
                    pair_safe = (
                        subset_mean <= min(1.50, kept_median + 0.40) and
                        subset_max <= min(2.50, max(kept_median + 1.20, 1.20)) and
                        dropped_min >= max(kept_median * 2.1, kept_median + 0.55)
                    )
                    if not pair_safe:
                        continue
                improvement_score = (
                    (current_max - subset_max) * 0.55 +
                    (current_mean - subset_mean) * 0.30 +
                    (current_median - subset_median) * 0.15
                )
                significant_improvement = (
                    subset_mean <= current_mean - 0.10 or
                    subset_max <= current_max - 0.80 or
                    (
                        subset_mean <= current_mean * 0.88 and
                        subset_max <= current_max * 0.75 and
                        subset_median <= current_median
                    )
                )
                if not significant_improvement:
                    continue

                candidate_fix = {
                    'point': np.asarray(subset_refined_point, dtype=np.float64).reshape(3),
                    'metrics': subset_metrics,
                    'pair': (camera_a_id, camera_b_id),
                    'dropped_camera_ids': dropped_camera_ids,
                    'improvement_score': float(improvement_score),
                }
                if best_fix is None or (
                    candidate_fix['improvement_score'] > best_fix['improvement_score'] + 1e-6 or
                    (
                        abs(candidate_fix['improvement_score'] - best_fix['improvement_score']) <= 1e-6 and
                        float(candidate_fix['metrics']['max_error']) < float(best_fix['metrics']['max_error'])
                    )
                ):
                    best_fix = candidate_fix

        if best_fix is None:
            continue

        if strict_track_consistency and best_fix['dropped_camera_ids']:
            kept_views = len(best_fix.get('metrics', {}).get('camera_ids', [])) if best_fix.get('metrics') else 0
            if kept_views < required_track_length:
                strict_drop = True
                if kept_views == 2 and 'pair' in best_fix:
                    # Проверяем параллакс и ошибку для сохранения 2-видовой точки
                    cam_a, cam_b = best_fix['pair']
                    if cam_a in cameras and cam_b in cameras:
                        R_a, t_a = cameras[cam_a]
                        R_b, t_b = cameras[cam_b]
                        C_a = -np.asarray(R_a).T @ np.asarray(t_a).ravel()
                        C_b = -np.asarray(R_b).T @ np.asarray(t_b).ravel()
                        pt = np.asarray(best_fix['point'])
                        ray_a = C_a - pt
                        ray_b = C_b - pt
                        norm_a = np.linalg.norm(ray_a)
                        norm_b = np.linalg.norm(ray_b)
                        if norm_a > 1e-6 and norm_b > 1e-6:
                            cos_alpha = np.clip(np.dot(ray_a, ray_b) / (norm_a * norm_b), -1.0, 1.0)
                            angle_deg = np.degrees(np.arccos(cos_alpha))
                            best_max_err = float(best_fix['metrics'].get('max_error', float('inf')))
                            best_mean_err = float(best_fix['metrics'].get('mean_error', float('inf')))
                            if angle_deg >= 8.0 and best_max_err < 1.2 and best_mean_err < 0.8:
                                strict_drop = False
                                print(f"  - Точка {point_id}: сохранена с 2 видами (parallax {angle_deg:.1f}°, пары: {cam_a}-{cam_b}, max_err {best_max_err:.2f}px)")

                if strict_drop:
                    if secondary_seed_points is not None:
                        secondary_seed_points[point_id] = np.asarray(best_fix['point'], dtype=np.float32).reshape(3)
                    if point_id in points_3d:
                        del points_3d[point_id]
                        removed_points += 1
                        print(
                            f"  - Точка {point_id}: удалена из реконструкции целиком, "
                            f"т.к. после удаления камер {','.join(best_fix['dropped_camera_ids'])} "
                            f"осталось бы < {required_track_length} видов"
                        )
                    continue

        points_3d[point_id] = best_fix['point'].astype(np.float32)
        repaired_points += 1
        for camera_id in best_fix['dropped_camera_ids']:
            if point_id in camera_points.get(camera_id, {}):
                del camera_points[camera_id][point_id]
                removed_observations += 1
                removed_by_camera[camera_id] = removed_by_camera.get(camera_id, 0) + 1

        best_mean = float(best_fix['metrics'].get('mean_error', float('inf')))
        best_max = float(best_fix['metrics'].get('max_error', float('inf')))
        print(
            f"  - Точка {point_id}: "
            f"mean {current_mean:.2f}px -> {best_mean:.2f}px, "
            f"max {current_max:.2f}px -> {best_max:.2f}px, "
            f"удалены камеры {','.join(best_fix['dropped_camera_ids'])}, "
            f"pair {best_fix['pair'][0]}-{best_fix['pair'][1]}"
        )

    if repaired_points <= 0:
        if removed_points <= 0:
            print("Point-track repair: улучшений не найдено")
    else:
        for camera_id, count in sorted(removed_by_camera.items(), key=lambda item: _stable_sort_key(item[0])):
            print(f"  - Камера {camera_id}: удалено наблюдений {count}")

    return {
        'repaired_points': int(repaired_points),
        'removed_observations': int(removed_observations),
        'removed_points': int(removed_points),
    }


def remove_inconsistent_full_tracks(
    points_3d,
    cameras,
    camera_points,
    K,
    dist_coeffs=None,
    target_mean=0.5,
    target_p95=1.0,
    target_max=1.0,
    min_track_length=3,
    min_points_remaining=12,
    min_camera_support=3,
    observation_confidences=None,
    line_support_data=None,
    camera_intrinsics=None,
):
    if not points_3d or not cameras:
        return 0

    min_track_length = max(3, int(min_track_length))
    min_points_remaining = max(8, int(min_points_remaining))
    rejection_candidates = []

    max_limit = max(float(target_max) * 1.60, float(target_p95) * 1.35, 1.25)
    p95_limit = max(float(target_p95) * 1.35, float(target_mean) * 2.25, 0.95)
    mean_limit = max(float(target_mean) * 1.75, float(target_p95) * 0.85, 0.60)
    asymmetry_limit = max(float(target_max) * 0.85, float(target_p95) * 0.75, 0.45)

    for point_id in sorted(points_3d.keys(), key=_stable_sort_key):
        point_observations = {
            str(camera_id): camera_points[str(camera_id)][point_id]
            for camera_id in sorted(cameras.keys(), key=_stable_sort_key)
            if point_id in camera_points.get(str(camera_id), {})
        }
        if len(point_observations) < min_track_length:
            continue

        point_observation_confidences = _extract_point_observation_confidences(
            observation_confidences,
            point_id,
            point_observations.keys(),
        )

        accepted, refined_point, metrics = evaluate_multiview_point(
            points_3d[point_id],
            point_observations,
            cameras,
            K,
            dist_coeffs,
            allow_subset=False,
            min_subset_views=min_track_length,
            observation_confidences=point_observation_confidences,
            point_id=point_id,
            points_3d_context=points_3d,
            line_support_data=line_support_data,
            camera_intrinsics=camera_intrinsics,
        )
        if metrics is None:
            continue

        finite_entries = []
        for camera_id, error in zip(metrics.get('camera_ids', []), metrics.get('errors', [])):
            if np.isfinite(error):
                finite_entries.append((float(error), str(camera_id)))
        if len(finite_entries) < min_track_length:
            continue

        errors = np.asarray([entry[0] for entry in finite_entries], dtype=np.float64)
        mean_error = float(np.mean(errors))
        median_error = float(np.median(errors))
        p95_error = float(np.percentile(errors, 95))
        max_error = float(np.max(errors))
        asymmetry = max_error - median_error

        severe_tail = max_error > max_limit
        p95_bad = p95_error > p95_limit
        mean_bad = mean_error > mean_limit
        asymmetry_bad = asymmetry > asymmetry_limit
        reject = (
            (not bool(accepted)) or
            severe_tail or
            (p95_bad and asymmetry_bad) or
            (mean_bad and asymmetry_bad)
        )
        if not reject:
            continue

        reason_parts = []
        if not bool(accepted):
            reason_parts.append("full_track_rejected")
        if severe_tail:
            reason_parts.append(f"max>{max_limit:.2f}")
        if p95_bad and asymmetry_bad:
            reason_parts.append(f"p95>{p95_limit:.2f}+asym")
        if mean_bad and asymmetry_bad:
            reason_parts.append(f"mean>{mean_limit:.2f}+asym")

        severity = (
            max(0.0, max_error - max_limit) * 0.55 +
            max(0.0, p95_error - p95_limit) * 0.25 +
            max(0.0, mean_error - mean_limit) * 0.10 +
            max(0.0, asymmetry - asymmetry_limit) * 0.10 +
            (2.0 if not bool(accepted) else 0.0)
        )
        rejection_candidates.append({
            'point_id': point_id,
            'severity': float(severity),
            'mean': mean_error,
            'p95': p95_error,
            'max': max_error,
            'asymmetry': asymmetry,
            'reasons': ", ".join(reason_parts),
            'worst_observations': sorted(finite_entries, reverse=True)[:3],
            'camera_ids': sorted(point_observations.keys(), key=_stable_sort_key),
            'refined_point': np.asarray(refined_point, dtype=np.float64).reshape(3),
        })

    if not rejection_candidates:
        print("Strict full-track gate: конфликтных полных треков не найдено")
        return 0

    effective_min_points_remaining = max(
        6,
        min(
            int(min_points_remaining),
            max(len(cameras) + 2, 6),
            max(len(points_3d) - 1, 6),
        ),
    )
    effective_min_camera_support = max(2, int(min_camera_support))
    camera_support = {}
    current_point_ids = set(points_3d.keys())
    for camera_id in sorted(cameras.keys(), key=_stable_sort_key):
        camera_id_str = str(camera_id)
        camera_support[camera_id_str] = sum(
            1
            for point_id in current_point_ids
            if point_id in camera_points.get(camera_id_str, {})
        )

    removed = 0
    skipped_point_limit = 0
    skipped_camera_coverage = 0
    current_points_remaining = len(points_3d)
    print("Strict full-track gate:")
    for candidate in sorted(rejection_candidates, key=lambda item: item['severity'], reverse=True):
        point_id = candidate['point_id']
        if point_id not in points_3d:
            continue

        if current_points_remaining - 1 < effective_min_points_remaining:
            skipped_point_limit += 1
            continue

        violating_cameras = [
            str(camera_id)
            for camera_id in candidate.get('camera_ids', [])
            if camera_support.get(str(camera_id), 0) - 1 < effective_min_camera_support
        ]
        if violating_cameras:
            skipped_camera_coverage += 1
            continue

        del points_3d[point_id]
        removed += 1
        current_points_remaining -= 1
        for camera_id in candidate.get('camera_ids', []):
            camera_id_str = str(camera_id)
            if camera_id_str in camera_support:
                camera_support[camera_id_str] = max(0, int(camera_support[camera_id_str]) - 1)
        worst_summary = ", ".join(
            f"{camera_id}:{error:.2f}px"
            for error, camera_id in candidate['worst_observations']
        )
        print(
            f"  - Точка {point_id}: удалена целиком, "
            f"mean={candidate['mean']:.2f}px, "
            f"p95={candidate['p95']:.2f}px, "
            f"max={candidate['max']:.2f}px, "
            f"asym={candidate['asymmetry']:.2f}px, "
            f"причина={candidate['reasons']}, "
            f"худшие [{worst_summary}]"
        )

    if skipped_point_limit > 0:
        print(
            "  - Пропущено кандидатов из-за ограничения на минимум точек: "
            f"{skipped_point_limit} (effective_min_points_remaining={effective_min_points_remaining})"
        )
    if skipped_camera_coverage > 0:
        print(
            "  - Пропущено кандидатов из-за ограничения на покрытие камер: "
            f"{skipped_camera_coverage} (min_camera_support={effective_min_camera_support})"
        )

    return int(removed)

def _calculate_reprojection_errors(calib_data):
    """
    Локальный расчет ошибок репроекции без зависимости от calibration_core.
    """
    try:
        cameras = calib_data.get('cameras', {})
        points_3d = calib_data.get('points_3d', {})
        camera_points = calib_data.get('camera_points', {})
        K = calib_data.get('K')

        if not cameras or not points_3d or K is None:
            return 0.0, {}, {}

        K = np.asarray(K, dtype=np.float64)
        dist_coeffs = calib_data.get('dist_coeffs')
        if dist_coeffs is None:
            dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        else:
            dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

        errors_by_camera = {}
        errors_by_point = {}
        all_errors = []

        for camera_id, pose in cameras.items():
            if pose is None or len(pose) != 2:
                continue

            observations = camera_points.get(str(camera_id), {})
            if not observations:
                continue

            R, t = pose
            R = np.asarray(R, dtype=np.float64)
            t = np.asarray(t, dtype=np.float64).reshape(3, 1)

            try:
                rvec, _ = cv2.Rodrigues(R)
            except Exception:
                continue

            camera_errors = []

            for point_id, point_2d in observations.items():
                if point_id not in points_3d:
                    continue

                point_3d = np.asarray(points_3d[point_id], dtype=np.float64).reshape(1, 3)
                observed_point = np.asarray(point_2d, dtype=np.float64).reshape(2)

                projected_point, _ = cv2.projectPoints(point_3d, rvec, t, K, dist_coeffs)
                projected_point = projected_point.reshape(2)

                error = float(np.linalg.norm(projected_point - observed_point))
                camera_errors.append(error)
                errors_by_point.setdefault(point_id, []).append(error)
                all_errors.append(error)

            if camera_errors:
                errors_by_camera[camera_id] = camera_errors

        for point_id, point_errors in list(errors_by_point.items()):
            errors_by_point[point_id] = float(np.mean(point_errors))

        total_error = float(np.mean(all_errors)) if all_errors else 0.0
        return total_error, errors_by_point, errors_by_camera

    except Exception as e:
        print(f"Ошибка при вычислении ошибок репроекции в triangulation.py: {str(e)}")
        traceback.print_exc()
        return 0.0, {}, {}

def filter_outliers_by_reprojection_error(calib_data, absolute_threshold=10.0, sigma_multiplier=2.5, mad_multiplier=3.0):
    """
    Улучшенная функция фильтрации выбросов на основе ошибок репроекции.
    Использует комбинацию нескольких методов для более надежного определения выбросов.
    
    Args:
        calib_data: Данные калибровки
        absolute_threshold: Абсолютный порог ошибки репроекции в пикселях
        sigma_multiplier: Множитель для определения выбросов по стандартному отклонению
        mad_multiplier: Множитель для определения выбросов по MAD (Median Absolute Deviation)
    
    Returns:
        tuple: (removed_points, old_error, new_error, outliers_by_camera)
            - количество удаленных точек
            - средняя ошибка репроекции до фильтрации
            - средняя ошибка репроекции после фильтрации
            - словарь распределения выбросов по камерам
    """
    from scipy import stats
    import numpy as np
    
    # Если нет данных или камер, возвращаем 0
    if not calib_data or 'cameras' not in calib_data or not calib_data['cameras']:
        return 0, 0.0, 0.0, {}
    
    print("Начало фильтрации выбросов...")
    
    # Вычисляем текущие ошибки репроекции
    total_error, errors_by_point, errors_by_camera = _calculate_reprojection_errors(calib_data)
    print(f"Исходная средняя ошибка репроекции: {total_error:.4f} пикселей")
    
    # Если нет точек, возвращаем 0
    if not errors_by_point:
        return 0, 0.0, 0.0, {}
    
    # Собираем все ошибки в список
    all_errors = list(errors_by_point.values())
    total_points = len(all_errors)
    print(f"Общее количество точек: {total_points}")
    
    # Вычисляем статистику ошибок
    median_error = np.median(all_errors)
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    mad = stats.median_abs_deviation(all_errors, scale='normal')
    
    # Определяем пороги для фильтрации по разным метрикам
    # 1. Абсолютный порог (должен быть настроен под конкретное приложение)
    # 2. Порог по сигма (стандартное отклонение)
    # 3. Порог по MAD (более устойчив к выбросам, чем стандартное отклонение)
    
    sigma_threshold = mean_error + sigma_multiplier * std_error
    mad_threshold = median_error + mad_multiplier * mad
    
    # Комбинируем пороги - выбираем наиболее строгий из них, но не жестче абсолютного порога
    if absolute_threshold > 0:
        combined_threshold = min(absolute_threshold, mad_threshold)
    else:
        combined_threshold = mad_threshold
    
    # Выводим информацию о статистике и порогах
    print("Статистика ошибок репроекции:")
    print(f"  - Медиана: {median_error:.4f} пикселей")
    print(f"  - Среднее: {mean_error:.4f} пикселей")
    print(f"  - Стандартное отклонение: {std_error:.4f} пикселей")
    print(f"  - MAD: {mad:.4f} пикселей")
    print(f"  - Пороговые значения:")
    print(f"    - Абсолютный порог: {absolute_threshold:.4f} пикселей")
    print(f"    - Sigma порог ({sigma_multiplier}x std): {sigma_threshold:.4f} пикселей")
    print(f"    - MAD порог ({mad_multiplier}): {mad_threshold:.4f} пикселей")
    print(f"  - Выбранный порог (combined): {combined_threshold:.4f} пикселей")
    
    # Находим точки с ошибкой выше порога
    points_to_remove = set()
    for point_id, error in errors_by_point.items():
        if error > combined_threshold:
            points_to_remove.add(point_id)
    
    # Собираем статистику по выбросам в камерах
    outliers_by_camera = {}
    for camera_id, cam_errors in errors_by_camera.items():
        camera_outliers = 0
        for i, point_id in enumerate(calib_data['camera_points'][camera_id].keys()):
            if i < len(cam_errors) and point_id in points_to_remove:
                camera_outliers += 1
        if camera_outliers > 0:
            outliers_by_camera[camera_id] = camera_outliers
    
    # Если нет выбросов, просто возвращаем исходные данные
    if not points_to_remove:
        print("\nРезультаты фильтрации выбросов:")
        print(f"  - Удалено точек: 0 из {total_points} (0.00%)")
        print(f"  - Ошибка репроекции до: {total_error:.4f} пикселей")
        print(f"  - Ошибка репроекции после: {total_error:.4f} пикселей")
        print(f"  - Снижение ошибки: 0.00%")
        print("\nРаспределение выбросов по камерам:")
        return 0, total_error, total_error, {}
    
    # Сохраняем копию исходных 3D точек
    original_points_3d = calib_data['points_3d'].copy()
    
    # Удаляем точки с ошибкой выше порога
    for point_id in points_to_remove:
        if point_id in calib_data['points_3d']:
            del calib_data['points_3d'][point_id]
    
    # Вычисляем новые ошибки репроекции
    new_total_error, _, _ = _calculate_reprojection_errors(calib_data)
    
    # Проверяем, улучшилась ли ошибка репроекции
    error_reduction = (total_error - new_total_error) / total_error * 100 if total_error > 0 else 0
    
    # Если ошибка возросла или изменилась незначительно, восстанавливаем точки
    if new_total_error > total_error or error_reduction < 1.0:
        calib_data['points_3d'] = original_points_3d
        print("\nРезультаты фильтрации выбросов:")
        print(f"  - Отмена удаления точек: ошибка после фильтрации ({new_total_error:.4f}) > ошибки до ({total_error:.4f})")
        print(f"  - Восстановлено {len(points_to_remove)} точек")
        return 0, total_error, total_error, {}
    
    # Выводим результаты фильтрации
    print("\nРезультаты фильтрации выбросов:")
    print(f"  - Удалено точек: {len(points_to_remove)} из {total_points} ({len(points_to_remove) / total_points * 100:.2f}%)")
    print(f"  - Ошибка репроекции до: {total_error:.4f} пикселей")
    print(f"  - Ошибка репроекции после: {new_total_error:.4f} пикселей")
    print(f"  - Снижение ошибки: {error_reduction:.2f}%")
    
    # Выводим распределение выбросов по камерам
    print("\nРаспределение выбросов по камерам:")
    for camera_id, count in outliers_by_camera.items():
        image_path = calib_data['images'].get(camera_id, 'Unknown')
        print(f"  - Камера {camera_id} ({image_path}): {count} выбросов")
    
    # Суммарная информация
    if len(points_to_remove) > 0:
        print(f"Фильтрация выбросов: удалено {len(points_to_remove)} точек, ошибка снижена на {error_reduction:.2f}%")
    
    return len(points_to_remove), total_error, new_total_error, outliers_by_camera

def triangulate_new_points(
    points_3d,
    cameras,
    camera_points,
    K,
    dist_coeffs,
    new_camera_id,
    strict_track_consistency=True,
    debug_logging=False,
    observation_confidences=None,
    multiview_refine_mode="balanced",
    line_support_data=None,
):
    """
    Триангулирует новые точки для добавленной камеры.

    Args:
        points_3d: Словарь 3D точек {id: point}
        cameras: Словарь камер {id: (R, t)}
        camera_points: Словарь точек для каждой камеры {camera_id: {point_id: point_2d}}
        K: Матрица калибровки камеры
        dist_coeffs: Коэффициенты дисторсии
        new_camera_id: ID новой камеры

    Returns:
        dict: Новые 3D точки {point_id: point_3d}
    """
    new_points = {}
    new_camera_id = str(new_camera_id)

    # Получаем позу новой камеры
    if new_camera_id not in cameras:
        print(f"Камера {new_camera_id} не найдена в списке камер")
        return new_points
    
    candidate_point_ids = [
        point_id
        for point_id in sorted(camera_points.get(new_camera_id, {}).keys(), key=_stable_sort_key)
        if point_id not in points_3d
    ]

    for point_id in candidate_point_ids:
        point_observations = {
            str(camera_id): camera_points[str(camera_id)][point_id]
            for camera_id in sorted(cameras.keys(), key=_stable_sort_key)
            if point_id in camera_points.get(str(camera_id), {})
        }
        if len(point_observations) < 2:
            continue

        point_observation_confidences = _extract_point_observation_confidences(
            observation_confidences,
            point_id,
            point_observations.keys(),
        )

        track_camera_ids = sorted(point_observations.keys(), key=_stable_sort_key)
        if new_camera_id not in track_camera_ids:
            continue

        seed_pairs = [
            (other_id, new_camera_id)
            for other_id in track_camera_ids
            if other_id != new_camera_id
        ]
        if not seed_pairs:
            continue

        best_candidate = None
        for other_id, seed_camera_id in seed_pairs:
            candidate_point = _triangulate_global_point_from_pair(
                other_id,
                seed_camera_id,
                point_observations,
                cameras,
                K,
                dist_coeffs=dist_coeffs,
                debug_logging=debug_logging,
            )
            if candidate_point is None:
                continue

            full_min_views = min(3, len(track_camera_ids))
            full_accepted, full_refined_point, full_metrics = evaluate_multiview_point(
                candidate_point,
                point_observations,
                cameras,
                K,
                dist_coeffs,
                allow_subset=False,
                min_subset_views=full_min_views,
                observation_confidences=point_observation_confidences,
                refine_mode=multiview_refine_mode,
                point_id=point_id,
                points_3d_context=points_3d,
                line_support_data=line_support_data,
            )
            if full_metrics is None:
                continue

            error_map = {}
            for camera_id, error in zip(full_metrics.get('camera_ids', []), full_metrics.get('errors', [])):
                error_map[str(camera_id)] = float(error)

            accepted = False
            refined_point = np.asarray(full_refined_point, dtype=np.float64).reshape(3)
            metrics = dict(full_metrics)
            used_camera_ids = [
                str(camera_id)
                for camera_id in full_metrics.get('used_camera_ids', full_metrics.get('camera_ids', []))
            ]
            omitted_camera_ids = [camera_id for camera_id in track_camera_ids if camera_id not in used_camera_ids]
            acceptance_mode = "full_track"

            if full_accepted:
                accepted = True
            else:
                subset_accepted, subset_refined_point, subset_metrics = evaluate_multiview_point(
                    candidate_point,
                    point_observations,
                    cameras,
                    K,
                    dist_coeffs,
                    allow_subset=True,
                    min_subset_views=2,
                    observation_confidences=point_observation_confidences,
                    refine_mode=multiview_refine_mode,
                    point_id=point_id,
                    points_3d_context=points_3d,
                    line_support_data=line_support_data,
                )
                if subset_accepted and subset_metrics is not None:
                    subset_used_camera_ids = [
                        str(camera_id)
                        for camera_id in subset_metrics.get('camera_ids', [])
                    ]
                    subset_omitted_camera_ids = [
                        camera_id
                        for camera_id in track_camera_ids
                        if camera_id not in subset_used_camera_ids
                    ]
                    if new_camera_id in subset_used_camera_ids:
                        kept_errors = np.asarray(
                            [
                                error_map.get(camera_id, float('inf'))
                                for camera_id in subset_used_camera_ids
                                if np.isfinite(error_map.get(camera_id, float('inf')))
                            ],
                            dtype=np.float64,
                        )
                        dropped_errors = np.asarray(
                            [
                                error_map.get(camera_id, float('inf'))
                                for camera_id in subset_omitted_camera_ids
                                if np.isfinite(error_map.get(camera_id, float('inf')))
                            ],
                            dtype=np.float64,
                        )

                        subset_safe = False
                        if len(track_camera_ids) <= 2 and len(subset_used_camera_ids) == 2:
                            subset_safe = True
                        elif len(track_camera_ids) == 3 and len(subset_used_camera_ids) == 2 and dropped_errors.size >= 1:
                            kept_median = float(np.median(kept_errors)) if kept_errors.size else float('inf')
                            dropped_min = float(np.min(dropped_errors))
                            subset_safe = (
                                dropped_min >= max(kept_median * 1.8, kept_median + 0.45) and
                                float(subset_metrics.get('max_error', float('inf'))) <= max(kept_median + 0.90, 2.25)
                            )
                        elif len(track_camera_ids) >= 4 and len(subset_used_camera_ids) >= 3 and dropped_errors.size >= 1:
                            kept_median = float(np.median(kept_errors)) if kept_errors.size else float('inf')
                            dropped_median = float(np.median(dropped_errors))
                            subset_safe = (
                                dropped_median >= max(kept_median * 1.5, kept_median + 0.35) and
                                float(subset_metrics.get('mean_error', float('inf'))) <= max(kept_median + 0.60, 2.50)
                            )

                        if strict_track_consistency and subset_omitted_camera_ids:
                            subset_safe = False

                        if subset_safe:
                            accepted = True
                            refined_point = np.asarray(subset_refined_point, dtype=np.float64).reshape(3)
                            metrics = dict(subset_metrics)
                            used_camera_ids = subset_used_camera_ids
                            omitted_camera_ids = subset_omitted_camera_ids
                            metrics['used_camera_ids'] = list(used_camera_ids)
                            metrics['dropped_camera_ids'] = list(omitted_camera_ids)
                            acceptance_mode = "track_subset"

            if not accepted:
                continue

            used_count = len(used_camera_ids)
            dropped_count = len(omitted_camera_ids)
            mean_error = float(metrics.get('mean_error', float('inf')))
            median_error = float(metrics.get('median_error', float('inf')))
            max_error = float(metrics.get('max_error', float('inf')))
            used_observations = {
                camera_id: point_observations[camera_id]
                for camera_id in used_camera_ids
                if camera_id in point_observations
            }
            used_observation_confidences = {
                camera_id: point_observation_confidences.get(camera_id, 1.0)
                for camera_id in used_camera_ids
            }
            track_summary = _summarize_track_candidate(
                refined_point,
                used_observations,
                cameras,
                K,
                dist_coeffs,
                observation_confidences=used_observation_confidences,
                point_id=point_id,
                points_3d_context=points_3d,
                line_support_data=line_support_data,
            )
            geometry_gate = _evaluate_track_geometry_gate(
                track_summary,
                len(used_camera_ids),
                strict_track_consistency=strict_track_consistency,
                candidate_stage="new_point",
            )
            if not geometry_gate['accepted']:
                _debug_print(
                    debug_logging,
                    f"Новая точка {point_id} отклонена геометрическим gate: "
                    f"{geometry_gate['reason']}, "
                    f"parallax={track_summary.get('min_parallax_deg')}-{track_summary.get('max_parallax_deg')}, "
                    f"baseline={track_summary.get('min_baseline')}-{track_summary.get('max_baseline')}, "
                    f"mean={mean_error:.2f}px, max={max_error:.2f}px"
                )
                continue
            score = (
                used_count * 2.5 -
                dropped_count * 0.75 -
                mean_error * 0.35 -
                max_error * 0.15 -
                median_error * 0.10
            )
            if acceptance_mode == "full_track":
                score += 1.5
            if new_camera_id in used_camera_ids:
                score += 0.5
            score += float(geometry_gate['geometry_score']) * 1.50

            candidate_entry = {
                'point': np.asarray(refined_point, dtype=np.float64).reshape(3),
                'metrics': metrics,
                'score': float(score),
                'used_camera_ids': list(used_camera_ids),
                'dropped_camera_ids': list(omitted_camera_ids),
                'pair': (other_id, seed_camera_id),
                'mode': acceptance_mode,
                'track_summary': track_summary,
                'geometry_gate': geometry_gate,
            }
            if best_candidate is None or (
                candidate_entry['score'] > best_candidate['score'] + 1e-6 or
                (
                    abs(candidate_entry['score'] - best_candidate['score']) <= 1e-6 and
                    len(candidate_entry['used_camera_ids']) > len(best_candidate['used_camera_ids'])
                ) or
                (
                    abs(candidate_entry['score'] - best_candidate['score']) <= 1e-6 and
                    len(candidate_entry['used_camera_ids']) == len(best_candidate['used_camera_ids']) and
                    float(candidate_entry['metrics'].get('max_error', float('inf'))) <
                    float(best_candidate['metrics'].get('max_error', float('inf')))
                )
            ):
                best_candidate = candidate_entry

        if best_candidate is None:
            continue

        new_points[point_id] = np.asarray(best_candidate['point'], dtype=np.float32)
        _debug_print(
            debug_logging,
            f"Новая точка {point_id} принята по {best_candidate['mode']}: "
            f"pair {best_candidate['pair'][0]}-{best_candidate['pair'][1]}, "
            f"used={best_candidate['used_camera_ids']}, "
            f"dropped={best_candidate['dropped_camera_ids']}, "
            f"mean={best_candidate['metrics'].get('mean_error', float('inf')):.2f}px, "
            f"max={best_candidate['metrics'].get('max_error', float('inf')):.2f}px, "
            f"parallax={best_candidate['track_summary'].get('min_parallax_deg')}-"
            f"{best_candidate['track_summary'].get('max_parallax_deg')}, "
            f"baseline={best_candidate['track_summary'].get('min_baseline')}-"
            f"{best_candidate['track_summary'].get('max_baseline')}"
        )

    print(f"Триангулировано {len(new_points)} новых точек")
    return new_points


def triangulate_remaining_tracks(
    points_3d,
    cameras,
    camera_points,
    K,
    dist_coeffs,
    min_track_length=3,
    strict_track_consistency=True,
    debug_logging=False,
    max_points=None,
    observation_confidences=None,
    line_support_data=None,
):
    new_points = {}
    if not cameras:
        return new_points

    min_track_length = max(2, int(min_track_length))
    reconstructed_camera_ids = sorted(cameras.keys(), key=_stable_sort_key)
    candidate_point_ids = sorted(
        {
            point_id
            for camera_id in reconstructed_camera_ids
            for point_id in camera_points.get(str(camera_id), {}).keys()
            if point_id not in points_3d
        },
        key=_stable_sort_key,
    )

    accepted_count = 0
    for point_id in candidate_point_ids:
        point_observations = {
            str(camera_id): camera_points[str(camera_id)][point_id]
            for camera_id in reconstructed_camera_ids
            if point_id in camera_points.get(str(camera_id), {})
        }
        if len(point_observations) < min_track_length:
            continue

        point_observation_confidences = _extract_point_observation_confidences(
            observation_confidences,
            point_id,
            point_observations.keys(),
        )

        track_camera_ids = sorted(point_observations.keys(), key=_stable_sort_key)
        seed_pairs = list(combinations(track_camera_ids, 2))
        if not seed_pairs:
            continue

        best_candidate = None
        for camera_a_id, camera_b_id in seed_pairs:
            pair_context = nullcontext() if debug_logging else redirect_stdout(io.StringIO())
            with pair_context:
                candidate_point = _triangulate_global_point_from_pair(
                    camera_a_id,
                    camera_b_id,
                    point_observations,
                    cameras,
                    K,
                    dist_coeffs=dist_coeffs,
                    debug_logging=debug_logging,
                )
            if candidate_point is None:
                continue

            full_min_views = min(max(2, min_track_length), len(track_camera_ids))
            full_accepted, full_refined_point, full_metrics = evaluate_multiview_point(
                candidate_point,
                point_observations,
                cameras,
                K,
                dist_coeffs,
                allow_subset=False,
                min_subset_views=full_min_views,
                observation_confidences=point_observation_confidences,
                point_id=point_id,
                points_3d_context=points_3d,
                line_support_data=line_support_data,
            )
            if full_metrics is None:
                continue

            accepted = bool(full_accepted)
            refined_point = np.asarray(full_refined_point, dtype=np.float64).reshape(3)
            metrics = dict(full_metrics)
            used_camera_ids = [
                str(camera_id)
                for camera_id in metrics.get('used_camera_ids', metrics.get('camera_ids', []))
            ]
            dropped_camera_ids = [
                camera_id for camera_id in track_camera_ids
                if camera_id not in used_camera_ids
            ]
            mode = "full_track"

            if not accepted and not strict_track_consistency:
                subset_accepted, subset_refined_point, subset_metrics = evaluate_multiview_point(
                    candidate_point,
                    point_observations,
                    cameras,
                    K,
                    dist_coeffs,
                    allow_subset=True,
                    min_subset_views=max(2, min_track_length - 1),
                    observation_confidences=point_observation_confidences,
                    point_id=point_id,
                    points_3d_context=points_3d,
                    line_support_data=line_support_data,
                )
                if subset_accepted and subset_metrics is not None:
                    accepted = True
                    refined_point = np.asarray(subset_refined_point, dtype=np.float64).reshape(3)
                    metrics = dict(subset_metrics)
                    used_camera_ids = [
                        str(camera_id)
                        for camera_id in metrics.get('used_camera_ids', metrics.get('camera_ids', []))
                    ]
                    dropped_camera_ids = [
                        camera_id for camera_id in track_camera_ids
                        if camera_id not in used_camera_ids
                    ]
                    mode = "track_subset"

            if strict_track_consistency and dropped_camera_ids:
                accepted = False

            if not accepted:
                continue

            mean_error = float(metrics.get('mean_error', float('inf')))
            median_error = float(metrics.get('median_error', float('inf')))
            max_error = float(metrics.get('max_error', float('inf')))
            track_length = len(track_camera_ids)
            if track_length <= 2:
                backfill_safe = (
                    mean_error <= 0.95 and
                    max_error <= 1.35
                )
            elif track_length == 3:
                backfill_safe = (
                    mean_error <= 1.15 and
                    max_error <= 1.75
                )
            else:
                backfill_safe = (
                    mean_error <= 1.35 and
                    max_error <= 2.25
                )
            if not backfill_safe:
                continue
            used_observations = {
                camera_id: point_observations[camera_id]
                for camera_id in used_camera_ids
                if camera_id in point_observations
            }
            used_observation_confidences = {
                camera_id: point_observation_confidences.get(camera_id, 1.0)
                for camera_id in used_camera_ids
            }
            track_summary = _summarize_track_candidate(
                refined_point,
                used_observations,
                cameras,
                K,
                dist_coeffs,
                observation_confidences=used_observation_confidences,
                point_id=point_id,
                points_3d_context=points_3d,
                line_support_data=line_support_data,
            )
            geometry_gate = _evaluate_track_geometry_gate(
                track_summary,
                len(used_camera_ids),
                strict_track_consistency=strict_track_consistency,
                candidate_stage="backfill",
            )
            if not geometry_gate['accepted']:
                _debug_print(
                    debug_logging,
                    f"Global backfill точка {point_id} отклонена геометрическим gate: "
                    f"{geometry_gate['reason']}, "
                    f"parallax={track_summary.get('min_parallax_deg')}-{track_summary.get('max_parallax_deg')}, "
                    f"baseline={track_summary.get('min_baseline')}-{track_summary.get('max_baseline')}, "
                    f"mean={mean_error:.2f}px, max={max_error:.2f}px"
                )
                continue
            score = (
                len(used_camera_ids) * 2.2 -
                mean_error * 0.35 -
                max_error * 0.15 -
                median_error * 0.10
            )
            if mode == "full_track":
                score += 1.25
            score += float(geometry_gate['geometry_score']) * 1.25

            candidate_entry = {
                'point': np.asarray(refined_point, dtype=np.float64).reshape(3),
                'metrics': metrics,
                'score': float(score),
                'used_camera_ids': list(used_camera_ids),
                'dropped_camera_ids': list(dropped_camera_ids),
                'pair': (camera_a_id, camera_b_id),
                'mode': mode,
                'track_summary': track_summary,
                'geometry_gate': geometry_gate,
            }
            if best_candidate is None or (
                candidate_entry['score'] > best_candidate['score'] + 1e-6 or
                (
                    abs(candidate_entry['score'] - best_candidate['score']) <= 1e-6 and
                    len(candidate_entry['used_camera_ids']) > len(best_candidate['used_camera_ids'])
                ) or
                (
                    abs(candidate_entry['score'] - best_candidate['score']) <= 1e-6 and
                    len(candidate_entry['used_camera_ids']) == len(best_candidate['used_camera_ids']) and
                    float(candidate_entry['metrics'].get('max_error', float('inf'))) <
                    float(best_candidate['metrics'].get('max_error', float('inf')))
                )
            ):
                best_candidate = candidate_entry

        if best_candidate is None:
            continue

        new_points[point_id] = np.asarray(best_candidate['point'], dtype=np.float32)
        accepted_count += 1
        _debug_print(
            debug_logging,
            f"Global backfill точка {point_id} принята по {best_candidate['mode']}: "
            f"pair {best_candidate['pair'][0]}-{best_candidate['pair'][1]}, "
            f"used={best_candidate['used_camera_ids']}, "
            f"mean={best_candidate['metrics'].get('mean_error', float('inf')):.2f}px, "
            f"max={best_candidate['metrics'].get('max_error', float('inf')):.2f}px, "
            f"parallax={best_candidate['track_summary'].get('min_parallax_deg')}-"
            f"{best_candidate['track_summary'].get('max_parallax_deg')}, "
            f"baseline={best_candidate['track_summary'].get('min_baseline')}-"
            f"{best_candidate['track_summary'].get('max_baseline')}"
        )

        if max_points is not None and accepted_count >= int(max_points):
            break

    print(f"Global track backfill: добавлено {len(new_points)} новых точек")
    return new_points


def diagnose_unreconstructed_tracks(
    points_3d,
    cameras,
    camera_points,
    K,
    dist_coeffs=None,
    min_track_length=2,
    strict_track_consistency=True,
    top_k=5,
    debug_logging=False,
    observation_confidences=None,
    line_support_data=None,
):
    if not cameras:
        return []

    min_track_length = max(2, int(min_track_length))
    reconstructed_camera_ids = sorted(cameras.keys(), key=_stable_sort_key)
    candidate_point_ids = sorted(
        {
            point_id
            for camera_id in reconstructed_camera_ids
            for point_id in camera_points.get(str(camera_id), {}).keys()
            if point_id not in points_3d
        },
        key=_stable_sort_key,
    )

    diagnostics = []
    for point_id in candidate_point_ids:
        point_observations = {
            str(camera_id): camera_points[str(camera_id)][point_id]
            for camera_id in reconstructed_camera_ids
            if point_id in camera_points.get(str(camera_id), {})
        }
        track_camera_ids = sorted(point_observations.keys(), key=_stable_sort_key)
        track_length = len(track_camera_ids)
        point_observation_confidences = _extract_point_observation_confidences(
            observation_confidences,
            point_id,
            point_observations.keys(),
        )
        if track_length < min_track_length:
            diagnostics.append({
                'point_id': point_id,
                'track_length': track_length,
                'reason': 'too_few_views',
                'pair_success_count': 0,
                'full_accept_count': 0,
                'subset_accept_count': 0,
                'strict_blocked_count': 0,
                'best_pair': None,
                'best_mode': None,
                'best_mean': None,
                'best_max': None,
                'baseline': None,
                'best_point': None,
            })
            continue

        pair_success_count = 0
        full_accept_count = 0
        subset_accept_count = 0
        strict_blocked_count = 0
        best_candidate = None

        for camera_a_id, camera_b_id in combinations(track_camera_ids, 2):
            pair_context = nullcontext() if debug_logging else redirect_stdout(io.StringIO())
            with pair_context:
                candidate_point = _triangulate_global_point_from_pair(
                    camera_a_id,
                    camera_b_id,
                    point_observations,
                    cameras,
                    K,
                    dist_coeffs=dist_coeffs,
                    debug_logging=debug_logging,
                )
            if candidate_point is None:
                continue

            pair_success_count += 1
            full_min_views = min(3, track_length)
            full_accepted, full_refined_point, full_metrics = evaluate_multiview_point(
                candidate_point,
                point_observations,
                cameras,
                K,
                dist_coeffs,
                allow_subset=False,
                min_subset_views=full_min_views,
                observation_confidences=point_observation_confidences,
                point_id=point_id,
                points_3d_context=points_3d,
                line_support_data=line_support_data,
            )

            if full_metrics is not None:
                full_mean = float(full_metrics.get('mean_error', float('inf')))
                full_max = float(full_metrics.get('max_error', float('inf')))
                full_entry = {
                    'pair': (camera_a_id, camera_b_id),
                    'mode': 'full_track',
                    'mean': full_mean,
                    'max': full_max,
                    'used_count': len(full_metrics.get('used_camera_ids', full_metrics.get('camera_ids', []))),
                    'point': np.asarray(candidate_point if full_refined_point is None else full_refined_point, dtype=np.float64).reshape(3),
                }
                if best_candidate is None or (
                    full_entry['used_count'] > best_candidate['used_count'] or
                    (
                        full_entry['used_count'] == best_candidate['used_count'] and
                        (
                            full_entry['mean'] < best_candidate['mean'] - 1e-6 or
                            (
                                abs(full_entry['mean'] - best_candidate['mean']) <= 1e-6 and
                                full_entry['max'] < best_candidate['max']
                            )
                        )
                    )
                ):
                    best_candidate = full_entry

            if full_accepted:
                full_accept_count += 1
                continue

            subset_accepted, subset_refined_point, subset_metrics = evaluate_multiview_point(
                candidate_point,
                point_observations,
                cameras,
                K,
                dist_coeffs,
                allow_subset=True,
                min_subset_views=2,
                observation_confidences=point_observation_confidences,
                point_id=point_id,
                points_3d_context=points_3d,
                line_support_data=line_support_data,
            )
            if not subset_accepted or subset_metrics is None:
                continue

            subset_used_camera_ids = [
                str(camera_id)
                for camera_id in subset_metrics.get('used_camera_ids', subset_metrics.get('camera_ids', []))
            ]
            dropped_camera_ids = [
                camera_id for camera_id in track_camera_ids
                if camera_id not in subset_used_camera_ids
            ]
            if dropped_camera_ids:
                subset_accept_count += 1
                if strict_track_consistency:
                    strict_blocked_count += 1

            subset_mean = float(subset_metrics.get('mean_error', float('inf')))
            subset_max = float(subset_metrics.get('max_error', float('inf')))
            subset_entry = {
                'pair': (camera_a_id, camera_b_id),
                'mode': 'subset' if dropped_camera_ids else 'full_track',
                'mean': subset_mean,
                'max': subset_max,
                'used_count': len(subset_used_camera_ids),
                'point': np.asarray(candidate_point if subset_refined_point is None else subset_refined_point, dtype=np.float64).reshape(3),
            }
            if best_candidate is None or (
                subset_entry['used_count'] > best_candidate['used_count'] or
                (
                    subset_entry['used_count'] == best_candidate['used_count'] and
                    (
                        subset_entry['mean'] < best_candidate['mean'] - 1e-6 or
                        (
                            abs(subset_entry['mean'] - best_candidate['mean']) <= 1e-6 and
                            subset_entry['max'] < best_candidate['max']
                        )
                    )
                )
            ):
                best_candidate = subset_entry

        if full_accept_count > 0:
            reason = 'backfillable_full_track'
        elif strict_blocked_count > 0:
            reason = 'strict_subset_only'
        elif subset_accept_count > 0:
            reason = 'subset_only'
        elif pair_success_count > 0:
            reason = 'full_track_rejected'
        else:
            reason = 'pair_triangulation_failed'

        baseline = None
        if best_candidate is not None and best_candidate['pair'] is not None:
            camera_a_id, camera_b_id = best_candidate['pair']
            R_a, t_a = cameras[camera_a_id]
            R_b, t_b = cameras[camera_b_id]
            center_a = _camera_center_from_pose(R_a, t_a)
            center_b = _camera_center_from_pose(R_b, t_b)
            baseline = float(np.linalg.norm(center_b - center_a))

        track_summary = (
            _summarize_track_candidate(
                best_candidate['point'],
                point_observations,
                cameras,
                K,
                dist_coeffs,
                observation_confidences=point_observation_confidences,
                point_id=point_id,
                points_3d_context=points_3d,
                line_support_data=line_support_data,
            )
            if best_candidate is not None else {
                'conflict_class': 'insufficient_support',
                'worst_observations': [],
                'median_error': None,
                'max_error': None,
                'asymmetry': None,
                'min_parallax_deg': None,
                'max_parallax_deg': None,
                'min_baseline': None,
                'max_baseline': None,
                'high_error_count': 0,
            }
        )

        diagnostics.append({
            'point_id': point_id,
            'track_length': track_length,
            'reason': reason,
            'pair_success_count': int(pair_success_count),
            'full_accept_count': int(full_accept_count),
            'subset_accept_count': int(subset_accept_count),
            'strict_blocked_count': int(strict_blocked_count),
            'best_pair': None if best_candidate is None else best_candidate['pair'],
            'best_mode': None if best_candidate is None else best_candidate['mode'],
            'best_mean': None if best_candidate is None else float(best_candidate['mean']),
            'best_max': None if best_candidate is None else float(best_candidate['max']),
            'baseline': baseline,
            'best_point': None if best_candidate is None else np.asarray(best_candidate['point'], dtype=np.float64).reshape(3),
            'conflict_class': track_summary.get('conflict_class'),
            'worst_observations': list(track_summary.get('worst_observations', [])),
            'error_asymmetry': track_summary.get('asymmetry'),
            'min_parallax_deg': track_summary.get('min_parallax_deg'),
            'max_parallax_deg': track_summary.get('max_parallax_deg'),
            'min_baseline': track_summary.get('min_baseline'),
            'max_baseline': track_summary.get('max_baseline'),
            'high_error_count': int(track_summary.get('high_error_count', 0)),
        })

    diagnostics.sort(
        key=lambda item: (
            item['track_length'],
            item['strict_blocked_count'],
            item['pair_success_count'],
            0.0 if item['best_max'] is None else item['best_max'],
        ),
        reverse=True,
    )
    if top_k is None or int(top_k) <= 0:
        return diagnostics
    return diagnostics[:max(1, int(top_k))]
    
def triangulate_points_with_uncertainty(pts1, pts2, K, R, t, points_confidence=None, dist_coeffs=None):
    """
    Триангулирует точки с учетом неопределенности и надежности соответствий.
    
    Args:
        pts1, pts2: 2D точки на изображениях
        K: Матрица калибровки
        R, t: Относительная поза второй камеры
        points_confidence: Массив весов надежности для точек (опционально)
        
    Returns:
        tuple: (points_3d, mask, uncertainties)
    """
    # Если надежность не задана, используем равномерные веса
    if points_confidence is None:
        points_confidence = np.ones(len(pts1))
    
    # Нормализуем точки
    pts1_norm = normalize_points(pts1, K, dist_coeffs=dist_coeffs)
    pts2_norm = normalize_points(pts2, K, dist_coeffs=dist_coeffs)
    
    # Создаем матрицы проекции
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))
    
    # Триангулируем точки
    points_4d = cv2.triangulatePoints(P1, P2, pts1_norm.T, pts2_norm.T)
    points_3d = points_4d[:3, :] / points_4d[3, :]
    
    # Вычисляем неопределенность для каждой точки
    uncertainties = np.zeros(points_3d.shape[1])
    baseline = np.linalg.norm(t)
    
    for i in range(points_3d.shape[1]):
        point = points_3d[:, i]
        
        # Вектора от центров камер к точке
        vec1 = point
        vec2 = point - (-R.T @ t.ravel())  # центр второй камеры в системе координат первой
        
        # Угол между векторами (параллакс)
        cos_parallax = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
        parallax = np.arccos(np.clip(cos_parallax, -1.0, 1.0))
        
        # Неопределенность обратно пропорциональна параллаксу и базовой линии
        if parallax > 1e-3:  # избегаем деления на 0
            uncertainties[i] = baseline / (np.sin(parallax) + 1e-10)
        else:
            uncertainties[i] = float('inf')  # очень высокая неопределенность
    
    # Формируем маску валидных точек на основе неопределенности и надежности
    # Учитываем как геометрическую стабильность, так и надежность соответствий
    geometric_stability = uncertainties < np.percentile(uncertainties, 95)  # Отбрасываем 5% самых неопределенных точек
    correspondence_reliability = points_confidence > 0.3  # Минимальная надежность соответствия
    
    mask = geometric_stability & correspondence_reliability
    
    return points_3d.T, mask, uncertainties

def weighted_triangulation(pts1, pts2, K, R, t, weights=None, dist_coeffs=None):
    """
    Выполняет взвешенную триангуляцию точек с учетом надежности соответствий.
    
    Args:
        pts1, pts2: 2D точки на изображениях
        K: Матрица калибровки
        R, t: Относительная поза второй камеры
        weights: Веса для каждой пары точек (опционально)
        
    Returns:
        tuple: (points_3d, mask)
    """
    if weights is None:
        # Если веса не заданы, используем стандартную триангуляцию
        return triangulate_points(pts1, pts2, K, R, t, robust=True, dist_coeffs=dist_coeffs)
    
    # Нормализуем точки
    pts1_norm = normalize_points(pts1, K, dist_coeffs=dist_coeffs)
    pts2_norm = normalize_points(pts2, K, dist_coeffs=dist_coeffs)
    
    # Создаем матрицы проекции
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))
    
    # Преобразуем веса в корректный формат
    weights = np.asarray(weights)
    
    # Используем взвешенную триангуляцию
    # Создаем расширенные матрицы с учетом весов
    n_points = len(pts1_norm)
    points_3d = np.zeros((3, n_points))
    
    # Для каждой точки выполняем взвешенную триангуляцию
    for i in range(n_points):
        # Используем алгебраический метод взвешенной триангуляции
        # Составляем систему уравнений с учетом весов
        
        # Вектора для составления системы уравнений
        A_row1 = weights[i] * (pts1_norm[i, 0] * P1[2, :] - P1[0, :])  # Взвешенная разница для первой камеры
        A_row2 = weights[i] * (pts1_norm[i, 1] * P1[2, :] - P1[1, :])
        A_row3 = weights[i] * (pts2_norm[i, 0] * P2[2, :] - P2[0, :])  # Взвешенная разница для второй камеры
        A_row4 = weights[i] * (pts2_norm[i, 1] * P2[2, :] - P2[1, :])
        
        # Составляем матрицу A для SVD
        A = np.vstack((A_row1, A_row2, A_row3, A_row4))
        
        # Решаем систему уравнений с помощью SVD
        try:
            _, _, Vt = np.linalg.svd(A)
            point_3d_hom = Vt[-1, :]  # Последний вектор строки - решение
            if point_3d_hom[3] != 0:
                point_3d = point_3d_hom[:3] / point_3d_hom[3]
            else:
                # Если точка в бесконечности, используем альтернативный метод
                point_3d = np.zeros(3)
        except np.linalg.LinAlgError:
            # Если SVD не сходится, используем стандартную триангуляцию для этой точки
            single_pts1 = pts1_norm[i:i+1].T
            single_pts2 = pts2_norm[i:i+1].T
            point_4d = cv2.triangulatePoints(P1, P2, single_pts1, single_pts2)
            point_3d = (point_4d[:3, 0] / point_4d[3, 0]).flatten()
        
        points_3d[:, i] = point_3d
    
    # Проверяем, что точки находятся перед обеими камерами
    mask = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        point = points_3d[:, i]
        
        # Проверяем, что точка перед первой камерой
        if point[2] <= 0.01:
            mask[i] = False
            continue
            
        # Преобразуем точку в систему координат второй камеры
        point2 = R @ point + t.ravel()
        
        # Проверяем, что точка перед второй камерой
        if point2[2] <= 0.01:
            mask[i] = False
            continue
    
    return points_3d.T, mask

def refine_3d_point(point_3d, point_2d_observations, cameras, K, dist_coeffs=None):
    """
    Уточняет положение 3D точки с помощью оптимизации ошибки репроекции.
    
    Args:
        point_3d: начальное приближение 3D точки
        point_2d_observations: словарь {camera_id: point_2d}
        cameras: словарь {camera_id: (R, t)}
        K: матрица калибровки камеры
        dist_coeffs: коэффициенты дисторсии
        
    Returns:
        np.ndarray: уточненная 3D точка
    """
    from scipy.optimize import minimize
    
    # Функция ошибки репроекции для оптимизации
    def reprojection_error(x):
        p = x.reshape(3)
        total_error = 0
        penalty = 0  # Штраф за точки, находящиеся позади камер
        
        for cam_id, point_2d in point_2d_observations.items():
            if cam_id not in cameras:
                continue
                
            R, t = cameras[cam_id]
            
            # Преобразуем 3D точку в координаты камеры
            p_cam = R.T @ (p - t.reshape(3))
            
            # Если точка позади камеры, добавляем штраф
            if p_cam[2] <= 0:
                penalty += 1000  # Большой штраф для точек позади камеры
                continue  # Пропускаем вычисление ошибки репроекции для этой камеры
                
            # Проецируем точку на изображение
            p_2d = K @ p_cam
            p_2d = p_2d[:2] / p_2d[2]
            
            # Вычисляем ошибку репроекции
            error = np.linalg.norm(p_2d - point_2d)
            total_error += error * error
            
        return total_error + penalty
        
    # Запускаем оптимизацию
    try:
        result = minimize(reprojection_error, point_3d, method='L-BFGS-B')
        if result.success:
            # Дополнительно проверяем, что уточненная точка находится перед всеми камерами
            refined_point = result.x
            all_in_front = True
            
            for cam_id, (_, t) in cameras.items():
                if cam_id not in point_2d_observations:
                    continue
                    
                # Преобразуем точку в систему координат камеры
                R, t = cameras[cam_id]
                p_cam = R.T @ (refined_point - t.reshape(3))
                
                # Проверяем, что точка перед камерой
                if p_cam[2] <= 0:
                    all_in_front = False
                    break
            
            # Если точка находится перед всеми камерами, возвращаем уточненную
            if all_in_front:
                return refined_point
            else:
                # Если уточненная точка позади какой-либо камеры, возвращаем исходную
                return point_3d
    except Exception as e:
        print(f"Ошибка при уточнении 3D точки: {str(e)}")
        pass
        
    return point_3d

def refine_all_points_3d(points_3d, camera_points, cameras, K, dist_coeffs=None, max_iterations=3):
    """
    Уточняет все 3D точки с использованием всех доступных наблюдений.

    Args:
        points_3d: Словарь 3D точек {point_id: point_3d}
        camera_points: Словарь точек для каждой камеры {camera_id: {point_id: point_2d}}
        cameras: Словарь поз камер {camera_id: (R, t)}
        K: Матрица калибровки
        dist_coeffs: Коэффициенты дисторсии
        max_iterations: Максимальное количество итераций уточнения
        
    Returns:
        dict: Уточненные 3D точки {point_id: point_3d}
    """
    refined_points = points_3d.copy()
    
    for iteration in range(max_iterations):
        print(f"Итерация уточнения точек: {iteration + 1}/{max_iterations}")
        
        updated_count = 0
        total_error_before = 0
        total_error_after = 0
        
        for point_id, point_3d in points_3d.items():
            # Собираем все наблюдения для этой точки
            observations = {}
            for camera_id, points in camera_points.items():
                if point_id in points:
                    observations[camera_id] = points[point_id]
            
            # Если точка видна менее чем в 2 камерах, пропускаем
            if len(observations) < 2:
                continue
            
            # Уточняем точку
            refined_point = refine_3d_point(point_3d, observations, cameras, K, dist_coeffs)
            
            # Проверяем, изменилась ли точка значительно
            change = np.linalg.norm(refined_point - point_3d)
            if change > 1e-6:  # Если изменение больше порога
                refined_points[point_id] = refined_point
                updated_count += 1
        
        print(f"  - Обновлено точек: {updated_count}")
        
        # Если на последующих итерациях не обновлено точек, выходим
        if updated_count == 0 and iteration > 0:
            print(f"  - Нет изменений, прекращение итераций")
            break
    
    print(f"Уточнение точек завершено: обновлено {len([k for k in refined_points if not np.allclose(refined_points[k], points_3d[k])])} точек")
    return refined_points

def compute_focal_length_from_exif(image_path, image_width, image_height):
    """
    Вычисляет фокусное расстояние камеры из EXIF данных изображения.

    Args:
        image_path: Путь к изображению
        image_width: Ширина изображения в пикселях
        image_height: Высота изображения в пикселях

    Returns:
        float or None: Фокусное расстояние в пикселях или None, если не удалось извлечь
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        img = Image.open(image_path)

        exif_data = img._getexif()
        if exif_data is None:
            return None

        # Построим словарь tag_name -> value для удобства
        exif_by_name = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            exif_by_name[tag_name] = value

        # --- FocalLengthIn35mmFilm: самый надёжный источник ---
        focal_35mm = exif_by_name.get("FocalLengthIn35mmFilm")
        if focal_35mm is not None:
            focal_35mm_val = float(focal_35mm)
            if focal_35mm_val > 0:
                # 35mm эквивалент -> пиксели: f_px = f_35mm / 36mm * image_width
                # (горизонтальный FoV совпадает с 36mm шириной кадра)
                focal_px = focal_35mm_val * image_width / 36.0
                print(f"EXIF FocalLengthIn35mmFilm: {focal_35mm_val}мм -> {focal_px:.1f}px")
                return float(focal_px)

        # --- FocalLength + sensor size ---
        focal_raw = exif_by_name.get("FocalLength")
        if focal_raw is None:
            return None

        # Pillow может вернуть IFDRational, tuple или float
        if isinstance(focal_raw, tuple) and len(focal_raw) == 2:
            focal_length_mm = float(focal_raw[0]) / float(focal_raw[1])
        else:
            focal_length_mm = float(focal_raw)

        if focal_length_mm <= 0:
            return None
        print(f"EXIF фокусное расстояние: {focal_length_mm}мм")

        # Попытка вычислить реальный размер сенсора из FocalPlane*Resolution
        sensor_width_mm = None
        fp_x_res = exif_by_name.get("FocalPlaneXResolution")
        fp_unit = exif_by_name.get("FocalPlaneResolutionUnit", 2)  # 2=inch, 3=cm

        if fp_x_res is not None:
            if isinstance(fp_x_res, tuple) and len(fp_x_res) == 2:
                x_res_value = float(fp_x_res[0]) / float(fp_x_res[1])
            else:
                x_res_value = float(fp_x_res)

            if x_res_value > 0:
                # Конвертируем в pixels/mm
                unit_val = int(fp_unit) if fp_unit is not None else 2
                if unit_val == 2:  # дюймы
                    x_res_mm = x_res_value / 25.4
                elif unit_val == 3:  # сантиметры
                    x_res_mm = x_res_value / 10.0
                else:
                    x_res_mm = x_res_value / 25.4  # fallback to inches

                sensor_width_mm = image_width / x_res_mm
                print(f"EXIF: вычислен размер сенсора: {sensor_width_mm:.2f}мм")

        if sensor_width_mm is not None and sensor_width_mm > 0:
            focal_px = focal_length_mm * image_width / sensor_width_mm
        else:
            # Fallback: используем 36mm (полный кадр)
            focal_px = focal_length_mm * image_width / 36.0

        print(f"Преобразованное фокусное расстояние: {focal_px:.1f} пикселей")
        return float(focal_px)

    except ImportError:
        print("Библиотека PIL не доступна, пропускаем чтение EXIF")
    except Exception as e:
        print(f"Ошибка при чтении EXIF: {str(e)}")

    return None

def estimate_focal_length_from_vanishing_points_advanced(image_path, K_initial, points_2d=None, points_3d=None):
    """
    Улучшенная оценка фокусного расстояния на основе точек схода с использованием
    геометрического анализа изображения.
    
    Args:
        image_path: Путь к изображению
        K_initial: Начальная матрица калибровки
        points_2d: 2D точки на изображении (опционально)
        points_3d: Соответствующие 3D точки (опционально)
        
    Returns:
        tuple: (fx, fy) улучшенные фокусные расстояния или (None, None), если не удалось оценить
    """
    try:
        import cv2
        import numpy as np
        from scipy.optimize import minimize
        
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            return None, None
            
        image_height, image_width = image.shape[:2]
        
        # Обнаруживаем линии на изображении
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Используем более чувствительные параметры для обнаружения линий
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                               minLineLength=int(image_width*0.05), maxLineGap=int(image_width*0.02))
        
        if lines is None or len(lines) < 10:
            print(f"Недостаточно линий для оценки по точкам схода: {len(lines) if lines is not None else 0}")
            return None, None
            
        # Группируем линии по направлениям для нахождения точек схода
        # Преобразуем углы в декартовы координаты для корректной кластеризации
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2-y1, x2-x1)
            angles.append(angle)
            
        angles = np.array(angles)
        angles_cartesian = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # Используем DBSCAN для кластеризации направлений линий
        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.3, min_samples=5).fit(angles_cartesian)
            cluster_labels = np.asarray(clustering.labels_, dtype=np.int32)
            unique_labels = set(cluster_labels.tolist())
            unique_labels.discard(-1)  # Удаляем шум
        except ImportError:
            # Альтернативная кластеризация без sklearn
            unique_labels = set()
            clusters = {}
            cluster_counts = {}
            cluster_labels = np.full(len(angles), -1, dtype=np.int32)
            angle_tolerance = 0.2  # Радианы (примерно 11.5 градусов)
            
            for i, angle in enumerate(angles):
                assigned = False
                for label, center_angle in clusters.items():
                    if min(abs(angle - center_angle), 2*np.pi - abs(angle - center_angle)) < angle_tolerance:
                        count = cluster_counts[label]
                        clusters[label] = (center_angle * count + angle) / (count + 1)
                        cluster_counts[label] = count + 1
                        cluster_labels[i] = label
                        assigned = True
                        break
                
                if not assigned:
                    new_label = len(clusters)
                    unique_labels.add(new_label)
                    clusters[new_label] = float(angle)
                    cluster_counts[new_label] = 1
                    cluster_labels[i] = new_label
        
        if len(unique_labels) < 2:
            print(f"Недостаточно направлений для оценки фокусного расстояния: {len(unique_labels)}")
            return None, None
            
        # Находим точки схода для каждого направления
        vanishing_points = []
        for label in unique_labels:
            line_indices = np.where(cluster_labels == label)[0]
            if len(line_indices) < 3:
                continue
                
            # Для нахождения точки схода используем метод наименьших квадратов
            # для пересечения линий в кластере
            A = []
            b = []
            for idx in line_indices:
                x1, y1, x2, y2 = lines[idx][0]
                # Уравнение линии: ax + by + c = 0
                a = y2 - y1
                b_coeff = x1 - x2
                c = x2*y1 - x1*y2
                # Нормализуем коэффициенты
                norm = np.sqrt(a**2 + b_coeff**2)
                if norm > 0:
                    a, b_coeff, c = a/norm, b_coeff/norm, c/norm
                    A.append([a, b_coeff])
                    b.append(-c)
            
            if len(A) >= 2:
                A = np.array(A)
                b = np.array(b)
                
                # Решаем систему уравнений Ax = b методом наименьших квадратов
                try:
                    vp = np.linalg.lstsq(A, b, rcond=None)[0]
                    if np.all(np.isfinite(vp)):
                        vanishing_points.append(vp)
                except np.linalg.LinAlgError:
                    continue
        
        # Используем найденные точки схода для оценки фокусного расстояния
        # Если есть ортогональные направления, их точки схода должны удовлетворять условию:
        # (vp1 - cc) · (vp2 - cc) + f² = 0, где cc - главная точка, f - фокусное расстояние
        if len(vanishing_points) >= 2:
            focal_estimates = []
            cx = K_initial[0, 2]  # Главная точка
            cy = K_initial[1, 2]
            
            for i in range(len(vanishing_points)):
                for j in range(i+1, len(vanishing_points)):
                    vp1 = vanishing_points[i]
                    vp2 = vanishing_points[j]
                    
                    # Вычисляем скалярное произведение в нормализованных координатах
                    # Преобразуем точки схода в нормализованные координаты (относительно главной точки и фокусного расстояния)
                    x1_norm = (vp1[0] - cx) / K_initial[0, 0]  # Используем текущее приближение
                    y1_norm = (vp1[1] - cy) / K_initial[1, 1]
                    x2_norm = (vp2[0] - cx) / K_initial[0, 0]
                    y2_norm = (vp2[1] - cy) / K_initial[1, 1]
                    
                    # Вектора из главной точки к точкам схода в нормализованных координатах
                    v1 = np.array([x1_norm, y1_norm, 1])
                    v2 = np.array([x2_norm, y2_norm, 1])
                    
                    # Для ортогональных направлений в 3D пространстве: v1 · v2 = 0
                    # В нормализованных координатах: (vp1 - cc) · (vp2 - cc) + f² = 0
                    # где cc - главная точка
                    dot_product = (vp1[0] - cx) * (vp2[0] - cx) + (vp1[1] - cy) * (vp2[1] - cy)
                    
                    # Если точки схода близки к ортогональности, можно оценить фокусное расстояние
                    if dot_product < 0:  # Это условие указывает на потенциальную ортогональность
                        estimated_focal = np.sqrt(-dot_product)
                        if 100 < estimated_focal < 10000:  # Разумный диапазон фокусных расстояний
                            focal_estimates.append(estimated_focal)
            
            if focal_estimates:
                # Используем медиану для устойчивости к выбросам
                median_focal = np.median(focal_estimates)
                
                # Проверяем, насколько согласованы оценки
                std_focal = np.std(focal_estimates)
                if len(focal_estimates) > 1 and std_focal / median_focal < 0.3:  # Коэффициент вариации < 30%
                    print(f"Оценка фокусного расстояния из точек схода: {median_focal:.2f} пикселей")
                    return median_focal, median_focal  # Предполагаем квадратные пиксели
    
    except Exception as e:
        print(f"Ошибка при оценке фокусного расстояния из точек схода: {str(e)}")
        import traceback
        traceback.print_exc()
        
    return None, None

def estimate_focal_length_from_geometric_constraints(image_path, K_initial, points_2d=None, points_3d=None):
    """
    Оценивает фокусное расстояние на основе геометрических ограничений и перспективных искажений.
    
    Args:
        image_path: Путь к изображению
        K_initial: Начальная матрица калибровки
        points_2d: 2D точки на изображении (опционально)
        points_3d: Соответствующие 3D точки (опционально)
        
    Returns:
        tuple: (fx, fy) оценка фокусного расстояния или (None, None), если не удалось оценить
    """
    try:
        import cv2
        import numpy as np
        
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            return None, None
            
        image_height, image_width = image.shape[:2]
        
        # Анализируем перспективные искажения
        # Для этого используем метод оценки по размеру объектов в разных частях изображения
        
        # Вычисляем градиенты изображения для обнаружения краев
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Определяем области с высокими градиентами (края объектов)
        high_gradient_threshold = np.percentile(magnitude, 80)  # Берем 80% перцентиль
        high_gradient_mask = magnitude > high_gradient_threshold
        
        # Разбиваем изображение на сетку для анализа перспективных искажений
        grid_size = 4
        cell_height = image_height // grid_size
        cell_width = image_width // grid_size
        
        cell_sizes = []
        for i in range(grid_size):
            for j in range(grid_size):
                y_start = i * cell_height
                y_end = min((i + 1) * cell_height, image_height)
                x_start = j * cell_width
                x_end = min((j + 1) * cell_width, image_width)
                
                # Вычисляем среднюю величину градиента в ячейке
                cell_gradient = magnitude[y_start:y_end, x_start:x_end]
                cell_size = np.mean(cell_gradient[high_gradient_mask[y_start:y_end, x_start:x_end]])
                
                if not np.isnan(cell_size) and cell_size > 0:
                    cell_sizes.append(((x_start + x_end) / 2, (y_start + y_end) / 2, cell_size))
        
        if len(cell_sizes) < 4:
            print("Недостаточно информации для анализа перспективных искажений")
            return None, None
        
        # Сортируем ячейки по величине градиента (от больших к меньшим)
        cell_sizes.sort(key=lambda x: x[2], reverse=True)
        
        # Берем крайние ячейки (самые большие и самые маленькие)
        if len(cell_sizes) >= 2:
            far_cell = cell_sizes[-1]  # Самая маленькая (далеко)
            near_cell = cell_sizes[0]   # Самая большая (близко)
            
            # Расстояние от центра изображения до ячеек
            center_x, center_y = image_width / 2, image_height / 2
            dist_far = np.sqrt((far_cell[0] - center_x)**2 + (far_cell[1] - center_y)**2)
            dist_near = np.sqrt((near_cell[0] - center_x)**2 + (near_cell[1] - center_y)**2)
            
            # Размеры в этих ячейках (обратно пропорциональны расстоянию)
            size_far = far_cell[2]
            size_near = near_cell[2]
            
            # Если мы знаем, что объекты в сцене одного размера, то отношение размеров
            # обратно пропорционально отношению расстояний
            if size_far > 0 and dist_far > 0:
                # Отношение расстояний к объектам
                distance_ratio = size_near / size_far  # Чем больше размер, тем ближе объект
                
                # Используем это отношение для оценки фокусного расстояния
                # При известной геометрии сцены и фокусном расстоянии:
                # f = (distance_to_object * real_size) / projected_size
                # Мы можем оценить f на основе перспективных искажений
                
                # Приблизительная оценка: если объект в центре в 2 раза больше, чем на краю,
                # и находится в 2 раза ближе, то фокусное расстояние можно оценить как:
                # f ≈ (размер_изображения / угол_покрытия) * tan(угол_покрытия/2)
                
                # Используем отношение размеров и расстояний для грубой оценки
                angular_coverage_factor = distance_ratio / (dist_near / dist_far) if dist_near > 0 else 1
                
                # Оценка фокусного расстояния из отношения перспективных искажений
                estimated_focal = image_width * 0.5 / np.tan(np.arccos(1 / (angular_coverage_factor + 1)))
                
                if np.isfinite(estimated_focal) and 100 < estimated_focal < 10000:
                    print(f"Оценка фокусного расстояния из перспективных искажений: {estimated_focal:.2f} пикселей")
                    return estimated_focal, estimated_focal
    
    except Exception as e:
        print(f"Ошибка при оценке фокусного расстояния из геометрических ограничений: {str(e)}")
        import traceback
        traceback.print_exc()
        
    return None, None

def refine_focal_length_estimation_multi_method(image_path, K_initial, points_2d=None, points_3d=None, image_width=None, image_height=None):
    """
    Комплексная оценка фокусного расстояния с использованием нескольких методов и их комбинации.
    
    Args:
        image_path: Путь к изображению
        K_initial: Начальная матрица калибровки
        points_2d: 2D точки на изображении (опционально)
        points_3d: Соответствующие 3D точки (опционально)
        image_width: Ширина изображения (опционально, если не указана, будет извлечена из файла)
        image_height: Высота изображения (опционально, если не указана, будет извлечена из файла)
        
    Returns:
        tuple: (fx, fy) улучшенные фокусные расстояния
    """
    try:
        import cv2
        import numpy as np
        
        # Получаем размеры изображения, если не указаны
        if image_width is None or image_height is None:
            image = cv2.imread(image_path)
            if image is not None:
                h, w = image.shape[:2]
                if image_width is None:
                    image_width = w
                if image_height is None:
                    image_height = h
            else:
                print(f"Не удалось определить размеры изображения: {image_path}")
                return K_initial[0, 0], K_initial[1, 1]
        
        # Создаем структуру для хранения оценок и их достоверности
        FocalEstimate = lambda fx, fy, confidence, method: (fx, fy, confidence, method)
        estimates = []
        
        # 1. Оценка из EXIF данных
        exif_focal = compute_focal_length_from_exif(image_path, image_width, image_height)
        if exif_focal is not None:
            estimates.append(FocalEstimate(exif_focal, exif_focal, 0.9, "EXIF"))
            print(f"  - EXIF метод: {exif_focal:.2f}px (достоверность: 0.9)")
        
        # 2. Оценка из точек схода
        vp_fx, vp_fy = estimate_focal_length_from_vanishing_points_advanced(image_path, K_initial, points_2d, points_3d)
        if vp_fx is not None and vp_fy is not None:
            estimates.append(FocalEstimate(vp_fx, vp_fy, 0.7, "vanishing_points"))
            print(f"  - Точки схода: fx={vp_fx:.2f}, fy={vp_fy:.2f} (достоверность: 0.7)")
        
        # 3. Оценка из геометрических ограничений
        geo_fx, geo_fy = estimate_focal_length_from_geometric_constraints(image_path, K_initial, points_2d, points_3d)
        if geo_fx is not None and geo_fy is not None:
            estimates.append(FocalEstimate(geo_fx, geo_fy, 0.6, "geometric_constraints"))
            print(f"  - Геометрические ограничения: fx={geo_fx:.2f}, fy={geo_fy:.2f} (достоверность: 0.6)")
        
        # 4. Используем начальную оценку как резерв
        initial_fx, initial_fy = K_initial[0, 0], K_initial[1, 1]
        estimates.append(FocalEstimate(initial_fx, initial_fy, 0.4, "initial_guess"))
        print(f"  - Начальное приближение: fx={initial_fx:.2f}, fy={initial_fy:.2f} (достоверность: 0.4)")
        
        # 5. Оценка на основе типичных значений для размера изображения
        # Для изображения 1600x1200 с 35мм сенсором, типичные значения фокусного расстояния 24-200мм
        # что соответствует примерно 1000-8000 пикселям для этого размера изображения
        typical_ranges = [
            (image_width * 0.5, 0.3, "wide_angle"),      # 24мм эквивалент
            (image_width * 0.7, 0.5, "standard"),       # 35мм эквивалент
            (image_width * 1.0, 0.6, "normal"),         # 50мм эквивалент
            (image_width * 1.7, 0.4, "portrait"),       # 85мм эквивалент
            (image_width * 2.0, 0.2, "telephoto")       # 100мм+ эквивалент
        ]
        
        for focal_val, conf, desc in typical_ranges:
            estimates.append(FocalEstimate(focal_val, focal_val, conf, f"typical_{desc}"))
            print(f"  - Типичное значение ({desc}): {focal_val:.2f}px (достоверность: {conf})")
        
        if not estimates:
            print("Нет доступных оценок фокусного расстояния, используем начальное приближение")
            return initial_fx, initial_fy
        
        # Комбинируем оценки на основе достоверности
        # Используем взвешенное среднее, но с учетом согласованности оценок
        total_weight = sum(est[2] for est in estimates)
        if total_weight > 0:
            # Вычисляем взвешенное среднее
            weighted_fx = sum(est[0] * est[2] for est in estimates) / total_weight
            weighted_fy = sum(est[1] * est[2] for est in estimates) / total_weight
            
            # Вычисляем стандартное отклонение оценок для проверки согласованности
            fx_values = np.array([est[0] for est in estimates])
            fy_values = np.array([est[1] for est in estimates])
            confidences = np.array([est[2] for est in estimates])
            
            mean_fx = np.average(fx_values, weights=confidences)
            mean_fy = np.average(fy_values, weights=confidences)
            std_fx = np.sqrt(np.average((fx_values - mean_fx)**2, weights=confidences))
            std_fy = np.sqrt(np.average((fy_values - mean_fy)**2, weights=confidences))
            
            # Если стандартное отклонение велико, это указывает на низкую согласованность
            # В таких случаях мы можем повысить вес более надежных методов
            cv_fx = std_fx / (mean_fx + 1e-10) if mean_fx != 0 else float('inf')
            cv_fy = std_fy / (mean_fy + 1e-10) if mean_fy != 0 else float('inf')
            
            print(f"Статистика оценок: среднее fx={mean_fx:.2f}, fy={mean_fy:.2f}, CV fx={cv_fx:.3f}, CV fy={cv_fy:.3f}")
            
            # Если коэффициент вариации больше 0.3 (30%), это указывает на высокую вариативность
            if cv_fx > 0.3 or cv_fy > 0.3:
                print("Высокая вариативность оценок, используем только высокодостоверные методы")
                
                # Используем только оценки с высокой достоверностью (>0.6)
                high_conf_estimates = [est for est in estimates if est[2] > 0.6]
                
                if high_conf_estimates:
                    high_conf_total_weight = sum(est[2] for est in high_conf_estimates)
                    if high_conf_total_weight > 0:
                        weighted_fx = sum(est[0] * est[2] for est in high_conf_estimates) / high_conf_total_weight
                        weighted_fy = sum(est[1] * est[2] for est in high_conf_estimates) / high_conf_total_weight
                        print(f"  - Используем только высокодостоверные оценки: fx={weighted_fx:.2f}, fy={weighted_fy:.2f}")
                    else:
                        # Если нет высокодостоверных оценок, используем медиану
                        weighted_fx = np.median([est[0] for est in estimates])
                        weighted_fy = np.median([est[1] for est in estimates])
                        print(f"  - Используем медиану оценок: fx={weighted_fx:.2f}, fy={weighted_fy:.2f}")
                else:
                    # Если нет высокодостоверных оценок, используем медиану
                    weighted_fx = np.median([est[0] for est in estimates])
                    weighted_fy = np.median([est[1] for est in estimates])
                    print(f"  - Используем медиану оценок: fx={weighted_fx:.2f}, fy={weighted_fy:.2f}")
            else:
                print(f"Оценки согласованы, используем взвешенное среднее: fx={weighted_fx:.2f}, fy={weighted_fy:.2f}")
        
        # Ограничиваем результат в разумных пределах
        min_focal = min(image_width, image_height) * 0.1  # 0.1 * минимальное изображение
        max_focal = max(image_width, image_height) * 2.0  # 2.0 * максимальное изображение
        
        final_fx = np.clip(weighted_fx, min_focal, max_focal)
        final_fy = np.clip(weighted_fy, min_focal, max_focal)
        
        print(f"Итоговая оценка фокусного расстояния: fx={final_fx:.2f}, fy={final_fy:.2f}")
        
        return final_fx, final_fy
        
    except Exception as e:
        print(f"Ошибка при комплексной оценке фокусного расстояния: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # В случае ошибки возвращаем начальные значения
        return K_initial[0, 0], K_initial[1, 1]


def enhanced_triangulation_with_uncertainty(pts1, pts2, K, R, t, points_confidence=None, parallax_threshold=2.0, reprojection_threshold=3.0):
   """
   Улучшенная триангуляция с учетом неопределенности и надежности соответствий.
   """
   if points_confidence is None:
       points_confidence = np.ones(len(pts1))
   
   # Нормализуем точки
   pts1_norm = normalize_points(pts1, K)
   pts2_norm = normalize_points(pts2, K)
   
   # Создаем матрицы проекции
   P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
   P2 = K @ np.hstack((R, t.reshape(3, 1)))
   
   # Триангулируем точки
   points_4d = cv2.triangulatePoints(P1, P2, pts1_norm.T, pts2_norm.T)
   points_3d = points_4d[:3, :] / points_4d[3, :]
   
   # Вычисляем неопределенность для каждой точки
   uncertainties = np.zeros(points_3d.shape[1])
   baseline = np.linalg.norm(t)
   
   # Формируем маску валидных точек
   mask = np.ones(points_3d.shape[1], dtype=bool)
   
   for i in range(points_3d.shape[1]):
       point = points_3d[:, i]
       
       # Вектора от центров камер к точке
       vec1 = point
       vec2 = point - (-R.T @ t.ravel())  # центр второй камеры в системе координат первой
       
       # Угол между векторами (параллакс)
       cos_parallax = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
       parallax = np.arccos(np.clip(cos_parallax, -1.0, 1.0))
       parallax_deg = np.degrees(parallax)
       
       # Проверяем угол параллакса
       if parallax_deg < parallax_threshold:
           mask[i] = False
           continue
           
       # Неопределенность обратно пропорциональна параллаксу и базовой линии
       if parallax > 1e-3: # избегаем деления на 0
           uncertainties[i] = baseline / (np.sin(parallax) + 1e-10)
       else:
           uncertainties[i] = float('inf')  # очень высокая неопределенность
           
       # Проверяем, что точка находится перед обеими камерами
       if point[2] <= 0.01:
           mask[i] = False
           continue
           
       # Преобразуем точку в систему координат второй камеры
       point2 = R @ point + t.ravel()
       if point2[2] <= 0.01:
           mask[i] = False
           continue
           
       # Проверяем ошибку репроекции
       proj1 = K @ point.reshape(3, 1)
       proj1 = proj1[:2] / proj1[2]
       proj2 = K @ point2.reshape(3, 1)
       proj2 = proj2[:2] / proj2[2]
       
       error1 = np.linalg.norm(proj1.ravel() - pts1[i])
       error2 = np.linalg.norm(proj2.ravel() - pts2[i])
       
       if error1 > reprojection_threshold or error2 > reprojection_threshold:
           mask[i] = False
           continue
   
   # Учитываем надежность соответствий
   confidence_mask = points_confidence > 0.3
   mask = mask & confidence_mask
   
   return points_3d.T, mask, uncertainties


def filter_triangulated_points_by_quality(points_3d, mask, pts1, pts2, K, R, t, uncertainties=None,
                                       parallax_threshold=2.0, reprojection_threshold=3.0, uncertainty_threshold=0.95):
   """
   Фильтрация 3D точек по качеству на основе нескольких критериев.
   """
   if uncertainties is None:
       # Вычисляем неопределенности, если не предоставлены
       _, _, uncertainties = enhanced_triangulation_with_uncertainty(pts1, pts2, K, R, t)
   
   # Создаем маску для фильтрации
   quality_mask = mask.copy()
   
   # Фильтрация по неопределенности
   uncertainty_threshold_value = np.percentile(uncertainties[quality_mask], uncertainty_threshold * 100)
   high_uncertainty_mask = uncertainties > uncertainty_threshold_value
   quality_mask = quality_mask & ~high_uncertainty_mask
   
   # Проверяем точки на численную стабильность
   for i in range(points_3d.shape[0]):
       if not quality_mask[i]:
           continue
           
       point = points_3d[i]
       point2 = R @ point + t.ravel()
       
       # Проверяем, что точка не содержит NaN или Inf
       if np.any(np.isnan(point)) or np.any(np.isinf(point)) or np.any(np.isnan(point2)) or np.any(np.isinf(point2)):
           quality_mask[i] = False
           continue
           
       # Проверяем, что точка не слишком далеко от начала координат
       if np.any(np.abs(point) > 10000) or np.any(np.abs(point2) > 10000):
           quality_mask[i] = False
           continue
   
   # Фильтрация по геометрическим критериям
   for i in range(points_3d.shape[0]):
       if not quality_mask[i]:
           continue
           
       point = points_3d[i]
       point2 = R @ point + t.ravel()
       
       # Проверяем угол параллакса
       vec1 = point
       vec2 = point - (-R.T @ t.ravel())  # центр второй камеры в системе координат первой
       cos_parallax = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
       parallax = np.arccos(np.clip(cos_parallax, -1.0, 1.0))
       parallax_deg = np.degrees(parallax)
       
       if parallax_deg < parallax_threshold:
           quality_mask[i] = False
           continue
   
   # Фильтрация по ошибке репроекции
   for i in range(points_3d.shape[0]):
       if not quality_mask[i]:
           continue
           
       point = points_3d[i]
       point2 = R @ point + t.ravel()
       
       # Проецируем точки обратно на изображения
       proj1 = K @ point.reshape(3, 1)
       proj1 = proj1[:2] / proj1[2]
       proj2 = K @ point2.reshape(3, 1)
       proj2 = proj2[:2] / proj2[2]
       
       error1 = np.linalg.norm(proj1.ravel() - pts1[i])
       error2 = np.linalg.norm(proj2.ravel() - pts2[i])
       
       if error1 > reprojection_threshold or error2 > reprojection_threshold:
           quality_mask[i] = False
           continue
   
   return quality_mask
