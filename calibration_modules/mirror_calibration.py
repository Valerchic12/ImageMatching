"""
Модуль для реализации двухсторонней (зеркальной) 360° калибровки.
Позволяет создавать зеркальные копии камер и точек для получения полного обзора симметричного объекта.
"""

import numpy as np
import logging
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def estimate_symmetry_plane(calib_data: Dict, method: str = 'auto', manual_points: List = None) -> np.ndarray:
    """
    Определяет плоскость симметрии на основе существующих 3D точек или указанных пользователем точек.
    
    Args:
        calib_data: Данные калибровки
        method: Метод определения плоскости ('auto', 'manual', 'principal')
        manual_points: Список из трех point_id точек для ручного определения плоскости
    
    Returns:
        np.ndarray: Параметры плоскости симметрии [a, b, c, d] для уравнения ax + by + cz + d = 0
    """
    # Проверка наличия точек в принципе
    if 'points_3d' not in calib_data or not calib_data['points_3d']:
        logger.error("Нет 3D точек для определения плоскости симметрии")
        # Возвращаем плоскость YZ по умолчанию (x = 0)
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    if method == 'auto':
        # Автоматический метод - используем все 3D точки для оценки плоскости симметрии
        return auto_estimate_symmetry_plane(calib_data)
    elif method == 'manual' and manual_points and len(manual_points) >= 3:
        # Ручной метод - используем указанные точки
        return manual_estimate_symmetry_plane(calib_data, manual_points)
    elif method == 'principal':
        # Метод главных компонент - оценка на основе анализа PCA
        return principal_components_symmetry_plane(calib_data)
    else:
        logger.error(f"Неподдерживаемый метод определения плоскости симметрии: {method}")
        # Возвращаем плоскость YZ по умолчанию (x = 0)
        return np.array([1.0, 0.0, 0.0, 0.0])

def auto_estimate_symmetry_plane(calib_data: Dict) -> np.ndarray:
    """
    Автоматически определяет плоскость симметрии на основе распределения 3D точек.
    
    Args:
        calib_data: Данные калибровки
    
    Returns:
        np.ndarray: Параметры плоскости симметрии [a, b, c, d]
    """
    logger.info("Автоматическое определение плоскости симметрии")
    
    # Проверяем, есть ли 3D точки
    if not calib_data.get('points_3d'):
        logger.error("Нет 3D точек для определения плоскости симметрии")
        return np.array([1.0, 0.0, 0.0, 0.0])  # Плоскость YZ по умолчанию
    
    # Преобразуем 3D точки в массив numpy
    points_3d = np.array([point for point in calib_data['points_3d'].values()])
    
    # Проверяем, достаточно ли точек для анализа
    if len(points_3d) < 3:
        logger.error(f"Недостаточно точек для определения плоскости симметрии (найдено {len(points_3d)}, нужно минимум 3)")
        return np.array([1.0, 0.0, 0.0, 0.0])  # Плоскость YZ по умолчанию
    
    try:
        # Вычисляем центр масс точек
        centroid = np.mean(points_3d, axis=0)
        logger.info(f"Центр масс 3D точек: {centroid}")
        
        # Вычисляем ковариационную матрицу
        centered_points = points_3d - centroid
        cov_matrix = np.dot(centered_points.T, centered_points) / len(points_3d)
        
        # Вычисляем собственные векторы и собственные значения
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        
        # Проверяем, что все собственные значения корректны
        if np.any(np.isnan(eigvals)) or np.any(np.isinf(eigvals)):
            logger.error("Некорректные собственные значения при анализе распределения точек")
            return np.array([1.0, 0.0, 0.0, 0.0])  # Плоскость YZ по умолчанию
        
        # Сортируем собственные векторы по убыванию собственных значений
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        logger.info(f"Собственные значения: {eigvals}")
        
        # Используем собственный вектор с наименьшим собственным значением как нормаль плоскости
        normal = eigvecs[:, 2]  # Третий (наименьший) собственный вектор
        
        # Нормализуем вектор
        normal_magnitude = np.linalg.norm(normal)
        if normal_magnitude < 1e-10:
            logger.error("Слишком малая величина нормали плоскости")
            return np.array([1.0, 0.0, 0.0, 0.0])  # Плоскость YZ по умолчанию
        
        normal = normal / normal_magnitude
        
        # Вычисляем параметр d плоскости ax + by + cz + d = 0
        d = -np.dot(normal, centroid)
        
        # Формируем параметры плоскости
        plane_params = np.array([normal[0], normal[1], normal[2], d])
        
        logger.info(f"Определена плоскость симметрии: {plane_params}")
        return plane_params
        
    except Exception as e:
        logger.error(f"Ошибка при определении плоскости симметрии: {str(e)}")
        import traceback
        traceback.print_exc()
        return np.array([1.0, 0.0, 0.0, 0.0])  # Плоскость YZ по умолчанию

def manual_estimate_symmetry_plane(calib_data: Dict, point_ids: List[int]) -> np.ndarray:
    """
    Определяет плоскость симметрии на основе трех точек, указанных пользователем.
    
    Args:
        calib_data: Данные калибровки
        point_ids: Список из трех ID точек
    
    Returns:
        np.ndarray: Параметры плоскости симметрии [a, b, c, d]
    """
    logger.info(f"Ручное определение плоскости симметрии по точкам: {point_ids}")
    
    # Проверяем, что у нас есть хотя бы 3 точки
    if len(point_ids) < 3:
        logger.error("Необходимо указать минимум 3 точки для определения плоскости")
        return np.array([1.0, 0.0, 0.0, 0.0])  # Плоскость YZ по умолчанию
    
    try:
        # Получаем координаты точек
        points = []
        for point_id in point_ids[:3]:  # Используем только первые 3 точки
            if point_id in calib_data.get('points_3d', {}):
                points.append(calib_data['points_3d'][point_id])
            else:
                logger.error(f"Точка с ID {point_id} не найдена")
                return np.array([1.0, 0.0, 0.0, 0.0])  # Плоскость YZ по умолчанию
        
        # Преобразуем в массив numpy
        points = np.array(points)
        
        # Вычисляем векторы между точками
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        
        # Проверяем, что векторы не коллинеарны
        if np.linalg.norm(v1) < 1e-10 or np.linalg.norm(v2) < 1e-10:
            logger.error("Указанные точки совпадают или слишком близки друг к другу")
            return np.array([1.0, 0.0, 0.0, 0.0])  # Плоскость YZ по умолчанию
        
        # Проверяем на коллинеарность через векторное произведение
        cross_product = np.cross(v1, v2)
        if np.linalg.norm(cross_product) < 1e-10:
            logger.error("Указанные точки образуют прямую линию, невозможно определить плоскость")
            return np.array([1.0, 0.0, 0.0, 0.0])  # Плоскость YZ по умолчанию
        
        # Вычисляем нормаль к плоскости как векторное произведение
        normal = cross_product
        
        # Нормализуем вектор
        normal_magnitude = np.linalg.norm(normal)
        normal = normal / normal_magnitude
        
        # Вычисляем параметр d плоскости ax + by + cz + d = 0
        d = -np.dot(normal, points[0])
        
        # Формируем параметры плоскости
        plane_params = np.array([normal[0], normal[1], normal[2], d])
        
        logger.info(f"Определена плоскость симметрии: {plane_params}")
        return plane_params
        
    except Exception as e:
        logger.error(f"Ошибка при ручном определении плоскости симметрии: {str(e)}")
        import traceback
        traceback.print_exc()
        return np.array([1.0, 0.0, 0.0, 0.0])  # Плоскость YZ по умолчанию

def principal_components_symmetry_plane(calib_data: Dict) -> np.ndarray:
    """
    Определяет плоскость симметрии на основе анализа главных компонент распределения точек.
    
    Args:
        calib_data: Данные калибровки
    
    Returns:
        np.ndarray: Параметры плоскости симметрии [a, b, c, d]
    """
    logger.info("Определение плоскости симметрии методом главных компонент")
    
    # Проверяем, есть ли 3D точки
    if not calib_data.get('points_3d'):
        logger.error("Нет 3D точек для определения плоскости симметрии")
        return np.array([1.0, 0.0, 0.0, 0.0])  # Плоскость YZ по умолчанию
    
    # Преобразуем 3D точки в массив numpy
    points_3d = np.array([point for point in calib_data['points_3d'].values()])
    
    # Проверяем, достаточно ли точек для анализа
    if len(points_3d) < 3:
        logger.error(f"Недостаточно точек для метода главных компонент (найдено {len(points_3d)}, нужно минимум 3)")
        return np.array([1.0, 0.0, 0.0, 0.0])  # Плоскость YZ по умолчанию
    
    try:
        # Вычисляем центр масс точек
        centroid = np.mean(points_3d, axis=0)
        
        # Центрируем точки
        centered_points = points_3d - centroid
        
        # Выполняем PCA
        _, _, vh = np.linalg.svd(centered_points)
        
        # Первые два собственных вектора определяют плоскость наибольшего разброса точек
        # Третий собственный вектор (с наименьшим собственным значением) - нормаль к этой плоскости
        normal = vh[2, :]
        
        # Проверяем нормаль
        normal_magnitude = np.linalg.norm(normal)
        if normal_magnitude < 1e-10 or np.any(np.isnan(normal)) or np.any(np.isinf(normal)):
            logger.error("Некорректный вектор нормали при методе главных компонент")
            return np.array([1.0, 0.0, 0.0, 0.0])  # Плоскость YZ по умолчанию
        
        # Нормализуем вектор
        normal = normal / normal_magnitude
        
        # Вычисляем параметр d плоскости ax + by + cz + d = 0
        d = -np.dot(normal, centroid)
        
        # Формируем параметры плоскости
        plane_params = np.array([normal[0], normal[1], normal[2], d])
        
        logger.info(f"Определена плоскость симметрии методом PCA: {plane_params}")
        return plane_params
    
    except Exception as e:
        logger.error(f"Ошибка при определении плоскости симметрии методом PCA: {str(e)}")
        import traceback
        traceback.print_exc()
        return np.array([1.0, 0.0, 0.0, 0.0])  # Плоскость YZ по умолчанию

def reflect_point(point: np.ndarray, plane_params: np.ndarray) -> np.ndarray:
    """
    Отражает точку относительно заданной плоскости.
    
    Args:
        point: 3D координаты точки [x, y, z]
        plane_params: Параметры плоскости [a, b, c, d]
    
    Returns:
        np.ndarray: Отраженная точка
    """
    # Извлекаем нормаль плоскости
    normal = plane_params[:3]
    d = plane_params[3]
    
    # Нормализуем нормаль
    normal = normal / np.linalg.norm(normal)
    
    # Вычисляем расстояние от точки до плоскости
    dist = np.dot(normal, point) + d
    
    # Вычисляем отраженную точку
    reflected_point = point - 2 * dist * normal
    
    return reflected_point

def reflect_camera(R: np.ndarray, t: np.ndarray, plane_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Отражает камеру относительно заданной плоскости.
    
    Args:
        R: Матрица поворота камеры
        t: Вектор переноса камеры
        plane_params: Параметры плоскости [a, b, c, d]
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Отраженные матрица поворота и вектор переноса
    """
    # Извлекаем нормаль плоскости
    normal = plane_params[:3]
    d = plane_params[3]
    
    # Нормализуем нормаль
    normal = normal / np.linalg.norm(normal)
    
    # Матрица отражения
    reflection_matrix = np.eye(3) - 2 * np.outer(normal, normal)
    
    # Отражаем центр камеры
    # Центр камеры в мировых координатах: C = -R^T * t
    C = -R.T @ t
    
    # Отражаем центр камеры
    reflected_C = reflect_point(C, plane_params)
    
    # Отражаем матрицу поворота
    # Новая матрица поворота: R_reflected = reflection_matrix @ R
    reflected_R = reflection_matrix @ R
    
    # Вычисляем новый вектор переноса
    # t_reflected = -reflected_R @ reflected_C
    reflected_t = -reflected_R @ reflected_C
    
    return reflected_R, reflected_t

def create_mirrored_calibration(calib_data: Dict, plane_params: np.ndarray) -> Dict:
    """
    Создает зеркальную копию калибровки относительно заданной плоскости.
    
    Args:
        calib_data: Данные калибровки
        plane_params: Параметры плоскости симметрии [a, b, c, d]
    
    Returns:
        Dict: Расширенные данные калибровки с зеркальными камерами и точками
    """
    logger.info("Создание зеркальной калибровки")
    
    # Создаем копию данных калибровки
    mirrored_data = calib_data.copy()
    
    # Словарь для хранения информации о зеркальных камерах и точках
    if 'mirror_data' not in mirrored_data:
        mirrored_data['mirror_data'] = {
            'original_cameras': set(calib_data['cameras'].keys()),
            'mirrored_cameras': set(),
            'plane_params': plane_params,
            'camera_mapping': {}  # Словарь соответствия {original_id: mirrored_id}
        }
    
    # Зеркалируем камеры
    for camera_id, (R, t) in calib_data['cameras'].items():
        # Пропускаем уже зеркальные камеры
        if camera_id in mirrored_data['mirror_data']['mirrored_cameras']:
            continue
        
        # Создаем ID для зеркальной камеры
        mirrored_camera_id = f"mirror_{camera_id}"
        
        # Отражаем камеру
        mirrored_R, mirrored_t = reflect_camera(R, t, plane_params)
        
        # Добавляем зеркальную камеру
        mirrored_data['cameras'][mirrored_camera_id] = (mirrored_R, mirrored_t)
        
        # Обновляем информацию о зеркальных камерах
        mirrored_data['mirror_data']['mirrored_cameras'].add(mirrored_camera_id)
        mirrored_data['mirror_data']['camera_mapping'][camera_id] = mirrored_camera_id
        
        logger.info(f"Создана зеркальная камера {mirrored_camera_id} для камеры {camera_id}")
    
    # Зеркалируем 3D точки
    mirrored_points_3d = {}
    for point_id, point in calib_data['points_3d'].items():
        mirrored_point = reflect_point(point, plane_params)
        mirrored_point_id = f"mirror_{point_id}"
        mirrored_points_3d[mirrored_point_id] = mirrored_point
    
    # Добавляем зеркальные точки к существующим
    mirrored_data['points_3d'].update(mirrored_points_3d)
    
    # Создаем проекции зеркальных точек на зеркальные камеры
    for camera_id in calib_data['camera_points']:
        # Получаем соответствующую зеркальную камеру
        mirrored_camera_id = mirrored_data['mirror_data']['camera_mapping'].get(camera_id)
        if not mirrored_camera_id:
            continue
        
        # Инициализируем словарь точек для зеркальной камеры
        if mirrored_camera_id not in mirrored_data['camera_points']:
            mirrored_data['camera_points'][mirrored_camera_id] = {}
        
        # Получаем матрицу внутренних параметров
        K = mirrored_data['K']
        dist_coeffs = mirrored_data['dist_coeffs']
        
        # Получаем позу зеркальной камеры
        mirrored_R, mirrored_t = mirrored_data['cameras'][mirrored_camera_id]
        
        # Для каждой точки на исходной камере
        for point_id, point_2d in calib_data['camera_points'][camera_id].items():
            # Соответствующая зеркальная точка
            mirrored_point_id = f"mirror_{point_id}"
            
            # Если зеркальная 3D точка существует
            if mirrored_point_id in mirrored_data['points_3d']:
                # Получаем 3D координаты зеркальной точки
                point_3d = mirrored_data['points_3d'][mirrored_point_id]
                
                # Проецируем 3D точку на зеркальную камеру
                projected_point, _ = cv2.projectPoints(
                    point_3d.reshape(1, 3), 
                    cv2.Rodrigues(mirrored_R)[0], 
                    mirrored_t, 
                    K, 
                    dist_coeffs
                )
                
                # Сохраняем проекцию
                mirrored_data['camera_points'][mirrored_camera_id][mirrored_point_id] = projected_point.reshape(2)
    
    logger.info("Зеркальная калибровка создана успешно")
    return mirrored_data

def visualize_symmetry_plane(calib_data: Dict, scale: float = 1.0) -> Dict:
    """
    Создает данные для визуализации плоскости симметрии в Blender.
    
    Args:
        calib_data: Данные калибровки
        scale: Масштаб плоскости
    
    Returns:
        Dict: Данные для создания плоскости в Blender
    """
    if 'mirror_data' not in calib_data or 'plane_params' not in calib_data['mirror_data']:
        logger.error("Данные о плоскости симметрии отсутствуют")
        return None
    
    # Получаем параметры плоскости
    plane_params = calib_data['mirror_data']['plane_params']
    normal = plane_params[:3]
    d = plane_params[3]
    
    # Нормализуем нормаль
    normal = normal / np.linalg.norm(normal)
    
    # Проверяем наличие 3D точек
    if 'points_3d' not in calib_data or not calib_data['points_3d']:
        logger.warning("Нет 3D точек для определения размера плоскости, используем значения по умолчанию")
        centroid = np.array([0.0, 0.0, 0.0])
        plane_size = 5.0 * scale  # Разумный размер по умолчанию
    else:
        # Находим центр всех 3D точек
        all_points = np.array(list(calib_data['points_3d'].values()))
        
        # Проверяем, что массив точек не пустой
        if len(all_points) == 0:
            logger.warning("Пустой массив 3D точек, используем значения по умолчанию")
            centroid = np.array([0.0, 0.0, 0.0])
            plane_size = 5.0 * scale  # Разумный размер по умолчанию
        else:
            # Вычисляем центроид и размер
            centroid = np.mean(all_points, axis=0)
            # Вычисляем размер плоскости на основе размаха точек
            points_range = np.max(all_points, axis=0) - np.min(all_points, axis=0)
            plane_size = np.max(points_range) * scale
    
    # Вычисляем точку на плоскости, ближайшую к центроиду
    plane_point = centroid - (np.dot(normal, centroid) + d) * normal
    
    # Создаем локальные оси для плоскости
    # Первая ось - проекция оси X на плоскость
    x_axis = np.array([1, 0, 0])
    if abs(np.dot(normal, x_axis)) > 0.9:
        # Если нормаль близка к оси X, используем ось Y
        x_axis = np.array([0, 1, 0])
    
    # Проецируем ось X на плоскость
    x_axis = x_axis - np.dot(x_axis, normal) * normal
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Вторая ось - перпендикулярна первой и нормали
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Создаем вершины плоскости
    vertices = [
        plane_point + plane_size * (-x_axis - y_axis),
        plane_point + plane_size * (x_axis - y_axis),
        plane_point + plane_size * (x_axis + y_axis),
        plane_point + plane_size * (-x_axis + y_axis)
    ]
    
    # Возвращаем данные для визуализации
    return {
        'vertices': vertices,
        'normal': normal,
        'center': plane_point
    }

def generate_mirrored_points(calib_data: Dict) -> Dict:
    """
    Генерирует зеркальные точки на основе существующих 3D точек и плоскости симметрии.
    Функция предназначена для использования до выполнения основной калибровки.
    
    Args:
        calib_data: Данные калибровки, содержащие 'points_3d' и 'mirror_data'
    
    Returns:
        Dict: Словарь зеркальных точек {mirror_<id>: point_3d}
    """
    logger.info("Генерация зеркальных точек для калибровки")
    
    # Проверяем наличие необходимых данных
    if 'points_3d' not in calib_data or 'mirror_data' not in calib_data:
        logger.error("Отсутствуют данные о 3D точках или плоскости симметрии")
        return {}
    
    # Проверяем наличие параметров плоскости симметрии
    if 'plane_params' not in calib_data['mirror_data']:
        logger.error("Отсутствуют параметры плоскости симметрии")
        return {}
    
    # Получаем параметры плоскости
    plane_params = calib_data['mirror_data']['plane_params']
    
    # Словарь зеркальных точек
    mirrored_points = {}
    
    # Отражаем каждую точку относительно плоскости
    for point_id, point in calib_data['points_3d'].items():
        # Отражаем точку
        mirrored_point = reflect_point(point, plane_params)
        
        # Сохраняем отраженную точку с префиксом 'mirror_'
        mirrored_points[f"mirror_{point_id}"] = mirrored_point
    
    logger.info(f"Сгенерировано {len(mirrored_points)} зеркальных точек")
    return mirrored_points 