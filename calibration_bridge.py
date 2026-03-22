"""
Модуль-мост между аддоном и системой калибровки.
Преобразует данные из формата аддона в формат системы калибровки и обратно.
"""

try:
    import bpy
except:
    print("Использует тест без Блендер")
    pass
import os
import time
import logging
import traceback
import cv2

# Настройка логирования
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _camera_id_to_index(camera_id):
    """Преобразует ID камеры в целочисленный индекс, если это возможно."""
    if isinstance(camera_id, int):
        return camera_id

    if isinstance(camera_id, str):
        try:
            return int(camera_id)
        except ValueError:
            return None

    try:
        return int(camera_id)
    except (TypeError, ValueError):
        return None


def _camera_sort_key(camera_id):
    camera_str = str(camera_id)
    try:
        return (0, int(camera_str))
    except ValueError:
        return (1, camera_str)


def _format_triplet(values):
    return ", ".join(f"{float(value):.3f}" for value in values)


def _camera_center_from_rt(R, t):
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    return (-R.T @ t).reshape(3)


def _log_reconstruction_camera_centers(cameras_cv, image_map=None, title="Центры камер реконструкции"):
    if not cameras_cv:
        return

    logger.info(title)
    centers = []
    for camera_id in sorted(cameras_cv.keys(), key=_camera_sort_key):
        R, t = cameras_cv[camera_id]
        center = _camera_center_from_rt(R, t)
        centers.append(center)
        image_name = None
        if image_map:
            image_name = image_map.get(str(camera_id), image_map.get(camera_id))
        label = image_name if image_name else f"camera_{camera_id}"
        logger.info(f"  - Камера {camera_id} ({label}): C=({_format_triplet(center)})")

    centers = np.asarray(centers, dtype=np.float64)
    if len(centers) >= 1:
        min_corner = np.min(centers, axis=0)
        max_corner = np.max(centers, axis=0)
        logger.info(
            "  - Диапазон центров камер: "
            f"min=({_format_triplet(min_corner)}), max=({_format_triplet(max_corner)})"
        )


def _log_blender_camera_object_positions(camera_objects, title="Позиции камер в Blender после применения"):
    if not camera_objects:
        return

    logger.info(title)
    positions = []
    for camera_id in sorted(camera_objects.keys(), key=_camera_sort_key):
        camera_obj = camera_objects[camera_id]
        location = np.asarray(camera_obj.matrix_world.translation, dtype=np.float64).reshape(3)
        positions.append(location)
        logger.info(f"  - Камера {camera_id} ({camera_obj.name}): loc=({_format_triplet(location)})")

    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) >= 1:
        min_corner = np.min(positions, axis=0)
        max_corner = np.max(positions, axis=0)
        logger.info(
            "  - Диапазон позиций камер Blender: "
            f"min=({_format_triplet(min_corner)}), max=({_format_triplet(max_corner)})"
        )


def _log_point_cloud_bounds(points_3d, title="Границы облака 3D точек"):
    if not points_3d:
        return

    points = np.asarray(
        [np.asarray(point_3d, dtype=np.float64).reshape(3) for point_3d in points_3d.values()],
        dtype=np.float64
    )
    min_corner = np.min(points, axis=0)
    max_corner = np.max(points, axis=0)
    logger.info(
        f"{title}: min=({_format_triplet(min_corner)}), max=({_format_triplet(max_corner)})"
    )


def _get_visualization_point_cloud():
    """
    Возвращает облако точек для Blender-визуализации.

    Для моделинга используем не только strict-каркас points_3d, но и
    secondary_points_3d, если они были построены solver'ом.
    """
    points_3d = get_3d_points() or {}
    merged_points = dict(points_3d)

    calib_data = getattr(calibration, 'calibration_data', None)
    if isinstance(calib_data, dict):
        secondary_points = calib_data.get('secondary_points_3d') or {}
        for point_id, point_3d in secondary_points.items():
            if point_id not in merged_points:
                merged_points[point_id] = point_3d

    return merged_points


def _resolve_common_image_size(props, calib_data=None):
    image_sizes = []
    seen_sizes = set()

    for image_item in getattr(props, 'images', []):
        image = bpy.data.images.get(image_item.name)
        if image and image.size[0] > 0 and image.size[1] > 0:
            size = (int(image.size[0]), int(image.size[1]))
            image_sizes.append(size)
            seen_sizes.add(size)

    if len(seen_sizes) == 1:
        return image_sizes[0]

    if len(seen_sizes) > 1:
        logger.warning(
            f"Изображения имеют разные размеры: {sorted(seen_sizes)}. "
            "Для render resolution будет использован размер первого доступного изображения."
        )
        return image_sizes[0]

    if calib_data is not None:
        width = int(calib_data.get('image_width', 0) or 0)
        height = int(calib_data.get('image_height', 0) or 0)
        if width > 0 and height > 0:
            return width, height

    return None, None


def _sync_scene_render_to_images(props, calib_data=None):
    width, height = _resolve_common_image_size(props, calib_data=calib_data)
    if not width or not height:
        logger.warning("Не удалось определить размер изображений для синхронизации render resolution")
        return False

    render = bpy.context.scene.render
    render.resolution_x = int(width)
    render.resolution_y = int(height)
    render.resolution_percentage = 100
    render.pixel_aspect_x = 1.0
    render.pixel_aspect_y = 1.0
    logger.info(f"Render resolution синхронизирован с изображениями: {width}x{height}")
    return True


def _normalize_vector(vec, eps=1e-8):
    """Нормализует вектор, возвращая None для вырожденного случая."""
    arr = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(arr)
    if norm < eps:
        return None
    return arr / norm


def _camera_center_from_pose(R, t):
    """Возвращает центр камеры в мировых координатах реконструкции."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    return (-R.T @ t).reshape(3)


def _build_blender_scene_transform(cameras_cv, points_3d):
    """
    Строит жесткое преобразование для экспорта в Blender.

    Реконструкция SfM имеет произвольную глобальную ориентацию. Для Blender
    поворачиваем всю сцену так, чтобы плоскость движения камер стала
    горизонтальной, а облако точек оказалось центрировано вокруг начала координат.
    """
    identity = {
        'rotation': np.eye(3, dtype=np.float32),
        'translation': np.zeros(3, dtype=np.float32),
        'aligned': False,
    }

    if not cameras_cv or len(cameras_cv) < 3:
        return identity

    camera_centers = []
    camera_ups = []
    for R, t in cameras_cv.values():
        center = _camera_center_from_pose(R, t)
        camera_centers.append(center)

        # В OpenCV локальная ось Y направлена вниз, поэтому "up" = -Y.
        camera_up = np.asarray(R, dtype=np.float64).reshape(3, 3).T @ np.array([0.0, -1.0, 0.0])
        camera_up = _normalize_vector(camera_up)
        if camera_up is not None:
            camera_ups.append(camera_up)

    if len(camera_centers) < 3:
        return identity

    camera_centers = np.asarray(camera_centers, dtype=np.float64)
    camera_centroid = np.mean(camera_centers, axis=0)

    try:
        _, singular_values, vh = np.linalg.svd(camera_centers - camera_centroid, full_matrices=False)
    except np.linalg.LinAlgError:
        return identity

    if len(singular_values) < 3 or singular_values[1] < 1e-6:
        return identity

    orbit_normal = _normalize_vector(vh[-1])
    if orbit_normal is None:
        return identity

    if camera_ups:
        average_up = _normalize_vector(np.mean(np.asarray(camera_ups, dtype=np.float64), axis=0))
        if average_up is not None and np.dot(average_up, orbit_normal) < 0.0:
            orbit_normal = -orbit_normal

    if points_3d:
        point_values = [
            np.asarray(point_3d, dtype=np.float64).reshape(3)
            for point_3d in points_3d.values()
        ]
        scene_origin = np.mean(np.asarray(point_values, dtype=np.float64), axis=0)
    else:
        scene_origin = camera_centroid

    # Направление от среднего положения камер к объекту задает "перед" сцены.
    orbit_forward = scene_origin - camera_centroid
    orbit_forward = orbit_forward - orbit_normal * np.dot(orbit_forward, orbit_normal)
    orbit_forward = _normalize_vector(orbit_forward)

    if orbit_forward is None:
        primary_axis = vh[0] - orbit_normal * np.dot(vh[0], orbit_normal)
        orbit_right = _normalize_vector(primary_axis)
        if orbit_right is None:
            return identity
        orbit_forward = _normalize_vector(np.cross(orbit_normal, orbit_right))
        if orbit_forward is None:
            return identity
        orbit_right = _normalize_vector(np.cross(orbit_forward, orbit_normal))
    else:
        orbit_right = _normalize_vector(np.cross(orbit_forward, orbit_normal))
        if orbit_right is None:
            return identity
        orbit_forward = _normalize_vector(np.cross(orbit_normal, orbit_right))
        if orbit_forward is None:
            return identity

    source_basis = np.column_stack([orbit_right, orbit_forward, orbit_normal])
    if np.linalg.det(source_basis) < 0.0:
        orbit_right = -orbit_right
        source_basis = np.column_stack([orbit_right, orbit_forward, orbit_normal])

    rotation = source_basis.T
    translation = -rotation @ scene_origin

    return {
        'rotation': rotation.astype(np.float32),
        'translation': translation.astype(np.float32),
        'aligned': True,
        'scene_origin': np.asarray(scene_origin, dtype=np.float32),
        'orbit_normal': np.asarray(orbit_normal, dtype=np.float32),
    }


def _get_blender_scene_transform(calib_data, cameras_cv=None, points_3d=None):
    """Получает или вычисляет transform для визуализации реконструкции в Blender."""
    if calib_data is None:
        return _build_blender_scene_transform(cameras_cv or {}, points_3d or {})

    transform = calib_data.get('blender_scene_transform')
    if transform is not None:
        return transform

    if cameras_cv is None:
        cameras_cv = calib_data.get('cameras', {})
    if points_3d is None:
        points_3d = calib_data.get('points_3d', {})

    transform = _build_blender_scene_transform(cameras_cv or {}, points_3d or {})
    calib_data['blender_scene_transform'] = transform

    if transform.get('aligned'):
        normal = transform.get('orbit_normal')
        logger.info(
            "Выравнивание сцены для Blender: orbit normal=(%.3f, %.3f, %.3f)",
            float(normal[0]),
            float(normal[1]),
            float(normal[2]),
        )

    return transform


def _transform_point_for_blender(point_3d, scene_transform):
    """Применяет transform визуализации Blender к 3D-точке."""
    point = np.asarray(point_3d, dtype=np.float32).reshape(3)
    if not scene_transform:
        return point

    rotation = np.asarray(scene_transform.get('rotation', np.eye(3)), dtype=np.float32).reshape(3, 3)
    translation = np.asarray(scene_transform.get('translation', np.zeros(3)), dtype=np.float32).reshape(3)
    return rotation @ point + translation


def _transform_camera_pose_for_blender(R, t, scene_transform):
    """Преобразует позу камеры в выровненную мировую систему Blender."""
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    camera_center = _camera_center_from_pose(R, t).astype(np.float32)
    camera_to_world_cv = R.T.astype(np.float32)

    if not scene_transform:
        return camera_to_world_cv, camera_center

    rotation = np.asarray(scene_transform.get('rotation', np.eye(3)), dtype=np.float32).reshape(3, 3)
    translation = np.asarray(scene_transform.get('translation', np.zeros(3)), dtype=np.float32).reshape(3)

    aligned_camera_to_world = rotation @ camera_to_world_cv
    aligned_center = rotation @ camera_center + translation
    return aligned_camera_to_world, aligned_center


def apply_cameras_to_blender(camera_objects, props):
    """
    Применяет позы и параметры камеры к уже созданным объектам Blender.
    """
    if not camera_objects:
        logger.error("Невозможно применить позы камер: список объектов пуст")
        return False

    try:
        import mathutils
    except ImportError:
        logger.error("Невозможно применить позы камер: mathutils недоступен")
        return False

    cameras_cv = get_camera_poses()
    calib_data = getattr(calibration, 'calibration_data', None)
    if not cameras_cv or not calib_data:
        logger.error("Невозможно применить позы камер: нет данных калибровки")
        return False

    common_K = calib_data.get('K')
    sensor_width = getattr(props, 'sensor_width', 36.0)
    for camera_id, camera_obj in camera_objects.items():
        if camera_id not in cameras_cv:
            logger.warning(f"Поза для камеры {camera_id} не найдена")
            continue

        R, t = cameras_cv[camera_id]
        R = np.asarray(R, dtype=np.float32).reshape(3, 3)
        t = np.asarray(t, dtype=np.float32).reshape(3, 1)

        # Преобразование системы координат OpenCV -> Blender.
        R_flip = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ], dtype=np.float32)
        R_blender = R.T @ R_flip
        camera_center = (-R.T @ t).reshape(3)

        matrix = mathutils.Matrix.Identity(4)
        for i in range(3):
            for j in range(3):
                matrix[i][j] = float(R_blender[i, j])
            matrix[i][3] = float(camera_center[i])

        camera_obj.matrix_world = matrix
        camera_obj.data.sensor_fit = 'HORIZONTAL'
        camera_obj.data.sensor_width = sensor_width

        image_width = None
        image_height = None
        camera_index = _camera_id_to_index(camera_id)
        if camera_index is not None and 0 <= camera_index < len(props.images):
            image = bpy.data.images.get(props.images[camera_index].name)
            if image and image.size[0] > 0 and image.size[1] > 0:
                image_width = image.size[0]
                image_height = image.size[1]

        if not image_width:
            image_width = calib_data.get('image_width', 1600)
        if not image_height:
            image_height = calib_data.get('image_height', 1200)

        individual_K = calib_data.get(f'K_{camera_id}')
        K = individual_K if individual_K is not None else common_K
        if K is None:
            logger.warning(f"Для камеры {camera_id} матрица калибровки не найдена")
            continue

        fx = float(K[0, 0])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        focal_mm = fx * sensor_width / float(image_width)
        camera_obj.data.lens = focal_mm

        # Подстраиваем principal point под Blender shifts.
        camera_obj.data.shift_x = -((cx - image_width * 0.5) / image_width)
        camera_obj.data.shift_y = (cy - image_height * 0.5) / image_width

    return True


def calculate_camera_fov(camera_matrix, image_width, image_height):
    """
    Рассчитывает горизонтальный и вертикальный FOV по матрице калибровки.
    """
    try:
        if camera_matrix is None:
            return None, None

        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1]) if float(camera_matrix[1, 1]) != 0 else fx
        if fx <= 0 or fy <= 0 or image_width <= 0 or image_height <= 0:
            return None, None

        horizontal_fov = 2.0 * np.degrees(np.arctan(float(image_width) / (2.0 * fx)))
        vertical_fov = 2.0 * np.degrees(np.arctan(float(image_height) / (2.0 * fy)))
        return horizontal_fov, vertical_fov
    except Exception as e:
        logger.error(f"Ошибка при расчете базового FOV: {str(e)}")
        return None, None

# Import from centralized import system
from core_imports import (
    check_core_dependencies,
    get_calibration_modules,
    get_main_utils
)

# Импортируем модуль калибровки с обработкой относительных импортов
try:
    from . import calibration
except ImportError:
    import calibration  # fallback для случаев, когда аддон запускается как скрипт

# Проверяем наличие необходимых пакетов перед импортом
DEPENDENCIES_INSTALLED = check_core_dependencies()

# Безопасные импорты
if DEPENDENCIES_INSTALLED:
    try:
        import numpy as np
        # Импортируем систему калибровки
        calibration_core, camera_pose, triangulation, bundle_adjustment = get_calibration_modules()
        utils_module = get_main_utils()
        
        # Импортируем функции из calibration.py
        try:
            from .calibration import (
                init_calibration,
                set_points_from_blender,
                run_calibration,
                get_camera_poses,
                get_3d_points,
                calibration_data,
                set_camera_parameters
            )
        except ImportError:
            from calibration import (
                init_calibration,
                set_points_from_blender,
                run_calibration,
                get_camera_poses,
                get_3d_points,
                calibration_data,
                set_camera_parameters
            )
        DEPENDENCIES_INSTALLED = True
    except ImportError as e:
        logger.error(f"Ошибка при импорте модулей: {str(e)}")
        DEPENDENCIES_INSTALLED = False
else:
    DEPENDENCIES_INSTALLED = False

def convert_addon_points_to_calibration_format(props):
    """
    Преобразует точки из формата аддона в формат системы калибровки.
    
    Args:
        props: Свойства аддона CAMCALIB_props
    
    Returns:
        dict: Словарь данных точек в формате {camera_id: {point_id: point_2d}}
    """
    if not DEPENDENCIES_INSTALLED:
        logger.error("Невозможно конвертировать точки: отсутствуют зависимости")
        return {}
    
    # Импортируем необходимые модули
    from calibration_modules.utils import invert_y_coordinate
    
    logger.info("Конвертация точек из формата аддона в формат системы калибровки")
    logger.info(f"Количество изображений: {len(props.images)}")
    logger.info(f"Количество групп точек: {len(props.point_groups)}")
        
    camera_points = {}
    
    # Для каждого изображения (камеры)
    for img_idx, image_item in enumerate(props.images):
        # Подсчитываем количество точек на изображении
        placed_points = sum(1 for p in image_item.points if p.is_placed and p.point_group_id >= 0)
        if placed_points <= 0:
            logger.info(f"Пропуск изображения {img_idx} ({image_item.name}): нет связанных точек")
            continue

        camera_id = img_idx  # Используем индекс изображения как ID камеры
        camera_points[camera_id] = {}
        
        logger.info(f"Обработка изображения {img_idx} ({image_item.name})")
        logger.info(f"  - Всего размещенных точек: {placed_points}")
        
        # Получаем реальную высоту изображения из Blender
        image = bpy.data.images.get(image_item.name)
        if image:
            image_height = image.size[1]
            logger.info(f"  - Размер изображения: {image.size[0]}x{image_height}")
        else:
            # Предполагаем стандартную высоту, если изображение недоступно
            image_height = 1200
            logger.warning(f"  - Не удалось получить реальный размер изображения {image_item.name}, используется значение по умолчанию 1200")
        
        # Для каждой точки на изображении
        for point_idx, point in enumerate(image_item.points):
            if point.is_placed and point.point_group_id >= 0:
                # Получаем координаты точки
                x, y = point.location_2d
                
                # Инвертируем Y-координату для перехода из системы координат Blender в OpenCV
                y_opencv = invert_y_coordinate(y, image_height=image_height)
                
                # Сохраняем точку как 2D координаты
                point_id = point.point_group_id
                point_2d = np.array([x, y_opencv], dtype=np.float32)
                if camera_id not in camera_points:
                    camera_points[camera_id] = {}
                camera_points[camera_id][point_id] = point_2d
                
                logger.debug(f"  - Точка {point_idx}: группа {point_id}, координаты Blender: ({x}, {y}), координаты OpenCV: ({x}, {y_opencv}), формат: {point_2d.shape}")
    
    # Выводим статистику по общим точкам
    logger.info("Статистика по общим точкам между изображениями:")
    for i in range(len(props.images) - 1):
        for j in range(i + 1, len(props.images)):
            if i in camera_points and j in camera_points:
                common_points = set(camera_points[i].keys()) & set(camera_points[j].keys())
                logger.info(f"  - Пара изображений {i}-{j}: {len(common_points)} общих точек")
                
                # Выводим список общих точек
                if common_points:
                    logger.debug(f"    - Общие точки: {sorted(list(common_points))}")
    
    # Проверяем, есть ли хотя бы одна пара изображений с достаточным количеством общих точек
    has_valid_pair = False
    for i in range(len(props.images) - 1):
        for j in range(i + 1, len(props.images)):
            if i in camera_points and j in camera_points:
                common_points = set(camera_points[i].keys()) & set(camera_points[j].keys())
                if len(common_points) >= 8:
                    has_valid_pair = True
                    logger.info(f"Найдена валидная пара изображений: {i}-{j} с {len(common_points)} общими точками")
                    break
        if has_valid_pair:
            break
    
    if not has_valid_pair:
        logger.error("Не найдено ни одной пары изображений с достаточным количеством общих точек (минимум 8)")
    
    return camera_points

def estimate_camera_matrix_from_addon(props):
    """
    Оценивает матрицу камеры из свойств аддона.
    
    Args:
        props: Свойства аддона CAMCALIB_props
    
    Returns:
        tuple[numpy.ndarray, dict | None]: Матрица камеры 3x3 и метаданные начальной оценки focal
    """
    import bpy
    import numpy as np
    import os
    
    # Проверяем, есть ли у нас изображения
    if not props.images:
        logger.warning("Нет изображений для оценки матрицы камеры")
        K = np.array([
            [1000, 0, 500],
            [0, 1000, 500],
            [0, 0, 1]
        ], dtype=np.float32)
        return K, {
            'mode': 'default',
            'focal_px': 1000.0,
            'confidence': 0.0,
            'fallback_focal_px': 1000.0,
        }
    
    # Берем первое изображение для оценки
    image_name = props.images[0].name
    image = bpy.data.images.get(image_name)
    image_path = bpy.path.abspath(image.filepath) if image and image.filepath else None
    
    if image is None:
        logger.warning(f"Не удалось получить изображение {image_name}, используется матрица по умолчанию")
        K = np.array([
            [1000, 0, 500],
            [0, 1000, 500],
            [0, 0, 1]
        ], dtype=np.float32)
        return K, {
            'mode': 'default',
            'focal_px': 1000.0,
            'confidence': 0.0,
            'fallback_focal_px': 1000.0,
        }
    
    # Размер изображения
    width, height = image.size
    logger.info(f"Размер изображения для оценки матрицы: {width}x{height}")
    
    # Получаем параметры из свойств аддона. Для solver это только слабые подсказки:
    # математический bootstrap теперь ведется в пикселях, без опоры на mm.
    sensor_width_mm = props.sensor_width
    initial_focal_length_mm = props.focal_length if hasattr(props, 'focal_length') else 50.0

    from calibration import compute_default_focal_prior_px

    logger.info("UI-подсказки камеры (не источник истины для solver):")
    logger.info(f"  - Ширина сенсора: {sensor_width_mm}мм")
    logger.info(f"  - Focal hint: {initial_focal_length_mm}мм")

    # Базовый fallback в пикселях больше не зависит от sensor_width/mm.
    standard_focal_length_pix = compute_default_focal_prior_px(width, height)
    logger.info(f"Pixel bootstrap prior: fx~fy~{standard_focal_length_pix:.2f}px")
    
    # Центр изображения
    cx = width / 2
    cy = height / 2
    
    # Используем новую функцию для предварительной оценки фокусного расстояния
    if image_path and os.path.exists(image_path):
        logger.info(f"Запуск расширенной оценки фокусного расстояния на основе изображения: {image_path}")
        try:
            from calibration import estimate_initial_focal_length
            
            # provided_focal_mm здесь только fallback-prior, а не жесткое целевое значение.
            fx_pix, fy_pix, confidence = estimate_initial_focal_length(
                image_path, 
                width, 
                height,
            )
            logger.info(f"Результат расширенной оценки: fx={fx_pix:.2f}, fy={fy_pix:.2f}, точность={confidence:.2f}")
            estimate_mode = 'image_based'
            
            # Если image-based оценка не дала сигнала, откатываемся в простой fallback.
            if confidence < 0.15:
                logger.warning(f"Низкая достоверность оценки фокусного расстояния ({confidence:.2f} < 0.15)")
                logger.info(f"Возврат к стандартной оценке: {standard_focal_length_pix:.2f}")
                fx_pix = fy_pix = standard_focal_length_pix
                estimate_mode = 'fallback'
            else:
                diff_percent = abs((fx_pix - standard_focal_length_pix) / max(standard_focal_length_pix, 1e-6)) * 100
                logger.info(f"Расхождение со fallback-оценкой: {diff_percent:.1f}%")
        except ImportError:
            logger.warning("Функция estimate_initial_focal_length не найдена. Используется стандартная оценка.")
            logger.info(f"Стандартная оценка фокусного расстояния: {standard_focal_length_pix:.2f}")
            fx_pix = fy_pix = standard_focal_length_pix
            confidence = 0.0
            estimate_mode = 'fallback'
        except Exception as e:
            logger.error(f"Ошибка при расширенной оценке фокусного расстояния: {e}")
            logger.warning(f"Низкая достоверность оценки фокусного расстояния (0.00 < 0.3)")
            logger.info(f"Возврат к стандартной оценке: {standard_focal_length_pix:.2f}")
            fx_pix = fy_pix = standard_focal_length_pix
            confidence = 0.0
            estimate_mode = 'fallback'
    else:
        logger.info(f"Изображение недоступно для анализа, используется стандартная оценка фокусного расстояния")
        fx_pix = fy_pix = standard_focal_length_pix
        confidence = 0.0
        estimate_mode = 'fallback'
    
    bootstrap_choice = None
    try:
        from calibration import resolve_bootstrap_focal_estimate
        bootstrap_choice = resolve_bootstrap_focal_estimate(
            {
                'focal_px': float((float(fx_pix) + float(fy_pix)) * 0.5),
                'confidence': float(confidence),
                'fallback_focal_px': float(standard_focal_length_pix),
            },
            min_confidence_for_direct_use=0.35,
        )
    except Exception:
        bootstrap_choice = None

    if bootstrap_choice and bootstrap_choice.get('source') != 'image_based':
        chosen_bootstrap_fx = float(bootstrap_choice['focal_px'])
        logger.info(
            "Bootstrap K переключен на fallback: "
            f"source={bootstrap_choice.get('source')}, "
            f"confidence={float(confidence):.2f}, fx={chosen_bootstrap_fx:.2f}"
        )
        fx_pix = fy_pix = chosen_bootstrap_fx
        estimate_mode = 'fallback'

    # Создаем матрицу камеры с полученными значениями
    K = np.array([
        [fx_pix, 0, cx],
        [0, fy_pix, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    logger.info(f"Итоговая оценка начальной матрицы камеры:")
    logger.info(f"  - Фокусное расстояние в пикселях: fx={fx_pix:.2f}, fy={fy_pix:.2f}")
    logger.info(
        "  - UI-эквивалент фокусного расстояния в мм: "
        f"{fx_pix * sensor_width_mm / width:.2f}мм (по текущему sensor_width)"
    )
    logger.info(f"  - Центр изображения: ({cx:.2f}, {cy:.2f})")
    logger.info(f"  - Матрица K:\n{K}")

    focal_estimate = {
        'mode': estimate_mode,
        'focal_px': float((float(fx_pix) + float(fy_pix)) * 0.5),
        'confidence': float(confidence),
        'fallback_focal_px': float(standard_focal_length_pix),
        'image_estimates': [{
            'image_name': str(image_name),
            'image_path': str(image_path) if image_path else None,
            'focal_px': float((float(fx_pix) + float(fy_pix)) * 0.5),
            'confidence': float(confidence),
        }],
    }

    return K, focal_estimate

def run_calibration_from_addon(props, min_points_for_camera=4, bundle_method='trf', 
                           bundle_ftol=1e-8, max_bundle_iterations=3, 
                           ransac_threshold=8.0, confidence=0.99, max_attempts=3,
                           max_reprojection_error=10.0, focal_range=(800, 3000),
                           adapt_initial_focal=True, check_focal_consistency=True,
                           auto_correct_focal=False, force_same_focal=False,
                           progress_callback=None, create_cameras=True):
    """
    Запускает калибровку из аддона Blender.
    
    Args:
        props: Свойства аддона CAMCALIB_props
        min_points_for_camera: Минимальное количество общих точек для добавления камеры
        bundle_method: Метод оптимизации для bundle adjustment ('trf', 'dogbox', 'lm')
        bundle_ftol: Порог сходимости по функции для bundle adjustment
        max_bundle_iterations: Максимальное количество итераций bundle adjustment
        ransac_threshold: Порог для RANSAC при оценке позы камеры
        confidence: Уровень доверия для RANSAC
        max_attempts: Максимальное количество попыток добавления камер
        max_reprojection_error: Максимальная допустимая ошибка репроекции при фильтрации точек (пикселей)
        focal_range: Кортеж (min_focal, max_focal) с реалистичными ограничениями для фокусного расстояния в пикселях
        adapt_initial_focal: Использовать ли механизм адаптации начального фокусного расстояния
        check_focal_consistency: Проверять согласованность фокусных расстояний между камерами
        auto_correct_focal: Автоматически корректировать аномальные фокусные расстояния
        force_same_focal: Принудительно использовать одинаковое фокусное расстояние для всех камер
    
    Returns:
        bool: True, если калибровка успешна, иначе False
    """
    if not DEPENDENCIES_INSTALLED:
        logger.error("Невозможно выполнить калибровку: отсутствуют зависимости")
        return False

    def _report_progress(progress_value, status_text):
        if progress_callback is None:
            return
        try:
            progress_callback(progress_value, status_text)
        except Exception:
            logger.exception("Ошибка в progress callback")
        
    try:
        logger.info("=" * 80)
        logger.info("НАЧАЛО ПРОЦЕССА КАЛИБРОВКИ")
        logger.info("=" * 80)
        _report_progress(5.0, "Проверка входных данных...")
        
        # Выводим информацию о данных
        logger.info(f"Количество изображений: {len(props.images)}")
        logger.info(f"Количество групп точек: {len(props.point_groups)}")

        candidate_images = []
        for img_idx, image_item in enumerate(props.images):
            points_count = sum(1 for p in image_item.points if p.is_placed and p.point_group_id >= 0)
            logger.info(f"Изображение {img_idx} ({image_item.name}): {points_count} точек")
            if points_count > 0:
                candidate_images.append((img_idx, image_item, points_count))

        if len(candidate_images) < 2:
            logger.error("Недостаточно изображений с точками для калибровки (минимум 2)")
            return False
        
        # Проверяем наличие общих точек между изображениями
        logger.info("Проверка общих точек между изображениями:")
        valid_pairs = []
        for left_index in range(len(candidate_images) - 1):
            for right_index in range(left_index + 1, len(candidate_images)):
                i, img1, _ = candidate_images[left_index]
                j, img2, _ = candidate_images[right_index]
                
                # Находим общие точки
                common_points = []
                for p1 in img1.points:
                    if p1.is_placed and p1.point_group_id >= 0:
                        for p2 in img2.points:
                            if p2.is_placed and p2.point_group_id >= 0 and p2.point_group_id == p1.point_group_id:
                                common_points.append(p1.point_group_id)
                                break
                
                logger.info(f"  - Пара изображений {i}-{j}: {len(common_points)} общих точек")
                if len(common_points) >= min_points_for_camera:
                    logger.info(f"  - Общие группы: {common_points}")
                    valid_pairs.append((i, j, len(common_points), common_points))
                    logger.info(f"Найдена валидная пара изображений: {i}-{j}")
        
        if not valid_pairs:
            logger.error(f"Нет пар изображений с достаточным количеством общих точек (минимум {min_points_for_camera})")
            return False
        
        # Сортируем пары по количеству общих точек
        valid_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Инициализируем калибровку
        _report_progress(15.0, "Инициализация калибровки...")
        logger.info("Инициализация калибровки...")
        calibration.init_calibration()
        logger.info("Калибровка инициализирована")
        
        # Оцениваем начальную матрицу камеры
        _report_progress(25.0, "Оценка матрицы камеры...")
        logger.info("Оценка начальной матрицы камеры...")
        K, initial_focal_estimate = estimate_camera_matrix_from_addon(props)
        if K is None:
            logger.error("Не удалось оценить начальную матрицу камеры")
            return False
            
        logger.info(f"Начальная матрица камеры:\n{K}")
        logger.info("Фокусное расстояние будет оптимизировано в процессе калибровки")
        calibration.set_camera_parameters(0, K)  # Используем одну матрицу для всех камер
        if getattr(calibration, 'calibration_data', None) is not None and initial_focal_estimate is not None:
            calibration.calibration_data['initial_focal_estimate'] = initial_focal_estimate
        
        # Конвертируем точки из аддона
        _report_progress(35.0, "Конвертация точек...")
        logger.info("Конвертация точек из аддона...")
        camera_points = convert_addon_points_to_calibration_format(props)
        
        # Проверяем наличие общих точек между камерами
        logger.info("Статистика по общим точкам между изображениями:")
        for i in range(len(props.images) - 1):
            for j in range(i + 1, len(props.images)):
                if i in camera_points and j in camera_points:
                    common_points = set(camera_points[i].keys()) & set(camera_points[j].keys())
                    logger.info(f"  - Пара изображений {i}-{j}: {len(common_points)} общих точек")
                    
                    if len(common_points) >= min_points_for_camera:
                        logger.info(f"    - Общие точки: {sorted(list(common_points))}")
        
        # Проверяем, что существует хотя бы одна валидная стартовая пара.
        # Сам выбор стартовой пары делегируем calibration_core, чтобы Blender-ран
        # использовал ту же bootstrap-логику, что и test_calibration_logic.py.
        bootstrap_pairs = []
        for i, j, count, points in valid_pairs:
            if i in camera_points and j in camera_points:
                common_points = set(camera_points[i].keys()) & set(camera_points[j].keys())
                if len(common_points) >= min_points_for_camera:
                    bootstrap_pairs.append((i, j, len(common_points)))

        if not bootstrap_pairs:
            logger.error("Не удалось найти валидную пару изображений для инициализации")
            return False

        _report_progress(45.0, "Подготовка стартовых пар...")
        logger.info("Стартовая пара не фиксируется в bridge, выбор делегирован calibration_core")
        for i, j, count in bootstrap_pairs[:5]:
            logger.info(f"  - Кандидат стартовой пары: {i}-{j} ({count} общих точек)")
        
        # Добавляем точки в калибровку
        _report_progress(55.0, "Передача точек в solver...")
        logger.info("Добавление точек в калибровку...")
        for camera_id, points in camera_points.items():
            logger.info(f"Камера {camera_id}: {len(points)} точек")
            logger.info(f"  - Точки: {sorted(list(points.keys()))}")
            set_points_from_blender(camera_id, points)
        
        # Адаптируем диапазон фокусного расстояния в зависимости от разрешения изображений
        if len(candidate_images) > 0:
            first_image = bpy.data.images.get(candidate_images[0][1].name)
            if first_image:
                image_width = first_image.size[0]
                
                # Корректируем диапазон в зависимости от разрешения изображения
                # Например, для изображения 800x600 диапазон (800, 3000) подходит
                # Но для изображения 4000x3000 нужен больший диапазон, примерно (4000, 15000)
                width_factor = image_width / 800  # Сравниваем с базовым разрешением 800
                if width_factor > 1.5:  # Значительно больший размер изображения
                    adjusted_range = (
                        int(focal_range[0]),
                        int(focal_range[1] * width_factor)
                    )
                    logger.info(f"Корректируем диапазон фокусного расстояния для изображения {image_width}x{first_image.size[1]}")
                    logger.info(f"Исходный диапазон: {focal_range} -> Скорректированный: {adjusted_range}")
                    focal_range = adjusted_range
        
        logger.info(f"Используемый диапазон фокусного расстояния: {focal_range[0]}-{focal_range[1]} пикселей")
        
        # Запускаем калибровку с оптимизацией фокусного расстояния
        _report_progress(65.0, "Запуск реконструкции...")
        logger.info("Запуск процесса калибровки с оптимизацией фокусного расстояния...")
        start_time = time.time()
        
        success = calibration.run_calibration(
            initial_pair=None,
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
        
        end_time = time.time()
        logger.info(f"Процесс калибровки завершен за {end_time - start_time:.2f} секунд")
        
        if not success:
            logger.error("Не удалось выполнить калибровку")
            return False
        
        # Получаем результаты
        logger.info("Получение результатов калибровки...")
        _report_progress(90.0, "Получение результатов калибровки...")
        camera_poses = get_camera_poses()
        points_3d = _get_visualization_point_cloud()
        
        if not camera_poses or len(camera_poses) < 2:
            logger.error("Не удалось получить положения камер")
            return False
        
        if not points_3d or len(points_3d) < 3:
            logger.error("Недостаточно 3D точек в результате калибровки")
            return False
        
        logger.info(f"Получено {len(camera_poses)} положений камер и {len(points_3d)} 3D точек")
        _log_reconstruction_camera_centers(
            camera_poses,
            calibration.calibration_data.get('images', {}) if calibration.calibration_data else {},
        )
        _log_point_cloud_bounds(points_3d)
        
        # Создаем камеры в сцене Blender
        if create_cameras:
            _report_progress(93.0, "Создание камер в Blender...")
            logger.info("Создание камер в сцене Blender...")
            try:
                create_cameras_from_calibration(props)
                logger.info("Камеры успешно созданы и настроены")
            except Exception as e:
                logger.error(f"Ошибка при создании камер: {str(e)}")
                traceback.print_exc()
                # Не прерываем процесс, продолжаем
        
        # Обновляем информацию о группах точек из 3D точек
        _report_progress(96.0, "Обновление групп точек...")
        logger.info("Обновление информации о группах точек...")
        try:
            update_point_groups_from_calibration(props)
            logger.info("Информация о группах точек обновлена")
        except Exception as e:
            logger.error(f"Ошибка при обновлении информации о группах точек: {str(e)}")
            traceback.print_exc()
            # Не прерываем процесс, продолжаем
        
        # Выводим информацию о фокусных расстояниях камер
        _report_progress(100.0, "Калибровка завершена")
        logger.info("Фокусные расстояния и поля зрения камер:")
        for cam_idx, camera_id in enumerate(camera_poses.keys()):
            if f'K_{camera_id}' in calibration.calibration_data:
                K = calibration.calibration_data[f'K_{camera_id}']
                fx = K[0, 0]
                fy = K[1, 1] if K[1, 1] != 0 else K[0, 0]
                
                logger.info(f"  - Камера {cam_idx} (индивидуальная матрица): fx={fx:.2f}, fy={fy:.2f}")
                
                # Пробуем получить информацию о соответствующем изображении
                if cam_idx < len(props.images):
                    img_name = props.images[cam_idx].name
                    first_image = bpy.data.images.get(props.images[0].name) if len(props.images) > 0 else None
                    if first_image:
                        logger.info(f"  - Камера {cam_idx} ({img_name}): {fx * props.sensor_width / first_image.size[0]:.2f}мм")
                    else:
                        logger.info(f"  - Камера {cam_idx} ({img_name}): фокусное расстояние {fx:.2f} пикселей (не удалось получить размер первого изображения)")
                    
                    # Расчет поля зрения
                    try:
                        img = bpy.data.images.get(img_name)
                        if img:
                            logger.info(f"Размеры изображения {img_name}: {img.size[0]}x{img.size[1]}")
                            fov_h, fov_v = calculate_camera_fov(K, img.size[0], img.size[1])
                            logger.info(f"Расчет FOV для Camera {cam_idx} ({img.size[0]}x{img.size[1]}): гор.={fov_h:.4f}°, верт.={fov_v:.4f}°")
                            
                            logger.info(f"Итоговое поле зрения для камеры {cam_idx}:")
                            logger.info(f"  - Горизонтальное FOV: {fov_h:.4f}°")
                            logger.info(f"  - Вертикальное FOV: {fov_v:.4f}°")
                    except Exception as e:
                        logger.error(f"Ошибка при расчете поля зрения для камеры {cam_idx}: {str(e)}")
        # Возвращаем успех
        logger.info("=" * 80)
        logger.info("КАЛИБРОВКА УСПЕШНО ЗАВЕРШЕНА")
        logger.info("=" * 80)
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении калибровки: {str(e)}")
        logger.error("=" * 80)
        logger.error("КАЛИБРОВКА ПРЕРВАНА ИЗ-ЗА ОШИБКИ")
        logger.error("=" * 80)
        traceback.print_exc()
        return False

def create_cameras_from_calibration(props, camera_name_prefix="Camera_"):
    """
    Создает камеры в Blender на основе результатов калибровки.
    
    Args:
        props: Свойства аддона CAMCALIB_props
        camera_name_prefix: Префикс для имен создаваемых камер
    
    Returns:
        list: Список созданных объектов камер
    """
    if not DEPENDENCIES_INSTALLED:
        logger.error("Невозможно создать камеры: отсутствуют зависимости")
        return []
    
    # Получаем результаты калибровки
    cameras_cv = get_camera_poses()
    points_3d = _get_visualization_point_cloud()
    
    # Получаем словарь изображений из калибровки
    from .calibration import calibration_data
    images_dict = calibration_data.get('images', {}) if calibration_data else {}

    _sync_scene_render_to_images(props, calib_data=calibration_data)
    
    if not cameras_cv:
        logger.error("Нет данных о позах камер")
        return []
    
    # Создаем словарь для хранения объектов камер
    camera_objects = {}
    
    # Для каждой камеры
    for camera_id, (R, t) in cameras_cv.items():
        camera_index = _camera_id_to_index(camera_id)

        # Получаем имя изображения, если оно существует
        image_name = None
        if camera_index is not None and 0 <= camera_index < len(props.images):
            image_name = props.images[camera_index].name
        
        # Имя камеры - используем имя изображения, если доступно
        camera_name = f"{image_name}" if image_name else f"{camera_name_prefix}{camera_id}"
        
        # Создаем объект камеры, если его еще нет
        if camera_name not in bpy.data.objects:
            camera_data = bpy.data.cameras.new(camera_name)
            camera_obj = bpy.data.objects.new(camera_name, camera_data)
            bpy.context.collection.objects.link(camera_obj)
            
            # Настраиваем базовые параметры камеры
            # Фокусное расстояние будет установлено в apply_cameras_to_blender
            camera_data.show_background_images = True  # Показывать фоновые изображения
            camera_data.display_size = 1.0  # Увеличиваем размер отображения камеры
            
            # Настройка CROP для корректного отображения точек
            camera_data.show_passepartout = True
            camera_data.passepartout_alpha = 0.9
            
            # Настройка отображения камеры в 3D виде
            camera_data.show_name = True    # Показывать имя камеры
        else:
            camera_obj = bpy.data.objects[camera_name]
            camera_data = camera_obj.data
            
            # Обновляем параметры существующей камеры
            camera_data.display_size = 1.0
            camera_data.show_name = True
        
        # Загружаем изображение в камеру
        if image_name:
            # Проверяем, есть ли уже фоновое изображение
            if not camera_data.background_images:
                bg = camera_data.background_images.new()
            else:
                bg = camera_data.background_images[0]
                
            # Поиск изображения в Blender
            if image_name in bpy.data.images:
                bg.image = bpy.data.images[image_name]
                bg.alpha = 1.0
                bg.display_depth = 'BACK'  # Отображение позади объектов сцены
                # FIT не режет изображение при несовпадении aspect ratio viewport/render.
                bg.frame_method = 'FIT'
        
        # Сохраняем объект камеры
        if 'camera_objects' not in locals():
            camera_objects = {}
        camera_objects[camera_id] = camera_obj
    
    # Применяем позы камер к созданным объектам
    # Эта функция также установит правильное фокусное расстояние на основе матрицы калибровки
    result = apply_cameras_to_blender(camera_objects, props)
    
    if result:
        logger.info("Камеры успешно созданы и настроены")
        _log_blender_camera_object_positions(camera_objects)
        
        # Выводим информацию о фокусном расстоянии для каждой камеры
        logger.info("Фокусные расстояния и поля зрения камер:")
        
        # Получаем общую матрицу калибровки
        K = None
        if hasattr(calibration, 'calibration_data') and calibration.calibration_data:
            K = calibration.calibration_data.get('K')
        elif 'calibration_data' in globals() and calibration_data:
            K = calibration_data.get('K')
        elif 'calibration_data' in locals() and calibration_data:
            K = calibration_data.get('K')
        else:
            # Пытаемся получить данные калибровки из модуля
            try:
                from .calibration import calibration_data
                if calibration_data:
                    K = calibration_data.get('K')
            except ImportError:
                try:
                    from calibration import calibration_data
                    if calibration_data:
                        K = calibration_data.get('K')
                except ImportError:
                    pass
        
        for camera_id, camera_obj in camera_objects.items():
            camera_index = _camera_id_to_index(camera_id)

            # Проверяем, есть ли индивидуальная матрица калибровки для этой камеры
            individual_K = None
            if hasattr(calibration, 'calibration_data') and calibration.calibration_data:
                individual_K = calibration.calibration_data.get(f'K_{camera_id}')
            elif 'calibration_data' in globals() and calibration_data:
                individual_K = calibration_data.get(f'K_{camera_id}')
            elif 'calibration_data' in locals() and calibration_data:
                individual_K = calibration_data.get(f'K_{camera_id}')
            else:
                # Пытаемся получить данные калибровки из модуля
                try:
                    from .calibration import calibration_data
                    if calibration_data:
                        individual_K = calibration_data.get(f'K_{camera_id}')
                except ImportError:
                    try:
                        from calibration import calibration_data
                        if calibration_data:
                            individual_K = calibration_data.get(f'K_{camera_id}')
                    except ImportError:
                        pass
            
            if individual_K is not None:
                # Используем индивидуальную матрицу
                fx = individual_K[0, 0]
                fy = individual_K[1, 1]
                logger.info(f"  - Камера {camera_id} (индивидуальная матрица): fx={fx:.2f}, fy={fy:.2f}")
            elif K is not None:
                # Используем общую матрицу
                fx = K[0, 0]
                fy = K[1, 1]
                logger.info(f"  - Камера {camera_id} (общая матрица): fx={fx:.2f}, fy={fy:.2f}")
            else:
                logger.warning(f"  - Камера {camera_id}: матрица калибровки не найдена")
                continue  # Пропускаем камеру, если нет матрицы калибровки
            
            # Выводим установленное фокусное расстояние в мм
            logger.info(f"  - Камера {camera_id} ({camera_obj.name}): {camera_obj.data.lens:.2f}мм")
            
            # Рассчитываем и выводим поле зрения
            horizontal_fov, vertical_fov = calculate_fov_from_calibration(props, camera_id)
            if horizontal_fov is not None and vertical_fov is not None:
                logger.info(f"  - Поле зрения: горизонтальное={horizontal_fov:.2f}°, вертикальное={vertical_fov:.2f}°")
                
                # Устанавливаем угол камеры, если свойство доступно
                if hasattr(camera_obj.data, 'angle'):
                    # Определяем, какой угол использовать в зависимости от соотношения сторон
                    image_width = None
                    image_height = None
                    
                    # Получаем размеры изображения из Blender
                    if camera_index is not None and 0 <= camera_index < len(props.images):
                        image_name = props.images[camera_index].name
                        image = bpy.data.images.get(image_name)
                        if image:
                            image_width = image.size[0]
                            image_height = image.size[1]
                    
                    # Если не удалось получить размеры, используем значения по умолчанию
                    if not image_width or not image_height:
                        image_width = 1600
                        image_height = 1200
                    
                    if image_width >= image_height:
                        # Горизонтальное изображение
                        camera_obj.data.angle = np.radians(horizontal_fov)
                        logger.info(f"  - Установлен горизонтальный угол камеры: {horizontal_fov:.2f}°")
                    else:
                        # Вертикальное изображение
                        camera_obj.data.angle = np.radians(vertical_fov)
                        logger.info(f"  - Установлен вертикальный угол камеры: {vertical_fov:.2f}°")
    else:
        logger.error("Не удалось применить позы камер")
    
    # Создаем точки в 3D для визуализации как пустышки (Empty)
    if points_3d:
        # Создаем пустой объект для группировки точек
        if "CalibrationPoints" not in bpy.data.objects:
            empty = bpy.data.objects.new("CalibrationPoints", None)
            bpy.context.collection.objects.link(empty)
        else:
            empty = bpy.data.objects["CalibrationPoints"]
        
        # Создаем точки как пустышки (Empty)
        for point_id, point_3d in points_3d.items():
            point_name = f"Point_{point_id}"
            
            # Ищем существующую точку или создаем новую
            if point_name not in bpy.data.objects:
                # Создаем пустышку (Empty)
                obj = bpy.data.objects.new(point_name, None)
                
                # Устанавливаем тип пустышки Plain Axes для всех точек
                obj.empty_display_type = 'PLAIN_AXES'
                
                # Устанавливаем размер пустышки
                obj.empty_display_size = 0.2
                
                bpy.context.collection.objects.link(obj)
                
                # Делаем дочерним к пустому объекту
                obj.parent = empty
            else:
                obj = bpy.data.objects[point_name]
                # Обновляем тип пустышки на Plain Axes
                obj.empty_display_type = 'PLAIN_AXES'
            
            # Устанавливаем позицию точки
            obj.location = point_3d
    
    return list(camera_objects.values())

def update_point_groups_from_calibration(props):
    """
    Обновляет группы точек в аддоне на основе результатов калибровки.
    
    Args:
        props: Свойства аддона CAMCALIB_props
    
    Returns:
        bool: True, если обновление успешно, иначе False
    """
    if not DEPENDENCIES_INSTALLED:
        logger.error("Невозможно обновить группы точек: отсутствуют зависимости")
        return False
        
    # Получаем результаты калибровки
    points_3d = _get_visualization_point_cloud()
    calib_data = getattr(calibration, 'calibration_data', None)
    final_camera_points = calib_data.get('camera_points', {}) if isinstance(calib_data, dict) else {}
    
    if not points_3d:
        logger.error("Нет данных о 3D точках")
        return False
    
    # Сначала сбрасываем статус калибровки для всех точек
    for img in props.images:
        for point in img.points:
            point.is_calibrated = False
            point.calibration_failed = False
    
    # Обновляем координаты 3D для каждой группы точек
    for point_id, point_3d in points_3d.items():
        # Ищем группу точек с соответствующим ID
        found = False
        for i, group in enumerate(props.point_groups):
            if i == point_id:  # Предполагаем, что индекс группы соответствует ID точки
                # Обновляем координаты 3D
                group.location_3d = point_3d
                found = True
                
                # Помечаем только те 2D-наблюдения, которые реально остались в solver
                for camera_index, img in enumerate(props.images):
                    surviving_ids = {
                        int(point_key)
                        for point_key in final_camera_points.get(str(camera_index), {}).keys()
                    }
                    for point in img.points:
                        if point.point_group_id == i and point.is_placed and i in surviving_ids:
                            point.is_calibrated = True
                break
        
        # Если группа не найдена, создаем новую
        if not found:
            # Добавляем новую группу через оператор add_point_group
            bpy.ops.camera_calibration.add_point_group()
            new_group_idx = props.active_point_group_index
            
            if new_group_idx >= 0 and new_group_idx < len(props.point_groups):
                group = props.point_groups[new_group_idx]
                group.location_3d = point_3d
    
    # Получаем диагностику нереконструированных точек
    _REASON_LABELS = {
        'too_few_views': 'Слишком мало видов (точка видна менее чем на 2 камерах)',
        'pair_triangulation_failed': 'Триангуляция не удалась (малый параллакс или вырожденная геометрия)',
        'full_track_rejected': 'Трек отклонён (высокая ошибка репроекции на всех парах)',
        'strict_subset_only': 'Точка видна с подмножества камер, но strict-режим запрещает частичные треки',
        'subset_only': 'Точка принята только частью камер (конфликт наблюдений)',
        'backfillable_full_track': 'Точка подходит для восстановления, но не была добавлена на этом этапе',
    }
    _CONFLICT_LABELS = {
        'single_view_conflict': 'Конфликт одной камеры — одна камера даёт аномально большую ошибку',
        'multi_view_conflict': 'Конфликт нескольких камер — наблюдения не согласуются между собой',
        'global_tension': 'Глобальное напряжение — точка плохо вписывается в общую реконструкцию',
        'depth_instability': 'Нестабильная глубина — малый параллакс, точка далеко или на оптической оси',
        'insufficient_support': 'Недостаточно поддержки — мало камер видят эту точку',
    }

    try:
        try:
            from .calibration import get_unreconstructed_diagnostics
        except ImportError:
            from calibration import get_unreconstructed_diagnostics
        unreconstructed_diags = get_unreconstructed_diagnostics()
        logger.info(f"Получено {len(unreconstructed_diags)} диагностик нереконструированных точек")
    except Exception as e:
        logger.warning(f"Не удалось получить диагностику нереконструированных точек: {e}")
        unreconstructed_diags = []

    diag_by_point_id = {}
    for diag in unreconstructed_diags:
        pid = diag.get('point_id')
        if pid is not None:
            diag_by_point_id[pid] = diag

    # Проверяем точки, которые должны были быть откалиброваны, но не были
    # Такие точки помечаем как calibration_failed и записываем причину
    for img in props.images:
        for point in img.points:
            if point.is_placed and point.point_group_id >= 0 and not point.is_calibrated:
                point.calibration_failed = True

    # Собираем множество всех point_id с размещёнными наблюдениями
    all_placed_point_ids = set()
    for img in props.images:
        for point in img.points:
            if point.is_placed and point.point_group_id >= 0:
                all_placed_point_ids.add(point.point_group_id)

    # Заполняем rejection_reason для групп точек
    for i, group in enumerate(props.point_groups):
        if i in points_3d:
            group.rejection_reason = ""
            continue

        diag = diag_by_point_id.get(i)
        if diag is None:
            if i not in all_placed_point_ids:
                group.rejection_reason = ""
            else:
                group.rejection_reason = "Точка удалена в процессе калибровки (наблюдения не согласовались с реконструкцией)"
            continue

        reason_code = diag.get('reason', '')
        reason_text = _REASON_LABELS.get(reason_code, f'Причина: {reason_code}')

        parts = [reason_text]

        conflict_class = diag.get('conflict_class')
        if conflict_class and conflict_class in _CONFLICT_LABELS:
            parts.append(_CONFLICT_LABELS[conflict_class])

        best_mean = diag.get('best_mean')
        best_max = diag.get('best_max')
        if best_mean is not None and best_max is not None:
            parts.append(f'Лучший результат: mean={best_mean:.2f}px, max={best_max:.2f}px')

        worst_obs = diag.get('worst_observations', [])
        if worst_obs:
            worst_cams = [f"камера {w.get('camera_id', '?')}={w.get('error', 0):.1f}px" for w in worst_obs[:3]]
            parts.append(f'Проблемные камеры: {", ".join(worst_cams)}')

        track_length = diag.get('track_length', 0)
        parts.append(f'Видов: {track_length}')

        group.rejection_reason = " | ".join(parts)
    
    return True

def visualize_calibration_results(save_path=None):
    """
    Визуализирует результаты калибровки и сохраняет изображение.
    
    Args:
        save_path: Путь для сохранения визуализации (опционально)
    
    Returns:
        bool: True, если визуализация создана успешно, иначе False
    """
    if not DEPENDENCIES_INSTALLED:
        logger.error("Невозможно визуализировать результаты: отсутствуют зависимости")
        return False
        
    try:
        # Путь для сохранения по умолчанию
        if save_path is None:
            save_path = os.path.join(bpy.path.abspath("//"), "calibration_results.png")
        return True
    
    except Exception as e:
        logger.error(f"Ошибка при визуализации: {str(e)}")
        traceback.print_exc()
        return False

def calculate_fov_from_calibration(props, camera_id=None):
    """
    Рассчитывает поле зрения (FOV) камеры на основе результатов калибровки.
    Использует статистический анализ нескольких расчетов для повышения точности.
    
    Args:
        props: Свойства аддона CAMCALIB_props
        camera_id: ID камеры для расчета FOV (если None, используется общая матрица калибровки)
    
    Returns:
        tuple: (horizontal_fov, vertical_fov) - углы в градусах
    """
    if not DEPENDENCIES_INSTALLED:
        logger.error("Невозможно рассчитать FOV: отсутствуют зависимости")
        return None, None
    
    calib_data = getattr(calibration, 'calibration_data', None)
    
    if not calib_data:
        logger.error("Невозможно рассчитать FOV: нет данных калибровки")
        return None, None
    
    # Получаем матрицу калибровки
    if camera_id is not None:
        # Пытаемся получить индивидуальную матрицу для указанной камеры
        camera_matrix = calib_data.get(f'K_{camera_id}')
        if camera_matrix is None:
            # Если индивидуальной матрицы нет, используем общую
            camera_matrix = calib_data.get('K')
            logger.info(f"Для камеры {camera_id} используется общая матрица калибровки")
    else:
        # Используем общую матрицу калибровки
        camera_matrix = calib_data.get('K')
    
    if camera_matrix is None:
        logger.error("Невозможно рассчитать FOV: матрица калибровки не найдена")
        return None, None
    
    # Получаем размеры изображения
    image_width = None
    image_height = None
    image_resolutions = []
    
    try:
        camera_index = _camera_id_to_index(camera_id) if camera_id is not None else None

        # Если указан camera_id, берем размеры соответствующего изображения
        if camera_index is not None and 0 <= camera_index < len(props.images):
            image_item = props.images[camera_index]
            image = bpy.data.images.get(image_item.name)
            if image and image.size[0] > 0 and image.size[1] > 0:
                image_width = image.size[0]
                image_height = image.size[1]
                logger.info(f"Размеры изображения {image_item.name}: {image_width}x{image_height}")
                image_resolutions.append((image_width, image_height, f"Camera {camera_id}"))
        
        # Если camera_id не указан или не удалось получить размеры, собираем размеры всех доступных изображений
        if len(props.images) > 0:
            for i, image_item in enumerate(props.images):
                if camera_index is not None and i == camera_index:
                    continue  # Пропускаем, если уже обработали эту камеру
                    
                image = bpy.data.images.get(image_item.name)
                if image and image.size[0] > 0 and image.size[1] > 0:
                    w, h = image.size[0], image.size[1]
                    # Проверяем, что такого разрешения еще нет в списке
                    if not any(abs(w-r[0])/max(w,r[0])<0.01 and abs(h-r[1])/max(h,r[1])<0.01 for r in image_resolutions):
                        image_resolutions.append((w, h, f"Camera {i}"))
                    # Если основного разрешения еще нет, используем это
                    if not image_width or not image_height:
                        image_width = w
                        image_height = h
                        logger.info(f"Используем размеры изображения {image_item.name}: {image_width}x{image_height}")
    except Exception as e:
        logger.error(f"Ошибка при получении размеров изображения: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Проверяем, получили ли мы размеры изображения
    if not image_width or not image_height:
        # Если не удалось получить из изображений, проверяем свойства аддона
        try:
            if hasattr(props, 'image_width') and hasattr(props, 'image_height'):
                if props.image_width > 0 and props.image_height > 0:
                    image_width = props.image_width
                    image_height = props.image_height
                    logger.info(f"Используем размеры изображения из свойств аддона: {image_width}x{image_height}")
                    image_resolutions.append((image_width, image_height, "Addon properties"))
        except Exception as e:
            logger.error(f"Ошибка при получении размеров из свойств аддона: {str(e)}")
    
    # Если все еще не удалось получить размеры, используем значения по умолчанию
    if not image_width or not image_height:
        # Используем стандартное соотношение сторон 4:3 и типичное разрешение для фотографий
        image_width = 1920
        image_height = 1080
        logger.warning(f"Не удалось определить размер изображения, используются значения по умолчанию: {image_width}x{image_height}")
        image_resolutions.append((image_width, image_height, "Default values"))
    
    # Убеждаемся, что в основном разрешении есть валидные данные
    if image_width <= 0 or image_height <= 0:
        logger.error(f"Некорректные размеры изображения: {image_width}x{image_height}")
        image_width = 1920
        image_height = 1080
        logger.warning(f"Принудительно установлены стандартные размеры: {image_width}x{image_height}")
    
    # Лимитируем количество разрешений для обработки (максимум 5)
    if len(image_resolutions) > 5:
        logger.info(f"Большое количество разрешений ({len(image_resolutions)}), ограничиваем до 5")
        image_resolutions = image_resolutions[:5]
    
    # Подготовка для многократного расчета FOV для повышения точности
    fov_results = []
    
    # Выполняем расчеты для всех собранных разрешений
    import numpy as np
    
    for res_width, res_height, res_source in image_resolutions:
        # Для первого (основного) разрешения используем оригинальную матрицу
        if res_width == image_width and res_height == image_height:
            h_fov, v_fov = calculate_camera_fov(camera_matrix, res_width, res_height)
            if h_fov is not None and v_fov is not None:
                logger.info(f"Расчет FOV для {res_source} ({res_width}x{res_height}): гор.={h_fov:.4f}°, верт.={v_fov:.4f}°")
                fov_results.append((h_fov, v_fov, res_source))
        else:
            # Для других разрешений масштабируем матрицу калибровки
            try:
                scale_x = res_width / image_width
                scale_y = res_height / image_height
                
                # Создаем копию матрицы и масштабируем ее
                adj_matrix = np.array(camera_matrix, copy=True)
                adj_matrix[0, 0] *= scale_x  # fx
                adj_matrix[1, 1] *= scale_y  # fy
                adj_matrix[0, 2] *= scale_x  # cx
                adj_matrix[1, 2] *= scale_y  # cy
                
                # Рассчитываем FOV с адаптированной матрицей
                h_fov, v_fov = calculate_camera_fov(adj_matrix, res_width, res_height)
                
                if h_fov is not None and v_fov is not None:
                    logger.info(f"Расчет FOV для {res_source} ({res_width}x{res_height}): гор.={h_fov:.4f}°, верт.={v_fov:.4f}°")
                    fov_results.append((h_fov, v_fov, res_source))
            except Exception as e:
                logger.error(f"Ошибка при расчете FOV для {res_source} ({res_width}x{res_height}): {str(e)}")
    
    # Если нет результатов, пытаемся выполнить хотя бы один расчет с основным разрешением
    if not fov_results:
        logger.warning("Не удалось выполнить ни одного успешного расчета FOV, пробуем снова с базовым разрешением")
        h_fov, v_fov = calculate_camera_fov(camera_matrix, image_width, image_height)
        if h_fov is not None and v_fov is not None:
            logger.info(f"Базовый расчет FOV ({image_width}x{image_height}): гор.={h_fov:.4f}°, верт.={v_fov:.4f}°")
            fov_results.append((h_fov, v_fov, "Base calculation"))
    
    # Если все еще нет результатов, возвращаем None
    if not fov_results:
        logger.error("Не удалось рассчитать FOV ни для одного разрешения")
        return None, None
    
    # Статистический анализ результатов
    if len(fov_results) > 1:
        # Разделяем результаты на горизонтальные и вертикальные FOV
        h_fovs = [r[0] for r in fov_results]
        v_fovs = [r[1] for r in fov_results]
        
        # Вычисляем базовую статистику
        h_mean = np.mean(h_fovs)
        h_median = np.median(h_fovs)
        h_std = np.std(h_fovs)
        h_min, h_max = min(h_fovs), max(h_fovs)
        
        v_mean = np.mean(v_fovs)
        v_median = np.median(v_fovs)
        v_std = np.std(v_fovs)
        v_min, v_max = min(v_fovs), max(v_fovs)
        
        # Выводим детальную статистику
        logger.info("Статистика расчета FOV:")
        logger.info(f"Горизонтальное FOV: среднее={h_mean:.4f}°, медиана={h_median:.4f}°, стд.откл.={h_std:.4f}°, диапазон=[{h_min:.4f}°, {h_max:.4f}°]")
        logger.info(f"Вертикальное FOV: среднее={v_mean:.4f}°, медиана={v_median:.4f}°, стд.откл.={v_std:.4f}°, диапазон=[{v_min:.4f}°, {v_max:.4f}°]")
        
        # Проверяем выбросы (отклонения более 3 стандартных отклонений от среднего)
        h_outliers = [(h, src) for h, v, src in fov_results if abs(h - h_mean) > 3 * h_std]
        v_outliers = [(v, src) for h, v, src in fov_results if abs(v - v_mean) > 3 * v_std]
        
        if h_outliers:
            logger.warning(f"Обнаружены выбросы в горизонтальном FOV: {h_outliers}")
        
        if v_outliers:
            logger.warning(f"Обнаружены выбросы в вертикальном FOV: {v_outliers}")
        
        # Используем медиану для устойчивости к выбросам, если есть большой разброс данных
        # или если обнаружены выбросы
        if h_std / h_mean > 0.05 or h_outliers:
            logger.info(f"Используем медиану для горизонтального FOV из-за значительного разброса или выбросов")
            horizontal_fov = h_median
        else:
            logger.info(f"Используем среднее значение для горизонтального FOV")
            horizontal_fov = h_mean
            
        if v_std / v_mean > 0.05 or v_outliers:
            logger.info(f"Используем медиану для вертикального FOV из-за значительного разброса или выбросов")
            vertical_fov = v_median
        else:
            logger.info(f"Используем среднее значение для вертикального FOV")
            vertical_fov = v_mean
    else:
        # Если только один результат, используем его напрямую
        horizontal_fov, vertical_fov, _ = fov_results[0]
    
    # Проверка значений на физическую реалистичность
    if horizontal_fov < 1.0 or horizontal_fov > 180.0:
        logger.warning(f"Нереалистичное значение горизонтального FOV: {horizontal_fov:.4f}°")
        horizontal_fov = max(1.0, min(horizontal_fov, 180.0))
        logger.info(f"Ограничиваем горизонтальное FOV до {horizontal_fov:.4f}°")
    
    if vertical_fov < 1.0 or vertical_fov > 180.0:
        logger.warning(f"Нереалистичное значение вертикального FOV: {vertical_fov:.4f}°")
        vertical_fov = max(1.0, min(vertical_fov, 180.0))
        logger.info(f"Ограничиваем вертикальное FOV до {vertical_fov:.4f}°")
    
    # Проверяем соотношение сторон FOV с соотношением сторон изображения
    image_aspect = image_width / image_height
    fov_aspect = horizontal_fov / vertical_fov
    aspect_diff = abs(image_aspect - fov_aspect) / image_aspect
    
    if aspect_diff > 0.15:  # Различие более 15%
        logger.warning(f"Значительное расхождение между соотношением сторон изображения ({image_aspect:.4f}) " 
                      f"и соотношением FOV ({fov_aspect:.4f}): {aspect_diff:.2%}")
        
        # Если разница очень большая, можно скорректировать один из FOV для получения правильного соотношения
        if aspect_diff > 0.3:  # Разница более 30%
            logger.warning(f"Критическое расхождение в соотношении сторон, выполняем корректировку")
            # Корректируем вертикальное FOV на основе горизонтального и соотношения сторон изображения
            corrected_v_fov = horizontal_fov / image_aspect
            
            # Проверяем, что корректировка не приводит к нереалистичным значениям
            if 1.0 <= corrected_v_fov <= 180.0:
                logger.info(f"Корректируем вертикальное FOV: {vertical_fov:.4f}° -> {corrected_v_fov:.4f}°")
                vertical_fov = corrected_v_fov
    
    # Округляем результаты до 4 знаков после запятой для стабильности
    horizontal_fov = round(horizontal_fov, 4)
    vertical_fov = round(vertical_fov, 4)
    
    logger.info(f"Итоговое поле зрения для {'камеры '+str(camera_id) if camera_id is not None else 'общей матрицы'}:")
    logger.info(f"  - Горизонтальное FOV: {horizontal_fov:.4f}°")
    logger.info(f"  - Вертикальное FOV: {vertical_fov:.4f}°")
    
    # Если использовалась общая матрица, сохраняем результаты в свойствах аддона
    if camera_id is None:
        try:
            props.horizontal_fov = horizontal_fov
            props.vertical_fov = vertical_fov
            logger.info(f"FOV сохранено в свойствах аддона")
        except Exception as e:
            logger.error(f"Ошибка при сохранении FOV в свойствах аддона: {str(e)}")
    
    return horizontal_fov, vertical_fov

# Добавляем функции для работы с двухсторонней калибровкой
def has_calibration_data():
    """
    Проверяет наличие данных калибровки.
    
    Returns:
        bool: True, если данные калибровки существуют
    """
    from calibration import calibration_data
    
    return calibration_data is not None and 'cameras' in calibration_data and len(calibration_data['cameras']) > 0

def get_calibration_data():
    """
    Возвращает данные калибровки.
    
    Returns:
        dict: Данные калибровки
    """
    from calibration import calibration_data
    
    return calibration_data

def update_calibration_data(new_calibration_data):
    """
    Обновляет данные калибровки.
    
    Args:
        new_calibration_data: Новые данные калибровки
    """
    # Обновляем данные в модуле calibration
    import calibration
    calibration.calibration_data = new_calibration_data
    
    logger.info("Данные калибровки обновлены")

def create_mirrored_cameras_in_blender(props, mirror_props, calib_data, camera_name_prefix="MirrorCamera_"):
    """
    Создает зеркальные камеры в сцене Blender.
    
    Args:
        props: Свойства аддона CAMCALIB_props
        mirror_props: Свойства зеркальной калибровки MIRROR_CALIBRATION_props
        calib_data: Данные калибровки
        camera_name_prefix: Префикс для имени камеры
    
    Returns:
        bool: True, если камеры успешно созданы
    """
    if not DEPENDENCIES_INSTALLED:
        logger.error("Невозможно создать зеркальные камеры: отсутствуют зависимости")
        return False
    
    try:
        import numpy as np
        import mathutils
    except ImportError as e:
        logger.error(f"Ошибка импорта модулей: {str(e)}")
        return False
    
    # Проверяем наличие данных зеркальной калибровки
    if 'mirror_data' not in calib_data or 'mirrored_cameras' not in calib_data['mirror_data']:
        logger.error("Отсутствуют данные зеркальной калибровки")
        return False
    
    # Получаем список зеркальных камер
    mirrored_cameras = calib_data['mirror_data']['mirrored_cameras']
    logger.info(f"Количество зеркальных камер: {len(mirrored_cameras)}")
    
    # Удаляем существующие объекты зеркальных камер
    for obj in bpy.data.objects:
        if obj.name.startswith(camera_name_prefix):
            bpy.data.objects.remove(obj)
            logger.info(f"Удален объект камеры: {obj.name}")
    
    # Для каждой зеркальной камеры создаем объект в Blender
    for camera_id in mirrored_cameras:
        if camera_id in calib_data['cameras']:
            R, t = calib_data['cameras'][camera_id]
            
            # Получаем параметры камеры
            if isinstance(camera_id, str) and camera_id.startswith("mirror_") and camera_id[7:].isdigit():
                original_id = int(camera_id[7:])
                
                # Ищем оригинальное изображение
                original_image = None
                for img in props.images:
                    if img.id == original_id:
                        original_image = img
                        break
                
                # Создаем имя для камеры
                if original_image:
                    camera_name = f"{camera_name_prefix}{original_image.name}"
                else:
                    camera_name = f"{camera_name_prefix}{camera_id}"
            else:
                camera_name = f"{camera_name_prefix}{camera_id}"
            
            # Преобразование из OpenCV в Blender
            # В OpenCV ось Z направлена вперед, в Blender - назад
            R_flip = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ], dtype=np.float32)

            R = np.asarray(R, dtype=np.float32).reshape(3, 3)
            t = np.asarray(t, dtype=np.float32).reshape(3, 1)
            R_blender = R.T @ R_flip
            C = (-R.T @ t).reshape(3)
            
            # Создаем объект камеры в Blender
            camera_data = bpy.data.cameras.new(f"{camera_id}_data")
            camera_obj = bpy.data.objects.new(camera_name, camera_data)
            
            # Добавляем камеру в сцену
            bpy.context.collection.objects.link(camera_obj)
            
            # Устанавливаем позицию и ориентацию
            # Создаем матрицу 4x4
            matrix = mathutils.Matrix.Identity(4)
            
            # Устанавливаем ротацию (верхняя левая часть 3x3)
            for i in range(3):
                for j in range(3):
                    matrix[i][j] = float(R_blender[i, j])
            
            # Устанавливаем позицию (верхняя правая часть 3x1)
            matrix[0][3] = float(C[0])
            matrix[1][3] = float(C[1])
            matrix[2][3] = float(C[2])
            
            camera_obj.matrix_world = matrix
            
            # Настраиваем отображение камеры
            camera_obj.data.display_size = 0.5
            camera_obj.data.show_limits = True
            camera_obj.data.clip_start = 0.1
            camera_obj.data.clip_end = 100.0
            
            # Настраиваем фокусное расстояние камеры
            if calib_data['K'] is not None:
                # Фокусное расстояние в пикселях
                focal_length_pixels = calib_data['K'][0, 0]
                
                # Находим соответствующее изображение
                for img in props.images:
                    if isinstance(camera_id, str) and camera_id.startswith("mirror_") and camera_id[7:].isdigit():
                        if img.id == int(camera_id[7:]):
                            image = bpy.data.images.get(img.name)
                            if image:
                                # Ширина сенсора
                                sensor_width = props.sensor_width
                                
                                # Ширина изображения
                                width = image.size[0]
                                
                                # Фокусное расстояние в миллиметрах
                                focal_mm = focal_length_pixels * sensor_width / width
                                
                                # Устанавливаем фокусное расстояние камеры
                                camera_obj.data.lens = focal_mm
                                break
            
            logger.info(f"Создана зеркальная камера: {camera_name}")
    logger.info("Зеркальные камеры успешно созданы в Blender")
    return True

def create_symmetry_plane_in_blender(mirror_props, calib_data):
    """
    Создает плоскость симметрии в сцене Blender.
    
    Args:
        mirror_props: Свойства зеркальной калибровки MIRROR_CALIBRATION_props
        calib_data: Данные калибровки
    
    Returns:
        bool: True, если плоскость успешно создана
    """
    if not DEPENDENCIES_INSTALLED:
        logger.error("Невозможно создать плоскость симметрии: отсутствуют зависимости")
        return False
    
    # Проверяем наличие данных о плоскости симметрии
    if 'mirror_data' not in calib_data or 'plane_visualization' not in calib_data['mirror_data']:
        logger.error("Отсутствуют данные о плоскости симметрии")
        return False
    
    # Получаем данные о плоскости
    plane_data = calib_data['mirror_data']['plane_visualization']
    
    # Удаляем существующий объект плоскости, если он есть
    if "SymmetryPlane" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["SymmetryPlane"])
        logger.info("Удален существующий объект плоскости симметрии")
    
    # Создаем новую сетку
    mesh = bpy.data.meshes.new("SymmetryPlaneMesh")
    
    # Создаем вершины и грани
    vertices = [tuple(v) for v in plane_data['vertices']]
    faces = [(0, 1, 2, 3)]
    
    # Заполняем сетку данными
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    
    # Создаем объект
    obj = bpy.data.objects.new("SymmetryPlane", mesh)
    
    # Добавляем объект в сцену
    bpy.context.collection.objects.link(obj)
    
    # Делаем объект полупрозрачным
    obj.display_type = 'WIRE' if mirror_props.show_symmetry_plane else 'BOUNDS'
    
    # Создаем материал для плоскости
    if "SymmetryPlaneMaterial" not in bpy.data.materials:
        mat = bpy.data.materials.new("SymmetryPlaneMaterial")
        mat.use_nodes = True
        
        # Очищаем существующие узлы
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        
        # Создаем узлы
        output_node = nodes.new(type="ShaderNodeOutputMaterial")
        principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
        
        # Устанавливаем параметры
        principled_node.inputs["Base Color"].default_value = (0.2, 0.8, 0.2, 0.5)  # Зеленый, полупрозрачный
        principled_node.inputs["Alpha"].default_value = 0.3
        
        # Соединяем узлы
        links.new(principled_node.outputs["BSDF"], output_node.inputs["Surface"])
        
        # Настраиваем прозрачность
        mat.blend_method = 'BLEND'
        mat.shadow_method = 'NONE'
    else:
        mat = bpy.data.materials["SymmetryPlaneMaterial"]
    
    # Назначаем материал
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat
    
    logger.info("Плоскость симметрии успешно создана в Blender")
    return True

def toggle_symmetry_plane_visibility(mirror_props):
    """
    Переключает видимость плоскости симметрии.
    
    Args:
        mirror_props: Свойства зеркальной калибровки MIRROR_CALIBRATION_props
    
    Returns:
        bool: True, если операция успешна
    """
    if "SymmetryPlane" in bpy.data.objects:
        obj = bpy.data.objects["SymmetryPlane"]
        obj.display_type = 'WIRE' if mirror_props.show_symmetry_plane else 'BOUNDS'
        obj.hide_viewport = not mirror_props.show_symmetry_plane
        logger.info(f"Видимость плоскости симметрии изменена на: {mirror_props.show_symmetry_plane}")
        return True
    else:
        logger.warning("Объект плоскости симметрии не найден")
        return False

def toggle_cameras_visibility(mirror_props):
    """
    Переключает видимость камер.
    
    Args:
        mirror_props: Свойства зеркальной калибровки MIRROR_CALIBRATION_props
    
    Returns:
        bool: True, если операция успешна
    """
    # Переключаем видимость оригинальных камер
    for obj in bpy.data.objects:
        if obj.name.startswith("Camera_") and not obj.name.startswith("MirrorCamera_"):
            obj.hide_viewport = not mirror_props.show_original_cameras
    
    # Переключаем видимость зеркальных камер
    for obj in bpy.data.objects:
        if obj.name.startswith("MirrorCamera_"):
            obj.hide_viewport = not mirror_props.show_mirrored_cameras
    
    logger.info(f"Видимость камер изменена: оригинальные - {mirror_props.show_original_cameras}, зеркальные - {mirror_props.show_mirrored_cameras}")
    return True
