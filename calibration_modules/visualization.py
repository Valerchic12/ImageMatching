"""
Функции для визуализации 3D точек, камер и других результатов калибровки.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union

def plot_cameras_and_points(points_3d, cameras, ax=None, camera_size=0.1, point_size=5):
    """
    Визуализирует 3D точки и камеры.

    Args:
        points_3d: Словарь 3D точек {id: point}
        cameras: Словарь камер {id: (R, t)}
        ax: Существующий объект matplotlib Axes3D (опционально)
        camera_size: Размер визуализации камеры
        point_size: Размер точек
    
    Returns:
        matplotlib.axes.Axes: Объект осей matplotlib
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Построение точек
    points = np.array(list(points_3d.values()))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=point_size, c='blue', marker='o')
    
    # Построение камер
    for cam_id, (R, t) in cameras.items():
        # Определяем оси координат камеры
        axes = camera_size * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        # Преобразуем оси из координат камеры в мировые координаты
        axes_world = []
        for axis in axes:
            # Для точки начала координат (0, 0, 0) в системе координат камеры
            if np.all(axis == 0):
                axes_world.append(-R.T @ t)
            else:
                # Для осей X, Y, Z
                axes_world.append((-R.T @ t) + (R.T @ axis))
        
        # Отрисовка позиции камеры
        cam_pos = axes_world[0]
        ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], s=100, c='red', marker='x')
        
        # Отрисовка осей камеры
        colors = ['r', 'g', 'b']
        for i, color in enumerate(colors):
            ax.plot([cam_pos[0], axes_world[i+1][0]], 
                    [cam_pos[1], axes_world[i+1][1]], 
                    [cam_pos[2], axes_world[i+1][2]], 
                    color=color, linewidth=2)
        
        # Добавляем надпись с ID камеры
        ax.text(cam_pos[0], cam_pos[1], cam_pos[2], f' {cam_id}', fontsize=10)
    
    # Настройка осей и меток
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Reconstruction')
    
    # Вычисление масштаба графика
    if len(points) > 0:
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
        
        # Находим самое большое расстояние для создания куба
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        
        # Находим центр точек
        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2
        z_mid = (z_max + z_min) / 2
        
        # Устанавливаем диапазон осей
        ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
        ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
        ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
    
    return ax

def draw_camera_positions_in_2d(cameras, ax=None, size=0.5, color_by_id=True):
    """
    Отрисовывает позиции камер в 2D (вид сверху).
    
    Args:
        cameras: Словарь камер {id: (R, t)}
        ax: Существующий объект matplotlib Axes (опционально)
        size: Размер маркера для отрисовки камеры
        color_by_id: Если True, цвет камеры зависит от её ID
    
    Returns:
        matplotlib.axes.Axes: Объект осей matplotlib
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Отрисовка позиций камер
    positions = []
    for cam_id, (R, t) in cameras.items():
        # Вычисляем позицию камеры
        cam_pos = -R.T @ t
        positions.append(cam_pos)
        
        # Определяем направление камеры (оптической оси)
        cam_direction = R.T @ np.array([0, 0, 1])
        
        # Отрисовываем позицию камеры
        if color_by_id:
            color = plt.cm.tab10(cam_id % 10)
        else:
            color = 'red'
        
        ax.scatter(cam_pos[0], cam_pos[2], c=[color], s=100, marker='o', label=f'Camera {cam_id}')
        
        # Отрисовываем направление камеры (проекция оптической оси)
        ax.arrow(cam_pos[0], cam_pos[2], 
                 cam_direction[0] * size, cam_direction[2] * size, 
                 head_width=0.05, head_length=0.1, fc=color, ec=color)
        
        # Добавляем ID камеры
        ax.text(cam_pos[0], cam_pos[2], f' {cam_id}', fontsize=10)
    
    # Добавляем сетку
    ax.grid(True)
    
    # Настройка заголовка и меток
    ax.set_title('Позиции камер (вид сверху)')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    
    # Настраиваем масштаб осей, чтобы соотношение было 1:1
    ax.set_aspect('equal')
    
    # Если у нас есть хотя бы две камеры, настраиваем оси так, чтобы все камеры были видны
    if len(positions) >= 2:
        positions = np.array(positions)
        x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
        z_min, z_max = np.min(positions[:, 2]), np.max(positions[:, 2])
        
        # Добавляем небольшой отступ
        x_range = x_max - x_min
        z_range = z_max - z_min
        max_range = max(x_range, z_range)
        
        # Если диапазон слишком мал, устанавливаем минимальный диапазон
        if max_range < 1:
            max_range = 1
        
        # Находим центры осей
        x_mid = (x_max + x_min) / 2
        z_mid = (z_max + z_min) / 2
        
        # Устанавливаем диапазон осей
        ax.set_xlim(x_mid - max_range * 0.6, x_mid + max_range * 0.6)
        ax.set_ylim(z_mid - max_range * 0.6, z_mid + max_range * 0.6)
    
    return ax

def draw_points_on_image(image, points, colors=None, radius=3, thickness=2):
    """
    Отрисовывает точки на изображении.
    
    Args:
        image: Исходное изображение
        points: Список точек в формате [(x, y), ...]
        colors: Список цветов для каждой точки или один цвет для всех точек
        radius: Радиус окружности
        thickness: Толщина линии окружности
    
    Returns:
        np.ndarray: Изображение с отрисованными точками
    """
    # Создаем копию изображения
    img_copy = image.copy()
    
    # Преобразуем точки в список
    if not isinstance(points, list):
        points = points.tolist() if hasattr(points, 'tolist') else list(points)
    
    # Настраиваем цвета
    if colors is None:
        colors = (0, 255, 0)  # Зеленый цвет по умолчанию
    
    # Если передан один цвет, применяем его ко всем точкам
    if not isinstance(colors, list):
        colors = [colors] * len(points)
    
    # Отрисовываем точки
    for i, point in enumerate(points):
        # Преобразуем координаты в целые числа
        x, y = int(round(point[0])), int(round(point[1]))
        
        # Проверяем, что точка находится в пределах изображения
        if 0 <= x < img_copy.shape[1] and 0 <= y < img_copy.shape[0]:
            cv2.circle(img_copy, (x, y), radius, colors[i], thickness)
    
    return img_copy

def draw_reprojected_points(image, points_3d, camera_matrix, dist_coeffs, R, t, colors=None):
    """
    Отрисовывает проекции 3D точек на изображении.
    
    Args:
        image: Исходное изображение
        points_3d: Словарь 3D точек {id: point}
        camera_matrix: Матрица внутренних параметров камеры (K)
        dist_coeffs: Коэффициенты дисторсии
        R: Матрица поворота камеры
        t: Вектор перемещения камеры
        colors: Цвета для отрисовки точек
    
    Returns:
        np.ndarray: Изображение с отрисованными проекциями точек
    """
    # Создаем копию изображения
    img_copy = image.copy()
    
    # Преобразуем точки в массив numpy
    points_array = np.array(list(points_3d.values()), dtype=np.float32)
    
    # Вычисляем проекции точек
    rvec, _ = cv2.Rodrigues(R)
    projected_points, _ = cv2.projectPoints(points_array, rvec, t, camera_matrix, dist_coeffs)
    
    # Отрисовываем проекции
    projected_points = projected_points.reshape(-1, 2)
    img_copy = draw_points_on_image(img_copy, projected_points, colors)
    
    return img_copy

def compare_point_observations(image, observed_points, projected_points, colors=None):
    """
    Сравнивает наблюдаемые и проецируемые точки на изображении.
    
    Args:
        image: Исходное изображение
        observed_points: Список наблюдаемых 2D точек
        projected_points: Список проецируемых 2D точек
        colors: Цвета для отрисовки (опционально)
    
    Returns:
        np.ndarray: Изображение с отрисованными точками и линиями
    """
    # Создаем копию изображения
    img_copy = image.copy()
    
    # Преобразуем точки в массивы numpy, если они не являются таковыми
    if not isinstance(observed_points, np.ndarray):
        observed_points = np.array(observed_points)
    
    if not isinstance(projected_points, np.ndarray):
        projected_points = np.array(projected_points)
    
    # Настраиваем цвета
    if colors is None:
        observed_color = (0, 255, 0)  # Зеленый для наблюдаемых точек
        projected_color = (255, 0, 0)  # Красный для проецируемых точек
        line_color = (0, 0, 255)       # Синий для линий
    else:
        observed_color, projected_color, line_color = colors
    
    # Отрисовываем наблюдаемые точки (зеленым)
    img_copy = draw_points_on_image(img_copy, observed_points, observed_color)
    
    # Отрисовываем проецируемые точки (красным)
    img_copy = draw_points_on_image(img_copy, projected_points, projected_color)
    
    # Отрисовываем линии между соответствующими точками
    for i in range(len(observed_points)):
        p1 = (int(round(observed_points[i][0])), int(round(observed_points[i][1])))
        p2 = (int(round(projected_points[i][0])), int(round(projected_points[i][1])))
        
        # Проверяем, что точки находятся в пределах изображения
        if (0 <= p1[0] < img_copy.shape[1] and 0 <= p1[1] < img_copy.shape[0] and
            0 <= p2[0] < img_copy.shape[1] and 0 <= p2[1] < img_copy.shape[0]):
            cv2.line(img_copy, p1, p2, line_color, 1)
    
    return img_copy

def plot_reprojection_errors(errors, camera_ids=None):
    """
    Строит график ошибок репроекции для каждой камеры.
    
    Args:
        errors: Словарь ошибок репроекции по камерам {camera_id: [errors]}
        camera_ids: Список ID камер для построения графика (опционально)
    
    Returns:
        matplotlib.figure.Figure: Объект фигуры matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Если ID камер не указан, используем все камеры
    if camera_ids is None:
        camera_ids = sorted(errors.keys())
    
    # Данные для построения
    data = []
    labels = []
    
    for cam_id in camera_ids:
        if cam_id in errors and len(errors[cam_id]) > 0:
            data.append(errors[cam_id])
            labels.append(f'Camera {cam_id}')
    
    # Строим боксплот
    if data:
        ax.boxplot(data, labels=labels)
        ax.set_title('Ошибки репроекции по камерам')
        ax.set_ylabel('Ошибка репроекции (пиксели)')
        ax.grid(True, linestyle='--', alpha=0.7)
    else:
        ax.text(0.5, 0.5, 'Нет данных для отображения', 
                ha='center', va='center', fontsize=12)
    
    return fig 

def generate_error_heatmap(points_2d, errors, image_size=(1920, 1080), grid_size=(20, 20)):
    """
    Создает тепловую карту (heatmap) ошибок репроекции по полю зрения камеры.
    Помогает выявлять проблемные зоны линзы (например, края) после калибровки.
    
    Args:
        points_2d: np.ndarray формы (N, 2) с координатами точек на изображении.
        errors: np.ndarray формы (N,) с величинами ошибок (например, L2 норма).
        image_size: Разрешение изображения (ширина, высота).
        grid_size: Количество ячеек сетки (по горизонтали, по вертикали).
        
    Returns:
        matplotlib.figure.Figure: Объект фигуры matplotlib с тепловой картой.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    width, height = image_size
    bins_x = grid_size[0]
    bins_y = grid_size[1]
    
    # Создаем сетку
    error_sum = np.zeros((bins_y, bins_x), dtype=np.float64)
    count = np.zeros((bins_y, bins_x), dtype=np.int32)
    
    # Распределяем по корзинам
    points_2d = np.asarray(points_2d)
    errors = np.asarray(errors)
    
    for i in range(len(points_2d)):
        x, y = points_2d[i]
        if 0 <= x < width and 0 <= y < height:
            bx = int((x / width) * bins_x)
            by = int((y / height) * bins_y)
            bx = min(max(bx, 0), bins_x - 1)
            by = min(max(by, 0), bins_y - 1)
            error_sum[by, bx] += errors[i]
            count[by, bx] += 1
            
    # Вычисляем среднюю ошибку в каждой ячейке
    heatmap = np.zeros_like(error_sum)
    mask = count > 0
    heatmap[mask] = error_sum[mask] / count[mask]
    
    # Отрисовываем
    im = ax.imshow(heatmap, cmap='hot', interpolation='nearest', 
                   extent=[0, width, height, 0], aspect='auto', vmin=0)
    plt.colorbar(im, ax=ax, label='Средняя ошибка репроекции (пиксели)')
    
    ax.set_title('Тепловая карта ошибок репроекции (Heatmap)')
    ax.set_xlabel('X (пиксели)')
    ax.set_ylabel('Y (пиксели)')
    
    # Добавляем полупрозрачные точки поверх для контекста
    ax.scatter(points_2d[:, 0], points_2d[:, 1], c='white', s=5, alpha=0.3)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    
    fig.tight_layout()
    return fig