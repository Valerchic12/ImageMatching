"""
Модуль для работы с кривыми Безье в редакторе изображений
"""

import bpy
import numpy as np
from mathutils import Vector
import logging
import gpu
from gpu_extras.batch import batch_for_shader
import gpu.state

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BezierPoint:
    """Класс для представления точки на кривой Безье"""
    def __init__(self, position=Vector((0, 0)), is_handle=False, handle_type='CONTROL'):
        """
        Инициализация точки Безье
        
        Args:
            position (Vector): Позиция точки в 2D пространстве изображения
            is_handle (bool): Является ли точка хендлером (контрольной точкой)
            handle_type (str): Тип хендлера ('CONTROL', 'START', 'END')
        """
        self.position = position
        self.is_handle = is_handle
        self.handle_type = handle_type
        self.selected = False
        self.hovered = False
    
    def set_position(self, position):
        """Установка новой позиции точки"""
        self.position = position
    
    def get_position(self):
        """Получение позиции точки"""
        return self.position


class BezierCurve:
    """Класс для работы с кривой Безье"""
    def __init__(self):
        """Инициализация кривой Безье"""
        self.control_points = []  # Контрольные точки (хендлеры)
        self.curve_points = []    # Точки на кривой для калибровки
        self.num_handles = 3      # Количество контрольных точек по умолчанию (кроме начальной и конечной)
        self.num_points = 5      # Количество точек на кривой по умолчанию
        self.is_complete = False  # Флаг завершенности кривой (установлены начальная и конечная точки)
        self.preview_enabled = True  # Флаг предварительного просмотра
    
    def set_start_point(self, position):
        """Установка начальной точки кривой"""
        if not self.control_points:
            start_point = BezierPoint(position, is_handle=True, handle_type='START')
            self.control_points.append(start_point)
            logger.info(f"Установлена начальная точка кривой: {position}")
            return True
        return False
    
    def set_end_point(self, position):
        """Установка конечной точки кривой и генерация промежуточных контрольных точек"""
        if len(self.control_points) == 1:
            end_point = BezierPoint(position, is_handle=True, handle_type='END')
            self.control_points.append(end_point)
            self.is_complete = True
            self._generate_control_points()
            self._generate_curve_points()
            logger.info(f"Установлена конечная точка кривой: {position}")
            return True
        return False
    
    def _generate_control_points(self):
        """Генерация промежуточных контрольных точек кривой"""
        if len(self.control_points) < 2:
            return
        
        start_pos = self.control_points[0].get_position()
        end_pos = self.control_points[-1].get_position()
        
        # Удаляем существующие промежуточные контрольные точки
        self.control_points = [self.control_points[0], self.control_points[-1]]
        
        # Вычисляем направление и длину линии от начала до конца
        line_vec = end_pos - start_pos
        line_length = line_vec.length
        
        # Вычисляем нормаль к линии для создания естественной кривизны
        # используем вектор, перпендикулярный направлению линии
        if abs(line_vec.y) > abs(line_vec.x):
            # Если линия более вертикальная
            normal = Vector((-line_vec.y, line_vec.x)).normalized()
        else:
            # Если линия более горизонтальная
            normal = Vector((line_vec.y, -line_vec.x)).normalized()
        
        # Генерируем равномерно распределенные контрольные точки между началом и концом
        for i in range(1, self.num_handles + 1):
            t = i / (self.num_handles + 1)
            
            # Базовая позиция на прямой линии
            base_pos = start_pos.lerp(end_pos, t)
            
            # Рассчитываем смещение для создания параболической формы
            # Максимальный изгиб в середине (при t = 0.5)
            offset_factor = 4 * t * (1 - t)  # Параболическая функция, максимум при t=0.5
            
            # Масштабируем смещение в зависимости от длины линии
            offset_scale = line_length * 0.2  # 20% от длины линии
            
            # Добавляем смещение в направлении нормали
            offset = normal * offset_scale * offset_factor
            
            control_point = BezierPoint(base_pos + offset, is_handle=True, handle_type='CONTROL')
            # Вставляем перед конечной точкой
            self.control_points.insert(-1, control_point)
        
        logger.info(f"Сгенерировано {self.num_handles} промежуточных контрольных точек")
    
    def _generate_curve_points(self):
        """Генерация точек на кривой Безье для калибровки с более равномерным распределением"""
        if len(self.control_points) < 2:
            return
        
        self.curve_points = []
        
        # Получаем массив контрольных точек для вычислений
        control_points = [cp.get_position() for cp in self.control_points]
        
        # Используем улучшенный алгоритм для более равномерного распределения точек
        # при большем количестве контрольных точек
        num_points = self.num_points
        
        # Используем модифицированное распределение параметра t
        # для более равномерного распределения точек на кривой
        for i in range(num_points):
            # Используем корректированный параметр t
            # Линейное распределение дает неравномерные расстояния на кривой
            # Мы можем использовать другие формулы для более равномерного распределения
            t = i / (num_points - 1) if num_points > 1 else 0
            
            # Вычисляем точку на кривой
            point_pos = self._bezier_point_optimized(control_points, t)
            
            # Добавляем точку в список точек кривой
            self.curve_points.append(BezierPoint(point_pos))
        
        # Оптимизация: если точек много, можно пересчитать расстояния и перераспределить
        if num_points > 10 and len(control_points) > 3:
            self._optimize_point_distribution()
        
        logger.info(f"Сгенерировано {len(self.curve_points)} точек на кривой")
    
    def _optimize_point_distribution(self):
        """Оптимизирует распределение точек по кривой, делая его более равномерным"""
        if len(self.curve_points) < 3:
            return
        
        # Вычисляем длины сегментов между соседними точками
        segments = []
        total_length = 0
        
        for i in range(1, len(self.curve_points)):
            prev_pos = self.curve_points[i-1].get_position()
            curr_pos = self.curve_points[i].get_position()
            length = (curr_pos - prev_pos).length
            segments.append(length)
            total_length += length
        
        # Если разброс длин слишком большой, перераспределяем точки
        max_length = max(segments)
        min_length = min(segments)
        
        # Если отношение максимальной длины к минимальной слишком велико, оптимизируем
        if max_length > min_length * 3 and len(self.curve_points) > 5:
            logger.debug(f"Оптимизация распределения точек. Макс/мин: {max_length/min_length:.2f}")
            
            # Получаем контрольные точки для вычислений
            control_points = [cp.get_position() for cp in self.control_points]
            
            # Вычисляем приблизительную длину кривой с помощью адаптивного подхода
            curve_length = self._estimate_curve_length(control_points)
            
            # Создаем новый набор точек с более равномерным распределением
            new_points = []
            segment_length = curve_length / (self.num_points - 1)
            
            # Первая точка всегда остается на месте
            new_points.append(self.curve_points[0])
            
            # Используем алгоритм бинарного поиска для нахождения параметра t,
            # соответствующего заданному расстоянию вдоль кривой
            current_length = 0
            for i in range(1, self.num_points - 1):
                target_length = i * segment_length
                t = self._find_t_for_arc_length(control_points, target_length, curve_length)
                point_pos = self._bezier_point_optimized(control_points, t)
                new_points.append(BezierPoint(point_pos))
            
            # Последняя точка всегда должна быть на конце кривой
            new_points.append(self.curve_points[-1])
            
            # Обновляем список точек
            self.curve_points = new_points
    
    def _estimate_curve_length(self, control_points, segments=100):
        """Оценивает длину кривой Безье путем суммирования длин сегментов"""
        length = 0
        prev_point = None
        
        for i in range(segments + 1):
            t = i / segments
            point = self._bezier_point_optimized(control_points, t)
            
            if prev_point:
                length += (point - prev_point).length
            
            prev_point = point
        
        return length
    
    def _find_t_for_arc_length(self, control_points, target_length, total_length, epsilon=0.001):
        """
        Находит параметр t, соответствующий заданной длине дуги на кривой Безье
        с использованием бинарного поиска
        """
        if target_length <= 0:
            return 0
        if target_length >= total_length:
            return 1
        
        t_min = 0
        t_max = 1
        t_mid = 0.5
        max_iterations = 20
        iteration = 0
        
        while iteration < max_iterations:
            length = self._measure_arc_length(control_points, 0, t_mid)
            
            if abs(length - target_length) < epsilon:
                break
            
            if length < target_length:
                t_min = t_mid
            else:
                t_max = t_mid
            
            t_mid = (t_min + t_max) / 2
            iteration += 1
        
        return t_mid
    
    def _measure_arc_length(self, control_points, t_start, t_end, segments=20):
        """Измеряет длину дуги кривой Безье между параметрами t_start и t_end"""
        length = 0
        prev_point = None
        
        for i in range(segments + 1):
            t = t_start + (t_end - t_start) * (i / segments)
            point = self._bezier_point_optimized(control_points, t)
            
            if prev_point:
                length += (point - prev_point).length
            
            prev_point = point
        
        return length
    
    def _bezier_point_optimized(self, control_points, t):
        """
        Оптимизированное вычисление точки на кривой Безье с использованием алгоритма де Кастельжо
        
        Args:
            control_points (list): Список контрольных точек
            t (float): Параметр кривой (0-1)
            
        Returns:
            Vector: Позиция точки на кривой
        """
        # Копируем контрольные точки для работы алгоритма
        points = control_points.copy()
        n = len(points)
        
        # Применяем алгоритм де Кастельжо
        for r in range(1, n):
            for i in range(n - r):
                points[i] = (1 - t) * points[i] + t * points[i + 1]
        
        # Возвращаем итоговую точку (первая точка после всех итераций)
        return points[0]
    
    def _bezier_point(self, control_points, t, n=None):
        """
        Вычисление точки на кривой Безье для заданного параметра t (рекурсивный метод)
        
        Args:
            control_points (list): Список контрольных точек
            t (float): Параметр кривой (0-1)
            n (int, optional): Степень кривой
            
        Returns:
            Vector: Позиция точки на кривой
        """
        if n is None:
            n = len(control_points) - 1
        
        # Алгоритм де Кастельжо для вычисления точки на кривой Безье
        if n == 0:
            return control_points[0]
        
        return (1 - t) * self._bezier_point(control_points[:-1], t, n - 1) + \
               t * self._bezier_point(control_points[1:], t, n - 1)
    
    def update_control_point(self, index, position):
        """Обновление позиции контрольной точки"""
        if 0 <= index < len(self.control_points):
            self.control_points[index].set_position(position)
            self._generate_curve_points()
            logger.info(f"Обновлена контрольная точка {index}: {position}")
            return True
        return False
    
    def set_num_handles(self, num_handles):
        """Установка количества контрольных точек"""
        if num_handles >= 0:
            self.num_handles = num_handles
            if self.is_complete:
                self._generate_control_points()
                self._generate_curve_points()
            logger.info(f"Установлено новое количество контрольных точек: {num_handles}")
            return True
        return False
    
    def set_num_points(self, num_points):
        """Установка количества точек на кривой"""
        if num_points >= 2:
            self.num_points = num_points
            if self.is_complete:
                self._generate_curve_points()
            logger.info(f"Установлено новое количество точек на кривой: {num_points}")
            return True
        return False
    
    def set_preview_enabled(self, enabled):
        """Включение/выключение предварительного просмотра"""
        self.preview_enabled = enabled
        logger.info(f"Предварительный просмотр {'включен' if enabled else 'выключен'}")
    
    def find_closest_handle(self, position, max_distance=20):
        """
        Поиск ближайшей контрольной точки к заданной позиции
        
        Args:
            position (Vector): Позиция курсора мыши в пространстве изображения
            max_distance (float): Максимальное расстояние для срабатывания выбора
            
        Returns:
            tuple: (индекс ближайшей точки, расстояние) или (None, None) если точка не найдена
        """
        if not self.control_points:
            logger.debug("Нет контрольных точек для выбора")
            return None, None
            
        closest_index = None
        min_distance = float('inf')
        
        # Вывод отладочной информации
        logger.debug(f"Поиск ближайшей точки к позиции: ({position.x}, {position.y})")
        logger.debug(f"Максимальное расстояние поиска: {max_distance}")
        logger.debug(f"Контрольные точки: {[(i, p.get_position().x, p.get_position().y) for i, p in enumerate(self.control_points)]}")
        
        # Оптимизация: сначала проверяем начальную и конечную точки
        # Пользователи часто хотят выбрать именно их
        special_points = []
        if len(self.control_points) > 0:
            # Добавляем начальную точку
            special_points.append((0, self.control_points[0]))
        if len(self.control_points) > 1:
            # Добавляем конечную точку
            special_points.append((len(self.control_points) - 1, self.control_points[-1]))
            
        # Проверяем специальные точки первыми
        for idx, point in special_points:
            point_pos = point.get_position()
            
            # Быстрое вычисление квадрата расстояния (без извлечения корня)
            dx = point_pos.x - position.x
            dy = point_pos.y - position.y
            distance_squared = dx*dx + dy*dy
            
            # Сравниваем с квадратом максимального расстояния для эффективности
            max_distance_squared = max_distance * max_distance
            
            if distance_squared <= max_distance_squared:
                distance = distance_squared ** 0.5  # Вычисляем корень только если точка в радиусе
                
                # Если это первая найденная точка или она ближе предыдущей ближайшей
                if closest_index is None or distance < min_distance:
                    min_distance = distance
                    closest_index = idx
                    
                    # Если точка очень близко, можем сразу вернуть ее
                    if distance < max_distance / 4:
                        logger.debug(f"Найдена очень близкая специальная точка: {idx}, расстояние={distance}")
                        return idx, distance
        
        # Проверяем остальные точки только если специальные не были найдены достаточно близко
        for i, point in enumerate(self.control_points):
            # Пропускаем специальные точки, которые уже проверены
            if i == 0 or i == len(self.control_points) - 1:
                continue
                
            point_pos = point.get_position()
            
            # Быстрое вычисление квадрата расстояния
            dx = point_pos.x - position.x
            dy = point_pos.y - position.y
            distance_squared = dx*dx + dy*dy
            
            # Логируем для отладки
            logger.debug(f"Точка {i}: pos=({point_pos.x}, {point_pos.y}), квадрат расстояния={distance_squared}")
            
            # Сравниваем с квадратом максимального расстояния
            max_distance_squared = max_distance * max_distance
            
            if distance_squared <= max_distance_squared:
                distance = distance_squared ** 0.5  # Вычисляем корень только при необходимости
                
                if closest_index is None or distance < min_distance:
                    min_distance = distance
                    closest_index = i
        
        # Вывод результата
        if closest_index is not None:
            logger.debug(f"Найдена ближайшая точка: {closest_index}, расстояние: {min_distance}")
            return closest_index, min_distance
        else:
            logger.debug("Не найдено точек в пределах допустимого расстояния")
            return None, None
    
    def reset(self):
        """Сброс кривой"""
        self.control_points = []
        self.curve_points = []
        self.is_complete = False
        logger.info("Кривая сброшена")


class BezierCurveManager:
    """Класс для управления несколькими кривыми Безье"""
    def __init__(self):
        """Инициализация менеджера кривых"""
        self.curves = []
        self.active_curve_index = -1
        self.mode = 'SINGLE_POINTS'  # 'SINGLE_POINTS' или 'BEZIER_CURVE'
    
    def set_mode(self, mode):
        """Установка режима работы"""
        if mode in ['SINGLE_POINTS', 'BEZIER_CURVE']:
            self.mode = mode
            logger.info(f"Установлен режим: {mode}")
            return True
        return False
    
    def new_curve(self):
        """Создание новой кривой"""
        curve = BezierCurve()
        self.curves.append(curve)
        self.active_curve_index = len(self.curves) - 1
        logger.info(f"Создана новая кривая, индекс: {self.active_curve_index}")
        return curve
    
    def get_active_curve(self):
        """Получение активной кривой"""
        if self.curves:  # Если есть хотя бы одна кривая
            if 0 <= self.active_curve_index < len(self.curves):
                return self.curves[self.active_curve_index]
            # Если индекс недействителен, но есть кривые, вернуть первую кривую
            print(f"Не нашли кривую: {self.curves[0]}")
            return self.curves[0]
        return None
    
    def set_active_curve(self, index):
        """Установка активной кривой"""
        if 0 <= index < len(self.curves):
            self.active_curve_index = index
            logger.info(f"Установлена активная кривая, индекс: {index}")
            return True
        return False
    
    def delete_curve(self, index=None):
        """Удаление кривой"""
        if index is None:
            index = self.active_curve_index
        
        if 0 <= index < len(self.curves):
            deleted = self.curves.pop(index)
            logger.info(f"Удалена кривая, индекс: {index}")
            
            # Обновляем индекс активной кривой
            if len(self.curves) == 0:
                self.active_curve_index = -1
            elif self.active_curve_index >= len(self.curves):
                self.active_curve_index = len(self.curves) - 1
            return True
        return False
    
    def get_all_calibration_points(self):
        """
        Получение всех точек калибровки со всех кривых
        
        Returns:
            list: Список позиций всех точек калибровки
        """
        all_points = []
        for curve in self.curves:
            all_points.extend([point.get_position() for point in curve.curve_points])
        return all_points


# Вспомогательные функции для отрисовки элементов - эти функции будут использованы в bezier_ui.py

def draw_handle_point(shader, x, y, color, size):
    """
    Отрисовка контрольной точки (хендлера) с улучшенным внешним видом
    
    Args:
        shader: Шейдер для отрисовки
        x, y: Координаты центра точки
        color: Цвет точки (r, g, b, a)
        size: Размер точки в пикселях
    """
    shader.uniform_float("color", color)
    gpu.state.blend_set('ALPHA')
    
    # Создаем круг для точки с большим количеством сегментов для более гладкого вида
    vertices = []
    num_segments = 24  # Увеличиваем количество сегментов для более гладкого круга
    
    for i in range(num_segments):
        angle = 2.0 * np.pi * i / num_segments
        vx = x + size * np.cos(angle)
        vy = y + size * np.sin(angle)
        vertices.append((vx, vy))
    
    # Отрисовка круга
    batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": vertices})
    batch.draw(shader)
    
    # Рисуем только контур круга без заполнения
    # Увеличиваем толщину линии для лучшей видимости
    gpu.state.line_width_set(2.0)
    batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": vertices})
    batch.draw(shader)

    # Возвращаем стандартную толщину линии
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')

def draw_cross_point(shader, x, y, color, size):
    """
    Отрисовка точки в виде крестика 
    
    Args:
        shader: Шейдер для отрисовки
        x, y: Координаты центра точки
        color: Цвет точки (r, g, b, a)
        size: Размер крестика в пикселях
    """
    shader.uniform_float("color", color)
    gpu.state.blend_set('ALPHA')
    
    # Увеличиваем толщину линии для лучшей видимости
    gpu.state.line_width_set(2.0)
    
    # Создаем линии крестика (горизонтальная и вертикальная)
    h_line = [(x - size, y), (x + size, y)]  # Горизонтальная линия
    v_line = [(x, y - size), (x, y + size)]  # Вертикальная линия
    
    # Отрисовка горизонтальной линии
    batch = batch_for_shader(shader, 'LINES', {"pos": h_line})
    batch.draw(shader)
    
    # Отрисовка вертикальной линии
    batch = batch_for_shader(shader, 'LINES', {"pos": v_line})
    batch.draw(shader)
    
    # Возвращаем стандартную толщину линии
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')

def draw_line(shader, x1, y1, x2, y2, color, width):
    """
    Отрисовка линии с улучшенной визуализацией
    
    Args:
        shader: Шейдер для отрисовки
        x1, y1: Координаты начальной точки
        x2, y2: Координаты конечной точки
        color: Цвет линии (r, g, b, a)
        width: Толщина линии в пикселях
    """
    shader.uniform_float("color", color)
    gpu.state.blend_set('ALPHA')
    
    # Устанавливаем толщину линии
    gpu.state.line_width_set(width)
    
    # Создаем линию
    vertices = [(x1, y1), (x2, y2)]
    batch = batch_for_shader(shader, 'LINES', {"pos": vertices})
    batch.draw(shader)
    
    # Если линия достаточно толстая, добавляем маленькие кружки на концах
    # для более плавного соединения с точками
    if width >= 2.0:
        # Маленькие кружки на концах линии для более гладкого соединения
        end_size = width * 0.6
        for point_x, point_y in [(x1, y1), (x2, y2)]:
            end_vertices = []
            num_segments = 12  # Небольшое количество сегментов для маленького круга
            
            for i in range(num_segments):
                angle = 2.0 * np.pi * i / num_segments
                vx = point_x + end_size * np.cos(angle)
                vy = point_y + end_size * np.sin(angle)
                end_vertices.append((vx, vy))
            
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": [(point_x, point_y)] + end_vertices})
            batch.draw(shader)
    
    # Восстанавливаем стандартную толщину линии
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def draw_text(shader, text, x, y, color):
    """
    Отрисовка текста с улучшенным визуальным представлением
    
    Args:
        shader: Шейдер для отрисовки
        text: Текст для отображения
        x, y: Координаты начала текста
        color: Цвет текста (r, g, b, a)
    """
    import blf
    
    # Настройка шрифта
    font_id = 0
    
    # Создаем тень для текста, чтобы он был лучше виден на любом фоне
    shadow_offset = 1.0
    shadow_color = (0.0, 0.0, 0.0, 0.7)  # Полупрозрачный черный для тени
    
    # Сначала рисуем тень
    blf.color(font_id, *shadow_color)
    blf.size(font_id, 14)
    blf.position(font_id, x + shadow_offset, y - shadow_offset, 0)
    blf.draw(font_id, text)
    
    # Затем рисуем основной текст
    blf.color(font_id, *color)
    blf.position(font_id, x, y, 0)
    blf.draw(font_id, text) 