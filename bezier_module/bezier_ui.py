"""
Модуль пользовательского интерфейса для работы с кривыми Безье
"""

import bpy
from bpy.props import (
    IntProperty,
    BoolProperty,
    EnumProperty,
    StringProperty,
    FloatProperty,
    FloatVectorProperty,
    PointerProperty,
    CollectionProperty
)
import numpy as np
from mathutils import Vector
import logging
import gpu
from gpu_extras.batch import batch_for_shader
import gpu.state
from . import bezier_curves

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Глобальные переменные для хранения состояния
current_bezier_state = 'NONE'
active_handle_index = -1
is_dragging_handle = False
active_drag_axis = None
hover_gizmo_axis = None
hover_gizmo_handle_index = -1
drag_origin_position = None

# Состояния для создания кривой
BEZIER_STATE_NONE = 'NONE'
BEZIER_STATE_FIRST_POINT = 'FIRST_POINT'
BEZIER_STATE_EDIT = 'EDIT'
BEZIER_STATE_ADD = 'ADD'

# Глобальные переменные для отслеживания
mouse_delta = Vector((0, 0))


def initialize_bezier_manager():
    """Инициализация менеджера кривых Безье"""
    if not hasattr(bpy.types, "im_bezier_manager_prop"):
        # Регистрируем свойство для всех сцен
        bpy.types.Scene.im_bezier_manager = bpy.props.PointerProperty(type=bpy.types.PropertyGroup)
        try:
            # В качестве атрибута класса храним один экземпляр менеджера для всего аддона
            bpy.types.im_bezier_manager_prop = bezier_curves.BezierCurveManager()
            logger.info("Инициализирован менеджер кривых Безье")
        except Exception as e:
            logger.error(f"Ошибка создания BezierCurveManager: {e}")
            return None


def get_bezier_manager():
    """Получение экземпляра менеджера кривых Безье"""
    if not hasattr(bpy.types, "im_bezier_manager_prop"):
        initialize_bezier_manager()
    return bpy.types.im_bezier_manager_prop


def reset_bezier_state():
    """Сбросить состояние кривой Безье"""
    global current_bezier_state, active_handle_index, is_dragging_handle
    global active_drag_axis, hover_gizmo_axis, hover_gizmo_handle_index, drag_origin_position
    current_bezier_state = BEZIER_STATE_NONE
    active_handle_index = -1
    is_dragging_handle = False
    active_drag_axis = None
    hover_gizmo_axis = None
    hover_gizmo_handle_index = -1
    drag_origin_position = None
    logger.info("Состояние кривой Безье сброшено")


def reset_bezier_curves():
    """Полный сброс кривых Безье - удаление всех кривых из менеджера и создание новой"""
    manager = get_bezier_manager()
    if manager:
        # Удаляем все кривые
        while len(manager.curves) > 0:
            manager.delete_curve(0)
        # Создаем новую кривую
        manager.new_curve()
        # Сбрасываем состояние
        reset_bezier_state()
        logger.info("Кривые Безье полностью сброшены")


def get_current_bezier_state():
    """Получить текущее состояние кривой Безье"""
    global current_bezier_state
    return current_bezier_state


def set_bezier_state(state):
    """Установка состояния работы с кривой Безье"""
    global current_bezier_state
    current_bezier_state = state
    logger.info(f"Установлено состояние работы с кривой Безье: {state}")


# Классы свойств для панели настроек
class BezierCurveProperties(bpy.types.PropertyGroup):
    """Свойства для настройки кривой Безье"""
    
    placement_mode: EnumProperty(
        name="Режим расстановки точек",
        description="Выбор между расстановкой одиночных точек или кривой Безье",
        items=[
            ('SINGLE_POINTS', "Одиночные точки", "Расстановка отдельных точек"),
            ('BEZIER_CURVE', "Кривая Безье", "Расстановка точек с помощью кривой Безье"),
        ],
        default='SINGLE_POINTS',
        update=lambda self, context: update_placement_mode(self, context)
    )
    
    num_control_points: IntProperty(
        name="Количество контрольных точек",
        description="Количество контрольных точек для кривой Безье",
        default=3,
        min=0,
        max=20,
        update=lambda self, context: update_control_points(self, context)
    )
    
    num_curve_points: IntProperty(
        name="Количество точек на кривой",
        description="Количество точек, которые будут созданы на кривой",
        default=5,
        min=2,
        max=100,
        update=lambda self, context: update_curve_points(self, context)
    )
    
    show_preview: BoolProperty(
        name="Предварительный просмотр",
        description="Показать предварительный просмотр точек на кривой",
        default=True,
        update=lambda self, context: update_preview(self, context)
    )
    
    placement_status: StringProperty(
        name="Статус расстановки",
        description="Текущий статус расстановки точек",
        default="Установите начальную и конечную точку"
    )


# Функции обновления свойств
def update_control_points(self, context):
    """Обновление количества контрольных точек"""
    manager = get_bezier_manager()
    curve = manager.get_active_curve()
    print(f'curve: {curve}')

    if curve and curve.is_complete:
        curve.set_num_handles(self.num_control_points)
        print(f'curve.set_num_handles: {curve.set_num_handles}')
        logger.info(f"Обновлено количество контрольных точек: {self.num_control_points}")
    else:
        print(f'Не изменили количество контрольных точек')


def update_curve_points(self, context):
    """Обновление количества точек на кривой"""
    manager = get_bezier_manager()
    curve = manager.get_active_curve()
    
    if curve and curve.is_complete:
        curve.set_num_points(self.num_curve_points)
        logger.info(f"Обновлено количество точек на кривой: {self.num_curve_points}")


def update_preview(self, context):
    """Обновление видимости предварительного просмотра"""
    manager = get_bezier_manager()
    curve = manager.get_active_curve()
    
    if curve:
        curve.set_preview_enabled(self.show_preview)
        logger.info(f"Предварительный просмотр {'включен' if self.show_preview else 'выключен'}")


def update_placement_mode(self, context):
    """Обновление режима расстановки точек"""
    manager = get_bezier_manager()
    manager.set_mode(self.placement_mode)
    
    # Сбрасываем состояние кривой при переключении режима
    reset_bezier_state()
    
    # Обновляем статус в зависимости от режима
    if self.placement_mode == 'BEZIER_CURVE':
        self.placement_status = "Установите начальную и конечную точку"
    else:
        self.placement_status = ""
    
    logger.info(f"Обновлен режим расстановки точек: {self.placement_mode}")


def _get_image_pos_to_region_coords(context, image_item, params):
    """Возвращает функцию преобразования координат изображения в координаты региона."""
    image_to_region_coords = None
    try:
        import sys
        for module_name in sys.modules.keys():
            if module_name.endswith('image_editor'):
                module = sys.modules[module_name]
                if hasattr(module, 'image_to_region_coords'):
                    image_to_region_coords = module.image_to_region_coords
                    break

        if not image_to_region_coords:
            logger.warning("Не найдена функция image_to_region_coords, используем локальное преобразование")
    except Exception as e:
        logger.error(f"Ошибка при поиске функции преобразования координат: {e}")

    def image_pos_to_region_coords(point_pos):
        if image_to_region_coords and context:
            try:
                result = image_to_region_coords(context, image_item, point_pos.x, point_pos.y, params)
                if result:
                    return result
            except Exception as e:
                logger.error(f"Ошибка при вызове image_to_region_coords: {e}")

        pos_x = params.get('pos_x', 0)
        pos_y = params.get('pos_y', 0)
        scale = params.get('scale', 1.0)
        if scale <= 0:
            scale = 1.0

        region_x = pos_x + point_pos.x * scale
        region_y = pos_y + point_pos.y * scale
        return region_x, region_y

    return image_pos_to_region_coords


def _get_bezier_gizmo_metrics(point):
    """Размеры маленького осевого gizmo вокруг контрольной точки."""
    is_emphasized = bool(point.selected or point.hovered)
    return {
        "arm_length": 18 if is_emphasized else 16,
        "inner_gap": 7 if is_emphasized else 6,
        "arm_hit_padding": 6,
        "dead_zone": 8 if is_emphasized else 7,
        "arrow_size": 5 if is_emphasized else 4,
        "center_radius": 5 if is_emphasized else 4,
    }


def _distance_to_segment(point, a, b):
    """Расстояние от точки до сегмента в экранных координатах."""
    ab = b - a
    ab_len_sq = ab.length_squared
    if ab_len_sq <= 1e-8:
        return (point - a).length

    t = max(0.0, min(1.0, (point - a).dot(ab) / ab_len_sq))
    projection = a + ab * t
    return (point - projection).length


def _find_gizmo_axis_hit(curve, mouse_pos, context, image_item, params):
    """Находит ось gizmo, если пользователь попал по осевой ручке."""
    if not curve or not curve.control_points or not params:
        return None, None, None

    image_pos_to_region_coords = _get_image_pos_to_region_coords(context, image_item, params)
    mouse_vec = Vector((mouse_pos.x, mouse_pos.y))

    best_handle_index = None
    best_axis = None
    best_distance = float('inf')

    for index, point in enumerate(curve.control_points):
        handle_x, handle_y = image_pos_to_region_coords(point.get_position())
        handle_pos = Vector((handle_x, handle_y))
        metrics = _get_bezier_gizmo_metrics(point)
        arm_length = metrics["arm_length"]
        inner_gap = metrics["inner_gap"]
        arm_hit_padding = metrics["arm_hit_padding"]
        dead_zone = metrics["dead_zone"]

        offset = mouse_vec - handle_pos
        radius = offset.length
        if radius < dead_zone or radius > arm_length + arm_hit_padding:
            continue

        x_distances = (
            _distance_to_segment(mouse_vec, Vector((handle_x - inner_gap, handle_y)), Vector((handle_x - arm_length, handle_y))),
            _distance_to_segment(mouse_vec, Vector((handle_x + inner_gap, handle_y)), Vector((handle_x + arm_length, handle_y))),
        )
        y_distances = (
            _distance_to_segment(mouse_vec, Vector((handle_x, handle_y - inner_gap)), Vector((handle_x, handle_y - arm_length))),
            _distance_to_segment(mouse_vec, Vector((handle_x, handle_y + inner_gap)), Vector((handle_x, handle_y + arm_length))),
        )

        x_distance = min(x_distances)
        y_distance = min(y_distances)

        if x_distance <= arm_hit_padding and x_distance < best_distance:
            best_handle_index = index
            best_axis = 'X'
            best_distance = x_distance

        if y_distance <= arm_hit_padding and y_distance < best_distance:
            best_handle_index = index
            best_axis = 'Y'
            best_distance = y_distance

    if best_handle_index is None:
        return None, None, None

    return best_handle_index, best_axis, best_distance


def _draw_bezier_axis_gizmo(shader, x, y, point, is_active_axis_x=False, is_active_axis_y=False):
    """Отрисовка маленького gizmo с блокировкой по осям."""
    metrics = _get_bezier_gizmo_metrics(point)
    arm_length = metrics["arm_length"]
    inner_gap = metrics["inner_gap"]
    arrow_size = metrics["arrow_size"]
    center_radius = metrics["center_radius"]

    x_color = (1.0, 0.28, 0.28, 0.96 if is_active_axis_x else 0.82)
    y_color = (0.22, 0.62, 1.0, 0.96 if is_active_axis_y else 0.82)

    def draw_line_and_arrow(start, end, color, line_width):
        shader.uniform_float("color", color)
        gpu.state.line_width_set(line_width)
        batch = batch_for_shader(shader, 'LINES', {"pos": [tuple(start), tuple(end)]})
        batch.draw(shader)

        direction = (end - start)
        if direction.length_squared <= 1e-8:
            return
        direction.normalize()
        perpendicular = Vector((-direction.y, direction.x))
        arrow_base = end - direction * arrow_size
        wing_a = arrow_base + perpendicular * (arrow_size * 0.55)
        wing_b = arrow_base - perpendicular * (arrow_size * 0.55)
        arrow_batch = batch_for_shader(
            shader,
            'LINES',
            {"pos": [tuple(wing_a), tuple(end), tuple(wing_b), tuple(end)]}
        )
        arrow_batch.draw(shader)

    line_width_x = 2.6 if is_active_axis_x else 1.7
    line_width_y = 2.6 if is_active_axis_y else 1.7

    draw_line_and_arrow(Vector((x - inner_gap, y)), Vector((x - arm_length, y)), x_color, line_width_x)
    draw_line_and_arrow(Vector((x + inner_gap, y)), Vector((x + arm_length, y)), x_color, line_width_x)
    draw_line_and_arrow(Vector((x, y - inner_gap)), Vector((x, y - arm_length)), y_color, line_width_y)
    draw_line_and_arrow(Vector((x, y + inner_gap)), Vector((x, y + arm_length)), y_color, line_width_y)

    # Черный центр для свободного перетаскивания.
    shader.uniform_float("color", (0.02, 0.02, 0.02, 0.96))
    center_vertices = [(x, y)]
    num_segments = 18
    for i in range(num_segments + 1):
        angle = 2.0 * np.pi * i / num_segments
        center_vertices.append((
            x + center_radius * np.cos(angle),
            y + center_radius * np.sin(angle),
        ))
    center_batch = batch_for_shader(shader, 'TRI_FAN', {"pos": center_vertices})
    center_batch.draw(shader)

    shader.uniform_float("color", (0.95, 0.95, 0.95, 0.9))
    gpu.state.line_width_set(1.2)
    outline_vertices = center_vertices[1:]
    outline_batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": outline_vertices})
    outline_batch.draw(shader)
    gpu.state.line_width_set(1.0)


def _set_selected_control_point(curve, selected_index):
    for index, point in enumerate(curve.control_points):
        point.selected = (index == selected_index)


# Операторы для работы с кривыми Безье
class CAMCALIB_OT_apply_bezier_points(bpy.types.Operator):
    """Применить точки с кривой Безье к калибровке"""
    bl_idname = "camera_calibration.apply_bezier_points"
    bl_label = "Применить точки Безье"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        props = scene.bezier_props
        manager = get_bezier_manager()
        
        if props.placement_mode != 'BEZIER_CURVE':
            self.report({'ERROR'}, "Режим кривой Безье не активен")
            return {'CANCELLED'}
        
        curve = manager.get_active_curve()
        if not curve or not curve.is_complete:
            self.report({'ERROR'}, "Кривая не создана или не завершена")
            return {'CANCELLED'}
        
        # Получаем точки калибровки с кривой
        points = [point.get_position() for point in curve.curve_points]
        
        if not points:
            self.report({'ERROR'}, "На кривой нет точек")
            return {'CANCELLED'}
        
        logger.info(f"Подготовлено {len(points)} точек для добавления")
        
        # Добавляем точки к активному изображению
        added_count = add_bezier_points_to_active_image(context, points)
        
        # Явная проверка на None для защиты от ошибок
        if added_count is None:
            logger.error("Функция add_bezier_points_to_active_image вернула None вместо числа")
            added_count = -1
        
        if added_count > 0:
            self.report({'INFO'}, f"Добавлено {added_count} точек к активному изображению")
        elif added_count == 0:
            self.report({'WARNING'}, "Не удалось добавить точки к изображению")
        else:
            self.report({'ERROR'}, "Произошла ошибка при добавлении точек")
        
        # Сбрасываем состояние для создания новой кривой
        reset_bezier_curves()
        manager = get_bezier_manager()
        manager.new_curve()
        props.placement_status = "Установите начальную и конечную точку"
        
        return {'FINISHED'}


class CAMCALIB_OT_clear_bezier(bpy.types.Operator):
    """Очистить текущую кривую Безье"""
    bl_idname = "camera_calibration.clear_bezier"
    bl_label = "Очистить кривую Безье"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        props = scene.bezier_props
        manager = get_bezier_manager()
        
        if props.placement_mode != 'BEZIER_CURVE':
            self.report({'ERROR'}, "Режим кривой Безье не активен")
            return {'CANCELLED'}
        
        # Удаляем текущую кривую и создаем новую
        manager.delete_curve()
        manager.new_curve()
        
        # Сбрасываем состояние
        reset_bezier_state()
        props.placement_status = "Установите начальную и конечную точку"
        
        self.report({'INFO'}, "Кривая Безье очищена")
        return {'FINISHED'}


# Функции для интеграции с редактором изображений
def draw_bezier_curve(shader, curve, image_item, context, params):
    """
    Отрисовка кривой Безье и её элементов
    
    Args:
        shader: Шейдер для отрисовки
        curve (BezierCurve): Кривая Безье
        image_item: Текущий элемент изображения
        context: Контекст Blender
        params: Параметры отображения изображения
    """
    # Проверяем, есть ли кривая
    if not curve:
        return
    
    # Проверяем, что params не None
    if params is None:
        logger.error("Не удалось получить параметры отображения изображения")
        return
    
    # Проверяем корректность image_item
    if not image_item:
        logger.error("Не удалось получить элемент изображения")
        return
    
    image_pos_to_region_coords = _get_image_pos_to_region_coords(context, image_item, params)
    
    # Отрисовка линий между контрольными точками для показа скелета кривой
    if len(curve.control_points) > 1:
        # Собираем все точки для линии контрольных точек
        control_line_points = []
        for cp in curve.control_points:
            pos = cp.get_position()
            x, y = image_pos_to_region_coords(pos)
            control_line_points.append((x, y))
        
        # Рисуем пунктирную линию для контрольных точек
        if control_line_points:
            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": control_line_points})
            shader.uniform_float("color", (0.8, 0.8, 0.8, 0.6))
            gpu.state.line_width_set(1.0)
            batch.draw(shader)
    
    # Отрисовка линии кривой Безье. Линия остается под точками превью.
    if curve.preview_enabled and curve.is_complete:
        if len(curve.curve_points) > 1:
            curve_line_points = []
            control_points = [cp.get_position() for cp in curve.control_points]
            num_visual_points = max(30, len(curve.curve_points) * 2)
            
            for i in range(num_visual_points):
                t = i / (num_visual_points - 1)
                points = control_points.copy()
                n = len(points)
                
                for r in range(1, n):
                    for j in range(n - r):
                        points[j] = (1 - t) * points[j] + t * points[j + 1]
                
                point_pos = points[0]
                x, y = image_pos_to_region_coords(point_pos)
                curve_line_points.append((x, y))
            
            if curve_line_points:
                batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": curve_line_points})
                shader.uniform_float("color", (0.05, 0.62, 1.0, 0.72))
                gpu.state.line_width_set(1.6)
                batch.draw(shader)
                gpu.state.line_width_set(1.0)

    # Отрисовка контрольных точек и gizmo поверх линии кривой.
    for i, point in enumerate(curve.control_points):
        pos = point.get_position()
        x, y = image_pos_to_region_coords(pos)

        if i == 0:
            color = (0.0, 1.0, 0.0, 1.0)
            size = 12
        elif i == len(curve.control_points) - 1:
            color = (1.0, 0.0, 0.0, 1.0)
            size = 12
        else:
            color = (1.0, 0.95, 0.2, 1.0)
            size = 10

        if point.selected:
            color = (1.0, 1.0, 1.0, 1.0)
            size += 2
        elif point.hovered:
            size += 2

        show_gizmo = bool(point.hovered or (is_dragging_handle and active_handle_index == i))
        if show_gizmo:
            is_x_active = bool(active_handle_index == i and active_drag_axis == 'X')
            is_y_active = bool(active_handle_index == i and active_drag_axis == 'Y')
            is_hover_x = bool(hover_gizmo_handle_index == i and hover_gizmo_axis == 'X')
            is_hover_y = bool(hover_gizmo_handle_index == i and hover_gizmo_axis == 'Y')
            _draw_bezier_axis_gizmo(
                shader,
                x,
                y,
                point,
                is_active_axis_x=(is_x_active or is_hover_x),
                is_active_axis_y=(is_y_active or is_hover_y),
            )
        else:
            bezier_curves.draw_handle_point(shader, x, y, color, size)

        if hasattr(bezier_curves, 'draw_text') and (point.selected or point.hovered):
            bezier_curves.draw_text(shader, str(i), x + 15, y + 5, color)

    # Отрисовка крестиков превью последней, чтобы они всегда были на переднем плане.
    if curve.preview_enabled and curve.is_complete:
        for point in curve.curve_points:
            pos = point.get_position()
            x, y = image_pos_to_region_coords(pos)
            bezier_curves.draw_cross_point(shader, x, y, (1.0, 0.4, 0.0, 1.0), 8)


def find_closest_handle_with_correction(curve, mouse_pos, y_correction=0.0, max_distance=20, context=None, image_item=None, params=None):
    """
    Обертка над find_closest_handle, которая применяет коррекцию координат
    перед поиском ближайшей точки. Учитывает параметры отображения изображения.
    
    Args:
        curve (BezierCurve): Кривая Безье
        mouse_pos (Vector): Позиция мыши в регионе
        y_correction (float): Устаревший параметр, остался для обратной совместимости
        max_distance (float): Максимальное расстояние для поиска
        context: Контекст Blender (для преобразования координат)
        image_item: Элемент изображения (для получения параметров)
        params: Параметры отображения изображения
        
    Returns:
        tuple: (индекс ближайшей точки, расстояние)
    """
    # Проверяем наличие кривой и контрольных точек
    if not curve or not curve.control_points:
        return None, None
    
    # Если нет параметров отображения - используем простой поиск
    if not params or not context or not image_item:
        logger.debug("Используем простой поиск без преобразования координат")
        return curve.find_closest_handle(mouse_pos, max_distance)
    
    try:
        # Импортируем функцию преобразования координат
        mouse_to_image_coords = None
        import sys
        for module_name in sys.modules.keys():
            if module_name.endswith('image_editor'):
                module = sys.modules[module_name]
                if hasattr(module, 'mouse_to_image_coords'):
                    mouse_to_image_coords = module.mouse_to_image_coords
                    break
        
        if not mouse_to_image_coords:
            logger.warning("Функция mouse_to_image_coords не найдена, используем простой поиск")
            return curve.find_closest_handle(mouse_pos, max_distance)
        
        # Получаем масштаб для корректировки максимального расстояния
        scale = params.get('scale', 1.0)
        
        # Проверка масштаба на корректность
        if scale <= 0:
            logger.warning(f"Некорректный масштаб: {scale}, устанавливаем 1.0")
            scale = 1.0
        
        # Получаем координаты мыши в пространстве региона
        region = context.region
        mouse_region_x = mouse_pos.x
        mouse_region_y = mouse_pos.y
        
        # Преобразуем координаты региона в абсолютные координаты для mouse_to_image_coords
        # Абсолютные координаты учитывают положение области относительно окна Blender
        mouse_x = mouse_region_x + region.x
        mouse_y = mouse_region_y + region.y
        
        # Преобразуем координаты мыши в координаты изображения через официальную функцию
        image_coords = mouse_to_image_coords(context, image_item, mouse_x, mouse_y, params)
        if not image_coords:
            logger.warning("Не удалось преобразовать координаты мыши в координаты изображения")
            # Пробуем ручное преобразование как запасной вариант
            pos_x = params.get('pos_x', 0)
            pos_y = params.get('pos_y', 0)
            
            # Ручное преобразование: обратное к формуле из image_to_region_coords
            image_x = (mouse_region_x - pos_x) / scale
            image_y = (mouse_region_y - pos_y) / scale
            
            image_coords = (image_x, image_y)
            logger.info(f"Используем ручное преобразование координат: {image_coords}")
        
        # Получаем координаты мыши в пространстве изображения
        image_x, image_y = image_coords
        
        logger.debug(f"Координаты мыши в регионе: ({mouse_region_x}, {mouse_region_y})")
        logger.debug(f"Абсолютные координаты мыши: ({mouse_x}, {mouse_y})")
        logger.debug(f"Преобразованные координаты в изображении: ({image_x}, {image_y})")
        
        # Корректируем максимальное расстояние с учетом масштаба изображения
        # При меньшем масштабе точки на экране ближе друг к другу
        # И требуется увеличить максимальное расстояние для удобства
        scaled_max_distance = max_distance / scale
        # Добавляем нижнюю границу для удобства
        scaled_max_distance = max(scaled_max_distance, 15.0)
        
        # Создаем временный вектор с координатами мыши в пространстве изображения
        mouse_image_pos = Vector((image_x, image_y))
        
        # Используем встроенный метод кривой для поиска ближайшей точки,
        # передавая координаты мыши в пространстве изображения
        return curve.find_closest_handle(mouse_image_pos, scaled_max_distance)
    
    except Exception as e:
        logger.error(f"Ошибка при поиске ближайшей точки: {e}")
        # Используем базовый метод как запасной вариант
        return curve.find_closest_handle(mouse_pos, max_distance)


def handle_bezier_mode_input(self, context, event, image_item, mouse_pos_image):
    """
    Обработка ввода пользователя в режиме работы с кривой Безье
    
    Args:
        context: Контекст Blender
        event: Событие ввода
        image_item: Текущий элемент изображения
        mouse_pos_image: Координаты мыши в пространстве изображения
    
    Returns:
        bool: True если ввод был обработан, иначе False
    """
    global current_bezier_state, active_handle_index, is_dragging_handle
    global active_drag_axis, drag_origin_position
    
    if not hasattr(event, 'mouse_region_x') or not hasattr(event, 'mouse_region_y'):
        logger.error("Событие не содержит координаты мыши region_x/region_y")
        return False
    
    props = context.scene.bezier_props
    
    # Проверяем, активен ли режим кривой Безье
    if props.placement_mode != 'BEZIER_CURVE':
        return False
    
    # Получаем параметры отображения изображения для корректного преобразования координат
    params = None
    try:
        import sys
        for module_name in sys.modules.keys():
            if module_name.endswith('image_editor'):
                module = sys.modules[module_name]
                if hasattr(module, 'get_image_display_params'):
                    params = module.get_image_display_params(context, image_item)
                    break
    except Exception as e:
        logger.error(f"Не удалось получить параметры отображения изображения: {e}")
    
    if params:
        logger.debug(f"Параметры отображения: pos_x={params.get('pos_x', 0)}, pos_y={params.get('pos_y', 0)}, scale={params.get('scale', 1.0)}")
    
    # ВАЖНОЕ ИЗМЕНЕНИЕ: Если это клик левой кнопкой мыши и активен режим кривой Безье,
    # всегда возвращаем True, чтобы блокировать установку обычных точек
    if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
        # Проверяем, находимся ли мы в режиме расстановки точек через свойство camera_calibration
        is_in_placing_mode = False
        if hasattr(context.scene, 'camera_calibration'):
            props_calib = context.scene.camera_calibration
            if hasattr(props_calib, 'is_placing_point'):
                is_in_placing_mode = props_calib.is_placing_point
        
        if not is_in_placing_mode:
            # Если мы не в режиме расстановки точек, просто блокируем событие
            logger.warning("Невозможно создать кривую: не в режиме расстановки точек")
            self.report({'WARNING'}, "Создание кривой доступно только в режиме расстановки точек")
            return True
        
        # Получаем менеджер кривых для дальнейшей обработки
        manager = get_bezier_manager()
        
        # Обработка событий в зависимости от состояния
        if current_bezier_state == BEZIER_STATE_NONE:
            # Создаем новую кривую
            curve = manager.new_curve()
            # Устанавливаем начальную точку
            mouse_pos_vec = Vector((event.mouse_region_x, event.mouse_region_y))
            
            # Если у нас есть параметры отображения, преобразуем координаты мыши в пространство изображения
            if params and mouse_pos_image:
                # Используем точно переданные координаты изображения
                image_x, image_y = mouse_pos_image
                
                # Проверка корректности координат - только для логирования
                image_width = getattr(image_item, 'width', 0) or getattr(image_item, 'size', [0, 0])[0] or 1600
                image_height = getattr(image_item, 'height', 0) or getattr(image_item, 'size', [0, 0])[1] or 1200
                
                if image_x < 0 or image_x > image_width * 1.5 or image_y < 0 or image_y > image_height * 1.5:
                    logger.warning(f"Координаты начальной точки вне типичных пределов: ({image_x}, {image_y})")
                    # Больше не корректируем координаты
                
                # Устанавливаем точно переданные координаты изображения
                logger.info(f"Установка начальной точки кривой в точных координатах: ({image_x}, {image_y})")
                curve.set_start_point(Vector((image_x, image_y)))
            else:
                # Если нет преобразованных координат, пытаемся преобразовать самостоятельно
                logger.warning("Нет преобразованных координат изображения для начальной точки, пытаемся преобразовать самостоятельно")
                
                try:
                    # Пытаемся найти функцию преобразования координат
                    import sys
                    mouse_to_image_coords = None
                    for module_name in sys.modules.keys():
                        if module_name.endswith('image_editor'):
                            module = sys.modules[module_name]
                            if hasattr(module, 'mouse_to_image_coords'):
                                mouse_to_image_coords = module.mouse_to_image_coords
                                break
                    
                    if mouse_to_image_coords and params:
                        # Преобразуем координаты мыши в координаты изображения
                        region = context.region
                        mouse_x = event.mouse_x  # Используем абсолютные координаты
                        mouse_y = event.mouse_y
                        image_coords = mouse_to_image_coords(context, image_item, mouse_x, mouse_y, params)
                        if image_coords:
                            image_x, image_y = image_coords
                            logger.info(f"Преобразованные координаты начальной точки: ({image_x}, {image_y})")
                            curve.set_start_point(Vector((image_x, image_y)))
                        else:
                            # Если не удалось преобразовать, используем подход с параметрами отображения
                            logger.warning("Не удалось преобразовать координаты начальной точки, используем прямой расчет")
                            pos_x = params.get('pos_x', 0)
                            pos_y = params.get('pos_y', 0)
                            scale = params.get('scale', 1.0)
                            
                            # Преобразование координат вручную
                            image_x = (event.mouse_region_x - pos_x) / scale
                            image_y = (event.mouse_region_y - pos_y) / scale
                            logger.info(f"Ручное преобразование координат начальной точки: ({image_x}, {image_y})")
                            curve.set_start_point(Vector((image_x, image_y)))
                    else:
                        # Если нет функции преобразования, используем координаты напрямую (как запасной вариант)
                        logger.warning("Не найдена функция mouse_to_image_coords для начальной точки, используем координаты напрямую")
                        curve.set_start_point(mouse_pos_vec)
                except Exception as e:
                    logger.error(f"Ошибка при преобразовании координат начальной точки: {e}")
                    # В случае ошибки используем координаты напрямую
                    curve.set_start_point(mouse_pos_vec)
            
            # Переходим в состояние ожидания второй точки
            set_bezier_state(BEZIER_STATE_FIRST_POINT)
            # Обновляем статус
            props.placement_status = "Установите конечную точку кривой"
            return True
        
        elif current_bezier_state == BEZIER_STATE_FIRST_POINT:
            curve = manager.get_active_curve()
            if curve:
                # Устанавливаем конечную точку
                mouse_pos_vec = Vector((event.mouse_region_x, event.mouse_region_y))
                
                # Если у нас есть параметры отображения, преобразуем координаты мыши в пространство изображения
                if params and mouse_pos_image:
                    # Используем напрямую преобразованные координаты изображения
                    image_x, image_y = mouse_pos_image
                    
                    # Проверка корректности координат - просто информативно
                    image_width = getattr(image_item, 'width', 0) or getattr(image_item, 'size', [0, 0])[0] or 1600
                    image_height = getattr(image_item, 'height', 0) or getattr(image_item, 'size', [0, 0])[1] or 1200
                    
                    if image_x < 0 or image_x > image_width * 1.5 or image_y < 0 or image_y > image_height * 1.5:
                        logger.warning(f"Координаты конечной точки вне типичных пределов: ({image_x}, {image_y})")
                        # Больше не меняем координаты, доверяем переданным значениям
                    
                    # Сохраняем точно переданные координаты для конечной точки
                    logger.info(f"Установка конечной точки кривой в точных координатах: ({image_x}, {image_y})")
                    curve.set_end_point(Vector((image_x, image_y)))
                else:
                    # Если нет преобразованных координат, пытаемся преобразовать самостоятельно
                    # Это запасной вариант, который должен использоваться редко
                    logger.warning("Нет преобразованных координат изображения, пытаемся преобразовать самостоятельно")
                    
                    try:
                        # Пытаемся найти функцию преобразования координат
                        import sys
                        mouse_to_image_coords = None
                        for module_name in sys.modules.keys():
                            if module_name.endswith('image_editor'):
                                module = sys.modules[module_name]
                                if hasattr(module, 'mouse_to_image_coords'):
                                    mouse_to_image_coords = module.mouse_to_image_coords
                                    break
                        
                        if mouse_to_image_coords and params:
                            # Преобразуем координаты мыши в координаты изображения
                            region = context.region
                            mouse_x = event.mouse_x  # Используем абсолютные координаты
                            mouse_y = event.mouse_y
                            image_coords = mouse_to_image_coords(context, image_item, mouse_x, mouse_y, params)
                            if image_coords:
                                image_x, image_y = image_coords
                                logger.info(f"Преобразованные координаты: ({image_x}, {image_y})")
                                curve.set_end_point(Vector((image_x, image_y)))
                            else:
                                # Если не удалось преобразовать, используем подход с параметрами отображения
                                logger.warning("Не удалось преобразовать координаты, используем прямой расчет")
                                pos_x = params.get('pos_x', 0)
                                pos_y = params.get('pos_y', 0)
                                scale = params.get('scale', 1.0)
                                
                                # Преобразование координат вручную
                                image_x = (event.mouse_region_x - pos_x) / scale
                                image_y = (event.mouse_region_y - pos_y) / scale
                                logger.info(f"Ручное преобразование координат: ({image_x}, {image_y})")
                                curve.set_end_point(Vector((image_x, image_y)))
                        else:
                            # Если нет функции преобразования, используем координаты напрямую (как запасной вариант)
                            logger.warning("Не найдена функция mouse_to_image_coords, используем координаты напрямую")
                            curve.set_end_point(mouse_pos_vec)
                    except Exception as e:
                        logger.error(f"Ошибка при преобразовании координат: {e}")
                        # В случае ошибки используем координаты напрямую
                        curve.set_end_point(mouse_pos_vec)
                
                # Устанавливаем количество контрольных точек и точек на кривой
                curve.set_num_handles(props.num_control_points)
                curve.set_num_points(props.num_curve_points)
                # Переходим в состояние редактирования
                set_bezier_state(BEZIER_STATE_EDIT)
                # Обновляем статус
                props.placement_status = (
                    "Редактирование кривой. Тяните точку свободно или за красную/зелёную ось для точного сдвига."
                )
            return True  # Всегда возвращаем True, даже если кривая не найдена
        
        elif current_bezier_state == BEZIER_STATE_EDIT:
            curve = manager.get_active_curve()
            if not curve:
                return True  # Всегда возвращаем True, чтобы блокировать обычные точки
            
            # Обрабатываем клик по контрольной точке
            mouse_pos_vec = Vector((event.mouse_region_x, event.mouse_region_y))
            
            # Логируем позицию клика для отладки
            logger.info(f"Клик в координатах региона: ({mouse_pos_vec.x}, {mouse_pos_vec.y})")
            
            gizmo_index, gizmo_axis, gizmo_distance = _find_gizmo_axis_hit(
                curve,
                mouse_pos_vec,
                context=context,
                image_item=image_item,
                params=params
            )

            if gizmo_index is not None and gizmo_axis is not None:
                _set_selected_control_point(curve, gizmo_index)
                active_handle_index = gizmo_index
                active_drag_axis = gizmo_axis
                is_dragging_handle = True
                drag_origin_position = curve.control_points[gizmo_index].get_position().copy()
                axis_label = 'X' if gizmo_axis == 'X' else 'Y'
                props.placement_status = (
                    f"Выбрана контрольная точка {gizmo_index}. Перетаскивание по оси {axis_label}."
                )
                logger.info(
                    f"Выбрана ось gizmo {gizmo_axis} для контрольной точки {gizmo_index}, "
                    f"расстояние={gizmo_distance}"
                )
                return True

            # Используем модифицированную функцию с коррекцией и преобразованием координат
            closest_index, distance = find_closest_handle_with_correction(
                curve, 
                mouse_pos_vec, 
                context=context, 
                image_item=image_item, 
                params=params
            )
            
            # Логируем результат поиска
            if closest_index is not None:
                control_point = curve.control_points[closest_index]
                control_pos = control_point.get_position()
            
            if closest_index is not None:
                _set_selected_control_point(curve, closest_index)
                active_handle_index = closest_index
                active_drag_axis = None
                is_dragging_handle = True
                drag_origin_position = curve.control_points[closest_index].get_position().copy()
                # Обновляем статус
                props.placement_status = f"Выбрана контрольная точка {closest_index}. Перетащите для изменения положения."
                logger.info(f"Выбрана контрольная точка {closest_index}")
            else:
                # Клик мимо точки
                active_handle_index = -1
                active_drag_axis = None
                drag_origin_position = None
                # Снимаем выделение со всех точек
                for point in curve.control_points:
                    point.selected = False
                
                logger.info(f"Клик мимо точки в координатах ({event.mouse_region_x}, {event.mouse_region_y})")
            
            return True  # Всегда возвращаем True, чтобы блокировать обычные точки
        
        # Для любых других состояний тоже блокируем создание обычных точек
        return True
    
    # Для режима кривой Безье обрабатываем другие события
    
    # Получаем менеджер кривых
    manager = get_bezier_manager()
    
    # Обработка событий в состоянии редактирования
    if current_bezier_state == BEZIER_STATE_EDIT:
        curve = manager.get_active_curve()
        if not curve:
            return False
        
        # Если нажата клавиша отмены, возвращаемся в начальное состояние
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            if not is_dragging_handle:
                # Удаляем кривую и возвращаемся в начальное состояние
                manager.delete_curve()
                set_bezier_state(BEZIER_STATE_NONE)
                # Обновляем статус
                props.placement_status = "Установите начальную точку кривой"
                logger.info("Отмена создания кривой, возврат в начальное состояние")
                return True
        
        # Если выбрана контрольная точка
        if active_handle_index >= 0:
            # Обработка перетаскивания
            if event.type == 'MOUSEMOVE' and is_dragging_handle:
                # Преобразуем координаты мыши в пространство изображения
                if params:
                    # Пытаемся получить преобразованные координаты изображения
                    image_coords = None
                    
                    # Если координаты уже преобразованы, используем их напрямую
                    if mouse_pos_image:
                        image_coords = mouse_pos_image
                        logger.debug(f"Используем переданные координаты изображения: {image_coords}")
                    else:
                        # Если нет, пытаемся найти функцию преобразования координат
                        try:
                            import sys
                            mouse_to_image_coords = None
                            for module_name in sys.modules.keys():
                                if module_name.endswith('image_editor'):
                                    module = sys.modules[module_name]
                                    if hasattr(module, 'mouse_to_image_coords'):
                                        mouse_to_image_coords = module.mouse_to_image_coords
                                        break
                            
                            if mouse_to_image_coords:
                                # Преобразуем абсолютные координаты мыши в координаты изображения
                                region = context.region
                                mouse_x = event.mouse_x
                                mouse_y = event.mouse_y
                                image_coords = mouse_to_image_coords(context, image_item, mouse_x, mouse_y, params)
                                logger.debug(f"Преобразованные координаты: {image_coords}")
                        except Exception as e:
                            logger.error(f"Ошибка при преобразовании координат мыши: {e}")
                    
                    # Если координаты получены, используем их для обновления точки
                    if image_coords:
                        image_x, image_y = image_coords
                        new_pos = Vector((image_x, image_y))
                        if drag_origin_position is not None:
                            if active_drag_axis == 'X':
                                new_pos.y = drag_origin_position.y
                            elif active_drag_axis == 'Y':
                                new_pos.x = drag_origin_position.x
                        
                        # Обновляем позицию контрольной точки
                        curve.update_control_point(active_handle_index, new_pos)
                        logger.info(f"Перемещение контрольной точки {active_handle_index} в координаты изображения: ({image_x}, {image_y})")
                    else:
                        # Если не удалось получить координаты, используем ручное преобразование
                        pos_x = params.get('pos_x', 0)
                        pos_y = params.get('pos_y', 0)
                        scale = params.get('scale', 1.0)
                        
                        # Преобразуем координаты региона в координаты изображения
                        image_x = (event.mouse_region_x - pos_x) / scale
                        image_y = (event.mouse_region_y - pos_y) / scale
                        new_pos = Vector((image_x, image_y))
                        if drag_origin_position is not None:
                            if active_drag_axis == 'X':
                                new_pos.y = drag_origin_position.y
                            elif active_drag_axis == 'Y':
                                new_pos.x = drag_origin_position.x
                        
                        # Обновляем позицию контрольной точки
                        curve.update_control_point(active_handle_index, new_pos)
                        logger.info(f"Перемещение контрольной точки {active_handle_index} с ручным преобразованием: ({image_x}, {image_y})")
                else:
                    # Если нет параметров отображения, используем координаты региона напрямую
                    mouse_pos_vec = Vector((event.mouse_region_x, event.mouse_region_y))
                    if drag_origin_position is not None:
                        if active_drag_axis == 'X':
                            mouse_pos_vec.y = drag_origin_position.y
                        elif active_drag_axis == 'Y':
                            mouse_pos_vec.x = drag_origin_position.x
                    curve.update_control_point(active_handle_index, mouse_pos_vec)
                    logger.info(f"Перемещение контрольной точки {active_handle_index} в координаты региона: {mouse_pos_vec}")
                
                return True
            
            # Завершение перетаскивания при отпускании кнопки мыши
            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and is_dragging_handle:
                is_dragging_handle = False
                active_drag_axis = None
                drag_origin_position = None
                # Обновляем статус
                props.placement_status = (
                    "Редактирование кривой. Тяните точку свободно или за красную/зелёную ось для точного сдвига."
                )
                logger.info(f"Завершение перетаскивания контрольной точки {active_handle_index}")
                return True
    
    # Для других типов событий возвращаем стандартное поведение
    return False


def draw_bezier_overlay(self, context, shader, image_item, params):
    """
    Отображение кривой Безье и контрольных точек на изображении
    
    Args:
        self: Экземпляр оператора редактора изображений
        context: Контекст Blender
        shader: Шейдер для отрисовки
        image_item: Элемент изображения
        params: Параметры отображения изображения
    """
    props = context.scene.bezier_props
    
    # Проверяем, активен ли режим кривой Безье
    if props.placement_mode != 'BEZIER_CURVE':
        return
    
    # Проверяем, находимся ли мы в режиме расстановки точек через свойство camera_calibration
    is_in_placing_mode = False
    if hasattr(context.scene, 'camera_calibration'):
        props_calib = context.scene.camera_calibration
        if hasattr(props_calib, 'is_placing_point'):
            is_in_placing_mode = props_calib.is_placing_point
    
    if not is_in_placing_mode:
        # Не отображаем кривую, если не в режиме расстановки точек
        return
    
    # Получаем менеджер кривых и активную кривую
    manager = get_bezier_manager()
    curve = manager.get_active_curve()
    
    # Если есть активная кривая, отображаем её
    if curve:
        draw_bezier_curve(shader, curve, image_item, context, params)


def update_bezier_hover_state(self, context, mouse_pos_image, mouse_pos_region=None):
    """
    Обновление состояния наведения курсора для контрольных точек кривой Безье
    
    Args:
        self: Экземпляр оператора
        context: Контекст Blender
        mouse_pos_image: Координаты мыши в пространстве изображения (могут быть None)
    
    Returns:
        bool: True если состояние наведения изменилось, иначе False
    """
    global hover_gizmo_axis, hover_gizmo_handle_index

    # Используем координаты мыши напрямую из события
    if mouse_pos_region is not None:
        mouse_region_x, mouse_region_y = mouse_pos_region
    elif hasattr(context, 'event') and hasattr(context.event, 'mouse_region_x') and hasattr(context.event, 'mouse_region_y'):
        mouse_region_x = context.event.mouse_region_x
        mouse_region_y = context.event.mouse_region_y
    else:
        # Если нет прямого доступа к координатам мыши, пытаемся использовать переданные координаты
        if not mouse_pos_image:
            hover_gizmo_axis = None
            hover_gizmo_handle_index = -1
            setattr(self, "_bezier_hover_handle_index", -1)
            setattr(self, "_bezier_hover_axis", None)
            return False
        mouse_region_x = mouse_pos_image[0]
        mouse_region_y = mouse_pos_image[1]
    
    # Получаем текущее изображение
    image_item = get_active_image_item(context)
    if not image_item:
        hover_gizmo_axis = None
        hover_gizmo_handle_index = -1
        setattr(self, "_bezier_hover_handle_index", -1)
        setattr(self, "_bezier_hover_axis", None)
        return False
    
    # Получаем параметры отображения изображения
    params = None
    try:
        # Пытаемся найти функцию для получения параметров
        import sys
        for module_name in sys.modules.keys():
            if module_name.endswith('image_editor'):
                module = sys.modules[module_name]
                if hasattr(module, 'get_image_display_params'):
                    params = module.get_image_display_params(context, image_item)
                    break
    except Exception as e:
        logger.error(f"Не удалось получить параметры отображения изображения: {e}")
    
    # Отладочная информация
    logger.debug(f"Обновление hover-состояния, координаты мыши: ({mouse_region_x}, {mouse_region_y})")
    
    props = context.scene.bezier_props
    
    # Проверяем, активен ли режим кривой Безье
    if props.placement_mode != 'BEZIER_CURVE':
        hover_gizmo_axis = None
        hover_gizmo_handle_index = -1
        setattr(self, "_bezier_hover_handle_index", -1)
        setattr(self, "_bezier_hover_axis", None)
        return False
    
    # Получаем менеджер кривых и активную кривую
    manager = get_bezier_manager()
    curve = manager.get_active_curve()
    
    # Проверяем, есть ли активная кривая
    if not curve:
        hover_gizmo_axis = None
        hover_gizmo_handle_index = -1
        setattr(self, "_bezier_hover_handle_index", -1)
        setattr(self, "_bezier_hover_axis", None)
        return False
    
    # Создаем Vector из координат мыши для сравнения с позициями точек
    mouse_pos_vec = Vector((mouse_region_x, mouse_region_y))
    
    # Логируем исходные координаты мыши и параметры
    if params:
        pos_x = params.get('pos_x', 0)
        pos_y = params.get('pos_y', 0)
        scale = params.get('scale', 1.0)
    
    gizmo_index, gizmo_axis, _ = _find_gizmo_axis_hit(
        curve,
        mouse_pos_vec,
        context=context,
        image_item=image_item,
        params=params
    )

    # Ищем ближайшую контрольную точку с коррекцией координат
    # Используем модифицированную функцию с передачей всех параметров
    closest_index, distance = find_closest_handle_with_correction(
        curve, 
        mouse_pos_vec, 
        context=context, 
        image_item=image_item, 
        params=params
    )

    effective_hover_index = gizmo_index if gizmo_index is not None else closest_index
    
    # Логируем результат поиска ближайшей точки
    if effective_hover_index is not None:
        control_point = curve.control_points[effective_hover_index]
        control_pos = control_point.get_position()
    else:
        pass
    
    # Обновляем состояние наведения на контрольных точках
    hover_changed = False
    for i, point in enumerate(curve.control_points):
        if i == effective_hover_index:
            if not point.hovered:
                point.hovered = True
                logger.debug(f"Точка {i} под курсором, расстояние: {distance}")
                hover_changed = True
        elif point.hovered:
            point.hovered = False
            hover_changed = True

    if hover_gizmo_axis != gizmo_axis or hover_gizmo_handle_index != (gizmo_index if gizmo_index is not None else -1):
        hover_changed = True

    hover_gizmo_axis = gizmo_axis
    hover_gizmo_handle_index = gizmo_index if gizmo_index is not None else -1
    setattr(self, "_bezier_hover_handle_index", effective_hover_index if effective_hover_index is not None else -1)
    setattr(self, "_bezier_hover_axis", gizmo_axis)
    
    return hover_changed


# Функция для получения функции преобразования координат из внешнего модуля
def get_image_to_region_coords_function(context):
    """
    Получение функции для преобразования координат изображения в координаты региона
    
    Returns:
        function: Функция преобразования координат
    """
    # Импортируем функцию из модуля image_editor
    try:
        # Используем правильный абсолютный импорт вместо относительного
        import sys
        import importlib
        
        # Пытаемся загрузить модуль из разных возможных мест
        module_name = None
        for module in sys.modules:
            if module.endswith('image_editor'):
                module_name = module
                break
        
        if module_name:
            module = sys.modules[module_name]
            if hasattr(module, 'image_to_region_coords'):
                # logger.info(f"Успешно импортирована функция image_to_region_coords из модуля {module_name}")
                return module.image_to_region_coords
        
        # Если не нашли через sys.modules, пробуем прямой импорт
        try:
            from image_editor import image_to_region_coords
            return image_to_region_coords
        except ImportError:
            pass
            
        logger.warning("Не удалось импортировать функцию преобразования координат")
    except Exception as e:
        logger.error(f"Ошибка при импорте функции преобразования координат: {e}")
    
    # Если не удалось импортировать, используем заглушку
    logger.warning("Используется заглушка для преобразования координат")
    
    def dummy_image_to_region_coords(context, image_item, x, y, params=None):
        # В заглушке проверяем наличие params, и используем их для корректного преобразования
        if params is not None:
            # Применяем формулу из оригинальной функции
            region_x = params['pos_x'] + x * params['scale']
            region_y = params['pos_y'] + y * params['scale']
            return (region_x, region_y)
        # Если params не предоставлены, возвращаем исходные координаты
        return x, y
    
    return dummy_image_to_region_coords


# Дополнительно добавляем функции для интеграции с редактором изображений
def get_active_image_item(context):
    """
    Получение активного элемента изображения
    
    Args:
        context: Контекст Blender
    
    Returns:
        image_item: Активный элемент изображения или None
    """
    # Проверяем наличие свойства camera_calibration в сцене
    if hasattr(context.scene, 'camera_calibration'):
        props_calib = context.scene.camera_calibration
        
        # Проверяем наличие active_image_index и images
        if hasattr(props_calib, 'active_image_index') and hasattr(props_calib, 'images'):
            active_image_index = props_calib.active_image_index
            if 0 <= active_image_index < len(props_calib.images):
                return props_calib.images[active_image_index]
            else:
                logger.warning(f"Неверный индекс активного изображения: {active_image_index}, доступно изображений: {len(props_calib.images)}")
        # Проверяем наличие active_image_idx и images (альтернативное имя свойства)
        elif hasattr(props_calib, 'active_image_idx') and hasattr(props_calib, 'images'):
            active_image_idx = props_calib.active_image_idx
            if 0 <= active_image_idx < len(props_calib.images):
                return props_calib.images[active_image_idx]
            else:
                logger.warning(f"Неверный индекс активного изображения: {active_image_idx}, доступно изображений: {len(props_calib.images)}")
        else:
            logger.warning("Не найдены атрибуты active_image_index/active_image_idx или images в camera_calibration")
    else:
        logger.warning("Не найдено свойство camera_calibration в context.scene")
    
    # Если не найден активный элемент через camera_calibration, пробуем через space_data
    if hasattr(context.space_data, 'image'):
        return context.space_data.image
    
    logger.error("Не удалось найти активное изображение никакими способами")
    return None


def add_bezier_points_to_active_image(context, points):
    """
    Добавление точек Безье к активному изображению
    
    Args:
        context: Контекст Blender
        points: Список точек в формате Vector или (x, y)
    
    Returns:
        int: Количество добавленных точек или -1 в случае ошибки
    """
    logger.info("------ Начало выполнения add_bezier_points_to_active_image ------")
    logger.info(f"Получено точек для добавления: {len(points) if points else 0}")
    
    if not points:
        logger.warning("Нет точек для добавления")
        logger.info("------ Завершение add_bezier_points_to_active_image: возврат 0 ------")
        return 0
    
    # Получаем активное изображение
    image_item = get_active_image_item(context)
    if not image_item:
        logger.error("Не найдено активное изображение")
        logger.info("------ Завершение add_bezier_points_to_active_image: возврат -1 ------")
        return -1
    
    # Проверяем исходные координаты точек для отладки
    logger.info("Исходные координаты точек (min/max):")
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    
    for point in points:
        if hasattr(point, 'x') and hasattr(point, 'y'):
            x, y = point.x, point.y
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[0], point[1]
        else:
            continue
        
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    
    logger.info(f"Диапазон X: {min_x} - {max_x}, диапазон Y: {min_y} - {max_y}")
    
    transformed_points = []  # Инициализируем список точек для добавления
    
    # ВАЖНАЯ МОДИФИКАЦИЯ: Всегда предполагаем, что точки находятся в пространстве региона
    # и требуют преобразования, так как это соответствует логике работы Bezier модуля
    are_points_in_image_space = False
    
    # Получаем размеры изображения для справки
    image_width = getattr(image_item, 'width', 0) or getattr(image_item, 'size', [0, 0])[0] or 1600
    image_height = getattr(image_item, 'height', 0) or getattr(image_item, 'size', [0, 0])[1] or 1200
    
    logger.info(f"Размеры изображения: {image_width}x{image_height}")
    logger.info("Точки будут преобразованы из пространства региона в пространство изображения")
    
    # Ищем функцию преобразования координат из region в image и параметры отображения
    try:
        import sys
        import importlib
        
        region_to_image_func = None
        params = None
        image_editor_module = None
        
        # Ищем модуль image_editor
        for module_name in sys.modules:
            if module_name.endswith('image_editor'):
                image_editor_module = sys.modules[module_name]
                logger.info(f"Найден модуль image_editor: {module_name}")
                break
        
        # Получаем функцию преобразования и параметры
        if image_editor_module:
            if hasattr(image_editor_module, 'mouse_to_image_coords'):
                region_to_image_func = image_editor_module.mouse_to_image_coords
                logger.info(f"Найдена функция mouse_to_image_coords в модуле {image_editor_module.__name__}")
            elif hasattr(image_editor_module, 'region_to_image_coords'):
                region_to_image_func = image_editor_module.region_to_image_coords
                logger.info(f"Найдена функция region_to_image_coords в модуле {image_editor_module.__name__}")
            
            if hasattr(image_editor_module, 'get_image_display_params'):
                params = image_editor_module.get_image_display_params(context, image_item)
                if params:
                    logger.info(f"Получены параметры отображения изображения: {params}")
                else:
                    logger.warning("Параметры отображения изображения не получены")
    except Exception as e:
        logger.warning(f"Ошибка при поиске функций преобразования координат: {e}")
        region_to_image_func = None
        params = None
    
    # Проверка и обработка параметров отображения
    if params:
        # Проверяем наличие всех необходимых параметров
        required_params = ['pos_x', 'pos_y', 'scale', 'image_width', 'image_height']
        missing_params = [param for param in required_params if param not in params]
        
        if missing_params:
            logger.warning(f"Отсутствуют необходимые параметры: {missing_params}")
        
        # Проверка корректности значения scale
        if 'scale' in params:
            # Корректируем подозрительные значения масштаба
            if params['scale'] <= 0:
                logger.warning(f"Недопустимое значение масштаба: {params['scale']}, устанавливаем значение по умолчанию")
                params['scale'] = 1.0
            elif params['scale'] > 100:
                logger.warning(f"Слишком большое значение масштаба: {params['scale']}, ограничиваем до 100")
                params['scale'] = min(params['scale'], 100)
            elif params['scale'] < 0.01:
                logger.warning(f"Слишком малое значение масштаба: {params['scale']}, устанавливаем минимальное значение")
                params['scale'] = 0.01
            
            # Логирование для отладки
            logger.info(f"Используется масштаб: {params['scale']}")
        
        # Проверяем наличие смещений offset_x и offset_y
        if 'offset_x' in params and 'offset_y' in params:
            logger.info(f"Обнаружены смещения: offset_x={params['offset_x']}, offset_y={params['offset_y']}")
    else:
        logger.warning("Не найдены параметры отображения изображения")
    
    # Определяем разницу в методах преобразования (если возможно)
    y_compensation = 0
    if region_to_image_func and params and len(points) > 0:
        test_point = points[0]
        if hasattr(test_point, 'x') and hasattr(test_point, 'y'):
            region_x, region_y = test_point.x, test_point.y
        elif isinstance(test_point, (list, tuple)) and len(test_point) >= 2:
            region_x, region_y = test_point[0], test_point[1]
        else:
            logger.warning(f"Неизвестный формат точки: {test_point}")
            region_x, region_y = 0, 0
        
        # Вычисляем координаты двумя способами и сравниваем
        # 1. Через функцию преобразования
        try:
            # Создаем фейковый event с координатами мыши
            class FakeEvent:
                def __init__(self, x, y):
                    self.mouse_x = x
                    self.mouse_y = y
                    
            # Получаем регион для корректного расчета
            region = context.region
            fake_event = FakeEvent(region_x + region.x, region_y + region.y)
            
            image_coords = region_to_image_func(context, image_item, fake_event.mouse_x, fake_event.mouse_y, params)
            if image_coords:
                image_x, image_y = image_coords
                
                # 2. Через ручной расчет по формуле
                image_x_manual = (region_x - params['pos_x']) / params['scale']
                image_y_manual = (region_y - params['pos_y']) / params['scale']
                
                # Вычисляем разницу для компенсации
                y_diff = image_y - image_y_manual
                
                # Если разница значительная, запоминаем её для компенсации
                if abs(y_diff) > 1.0:
                    y_compensation = y_diff
                    logger.info(f"Обнаружена разница в Y-координатах: {y_diff}, будет применена компенсация")
        except Exception as e:
            logger.warning(f"Ошибка при расчете компенсации: {e}")
    
    # Создаем преобразованные точки
    region = context.region
    
    # ПЕРВЫЙ МЕТОД: Используем найденную функцию преобразования
    if region_to_image_func and params:
        logger.info(f"Используем функцию {region_to_image_func.__name__} для преобразования координат с компенсацией Y: {y_compensation}")
        
        for point_pos in points:
            # Получаем координаты точки в пространстве региона
            if hasattr(point_pos, 'x') and hasattr(point_pos, 'y'):
                region_x, region_y = point_pos.x, point_pos.y
            elif isinstance(point_pos, (list, tuple)) and len(point_pos) >= 2:
                region_x, region_y = point_pos[0], point_pos[1]
            else:
                logger.warning(f"Неизвестный формат точки: {point_pos}")
                continue
            
            # Создаем фейковый event с координатами мыши
            class FakeEvent:
                def __init__(self, x, y):
                    self.mouse_x = x
                    self.mouse_y = y
                    
            # Абсолютные координаты для функции преобразования
            fake_event = FakeEvent(region_x + region.x, region_y + region.y)
            
            # Преобразуем координаты с использованием найденной функции
            image_coords = region_to_image_func(context, image_item, fake_event.mouse_x, fake_event.mouse_y, params)
            
            if image_coords:
                image_x, image_y = image_coords
                logger.info(f"Преобразование через функцию: ({region_x}, {region_y}) -> ({image_x}, {image_y})")
                transformed_points.append((image_x, image_y))
            else:
                # Если функция не смогла преобразовать, используем ручной метод
                image_x = (region_x - params['pos_x']) / params['scale']
                image_y = (region_y - params['pos_y']) / params['scale'] + y_compensation
                logger.info(f"Резервное преобразование: ({region_x}, {region_y}) -> ({image_x}, {image_y})")
                transformed_points.append((image_x, image_y))
    
    # ВТОРОЙ МЕТОД: Используем прямую формулу из image_editor.py с компенсацией
    elif params and 'pos_x' in params and 'pos_y' in params and 'scale' in params:
        logger.info(f"Используем прямую формулу из image_editor.py с компенсацией Y: {y_compensation}")
        
        for point_pos in points:
            # Получаем координаты точки в пространстве региона
            if hasattr(point_pos, 'x') and hasattr(point_pos, 'y'):
                region_x, region_y = point_pos.x, point_pos.y
            elif isinstance(point_pos, (list, tuple)) and len(point_pos) >= 2:
                region_x, region_y = point_pos[0], point_pos[1]
            else:
                logger.warning(f"Неизвестный формат точки: {point_pos}")
                continue
            
            # Преобразуем координаты по формуле из mouse_to_image_coords
            image_x = (region_x - params['pos_x']) / params['scale']
            image_y = (region_y - params['pos_y']) / params['scale'] + y_compensation
            
            logger.info(f"Преобразование по формуле: ({region_x}, {region_y}) -> ({image_x}, {image_y})")
            transformed_points.append((image_x, image_y))
    
    # ТРЕТИЙ МЕТОД: Резервный метод с масштабированием
    else:
        logger.warning("Используем резервный метод преобразования координат")
        
        # Определяем размеры изображения и области просмотра
        image_width = getattr(image_item, 'width', 0) or getattr(image_item, 'size', [0, 0])[0] or 1600
        image_height = getattr(image_item, 'height', 0) or getattr(image_item, 'size', [0, 0])[1] or 1200
        
        region_width = region.width if hasattr(region, 'width') else 1000
        region_height = region.height if hasattr(region, 'height') else 888
        
        logger.info(f"Размеры изображения: {image_width}x{image_height}, региона: {region_width}x{region_height}")
        
        # Улучшенная оценка параметров отображения на основе соотношения сторон
        aspect_ratio_img = image_width / image_height
        aspect_ratio_region = region_width / region_height
        
        # Определяем, какая сторона лимитирует (сохраняя соотношение сторон)
        if aspect_ratio_img > aspect_ratio_region:
            # Ширина ограничивает
            estimated_scale = region_width / (image_width * 1.1)  # Небольшой отступ
        else:
            # Высота ограничивает
            estimated_scale = region_height / (image_height * 1.1)  # Небольшой отступ
        
        # Ограничиваем масштаб разумными пределами
        estimated_scale = max(0.01, min(estimated_scale, 10.0))
        
        # Вычисляем позиции для центрирования изображения
        estimated_pos_x = (region_width - image_width * estimated_scale) / 2
        estimated_pos_y = (region_height - image_height * estimated_scale) / 2
        
        logger.info(f"Оценочные параметры: scale={estimated_scale}, pos_x={estimated_pos_x}, pos_y={estimated_pos_y}")
        
        # Если мы имеем точки кривой Безье, можем попытаться получить более точные параметры
        # на основе известных координат первой и последней точки
        if len(points) >= 2:
            # Если можем определить крайние точки в пространстве региона
            if all(hasattr(p, 'x') and hasattr(p, 'y') for p in [points[0], points[-1]]):
                first_point = points[0]
                last_point = points[-1]
                
                # Оценка на основе расстояния между точками
                region_distance = ((last_point.x - first_point.x)**2 + (last_point.y - first_point.y)**2)**0.5
                # Предполагаем, что точки должны быть на расстоянии не более 80% ширины изображения
                image_distance = image_width * 0.8
                
                # Корректируем масштаб на основе этой оценки
                distance_scale = region_distance / image_distance
                if 0.01 < distance_scale < 10.0:
                    logger.info(f"Корректируем масштаб на основе расстояния между точками: {distance_scale}")
                    estimated_scale = distance_scale
        
        for point_pos in points:
            if hasattr(point_pos, 'x') and hasattr(point_pos, 'y'):
                region_x, region_y = point_pos.x, point_pos.y
            elif isinstance(point_pos, (list, tuple)) and len(point_pos) >= 2:
                region_x, region_y = point_pos[0], point_pos[1]
            else:
                logger.warning(f"Неизвестный формат точки: {point_pos}")
                continue
            
            # Используем формулу с оценочными параметрами
            image_x = (region_x - estimated_pos_x) / estimated_scale
            image_y = (region_y - estimated_pos_y) / estimated_scale
            
            logger.info(f"Резервное преобразование: ({region_x}, {region_y}) -> ({image_x}, {image_y})")
            transformed_points.append((image_x, image_y))
    
    logger.info(f"Преобразовано {len(transformed_points)} точек для добавления")
    
    # Проверяем диапазоны преобразованных координат
    if transformed_points:
        min_x = min(p[0] for p in transformed_points)
        max_x = max(p[0] for p in transformed_points) 
        min_y = min(p[1] for p in transformed_points)
        max_y = max(p[1] for p in transformed_points)
        logger.info(f"Диапазон преобразованных точек - X: {min_x:.2f} - {max_x:.2f}, Y: {min_y:.2f} - {max_y:.2f}")
        
        # Проверяем, находятся ли точки в пределах изображения
        image_width = getattr(image_item, 'width', 0) or getattr(image_item, 'size', [0, 0])[0] or 1600
        image_height = getattr(image_item, 'height', 0) or getattr(image_item, 'size', [0, 0])[1] or 1200
        
        if min_x < 0 or max_x > image_width or min_y < 0 or max_y > image_height:
            logger.warning(f"Преобразованные точки выходят за пределы изображения ({image_width}x{image_height})")
    
    # Если не удалось преобразовать точки, возвращаем ошибку
    if not transformed_points:
        logger.error("Не удалось преобразовать ни одной точки")
        logger.info("------ Завершение add_bezier_points_to_active_image: возврат -1 ------")
        return -1
    
    added_count = 0
    
    try:
        # Проверка структуры объекта image_item
        has_points_collection = hasattr(image_item, 'points') and hasattr(image_item.points, 'add')
        has_location_2d = False
        has_co_2d = False
        
        # Проверяем, какой атрибут используется для хранения координат
        if has_points_collection and len(image_item.points) > 0:
            test_point = image_item.points[0]
            has_location_2d = hasattr(test_point, 'location_2d')
            has_co_2d = hasattr(test_point, 'co_2d')
        
        # Добавляем точки в зависимости от структуры
        if has_points_collection:
            for x, y in transformed_points:
                # Добавляем новую точку
                new_point = image_item.points.add()
                
                # Устанавливаем координаты в зависимости от структуры данных
                if has_location_2d:
                    new_point.location_2d = (x, y)
                    logger.debug(f"Установлены координаты для location_2d: {x}, {y}")
                elif has_co_2d:
                    new_point.co_2d[0] = x
                    new_point.co_2d[1] = y
                    logger.debug(f"Установлены координаты для co_2d: {x}, {y}")
                else:
                    # Пытаемся установить координаты напрямую, если это возможно
                    try:
                        new_point.x = x
                        new_point.y = y
                        logger.debug(f"Установлены координаты напрямую: {x}, {y}")
                    except Exception as e:
                        logger.error(f"Не удалось установить координаты для точки: {e}")
                        continue
                
                # Устанавливаем дополнительные атрибуты, если они есть
                if hasattr(new_point, 'is_placed'):
                    new_point.is_placed = True
                
                # Если есть свойство группы точек, устанавливаем активную группу
                if hasattr(new_point, 'point_group_id') and hasattr(context.scene.camera_calibration, 'active_point_group_index'):
                    new_point.point_group_id = context.scene.camera_calibration.active_point_group_index
                
                added_count += 1
            
            # Обновляем интерфейс
            for area in context.screen.areas:
                area.tag_redraw()
            
            image_name = getattr(image_item, 'name', 'unknown')
            logger.info(f"Добавлено {added_count} точек к изображению {image_name}")
            logger.info(f"------ Завершение add_bezier_points_to_active_image: возврат {added_count} ------")
            return added_count
        else:
            logger.error("У объекта изображения нет коллекции points или метода add")
            logger.error(f"Доступные атрибуты: {dir(image_item)}")
            logger.info("------ Завершение add_bezier_points_to_active_image: возврат -1 ------")
            return -1
    except Exception as e:
        logger.error(f"Ошибка при добавлении точек: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("------ Завершение add_bezier_points_to_active_image: возврат -1 из-за исключения ------")
        return -1


# Функции регистрации

def register():
    """Регистрация модуля Bezier UI"""
    initialize_bezier_manager()
    
    # Регистрация классов
    bpy.utils.register_class(BezierCurveProperties)
    
    # Отложенное добавление свойств через таймер
    def register_bezier_props():
        if hasattr(bpy.types, 'Scene'):
            # Добавляем свойства, если они еще не зарегистрированы
            if not hasattr(bpy.types.Scene, 'bezier_props'):
                bpy.types.Scene.bezier_props = bpy.props.PointerProperty(type=BezierCurveProperties)
                logger.info("Свойства Bezier добавлены к Scene")
            else:
                logger.info("Свойства Bezier уже зарегистрированы")
        else:
            logger.warning("bpy.types.Scene недоступен, отложим регистрацию")
            return 0.5  # Повторить через 0.5 секунд
            
        return None  # Не повторять таймер
    
    # Регистрируем таймер, если доступен
    if hasattr(bpy, 'app') and hasattr(bpy.app, 'timers'):
        bpy.app.timers.register(register_bezier_props, first_interval=0.5)
    else:
        # Если таймеры недоступны, пробуем прямую регистрацию с проверкой
        logger.warning("Таймеры bpy.app.timers недоступны, используем прямую регистрацию")
        if hasattr(bpy.types, 'Scene'):
            bpy.types.Scene.bezier_props = bpy.props.PointerProperty(type=BezierCurveProperties)
    
    logger.info("Модуль Bezier UI зарегистрирован")


def unregister():
    """Отмена регистрации модуля Bezier UI"""
    cleanup_bezier_manager()
    
    # Удаляем таймеры, если они есть
    if hasattr(bpy, 'app') and hasattr(bpy.app, 'timers'):
        for timer in list(bpy.app.timers.registered):
            if hasattr(timer, '__name__') and timer.__name__ == 'register_bezier_props':
                try:
                    bpy.app.timers.unregister(timer)
                except:
                    logger.warning("Не удалось удалить таймер register_bezier_props")
    
    # Отмена регистрации свойств
    if hasattr(bpy.types, 'Scene') and hasattr(bpy.types.Scene, 'bezier_props'):
        try:
            del bpy.types.Scene.bezier_props
            logger.info("Свойства Bezier удалены из Scene")
        except:
            logger.warning("Не удалось удалить свойства Bezier из Scene")
    
    # Отмена регистрации классов
    try:
        bpy.utils.unregister_class(BezierCurveProperties)
        logger.info("Класс BezierCurveProperties отменил регистрацию")
    except:
        logger.warning("Не удалось отменить регистрацию класса BezierCurveProperties")
    
    logger.info("Модуль Bezier UI отменил регистрацию")


def cleanup_bezier_manager():
    """Очистка менеджера кривых при выгрузке аддона"""
    if hasattr(bpy.types, "im_bezier_manager_prop"):
        del bpy.types.im_bezier_manager_prop
        if hasattr(bpy.types.Scene, "im_bezier_manager"):
            del bpy.types.Scene.im_bezier_manager
        logger.info("Менеджер кривых Безье удален")


def get_active_bezier_curve():
    """Получение активной кривой Безье
    
    Returns:
        BezierCurve: Активная кривая Безье или None, если нет активной кривой
    """
    manager = get_bezier_manager()
    if not manager:
        logger.warning("Менеджер кривых не инициализирован")
        return None
    
    return manager.get_active_curve()

def draw_callback_px(self, context):
    """
    Функция обратного вызова для отрисовки кривой Безье в режиме реального времени
    
    Args:
        self: Оператор Blender
        context: Контекст Blender
    """
    # Получаем активную кривую
    curve = get_active_bezier_curve()
    if not curve:
        return
    
    # Получаем активное изображение
    image_item = get_active_image(context)
    if not image_item:
        return
    
    # Создаем шейдер для отрисовки
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    # Параметры отрисовки
    params = {
        'region': context.region,
        'region_data': context.region_data,
        'v2d': context.region.view2d,
    }
    
    # Отрисовываем кривую
    draw_bezier_curve(shader, curve, image_item, context, params)
    


def get_active_image(context):
    """
    Получает активное изображение из контекста Blender
    
    Args:
        context: Контекст Blender
        
    Returns:
        Image: Активное изображение или None, если нет активного изображения
    """
    if context.area.type != 'IMAGE_EDITOR':
        return None
    
    return context.space_data.image


if __name__ == "__main__":
    register() 

