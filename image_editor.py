"""
Image Editor Module for Camera Calibration Add-on
Provides tools for placing and linking points on images for camera calibration
"""


try:
    import bpy
except:
    print("Использует тест без Блендер")
    pass

import numpy as np
import cv2
from bpy.types import Operator
from mathutils import Vector, Matrix
import gpu
import blf
from gpu_extras.batch import batch_for_shader
from gpu.types import GPUTexture
from . import calibration
from . import utils
from .properties import PREDEFINED_COLORS
from . import calibration_bridge
import os
import time
import math
import threading
import sys

# Импортируем модуль кривых Безье
try:
    from .bezier_module.bezier_ui import (
        initialize_bezier_manager,
        handle_bezier_mode_input,
        draw_bezier_overlay,
        update_bezier_hover_state,
        reset_bezier_state,
        get_bezier_manager,
    )
    bezier_module_imported = True
except ImportError:
    bezier_module_imported = False
    print("Внимание: модуль bezier_module не найден или не может быть импортирован")

# Определяем цвета для разных состояний точек
POINT_COLOR_UNASSIGNED = (0.2, 0.7, 1.0, 1.0)  # Голубой для точек без группы
POINT_COLOR_UNCALIBRATED = (1.0, 0.8, 0.0, 1.0)  # Желтый для точек в группе, но не откалиброванных
POINT_COLOR_CALIBRATED = (0.0, 0.8, 0.2, 1.0)  # Зеленый для откалиброванных точек
POINT_COLOR_FAILED = (1.0, 0.2, 0.2, 1.0)  # Красный для точек, которые не удалось откалибровать

# Добавляем новую глобальную переменную для кэширования данных изображений
_image_cache = {}

def get_gpu_texture(image):
    """Получить текстуру для GPU"""
    if image.gl_load():
        raise Exception("Не удалось загрузить изображение")
    
    # Создаем текстуру
    texture = gpu.texture.from_image(image)
    return texture

# Глобальные переменные для хранения состояния редактора
_editor_data = {
    "active_image_item": None,
    "mouse_x": 0,
    "mouse_y": 0,
    "is_magnifier_active": False,
    "magnifier_radius": 100,
    "magnifier_scale": 2.0,
    "calibration_mode": 'points',
    "active_point_index": -1,
    "is_dragging": False,
    "show_help": False
}

MIRROR_GROUP_SUFFIX = " Mirror"


def _find_point_group_index_by_name(props, group_name):
    """Ищет группу точек по имени."""
    target_name = str(group_name or "").strip()
    if not target_name:
        return -1

    for index, group in enumerate(getattr(props, "point_groups", [])):
        if str(getattr(group, "name", "")).strip() == target_name:
            return index
    return -1


def _get_paired_mirror_group_name(group_name):
    """Возвращает имя парной группы для зеркальной стороны."""
    base_name = str(group_name or "").strip()
    if not base_name:
        return MIRROR_GROUP_SUFFIX.strip()

    if base_name.endswith(MIRROR_GROUP_SUFFIX):
        original_name = base_name[:-len(MIRROR_GROUP_SUFFIX)].rstrip()
        return original_name or base_name

    return f"{base_name}{MIRROR_GROUP_SUFFIX}"


def _ensure_paired_mirror_group(props, source_group_id):
    """
    Возвращает id группы для зеркальной стороны.
    Base -> Base Mirror, Base Mirror -> Base.
    """
    if source_group_id is None or int(source_group_id) < 0:
        return -1

    source_group_id = int(source_group_id)
    if source_group_id >= len(props.point_groups):
        return -1

    source_group = props.point_groups[source_group_id]
    target_name = _get_paired_mirror_group_name(getattr(source_group, "name", ""))
    existing_index = _find_point_group_index_by_name(props, target_name)
    if existing_index >= 0:
        return existing_index

    new_group = props.point_groups.add()
    new_group.name = target_name
    color_index = (len(props.point_groups) - 1) % len(PREDEFINED_COLORS)
    new_group.color = PREDEFINED_COLORS[color_index]
    return len(props.point_groups) - 1

def draw_callback_px(op_self, context):
    """Функция отрисовки в области просмотра"""
    # Импортируем модули внутри функции для избежания проблем с областью видимости
    import math
    import time
    import gpu
    import blf
    import bpy
    from gpu_extras.batch import batch_for_shader
    
    try:
        props = context.scene.camera_calibration
        
        # Проверяем, что у нас есть активное изображение
        if len(props.images) == 0 or props.active_image_index < 0 or props.active_image_index >= len(props.images):
            print("Нет активного изображения для отображения")
            return
            
        # Получаем активное изображение
        image_item = props.images[props.active_image_index]
        
        # Отладочная информация о количестве точек
        points_count = sum(1 for p in image_item.points if p.is_placed)
        
        # Создаем параметры отображения изображения вместо вызова метода op_self.get_image_display_params
        params = get_image_display_params(context, image_item)
        if not params:
            print(f"Не удалось получить параметры отображения для {image_item.name}")
            return
        
        # Получаем размеры региона
        region = context.region
        region_width = region.width
        region_height = region.height
        
        # Создаем шейдер
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        
        # Запоминаем исходное состояние смешивания
        original_blend_mode = gpu.state.blend_get()
        
        # Включаем смешивание для прозрачности
        gpu.state.blend_set('ALPHA_PREMULT')
        
        # Отрисовка самого изображения
        image = None
        for img in bpy.data.images:
            if img.name == image_item.name:
                image = img
                break
        
        if image:
            # Создаем текстуру из изображения
            try:
                # Используем локальную функцию get_gpu_texture, определенную в начале файла
                texture = get_gpu_texture(image)
                
                # Создаем текстурный шейдер
                shader_tex = gpu.shader.from_builtin('IMAGE')
                
                # Создаем вершины для отрисовки текстуры
                vertices = ((params['pos_x'], params['pos_y']), 
                            (params['pos_x'] + params['width'], params['pos_y']),
                            (params['pos_x'] + params['width'], params['pos_y'] + params['height']),
                            (params['pos_x'], params['pos_y'] + params['height']))
                
                # Координаты текстуры (UV-координаты)
                # Для зеркальных изображений меняем порядок UV по горизонтали
                if hasattr(image_item, 'is_mirror') and image_item.is_mirror:
                    uvs = ((1, 0), (0, 0), (0, 1), (1, 1))
                else:
                    uvs = ((0, 0), (1, 0), (1, 1), (0, 1))
                
                # Создаем batch для отрисовки текстуры
                batch_tex = batch_for_shader(shader_tex, 'TRI_FAN', {
                    "pos": vertices,
                    "texCoord": uvs
                })
                
                # Привязываем текстуру и шейдер, затем отрисовываем
                shader_tex.bind()
                shader_tex.uniform_sampler("image", texture)
                batch_tex.draw(shader_tex)
                
            except Exception as e:
                print(f"Ошибка при отрисовке текстуры: {e}")
        
        # Рисуем рамку вокруг изображения
        vertices = [
            (params['pos_x'], params['pos_y']),
            (params['pos_x'] + params['width'], params['pos_y']),
            (params['pos_x'] + params['width'], params['pos_y'] + params['height']),
            (params['pos_x'], params['pos_y'] + params['height'])
        ]
        indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
        batch = batch_for_shader(shader, 'LINES', {"pos": vertices}, indices=indices)
        
        shader.bind()
        shader.uniform_float("color", (0.5, 0.5, 0.5, 0.5))
        batch.draw(shader)
        
        # Отрисовка самого изображения
        image = None
        for img in bpy.data.images:
            if img.name == image_item.name:
                image = img
                break
        
        if image:
            # Создаем текстуру из изображения
            try:
                # Используем локальную функцию get_gpu_texture, определенную в начале файла
                texture = get_gpu_texture(image)
                
                # Создаем текстурный шейдер
                shader_tex = gpu.shader.from_builtin('IMAGE')
                
                # Создаем вершины для отрисовки текстуры
                vertices = ((params['pos_x'], params['pos_y']), 
                            (params['pos_x'] + params['width'], params['pos_y']),
                            (params['pos_x'] + params['width'], params['pos_y'] + params['height']),
                            (params['pos_x'], params['pos_y'] + params['height']))
                
                # Координаты текстуры (UV-координаты)
                uvs = ((0, 0), (1, 0), (1, 1), (0, 1))
                
                # Создаем batch для отрисовки текстуры
                batch_tex = batch_for_shader(shader_tex, 'TRI_FAN', {
                    "pos": vertices,
                    "texCoord": uvs
                })
                
                # Привязываем текстуру и шейдер, затем отрисовываем
                shader_tex.bind()
                shader_tex.uniform_sampler("image", texture)
                batch_tex.draw(shader_tex)
                
            except Exception as e:
                print(f"Ошибка при отрисовке изображения: {str(e)}")
        
        # Определяем константы для цветов точек
        POINT_COLOR_UNASSIGNED = (0.2, 0.7, 1.0, 1.0)  # Голубой для точек без группы
        POINT_COLOR_UNCALIBRATED = (1.0, 0.8, 0.0, 1.0)  # Желтый для точек в группе, но не откалиброванных
        POINT_COLOR_CALIBRATED = (0.0, 0.8, 0.2, 1.0)  # Зеленый для откалиброванных точек
        POINT_COLOR_FAILED = (1.0, 0.2, 0.2, 1.0)  # Красный для точек, которые не удалось откалибровать
        
        # Статические данные об операторе из глобального словаря
        global _editor_data
        hover_point_index = _editor_data.get("active_point_index", -1)
        dragged_point_index = _editor_data.get("dragged_point_index", -1)
        show_help = _editor_data.get("show_help", False)
        is_magnifier_active = _editor_data.get("is_magnifier_active", False)
        mouse_x = _editor_data.get("mouse_x", 0)
        mouse_y = _editor_data.get("mouse_y", 0)
        
        # Рисуем точки
        for i, point in enumerate(image_item.points):
            if not point.is_placed:
                continue
                
            # Преобразуем координаты точки из координат изображения в координаты региона
            region_coords = image_to_region_coords(context, image_item, point.location_2d[0], point.location_2d[1], params)
            if not region_coords:
                continue
                
            x, y = region_coords
            
            # Определяем цвет точки в зависимости от группы и статуса калибровки
            if i == dragged_point_index:
                # Точка, которая сейчас перетаскивается
                color = (0.0, 1.0, 0.2, 1.0)  # Ярко-зелёный
                point_size = 12  # Увеличиваем размер перетаскиваемой точки
            elif point.point_group_id < 0:
                # Точка без группы
                color = POINT_COLOR_UNASSIGNED
                point_size = 7
            elif point.calibration_failed:
                # Точка, которую не удалось откалибровать
                color = POINT_COLOR_FAILED
                point_size = 7
            elif point.is_calibrated:
                # Успешно откалиброванная точка
                color = POINT_COLOR_CALIBRATED
                point_size = 7
            else:
                # Точка в группе, но не откалиброванная
                color = POINT_COLOR_UNCALIBRATED
                point_size = 7
            
            # Если это активная точка, рисуем её другим цветом
            if i == image_item.active_point_index:
                # Пульсирующий эффект для активной точки
                pulse = 0.5 + 0.5 * math.sin(time.time() * 5.0)
                color = (pulse, pulse, 1.0, 1.0)  # Голубой пульсирующий цвет
                point_size = 10
            
            # Если это точка под курсором, рисуем её с подсветкой
            if i == hover_point_index and i != dragged_point_index:
                point_size = 10
                # Рисуем ореол вокруг точки
                segments = 16
                circle_vertices = []
                radius = (point_size + 5) / 2
                for j in range(segments):
                    angle = 2 * math.pi * j / segments
                    circle_vertices.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
                
                # Рисуем ореол
                batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": circle_vertices})
                shader.bind()
                shader.uniform_float("color", (1.0, 1.0, 0.0, 0.5))
                batch.draw(shader)
            
            # Специальный эффект для перетаскиваемой точки
            if i == dragged_point_index:
                # Рисуем линию от точки до курсора
                shader.bind()
                drag_line_vertices = [(x, y), (mouse_x - region.x, mouse_y - region.y)]
                batch = batch_for_shader(shader, 'LINES', {"pos": drag_line_vertices})
                shader.uniform_float("color", (0.0, 1.0, 0.2, 0.7))  # Яркий зеленый
                batch.draw(shader)
                
                # Рисуем пульсирующий ореол вокруг перетаскиваемой точки
                segments = 32
                circle_vertices = []
                pulse = 0.5 + 0.5 * math.sin(time.time() * 8.0)  # Быстрая пульсация
                radius = (point_size + 8) * (0.8 + 0.2 * pulse)  # Пульсирующий размер
                
                for j in range(segments):
                    angle = 2 * math.pi * j / segments
                    circle_vertices.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
                
                # Рисуем ореол
                batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": circle_vertices})
                shader.bind()
                shader.uniform_float("color", (0.0, 1.0, 0.2, 0.3 + 0.3 * pulse))  # Пульсирующая прозрачность
                batch.draw(shader)
            
            # Рисуем крестик вместо круга
            cross_size = point_size
            cross_vertices = [
                # Горизонтальная линия
                (x - cross_size, y),
                (x + cross_size, y),
                # Вертикальная линия
                (x, y - cross_size),
                (x, y + cross_size)
            ]
            cross_indices = [(0, 1), (2, 3)]  # Определяем линии крестика
            
            # Рисуем крестик
            batch = batch_for_shader(shader, 'LINES', {"pos": cross_vertices}, indices=cross_indices)
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)
            
            # Рисуем номер точки
            font_id = 0  # По умолчанию используем встроенный шрифт
            blf.position(font_id, x + 10, y + 10, 0)
            blf.size(font_id, 12)  # Убираем третий параметр (DPI)
            blf.color(font_id, *color)
            
            # Отображаем номер группы вместо порядкового номера точки, если точка в группе
            point = image_item.points[i]
            if point.point_group_id >= 0:
                # Для точек в группе показываем номер группы + 1 (чтобы нумерация начиналась с 1)
                blf.draw(font_id, str(point.point_group_id + 1))
            else:
                # Для точек без группы показываем порядковый номер
                blf.draw(font_id, str(i + 1))
                    
        # Определяем высоту строки для текста
        line_height = 18
        ui_scale = context.preferences.system.ui_scale if hasattr(context.preferences.system, "ui_scale") else 1.0

        overlay_surface = (0.05, 0.07, 0.10, 0.88)
        overlay_surface_soft = (0.07, 0.09, 0.13, 0.92)
        overlay_border = (0.20, 0.33, 0.47, 0.95)
        overlay_accent = (0.19, 0.68, 1.00, 1.00)
        overlay_text = (0.94, 0.97, 1.00, 1.00)
        overlay_text_muted = (0.70, 0.80, 0.90, 1.00)
        overlay_key_fill = (0.10, 0.13, 0.18, 0.96)
        overlay_shadow = (0.0, 0.0, 0.0, 0.24)

        def scale_px(value):
            return value * ui_scale

        def apply_alpha(color, factor):
            if len(color) == 4:
                return (color[0], color[1], color[2], color[3] * factor)
            return (color[0], color[1], color[2], factor)

        def draw_overlay_card(x, y, width, height, fill_color=None, border_color=None, accent_color=None):
            fill_color = fill_color or overlay_surface
            border_color = border_color or overlay_border

            shadow_offset = scale_px(4)
            shadow_vertices = [
                (x + shadow_offset, y - shadow_offset),
                (x + width + shadow_offset, y - shadow_offset),
                (x + width + shadow_offset, y + height - shadow_offset),
                (x + shadow_offset, y + height - shadow_offset)
            ]
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": shadow_vertices})
            shader.bind()
            shader.uniform_float("color", overlay_shadow)
            batch.draw(shader)

            vertices = [
                (x, y),
                (x + width, y),
                (x + width, y + height),
                (x, y + height)
            ]
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": vertices})
            shader.bind()
            shader.uniform_float("color", fill_color)
            batch.draw(shader)

            if accent_color is not None:
                accent_height = max(2.0, scale_px(3))
                accent_vertices = [
                    (x, y + height - accent_height),
                    (x + width, y + height - accent_height),
                    (x + width, y + height),
                    (x, y + height)
                ]
                batch = batch_for_shader(shader, 'TRI_FAN', {"pos": accent_vertices})
                shader.bind()
                shader.uniform_float("color", accent_color)
                batch.draw(shader)

            gpu.state.line_width_set(1.5)
            batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": vertices})
            shader.bind()
            shader.uniform_float("color", border_color)
            batch.draw(shader)
            gpu.state.line_width_set(1.0)

        def get_overlay_top_margin():
            header_height = 0
            tool_header_height = 0
            area = getattr(context, "area", None)
            if area is not None:
                for area_region in area.regions:
                    if area_region.type == 'HEADER':
                        header_height = max(header_height, area_region.height)
                    elif area_region.type == 'TOOL_HEADER':
                        tool_header_height = max(tool_header_height, area_region.height)
            return max(scale_px(28), max(header_height, tool_header_height) + scale_px(14))

        def wrap_overlay_text(font_id, text, max_width):
            wrapped_lines = []
            for raw_line in str(text or "").splitlines():
                words = raw_line.split()
                if not words:
                    wrapped_lines.append("")
                    continue

                current_line = words[0]
                for word in words[1:]:
                    candidate = f"{current_line} {word}"
                    if blf.dimensions(font_id, candidate)[0] <= max_width:
                        current_line = candidate
                    else:
                        wrapped_lines.append(current_line)
                        current_line = word
                wrapped_lines.append(current_line)
            return wrapped_lines
            
        # Рисуем подсказки о цветах точек
        if points_count > 0 and show_help:  # Показываем подсказки только если есть точки И включен режим подсказок
            font_id = 0
            font_size = 12
            blf.size(font_id, font_size)
            
            # Отступы
            margin_x = 20
            margin_y = 20
            
            # Фон для легенды (полупрозрачный прямоугольник с закругленными углами)
            legend_width = 300
            legend_height = 100
            legend_x = 350 
            legend_y = region_height - legend_height - 100
            gpu.state.blend_set('ALPHA_PREMULT')
            draw_overlay_card(
                legend_x,
                legend_y,
                legend_width,
                legend_height,
                fill_color=overlay_surface_soft,
                border_color=overlay_border,
                accent_color=overlay_accent,
            )
            
            # Заголовок с подчеркиванием
            title_y = legend_y + legend_height - line_height
            blf.position(font_id, legend_x + 10, title_y, 0)
            blf.color(font_id, *overlay_text)
            title_text = "Обозначение цветов точек"
            blf.draw(font_id, title_text)
            
            # Подчеркивание заголовка
            underline_vertices = [
                (legend_x + 10, title_y - 5),
                (legend_x + 10 + blf.dimensions(font_id, title_text)[0], title_y - 5)
            ]
            batch = batch_for_shader(shader, 'LINES', {"pos": underline_vertices})
            shader.bind()
            shader.uniform_float("color", overlay_accent)
            batch.draw(shader)
            
            # Рисуем объяснения цветов с иконками вместо просто текста
            y_pos = legend_y + legend_height - 2*line_height - 10
            
            # Функция для рисования цветного квадрата перед текстом
            def draw_color_square(x, y, color, size=10):
                square_vertices = [
                    (x, y),
                    (x + size, y),
                    (x + size, y + size),
                    (x, y + size)
                ]
                batch = batch_for_shader(shader, 'TRI_FAN', {"pos": square_vertices})
                shader.bind()
                shader.uniform_float("color", color)
                batch.draw(shader)
                
                # Обводка квадрата
                batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": square_vertices})
                shader.bind()
                shader.uniform_float("color", (1.0, 1.0, 1.0, 0.5))
                batch.draw(shader)
            
            # Голубой - точки вне группы
            draw_color_square(legend_x + 15, y_pos, POINT_COLOR_UNASSIGNED)
            blf.position(font_id, legend_x + 35, y_pos, 0)
            blf.color(font_id, *overlay_text)
            blf.draw(font_id, "Точка вне группы")
            
            # Желтый - в группе, но не откалиброван
            y_pos -= line_height
            draw_color_square(legend_x + 15, y_pos, POINT_COLOR_UNCALIBRATED)
            blf.position(font_id, legend_x + 35, y_pos, 0)
            blf.color(font_id, *overlay_text)
            blf.draw(font_id, "В группе, не откалиброван")
            
            # Зеленый - успешно откалиброван
            y_pos -= line_height
            draw_color_square(legend_x + 15, y_pos, POINT_COLOR_CALIBRATED)
            blf.position(font_id, legend_x + 35, y_pos, 0)
            blf.color(font_id, *overlay_text)
            blf.draw(font_id, "Успешно откалиброван")
            
            # Красный - калибровка не удалась
            y_pos -= line_height
            draw_color_square(legend_x + 15, y_pos, POINT_COLOR_FAILED)
            blf.position(font_id, legend_x + 35, y_pos, 0)
            blf.color(font_id, *overlay_text)
            blf.draw(font_id, "Калибровка не удалась")
        
        # Рисуем легенду горячих клавиш
        if show_help:  # Проверяем флаг отображения подсказок
            font_id = 0
            font_size = 12
            blf.size(font_id, font_size)
            
            hotkey_sections = [
                ("Основное", [
                    ("ESC", "Закрыть редактор"),
                    ("H", "Показать или скрыть подсказки"),
                    ("C", "Калибровка / зеркальные точки"),
                ]),
                ("Разметка", [
                    ("ЛКМ", "Выбрать или добавить точку"),
                    ("A", "Режим добавления"),
                    ("L", "Режим связывания"),
                    ("DEL/X", "Удалить активную точку"),
                    ("R", "Убрать точку из группы"),
                    ("G", "Добавить новую группу"),
                    ("Z", "Лупа при ЛКМ в режиме точки"),
                ]),
                ("Навигация", [
                    ("СКМ", "Перемещение и масштаб"),
                    ("←/→", "Переключение изображений"),
                ]),
                ("Симметрия", [
                    ("M", "Режим симметрии"),
                    ("V / H", "Ориентация оси"),
                    ("S", "Сменить направление"),
                    ("←/→", "Сдвиг вертикальной оси"),
                    ("↑/↓", "Сдвиг горизонтальной оси"),
                ]),
            ]

            all_hotkeys = [item for _, items in hotkey_sections for item in items]
            item_font_size = max(11, int(12 * ui_scale))
            section_font_size = max(10, int(11 * ui_scale))
            title_font_size = max(13, int(14 * ui_scale))

            blf.size(font_id, item_font_size)
            key_box_width = 0.0
            max_description_width = 0.0
            for key, description in all_hotkeys:
                key_box_width = max(key_box_width, blf.dimensions(font_id, key)[0] + scale_px(18))
                max_description_width = max(max_description_width, blf.dimensions(font_id, description)[0])

            content_width = key_box_width + scale_px(14) + max_description_width
            hotkeys_legend_width = max(scale_px(360), content_width + scale_px(32))

            title_height = scale_px(30)
            section_row_height = scale_px(20)
            item_row_height = scale_px(21)
            section_gap = scale_px(8)
            content_height = scale_px(14)
            for section_index, (_, items) in enumerate(hotkey_sections):
                content_height += section_row_height
                content_height += len(items) * item_row_height
                if section_index != len(hotkey_sections) - 1:
                    content_height += section_gap
            content_height += scale_px(10)
            hotkeys_legend_height = title_height + content_height
            
            # Фиксированное положение нижнего края легенды
            hotkeys_legend_x = 20  # Размещаем в левом нижнем углу
            bottom_margin = 60  # Отступ от нижнего края экрана
            
            # Проверяем, не выходит ли легенда за верхний край экрана
            max_allowed_height = region_height - 40  # Максимальная допустимая высота (с отступом сверху)
            if hotkeys_legend_height > max_allowed_height:
                hotkeys_legend_height = max_allowed_height
            
            # Рассчитываем позицию нижнего края легенды
            hotkeys_legend_y = bottom_margin
            
            gpu.state.blend_set('ALPHA_PREMULT')
            draw_overlay_card(
                hotkeys_legend_x,
                hotkeys_legend_y,
                hotkeys_legend_width,
                hotkeys_legend_height,
                fill_color=overlay_surface_soft,
                border_color=overlay_border,
                accent_color=overlay_accent,
            )
            
            # Заголовок с подчеркиванием
            title_top_margin = scale_px(18)
            title_y = hotkeys_legend_y + hotkeys_legend_height - title_top_margin
            blf.position(font_id, hotkeys_legend_x + scale_px(12), title_y, 0)
            blf.color(font_id, *overlay_text)
            blf.size(font_id, title_font_size)
            title_text = "Горячие клавиши"
            title_text_width = blf.dimensions(font_id, title_text)[0]
            blf.draw(font_id, title_text)
            blf.size(font_id, item_font_size)
            
            # Подчеркивание заголовка
            underline_vertices = [
                (hotkeys_legend_x + scale_px(12), title_y - scale_px(6)),
                (hotkeys_legend_x + scale_px(12) + title_text_width, title_y - scale_px(6))
            ]
            batch = batch_for_shader(shader, 'LINES', {"pos": underline_vertices})
            shader.bind()
            shader.uniform_float("color", overlay_accent)
            batch.draw(shader)
            
            content_left = hotkeys_legend_x + scale_px(12)
            content_right = hotkeys_legend_x + hotkeys_legend_width - scale_px(12)
            y_pos = title_y - scale_px(24)
            content_clip_y = hotkeys_legend_y + scale_px(14)

            for section_index, (section_title, items) in enumerate(hotkey_sections):
                if y_pos < content_clip_y:
                    break

                blf.size(font_id, section_font_size)
                section_text = section_title.upper()
                section_width = blf.dimensions(font_id, section_text)[0]
                blf.position(font_id, content_left, y_pos, 0)
                blf.color(font_id, *overlay_text_muted)
                blf.draw(font_id, section_text)

                divider_start_x = content_left + section_width + scale_px(10)
                divider_y = y_pos + scale_px(6)
                if divider_start_x < content_right:
                    divider_vertices = [
                        (divider_start_x, divider_y),
                        (content_right, divider_y),
                    ]
                    batch = batch_for_shader(shader, 'LINES', {"pos": divider_vertices})
                    shader.bind()
                    shader.uniform_float("color", (overlay_border[0], overlay_border[1], overlay_border[2], 0.65))
                    batch.draw(shader)

                y_pos -= section_row_height
                blf.size(font_id, item_font_size)

                for key, description in items:
                    if y_pos < content_clip_y:
                        break

                    key_height = scale_px(17)
                    key_x = content_left
                    key_y = y_pos - scale_px(3)

                    key_vertices = [
                        (key_x, key_y),
                        (key_x + key_box_width, key_y),
                        (key_x + key_box_width, key_y + key_height),
                        (key_x, key_y + key_height)
                    ]
                    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": key_vertices})
                    shader.bind()
                    shader.uniform_float("color", overlay_key_fill)
                    batch.draw(shader)

                    batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": key_vertices})
                    shader.bind()
                    shader.uniform_float("color", overlay_border)
                    batch.draw(shader)

                    key_text_width = blf.dimensions(font_id, key)[0]
                    key_text_x = key_x + (key_box_width - key_text_width) / 2
                    blf.position(font_id, key_text_x, y_pos, 0)
                    blf.color(font_id, *overlay_accent)
                    blf.draw(font_id, key)

                    blf.position(font_id, key_x + key_box_width + scale_px(12), y_pos, 0)
                    blf.color(font_id, *overlay_text)
                    blf.draw(font_id, description)

                    y_pos -= item_row_height

                if section_index != len(hotkey_sections) - 1:
                    y_pos -= section_gap
        else:
            # Если подсказки скрыты, показываем минимальную подсказку о клавише H
            font_id = 0
            font_size = 12
            blf.size(font_id, font_size)
            
            # Рисуем полупрозрачный фон для подсказки
            hint_width = 250
            hint_height = 30  # Увеличиваем высоту для лучшей видимости
            hint_x = 20
            hint_y = 60  # Увеличиваем отступ от нижнего края экрана
            
            # Убедимся, что режим смешивания активен перед рисованием
            gpu.state.blend_set('ALPHA_PREMULT')
            
            draw_overlay_card(
                hint_x,
                hint_y,
                hint_width,
                hint_height,
                fill_color=overlay_surface_soft,
                border_color=overlay_border,
                accent_color=overlay_accent,
            )
            
            # Добавляем иконку клавиши H
            key_width = 20
            key_height = 20
            key_x = hint_x + 10
            key_y = hint_y + 5
            
            # Фон для клавиши
            key_vertices = [
                (key_x, key_y),
                (key_x + key_width, key_y),
                (key_x + key_width, key_y + key_height),
                (key_x, key_y + key_height)
            ]
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": key_vertices})
            shader.bind()
            shader.uniform_float("color", overlay_key_fill)
            batch.draw(shader)
            
            # Рамка для клавиши
            batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": key_vertices})
            shader.bind()
            shader.uniform_float("color", overlay_border)
            batch.draw(shader)
            
            # Текст клавиши H
            blf.position(font_id, key_x + 6, key_y + 5, 0)
            blf.color(font_id, *overlay_accent)
            blf.draw(font_id, "H")
            
            # Текст подсказки
            blf.position(font_id, key_x + key_width + 10, key_y + 5, 0)
            blf.color(font_id, *overlay_text)
            blf.draw(font_id, "Показать подсказки")

        notice_text = getattr(op_self, "_overlay_notification_text", "")
        notice_until = float(getattr(op_self, "_overlay_notification_expires_at", 0.0) or 0.0)
        if notice_text and notice_until > time.time():
            notice_level = str(getattr(op_self, "_overlay_notification_level", "INFO") or "INFO").upper()
            notice_style = {
                'INFO': {
                    'label': 'ИНФО',
                    'accent': overlay_accent,
                },
                'WARNING': {
                    'label': 'ВНИМАНИЕ',
                    'accent': (1.00, 0.76, 0.28, 1.0),
                },
                'ERROR': {
                    'label': 'ОШИБКА',
                    'accent': (1.00, 0.38, 0.38, 1.0),
                },
            }.get(notice_level, {
                'label': 'ИНФО',
                'accent': overlay_accent,
            })

            remaining_time = notice_until - time.time()
            fade_factor = 1.0 if remaining_time >= 0.35 else max(0.0, remaining_time / 0.35)
            notice_font_size = max(11, int(12 * ui_scale))
            notice_label_size = max(10, int(11 * ui_scale))
            notice_max_width = min(scale_px(560), region_width * 0.46)

            blf.size(font_id, notice_font_size)
            wrapped_notice_lines = wrap_overlay_text(font_id, notice_text, notice_max_width)
            if len(wrapped_notice_lines) > 3:
                wrapped_notice_lines = wrapped_notice_lines[:3]

            line_widths = [blf.dimensions(font_id, line)[0] for line in wrapped_notice_lines] or [0.0]
            max_line_width = max(line_widths)
            blf.size(font_id, notice_label_size)
            label_text = notice_style['label']
            label_width = blf.dimensions(font_id, label_text)[0]

            toast_padding_x = scale_px(16)
            toast_padding_y = scale_px(11)
            toast_gap = scale_px(5)
            toast_line_height = max(scale_px(16), notice_font_size + scale_px(3))
            toast_width = max(max_line_width, label_width) + 2 * toast_padding_x
            toast_height = toast_padding_y * 2 + scale_px(14) + toast_gap + len(wrapped_notice_lines) * toast_line_height
            toast_x = region_width / 2 - toast_width / 2
            toast_y = scale_px(56)

            accent_color = apply_alpha(notice_style['accent'], fade_factor)
            draw_overlay_card(
                toast_x,
                toast_y,
                toast_width,
                toast_height,
                fill_color=apply_alpha(overlay_surface_soft, fade_factor),
                border_color=apply_alpha(notice_style['accent'], 0.65 * fade_factor),
                accent_color=accent_color,
            )

            blf.size(font_id, notice_label_size)
            blf.position(font_id, toast_x + toast_padding_x, toast_y + toast_height - toast_padding_y - scale_px(12), 0)
            blf.color(font_id, *accent_color)
            blf.draw(font_id, label_text)

            blf.size(font_id, notice_font_size)
            text_y = toast_y + toast_height - toast_padding_y - scale_px(12) - toast_gap - toast_line_height
            for line in wrapped_notice_lines:
                blf.position(font_id, toast_x + toast_padding_x, text_y, 0)
                blf.color(font_id, *apply_alpha(overlay_text, fade_factor))
                blf.draw(font_id, line)
                text_y -= toast_line_height
        
        # Отображаем активный режим работы и подсказки
        # Отображаем информацию о текущем режиме в верхней части экрана
        font_id = 0
        font_size = 16
        blf.size(font_id, font_size)
        
        # Определяем режим работы
        mode_text = "Просмотр"
        mode_color = (0.60, 0.69, 0.80, 1.0)
        
        if props.is_placing_point:
            mode_text = "Добавление точек"
            mode_color = (0.11, 0.72, 1.00, 1.0)
        elif props.is_linking_points:
            mode_text = "Связывание точек"
            mode_color = (1.00, 0.76, 0.28, 1.0)
        elif op_self._mirror_mode_active:
            orientation = "вертикальная" if op_self._mirror_axis_orientation == 'vertical' else "горизонтальная"
            direction = "слева направо" if op_self._mirror_direction == 'left_to_right' else "справа налево"
            mode_text = f"Симметрия: {orientation}, {direction}"
            mode_color = (0.30, 0.86, 0.58, 1.0)
        elif hasattr(op_self, '_is_box_select_active') and op_self._is_box_select_active:
            mode_text = "Выделение рамкой"
            mode_color = (1.00, 0.58, 0.28, 1.0)
        
        # Добавляем информацию о количестве точек
        group_points = sum(1 for p in image_item.points if p.is_placed and p.point_group_id >= 0)
        free_points = sum(1 for p in image_item.points if p.is_placed and p.point_group_id < 0)
        
        meta_parts = []
        if props.active_point_group_index >= 0 and props.active_point_group_index < len(props.point_groups):
            group_name = props.point_groups[props.active_point_group_index].name
            meta_parts.append(f"Группа: {group_name}")
        
        meta_parts.append(f"{group_points} в группах")
        meta_parts.append(f"{free_points} свободных")
        mode_meta_text = "  •  ".join(meta_parts)

        title_font_size = max(14, int(17 * ui_scale))
        meta_font_size = max(11, int(12 * ui_scale))
        hint_font_size = max(10, int(11 * ui_scale))

        blf.size(font_id, title_font_size)
        title_width, title_height = blf.dimensions(font_id, mode_text)
        blf.size(font_id, meta_font_size)
        meta_width, meta_height = blf.dimensions(font_id, mode_meta_text)

        card_padding_x = scale_px(16)
        card_padding_top = scale_px(13)
        card_padding_bottom = scale_px(11)
        title_meta_gap = scale_px(5)

        mode_bg_width = max(title_width, meta_width) + 2 * card_padding_x
        mode_bg_height = title_height + meta_height + card_padding_top + card_padding_bottom + title_meta_gap
        mode_bg_x = region_width / 2 - mode_bg_width / 2
        mode_bg_y = region_height - get_overlay_top_margin() - mode_bg_height

        draw_overlay_card(
            mode_bg_x,
            mode_bg_y,
            mode_bg_width,
            mode_bg_height,
            fill_color=overlay_surface_soft,
            border_color=tuple((mode_color[i] if i < 3 else 1.0) * 0.75 for i in range(4)),
            accent_color=mode_color,
        )

        title_y = mode_bg_y + mode_bg_height - card_padding_top - title_height
        blf.size(font_id, title_font_size)
        blf.position(font_id, mode_bg_x + card_padding_x, title_y, 0)
        blf.color(font_id, *mode_color)
        blf.draw(font_id, mode_text)

        meta_y = mode_bg_y + card_padding_bottom
        blf.size(font_id, meta_font_size)
        blf.position(font_id, mode_bg_x + card_padding_x, meta_y, 0)
        blf.color(font_id, *overlay_text)
        blf.draw(font_id, mode_meta_text)
        
        # Отображаем подсказку по текущему режиму под заголовком режима
        hint_text = ""
        if props.is_placing_point:
            hint_text = "Кликните для добавления точки, Esc - отмена режима"
        elif props.is_linking_points:
            hint_text = "Кликните на точку для привязки к активной группе, Esc - отмена режима"
        elif op_self._mirror_mode_active:
            hint_text = "V/H - ориентация оси, стрелки - перемещение оси, S - направление, C - создать точки, Esc - выход"
        elif hasattr(op_self, '_is_box_select_active') and op_self._is_box_select_active:
            hint_text = "Зажмите ЛКМ и растяните рамку для выбора точек, Shift - добавление к выделению, Esc - выход"
        
        if hint_text:
            blf.size(font_id, hint_font_size)
            hint_width, hint_height = blf.dimensions(font_id, hint_text)
            hint_padding_x = scale_px(14)
            hint_padding_y = scale_px(8)
            hint_bg_width = hint_width + 2 * hint_padding_x
            hint_bg_height = hint_height + 2 * hint_padding_y
            hint_bg_x = region_width / 2 - hint_bg_width / 2
            hint_bg_y = mode_bg_y - hint_bg_height - scale_px(10)

            draw_overlay_card(
                hint_bg_x,
                hint_bg_y,
                hint_bg_width,
                hint_bg_height,
                fill_color=overlay_surface,
                border_color=(mode_color[0], mode_color[1], mode_color[2], 0.55),
                accent_color=None,
            )

            blf.position(font_id, hint_bg_x + hint_padding_x, hint_bg_y + hint_padding_y, 0)
            blf.color(font_id, *overlay_text_muted)
            blf.draw(font_id, hint_text)
        
        # После отрисовки существующих точек добавляем отрисовку симметрии
        if op_self._mirror_mode_active:
            # Очищаем предварительный список зеркальных точек
            op_self._mirror_point_preview = []
            
            # Отрисовываем ось симметрии
            op_self.draw_mirror_axis(context, shader, params)
            
            # Отрисовываем предпросмотр зеркальных точек
            op_self.draw_mirror_points_preview(context, shader, params)
        
        # Отрисовываем рамку выделения, если активен режим выделения рамкой
        if hasattr(op_self, '_is_box_selecting') and op_self._is_box_selecting:
            # Переключаемся на режим прозрачности для рамки выделения
            op_self.draw_selection_box(context, shader)
        
        # Отображаем выделенные точки (подсветка)
        if hasattr(op_self, '_selected_points') and op_self._selected_points:
            # Сохраняем режим смешивания для выделенных точек
            gpu.state.blend_set('ALPHA')
            
            # Цвет для выделенных точек
            selection_color = (1.0, 0.8, 0.0, 0.8)  # Желтый цвет для выделения
            
            # Сначала рисуем только подсветку для всех выделенных точек
            for point_idx in op_self._selected_points:
                if point_idx >= 0 and point_idx < len(image_item.points):
                    point = image_item.points[point_idx]
                    if not point.is_placed:
                        continue
                    
                    # Преобразуем координаты точки из координат изображения в координаты региона
                    region_coords = image_to_region_coords(context, image_item, point.location_2d[0], point.location_2d[1], params)
                    if not region_coords:
                        continue
                    
                    x, y = region_coords
                    
                    # Рисуем кружок подсветки вокруг точки
                    segments = 16
                    circle_vertices = []
                    radius = 15  # Радиус кружка подсветки
                    
                    for j in range(segments):
                        angle = 2 * math.pi * j / segments
                        circle_vertices.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
                    
                    # Рисуем заполненный кружок с полупрозрачностью
                    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": [(x, y)] + circle_vertices})
                    shader.bind()
                    shader.uniform_float("color", (selection_color[0], selection_color[1], selection_color[2], 0.2))
                    batch.draw(shader)
                
                    # Рисуем контур кружка
                    batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": circle_vertices})
                    shader.bind()
                    shader.uniform_float("color", selection_color)
                    batch.draw(shader)

        # Восстанавливаем исходный режим смешивания в конце отрисовки
        gpu.state.blend_set(original_blend_mode)
        
        # Восстанавливаем состояние blending
        gpu.state.blend_set('NONE')
        
        # Отображаем кривые Безье, если модуль доступен
        if bezier_module_imported:
            # Используем уже полученный image_item вместо вызова get_current_image_item
            if image_item:
                # Получаем параметры отображения
                params = get_image_display_params(context, image_item)
                
                # Создаем шейдер для отрисовки примитивов
                shader = gpu.shader.from_builtin('UNIFORM_COLOR')
                shader.bind()
                
                # Отображение кривой Безье и контрольных точек
                draw_bezier_overlay(op_self, context, shader, image_item, params)
        
        # Отображаем предпросмотр зеркальных точек, если включен режим двухсторонней калибровки
        props = context.scene.camera_calibration
        if props.use_mirror_calibration and props.mirror_preview_enabled and props.symmetry_plane_created:
            try:
                # Импортируем необходимые модули
                from .calibration import calibration_data
                from .calibration_modules import mirror_calibration
                
                # Проверяем наличие данных о плоскости симметрии
                if (calibration_data and 'mirror_data' in calibration_data and 
                    'plane_params' in calibration_data['mirror_data']):
                    
                    # Получаем параметры плоскости симметрии
                    plane_params = calibration_data['mirror_data']['plane_params']
                    
                    # Для каждой группы точек, создаем предпросмотр зеркальной точки
                    for i, group in enumerate(props.point_groups):
                        # Получаем 3D координаты точки
                        point_3d = np.array(group.location_3d)
                        
                        # Отражаем точку относительно плоскости симметрии
                        mirror_point_3d = mirror_calibration.reflect_point(point_3d, plane_params)
                        
                        # Находим точки этой группы на текущем изображении
                        for j, point in enumerate(image_item.points):
                            if point.is_placed and point.point_group_id == i:
                                # Проецируем отраженную 3D точку на изображение
                                if 'K' in calibration_data and props.active_image_index in calibration_data.get('cameras', {}):
                                    K = calibration_data['K']
                                    dist_coeffs = calibration_data.get('dist_coeffs', np.zeros(5))
                                    R, t = calibration_data['cameras'][props.active_image_index]
                                    
                                    # Проецируем точку на изображение
                                    mirror_point_3d_reshaped = mirror_point_3d.reshape(1, 3)
                                    projected_point, _ = cv2.projectPoints(
                                        mirror_point_3d_reshaped, 
                                        cv2.Rodrigues(R)[0], 
                                        t, 
                                        K, 
                                        dist_coeffs
                                    )
                                    
                                    # Получаем координаты на изображении
                                    img_x, img_y = projected_point.reshape(2)
                                    
                                    # Преобразуем координаты изображения в координаты региона
                                    region_coords = image_to_region_coords(context, image_item, img_x, img_y, params)
                                    if region_coords:
                                        x, y = region_coords
                                        
                                        # Рисуем зеркальную точку пунктирным кругом
                                        mirror_color = props.mirror_point_color
                                        shader.bind()
                                        shader.uniform_float("color", (mirror_color[0], mirror_color[1], mirror_color[2], 0.7))
                                        
                                        # Рисуем пунктирный круг
                                        import math
                                        gpu.state.line_width_set(1.5)
                                        
                                        # К сожалению, в более новых версиях Blender API для пунктирных линий изменился
                                        # Поэтому рисуем просто круг
                                        num_segments = 16
                                        radius = 8
                                        
                                        # Создаем вершины круга
                                        vertices = []
                                        for i in range(num_segments):
                                            angle = 2.0 * math.pi * i / num_segments
                                            vx = x + math.cos(angle) * radius
                                            vy = y + math.sin(angle) * radius
                                            vertices.append((vx, vy))
                                        
                                        # Рисуем круг
                                        batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": vertices})
                                        batch.draw(shader)
                                        
                                        # Рисуем текст с ID группы
                                        if props.show_point_ids:
                                            # Настраиваем шрифт
                                            blf.size(0, 12, 72)
                                            blf.color(0, mirror_color[0], mirror_color[1], mirror_color[2], 0.9)
                                            
                                            # Рисуем ID
                                            blf.position(0, x + radius + 5, y - 5, 0)
                                            blf.draw(0, f"M{i}")
            except Exception as e:
                print(f"Ошибка при отображении предпросмотра зеркальных точек: {str(e)}")
                import traceback
                traceback.print_exc()

        # Лупа рисуется поверх остальных элементов только в режиме постановки точки.
        if is_magnifier_active:
            draw_magnifier(context, shader, image_item, params, mouse_x, mouse_y)
        
    except Exception as e:
        # Обрабатываем ошибки при рендеринге
        print(f"Ошибка в draw_callback_px: {str(e)}")
        import traceback
        traceback.print_exc()
        
# Функции, которые заменят методы из класса
def get_image_display_params(context, image_item):
    """
    Получить параметры отображения изображения с использованием кэша.
    
    Args:
        context: Контекст Blender
        image_item: Элемент изображения из props.images
        
    Returns:
        dict: Словарь с параметрами отображения или None, если изображение не найдено
    """
    global _image_cache
    import os
    import bpy
    import time
    
    # Проверяем кэш
    if image_item.name in _image_cache:
        cache_data = _image_cache[image_item.name]
        image = cache_data['image']
        image_width = cache_data['width']
        image_height = cache_data['height']
        # Обновляем временную метку доступа
        cache_data['timestamp'] = time.time()
    else:
        # Если изображения нет в кэше, загружаем его стандартным способом
        image = None
        
        # Сначала пытаемся найти изображение по имени (это более надежно при сохранении/загрузке проекта)
        image = bpy.data.images.get(image_item.name)
        
        # Если это зеркальное изображение и мы его не нашли, попробуем найти оригинал
        if not image and image_item.name.startswith("Mirror_"):
            # Проверяем, связано ли оно с оригиналом
            original_name = image_item.name[7:]  # Убираем "Mirror_"
            image = bpy.data.images.get(original_name)
        
        # Если всё еще не нашли по имени, пробуем искать по пути (для обратной совместимости)
        if not image and hasattr(image_item, 'filepath') and image_item.filepath:
            for img in bpy.data.images:
                if os.path.normpath(img.filepath) == os.path.normpath(image_item.filepath):
                    image = img
                    break
            
        if not image:
            print(f"Не удалось получить параметры отображения для {image_item.name}")
            return None
        
        # Получаем размеры изображения
        image_width = image.size[0]
        image_height = image.size[1]
        
        # Добавляем в кэш
        _image_cache[image_item.name] = {
            'width': image_width,
            'height': image_height,
            'image': image,
            'timestamp': time.time()
        }
    
    if image_width == 0 or image_height == 0:
        return None
    
    # Получаем масштаб интерфейса для корректной работы на разных разрешениях экрана
    ui_scale = context.preferences.system.ui_scale if hasattr(context.preferences.system, "ui_scale") else 1.0
    dpi_fac = context.preferences.system.dpi / 72.0 if hasattr(context.preferences.system, "dpi") else 1.0
    
    # Получаем размеры региона (область в интерфейсе Blender)
    region = context.region
    region_width = region.width
    region_height = region.height
    
    # Определяем N-панель для последующих проверок, но не используем для масштабирования
    n_panel_width = 0
    for region_ui in context.area.regions:
        if region_ui.type == 'UI' and region_ui.width > 1:
            n_panel_width = region_ui.width
            break
    
    # Вычисляем масштаб и позицию изображения, используя полную ширину региона
    # независимо от размера N-панели
    base_scale = min((region_width * 0.8) / image_width, (region_height * 0.8) / image_height)
    scale = base_scale * image_item.scale
    
    width = image_width * scale
    height = image_height * scale
    
    # Позиционируем изображение по центру региона, без учета N-панели для масштаба
    # но сохраняем информацию о N-панели для корректной проверки нажатий
    pos_x = region_width / 2 - width / 2 + image_item.offset_x
    pos_y = region_height / 2 - height / 2 + image_item.offset_y
    
    return {
        'pos_x': pos_x,
        'pos_y': pos_y,
        'width': width,
        'height': height,
        'scale': scale,
        'offset_x': image_item.offset_x,
        'offset_y': image_item.offset_y,
        'image_width': image_width,
        'image_height': image_height,
        'ui_scale': ui_scale,
        'dpi_fac': dpi_fac,
        'n_panel_width': n_panel_width,
        'region_width': region_width,
        'region_height': region_height
    }

def image_to_region_coords(context, image_item, image_x, image_y, params=None):
    """
    Преобразует координаты изображения в координаты региона.
    
    Args:
        context: Контекст Blender
        image_item: Элемент изображения
        image_x: X-координата на изображении
        image_y: Y-координата на изображении
        params: Опциональные предварительно вычисленные параметры отображения
        
    Returns:
        tuple: (region_x, region_y) - координаты в регионе или None, если преобразование невозможно
    """
    # Получаем параметры отображения изображения, если они не переданы
    if params is None:
        params = get_image_display_params(context, image_item)
    if not params:
        return None
        
    # Проверяем UI масштаб для корректной работы на разных разрешениях экрана
    ui_scale = context.preferences.system.ui_scale if hasattr(context.preferences.system, "ui_scale") else 1.0
    dpi_fac = context.preferences.system.dpi / 72.0 if hasattr(context.preferences.system, "dpi") else 1.0
        
    # Преобразуем координаты изображения в координаты региона
    region_x = params['pos_x'] + image_x * params['scale']
    region_y = params['pos_y'] + image_y * params['scale']
    
    return (region_x, region_y)

def draw_magnifier(context, shader, image_item, params, mouse_x, mouse_y):
    """Отрисовка лупы в виде круга, увеличивающей область под курсором"""
    import gpu
    import blf
    import math
    import time
    import bpy
    from gpu_extras.batch import batch_for_shader
    
    # Получаем настройки лупы из _editor_data
    global _editor_data
    is_magnifier_active = _editor_data.get("is_magnifier_active", False)
    
    if not is_magnifier_active:
        return
    
    # Получаем UI масштаб для корректной работы на разных разрешениях экрана
    ui_scale = context.preferences.system.ui_scale if hasattr(context.preferences.system, "ui_scale") else 1.0
    dpi_fac = context.preferences.system.dpi / 72.0 if hasattr(context.preferences.system, "dpi") else 1.0
    
    # Адаптируем размер лупы к разрешению экрана
    magnifier_radius = 120  # Базовый радиус
    magnifier_radius = int(magnifier_radius * ui_scale * dpi_fac)
    
    # Устанавливаем масштаб изображения в лупе с учетом разрешения
    magnifier_scale = 4.0 * min(ui_scale * dpi_fac, 2.0)  # Ограничиваем максимальное увеличение
    
    # Получаем изображение для отрисовки его в лупе
    image = None
    for img in bpy.data.images:
        if img.name == image_item.name:
            image = img
            break
    
    if not image:
        return
    
    # Получаем размеры региона
    region = context.region
    region_width = region.width
    region_height = region.height
    
    # Проверяем, что курсор находится в пределах региона
    mouse_region_x = mouse_x - region.x
    mouse_region_y = mouse_y - region.y
    if not (0 <= mouse_region_x <= region_width and 0 <= mouse_region_y <= region_height):
        return
    
    # Получаем текущий режим работы
    props = context.scene.camera_calibration
    is_placing_point = props.is_placing_point
    is_dragging_point = _editor_data.get("is_dragging", False)
    
    # Настройки лупы
    magnifier_radius = _editor_data.get("magnifier_radius", 120)
    magnifier_scale = _editor_data.get("magnifier_scale", 4.0)
    
    # Координаты центра лупы в координатах региона.
    center_x = mouse_region_x
    center_y = mouse_region_y
    
    # Координаты левого нижнего угла лупы
    magnifier_x = center_x - magnifier_radius
    magnifier_y = center_y - magnifier_radius
    
    # Получаем координаты точки в координатах изображения
    image_coords = mouse_to_image_coords(context, image_item, mouse_x, mouse_y, params)
    if not image_coords:
        return
    
    image_x, image_y = image_coords
    
    # Проверка, что точка находится на изображении
    if not (0 <= image_x < params['image_width'] and 0 <= image_y < params['image_height']):
        return
    
    # Убедимся, что режим смешивания активен перед рисованием
    gpu.state.blend_set('ALPHA_PREMULT')
    
    # Создаем круглую форму для лупы
    segments = 32  # Количество сегментов круга для плавности
    circle_vertices = []
    circle_indices = []
    
    # Создаем вершины для края круга
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        x = center_x + magnifier_radius * math.cos(angle)
        y = center_y + magnifier_radius * math.sin(angle)
        circle_vertices.append((x, y))
    
    # Создаем индексы для рисования линий круга
    for i in range(segments):
        circle_indices.append((i, (i + 1) % segments))
    
    # Создаем вершины для заполнения круга
    fill_vertices = [(center_x, center_y)]  # Центр круга
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        x = center_x + magnifier_radius * math.cos(angle)
        y = center_y + magnifier_radius * math.sin(angle)
        fill_vertices.append((x, y))
    
    # Создаем индексы для заполнения круга треугольниками
    fill_indices = []
    for i in range(1, segments):
        fill_indices.append((0, i, i + 1))
    fill_indices.append((0, segments, 1))  # Замыкаем круг
    
    # Рисуем фон лупы (темнее в режиме расстановки или перетаскивания для контраста)
    bg_color = (0.1, 0.1, 0.12, 0.9)
    if is_placing_point:
        # Слегка голубоватый фон в режиме расстановки
        bg_color = (0.1, 0.15, 0.2, 0.9)
    
    batch = batch_for_shader(shader, 'TRIS', {"pos": fill_vertices}, indices=fill_indices)
    shader.bind()
    shader.uniform_float("color", bg_color)
    batch.draw(shader)
    
    # Отрисовываем текстуру изображения в лупе
    try:
        # Получаем текстуру
        texture = get_gpu_texture(image)
        
        if texture:
            # Создаем текстурный шейдер
            shader_tex = gpu.shader.from_builtin('IMAGE')
            
            # Размер области изображения, которая будет показана в лупе
            view_size = (magnifier_radius * 2) / (params['scale'] * magnifier_scale)
            half_view = view_size / 2
            
            # Координаты изображения для увеличенной области
            min_img_x = max(0, image_x - half_view)
            min_img_y = max(0, image_y - half_view)
            max_img_x = min(params['image_width'], image_x + half_view)
            max_img_y = min(params['image_height'], image_y + half_view)
            
            # Нормализованные UV координаты для текстуры
            min_u = min_img_x / params['image_width']
            min_v = min_img_y / params['image_height']
            max_u = max_img_x / params['image_width']
            max_v = max_img_y / params['image_height']
            
            # Создаем UV-координаты для текстуры по кругу
            tex_vertices = [(center_x, center_y)]  # Центр круга
            tex_uvs = [(min_u + (max_u - min_u) / 2, min_v + (max_v - min_v) / 2)]  # Центр текстуры
            
            for i in range(segments + 1):
                angle = 2.0 * math.pi * i / segments
                cos_val = math.cos(angle)
                sin_val = math.sin(angle)
                
                # Координаты вершины по кругу
                x = center_x + magnifier_radius * cos_val
                y = center_y + magnifier_radius * sin_val
                
                # UV-координаты, смещенные от центра для правильного масштабирования
                u = min_u + (max_u - min_u) / 2 + (max_u - min_u) / 2 * cos_val
                v = min_v + (max_v - min_v) / 2 + (max_v - min_v) / 2 * sin_val
                
                tex_vertices.append((x, y))
                tex_uvs.append((u, v))
            
            # Создаем индексы для отрисовки текстуры треугольниками
            tex_indices = []
            for i in range(1, segments):
                tex_indices.append((0, i, i + 1))
            tex_indices.append((0, segments, 1))  # Замыкаем круг
            
            # Создаем batch для отрисовки текстуры
            batch_tex = batch_for_shader(shader_tex, 'TRIS', {
                "pos": tex_vertices,
                "texCoord": tex_uvs
            }, indices=tex_indices)
            
            # Привязываем текстуру и шейдер, затем отрисовываем
            shader_tex.bind()
            shader_tex.uniform_sampler("image", texture)
            batch_tex.draw(shader_tex)
    except Exception as e:
        print(f"Ошибка при отрисовке текстуры в лупе: {str(e)}")
    
    # Отображаем точки в лупе
    try:
        # Отображаем точки, которые попадают в область лупы
        for i, point in enumerate(image_item.points):
            if not point.is_placed:
                continue
                
            # Преобразуем координаты точки из координат изображения в координаты региона
            region_coords = image_to_region_coords(context, image_item, point.location_2d[0], point.location_2d[1], params)
            if not region_coords:
                continue
                
            x, y = region_coords
            
            # Проверяем, что точка находится внутри круга лупы
            dx = x - center_x
            dy = y - center_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance <= magnifier_radius:
                # Определяем цвет точки в зависимости от группы и статуса калибровки
                if point.point_group_id < 0:
                    # Точка без группы
                    color = (0.2, 0.7, 1.0, 1.0)  # POINT_COLOR_UNASSIGNED
                elif point.calibration_failed:
                    # Точка, которую не удалось откалибровать
                    color = (1.0, 0.2, 0.2, 1.0)  # POINT_COLOR_FAILED
                elif point.is_calibrated:
                    # Успешно откалиброванная точка
                    color = (0.0, 0.8, 0.2, 1.0)  # POINT_COLOR_CALIBRATED
                else:
                    # Точка в группе, но не откалиброванная
                    color = (1.0, 0.8, 0.0, 1.0)  # POINT_COLOR_UNCALIBRATED
                
                # Если это активная точка, рисуем её другим цветом
                if i == image_item.active_point_index:
                    # Пульсирующий эффект для активной точки
                    pulse = 0.5 + 0.5 * math.sin(time.time() * 5.0)
                    color = (pulse, pulse, 1.0, 1.0)  # Голубой пульсирующий цвет
                
                # Масштабированные координаты точки в лупе
                # Проблема в том, что мы применяем масштабирование к смещению точки от центра лупы
                # Но нам нужно правильно перенести точку из её исходной позиции в увеличенную область
                
            # Координаты курсора в координатах изображения - центр лупы.
            image_center = mouse_to_image_coords(
                context,
                image_item,
                mouse_x,
                mouse_y,
                params,
            )
            if not image_center:
                continue
                    
                # Вычисляем смещение точки от центра в координатах изображения
                img_dx = point.location_2d[0] - image_center[0]
                img_dy = point.location_2d[1] - image_center[1]
                
                # Применяем масштабирование к этому смещению и добавляем к центру лупы
                lupe_x = center_x + img_dx * magnifier_scale
                lupe_y = center_y + img_dy * magnifier_scale
                
                # Рисуем крестик в лупе
                cross_size = 10  # Размер крестика в лупе
                cross_vertices = [
                    # Горизонтальная линия
                    (lupe_x - cross_size, lupe_y),
                    (lupe_x + cross_size, lupe_y),
                    # Вертикальная линия
                    (lupe_x, lupe_y - cross_size),
                    (lupe_x, lupe_y + cross_size)
                ]
                cross_indices = [(0, 1), (2, 3)]  # Определяем линии крестика
                
                # Рисуем крестик
                batch = batch_for_shader(shader, 'LINES', {"pos": cross_vertices}, indices=cross_indices)
                shader.bind()
                shader.uniform_float("color", color)
                batch.draw(shader)
                
                # Добавляем отображение номера точки в лупе
                font_id = 0
                blf.position(font_id, lupe_x + 12, lupe_y + 12, 0)
                blf.size(font_id, 12)
                blf.color(font_id, *color)
                
                # Отображаем номер группы или порядковый номер
                if point.point_group_id >= 0:
                    # Для точек в группе показываем номер группы + 1 (нумерация с 1)
                    blf.draw(font_id, str(point.point_group_id + 1))
                else:
                    # Для точек без группы показываем порядковый номер
                    blf.draw(font_id, str(i + 1))
    except Exception as e:
        print(f"Ошибка при отрисовке точек в лупе: {str(e)}")
    
    # Определяем цвет границы в зависимости от режима
    border_color = (0.7, 0.8, 0.9, 1.0)  # Обычный режим - голубоватый
    if is_placing_point:
        # В режиме расстановки точек - яркий голубой
        border_color = (0.3, 0.8, 1.0, 1.0)
    elif is_dragging_point:
        # В режиме перетаскивания - желтоватый
        border_color = (1.0, 0.9, 0.3, 1.0)
    
    # Рисуем границу круга с большей толщиной
    gpu.state.line_width_set(2.0)  # Увеличиваем толщину линии для лучшей видимости
    batch = batch_for_shader(shader, 'LINES', {"pos": circle_vertices}, indices=circle_indices)
    shader.bind()
    shader.uniform_float("color", border_color)
    batch.draw(shader)
    gpu.state.line_width_set(1.0)  # Возвращаем стандартную толщину
    
    # Добавляем перекрестие в центре
    cross_size = 10
    cross_color = (1.0, 0.0, 0.0, 0.8)  # Красный по умолчанию
    
    # В режиме добавления точек делаем перекрестие более заметным
    if is_placing_point:
        cross_size = 15
        cross_color = (1.0, 0.3, 0.3, 1.0)  # Более яркий красный
    
    cross_vertices = [
        (center_x - cross_size, center_y),
        (center_x + cross_size, center_y),
        (center_x, center_y - cross_size),
        (center_x, center_y + cross_size)
    ]
    cross_indices = [(0, 1), (2, 3)]
    
    gpu.state.line_width_set(2.0)  # Увеличиваем толщину для перекрестия
    batch = batch_for_shader(shader, 'LINES', {"pos": cross_vertices}, indices=cross_indices)
    shader.bind()
    shader.uniform_float("color", cross_color)
    batch.draw(shader)
    
    # Добавляем маленький кружок в центр для точного позиционирования
    circle_radius = 3
    circle_segments = 16
    center_circle_vertices = []
    
    for i in range(circle_segments):
        angle = 2.0 * math.pi * i / circle_segments
        x = center_x + circle_radius * math.cos(angle)
        y = center_y + circle_radius * math.sin(angle)
        center_circle_vertices.append((x, y))
    
    center_circle_indices = []
    for i in range(circle_segments):
        center_circle_indices.append((i, (i + 1) % circle_segments))
    
    batch = batch_for_shader(shader, 'LINES', {"pos": center_circle_vertices}, indices=center_circle_indices)
    shader.bind()
    shader.uniform_float("color", (1.0, 0.3, 0.3, 1.0))  # Яркий красный для центра
    batch.draw(shader)
    
    gpu.state.line_width_set(1.0)  # Возвращаем стандартную толщину
    
    # Добавляем заголовок с увеличением
    font_id = 0
    blf.size(font_id, 12)
    text = f"{int(magnifier_scale)}x"
    if is_placing_point:
        text = f"{int(magnifier_scale)}x (режим добавления)"
    elif is_dragging_point:
        text = f"{int(magnifier_scale)}x (перемещение)"
    
    text_width, text_height = blf.dimensions(font_id, text)
    blf.position(font_id, center_x - text_width / 2, center_y + magnifier_radius - 20, 0)
    blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
    blf.draw(font_id, text)

def mouse_to_image_coords(context, image_item, mouse_x, mouse_y, params=None):
    """
    Преобразует координаты мыши в координаты изображения.
    
    Args:
        context: Контекст Blender
        image_item: Элемент изображения
        mouse_x: X-координата мыши (абсолютная)
        mouse_y: Y-координата мыши (абсолютная)
        params: Опциональные предварительно вычисленные параметры отображения
        
    Returns:
        tuple: (image_x, image_y) - координаты на изображении или None, если преобразование невозможно
    """
    # Получаем параметры отображения изображения, если они не переданы
    if params is None:
        params = get_image_display_params(context, image_item)
    if not params:
        return None
        
    # Проверяем UI масштаб для корректной работы на разных разрешениях экрана
    ui_scale = context.preferences.system.ui_scale if hasattr(context.preferences.system, "ui_scale") else 1.0
    dpi_fac = context.preferences.system.dpi / 72.0 if hasattr(context.preferences.system, "dpi") else 1.0
    
    # Переводим абсолютные координаты мыши в координаты относительно региона
    region = context.region
    mouse_region_x = mouse_x - region.x
    mouse_region_y = mouse_y - region.y

    # Преобразуем координаты мыши в координаты изображения
    # Учитываем масштаб UI и DPI
    image_x = (mouse_region_x - params['pos_x']) / params['scale']
    image_y = (mouse_region_y - params['pos_y']) / params['scale']
    
    return (image_x, image_y)

class CAMCALIB_line:
    """Класс для хранения информации о линии"""
    def __init__(self):
        self.start = None
        self.end = None
        self.direction = None  # 0 - X, 1 - Y, 2 - Z
        self.is_complete = False

class CAMCALIB_OT_image_editor(Operator):
    """Редактор изображений для работы с опорными точками"""
    bl_idname = "camera_calibration.image_editor"
    bl_label = "Редактор изображений"
    bl_options = {'REGISTER', 'UNDO'}  # Добавляем поддержку undo и register
    
    _mouse_x: int = 0
    _mouse_y: int = 0
    _is_dragging: bool = False
    _is_mmb_dragging: bool = False
    _drag_start_x: int = 0
    _drag_start_y: int = 0
    _initial_offset: Vector = None
    _is_magnifier_active: bool = False
    _magnifier_radius: int = 120  # Увеличиваем радиус для более удобного просмотра
    _magnifier_scale: float = 4.0  # Увеличиваем масштаб для лучшей детализации
    _is_point_dragging: bool = False
    _dragged_point_index: int = -1
    _min_point_distance: float = 25.0  # Значительно увеличиваем область выделения точки
    _active_area = None  # Сохраняем активную область
    _hover_point_index: int = -1  # Индекс точки под курсором для подсветки
    _is_lmb_pressed: bool = False  # Флаг для отслеживания нажатия ЛКМ
    _user_toggle_magnifier: bool = False  # Флаг для хранения состояния лупы, включенной пользователем
    _pending_group_move_confirmation = None  # Подтверждение переноса уже существующей точки группы
    
    # Добавляем переменные для режима симметрии
    _mirror_mode_active: bool = False  # Флаг активности режима симметрии
    _mirror_point_preview: list = []  # Список для хранения предпросмотра зеркальных точек
    _mirror_axis_position: float = 0.5  # Позиция оси симметрии (0-1, по умолчанию в центре)
    _mirror_axis_orientation: str = 'vertical'  # Ориентация оси ('vertical', 'horizontal')
    _mirror_direction: str = 'left_to_right'  # Направление симметрии ('left_to_right', 'right_to_left')
    _is_adjusting_mirror_axis: bool = False  # Флаг для перетаскивания оси симметрии
    
    # Добавляем переменные для режима выделения рамкой
    _is_box_select_active: bool = False  # Флаг активности режима выделения рамкой
    _box_select_start_x: int = 0  # Координата X начала выделения
    _box_select_start_y: int = 0  # Координата Y начала выделения
    _box_select_end_x: int = 0  # Координата X конца выделения
    _box_select_end_y: int = 0  # Координата Y конца выделения
    _is_box_selecting: bool = False  # Флаг процесса выделения (когда ЛКМ нажата)
    _selected_points: list = []  # Список выделенных точек
    
    _calibration_mode: str = 'points'  # 'points' или 'lines'
    _lines = None
    _current_line: CAMCALIB_line = None
    
    # Хранилище для handle (callback)
    _handle = None
    
    # Добавляем переменные для отслеживания быстрого переключения
    _last_switch_time = 0
    _key_repeat_delay = 0.2  # минимальная задержка между переключениями в секундах
    _preload_timer = None
    _overlay_notification_text: str = ""
    _overlay_notification_level: str = 'INFO'
    _overlay_notification_expires_at: float = 0.0
    _bezier_hover_handle_index: int = -1
    _bezier_hover_axis: str = None

    def __getattribute__(self, name):
        if name == "report":
            try:
                return object.__getattribute__(self, "_notify")
            except Exception:
                pass
        return object.__getattribute__(self, name)
    
    def __init__(self):
        """Инициализация объекта редактора"""
        self._magnifier_radius = 120  # Увеличиваем радиус для более удобного просмотра
        self._magnifier_scale = 4.0   # Увеличиваем масштаб для лучшей детализации
        self._zoom_factor = 1.0
        self._show_help = False  # Скрываем подсказки по умолчанию
        self._lines = []
        self._is_lmb_pressed = False  # Инициализация флага нажатия левой кнопки мыши
        self._user_toggle_magnifier = False  # Начальное состояние лупы, управляемой пользователем
        self._is_placing_point = False  # Инициализация флага режима расстановки точек
        self._mirror_mode_active = False  # Инициализация режима симметрии
        self._mirror_point_preview = []  # Инициализация списка предпросмотра зеркальных точек
        self._selected_points = []  # Инициализация списка выделенных точек
        self._is_box_selecting = False  # Инициализация флага выделения рамкой
        self._is_box_select_active = False  # Инициализация активности режима выделения рамкой
        self._pending_group_move_confirmation = None
        self._overlay_notification_text = ""
        self._overlay_notification_level = 'INFO'
        self._overlay_notification_expires_at = 0.0
        self._bezier_hover_handle_index = -1
        self._bezier_hover_axis = None
        
        # Обновляем глобальные данные
        global _editor_data
        _editor_data["is_magnifier_active"] = self._is_magnifier_active
        _editor_data["magnifier_radius"] = self._magnifier_radius
        _editor_data["magnifier_scale"] = self._magnifier_scale
        _editor_data["calibration_mode"] = self._calibration_mode
        _editor_data["is_dragging"] = self._is_dragging
        _editor_data["show_help"] = self._show_help
        _editor_data["mouse_x"] = self._mouse_x
        _editor_data["mouse_y"] = self._mouse_y
        self._last_switch_time = 0

    def _clear_pending_group_move_confirmation(self):
        self._pending_group_move_confirmation = None

    def _find_group_point_index(self, image_item, group_id, exclude_index=None):
        if image_item is None or group_id is None or int(group_id) < 0:
            return -1

        target_group_id = int(group_id)
        for index, point in enumerate(image_item.points):
            if exclude_index is not None and int(index) == int(exclude_index):
                continue
            if point.is_placed and int(point.point_group_id) == target_group_id:
                return index
        return -1

    def _activate_group_from_point(self, props, point):
        group_id = int(getattr(point, "point_group_id", -1))
        if 0 <= group_id < len(props.point_groups):
            props.active_point_group_index = group_id
            return True
        return False

    def _get_group_name(self, props, group_id):
        group_id = int(group_id)
        if 0 <= group_id < len(props.point_groups):
            return props.point_groups[group_id].name
        return str(group_id)

    def _queue_group_move_confirmation(self, props, group_id, existing_point_index, target_point_index=None):
        self._pending_group_move_confirmation = {
            "image_index": int(props.active_image_index),
            "group_id": int(group_id),
            "point_index": int(existing_point_index),
        }
        if target_point_index is not None:
            self._pending_group_move_confirmation["target_point_index"] = int(target_point_index)

    def _has_pending_group_move_confirmation(self, props, group_id, existing_point_index, target_point_index=None):
        pending = self._pending_group_move_confirmation
        if not pending:
            return False
        is_match = (
            int(pending.get("image_index", -1)) == int(props.active_image_index) and
            int(pending.get("group_id", -1)) == int(group_id) and
            int(pending.get("point_index", -1)) == int(existing_point_index)
        )
        if not is_match:
            return False
        if target_point_index is None:
            return "target_point_index" not in pending
        return int(pending.get("target_point_index", -1)) == int(target_point_index)

    def _sync_modal_cursor(self, context, props):
        if self._is_box_select_active or self._is_box_selecting:
            context.window.cursor_modal_set('CROSSHAIR')
            return

        if self._is_point_dragging:
            if props.is_placing_point and self._user_toggle_magnifier and self._is_lmb_pressed:
                context.window.cursor_modal_set('CROSSHAIR')
            else:
                context.window.cursor_modal_set('HAND')
            return

        if self._hover_point_index >= 0:
            context.window.cursor_modal_set('EYEDROPPER')
            return

        if getattr(self, "_bezier_hover_handle_index", -1) >= 0:
            context.window.cursor_modal_set('HAND')
            return

        if props.is_placing_point:
            context.window.cursor_modal_set('CROSSHAIR')
            return

        context.window.cursor_modal_restore()

    def _deactivate_box_select(self):
        was_active = self._is_box_select_active or self._is_box_selecting
        self._is_box_select_active = False
        self._is_box_selecting = False
        return was_active

    def _deactivate_point_modes(self, props):
        was_active = bool(props.is_placing_point or props.is_linking_points)
        props.is_placing_point = False
        self._is_placing_point = False
        props.is_linking_points = False
        self._clear_pending_group_move_confirmation()
        return was_active

    def _clear_overlay_notification(self):
        self._overlay_notification_text = ""
        self._overlay_notification_level = 'INFO'
        self._overlay_notification_expires_at = 0.0

    def _show_overlay_notification(self, context, message, level='INFO', duration=None):
        if not message:
            return

        level = str(level or 'INFO').upper()
        default_durations = {
            'INFO': 2.8,
            'WARNING': 4.0,
            'ERROR': 5.0,
        }

        self._overlay_notification_text = str(message)
        self._overlay_notification_level = level
        self._overlay_notification_expires_at = time.time() + float(duration or default_durations.get(level, 3.0))

        area = self._active_area or getattr(context, "area", None)
        if area is not None:
            try:
                area.tag_redraw()
            except Exception:
                pass

    def _notify(self, type, message):
        level = 'INFO'
        if isinstance(type, (set, list, tuple)) and len(type) > 0:
            level = next(iter(type))
        elif isinstance(type, str):
            level = type

        context = None
        try:
            context = bpy.context
        except Exception:
            context = None

        use_overlay = bool(getattr(self, "_handle", None) and getattr(self, "_active_area", None))
        if context is not None and use_overlay:
            self._show_overlay_notification(context, message, level=level)
            return

        try:
            super().report(type, message)
        except Exception:
            pass

    def report(self, type, message):
        self._notify(type, message)
    
    def adapt_ui_parameters(self, context):
        """Адаптирует параметры интерфейса на основе масштаба UI и DPI пользователя"""
        # Получаем масштаб интерфейса для корректной работы на разных разрешениях экрана
        self._ui_scale = context.preferences.system.ui_scale if hasattr(context.preferences.system, "ui_scale") else 1.0
        self._dpi_fac = context.preferences.system.dpi / 72.0 if hasattr(context.preferences.system, "dpi") else 1.0
        
        # Адаптируем параметры лупы
        self._magnifier_radius = int(120 * self._ui_scale * self._dpi_fac)
        
        # Адаптируем пороговое значение для обнаружения точек
        self._min_point_distance = 25.0 * self._ui_scale * self._dpi_fac
        
        # Выводим информацию в консоль для отладки
        print(f"Адаптация UI: масштаб={self._ui_scale}, DPI фактор={self._dpi_fac}")
        print(f"Адаптированный радиус лупы: {self._magnifier_radius}")
        print(f"Адаптированное пороговое значение для точек: {self._min_point_distance}")
    
    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "Редактор должен быть открыт из 3D вида")
            return {'CANCELLED'}
        
        # Адаптируем параметры интерфейса
        self.adapt_ui_parameters(context)
        
        # Инициализируем параметры для отслеживания изменений интерфейса
        region = context.region
        self._prev_region_width = region.width
        self._prev_region_height = region.height
        
        # Инициализируем информацию о N-панели
        self._prev_n_panel_width = 0
        for region_ui in context.area.regions:
            if region_ui.type == 'UI' and region_ui.width > 1:
                self._prev_n_panel_width = region_ui.width
                break
        
        props = context.scene.camera_calibration
        
        # Проверяем наличие изображений
        if len(props.images) == 0:
            self.report({'WARNING'}, "Нет загруженных изображений")
            return {'CANCELLED'}
        
        # Проверяем наличие активного изображения
        if props.active_image_index < 0:
            props.active_image_index = 0
        
        # Назначаем обработчик для отрисовки
        args = (self, context)
        try:
            self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, args, 'WINDOW', 'POST_PIXEL')
        except Exception as e:
            self.report({'ERROR'}, f"Не удалось добавить обработчик рисования: {str(e)}")
            return {'CANCELLED'}
        
        # Запоминаем активную область
        self._active_area = context.area
        
        # Режим отображения текущего оператора
        # Синхронизируем состояние размещения точек с настройками
        self._is_placing_point = props.is_placing_point
        
        context.window_manager.modal_handler_add(self)
        
        # Предварительно загружаем изображения при запуске
        preload_images(props)
        
        # Инициализируем менеджер кривых Безье
        if bezier_module_imported:
            initialize_bezier_manager()
            print("Менеджер кривых Безье инициализирован")
        
        # Устанавливаем флаг активности редактора
        context.scene.camera_calibration.image_editor_active = True
        
        # Выводим подсказку о горячих клавишах
        self.report({'INFO'}, "Нажмите H для просмотра подсказок. ESC - для выхода из режимов или закрытия редактора.")
        
        return {'RUNNING_MODAL'}

    def track_interface_changes(self, context):
        """Отслеживает изменения в интерфейсе и окне Blender"""
        # Сохраняем текущие размеры региона и активной области
        region = context.region
        current_width = region.width
        current_height = region.height
        
        # Сохраняем информацию о N-панели
        n_panel_width = 0
        for region_ui in self._active_area.regions:
            if region_ui.type == 'UI' and region_ui.width > 1:
                n_panel_width = region_ui.width
                break
        
        # Если у нас есть сохраненные предыдущие размеры, проверяем изменения
        if hasattr(self, '_prev_region_width'):
            width_changed = self._prev_region_width != current_width
            height_changed = self._prev_region_height != current_height
            n_panel_changed = self._prev_n_panel_width != n_panel_width
            
            # Если что-то изменилось, обновляем интерфейс
            if width_changed or height_changed or n_panel_changed:
                # Выводим информацию о изменениях для отладки
                if width_changed:
                    print(f"Изменилась ширина региона: {self._prev_region_width} -> {current_width}")
                if height_changed:
                    print(f"Изменилась высота региона: {self._prev_region_height} -> {current_height}")
                if n_panel_changed:
                    print(f"Изменилась ширина N-панели: {self._prev_n_panel_width} -> {n_panel_width}")
                
                # Обновляем масштаб и положение точек
                self._active_area.tag_redraw()
        
        # Сохраняем текущие значения для последующего сравнения
        self._prev_region_width = current_width
        self._prev_region_height = current_height
        self._prev_n_panel_width = n_panel_width
    
    def modal(self, context, event):
        """Обработка событий"""
        global _editor_data
        try:
            # Обновляем положение мыши для использования в отрисовке
            self._mouse_x = event.mouse_x
            self._mouse_y = event.mouse_y
            
            # Обновляем глобальные данные
            _editor_data["mouse_x"] = self._mouse_x
            _editor_data["mouse_y"] = self._mouse_y
            
            # Получаем масштаб интерфейса для корректной работы на разных разрешениях экрана
            ui_scale = context.preferences.system.ui_scale if hasattr(context.preferences.system, "ui_scale") else 1.0
            dpi_fac = context.preferences.system.dpi / 72.0 if hasattr(context.preferences.system, "dpi") else 1.0
            
            # Проверяем, что область все еще существует
            if not self._active_area or self._active_area.as_pointer() not in [area.as_pointer() for area in context.screen.areas]:
                self.cleanup(context)
                return {'CANCELLED'}
            
            # Отслеживаем изменения в интерфейсе и окне Blender
            self.track_interface_changes(context)
            
            # Обновляем список активных регионов
            for region in self._active_area.regions:
                if region.type == 'UI' and hasattr(region, 'width'):
                    _editor_data["n_panel_width"] = region.width
                    break
                    
            # Проверяем, находится ли курсор в нашей области и не над N-панелью
            is_in_area = self.check_active_area(context, event)
            
            # Если курсор не в нашей области или над N-панелью, пропускаем события мыши
            if not is_in_area:
                # Если курсор вышел из области во время перетаскивания,
                # мы все равно обрабатываем события, чтобы не потерять drag & drop
                if not self._is_dragging and not self._is_point_dragging and not self._is_mmb_dragging:
                    context.window.cursor_modal_restore()
                    return {'PASS_THROUGH'}
            
            props = context.scene.camera_calibration
            
            # Лупа активна только в режиме постановки точки:
            # пользователь включил ее через Z и удерживает ЛКМ.
            self._is_magnifier_active = (
                self._user_toggle_magnifier and
                self._is_lmb_pressed and
                props.is_placing_point
            )
            
            # Обновляем информацию для отрисовки лупы и статуса перетаскивания
            _editor_data["is_dragging"] = self._is_point_dragging
            _editor_data["is_magnifier_active"] = self._is_magnifier_active
            _editor_data["active_point_index"] = self._hover_point_index
            _editor_data["show_help"] = self._show_help
            _editor_data["dragged_point_index"] = self._dragged_point_index if self._is_point_dragging else -1
            # Обновляем информацию о режимах
            _editor_data["is_placing_point"] = props.is_placing_point
            _editor_data["is_linking_points"] = props.is_linking_points
            _editor_data["mirror_mode_active"] = self._mirror_mode_active
            _editor_data["mirror_axis_orientation"] = self._mirror_axis_orientation
            
            # Получаем координаты мыши относительно региона с учетом масштаба интерфейса
            region = context.region
            mouse_region_x = event.mouse_x - region.x
            mouse_region_y = event.mouse_y - region.y
            
            image_item = None
            if 0 <= props.active_image_index < len(props.images):
                image_item = props.images[props.active_image_index]

            previous_hover = self._hover_point_index
            if image_item is not None and not self._is_point_dragging:
                self._hover_point_index = self.find_point_under_cursor(
                    context,
                    image_item,
                    event.mouse_x,
                    event.mouse_y,
                )
            elif image_item is None:
                self._hover_point_index = -1

            if previous_hover != self._hover_point_index and self._active_area:
                self._active_area.tag_redraw()

            self._sync_modal_cursor(context, props)
            
            # Обработка событий клавиатуры
            if event.type == 'ESC' and event.value == 'PRESS':
                self._clear_pending_group_move_confirmation()
                if self._is_point_dragging and props.is_placing_point:
                    # Если мы перетаскиваем новую точку, отменяем её создание
                    image_item = props.images[props.active_image_index]
                    if self._dragged_point_index >= 0 and self._dragged_point_index < len(image_item.points):
                        # Удаляем точку, которую мы только что создали и перетаскивали
                        image_item.points[self._dragged_point_index].is_placed = False
                        self._is_point_dragging = False
                        self._dragged_point_index = -1
                        self.report({'INFO'}, "Отмена создания новой точки")
                        context.window.cursor_modal_set('CROSSHAIR')
                        return {'RUNNING_MODAL'}
                elif props.is_placing_point or props.is_linking_points:
                    props.is_placing_point = False
                    self._is_placing_point = False  # Синхронизируем состояние
                    props.is_linking_points = False
                    # Восстанавливаем стандартный курсор
                    context.window.cursor_modal_restore()
                    return {'RUNNING_MODAL'}
                elif self._mirror_mode_active:
                    # Если активен режим симметрии, выключаем его
                    self._mirror_mode_active = False
                    self.report({'INFO'}, "Режим симметрии выключен")
                    return {'RUNNING_MODAL'}
                elif self._is_box_select_active:
                    # Если активен режим выделения рамкой, выключаем его
                    self._is_box_select_active = False
                    self._is_box_selecting = False
                    self.report({'INFO'}, "Режим выделения рамкой выключен")
                    return {'RUNNING_MODAL'}
                else:
                    # Окончательный выход из редактора
                    print("Test")
                    self.cleanup(context)
                    self.report({'INFO'}, "Выход из редактора изображений")
                    return {'FINISHED'}  # Используем FINISHED вместо CANCELLED для нормального завершения
            
            elif event.type == 'M' and event.value == 'PRESS':
                # Переключаем режим симметрии
                self.toggle_mirror_mode(context)
                # Показываем сообщение о текущем состоянии
                if self._mirror_mode_active:
                    self.report({'INFO'}, "Режим симметрии включен. Клавиши: V-вертикальная ось, H-горизонтальная, S-направление, стрелки-перемещение оси, C-создать симметричные точки")
                else:
                    self.report({'INFO'}, "Режим симметрии выключен")
                return {'RUNNING_MODAL'}
            
            elif event.type == 'S' and event.value == 'PRESS' and self._mirror_mode_active:
                # Переключаем направление симметрии
                self._mirror_direction = 'right_to_left' if self._mirror_direction == 'left_to_right' else 'left_to_right'
                direction_text = "слева направо" if self._mirror_direction == 'left_to_right' else "справа налево"
                self.report({'INFO'}, f"Направление симметрии: {direction_text}")
                return {'RUNNING_MODAL'}
            
            elif event.type in {'DEL', 'X'} and event.value == 'PRESS':
                # Удаление активной точки или выделенных точек
                image_item = props.images[props.active_image_index]
                
                # Проверяем, есть ли выделенные точки
                if hasattr(self, '_selected_points') and self._selected_points:
                    # Удаляем все выделенные точки
                    points_removed = 0
                    for point_index in sorted(self._selected_points, reverse=True):
                        if point_index >= 0 and point_index < len(image_item.points):
                            point = image_item.points[point_index]
                            if point.is_placed:
                                point.is_placed = False
                                point.point_group_id = -1
                                points_removed += 1
                    
                    # Очищаем список выделенных точек
                    self._selected_points = []
                    
                    if points_removed > 0:
                        self.report({'INFO'}, f"Удалено {points_removed} точек")
                    return {'RUNNING_MODAL'}
                    
                # Если нет выделенных точек, удаляем активную точку
                elif image_item.active_point_index >= 0 and image_item.active_point_index < len(image_item.points):
                    # Помечаем точку как неразмещенную вместо полного удаления
                    # Это позволяет избежать проблем с индексами
                    point = image_item.points[image_item.active_point_index]
                    if point.is_placed:
                        point.is_placed = False
                        point.point_group_id = -1
                        self.report({'INFO'}, "Точка удалена")
                        image_item.active_point_index = -1
                return {'RUNNING_MODAL'}
            
            elif event.type == 'R' and event.value == 'PRESS':
                # Исключение активной точки или выделенных точек из группы
                image_item = props.images[props.active_image_index]
                
                # Проверяем, есть ли выделенные точки
                if hasattr(self, '_selected_points') and self._selected_points:
                    # Исключаем все выделенные точки из групп
                    points_excluded = 0
                    for point_index in self._selected_points:
                        if point_index >= 0 and point_index < len(image_item.points):
                            point = image_item.points[point_index]
                            if point.is_placed and point.point_group_id >= 0:
                                point.point_group_id = -1
                                points_excluded += 1
                    
                    if points_excluded > 0:
                        self.report({'INFO'}, f"Исключено из групп {points_excluded} точек")
                    else:
                        self.report({'INFO'}, "Выделенные точки уже не принадлежат группам")
                    return {'RUNNING_MODAL'}
                
                # Если нет выделенных точек, исключаем активную точку
                elif image_item.active_point_index >= 0 and image_item.active_point_index < len(image_item.points):
                    point = image_item.points[image_item.active_point_index]
                    if point.is_placed and point.point_group_id >= 0:
                        # Сохраняем старый ID группы для отчета
                        old_group_id = point.point_group_id
                        # Исключаем точку из группы
                        point.point_group_id = -1
                        if old_group_id < len(props.point_groups):
                            self.report({'INFO'}, f"Точка исключена из группы {props.point_groups[old_group_id].name}")
                        else:
                            self.report({'INFO'}, f"Точка исключена из группы {old_group_id}")
                    elif point.is_placed:
                        self.report({'INFO'}, "Точка уже не принадлежит ни одной группе")
                return {'RUNNING_MODAL'}
            
            elif event.type == 'L' and event.value == 'PRESS':
                # Переключаем режим связывания точек
                self._clear_pending_group_move_confirmation()
                self._deactivate_box_select()
                props.is_linking_points = not props.is_linking_points
                props.is_placing_point = False  # Выключаем режим добавления точек
                self._is_placing_point = False  # Синхронизируем локальное состояние
                # Восстанавливаем стандартный курсор
                context.window.cursor_modal_restore()
                if props.is_linking_points:
                    self.report({'INFO'}, "Режим связывания точек включен")
                else:
                    self.report({'INFO'}, "Режим связывания точек выключен")
                return {'RUNNING_MODAL'}
            
            elif event.type == 'A' and event.value == 'PRESS':
                # Синхронизируем состояние переменных расстановки точек
                self._clear_pending_group_move_confirmation()
                self._deactivate_box_select()
                props.is_placing_point = not props.is_placing_point
                self._is_placing_point = props.is_placing_point
                
                props.is_linking_points = False  # Выключаем режим связывания точек
                
                # Проверяем, импортирован ли модуль безье и какой режим установлен
                bezier_mode = False
                try:
                    if hasattr(context.scene, 'bezier_props') and context.scene.bezier_props.placement_mode == 'BEZIER_CURVE':
                        bezier_mode = True
                except Exception as e:
                    print(f"Ошибка при проверке режима безье: {e}")
                
                # Устанавливаем курсор и выводим сообщение в зависимости от режима
                if self._is_placing_point:
                    context.window.cursor_modal_set('CROSSHAIR')
                    if bezier_mode:
                        self.report({'INFO'}, "Включен режим создания кривых Безье. Кликните для размещения точек кривой.")
                    else:
                        self.report({'INFO'}, "Включен режим расстановки точек. Кликните для размещения точки.")
                else:
                    context.window.cursor_modal_set('DEFAULT')
                    if bezier_mode:
                        self.report({'INFO'}, "Выключен режим создания кривых Безье. Кликните на точку для выбора.")
                    else:
                        self.report({'INFO'}, "Выключен режим расстановки точек. Кликните на точку для выбора.")
            
            elif event.type == 'T' and event.value == 'PRESS':
                # Размещаем тестовые точки
                self.setup_test_points(context)
                return {'RUNNING_MODAL'}
            
            elif event.type == 'G' and event.value == 'PRESS':
                bpy.ops.camera_calibration.add_point_group()
                return {'RUNNING_MODAL'}
            
            # Обработка стрелок для перемещения оси симметрии и переключения изображений
            elif event.type in {'LEFT_ARROW', 'RIGHT_ARROW'} and event.value == 'PRESS':
                if self._mirror_mode_active and self._mirror_axis_orientation == 'vertical':
                    # Если режим симметрии активен и ось вертикальная, двигаем ось
                    delta = -0.05 if event.type == 'LEFT_ARROW' else 0.05
                    self._mirror_axis_position = max(0.0, min(1.0, self._mirror_axis_position + delta))
                    self.report({'INFO'}, f"Позиция оси: {self._mirror_axis_position:.2f}")
                    return {'RUNNING_MODAL'}
                else:
                    # Если режим симметрии не активен или ось горизонтальная, переключаем изображения
                    import time
                    current_time = time.time()
                    # Проверяем, прошло ли достаточно времени с момента последнего переключения
                    if current_time - self._last_switch_time > self._key_repeat_delay and len(props.images) > 0:
                        if event.type == 'LEFT_ARROW':
                            props.active_image_index = (props.active_image_index - 1) % len(props.images)
                        else:  # RIGHT_ARROW
                            props.active_image_index = (props.active_image_index + 1) % len(props.images)

                        self._last_switch_time = current_time

                        def preload_neighbors():
                            import threading
                            threading.Timer(0.1, lambda: preload_images(props)).start()

                        preload_neighbors()
                return {'RUNNING_MODAL'}
            
            elif event.type in {'UP_ARROW', 'DOWN_ARROW'} and event.value == 'PRESS':
                if self._mirror_mode_active and self._mirror_axis_orientation == 'horizontal':
                    # Если режим симметрии активен и ось горизонтальная, двигаем ось
                    delta = 0.05 if event.type == 'UP_ARROW' else -0.05
                    self._mirror_axis_position = max(0.0, min(1.0, self._mirror_axis_position + delta))
                    self.report({'INFO'}, f"Позиция оси: {self._mirror_axis_position:.2f}")
                    return {'RUNNING_MODAL'}
            
            elif event.type == 'P' and event.value == 'PRESS':
                # Вывод данных для тестирования
                self.print_points_data(context)
                return {'RUNNING_MODAL'}
                
            elif event.type == 'C' and event.value == 'PRESS':
                if self._mirror_mode_active:
                    # Создаем все зеркальные точки
                    created_count = 0
                    current_image = self.get_current_image_item()
                    if current_image:
                        for point_idx in range(len(current_image.points)):
                            if current_image.points[point_idx].is_placed and self.create_mirror_point(context, point_idx):
                                created_count += 1
                        
                        if created_count > 0:
                            self.report({'INFO'}, f"Создано {created_count} зеркальных точек.")
                        else:
                            self.report({'INFO'}, "Нет точек для зеркального отражения.")
                    return {'RUNNING_MODAL'}
                else:
                    # Запуск калибровки через стандартный оператор вместо динамической калибровки
                    bpy.ops.camera_calibration.run_advanced_calibration('INVOKE_DEFAULT')
                    return {'RUNNING_MODAL'}
                
            # Клавиша V для установки вертикальной оси симметрии
            elif event.type == 'V' and event.value == 'PRESS' and self._mirror_mode_active:
                self._mirror_axis_orientation = 'vertical'
                self.report({'INFO'}, "Установлена вертикальная ось симметрии.")
                return {'RUNNING_MODAL'}
            
            # Клавиша H для установки горизонтальной оси симметрии
            elif event.type == 'H' and event.value == 'PRESS' and self._mirror_mode_active:
                if not self._show_help:  # Если не режим подсказок
                    self._mirror_axis_orientation = 'horizontal'
                    self.report({'INFO'}, "Установлена горизонтальная ось симметрии.")
                    return {'RUNNING_MODAL'}
            
            elif event.type == 'Z' and event.value == 'PRESS':
                # Переключаем режим лупы
                self._user_toggle_magnifier = not self._user_toggle_magnifier
                # Показываем пользователю статус лупы
                if self._user_toggle_magnifier:
                    self.report({'INFO'}, "Лупа включена (работает при зажатой ЛКМ в режиме постановки точки)")
                else:
                    self.report({'INFO'}, "Лупа отключена")
                return {'RUNNING_MODAL'}
            
            elif event.type == 'H' and event.value == 'PRESS':
                # Показать/скрыть подсказки
                self._show_help = not self._show_help
                _editor_data["show_help"] = self._show_help
                self.report({'INFO'}, f"Подсказки {'показаны' if self._show_help else 'скрыты'}")
                return {'RUNNING_MODAL'}
            
            elif event.type == 'B' and event.value == 'PRESS':
                # Включаем/выключаем режим выделения рамкой
                self._is_box_select_active = not self._is_box_select_active
                if self._is_box_select_active:
                    self._deactivate_point_modes(props)
                    # Очищаем предыдущее выделение при входе в режим
                    if not event.shift:
                        self._selected_points = []
                    self.report({'INFO'}, "Включен режим выделения рамкой. Используйте ЛКМ для выделения, Shift - добавление к выделению.")
                    self._sync_modal_cursor(context, props)
                else:
                    self.report({'INFO'}, f"Режим выделения рамкой выключен. Выделено {len(self._selected_points)} точек.")
                    self._sync_modal_cursor(context, props)
                return {'RUNNING_MODAL'}
            
            elif event.type == 'LEFTMOUSE':
                if event.value == 'PRESS':
                    # Отслеживаем нажатие ЛКМ для активации лупы
                    self._is_lmb_pressed = True
                    
                    # Проверяем, активен ли режим выделения рамкой
                    if self._is_box_select_active:
                        # Начинаем выделение рамкой
                        self._is_box_selecting = True
                        self._box_select_start_x = mouse_region_x
                        self._box_select_start_y = mouse_region_y
                        self._box_select_end_x = mouse_region_x
                        self._box_select_end_y = mouse_region_y
                        return {'RUNNING_MODAL'}
                    
                    # Проверяем, активен ли режим кривой Безье
                    is_bezier_mode_active = False
                    if hasattr(context.scene, 'bezier_props'):
                        if hasattr(context.scene.bezier_props, 'placement_mode'):
                            is_bezier_mode_active = (context.scene.bezier_props.placement_mode == 'BEZIER_CURVE')
                    
                    if props.is_placing_point and not is_bezier_mode_active:  # Добавляем проверку режима Безье
                        # Добавляем новую точку при нажатии и начинаем её перетаскивание
                        image_item = props.images[props.active_image_index]
                        params = get_image_display_params(context, image_item)
                        if params:
                            # Получаем изображение
                            image = None
                            for img in bpy.data.images:
                                if img.name == image_item.name:
                                    image = img
                                    break
                                    
                            if not image:
                                return {'RUNNING_MODAL'}
                                
                            # Преобразуем координаты мыши в координаты изображения
                            image_coords = mouse_to_image_coords(context, image_item, event.mouse_x, event.mouse_y)
                            if not image_coords:
                                return {'RUNNING_MODAL'}
                                
                            image_x, image_y = image_coords
                            
                            # Проверяем, что точка находится в пределах изображения
                            if (0 <= image_x <= image.size[0] and
                                0 <= image_y <= image.size[1]):
                                hovered_point_index = self.find_point_under_cursor(
                                    context,
                                    image_item,
                                    event.mouse_x,
                                    event.mouse_y,
                                )
                                if hovered_point_index >= 0 and hovered_point_index < len(image_item.points):
                                    hovered_point = image_item.points[hovered_point_index]
                                    self._activate_group_from_point(props, hovered_point)

                                # Проверяем, не слишком ли близко к существующим точкам
                                is_valid, closest_point_index = self.check_point_distance(image_item, (image_x, image_y))
                                
                                if is_valid:
                                    existing_group_point_index = self._find_group_point_index(
                                        image_item,
                                        props.active_point_group_index,
                                    )
                                    if existing_group_point_index >= 0:
                                        existing_group_point = image_item.points[existing_group_point_index]
                                        if self._has_pending_group_move_confirmation(
                                            props,
                                            props.active_point_group_index,
                                            existing_group_point_index,
                                        ):
                                            existing_group_point.location_2d = (image_x, image_y)
                                            image_item.active_point_index = existing_group_point_index
                                            self._is_point_dragging = True
                                            self._dragged_point_index = existing_group_point_index
                                            self._clear_pending_group_move_confirmation()
                                            group_name = self._get_group_name(props, props.active_point_group_index)
                                            self.report(
                                                {'INFO'},
                                                f"Точка группы {group_name} перенесена. "
                                                "Перемещайте мышь для точного позиционирования."
                                            )
                                            self._sync_modal_cursor(context, props)
                                            return {'RUNNING_MODAL'}

                                        image_item.active_point_index = existing_group_point_index
                                        self._queue_group_move_confirmation(
                                            props,
                                            props.active_point_group_index,
                                            existing_group_point_index,
                                        )
                                        group_name = self._get_group_name(props, props.active_point_group_index)
                                        self.report(
                                            {'WARNING'},
                                            f"В группе {group_name} уже есть точка на этом изображении. "
                                            "Кликните ещё раз в нужное место, чтобы перенести её."
                                        )
                                        self._sync_modal_cursor(context, props)
                                        return {'RUNNING_MODAL'}

                                    # Добавляем новую точку
                                    point = image_item.points.add()
                                    point.location_2d = (image_x, image_y)
                                    point.is_placed = True
                                    point.point_group_id = props.active_point_group_index
                                    new_point_index = len(image_item.points) - 1
                                    image_item.active_point_index = new_point_index
                                    
                                    # Начинаем перетаскивание новой точки
                                    self._is_point_dragging = True
                                    self._dragged_point_index = new_point_index
                                    self._clear_pending_group_move_confirmation()
                                    
                                    self.report({'INFO'}, f"Добавлена точка {len(image_item.points)} (группа {props.active_point_group_index}). Перемещайте мышь для позиционирования. ESC для отмены.")
                                    self._sync_modal_cursor(context, props)
                                elif closest_point_index >= 0:
                                    # Если точка слишком близко к существующей, выделяем существующую и начинаем её перетаскивание
                                    image_item.active_point_index = closest_point_index
                                    self._is_point_dragging = True
                                    self._dragged_point_index = closest_point_index
                                    self._clear_pending_group_move_confirmation()
                                    self.report({'INFO'}, "Перемещение существующей точки. Для завершения отпустите кнопку мыши.")
                                    
                                    # Если точка принадлежит другой группе, предлагаем переключиться на неё
                                    existing_point = image_item.points[closest_point_index]
                                    if self._activate_group_from_point(props, existing_point):
                                        self.report({'INFO'}, f"Активна группа {self._get_group_name(props, existing_point.point_group_id)}")
                                    self._sync_modal_cursor(context, props)
                    elif props.is_linking_points:
                        # Находим точку под курсором
                        image_item = props.images[props.active_image_index]
                        point_index = self.find_point_under_cursor(context, image_item, event.mouse_x, event.mouse_y)
                        if point_index >= 0:
                            point = image_item.points[point_index]
                            if point.point_group_id >= 0:
                                # Если точка уже в группе, делаем эту группу активной
                                props.active_point_group_index = point.point_group_id
                                image_item.active_point_index = point_index
                                self._clear_pending_group_move_confirmation()
                            else:
                                # Если точка без группы, привязываем её к активной группе
                                if props.active_point_group_index >= 0:
                                    existing_group_point_index = self._find_group_point_index(
                                        image_item,
                                        props.active_point_group_index,
                                        exclude_index=point_index,
                                    )
                                    if existing_group_point_index >= 0:
                                        existing_group_point = image_item.points[existing_group_point_index]
                                        if self._has_pending_group_move_confirmation(
                                            props,
                                            props.active_point_group_index,
                                            existing_group_point_index,
                                            target_point_index=point_index,
                                        ):
                                            existing_group_point.point_group_id = -1
                                            point.point_group_id = props.active_point_group_index
                                            self._clear_pending_group_move_confirmation()
                                            self.report(
                                                {'INFO'},
                                                f"Группа {self._get_group_name(props, props.active_point_group_index)} "
                                                "перенесена на выбранную точку."
                                            )
                                        else:
                                            image_item.active_point_index = point_index
                                            self._queue_group_move_confirmation(
                                                props,
                                                props.active_point_group_index,
                                                existing_group_point_index,
                                                target_point_index=point_index,
                                            )
                                            self.report(
                                                {'WARNING'},
                                                f"В группе {self._get_group_name(props, props.active_point_group_index)} "
                                                "уже есть точка на этом изображении. "
                                                "Кликните ещё раз по новой точке, чтобы заменить старую."
                                            )
                                            self._sync_modal_cursor(context, props)
                                            return {'RUNNING_MODAL'}
                                    else:
                                        point.point_group_id = props.active_point_group_index
                                        self._clear_pending_group_move_confirmation()
                                        self.report(
                                            {'INFO'},
                                            f"Точка привязана к группе {self._get_group_name(props, props.active_point_group_index)}"
                                        )
                                image_item.active_point_index = point_index
                    else:
                        # Проверяем, не кликнули ли мы на точку
                        image_item = props.images[props.active_image_index]
                        
                        # Используем нашу функцию для поиска точки под курсором
                        point_index = self.find_point_under_cursor(context, image_item, event.mouse_x, event.mouse_y)
                        
                        if point_index >= 0:
                            # Нашли точку под курсором - начинаем перетаскивание
                            self._is_point_dragging = True
                            self._dragged_point_index = point_index
                            image_item.active_point_index = point_index
                            
                            # Если зажат Shift, добавляем/удаляем точку из выделения
                            if hasattr(self, '_selected_points') and event.shift:
                                if point_index in self._selected_points:
                                    # Если точка уже выделена - исключаем её из выделения
                                    self._selected_points.remove(point_index)
                                    self.report({'INFO'}, f"Точка {point_index + 1} исключена из выделения. Всего выделено: {len(self._selected_points)}")
                                else:
                                    # Если точка не выделена - добавляем в выделение
                                    self._selected_points.append(point_index)
                                    self.report({'INFO'}, f"Точка {point_index + 1} добавлена в выделение. Всего выделено: {len(self._selected_points)}")
                            else:
                                # Если Shift не зажат, заменяем ыделение на текущую точку
                                self._selected_points = [point_index]
                                self.report({'INFO'}, f"Выделена точка {point_index + 1}. Перемещайте мышь для изменения позиции. Для завершения отпустите кнопку мыши.")
                            self._clear_pending_group_move_confirmation()
                            self._sync_modal_cursor(context, props)
                        else:
                            # Клик был не на точке - очищаем выделение, если не зажат Shift
                            if hasattr(self, '_selected_points') and len(self._selected_points) > 0 and not event.shift:
                                old_selection_count = len(self._selected_points)
                                self._selected_points = []
                                self.report({'INFO'}, f"Выделение очищено ({old_selection_count} точек)")
                            print(f"Клик мимо точки в координатах ({mouse_region_x}, {mouse_region_y})")
                elif event.value == 'RELEASE':
                    # Отслеживаем отпускание ЛКМ для деактивации лупы
                    self._is_lmb_pressed = False
                    
                    if self._is_point_dragging:
                        # Завершаем перетаскивание точки
                        self._is_point_dragging = False
                        self._dragged_point_index = -1
                        self._sync_modal_cursor(context, props)
                    
                    elif self._is_box_selecting:
                        # Завершаем выделение рамкой
                        self._is_box_selecting = False
                        
                        # Получаем активное изображение
                        image_item = props.images[props.active_image_index]
                        
                        # Выделяем точки в рамке
                        self.select_points_in_box(context, image_item, event.shift)
            
            elif event.type == 'MIDDLEMOUSE':
                if event.value == 'PRESS':
                    if event.shift:
                        # Масштабирование при зажатом Shift
                        self._is_mmb_dragging = True
                        self._drag_start_x = mouse_region_x
                        self._drag_start_y = mouse_region_y
                    else:
                        # Панорамирование
                        self._is_dragging = True
                        self._drag_start_x = mouse_region_x
                        self._drag_start_y = mouse_region_y
                        image_item = props.images[props.active_image_index]
                        self._initial_offset = Vector((image_item.offset_x, image_item.offset_y))
                elif event.value == 'RELEASE':
                    if self._is_dragging:
                        self._is_dragging = False
                    if self._is_mmb_dragging:
                        self._is_mmb_dragging = False
            
            elif event.type == 'MOUSEMOVE':
                if self._is_dragging:
                    # Перетаскивание изображения
                    image_item = props.images[props.active_image_index]
                    dx = mouse_region_x - self._drag_start_x
                    dy = mouse_region_y - self._drag_start_y
                    image_item.offset_x = self._initial_offset.x + dx
                    image_item.offset_y = self._initial_offset.y + dy
                
                elif self._is_point_dragging:
                    # Перетаскивание точки
                    image_item = props.images[props.active_image_index]
                    params = get_image_display_params(context, image_item)
                    if params:
                        # Получаем изображение
                        image = None
                        for img in bpy.data.images:
                            if img.name == image_item.name:
                                image = img
                                break
                                
                        if not image:
                            return {'RUNNING_MODAL'}
                            
                        # Преобразуем координаты мыши в координаты изображения
                        image_coords = mouse_to_image_coords(context, image_item, event.mouse_x, event.mouse_y)
                        if not image_coords:
                            return {'RUNNING_MODAL'}
                            
                        image_x, image_y = image_coords
                        
                        # Проверяем границы изображения
                        if not (0 <= image_x <= image.size[0] and 0 <= image_y <= image.size[1]):
                            return {'RUNNING_MODAL'}
                            
                        # Если перетаскиваемая точка в списке выделенных точек и есть другие выделенные точки,
                        # перемещаем все выделенные точки
                        if (hasattr(self, '_selected_points') and 
                            self._dragged_point_index in self._selected_points and 
                            len(self._selected_points) > 1):
                            
                            # Вычисляем смещение от начальной позиции точки
                            dragged_point = image_item.points[self._dragged_point_index]
                            dx = image_x - dragged_point.location_2d[0]
                            dy = image_y - dragged_point.location_2d[1]
                            
                            # Перемещаем все выделенные точки на то же смещение
                            for point_idx in self._selected_points:
                                if point_idx >= 0 and point_idx < len(image_item.points) and point_idx != self._dragged_point_index:
                                    point = image_item.points[point_idx]
                                    if point.is_placed:
                                        new_x = point.location_2d[0] + dx
                                        new_y = point.location_2d[1] + dy
                                        
                                        # Проверяем границы изображения для каждой точки
                                        if 0 <= new_x <= image.size[0] and 0 <= new_y <= image.size[1]:
                                            point.location_2d = (new_x, new_y)
                            
                            # Устанавливаем позицию перетаскиваемой точки последней
                            dragged_point.location_2d = (image_x, image_y)
                        else:
                            # Если нет группового выделения, перемещаем только текущую точку
                            point = image_item.points[self._dragged_point_index]
                            point.location_2d = (image_x, image_y)
                
                elif self._is_mmb_dragging:
                    # Масштабирование изображения
                    image_item = props.images[props.active_image_index]
                    dy = mouse_region_y - self._drag_start_y
                    # Более плавное масштабирование
                    scale_factor = 1.0 + dy * 0.01
                    image_item.scale *= scale_factor
                    image_item.scale = max(0.1, min(10.0, image_item.scale))
                    self._drag_start_y = mouse_region_y
                
                elif self._is_box_selecting:
                    # Обновление координат конца рамки выделения
                    self._box_select_end_x = mouse_region_x
                    self._box_select_end_y = mouse_region_y
            
            elif event.type == 'WHEELUPMOUSE':
                # Увеличение масштаба
                image_item = props.images[props.active_image_index]
                image_item.scale *= 1.1
                image_item.scale = min(10.0, image_item.scale)
            
            elif event.type == 'WHEELDOWNMOUSE':
                # Уменьшение масштаба
                image_item = props.images[props.active_image_index]
                image_item.scale *= 0.9
                image_item.scale = max(0.1, image_item.scale)
            
            # Обновляем интерфейс
            if self._active_area:
                self._active_area.tag_redraw()
            
            # Обрабатываем ввод для кривых Безье, если модуль доступен
            if bezier_module_imported:
                # Получаем текущее изображение напрямую из свойств
                props = context.scene.camera_calibration
                if props.active_image_index >= 0 and props.active_image_index < len(props.images):
                    image_item = props.images[props.active_image_index]
                    # Преобразование координат мыши в координаты изображения
                    mouse_pos_region = (event.mouse_region_x, event.mouse_region_y)
                    mouse_pos_image = mouse_to_image_coords(context, image_item, 
                                                                event.mouse_x, 
                                                                event.mouse_y)
                    
                    # Обновляем состояние наведения для контрольных точек кривой Безье
                    update_bezier_hover_state(self, context, mouse_pos_image, mouse_pos_region)
                    self._sync_modal_cursor(context, props)
                    
                    # Проверяем активен ли режим кривой Безье
                    try:
                        if context.scene.bezier_props.placement_mode == 'BEZIER_CURVE':
                            # Обработка ввода для кривых Безье
                            if handle_bezier_mode_input(self, context, event, image_item, mouse_pos_image):
                                # Если событие обработано в режиме кривой Безье, пропускаем обычную обработку
                                return {'RUNNING_MODAL'}
                    except AttributeError:
                        # Если свойства безье не зарегистрированы, игнорируем
                        pass
            
            return {'RUNNING_MODAL'}
        except Exception as e:
            # Обрабатываем любые ошибки в режиме modal
            print(f"Ошибка в modal: {str(e)}")
            import traceback
            traceback.print_exc()
            # Пробуем восстановиться и продолжить работу
            return {'RUNNING_MODAL'}
    
    
    def cleanup(self, context):
        """Очистка ресурсов"""
        if hasattr(self, '_handle'):
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            del self._handle
        
        context.window.cursor_modal_restore()
        
        # Сбрасываем режимы
        props = context.scene.camera_calibration
        props.is_placing_point = False
        self._is_placing_point = False  # Синхронизируем локальное состояние
        props.is_linking_points = False
        
        # Очищаем временные данные
        self._lines = []
        self._current_line = None
        self._active_area = None
        self._clear_overlay_notification()
        
        # Обновляем интерфейс
        for area in context.screen.areas:
            area.tag_redraw()
    
        # Очищаем кэш изображений при выходе из редактора
        global _image_cache
        
        # Оставляем в кэше только активное изображение, если оно есть
        props = context.scene.camera_calibration
        if props.active_image_index >= 0 and props.active_image_index < len(props.images):
            active_name = props.images[props.active_image_index].name
            for name in list(_image_cache.keys()):
                if name != active_name:
                    del _image_cache[name]
        else:
            # Полностью очищаем кэш
            _image_cache.clear()
    
    def __del__(self):
        """Деструктор для очистки ресурсов"""
        try:
            if hasattr(self, '_handle') and self._handle:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
                except (ValueError, TypeError, ReferenceError, AttributeError):
                    pass
        except:
            # Игнорируем все ошибки в деструкторе
            pass

    def setup_test_points(self, context):
        """Размещение тестовых точек"""
        props = context.scene.camera_calibration

        from .test_calibration_logic import get_setup_test_point_coordinates

        point_coordinates = get_setup_test_point_coordinates()
        
        # Очищаем существующие группы точек
        props.point_groups.clear()
        
        # Создаем необходимое количество групп точек
        max_group_id = -1
        for img_name, points in point_coordinates.items():
            for _, _, group_id in points:
                if group_id >= 0:  # Учитываем только положительные group_id
                    max_group_id = max(max_group_id, group_id)
        
        # Создаем группы точек
        for i in range(max_group_id + 1):
            group = props.point_groups.add()
            group.name = f"Группа {i}"
            # Назначаем цвет из предопределенных
            color_index = i % len(PREDEFINED_COLORS)
            group.color = PREDEFINED_COLORS[color_index]
        
        # Размещаем точки на изображениях
        for img_name, points in point_coordinates.items():
            # Ищем изображение по имени
            for img_index, img_item in enumerate(props.images):
                if img_item.name == img_name:
                    # Очищаем существующие точки
                    img_item.points.clear()
                    
                    # Добавляем новые точки
                    for x, y, group_id in points:
                        point = img_item.points.add()
                        point.location_2d = (x, y)
                        point.is_placed = True
                        point.point_group_id = group_id
                    break
        
        self.report({'INFO'}, f"Размещено {len(point_coordinates)} наборов тестовых точек")
        
        # Заставляем интерфейс перерисоваться
        if self._active_area:
            self._active_area.tag_redraw()

    def check_point_distance(self, image_item, new_pos, exclude_index=-1):
        """Проверяет, не слишком ли близко новая точка к существующим"""
        closest_point_index = -1
        min_distance = float('inf')
        
        for i, point in enumerate(image_item.points):
            if i == exclude_index or not point.is_placed:
                continue
            dx = point.location_2d[0] - new_pos[0]
            dy = point.location_2d[1] - new_pos[1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < min_distance:
                min_distance = distance
                closest_point_index = i
                
        # Если ближайшая точка находится в пределах 5 пикселей, возвращаем её индекс
        if min_distance < 3.0:
            return False, closest_point_index
            
        return True, -1
        
    def print_points_data(self, context):
        """Выводит данные о точках в консоль в формате, удобном для тестирования"""
        props = context.scene.camera_calibration
        
        # Получаем текущее активное изображение
        if props.active_image_index < 0 or props.active_image_index >= len(props.images):
            print("Нет активного изображения")
            return
        
        image_item = props.images[props.active_image_index]
        # Получаем объект изображения через Blender API
        image = None
        for img in bpy.data.images:
            if img.name == image_item.name:
                image = img
                break
        
        if image:
            # Правильно получаем размеры изображения
            image_width, image_height = image.size
            print("\nДанные для тестирования:")
            print(f"Размер изображения: {image_width}x{image_height}")
        else:
            print("\nДанные для тестирования:")
            print("Изображение не найдено")
        
        print("\nТочки для каждого изображения:")
        
        for img in props.images:
            # Для каждого изображения также выводим его реальные размеры
            img_object = None
            for bimg in bpy.data.images:
                if bimg.name == img.name:
                    img_object = bimg
                    break
                    
            if img_object:
                img_width, img_height = img_object.size
                print(f"\n# Изображение: {img.name} (размер: {img_width}x{img_height})")
            else:
                print(f"\n# Изображение: {img.name} (размер неизвестен)")
            
            print("points = [")
            for point in img.points:
                if point.is_placed and point.point_group_id >= 0:
                    print(f"    MockPoint(({point.location_2d[0]:.3f}, {point.location_2d[1]:.3f})), {point.point_group_id}),")
            print("]")
            print(f"test_images.append(MockImage('{img.name}', points))")
        
        print("\nГруппы точек:")
        print("point_groups = [", end="")
        groups = set()
        for img in props.images:
            for point in img.points:
                if point.is_placed and point.point_group_id >= 0:
                    groups.add(point.point_group_id)
        print(", ".join(str(g) for g in sorted(groups)), end="")
        print("]")

    def check_active_area(self, context, event):
        """Проверяет, находится ли курсор в активной области и не над N-панелью"""
        if not self._active_area:
            return False
        
        # Получаем координаты области
        x = self._active_area.x
        y = self._active_area.y
        width = self._active_area.width
        height = self._active_area.height
        
        # Получаем масштаб интерфейса для корректной работы на разных разрешениях экрана
        ui_scale = context.preferences.system.ui_scale if hasattr(context.preferences.system, "ui_scale") else 1.0
        dpi_fac = context.preferences.system.dpi / 72.0 if hasattr(context.preferences.system, "dpi") else 1.0
        
        # Динамически определяем области интерфейса
        ui_regions = {}
        for region in self._active_area.regions:
            ui_regions[region.type] = region
            
        # Проверяем наличие N-панели (UI region)
        n_panel_width = 0
        n_panel_x = 0
        if 'UI' in ui_regions and ui_regions['UI'].width > 1:
            n_panel = ui_regions['UI']
            n_panel_width = n_panel.width
            n_panel_x = n_panel.x
        
        # Проверяем наличие других панелей, которые могут перекрывать рабочую область
        header_height = 0
        if 'HEADER' in ui_regions:
            header_height = ui_regions['HEADER'].height
        
        tool_header_height = 0
        if 'TOOL_HEADER' in ui_regions:
            tool_header_height = ui_regions['TOOL_HEADER'].height
            
        # Проверка, находится ли курсор над N-панелью
        is_over_n_panel = False
        if n_panel_width > 0:
            is_over_n_panel = event.mouse_x >= n_panel_x
        
        # Общая проверка, находится ли курсор в активной рабочей области
        is_in_area = (x <= event.mouse_x <= x + width and 
                     y <= event.mouse_y <= y + height)
                     
        # Дополнительная проверка для панелей инструментов
        is_over_header = False
        if header_height > 0:
            is_over_header = (y <= event.mouse_y <= y + header_height)
            
        is_over_tool_header = False
        if tool_header_height > 0:
            is_over_tool_header = (y + header_height <= event.mouse_y <= y + header_height + tool_header_height)
        
        # Возвращаем True только если курсор в области И НЕ над N-панелью/заголовками
        return (is_in_area and not is_over_n_panel and not is_over_header and not is_over_tool_header)

    def find_point_under_cursor(self, context, image_item, mouse_x, mouse_y):
        """
        Находит точку, ближайшую к курсору мыши и возвращает её индекс.
        Возвращает -1, если точка не найдена в пределах порогового значения.
        """
        # Получаем параметры отображения изображения
        params = get_image_display_params(context, image_item)
        if not params:
            return -1
        
        # Получаем масштаб интерфейса для корректной работы на разных разрешениях экрана
        ui_scale = context.preferences.system.ui_scale if hasattr(context.preferences.system, "ui_scale") else 1.0
        dpi_fac = context.preferences.system.dpi / 72.0 if hasattr(context.preferences.system, "dpi") else 1.0
            
        # Преобразуем координаты мыши в координаты изображения
        image_coords = mouse_to_image_coords(context, image_item, mouse_x, mouse_y)
        if not image_coords:
            return -1
            
        image_x, image_y = image_coords
        
        # Настраиваем базовый порог обнаружения в пикселях изображения с учетом DPI и UI масштаба
        base_threshold = 10.0 * ui_scale * dpi_fac
        
        # Адаптивный порог с учетом масштаба изображения
        # При большом увеличении должен быть меньше, при уменьшении - больше
        adaptive_threshold = base_threshold / max(params['scale'], 0.1)
        
        # Ограничиваем максимальное и минимальное значение порога
        min_threshold = 5.0 * ui_scale * dpi_fac
        max_threshold = 200.0 * ui_scale * dpi_fac
        adaptive_threshold = max(min(adaptive_threshold, max_threshold), min_threshold)
        
        # Сохраняем текущее значение для доступа в других методах
        self._min_point_distance = adaptive_threshold
        
        closest_point_index = -1
        min_distance = float('inf')
        
        # Перебираем все точки и ищем ближайшую
        for i, point in enumerate(image_item.points):
            if not point.is_placed:
                continue
                
            dx = point.location_2d[0] - image_x
            dy = point.location_2d[1] - image_y
            distance = (dx*dx + dy*dy) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_point_index = i
                
        # Проверяем, находится ли ближайшая точка в пределах порога
        if min_distance <= adaptive_threshold:
            return closest_point_index
        return -1

    def auto_detect_points(self, context):
        """Автоматическое определение и сопоставление точек"""
        props = context.scene.camera_calibration
        
        try:
            num_groups = auto_calibrate(props.images)
            self.report({'INFO'}, f"Автоматически создано {num_groups} групп точек")
            return True
        except Exception as e:
            self.report({'ERROR'}, f"Ошибка при автоматическом определении точек: {str(e)}")
            return False

    def get_current_image_item(self):
        """Получает текущее изображение из свойств"""
        try:
            props = bpy.context.scene.camera_calibration
            if props.active_image_index >= 0 and props.active_image_index < len(props.images):
                return props.images[props.active_image_index]
            return None
        except Exception as e:
            print(f"Ошибка при получении текущего изображения: {str(e)}")
        return None

    def draw_mirror_axis(self, context, shader, params):
        """Отрисовывает ось симметрии в редакторе изображений"""
        if not self._mirror_mode_active:
            return
        
        props = context.scene.camera_calibration
        region = context.region
        
        # Получаем размеры области
        width = region.width
        height = region.height
        
        # Определяем координаты оси в зависимости от ориентации
        if self._mirror_axis_orientation == 'vertical':
            # Вертикальная ось
            x_pos = int(width * self._mirror_axis_position)
            
            # Рисуем вертикальную линию
            shader.bind()
            shader.uniform_float("color", (0.2, 0.8, 0.2, 0.7))
            
            batch = batch_for_shader(shader, 'LINES', {
                "pos": [(x_pos, 0), (x_pos, height)]
            })
            batch.draw(shader)
            
            # Отображаем направление симметрии
            font_id = 0
            blf.size(font_id, 14)
            direction_text = "→" if self._mirror_direction == 'left_to_right' else "←"
            blf.position(font_id, x_pos + 5, height - 30, 0)
            blf.color(font_id, 0.2, 0.8, 0.2, 0.9)
            blf.draw(font_id, direction_text)
            
        else:
            # Горизонтальная ось
            y_pos = int(height * self._mirror_axis_position)
            
            # Рисуем горизонтальную линию
            shader.bind()
            shader.uniform_float("color", (0.2, 0.8, 0.2, 0.7))
            
            batch = batch_for_shader(shader, 'LINES', {
                "pos": [(0, y_pos), (width, y_pos)]
            })
            batch.draw(shader)
            
            # Отображаем направление симметрии
            font_id = 0
            blf.size(font_id, 14)
            direction_text = "↓" if self._mirror_direction == 'left_to_right' else "↑"
            blf.position(font_id, 10, y_pos - 15, 0)
            blf.color(font_id, 0.2, 0.8, 0.2, 0.9)
            blf.draw(font_id, direction_text)
    
    def draw_mirror_points_preview(self, context, shader, params):
        """Отрисовывает предпросмотр зеркальных точек в редакторе изображений"""
        if not self._mirror_mode_active:
            return
        
        props = context.scene.camera_calibration
        image_item = self.get_current_image_item()
        if not image_item:
            return
        
        # Получаем размеры изображения и области
        region = context.region
        image_width = params['image_width']
        image_height = params['image_height']
        
        # Определяем положение оси симметрии в координатах изображения
        mirror_pos = 0
        if self._mirror_axis_orientation == 'vertical':
            mirror_pos = image_width * self._mirror_axis_position
        else:
            mirror_pos = image_height * self._mirror_axis_position
        
        # Проверяем наличие существующих точек и отображаем предпросмотр зеркальных точек
        for i, point in enumerate(image_item.points):
            if not point.is_placed:
                continue
            
            # Получаем координаты точки
            x, y = point.location_2d
            
            # Определяем, с какой стороны от оси находится точка
            if self._mirror_axis_orientation == 'vertical':
                is_on_source_side = (x < mirror_pos) if self._mirror_direction == 'left_to_right' else (x > mirror_pos)
            else:
                is_on_source_side = (y < mirror_pos) if self._mirror_direction == 'left_to_right' else (y > mirror_pos)
            
            # Если точка не на исходной стороне, пропускаем её
            if not is_on_source_side:
                continue
            
            # Вычисляем координаты зеркальной точки
            mirror_x = x
            mirror_y = y
            
            if self._mirror_axis_orientation == 'vertical':
                # Для вертикальной оси симметрии меняем X-координату
                mirror_x = 2 * mirror_pos - x
            else:
                # Для горизонтальной оси симметрии меняем Y-координату
                mirror_y = 2 * mirror_pos - y
            
            # Проверяем, не выходит ли зеркальная точка за пределы изображения
            if mirror_x < 0 or mirror_x >= image_width or mirror_y < 0 or mirror_y >= image_height:
                continue
            
            # Проверяем, нет ли уже точки в этом месте или рядом
            is_point_nearby = False
            min_distance = 5.0  # Минимальное расстояние между точками
            
            for j, existing_point in enumerate(image_item.points):
                if not existing_point.is_placed or j == i:
                    continue
                
                ex, ey = existing_point.location_2d
                distance = ((mirror_x - ex) ** 2 + (mirror_y - ey) ** 2) ** 0.5
                
                if distance < min_distance:
                    is_point_nearby = True
                    break
            
            # Если рядом уже есть точка, пропускаем создание зеркальной
            if is_point_nearby:
                continue
            
            # Преобразуем координаты зеркальной точки в координаты региона
            region_coords = image_to_region_coords(context, image_item, mirror_x, mirror_y, params)
            if not region_coords:
                continue
            
            # Рисуем предпросмотр зеркальной точки
            rx, ry = region_coords
            
            # Используем цвет исходной точки
            if point.point_group_id < 0:
                # Точка без группы
                color = POINT_COLOR_UNASSIGNED
            elif point.calibration_failed:
                # Точка, которую не удалось откалибровать
                color = POINT_COLOR_FAILED
            elif point.is_calibrated:
                # Успешно откалиброванная точка
                color = POINT_COLOR_CALIBRATED
            else:
                # Точка в группе, но не откалиброванная
                color = POINT_COLOR_UNCALIBRATED
            
            # Делаем цвет полупрозрачным для предпросмотра
            color = (color[0], color[1], color[2], 0.5)
            
            # Определяем размер отображаемой точки
            point_size = 7
            
            # Рисуем предпросмотр точки
            draw_point(shader, rx, ry, point_size, color)
            
            # Рисуем номер точки
            font_id = 0
            blf.position(font_id, rx + 10, ry + 10, 0)
            blf.size(font_id, 12)
            blf.color(font_id, *color)
            
            # Отображаем номер группы, если точка в группе
            if point.point_group_id >= 0:
                blf.draw(font_id, str(point.point_group_id + 1))
            else:
                blf.draw(font_id, str(i + 1))
            
            # Сохраняем информацию о предпросмотре для последующего создания
            self._mirror_point_preview.append((mirror_x, mirror_y, i))

    def toggle_mirror_mode(self, context):
        """Переключает режим симметрии"""
        self._mirror_mode_active = not self._mirror_mode_active
        
        # Если включен режим симметрии, устанавливаем ось по умолчанию в центре
        if self._mirror_mode_active:
            self._mirror_axis_position = 0.5
            self._mirror_axis_orientation = 'vertical'  # По умолчанию вертикальная ось
        
        return True

    def create_mirror_point(self, context, original_point_idx):
        """Создает зеркальную точку на основе оригинальной точки"""
        props = context.scene.camera_calibration
        image_item = self.get_current_image_item()
        
        if not image_item or original_point_idx < 0 or original_point_idx >= len(image_item.points):
            return False
        
        # Получаем оригинальную точку
        orig_point = image_item.points[original_point_idx]
        
        # Находим зеркальную точку в предварительном списке
        for mirror_x, mirror_y, idx in self._mirror_point_preview:
            if idx == original_point_idx:
                # Проверяем, нет ли уже точки в этом месте или рядом
                is_point_nearby = False
                min_distance = 5.0  # Минимальное расстояние между точками
                
                for j, existing_point in enumerate(image_item.points):
                    if not existing_point.is_placed or j == original_point_idx:
                        continue
                    
                    ex, ey = existing_point.location_2d
                    distance = ((mirror_x - ex) ** 2 + (mirror_y - ey) ** 2) ** 0.5
                    
                    if distance < min_distance:
                        is_point_nearby = True
                        self.report({'WARNING'}, f"Не удалось создать зеркальную точку: в этом месте уже есть точка")
                        break
                
                # Если рядом уже есть точка, пропускаем создание зеркальной
                if is_point_nearby:
                    continue
                
                # Создаем новую точку
                point = image_item.points.add()
                point.location_2d = (mirror_x, mirror_y)
                point.is_placed = True
                mirror_group_id = _ensure_paired_mirror_group(props, orig_point.point_group_id)
                point.point_group_id = mirror_group_id

                if mirror_group_id >= 0:
                    self.report(
                        {'INFO'},
                        f"Создана зеркальная точка {len(image_item.points)} в группе "
                        f"{self._get_group_name(props, mirror_group_id)}"
                    )
                else:
                    self.report({'INFO'}, f"Создана зеркальная точка {len(image_item.points)}")
                return True
        
        self.report({'WARNING'}, "Не удалось найти подходящую зеркальную точку")
        return False

    def __del__(self):
        """Деструктор для очистки ресурсов"""
        try:
            if hasattr(self, '_handle') and self._handle:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
                except (ValueError, TypeError, ReferenceError, AttributeError):
                    pass
        except:
            # Игнорируем все ошибки в деструкторе
            pass

    def select_points_in_box(self, context, image_item, add_to_selection=False):
        """
        Выбирает точки внутри рамки выделения
        
        Args:
            context: Контекст Blender
            image_item: Элемент изображения
            add_to_selection: Если True, добавляет к существующему выделению
        """
        # Получаем координаты рамки
        min_x = min(self._box_select_start_x, self._box_select_end_x)
        max_x = max(self._box_select_start_x, self._box_select_end_x)
        min_y = min(self._box_select_start_y, self._box_select_end_y)
        max_y = max(self._box_select_start_y, self._box_select_end_y)
        
        # Если рамка слишком маленькая, считаем это случайным кликом
        if max_x - min_x < 5 or max_y - min_y < 5:
            return
        
        # Получаем параметры отображения изображения
        params = get_image_display_params(context, image_item)
        if not params:
            return
        
        # Если не нужно добавлять к существующему выделению, очищаем список
        if not add_to_selection:
            self._selected_points = []
        
        # Перебираем все точки и проверяем, находятся ли они внутри рамки
        for i, point in enumerate(image_item.points):
            if not point.is_placed:
                continue
                
            # Преобразуем координаты точки из координат изображения в координаты региона
            region_coords = image_to_region_coords(context, image_item, point.location_2d[0], point.location_2d[1], params)
            if not region_coords:
                continue
                
            x, y = region_coords
            
            # Проверяем, находится ли точка внутри рамки
            if min_x <= x <= max_x and min_y <= y <= max_y:
                # Если точка еще не выбрана, добавляем ее в список
                if i not in self._selected_points:
                    self._selected_points.append(i)
                    
        # Выводим информацию о количестве выделенных точек
        if len(self._selected_points) > 0:
            self.report({'INFO'}, f"Выделено {len(self._selected_points)} точек")
        else:
            self.report({'INFO'}, "Нет точек в области выделения")

    def draw_selection_box(self, context, shader):
        """Отрисовывает рамку выделения"""
        if not self._is_box_selecting:
            return
            
        # Получаем координаты рамки выделения
        start_x = self._box_select_start_x
        start_y = self._box_select_start_y
        end_x = self._box_select_end_x
        end_y = self._box_select_end_y
        
        # Вершины прямоугольника
        vertices = [
            (start_x, start_y),
            (end_x, start_y),
            (end_x, end_y),
            (start_x, end_y)
        ]
        
        # Сохраняем текущее состояние смешивания и толщину линии
        current_blend_mode = gpu.state.blend_get()
        current_line_width = gpu.state.line_width_get()
        
        # Устанавливаем правильный режим смешивания для наложения цветов
        gpu.state.blend_set('ALPHA')
        
        # Рисуем полупрозрачное заполнение с более правильным накладываемым цветом
        batch = batch_for_shader(shader, 'TRI_FAN', {"pos": vertices})
        shader.bind()
        shader.uniform_float("color", (0.9, 0.5, 0.2, 0.12))  # Оптимизируем прозрачность
        batch.draw(shader)
        
        # Рисуем рамку выделения с повышенной видимостью
        gpu.state.line_width_set(2.0)  # Делаем линию заметнее
        
        # Внутренняя рамка (темнее)
        batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": vertices})
        shader.bind()
        shader.uniform_float("color", (0.9, 0.6, 0.2, 0.9))  # Более яркий и непрозрачный оранжевый
        batch.draw(shader)
        
        # Внешняя рамка (обводка для лучшей видимости)
        # Смещаем вершины для внешней обводки на 1 пиксель наружу
        outline_vertices = [
            (start_x - 1, start_y - 1),
            (end_x + 1, start_y - 1),
            (end_x + 1, end_y + 1),
            (start_x - 1, end_y + 1)
        ]
        batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": outline_vertices})
        shader.bind()
        shader.uniform_float("color", (0.0, 0.0, 0.0, 0.7))  # Темная обводка для контраста
        batch.draw(shader)
        
        # Восстанавливаем предыдущие настройки
        gpu.state.line_width_set(current_line_width)
        gpu.state.blend_set(current_blend_mode)

# Вспомогательная функция для отрисовки точки
def draw_point(shader, x, y, size, color):
    """Рисует точку с заданными параметрами"""
    import math
    from gpu_extras.batch import batch_for_shader
    
    # Создаем круг из точек
    segments = 16
    circle_vertices = []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        circle_vertices.append((x + size * math.cos(angle), y + size * math.sin(angle)))
    
    # Рисуем заполненный круг
    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": [(x, y)] + circle_vertices})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)

# Вспомогательная функция для предварительной загрузки изображений в кэш
def preload_images(props):
    """Предварительно загружает изображения в кэш для ускорения работы"""
    global _image_cache
    import bpy
    import time
    import os
    
    # Определяем, какие изображения нужно загрузить
    active_index = props.active_image_index
    if active_index < 0 or active_index >= len(props.images):
        return
    
    # Загружаем активное изображение и соседние
    indices_to_load = [active_index]
    if active_index > 0:
        indices_to_load.append(active_index - 1)
    if active_index < len(props.images) - 1:
        indices_to_load.append(active_index + 1)
    
    # Загружаем выбранные изображения в кэш
    for idx in indices_to_load:
        if idx >= 0 and idx < len(props.images):
            image_item = props.images[idx]
            if image_item.name not in _image_cache:
                # Загружаем изображение
                image = None
                for img in bpy.data.images:
                    if img.name == image_item.name:
                        image = img
                        break
                
                if not image and hasattr(image_item, 'filepath') and image_item.filepath:
                    for img in bpy.data.images:
                        if os.path.normpath(img.filepath) == os.path.normpath(image_item.filepath):
                            image = img
                            break
                
                if image:
                    # Добавляем в кэш
                    _image_cache[image_item.name] = {
                        'width': image.size[0],
                        'height': image.size[1],
                        'image': image,
                        'timestamp': time.time()
                    }
    
    # Очищаем старые элементы кэша
    current_time = time.time()
    cache_timeout = 60  # Время в секундах, после которого изображение удаляется из кэша
    
    # Создаем список ключей для удаления
    keys_to_remove = []
    for key, cache_entry in _image_cache.items():
        # Проверяем, является ли изображение активным или соседним
        is_active_or_neighbor = False
        for idx in indices_to_load:
            if idx >= 0 and idx < len(props.images) and props.images[idx].name == key:
                is_active_or_neighbor = True
                break
        
        # Если это не активное/соседнее изображение и оно старше timeout, удаляем его из кэша
        if not is_active_or_neighbor and current_time - cache_entry['timestamp'] > cache_timeout:
            keys_to_remove.append(key)
    
    # Удаляем старые элементы кэша
    for key in keys_to_remove:
        del _image_cache[key]

class CAMCALIB_OT_create_mirrored_images(Operator):
    """Создает зеркальные копии изображений с отраженными точками"""
    bl_idname = "camera_calibration.create_mirrored_images"
    bl_label = "Создать зеркальные изображения"
    bl_description = "Создает зеркальные копии всех изображений, которые не отмечены как исключенные из зеркалирования"
    
    def execute(self, context):
        props = context.scene.camera_calibration
        
        # Сохраняем оригинальный активный индекс
        original_active_index = props.active_image_index
        
        # Список индексов изображений, которые будут созданы
        images_to_mirror = []
        
        # Собираем индексы изображений для зеркалирования
        for idx, img in enumerate(props.images):
            # Пропускаем уже зеркальные изображения и те, что отмечены как исключенные
            if not img.name.startswith("Mirror_") and not img.is_mirror_excluded:
                images_to_mirror.append(idx)
        
        if not images_to_mirror:
            self.report({'WARNING'}, "Нет подходящих изображений для зеркалирования")
            return {'CANCELLED'}
        
        mirrored_count = 0
        try:
            # Импортируем numpy для обработки изображений
            import numpy as np
            
            # Создаем зеркальные копии изображений
            for idx in images_to_mirror:
                original_img = props.images[idx]
                
                # Проверяем, есть ли уже зеркальная копия этого изображения
                mirror_name = f"Mirror_{original_img.name}"
                existing_mirror = next((i for i, img in enumerate(props.images) if img.name == mirror_name), None)
                
                # Получаем объект изображения из Blender
                original_blender_image = None
                for img in bpy.data.images:
                    if img.name == original_img.name:
                        original_blender_image = img
                        break
                
                # Если не нашли изображение, пробуем по пути
                if original_blender_image is None and original_img.filepath:
                    for img in bpy.data.images:
                        if img.filepath == original_img.filepath:
                            original_blender_image = img
                            break
                
                # Если всё равно не нашли, пропускаем это изображение
                if original_blender_image is None:
                    self.report({'WARNING'}, f"Не удалось найти изображение {original_img.name}")
                    continue
                
                # Получаем размеры изображения
                width = original_blender_image.size[0]
                height = original_blender_image.size[1]
                
                # Проверяем, есть ли уже зеркальное изображение в bpy.data.images
                mirror_blender_image = bpy.data.images.get(mirror_name)
                
                # Если зеркального изображения ещё нет, создаем его
                if mirror_blender_image is None:
                    # Создаем новое изображение с тем же размером
                    mirror_blender_image = bpy.data.images.new(
                        name=mirror_name,
                        width=width,
                        height=height
                    )
                
                # Получаем пиксельные данные оригинального изображения
                pixel_count = width * height * 4  # RGBA формат
                original_pixels = np.zeros(pixel_count, dtype=np.float32)
                original_blender_image.pixels.foreach_get(original_pixels)
                
                # Преобразуем пиксельные данные в формат для зеркалирования
                original_pixels = original_pixels.reshape(height, width, 4)
                
                # Создаем зеркальное отражение по горизонтали
                mirrored_pixels = np.flip(original_pixels, axis=1).copy()
                
                # Преобразуем обратно в плоский массив для Blender
                mirrored_pixels = mirrored_pixels.flatten()
                
                # Записываем данные в новое изображение
                mirror_blender_image.pixels.foreach_set(mirrored_pixels)
                
                # Обновляем изображение, чтобы изменения стали видимыми
                mirror_blender_image.update()
                
                if existing_mirror is not None:
                    # Если зеркальная копия уже существует в коллекции, обновляем её
                    mirror_idx = existing_mirror
                    mirror_img = props.images[mirror_idx]
                else:
                    # Создаем новое зеркальное изображение в коллекции
                    mirror_img = props.images.add()
                    mirror_img.name = mirror_name
                    mirror_img.filepath = original_img.filepath
                    
                    # Новое изображение будет последним в списке
                    mirror_idx = len(props.images) - 1
                
                # Копируем все точки с зеркальным отражением
                mirror_img.points.clear()
                for point in original_img.points:
                    if point.is_placed:
                        # Создаем зеркальную точку
                        mirror_point = mirror_img.points.add()
                        # Зеркально отражаем координаты по горизонтали (инвертируем X)
                        mirror_point.location_2d = (width - point.location_2d[0], point.location_2d[1])
                        mirror_point.is_placed = True
                        mirror_point.point_group_id = _ensure_paired_mirror_group(props, point.point_group_id)
                
                # Копируем остальные свойства отображения
                mirror_img.scale = original_img.scale
                mirror_img.offset_x = original_img.offset_x
                mirror_img.offset_y = original_img.offset_y
                
                # Отмечаем как зеркальное изображение
                mirror_img.is_mirror = True
                mirror_img.mirror_source_index = idx
                
                mirrored_count += 1
            
            # Восстанавливаем активный индекс
            props.active_image_index = original_active_index
            
            if mirrored_count > 0:
                self.report({'INFO'}, f"Создано {mirrored_count} зеркальных изображений")
                # Очищаем кэш изображений, чтобы заставить перезагрузить изображения
                global _image_cache
                for key in list(_image_cache.keys()):
                    if key.startswith("Mirror_"):
                        del _image_cache[key]
                
                # Обновляем интерфейс для отображения новых изображений
                for area in context.screen.areas:
                    area.tag_redraw()
                    
                return {'FINISHED'}
            else:
                self.report({'WARNING'}, "Не удалось создать зеркальные изображения")
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Ошибка при создании зеркальных изображений: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

# Функции регистрации и отмены регистрации оператора
def register():
    bpy.utils.register_class(CAMCALIB_OT_image_editor)
    bpy.utils.register_class(CAMCALIB_OT_create_mirrored_images)
    print("Image Editor: модуль успешно зарегистрирован")

def unregister():
    bpy.utils.unregister_class(CAMCALIB_OT_create_mirrored_images)
    bpy.utils.unregister_class(CAMCALIB_OT_image_editor)
    print("Image Editor: модуль успешно отменил регистрацию")

if __name__ == "__main__":
    register()
