try:
    import bpy
except:
    print("Использует тест без Блендер")
    pass
from bpy.types import Operator
from bpy.props import StringProperty, FloatProperty, IntProperty, CollectionProperty, BoolProperty, EnumProperty
import os
from mathutils import Vector, Matrix
from . import utils
from . import image_editor
from . import calibration
from . import calibration_bridge
import numpy as np
import math

# Импортируем константы из image_editor
from .image_editor import POINT_COLOR_UNASSIGNED, POINT_COLOR_UNCALIBRATED

# Предопределенные контрастные цвета в формате RGB
PREDEFINED_COLORS = [
    (1.0, 0.0, 0.0),    # Красный
    (0.0, 1.0, 0.0),    # Зеленый
    (0.0, 0.0, 1.0),    # Синий
    (1.0, 0.5, 0.0),    # Оранжевый
    (0.5, 0.0, 1.0),    # Фиолетовый
    (0.0, 1.0, 1.0),    # Голубой
    (1.0, 0.0, 0.5),    # Розовый
    (0.5, 1.0, 0.0),    # Желто-зеленый
    (0.0, 0.5, 1.0),    # Индиго
    (1.0, 1.0, 0.0),    # Желтый
    (0.0, 0.8, 0.4),    # Весенне-зеленый
    (0.0, 0.8, 0.8),    # Морской
]

# Импортируем мост к новой системе калибровки
from . import calibration_bridge

# Импортируем модуль кривых Безье
try:
    from .bezier_module import get_bezier_points, reset_bezier_state, reset_bezier_curves
    bezier_module_imported = True
except ImportError:
    bezier_module_imported = False
    print("Внимание: модуль bezier_module не найден или не может быть импортирован")


def _count_grouped_points_on_image(image_item):
    return sum(1 for p in image_item.points if p.is_placed and p.point_group_id >= 0)


def _get_images_with_grouped_points(props):
    return [
        (index, image_item, _count_grouped_points_on_image(image_item))
        for index, image_item in enumerate(props.images)
        if _count_grouped_points_on_image(image_item) > 0
    ]

class CAMCALIB_OT_load_image(Operator):
    """Загрузить изображения для калибровки"""
    bl_idname = "camera_calibration.load_image"
    bl_label = "Загрузить изображения"
    
    directory: StringProperty(
        name="Директория",
        subtype='DIR_PATH'
    )
    
    files: CollectionProperty(
        name="Файлы",
        type=bpy.types.OperatorFileListElement
    )
    
    filter_glob: StringProperty(
        default="*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp",
        options={'HIDDEN'}
    )
    
    def execute(self, context):
        props = context.scene.camera_calibration
        loaded_count = 0
        reloaded_count = 0
        failed_count = 0
        skipped_count = 0
        
        print(f"Количество выбранных файлов: {len(self.files)}")  # Отладка
        
        for file_elem in self.files:
            filepath = os.path.join(self.directory, file_elem.name)
            print(f"Обработка файла: {filepath}")  # Отладка
            
            # Проверяем, есть ли уже такое изображение в списке
            image_name = os.path.basename(filepath)
            already_loaded = False
            existing_image_index = -1
            
            for i, img in enumerate(props.images):
                if os.path.normpath(img.filepath) == os.path.normpath(filepath):
                    already_loaded = True
                    existing_image_index = i
                    break
            
            # Проверяем, есть ли изображение в данных Blender
            blender_image_exists = False
            if already_loaded:
                for img in bpy.data.images:
                    if img.name == props.images[existing_image_index].name:
                        blender_image_exists = True
                        break
            
            # Если изображение уже загружено и существует в Blender, пропускаем его
            if already_loaded and blender_image_exists:
                print(f"Файл уже загружен: {filepath}")  # Отладка
                skipped_count += 1
                continue
            
            # Если изображение в списке, но отсутствует в Blender - перезагружаем
            if already_loaded and not blender_image_exists:
                print(f"Восстановление потерянного файла: {filepath}")  # Отладка
                image = utils.load_image_to_blender(filepath)
                if image is None:
                    failed_count += 1
                    print(f"Не удалось загрузить файл: {filepath}")  # Отладка
                    continue
                
                # Обновляем существующий элемент в списке
                props.images[existing_image_index].name = image.name
                props.images[existing_image_index].filepath = filepath
                reloaded_count += 1
                print(f"Успешно восстановлен файл: {filepath}")  # Отладка
                continue
            
            # Загружаем новое изображение
            image = utils.load_image_to_blender(filepath)
            if image is None:
                failed_count += 1
                print(f"Не удалось загрузить файл: {filepath}")  # Отладка
                continue
            
            # Добавляем информацию об изображении в коллекцию
            img_item = props.images.add()
            img_item.name = image.name
            img_item.filepath = filepath
            loaded_count += 1
            print(f"Успешно загружен файл: {filepath}")  # Отладка
        
        # Устанавливаем активное изображение на последнее загруженное
        if loaded_count > 0 or reloaded_count > 0:
            props.active_image_index = len(props.images) - 1
            print(f"Установлен активный индекс: {props.active_image_index}")
            print(f"Всего изображений: {len(props.images)}")
            print(f"Активное изображение: {props.images[props.active_image_index].name}")
            # Обновляем интерфейс
            for area in context.screen.areas:
                area.tag_redraw()
        
        # Формируем отчет о загрузке
        message_parts = []
        if loaded_count > 0:
            message_parts.append(f"Загружено {loaded_count} изображений")
        if reloaded_count > 0:
            message_parts.append(f"Восстановлено {reloaded_count} изображений")
        if skipped_count > 0:
            message_parts.append(f"Пропущено {skipped_count} (уже загружены)")
        if failed_count > 0:
            message_parts.append(f"Не удалось загрузить {failed_count}")
        
        if message_parts:
            self.report(
                {'INFO' if failed_count == 0 else 'WARNING'},
                ", ".join(message_parts)
            )
        else:
            self.report({'ERROR'}, "Не выбрано ни одного изображения")
            return {'CANCELLED'}
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class CAMCALIB_OT_remove_image(Operator):
    """Удалить изображение из списка"""
    bl_idname = "camera_calibration.remove_image"
    bl_label = "Удалить изображение"
    bl_options = {'REGISTER', 'UNDO'}
    
    index: IntProperty(name="Индекс", default=0)
    
    def execute(self, context):
        props = context.scene.camera_calibration
        
        # Проверяем, что индекс в допустимом диапазоне
        if self.index >= 0 and self.index < len(props.images):
            # Удаляем изображение из списка
            props.images.remove(self.index)
            
            # Если активное изображение было удалено, обновляем активный индекс
            if props.active_image_index >= len(props.images):
                props.active_image_index = max(0, len(props.images) - 1)
            
            self.report({'INFO'}, "Изображение удалено")
        else:
            self.report({'ERROR'}, "Неверный индекс изображения")
        
        return {'FINISHED'}

class CAMCALIB_OT_remove_all_images(Operator):
    """Удалить все изображения из списка"""
    bl_idname = "camera_calibration.remove_all_images"
    bl_label = "Удалить все изображения"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.camera_calibration
        
        # Удаляем все изображения
        props.images.clear()
        props.active_image_index = 0
        
        self.report({'INFO'}, "Все изображения удалены")
        return {'FINISHED'}

class CAMCALIB_OT_add_point(Operator):
    """Добавить опорную точку"""
    bl_idname = "camera_calibration.add_point"
    bl_label = "Добавить точку"
    
    def execute(self, context):
        # TODO: Реализовать добавление точки
        return {'FINISHED'}

class CAMCALIB_OT_add_point_group(Operator):
    """Добавить новую группу точек"""
    bl_idname = "camera_calibration.add_point_group"
    bl_label = "Добавить группу"
    
    def execute(self, context):
        props = context.scene.camera_calibration
        group = props.point_groups.add()
        group.name = f"Группа {len(props.point_groups)}"
        
        # Назначаем цвет из предопределенного списка
        color_index = (len(props.point_groups) - 1) % len(PREDEFINED_COLORS)
        group.color = PREDEFINED_COLORS[color_index]
        
        props.active_point_group_index = len(props.point_groups) - 1
        return {'FINISHED'}

class CAMCALIB_OT_remove_point_group(Operator):
    """Удалить группу точек"""
    bl_idname = "camera_calibration.remove_point_group"
    bl_label = "Удалить группу"
    
    def execute(self, context):
        props = context.scene.camera_calibration
        if props.active_point_group_index >= 0:
            group_id = props.active_point_group_index
            # Удаляем связи с этой группой у всех точек
            for img in props.images:
                for point in img.points:
                    if point.point_group_id == group_id:
                        point.point_group_id = -1
                    elif point.point_group_id > group_id:
                        point.point_group_id -= 1
            
            # Удаляем группу
            props.point_groups.remove(props.active_point_group_index)
            if props.active_point_group_index >= len(props.point_groups):
                props.active_point_group_index = len(props.point_groups) - 1
        return {'FINISHED'}

class CAMCALIB_OT_remove_point_from_group(Operator):
    """Исключить точку из группы"""
    bl_idname = "camera_calibration.remove_point_from_group"
    bl_label = "Исключить из группы"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.camera_calibration
        
        # Проверяем, есть ли активное изображение
        if props.active_image_index < 0 or props.active_image_index >= len(props.images):
            self.report({'ERROR'}, "Нет активного изображения")
            return {'CANCELLED'}
        
        # Получаем активное изображение
        image_item = props.images[props.active_image_index]
        
        # Проверяем, есть ли активная точка
        if image_item.active_point_index < 0 or image_item.active_point_index >= len(image_item.points):
            self.report({'ERROR'}, "Нет активной точки")
            return {'CANCELLED'}
        
        # Получаем активную точку
        point = image_item.points[image_item.active_point_index]
        
        # Проверяем, принадлежит ли точка группе
        if point.point_group_id < 0:
            self.report({'WARNING'}, "Точка уже не принадлежит ни одной группе")
            return {'CANCELLED'}
        
        # Сохраняем старое значение для отчета
        old_group_id = point.point_group_id
        
        # Исключаем точку из группы
        point.point_group_id = -1
        
        # Отчитываемся об успешном исключении
        self.report({'INFO'}, f"Точка исключена из группы {props.point_groups[old_group_id].name}")
        
        return {'FINISHED'}

class CAMCALIB_OT_clear_active_group(Operator):
    """Сбросить активную группу точек, чтобы новые точки создавались без привязки к группе"""
    bl_idname = "camera_calibration.clear_active_group"
    bl_label = "Сбросить группу"
    
    def execute(self, context):
        props = context.scene.camera_calibration
        props.active_point_group_index = -1
        self.report({'INFO'}, "Активная группа сброшена. Новые точки будут создаваться без привязки к группе.")
        return {'FINISHED'}

# Добавляем новый оператор для калибровки с использованием новой системы
class CAMCALIB_OT_run_advanced_calibration(Operator):
    """Запустить расширенную калибровку камеры"""
    bl_idname = "camera_calibration.run_advanced_calibration"
    bl_label = "Расширенная калибровка"
    bl_options = {'REGISTER', 'UNDO'}
    
    create_cameras: BoolProperty(
        name="Создать камеры",
        description="Создать объекты камер в сцене",
        default=True
    )

    quality_profile: EnumProperty(
        name="Режим",
        description="Готовый режим расширенной калибровки",
        items=[
            ('FAST', "Быстро", "Быстрее работает, но обычно оставляет меньше точек"),
            ('BALANCED', "Баланс", "Рекомендуемый режим для большинства сцен"),
            ('PRECISE', "Точно", "Строже к ошибкам, медленнее, обычно дает более чистый результат"),
            ('CUSTOM', "Свои настройки", "Ручная настройка всех параметров"),
        ],
        default='BALANCED'
    )

    show_technical_settings: BoolProperty(
        name="Показать тонкие настройки",
        description="Показать технические параметры solver-а",
        default=False
    )

    # Добавляем параметры для расширенной калибровки
    min_points_for_camera: IntProperty(
        name="Мин. точек для камеры",
        description="Минимальное количество общих точек для добавления камеры",
        default=4,
        min=3,
        max=20
    )
    
    bundle_method: EnumProperty(
        name="Метод оптимизации",
        description="Метод оптимизации для bundle adjustment",
        items=[
            ('trf', "Trust Region", "Trust Region Reflective - стабильный метод"),
            ('dogbox', "Dogbox", "Dogbox - быстрый метод для задач малой размерности"),
            ('lm', "Левенберг-Марквардт", "Левенберг-Марквардт - классический метод")
        ],
        default='trf'
    )
    
    bundle_ftol: FloatProperty(
        name="Точность оптимизации",
        description="Порог сходимости по функции для bundle adjustment",
        default=1e-8,
        min=1e-12,
        max=1e-4,
        soft_min=1e-10,
        soft_max=1e-6
    )
    
    max_bundle_iterations: IntProperty(
        name="Макс. итераций",
        description="Максимальное количество итераций bundle adjustment",
        default=3,
        min=1,
        max=10
    )
    
    ransac_threshold: FloatProperty(
        name="Порог RANSAC",
        description="Порог для RANSAC при оценке позы камеры",
        default=8.0,
        min=1.0,
        max=20.0
    )
    
    confidence: FloatProperty(
        name="Уровень доверия",
        description="Уровень доверия для RANSAC",
        default=0.99,
        min=0.8,
        max=0.999
    )
    
    max_attempts: IntProperty(
        name="Макс. попыток",
        description="Максимальное количество попыток добавления камер",
        default=3,
        min=1,
        max=10
    )
    
    # Добавляем новый параметр для максимальной ошибки репроекции
    max_reprojection_error: FloatProperty(
        name="Макс. ошибка репроекции",
        description="Максимальная допустимая ошибка репроекции при фильтрации точек (пикселей)",
        default=10.0,
        min=1.0,
        max=50.0
    )

    def _get_resolved_settings(self):
        if self.quality_profile == 'CUSTOM':
            return {
                "min_points_for_camera": self.min_points_for_camera,
                "bundle_method": self.bundle_method,
                "bundle_ftol": self.bundle_ftol,
                "max_bundle_iterations": self.max_bundle_iterations,
                "ransac_threshold": self.ransac_threshold,
                "confidence": self.confidence,
                "max_attempts": self.max_attempts,
                "max_reprojection_error": self.max_reprojection_error,
            }

        presets = {
            'FAST': {
                "min_points_for_camera": 4,
                "bundle_method": 'trf',
                "bundle_ftol": 1e-6,
                "max_bundle_iterations": 2,
                "ransac_threshold": 10.0,
                "confidence": 0.98,
                "max_attempts": 2,
                "max_reprojection_error": 12.0,
            },
            'BALANCED': {
                "min_points_for_camera": 4,
                "bundle_method": 'trf',
                "bundle_ftol": 1e-8,
                "max_bundle_iterations": 3,
                "ransac_threshold": 8.0,
                "confidence": 0.99,
                "max_attempts": 3,
                "max_reprojection_error": 10.0,
            },
            'PRECISE': {
                "min_points_for_camera": 5,
                "bundle_method": 'trf',
                "bundle_ftol": 1e-9,
                "max_bundle_iterations": 4,
                "ransac_threshold": 6.0,
                "confidence": 0.995,
                "max_attempts": 4,
                "max_reprojection_error": 6.0,
            },
        }
        return presets[self.quality_profile]

    def _get_profile_help_lines(self):
        if self.quality_profile == 'FAST':
            return [
                "Подходит для быстрого чернового результата.",
                "Обычно быстрее, но может оставить меньше точек в сцене.",
            ]
        if self.quality_profile == 'PRECISE':
            return [
                "Подходит, когда важнее чистота и точность, чем плотность облака.",
                "Работает медленнее и строже отбрасывает конфликтные треки.",
            ]
        if self.quality_profile == 'CUSTOM':
            return [
                "Полностью ручной режим.",
                "Используйте только если понимаете, что делают параметры ниже.",
            ]
        return [
            "Рекомендуемый режим по умолчанию.",
            "Обычно дает лучший компромисс между скоростью, плотностью и качеством.",
        ]
    
    def execute(self, context):
        props = context.scene.camera_calibration
        wm = context.window_manager

        def update_progress(progress_value, status_text):
            progress_value = float(max(0.0, min(100.0, progress_value)))
            if progress_value + 1e-6 < float(props.calibration_progress):
                return
            props.calibration_progress = progress_value
            props.calibration_status = str(status_text or "")
            try:
                wm.progress_update(progress_value)
            except Exception:
                pass

            try:
                for window in wm.windows:
                    screen = getattr(window, "screen", None)
                    if screen is None:
                        continue
                    for area in screen.areas:
                        area.tag_redraw()
            except Exception:
                pass
        
        images_with_points = _get_images_with_grouped_points(props)

        # Проверяем наличие изображений и точек
        if len(images_with_points) < 2:
            self.report({'ERROR'}, "Недостаточно изображений с точками для калибровки (минимум 2)")
            return {'CANCELLED'}
        
        # Проверяем наличие групп точек
        if len(props.point_groups) == 0:
            self.report({'ERROR'}, "Нет групп точек для калибровки")
            return {'CANCELLED'}
        
        # Пустые изображения исключаем из калибровки.
        # Проверку минимального числа точек оставляем только для кадров, где точки есть.
        min_points_per_image = 6
        for _, image_item, points_count in images_with_points:
            if points_count < min_points_per_image:
                self.report({'ERROR'}, f"На изображении {image_item.name} недостаточно точек (минимум {min_points_per_image})")
                return {'CANCELLED'}

        props.calibration_in_progress = True
        props.calibration_progress = 0.0
        props.calibration_status = "Подготовка к запуску..."
        wm.progress_begin(0, 100)

        try:
            resolved_settings = self._get_resolved_settings()

            # Запускаем калибровку через мост с настраиваемыми параметрами
            self.report({'INFO'}, "Запуск расширенной калибровки...")
            update_progress(2.0, "Запуск калибровки...")
            result = calibration_bridge.run_calibration_from_addon(
                props,
                min_points_for_camera=resolved_settings["min_points_for_camera"],
                bundle_method=resolved_settings["bundle_method"],
                bundle_ftol=resolved_settings["bundle_ftol"],
                max_bundle_iterations=resolved_settings["max_bundle_iterations"],
                ransac_threshold=resolved_settings["ransac_threshold"],
                confidence=resolved_settings["confidence"],
                max_attempts=resolved_settings["max_attempts"],
                max_reprojection_error=resolved_settings["max_reprojection_error"],
                create_cameras=self.create_cameras,
                progress_callback=update_progress,
            )
            
            if not result:
                props.calibration_progress = 0.0
                props.calibration_status = "Ошибка при выполнении калибровки"
                self.report({'ERROR'}, "Ошибка при выполнении калибровки")
                return {'CANCELLED'}
            
            # Устанавливаем флаг калибровки
            props.is_calibrated = True
            props.calibration_progress = 100.0
            props.calibration_status = "Калибровка завершена"
            
            self.report({'INFO'}, "Калибровка успешно завершена")
            return {'FINISHED'}
        finally:
            props.calibration_in_progress = False
            try:
                wm.progress_end()
            except Exception:
                pass
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=460)
    
    def draw(self, context):
        layout = self.layout

        mode_box = layout.box()
        mode_box.label(text="Режим калибровки", icon='CAMERA_DATA')
        mode_box.prop(self, "quality_profile", text="Профиль")
        for help_line in self._get_profile_help_lines():
            mode_box.label(text=help_line, icon='INFO')

        result_box = layout.box()
        result_box.label(text="Что сделать после расчета", icon='OUTLINER_OB_CAMERA')
        result_box.prop(self, "create_cameras")

        resolved = self._get_resolved_settings()
        summary_box = layout.box()
        summary_box.label(text="Что будет использоваться", icon='PRESET')
        summary_box.label(
            text=f"Добавление камер: минимум {resolved['min_points_for_camera']} общих точек, до {resolved['max_attempts']} попыток"
        )
        summary_box.label(
            text=f"Устойчивость: RANSAC {resolved['ransac_threshold']:.1f}px, confidence {resolved['confidence']:.3f}"
        )
        summary_box.label(
            text=f"Оптимизация: {resolved['bundle_method']}, {resolved['max_bundle_iterations']} прохода(ов)"
        )
        summary_box.label(
            text=f"Фильтрация: допустимая ошибка {resolved['max_reprojection_error']:.1f}px"
        )

        if self.quality_profile != 'CUSTOM':
            layout.prop(self, "show_technical_settings", text="Показать тонкие настройки")
            show_technical = self.show_technical_settings
        else:
            show_technical = True

        if show_technical:
            tech_box = layout.box()
            tech_box.label(text="Тонкие настройки solver-а", icon='SETTINGS')

            camera_box = tech_box.box()
            camera_box.label(text="Добавление камер")
            row = camera_box.row()
            row.prop(self, "min_points_for_camera")
            row.prop(self, "max_attempts")

            robust_box = tech_box.box()
            robust_box.label(text="Устойчивость к выбросам")
            row = robust_box.row()
            row.prop(self, "ransac_threshold")
            row.prop(self, "confidence")
            robust_box.prop(self, "max_reprojection_error")

            optimize_box = tech_box.box()
            optimize_box.label(text="Финальная оптимизация")
            row = optimize_box.row()
            row.prop(self, "bundle_method")
            row.prop(self, "max_bundle_iterations")
            optimize_box.prop(self, "bundle_ftol")

# Новый оператор для сохранения результатов калибровки
class CAMCALIB_OT_save_calibration(Operator):
    """Сохранить результаты калибровки в файл"""
    bl_idname = "camera_calibration.save_calibration"
    bl_label = "Сохранить калибровку"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: StringProperty(
        name="Файл калибровки",
        description="Путь для сохранения данных калибровки",
        default="calibration.json",
        subtype='FILE_PATH'
    )
    
    def execute(self, context):
        props = context.scene.camera_calibration
        
        # Проверяем, что калибровка выполнена
        if not props.is_calibrated:
            self.report({'ERROR'}, "Калибровка не выполнена")
            return {'CANCELLED'}
        
        # Сохраняем калибровку через мост
        from calibration import save_calibration
        
        result = save_calibration(self.filepath)
        
        if result:
            self.report({'INFO'}, f"Калибровка сохранена в {self.filepath}")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Ошибка при сохранении калибровки")
            return {'CANCELLED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

# Новый оператор для загрузки результатов калибровки
class CAMCALIB_OT_load_calibration(Operator):
    """Загрузить результаты калибровки из файла"""
    bl_idname = "camera_calibration.load_calibration"
    bl_label = "Загрузить калибровку"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: StringProperty(
        name="Файл калибровки",
        description="Путь к файлу калибровки",
        default="calibration.json",
        subtype='FILE_PATH'
    )
    
    def execute(self, context):
        props = context.scene.camera_calibration
        
        # Загружаем калибровку через мост
        from calibration import load_calibration
        
        result = load_calibration(self.filepath)
        
        if result:
            # Обновляем 3D координаты групп точек
            calibration_bridge.update_point_groups_from_calibration(props)
            
            # Устанавливаем флаг калибровки
            props.is_calibrated = True
            
            self.report({'INFO'}, f"Калибровка загружена из {self.filepath}")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Ошибка при загрузке калибровки")
            return {'CANCELLED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

def estimate_scene_scale(points_3d, camera_points):
    """
    Оценивает масштаб сцены на основе 3D точек и их проекций на камеры.
    
    Args:
        points_3d: Словарь 3D точек {point_id: point_3d}
        camera_points: Словарь точек на камерах {camera_id: [point_id1, point_id2, ...]}
    
    Returns:
        float: Оценка масштаба сцены
    """
    if not points_3d or len(points_3d) < 2:
        print("Недостаточно точек для оценки масштаба")
        return 1.0
    
    # Находим стабильные пары точек, которые видны на нескольких камерах
    point_visibility = {}
    for camera_id, points in camera_points.items():
        for point_id in points:
            if point_id not in point_visibility:
                point_visibility[point_id] = set()
            point_visibility[point_id].add(camera_id)
    
    # Точки, видимые на нескольких камерах (минимум 3)
    stable_points = [point_id for point_id, cameras in point_visibility.items() 
                    if len(cameras) >= 3 and point_id in points_3d]
    
    if len(stable_points) < 2:
        print("Недостаточно стабильных точек для оценки масштаба")
        return 1.0
    
    # Вычисляем расстояния между стабильными точками
    distances = []
    stable_pairs = []
    
    for i in range(len(stable_points)):
        for j in range(i+1, len(stable_points)):
            point_id1 = stable_points[i]
            point_id2 = stable_points[j]
            
            # Проверяем, что точки видны на общих камерах
            common_cameras = point_visibility[point_id1].intersection(point_visibility[point_id2])
            if len(common_cameras) >= 2:
                p1 = np.array(points_3d[point_id1])
                p2 = np.array(points_3d[point_id2])
                distance = np.linalg.norm(p2 - p1)
                
                # Проверяем, что расстояние не слишком маленькое
                if distance > 0.01:
                    distances.append(distance)
                    stable_pairs.append((point_id1, point_id2))
                    print(f"Стабильная пара точек {point_id1}-{point_id2}: расстояние {distance:.2f}")
    
    if distances:
        # Используем медианное расстояние как базовое
        median_dist = np.median(distances)
        print(f"\nМедианное расстояние между стабильными точками: {median_dist:.2f}")
        
        # Для автомобиля Mercedes SL65 AMG (на фото) типичная длина около 4.5 метров
        # Ширина около 1.8 метра, высота около 1.3 метра
        # Расстояние между колесами (колесная база) около 2.5 метров
        # Диаметр колеса около 0.7 метра
        
        # Анализируем размеры реконструкции
        points_array = np.array(list(points_3d.values()))
        min_coords = np.min(points_array, axis=0)
        max_coords = np.max(points_array, axis=0)
        dimensions = max_coords - min_coords
        diagonal = np.linalg.norm(dimensions)
        
        print(f"Текущие размеры реконструкции:")
        print(f"  X: {dimensions[0]:.2f}")
        print(f"  Y: {dimensions[1]:.2f}")
        print(f"  Z: {dimensions[2]:.2f}")
        print(f"  Диагональ: {diagonal:.2f}")
        
        # Оцениваем масштаб на основе диагонали
        # Типичная диагональ автомобиля ~5.5 метров
        target_diagonal = 5.5
        scale_from_diagonal = target_diagonal / diagonal
        
        # Оцениваем масштаб на основе медианного расстояния
        # Предполагаем, что медианное расстояние между точками примерно 1/4 длины автомобиля
        estimated_car_length = median_dist * 4
        scale_from_median = 4.5 / estimated_car_length
        
        print(f"Оценка длины автомобиля: {estimated_car_length:.2f} м")
        print(f"Масштаб на основе медианного расстояния: {scale_from_median:.4f}")
        print(f"Масштаб на основе диагонали: {scale_from_diagonal:.4f}")
        
        # Выбираем наиболее разумный масштаб
        if 0.5 < scale_from_median < 20 and 0.5 < scale_from_diagonal < 20:
            # Если оба масштаба в разумных пределах, берем среднее
            scale = (scale_from_median + scale_from_diagonal) / 2
        elif 0.5 < scale_from_median < 20:
            # Если только масштаб на основе медианы разумный
            scale = scale_from_median
        elif 0.5 < scale_from_diagonal < 20:
            # Если только масштаб на основе диагонали разумный
            scale = scale_from_diagonal
        else:
            # Если оба масштаба за пределами разумного, используем более консервативный
            scale = min(max(scale_from_median, 0.5), 20)
        
        print(f"Итоговый масштаб: {scale:.4f}")
        
        # Проверяем, что масштаб находится в разумных пределах
        if scale < 0.5 or scale > 20:
            print(f"Вычисленный масштаб {scale:.4f} выходит за разумные пределы, корректируем")
            scale = min(max(scale, 0.5), 20)
            print(f"Скорректированный масштаб: {scale:.4f}")
        
        return scale
    
    return 1.0  # Возвращаем масштаб по умолчанию, если не удалось оценить

class CAMCALIB_OT_apply_scale(Operator):
    """Применить масштаб к реконструкции"""
    bl_idname = "camera_calibration.apply_scale"
    bl_label = "Применить масштаб"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.camera_calibration
        
        if not calibration_bridge.DEPENDENCIES_INSTALLED:
            self.report({'ERROR'}, "Отсутствуют зависимости для калибровки")
            return {'CANCELLED'}
        
        # Проверяем, что калибровка выполнена
        if not props.is_calibrated:
            self.report({'ERROR'}, "Сначала выполните калибровку")
            return {'CANCELLED'}
        
        # Применяем масштаб
        calibration.apply_scale(props.scale_factor)
        
        # Обновляем камеры в сцене, если они есть
        calibration_bridge.create_cameras_from_calibration(props)
        
        self.report({'INFO'}, f"Масштаб {props.scale_factor:.4f} успешно применен")
        return {'FINISHED'}

class CAMCALIB_OT_estimate_scale_from_points(Operator):
    """Оценить масштаб на основе расстояния между точками"""
    bl_idname = "camera_calibration.estimate_scale_from_points"
    bl_label = "Оценить масштаб"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.camera_calibration
        
        if not calibration_bridge.DEPENDENCIES_INSTALLED:
            self.report({'ERROR'}, "Отсутствуют зависимости для калибровки")
            return {'CANCELLED'}
        
        # Проверяем, что калибровка выполнена
        if not props.is_calibrated:
            self.report({'ERROR'}, "Сначала выполните калибровку")
            return {'CANCELLED'}
        
        # Проверяем, что точки существуют
        if props.point_id1 not in calibration.calibration_data['points_3d'] or \
           props.point_id2 not in calibration.calibration_data['points_3d']:
            self.report({'ERROR'}, f"Точки с ID {props.point_id1} и {props.point_id2} не найдены")
            return {'CANCELLED'}
        
        # Получаем 3D точки
        p1 = calibration.calibration_data['points_3d'][props.point_id1]
        p2 = calibration.calibration_data['points_3d'][props.point_id2]
        
        # Вычисляем расстояние в пространстве реконструкции
        reconstruction_distance = np.linalg.norm(p2 - p1)
        
        # Вычисляем коэффициент масштаба
        if reconstruction_distance > 0:
            scale = props.real_distance / reconstruction_distance
            props.scale_factor = scale
            self.report({'INFO'}, f"Оценен масштаб: {scale:.4f}")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Расстояние между точками слишком мало")
            return {'CANCELLED'}

class CAMCALIB_OT_calibrate(Operator):
    """Запустить калибровку камеры с базовыми настройками"""
    bl_idname = "camera_calibration.calibrate"
    bl_label = "Калибровать"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.camera_calibration
        wm = context.window_manager

        def update_progress(progress_value, status_text):
            progress_value = float(max(0.0, min(100.0, progress_value)))
            if progress_value + 1e-6 < float(props.calibration_progress):
                return
            props.calibration_progress = progress_value
            props.calibration_status = str(status_text or "")
            try:
                wm.progress_update(progress_value)
            except Exception:
                pass

            try:
                for window in wm.windows:
                    screen = getattr(window, "screen", None)
                    if screen is None:
                        continue
                    for area in screen.areas:
                        area.tag_redraw()
            except Exception:
                pass
        
        if not calibration_bridge.DEPENDENCIES_INSTALLED:
            self.report({'ERROR'}, "Отсутствуют зависимости для калибровки")
            return {'CANCELLED'}
        
        images_with_points = _get_images_with_grouped_points(props)

        # Проверяем, что есть достаточно изображений
        if len(images_with_points) < 2:
            self.report({'ERROR'}, "Для калибровки необходимо минимум 2 изображения с точками")
            return {'CANCELLED'}
        
        # Проверяем наличие групп точек
        if len(props.point_groups) == 0:
            self.report({'ERROR'}, "Нет групп точек для калибровки")
            return {'CANCELLED'}
        
        # Пустые изображения не участвуют в калибровке.
        # Минимум точек проверяем только для реально используемых кадров.
        min_points_per_image = 6
        for _, image_item, points_count in images_with_points:
            if points_count < min_points_per_image:
                self.report({'ERROR'}, f"На изображении {image_item.name} недостаточно точек (минимум {min_points_per_image})")
                return {'CANCELLED'}
        
        # Если включен режим двухсторонней калибровки и создана плоскость симметрии,
        # добавляем симметричные точки перед выполнением калибровки
        use_mirror = props.use_mirror_calibration and props.symmetry_plane_created
        if use_mirror:
            try:
                from calibration import calibration_data
                from calibration_modules import mirror_calibration
                
                # Создаем временные данные для калибровки
                temp_calib_data = {}
                
                # Добавляем точки из групп в данные калибровки
                temp_calib_data['points_3d'] = {}
                for i, group in enumerate(props.point_groups):
                    temp_calib_data['points_3d'][i] = group.location_3d
                
                # Добавляем информацию о плоскости симметрии
                plane_params = calibration_data['mirror_data']['plane_params']
                temp_calib_data['mirror_data'] = {
                    'plane_params': plane_params
                }
                
                # Создаем зеркальные точки
                mirror_points = mirror_calibration.generate_mirrored_points(temp_calib_data)
                
                # Добавляем зеркальные точки как дополнительные группы
                if mirror_points:
                    # Перед добавлением сохраняем текущий индекс активной группы
                    active_index = props.active_point_group_index
                    
                    # Добавляем новые группы для зеркальных точек
                    for point_id, point_3d in mirror_points.items():
                        # Создаем новую группу
                        bpy.ops.camera_calibration.add_point_group()
                        new_group_idx = len(props.point_groups) - 1
                        
                        if new_group_idx >= 0:
                            # Получаем группу
                            group = props.point_groups[new_group_idx]
                            
                            # Задаем имя группы
                            original_id = int(point_id.split('_')[1])
                            group.name = f"Зеркало {props.point_groups[original_id].name}"
                            
                            # Устанавливаем 3D позицию
                            group.location_3d = point_3d
                            
                            # Устанавливаем цвет группы (используем зеркальный цвет)
                            group.color = props.mirror_point_color
                    
                    # Восстанавливаем активную группу
                    props.active_point_group_index = active_index
            
            except Exception as e:
                self.report({'WARNING'}, f"Ошибка при создании зеркальных точек: {str(e)}")
                import traceback
                traceback.print_exc()
        
        props.calibration_in_progress = True
        props.calibration_progress = 0.0
        props.calibration_status = "Подготовка к запуску..."
        wm.progress_begin(0, 100)

        try:
            # Запускаем быструю калибровку с оптимальными параметрами
            self.report({'INFO'}, "Запуск калибровки...")
            update_progress(2.0, "Запуск калибровки...")
            result = calibration_bridge.run_calibration_from_addon(
                props,
                min_points_for_camera=6,
                bundle_method='trf',
                bundle_ftol=1e-8,
                max_bundle_iterations=5,
                ransac_threshold=8.0,
                confidence=0.99,
                max_attempts=3,
                max_reprojection_error=10.0,
                create_cameras=True,
                progress_callback=update_progress
            )
            
            if not result:
                props.calibration_progress = 0.0
                props.calibration_status = "Ошибка при выполнении калибровки"
                self.report({'ERROR'}, "Ошибка при выполнении калибровки")
                return {'CANCELLED'}
            
            # Устанавливаем флаг калибровки
            props.is_calibrated = True
            props.calibration_progress = 100.0
            props.calibration_status = "Калибровка завершена"
            
            self.report({'INFO'}, "Калибровка успешно завершена")
            return {'FINISHED'}
        finally:
            props.calibration_in_progress = False
            try:
                wm.progress_end()
            except Exception:
                pass

class CAMCALIB_OT_advanced_calibrate(Operator):
    """Открыть диалог расширенной калибровки"""
    bl_idname = "camera_calibration.advanced_calibrate"
    bl_label = "Расширенная калибровка"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Открываем диалог расширенной калибровки
        bpy.ops.camera_calibration.run_advanced_calibration('INVOKE_DEFAULT')
        return {'FINISHED'}

class CAMCALIB_OT_calculate_fov(bpy.types.Operator):
    """Рассчитывает поле зрения (FOV) на основе калибровки"""
    bl_idname = "camera_calibration.calculate_fov"
    bl_label = "Рассчитать поле зрения"
    bl_description = "Рассчитывает горизонтальное и вертикальное поле зрения на основе калибровки"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        props = context.scene.camera_calibration
        return props.is_calibrated
    
    def execute(self, context):
        props = context.scene.camera_calibration
        
        # Импортируем функцию расчета FOV
        from .calibration_bridge import calculate_fov_from_calibration
        
        # Рассчитываем общее FOV
        horizontal_fov, vertical_fov = calculate_fov_from_calibration(props)
        
        if horizontal_fov is not None and vertical_fov is not None:
            self.report({'INFO'}, f"Поле зрения: горизонтальное={horizontal_fov:.2f}°, вертикальное={vertical_fov:.2f}°")
            
            # Сохраняем FOV в свойствах аддона
            props.horizontal_fov = horizontal_fov
            props.vertical_fov = vertical_fov
            
            # Рассчитываем FOV для каждой камеры
            for camera_id in range(len(props.images)):
                calculate_fov_from_calibration(props, camera_id)
            
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Не удалось рассчитать поле зрения")
            return {'CANCELLED'}

class CAMCALIB_OT_align_scene_to_camera(Operator):
    """Выровнять сцену относительно выбранной камеры"""
    bl_idname = "camera_calibration.align_scene_to_camera"
    bl_label = "Выровнять по камере"
    bl_options = {'REGISTER', 'UNDO'}
    
    align_axis: StringProperty(
        name="Ось выравнивания",
        description="Ось, по которой будет выровнена сцена",
        default="Y"
    )
    
    def execute(self, context):
        props = context.scene.camera_calibration
        
        # Проверяем, выбрана ли камера
        if not context.active_object or context.active_object.type != 'CAMERA':
            self.report({'ERROR'}, "Выберите камеру для выравнивания")
            return {'CANCELLED'}
        
        camera_obj = context.active_object
        
        # Создаем полную матрицу трансформации камеры
        camera_matrix = camera_obj.matrix_world.copy()
        
        # Создаем матрицу преобразования, которая переместит камеру в начало координат
        # и выровняет ее по выбранной оси
        align_matrix = Matrix.Identity(4)
        
        # Добавляем дополнительное вращение в зависимости от выбранной оси
        if self.align_axis == "Y":
            # Поворот, чтобы смотреть вдоль +Y (90 градусов вокруг X)
            align_matrix = Matrix.Rotation(math.radians(90), 4, 'X')
        elif self.align_axis == "Z":
            # Поворот, чтобы смотреть вдоль +Z (180 градусов вокруг Y)
            align_matrix = Matrix.Rotation(math.radians(180), 4, 'Y')
        elif self.align_axis == "X":
            # Поворот, чтобы смотреть вдоль +X (-90 градусов вокруг Y)
            align_matrix = Matrix.Rotation(math.radians(-90), 4, 'Y')
        else:
            self.report({'ERROR'}, f"Неизвестная ось выравнивания: {self.align_axis}")
            return {'CANCELLED'}
        
        # Вычисляем полную матрицу трансформации: сначала инвертируем матрицу камеры,
        # чтобы переместить ее в начало координат и сбросить ориентацию,
        # затем применяем дополнительное вращение для выравнивания по нужной оси
        transform_matrix = align_matrix @ camera_matrix.inverted()
        
        # Применяем эту трансформацию ко всем объектам сцены
        for obj in context.scene.objects:
            obj.matrix_world = transform_matrix @ obj.matrix_world
        
        # Также применяем трансформацию к камерам из калибровки, если они существуют
        if hasattr(props, 'cameras'):
            for cam in props.cameras:
                camera_name = f"Camera_{cam.camera_id}"
                if camera_name in bpy.data.objects and bpy.data.objects[camera_name] != camera_obj:
                    obj = bpy.data.objects[camera_name]
                    # Убеждаемся, что мы еще не обработали эту камеру
                    if obj not in context.scene.objects:
                        obj.matrix_world = transform_matrix @ obj.matrix_world
        
        self.report({'INFO'}, f"Сцена выровнена относительно камеры {camera_obj.name} по оси {self.align_axis}")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

class CAMCALIB_OT_select_point_group(Operator):
    """Активирует точку выбранной группы на текущем изображении"""
    bl_idname = "camera_calibration.select_point_group"
    bl_label = "Выбрать группу точек"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.camera_calibration
        
        # Проверяем, есть ли активная группа
        if props.active_point_group_index < 0 or props.active_point_group_index >= len(props.point_groups):
            return {'CANCELLED'}
        
        # Проверяем, есть ли активное изображение
        if props.active_image_index < 0 or props.active_image_index >= len(props.images):
            return {'CANCELLED'}
        
        # Получаем активную группу и изображение
        active_group_id = props.active_point_group_index
        image_item = props.images[props.active_image_index]
        
        # Ищем первую точку выбранной группы на текущем изображении
        for i, point in enumerate(image_item.points):
            if point.is_placed and point.point_group_id == active_group_id:
                # Нашли точку этой группы, делаем её активной
                image_item.active_point_index = i
                self.report({'INFO'}, f"Активирована точка {i+1} из группы {props.point_groups[active_group_id].name}")
                break
        
        return {'FINISHED'}

class CAMCALIB_OT_fix_images_fake_user(Operator):
    """Установить флаг Fake User для всех изображений, чтобы они не терялись при сохранении"""
    bl_idname = "camera_calibration.fix_images_fake_user"
    bl_label = "Закрепить изображения в проекте"
    bl_description = "Устанавливает флаг Fake User для всех изображений, чтобы они не удалялись при сохранении"
    
    def execute(self, context):
        # Устанавливаем флаг fake_user для всех изображений
        count = 0
        for img in bpy.data.images:
            if not img.use_fake_user:
                img.use_fake_user = True
                count += 1
        
        if count > 0:
            self.report({'INFO'}, f"Закреплено {count} изображений в проекте")
        else:
            self.report({'INFO'}, "Все изображения уже закреплены в проекте")
        
        return {'FINISHED'}

class CAMCALIB_OT_apply_bezier_points(bpy.types.Operator):
    """Применить точки с кривой Безье к калибровке"""
    bl_idname = "camera_calibration.apply_bezier_points"
    bl_label = "Применить точки Безье"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        if not bezier_module_imported:
            self.report({'ERROR'}, "Модуль кривых Безье не найден")
            return {'CANCELLED'}
        
        # Получаем свойства аддона
        props = context.scene.camera_calibration
        
        # Получаем точки с кривой
        points = get_bezier_points(context)
        
        if not points:
            self.report({'ERROR'}, "Нет точек для добавления")
            return {'CANCELLED'}
        
        # Получаем текущее изображение
        current_image_index = props.active_image_index
        if current_image_index < 0 or current_image_index >= len(props.images):
            self.report({'ERROR'}, "Не выбрано текущее изображение")
            return {'CANCELLED'}
        
        image_item = props.images[current_image_index]
        
        # Добавляем точки в список точек изображения
        added_count = 0
        for point in points:
            new_point = image_item.points.add()
            new_point.location_2d[0] = point.x
            new_point.location_2d[1] = point.y
            # Дополнительные настройки точки
            new_point.is_placed = True
            new_point.point_group_id = -1
            added_count += 1
        
        # Обновляем UI
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        
        # Сбрасываем состояние кривой для создания новой
        reset_bezier_curves()
        
        self.report({'INFO'}, f"Добавлено {added_count} точек для калибровки")
        return {'FINISHED'}

class CAMCALIB_OT_clear_bezier(bpy.types.Operator):
    """Очистить текущую кривую Безье"""
    bl_idname = "camera_calibration.clear_bezier"
    bl_label = "Очистить кривую Безье"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        if not bezier_module_imported:
            self.report({'ERROR'}, "Модуль кривых Безье не найден")
            return {'CANCELLED'}
        
        # Сбрасываем состояние кривой
        reset_bezier_curves()
        
        self.report({'INFO'}, "Кривая Безье очищена")
        return {'FINISHED'}

# Добавляем новые операторы для работы с симметрией
class CAMCALIB_OT_create_symmetry_plane(bpy.types.Operator):
    """Создает плоскость симметрии на основе выбранного метода"""
    bl_idname = "camera_calibration.create_symmetry_plane"
    bl_label = "Создать плоскость симметрии"
    bl_description = "Создать плоскость симметрии для двухсторонней калибровки"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.camera_calibration
        
        # Проверяем, что режим двухсторонней калибровки включен
        if not props.use_mirror_calibration:
            self.report({'ERROR'}, "Режим двухсторонней калибровки не включен")
            return {'CANCELLED'}
        
        try:
            # Импортируем модуль зеркальной калибровки
            from calibration_modules import mirror_calibration
            
            # Создаем временную структуру данных для определения плоскости симметрии
            points_3d = {}
            
            # Получаем точки из групп
            for i, group in enumerate(props.point_groups):
                points_3d[i] = group.location_3d
            
            # Проверяем, достаточно ли точек для определения плоскости
            if len(points_3d) < 3:
                self.report({'ERROR'}, "Недостаточно точек для определения плоскости симметрии (минимум 3)")
                return {'CANCELLED'}
            
            # Если выбран ручной метод, проверяем указанные точки
            manual_points = None
            method = props.symmetry_plane_method
            
            if method == 'manual':
                # Собираем ID точек для ручного определения плоскости
                point_ids = [props.symmetry_point1, props.symmetry_point2, props.symmetry_point3]
                
                # Проверяем, что все ID в пределах допустимых значений
                for point_id in point_ids:
                    if point_id >= len(props.point_groups):
                        self.report({'ERROR'}, f"Некорректный ID точки: {point_id}. Максимальный ID: {len(props.point_groups)-1}")
                        return {'CANCELLED'}
                
                manual_points = point_ids
            
            # Временная структура данных для калибровки
            calib_data = {
                'points_3d': points_3d
            }
            
            # Определяем плоскость симметрии
            plane_params = mirror_calibration.estimate_symmetry_plane(
                calib_data, method=method, manual_points=manual_points)
            
            # Сохраняем данные о плоскости симметрии в глобальную переменную
            from calibration import calibration_data
            if not calibration_data:
                from calibration import init_calibration
                init_calibration()
                calibration_data = init_calibration()
            
            # Сохраняем параметры плоскости
            if 'mirror_data' not in calibration_data:
                calibration_data['mirror_data'] = {}
            
            # Сохраняем параметры плоскости
            calibration_data['mirror_data']['plane_params'] = plane_params
            calibration_data['mirror_data']['method'] = method
            calibration_data['mirror_data']['manual_points'] = manual_points
            
            # Создаем визуализацию плоскости симметрии
            plane_data = mirror_calibration.visualize_symmetry_plane(
                calibration_data, scale=props.plane_scale)
            
            if plane_data:
                # Сохраняем данные плоскости для визуализации
                calibration_data['mirror_data']['plane_visualization'] = plane_data
                
                # Создаем объект плоскости в Blender
                self._create_plane_object(context, plane_data, props.show_symmetry_plane)
                
                # Устанавливаем флаг создания плоскости
                props.symmetry_plane_created = True
                
                self.report({'INFO'}, "Плоскость симметрии успешно создана")
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, "Не удалось создать данные для визуализации плоскости")
                return {'CANCELLED'}
        
        except Exception as e:
            self.report({'ERROR'}, f"Ошибка при создании плоскости симметрии: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
    
    def _create_plane_object(self, context, plane_data, show=True):
        """Создает объект плоскости в Blender"""
        
        # Удаляем существующий объект плоскости, если он есть
        if "SymmetryPlane" in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects["SymmetryPlane"])
        
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
        context.collection.objects.link(obj)
        
        # Настраиваем отображение плоскости
        obj.display_type = 'WIRE' if show else 'BOUNDS'
        obj.hide_viewport = not show
        
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
            
        return obj

class CAMCALIB_OT_update_symmetry_plane(bpy.types.Operator):
    """Обновляет плоскость симметрии"""
    bl_idname = "camera_calibration.update_symmetry_plane"
    bl_label = "Обновить плоскость симметрии"
    bl_description = "Обновить плоскость симметрии с новыми параметрами"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Вызываем оператор создания плоскости, который заменит существующую
        return bpy.ops.camera_calibration.create_symmetry_plane()

class CAMCALIB_OT_remove_symmetry_plane(bpy.types.Operator):
    """Удаляет плоскость симметрии"""
    bl_idname = "camera_calibration.remove_symmetry_plane"
    bl_label = "Удалить плоскость симметрии"
    bl_description = "Удалить плоскость симметрии и отключить режим двухсторонней калибровки"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.camera_calibration
        
        # Удаляем объект плоскости, если он существует
        if "SymmetryPlane" in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects["SymmetryPlane"])
        
        # Удаляем данные о плоскости симметрии из калибровки
        from calibration import calibration_data
        if calibration_data and 'mirror_data' in calibration_data:
            calibration_data.pop('mirror_data', None)
        
        # Сбрасываем флаг создания плоскости
        props.symmetry_plane_created = False
        
        self.report({'INFO'}, "Плоскость симметрии удалена")
        return {'FINISHED'}

# Обновляем регистрацию и дерегистрацию классов
classes = (
    CAMCALIB_OT_load_image,
    CAMCALIB_OT_remove_image,
    CAMCALIB_OT_remove_all_images,
    CAMCALIB_OT_add_point,
    CAMCALIB_OT_add_point_group,
    CAMCALIB_OT_remove_point_group,
    CAMCALIB_OT_run_advanced_calibration,
    CAMCALIB_OT_save_calibration,
    CAMCALIB_OT_load_calibration,
    CAMCALIB_OT_apply_scale,
    CAMCALIB_OT_estimate_scale_from_points,
    CAMCALIB_OT_calibrate,
    CAMCALIB_OT_advanced_calibrate,
    CAMCALIB_OT_calculate_fov,
    CAMCALIB_OT_align_scene_to_camera,
    CAMCALIB_OT_select_point_group,
    CAMCALIB_OT_remove_point_from_group,
    CAMCALIB_OT_clear_active_group,
    CAMCALIB_OT_fix_images_fake_user,
    CAMCALIB_OT_apply_bezier_points,
    CAMCALIB_OT_clear_bezier,
    CAMCALIB_OT_create_symmetry_plane,
    CAMCALIB_OT_update_symmetry_plane,
    CAMCALIB_OT_remove_symmetry_plane,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls) 
