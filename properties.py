try:
    import bpy
except:
    print("Использует тест без Блендер")
    pass
from bpy.props import (
    StringProperty, IntProperty, FloatVectorProperty, 
    CollectionProperty, PointerProperty, BoolProperty,
    FloatProperty, EnumProperty
)
import colorsys

SENSOR_WIDTH_PRESETS = {
    'MEDIUM_FORMAT': {
        'label': "Medium Format",
        'width_mm': 53.7,
        'description': "53.7 x 40.2 mm",
    },
    'FULL_FRAME': {
        'label': "Full Frame",
        'width_mm': 36.0,
        'description': "36.0 x 23.9 mm",
    },
    'APS_H': {
        'label': "APS-H",
        'width_mm': 27.9,
        'description': "27.9 x 18.6 mm",
    },
    'APS_C': {
        'label': "APS-C",
        'width_mm': 23.6,
        'description': "23.6 x 15.8 mm",
    },
    'FOUR_THIRDS': {
        'label': "4/3",
        'width_mm': 17.3,
        'description': "17.3 x 13.0 mm",
    },
    'ONE_INCH': {
        'label': '1"',
        'width_mm': 13.2,
        'description': "13.2 x 8.8 mm",
    },
    'ONE_OVER_1_63': {
        'label': '1/1.63"',
        'width_mm': 8.8,
        'description': "8.8 x 6.6 mm",
    },
    'ONE_OVER_2_3': {
        'label': '1/2.3"',
        'width_mm': 6.17,
        'description': "6.17 x 4.55 mm",
    },
    'ONE_OVER_3_2': {
        'label': '1/3.2"',
        'width_mm': 4.54,
        'description': "4.54 x 3.42 mm",
    },
}

SENSOR_PRESET_ITEMS = [
    (
        preset_id,
        data['label'],
        f"{data['description']} | width={data['width_mm']:.2f} mm",
    )
    for preset_id, data in SENSOR_WIDTH_PRESETS.items()
]
SENSOR_PRESET_ITEMS.append(
    ('CUSTOM', "Custom", "Своя ширина сенсора в мм")
)

_SENSOR_PRESET_SYNC = False


def _find_sensor_preset_id(sensor_width_mm, tolerance=0.05):
    try:
        sensor_width_mm = float(sensor_width_mm)
    except (TypeError, ValueError):
        return 'CUSTOM'

    for preset_id, data in SENSOR_WIDTH_PRESETS.items():
        if abs(float(data['width_mm']) - sensor_width_mm) <= tolerance:
            return preset_id
    return 'CUSTOM'


def _update_sensor_preset(self, context):
    global _SENSOR_PRESET_SYNC
    if _SENSOR_PRESET_SYNC:
        return

    preset_id = getattr(self, "sensor_preset", "CUSTOM")
    preset = SENSOR_WIDTH_PRESETS.get(preset_id)
    if preset is None:
        return

    target_width = float(preset['width_mm'])
    current_width = float(getattr(self, "sensor_width", target_width))
    if abs(current_width - target_width) <= 1e-6:
        return

    _SENSOR_PRESET_SYNC = True
    try:
        self.sensor_width = target_width
    finally:
        _SENSOR_PRESET_SYNC = False


def _update_sensor_width(self, context):
    global _SENSOR_PRESET_SYNC
    if _SENSOR_PRESET_SYNC:
        return

    matched_preset = _find_sensor_preset_id(getattr(self, "sensor_width", 0.0))
    current_preset = getattr(self, "sensor_preset", "CUSTOM")
    if current_preset == matched_preset:
        return

    _SENSOR_PRESET_SYNC = True
    try:
        self.sensor_preset = matched_preset
    finally:
        _SENSOR_PRESET_SYNC = False

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

class CAMCALIB_point(bpy.types.PropertyGroup):
    """Группа свойств для точки"""
    location_2d: FloatVectorProperty(
        name="2D Location",
        size=2,
        default=(0.0, 0.0)
    )
    is_placed: BoolProperty(
        name="Is Placed",
        default=False
    )
    point_group_id: IntProperty(
        name="Point Group ID",
        default=-1
    )
    is_calibrated: BoolProperty(
        name="Is Calibrated",
        description="Была ли точка успешно откалибрована",
        default=False
    )
    calibration_failed: BoolProperty(
        name="Calibration Failed",
        description="Не удалось откалибровать точку",
        default=False
    )

class CAMCALIB_point_group(bpy.types.PropertyGroup):
    """Группа свойств для группы точек"""
    name: StringProperty(
        name="Name",
        default="Group"
    )
    color: FloatVectorProperty(
        name="Color",
        subtype='COLOR',
        size=3,
        min=0.0,
        max=1.0,
        default=(0.8, 0.8, 0.8)  # Начальный цвет серый
    )
    location_3d: FloatVectorProperty(
        name="3D Location",
        size=3,
        default=(0.0, 0.0, 0.0)
    )
    rejection_reason: StringProperty(
        name="Rejection Reason",
        description="Причина, по которой точка не прошла калибровку",
        default=""
    )

class CAMCALIB_image(bpy.types.PropertyGroup):
    """Группа свойств для изображения"""
    name: StringProperty(
        name="Name",
        default=""
    )
    filepath: StringProperty(
        name="File Path",
        default=""
    )
    points: CollectionProperty(
        type=CAMCALIB_point
    )
    active_point_index: IntProperty(
        name="Active Point Index",
        default=-1
    )
    offset_x: FloatProperty(
        name="Offset X",
        default=0.0
    )
    offset_y: FloatProperty(
        name="Offset Y",
        default=0.0
    )
    scale: FloatProperty(
        name="Scale",
        default=1.0,
        min=0.1,
        max=10.0
    )
    is_mirror_excluded: BoolProperty(
        name="Исключить из зеркалирования",
        description="Исключить это изображение из процесса создания зеркальных версий (для фронтальных или задних видов)",
        default=False
    )
    is_mirror: BoolProperty(
        name="Зеркальное изображение",
        description="Является ли это изображение зеркальной копией другого изображения",
        default=False
    )
    mirror_source_index: IntProperty(
        name="Индекс исходного изображения",
        description="Индекс изображения, зеркальной копией которого является данное изображение",
        default=-1
    )

class CAMCALIB_props(bpy.types.PropertyGroup):
    """Группа свойств для калибровки камеры"""
    # Свойства для управления раскрытием/сворачиванием разделов интерфейса
    show_images_section: BoolProperty(
        name="Показать раздел изображений",
        description="Показать или скрыть раздел изображений",
        default=True
    )
    
    show_point_groups_section: BoolProperty(
        name="Показать раздел групп точек",
        description="Показать или скрыть раздел групп точек",
        default=True
    )
    
    show_calibration_section: BoolProperty(
        name="Показать раздел калибровки",
        description="Показать или скрыть раздел калибровки",
        default=True
    )
    
    show_fov_section: BoolProperty(
        name="Показать раздел FOV",
        description="Показать или скрыть раздел поля зрения",
        default=False
    )
    
    show_scale_section: BoolProperty(
        name="Показать раздел масштаба",
        description="Показать или скрыть раздел масштаба сцены",
        default=False
    )
    
    show_alignment_section: BoolProperty(
        name="Показать раздел выравнивания",
        description="Показать или скрыть раздел выравнивания сцены",
        default=False
    )
    
    show_settings_section: BoolProperty(
        name="Показать раздел настроек",
        description="Показать или скрыть раздел настроек",
        default=False
    )
    
    # Флаг активности редактора изображений
    image_editor_active: BoolProperty(
        name="Активность редактора изображений",
        description="Указывает, что редактор изображений в данный момент активен",
        default=False
    )
    
    # Настройки отображения точек в редакторе изображений
    show_points: BoolProperty(
        name="Показывать точки",
        description="Показывать точки на изображениях",
        default=True
    )
    
    show_point_ids: BoolProperty(
        name="ID точек",
        description="Показывать идентификаторы точек",
        default=True
    )
    
    show_point_groups: BoolProperty(
        name="Группы точек",
        description="Показывать группы точек разными цветами",
        default=True
    )
    
    show_magnifier: BoolProperty(
        name="Лупа",
        description="Показывать увеличительное стекло при наведении на точки",
        default=True
    )
    
    point_size: FloatProperty(
        name="Размер точек",
        description="Размер отображаемых точек на изображении",
        default=10.0,
        min=1.0,
        max=30.0
    )
    
    point_brightness: FloatProperty(
        name="Яркость точек",
        description="Яркость отображаемых точек на изображении",
        default=1.0,
        min=0.1,
        max=2.0
    )
    
    # Далее идут существующие свойства
    images: CollectionProperty(
        type=CAMCALIB_image
    )
    active_image_index: IntProperty(
        name="Active Image Index",
        default=-1
    )
    point_groups: CollectionProperty(
        type=CAMCALIB_point_group
    )
    active_point_group_index: IntProperty(
        name="Active Point Group Index",
        default=-1,
        update=lambda self, context: self.on_active_group_changed(context)
    )
    is_placing_point: BoolProperty(
        name="Is Placing Point",
        default=False
    )
    is_linking_points: BoolProperty(
        name="Is Linking Points",
        default=False
    )
    is_calibrated: BoolProperty(
        name="Is Calibrated",
        default=False
    )
    calibration_in_progress: BoolProperty(
        name="Calibration In Progress",
        description="Идет ли сейчас процесс калибровки",
        default=False
    )
    calibration_progress: FloatProperty(
        name="Calibration Progress",
        description="Текущий прогресс калибровки в процентах",
        default=0.0,
        min=0.0,
        max=100.0,
        subtype='PERCENTAGE'
    )
    calibration_status: StringProperty(
        name="Calibration Status",
        description="Текущий этап калибровки",
        default=""
    )
    calibration_error: FloatProperty(
        name="Calibration Error",
        default=0.0
    )
    camera_matrix: FloatVectorProperty(
        name="Camera Matrix",
        size=9,
        default=(0.0,) * 9
    )
    distortion_coefficients: FloatVectorProperty(
        name="Distortion Coefficients",
        size=4,
        default=(0.0,) * 4
    )
    sensor_preset: EnumProperty(
        name="Тип сенсора",
        description="Быстрый выбор стандартной ширины сенсора для Blender-представления камеры",
        items=SENSOR_PRESET_ITEMS,
        default='FULL_FRAME',
        update=_update_sensor_preset,
    )
    # Размер сенсора в миллиметрах (по умолчанию как у 35мм пленки)
    sensor_width: FloatProperty(
        name="Ширина сенсора",
        description="Используется для перевода fx в Blender lens/mm; не является источником истины для solver",
        default=36.0,
        min=1.0,
        unit='LENGTH',
        update=_update_sensor_width,
    )
    # Фокусное расстояние в миллиметрах
    focal_length: FloatProperty(
        name="Фокусное расстояние",
        description="Слабая UI-подсказка в мм для bootstrap и Blender-представления; solver опирается прежде всего на fx/fy в пикселях",
        default=50.0,
        min=1.0,
        max=2000.0,
        unit='LENGTH'
    )
    # Поле зрения (FOV) в градусах
    horizontal_fov: FloatProperty(
        name="Горизонтальное поле зрения",
        description="Горизонтальное поле зрения камеры в градусах (рассчитывается автоматически)",
        default=0.0,
        min=0.0,
        max=180.0,
        precision=2
    )
    vertical_fov: FloatProperty(
        name="Вертикальное поле зрения",
        description="Вертикальное поле зрения камеры в градусах (рассчитывается автоматически)",
        default=0.0,
        min=0.0,
        max=180.0,
        precision=2
    )
    # Свойства для масштаба
    scale_factor: FloatProperty(
        name="Масштаб",
        description="Коэффициент масштаба для реконструкции (1.0 = без изменений)",
        default=1.0,
        min=0.001,
        max=1000.0,
        precision=4
    )
    
    real_distance: FloatProperty(
        name="Реальное расстояние",
        description="Реальное расстояние между выбранными точками (в метрах)",
        default=1.0,
        min=0.001,
        max=1000.0,
        precision=4
    )
    
    point_id1: IntProperty(
        name="Точка 1",
        description="ID первой точки для измерения расстояния",
        default=0,
        min=0
    )
    
    point_id2: IntProperty(
        name="Точка 2",
        description="ID второй точки для измерения расстояния",
        default=1,
        min=0
    )

    # Добавляем свойства для двухсторонней калибровки
    use_mirror_calibration: BoolProperty(
        name="Двухсторонняя калибровка",
        description="Включить режим двухсторонней 360° калибровки",
        default=False
    )
    
    create_mirrored_images: BoolProperty(
        name="Создавать зеркальные изображения",
        description="Автоматически создавать зеркальные копии изображений для симметричной реконструкции",
        default=True
    )
    
    show_mirror_section: BoolProperty(
        name="Показать раздел симметрии",
        description="Показать или скрыть раздел настройки симметрии",
        default=False
    )
    
    symmetry_plane_method: EnumProperty(
        name="Метод определения плоскости",
        description="Метод, используемый для определения плоскости симметрии",
        items=[
            ('auto', "Автоматический", "Автоматически определить плоскость симметрии на основе распределения точек"),
            ('principal', "Главные компоненты", "Использовать метод главных компонент для определения плоскости"),
            ('manual', "Ручной", "Указать три точки для определения плоскости симметрии")
        ],
        default='auto'
    )
    
    symmetry_point1: IntProperty(
        name="Точка 1",
        description="ID первой точки для определения плоскости симметрии",
        default=0,
        min=0
    )
    
    symmetry_point2: IntProperty(
        name="Точка 2",
        description="ID второй точки для определения плоскости симметрии",
        default=1,
        min=0
    )
    
    symmetry_point3: IntProperty(
        name="Точка 3",
        description="ID третьей точки для определения плоскости симметрии",
        default=2,
        min=0
    )
    
    mirror_preview_enabled: BoolProperty(
        name="Предпросмотр зеркальных точек",
        description="Показывать предпросмотр зеркальных точек в редакторе",
        default=True
    )
    
    mirror_point_color: FloatVectorProperty(
        name="Цвет зеркальных точек",
        subtype='COLOR',
        size=3,
        min=0.0,
        max=1.0,
        default=(0.2, 0.6, 1.0)  # Светло-синий
    )
    
    symmetry_plane_created: BoolProperty(
        name="Плоскость создана",
        description="Флаг, указывающий на то, была ли создана плоскость симметрии",
        default=False
    )
    
    show_symmetry_plane: BoolProperty(
        name="Показать плоскость симметрии",
        description="Показать или скрыть плоскость симметрии в 3D-виде",
        default=True
    )
    
    plane_scale: FloatProperty(
        name="Масштаб плоскости",
        description="Масштаб визуализации плоскости симметрии",
        default=1.0,
        min=0.1,
        max=10.0
    )

    # ===== Параметры калибровки (persistent) =====
    quality_profile: EnumProperty(
        name="Режим калибровки",
        description="Готовый режим калибровки: Быстро/Баланс/Точно/Свои настройки",
        items=[
            ('FAST', "Быстро", "Быстрее работает, но обычно оставляет меньше точек"),
            ('BALANCED', "Баланс", "Рекомендуемый режим для большинства сцен"),
            ('PRECISE', "Точно", "Строже к ошибкам, медленнее, обычно дает более чистый результат"),
            ('CUSTOM', "Свои настройки", "Ручная настройка всех параметров"),
        ],
        default='BALANCED'
    )

    min_points_for_camera: IntProperty(
        name="Мин. точек для камеры",
        description="Минимальное количество точек, необходимое для добавления камеры",
        default=4,
        min=3,
        max=20
    )

    bundle_method: EnumProperty(
        name="Метод оптимизации",
        description="Метод оптимизации для пакетной корректировки",
        items=[
            ('trf', "Trust Region Reflective", "Надежный метод Trust Region Reflective"),
            ('dogbox', "Dogbox", "Метод Dogbox для сложных случаев"),
            ('lm', "Levenberg-Marquardt", "Классический LM алгоритм"),
        ],
        default='trf'
    )

    bundle_ftol: FloatProperty(
        name="Точность оптимизации",
        description="Относительная ошибка в целевой функции для остановки оптимизации",
        default=1e-8,
        min=1e-12,
        max=1e-4,
        precision=12
    )

    max_bundle_iterations: IntProperty(
        name="Макс. итераций оптимизации",
        description="Максимальное количество итераций пакетной корректировки",
        default=3,
        min=1,
        max=10
    )

    ransac_threshold: FloatProperty(
        name="Порог RANSAC",
        description="Пороговое расстояние в пикселях для определения выбросов",
        default=8.0,
        min=1.0,
        max=20.0,
        precision=1
    )

    confidence: FloatProperty(
        name="Уровень доверия RANSAC",
        description="Уровень доверия для RANSAC алгоритма",
        default=0.99,
        min=0.8,
        max=0.999,
        precision=3
    )

    max_attempts: IntProperty(
        name="Макс. попыток добавить камеру",
        description="Максимальное количество попыток добавить камеру в сцену",
        default=3,
        min=1,
        max=10
    )

    max_reprojection_error: FloatProperty(
        name="Макс. ошибка репроекции",
        description="Максимально допустимая ошибка репроекции в пикселях",
        default=10.0,
        min=1.0,
        max=50.0,
        precision=1
    )

    def on_active_group_changed(self, context):
        """Обработчик изменения активной группы точек"""
        # Вызываем оператор выбора точки, если мы не в процессе загрузки/инициализации
        if hasattr(context.scene, 'camera_calibration') and context.scene.camera_calibration == self:
            try:
                bpy.ops.camera_calibration.select_point_group()
            except:
                # Игнорируем ошибки при вызове оператора, чтобы не блокировать интерфейс
                pass

def register():
    bpy.utils.register_class(CAMCALIB_point)
    bpy.utils.register_class(CAMCALIB_point_group)
    bpy.utils.register_class(CAMCALIB_image)
    bpy.utils.register_class(CAMCALIB_props)
    bpy.types.Scene.camera_calibration = PointerProperty(type=CAMCALIB_props)

def unregister():
    # Проверяем, существует ли атрибут, перед его удалением
    if hasattr(bpy.types.Scene, 'camera_calibration'):
        del bpy.types.Scene.camera_calibration
    
    bpy.utils.unregister_class(CAMCALIB_props)
    bpy.utils.unregister_class(CAMCALIB_image)
    bpy.utils.unregister_class(CAMCALIB_point_group)
    bpy.utils.unregister_class(CAMCALIB_point) 
