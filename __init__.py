"""
Аддон для калибровки камер в Blender

Этот аддон предоставляет инструменты для калибровки камер на основе набора соответствующих точек,
определения позиции камеры и реконструкции 3D-сцены.
"""

bl_info = {
    "name": "Image Matching",
    "author": "Development Team",
    "version": (1, 2, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Image Matching",
    "description": "Плагин для точной калибровки камеры на основе фотографий",
    "warning": "",
    "wiki_url": "",
    "category": "3D View"
}

import logging
import sys
import os
import bpy
import traceback
import importlib

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Флаги для отслеживания успешного импорта модулей
properties_imported = False
operators_imported = False
panels_imported = False
image_editor_imported = False
bezier_module_imported = False
calibration_modules_imported = False
utils_imported = False
mirror_calibration_ui_imported = False  # Флаг для модуля зеркальной калибровки
mirror_calibration_imported = False  # Флаг для модуля расчетов зеркальной калибровки

# Базовый путь аддона для импорта модулей
base_path = os.path.dirname(os.path.abspath(__file__))

# Добавляем базовый путь аддона в sys.path, если его там нет
if base_path not in sys.path:
    sys.path.append(base_path)

# Защита кода удалена
# Был закомментирован импорт и использование code_protection
# try:
#     from . import code_protection
#     # Активируем двухуровневую защиту кода
#     code_protection.register()
#
#     # Устанавливаем хуки для защиты пакета IM_v2
#     # Хуки перехватывают импорт и расшифровывают зашифрованные файлы
#     code_protection.install_protection_hooks('IM_v2', base_path, code_protection.get_master_key())
# except ImportError as e:
#     print(f"Ошибка импорта модуля code_protection: {str(e)}")
# except Exception as e:
#     print(f"Ошибка при активации защиты кода: {str(e)}")
#     import traceback
#     traceback.print_exc()

# Импортируем модуль зависимостей
try:
    from . import dependencies
    # Регистрируем менеджер зависимостей
    dependencies.register()
except ImportError as e:
    print(f"Ошибка импорта модуля dependencies: {str(e)}")
    dependencies = None

# Для совместимости с Blender ниже 2.80
if not hasattr(bpy.app, 'timers'):
    import time
    import threading
    
    # Добавляем простую реализацию таймеров для старых версий Blender
    class SimpleTimers:
        def __init__(self):
            self.registered = []
            self.running = False
        
        def register(self, func, first_interval=0):
            if func not in self.registered:
                self.registered.append(func)
                if not self.running:
                    self.running = True
                    threading.Thread(target=self._timer_thread, args=(func, first_interval)).start()
        
        def unregister(self, func):
            if func in self.registered:
                self.registered.remove(func)
        
        def _timer_thread(self, func, delay):
            time.sleep(delay)
            while func in self.registered:
                result = func()
                if result is None:
                    self.registered.remove(func)
                    break
                time.sleep(max(0.1, result))
    
    bpy.app.timers = SimpleTimers()

# Класс настроек аддона
class CAMCALIB_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__
    
    # Свойства для сворачивания/разворачивания блоков
    show_dependencies: bpy.props.BoolProperty(
        name="Показать зависимости",
        description="Показать/скрыть раздел зависимостей",
        default=True
    )
    
    show_info: bpy.props.BoolProperty(
        name="Показать информацию",
        description="Показать/скрыть информацию и диагностику",
        default=True
    )
    
    def draw(self, context):
        layout = self.layout
        
        # Заголовок без блока лицензирования
        box = layout.box()
        box.label(text="Camera Calibration", icon='CAMERA_DATA')
        
        # Блок зависимостей (сворачиваемый)
        box = layout.box()
        row = box.row()
        row.prop(self, "show_dependencies", icon="TRIA_DOWN" if self.show_dependencies else "TRIA_RIGHT", icon_only=True, emboss=False)
        row.label(text="Зависимости")
        
        if self.show_dependencies:
            if dependencies:
                dependencies.draw_dependency_status(box)
        
        # Блок информации и диагностики (сворачиваемый)
        box = layout.box()
        row = box.row()
        row.prop(self, "show_info", icon="TRIA_DOWN" if self.show_info else "TRIA_RIGHT", icon_only=True, emboss=False)
        row.label(text="Информация и диагностика")
        
        if self.show_info:
            # Секция информации о разработчике
            info_box = box.box()
            info_box.label(text="Контактная информация", icon='INFO')
            info_box.label(text="Разработчик: Nultron")
            info_box.operator("wm.url_open", text="Проверить обновления").url = "https://t.me/IM_V2_Bot"
            info_box.label(text="© 2024 Все права защищены")
            
            # Удаляем также оператор диагностики лицензии

# Список модулей для импорта
core_modules = []

# Функция для регистрации свойств сцены
def register_scene_properties():
    """
    Регистрирует свойства сцены для аддона.
    """
    if not properties_imported:
        print("Не удается зарегистрировать свойства сцены: модуль properties не импортирован")
        return
    
    try:
        from . import properties as properties_module
        if hasattr(bpy.types, 'Scene'):
            # Регистрируем свойства для основных операций аддона
            if not hasattr(bpy.types.Scene, 'camera_calibration'):
                bpy.types.Scene.camera_calibration = bpy.props.PointerProperty(type=properties_module.CAMCALIB_props)
                print("Свойства сцены успешно зарегистрированы")
        else:
            print("Не удается получить доступ к bpy.types.Scene")
    except Exception as e:
        print(f"Ошибка при регистрации свойств сцены: {str(e)}")
        import traceback
        traceback.print_exc()

def register():
    # Регистрируем класс настроек аддона
    bpy.utils.register_class(CAMCALIB_AddonPreferences)
    
    # Динамически импортируем utils для предотвращения циклического импорта
    try:
        from . import utils as utils_module
        importlib.reload(utils_module)
        utils_imported = True
    except ImportError as e:
        print(f"Ошибка импорта модуля utils: {str(e)}")
        utils_imported = False
    
    # Перезагружаем модули при обновлении аддона
    for module in core_modules:
        importlib.reload(module)
    
    # Модуль лицензирования удален
    # Все проверки лицензии больше не требуются
    
    # Функция регистрации основных компонентов аддона (вызывается только при валидной лицензии)
    def register_addon_components():
        global properties_imported, operators_imported, panels_imported, image_editor_imported
        global bezier_module_imported, calibration_modules_imported, utils_imported
        global mirror_calibration_ui_imported  # Добавляем глобальную переменную
        
        # Проверяем зависимости перед регистрацией компонентов
        if dependencies is not None:
            dependency_status = dependencies.get_dependency_status(verbose=False)
            deps_installed = all(info['installed'] for info in dependency_status.values())
        else:
            deps_installed = True  # По умолчанию считаем, что зависимости установлены, если модуль не загружен
        
        if deps_installed:
            try:
                # Проверяем наличие обычных файлов модулей
                def check_module_path(module_name):
                    """Проверяет наличие модуля в обычном виде."""
                    import os  # Добавляем явный импорт os в функцию
                    module_path = os.path.join(base_path, f"{module_name}.py")
                    
                    if os.path.exists(module_path):
                        return True
                    else:
                        return False
                
                # Проверяем наличие ключевых модулей
                properties_exists = check_module_path("properties")
                operators_exists = check_module_path("operators")
                panels_exists = check_module_path("panels")
                image_editor_exists = check_module_path("image_editor")
                
                # Используем прямой относительный импорт модулей
                try:
                    from . import properties as properties_module
                    properties_imported = True
                except ImportError as e:
                    print(f"Ошибка импорта модуля properties: {str(e)}")
                    properties_imported = False
                
                try:
                    from . import operators as operators_module
                    operators_imported = True
                except ImportError as e:
                    print(f"Ошибка импорта модуля operators: {str(e)}")
                    operators_imported = False
                
                try:
                    from . import panels as panels_module
                    panels_imported = True
                except ImportError as e:
                    print(f"Ошибка импорта модуля panels: {str(e)}")
                    panels_imported = False
                
                try:
                    from . import image_editor as image_editor_module
                    image_editor_imported = True
                except ImportError as e:
                    print(f"Ошибка импорта модуля image_editor: {str(e)}")
                    image_editor_imported = False
                
                # Импортируем модули калибровки
                try:
                    from . import calibration_bridge as calibration_bridge_module
                    calibration_modules_imported = True
                    # Перезагружаем модуль калибровки
                    importlib.reload(calibration_bridge_module)
                except ImportError as e:
                    print(f"Ошибка импорта модуля calibration_bridge: {str(e)}")
                    calibration_modules_imported = False
                
                # Импортируем модуль безье
                try:
                    from . import bezier_module as bezier_module_module
                    importlib.reload(bezier_module_module)
                    bezier_module_imported = True
                except ImportError as e:
                    print(f"Ошибка импорта модуля bezier_module: {str(e)}")
                    bezier_module_imported = False
                
                # Импортируем модуль зеркальной калибровки
                try:
                    # Проверяем наличие файла mirror_calibration_ui.py
                    import os  # Явно импортируем os в области видимости функции
                    mirror_ui_path = os.path.join(base_path, "mirror_calibration_ui.py")
                    if os.path.exists(mirror_ui_path):
                        from . import mirror_calibration_ui as mirror_calibration_ui_module
                        importlib.reload(mirror_calibration_ui_module)
                        mirror_calibration_ui_imported = True
                    else:
                        print("Файл mirror_calibration_ui.py не найден, пропускаем импорт")
                        mirror_calibration_ui_imported = False
                except ImportError as e:
                    print(f"Ошибка импорта модуля mirror_calibration_ui: {str(e)}")
                    mirror_calibration_ui_imported = False
                
                # Импортируем модуль расчетов зеркальной калибровки
                try:
                    from .calibration_modules import mirror_calibration as mirror_calibration_module
                    mirror_calibration_imported = True
                except ImportError as e:
                    print(f"Ошибка импорта модуля mirror_calibration: {str(e)}")
                    mirror_calibration_imported = False
                
                # Регистрируем модули компонентов аддона
                if properties_imported:
                    properties_module.register()
                
                if operators_imported:
                    operators_module.register()
                
                if panels_imported:
                    panels_module.register()
                
                if image_editor_imported:
                    image_editor_module.register()
                
                if bezier_module_imported:
                    bezier_module_module.register()
                
                # Регистрируем модуль зеркальной калибровки
                if mirror_calibration_ui_imported:
                    mirror_calibration_ui_module.register()
                
                # Регистрируем свойства сцены
                register_scene_properties()
                
                # Регистрируем таймер для обеспечения fake user на изображениях
                if hasattr(bpy.app, 'timers'):
                    bpy.app.timers.register(ensure_fake_user_for_images, first_interval=2.0)
            except Exception as e:
                print(f"Ошибка при загрузке модулей аддона: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("Camera Calibration: Зависимости не установлены. Установите необходимые зависимости.")
    
    # Регистрируем основные компоненты без проверки лицензии
    register_addon_components()

# Функция для установки fake_user для всех изображений
def ensure_fake_user_for_images():
    """Устанавливает флаг Fake User для всех изображений в проекте"""
    try:
        # Проверяем доступность необходимых объектов в текущем контексте
        if not hasattr(bpy, 'data') or not hasattr(bpy.data, 'images'):
            print("Camera Calibration: bpy.data.images недоступно в текущем контексте")
            return None  # Возвращаем None, чтобы таймер не повторялся
            
        # Устанавливаем флаг fake_user для всех изображений
        count = 0
        for img in bpy.data.images:
            if not img.use_fake_user:
                img.use_fake_user = True
                count += 1
                
        if count > 0:
            print(f"Camera Calibration: установлен флаг Fake User для {count} изображений")
        
        return None  # Возвращаем None, чтобы таймер не повторялся
    except Exception as e:
        print(f"Camera Calibration: ошибка при установке Fake User: {str(e)}")
        return None  # Возвращаем None, чтобы таймер не повторялся

def unregister():
    # Удаляем все таймеры аддона
    if hasattr(bpy.app, 'timers'):
        # Попытка остановить таймеры, которые могли быть зарегистрированы
        for timer_func in [ensure_fake_user_for_images]:
            if timer_func in bpy.app.timers.registered:
                try:
                    bpy.app.timers.unregister(timer_func)
                except:
                    pass
    
    # Отменяем регистрацию модулей в обратном порядке
    try:
        # Импорт модулей для отмены регистрации
        try:
            from . import mirror_calibration_ui
            mirror_calibration_ui.unregister()
        except ImportError as e:
            print(f"Ошибка импорта модуля mirror_calibration_ui при выгрузке: {str(e)}")
        except Exception as e:
            print(f"Ошибка выгрузки модуля mirror_calibration_ui: {str(e)}")
            
        if bezier_module_imported:
            try:
                from . import bezier_module
                bezier_module.unregister()
            except:
                pass
        
        if panels_imported:
            try:
                from . import panels
                panels.unregister()
            except:
                pass
        
        if image_editor_imported:
            try:
                from . import image_editor
                image_editor.unregister()
            except:
                pass
        
        if operators_imported:
            try:
                from . import operators
                operators.unregister()
            except:
                pass
        
        if properties_imported:
            try:
                from . import properties
                properties.unregister()
            except:
                pass
    except Exception as e:
        print(f"Ошибка при отмене регистрации компонентов аддона: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Удаляем свойства сцены
    try:
        if hasattr(bpy.types, 'Scene') and hasattr(bpy.types.Scene, 'camera_calibration'):
            del bpy.types.Scene.camera_calibration
    except:
        pass
    
    # Модуль лицензирования удален, больше не требуется отменять его регистрацию
    
    # Отменяем регистрацию настроек аддона
    try:
        bpy.utils.unregister_class(CAMCALIB_AddonPreferences)
    except:
        pass
    
    print("Camera Calibration: аддон выгружен")

if __name__ == "__main__":
    register()
