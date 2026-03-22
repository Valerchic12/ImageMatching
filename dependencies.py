# Импортируем необходимые модули
import os
import sys
import subprocess
import importlib
import traceback
from importlib import util

# Проверяем, запущен ли скрипт в Blender
try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    # Создаем заглушку для bpy, чтобы не было ошибок при запуске скрипта вне Blender
    class FakeBpy:
        class props:
            StringProperty = lambda **kwargs: None
        class types:
            Operator = object
    bpy = FakeBpy()

def get_python_exe():
    """Get path to Python executable."""
    import sys
    return sys.executable

def get_pip_packages_list(force_refresh=False):
    """Get list of installed pip packages."""
    try:
        import subprocess
        python_exe = get_python_exe()
        result = subprocess.run(
            [python_exe, "-m", "pip", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace'
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except Exception as e:
        print(f"Error getting pip packages list: {str(e)}")
        return None

def check_package_installed(package_name, required_version=None, force_check=False):
    """Check if package is installed with optional version check."""
    try:
        # При принудительной проверке сначала очищаем кэш импорта
        if force_check:
            import importlib
            importlib.invalidate_caches()
            
            # Очищаем кэш системы
            import sys
            for key in list(sys.modules.keys()):
                if key.startswith(package_name) or (package_name == 'opencv-python' and key.startswith('cv2')):
                    try:
                        del sys.modules[key]
                    except:
                        pass

        # Метод 1: Использование pkg_resources
        import pkg_resources
        
        if force_check:
            # Очистить кэш pkg_resources (хак)
            try:
                if hasattr(pkg_resources, '_initialize_master_working_set'):
                    pkg_resources._initialize_master_working_set()
            except:
                pass
        
        try:
            pkg_resources.require(package_name)
            if required_version:
                installed_version = pkg_resources.get_distribution(package_name).version
                return installed_version >= required_version
            return True
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            # Если pkg_resources не нашел пакет, проверим другими способами
            pass
            
        # Метод 2: Прямой импорт (для проверки доступности модуля)
        try:
            # Попытка импортировать модуль напрямую
            module_name = package_name.replace('-', '_')  # opencvpython -> opencv_python
            
            # Для определенных пакетов используем специальные имена модулей
            if module_name == 'opencv_python':
                module_name = 'cv2'
            
            # Безопасный импорт с использованием importlib
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Если импорт успешен, но требуется проверка версии
                    if required_version and hasattr(module, '__version__'):
                        return module.__version__ >= required_version
                    return True
            except:
                # Если не сработало, пробуем стандартный импорт
                try:
                    module = __import__(module_name)
                    
                    # Если импорт успешен, но требуется проверка версии
                    if required_version and hasattr(module, '__version__'):
                        return module.__version__ >= required_version
                    return True
                except ImportError:
                    # Если и этот метод не сработал, проверим через pip
                    pass
        except Exception as e:
            print(f"Ошибка при проверке импорта модуля {module_name}: {str(e)}")
        
        # Метод 3: Проверка через pip list (самый медленный, но надежный)
        try:
            import subprocess
            python_exe = get_python_exe()
            result = subprocess.run(
                [python_exe, "-m", "pip", "list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                errors='replace'
            )
            if result.returncode == 0:
                packages_list = result.stdout.lower()
                if package_name.lower() in packages_list:
                    print(f"Пакет {package_name} найден в списке pip, но недоступен через импорт. Перезапустите Blender.")
                    return True
        except Exception as e:
            print(f"Ошибка при проверке пакета через pip list: {str(e)}")
        
        return False
    except Exception as e:
        print(f"Ошибка при проверке установки пакета {package_name}: {str(e)}")
        return False

def get_python_version():
    """Возвращает версию Python как строку и как кортеж."""
    import sys
    version_str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    version_tuple = (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    return version_str, version_tuple

def is_version_compatible(required_version, current_version):
    """Проверяет, совместима ли текущая версия с требуемой."""
    from packaging import version
    try:
        return version.parse(current_version) >= version.parse(required_version)
    except:
        # Если не можем использовать packaging, делаем простое сравнение строк
        return current_version >= required_version

def install_package(package_name, version=None):
    """Устанавливает пакет используя pip напрямую в директорию Blender."""
    try:
        import subprocess
        import site
        import sys
        import os
        
        python_exe = get_python_exe()
        
        # Получаем версию Python
        py_version_str, _ = get_python_version()
        
        # Получаем правильную директорию site-packages Blender
        blender_site_packages = None
        for path in site.getsitepackages():
            if "site-packages" in path:
                # Проверяем, относится ли путь к Blender
                if "Blender" in path or "blender" in path:
                    blender_site_packages = path
                    break
        
        # Если путь не найден, используем предположительный путь
        if not blender_site_packages:
            if 'win32' in sys.platform:
                # Для Windows
                blender_site_packages = os.path.join(os.path.dirname(os.path.dirname(python_exe)), 
                                                  "python", "Lib", "site-packages")
            else:
                # Для Linux/Mac
                blender_site_packages = os.path.join(os.path.dirname(os.path.dirname(python_exe)), 
                                                  "lib", "python" + py_version_str.rsplit(".", 1)[0], 
                                                  "site-packages")
        
        # Формируем спецификацию пакета
        if version:
            package_spec = f"{package_name}=={version}"
        else:
            package_spec = package_name
        
        # Выводим информацию
        print(f"Устанавливаем {package_spec} в директорию Blender...")
        print(f"Python: {python_exe}")
        print(f"Директория: {blender_site_packages}")
        
        # Выполняем прямую установку в директорию Blender
        result = subprocess.run(
            [python_exe, "-m", "pip", "install", f"--target={blender_site_packages}", package_spec],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace'
        )
        
        # Проверяем результат
        if result.returncode == 0:
            print(f"Установка {package_spec} успешна! Перезапустите Blender.")
            return True
        else:
            print(f"Ошибка при установке {package_spec}:")
            print(result.stderr)
            print("Вероятно, нужны права администратора.")
            return False
    except Exception as e:
        print(f"Ошибка: {e}")
        print("Вероятно, нужны права администратора.")
        return False

# Определяем известные пакеты и их версии
KNOWN_PACKAGES = {
    'numpy': {'version': '1.23.5'},
    'scipy': {'version': '1.10.1'},  # Совместимая с Python 3.10 версия
    'opencv-python': {'version': '4.7.0.72'}  # Доступная в репозитории версия
}

# Добавляем переменные для кэширования
_dependency_status_cache = None
_last_check_time = 0
_check_interval = 30  # Интервал обновления в секундах

def get_dependency_status(verbose=False, force_refresh=False):
    """Get status of all dependencies."""
    global _dependency_status_cache, _last_check_time
    
    # Проверяем, нужно ли обновлять кэш
    import time
    current_time = time.time()
    
    # Используем кэш, если он существует и не истек срок его действия, и не требуется принудительное обновление
    if not force_refresh and _dependency_status_cache is not None and (current_time - _last_check_time) < _check_interval:
        if verbose:
            print("Используется кэшированный статус зависимостей")
        return _dependency_status_cache
    
    # Если кэш недействителен или требуется обновление, получаем новый статус
    if verbose:
        print("Обновление статуса зависимостей...")
    
    packages = {}
    for package, info in KNOWN_PACKAGES.items():
        packages[package] = {
            'installed': check_package_installed(package, info['version'], force_check=force_refresh),
            'required_version': info['version']
        }
        
        if verbose:
            print(f"{package} (>= {info['version']}): {'установлен' if packages[package]['installed'] else 'не установлен'}")
    
    # Обновляем кэш и время последней проверки
    _dependency_status_cache = packages
    _last_check_time = current_time
    
    return packages

def draw_dependency_status(layout):
    """Отображает статус зависимостей и кнопки для их установки."""
    # Убираем импорт несуществующего модуля
    # from . import logger
    # log = logger.get_logger("dependencies")
    
    # Используем кэшированный статус по умолчанию, без обновления
    dependency_status = get_dependency_status(force_refresh=False)
    
    # Заголовок секции
    box = layout.box()
    row = box.row()
    row.label(text="Зависимости:")
    
    # Кнопка для обновления статуса
    row.operator("im.refresh_dependencies", text="Обновить статус", icon='FILE_REFRESH')
    
    # Создаем новую строку с предупреждением
    note_row = box.row()
    note_row.label(text="Для установки пакетов запустите Blender от имени администратора!")
    
    # Кнопка установки всех пакетов
    fix_row = box.row()
    fix_row.operator("im.install_all_dependencies", text="Установить все", icon='IMPORT')
    
    # Отображаем статус каждой зависимости
    for package_name, info in dependency_status.items():
        row = box.row()
        if info['installed']:
            row.label(text=f"{package_name}", icon='CHECKMARK')
        else:
            row.label(text=f"{package_name}", icon='X')
            
            # Если пакет не установлен, показываем кнопку для установки
            version = info['required_version']
            op = row.operator("im.install_dependency", text=f"Установить {package_name} {version}")
            op.package_name = package_name

# Вспомогательная функция для отображения сообщений пользователю
def show_message_to_user(message, title="Сообщение", icon='INFO'):
    """Отображает всплывающее сообщение пользователю"""
    def draw(self, context):
        lines = message.split('\n')
        for line in lines:
            self.layout.label(text=line)
    
    if IN_BLENDER:
        bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)

# Класс для установки всех зависимостей
class IMAGEMODELLER_OT_InstallAllDependencies(bpy.types.Operator):
    bl_idname = "im.install_all_dependencies"
    bl_label = "Установить все"
    bl_description = "Установить все необходимые Python-пакеты"
    bl_options = {'REGISTER', 'INTERNAL'}
    
    def execute(self, context):
        import site
        import subprocess
        import sys
        import os
        
        # Получаем путь к Python
        python_exe = get_python_exe()
        py_version_str, _ = get_python_version()
        
        # Получаем директорию site-packages Blender
        blender_site_packages = None
        for path in site.getsitepackages():
            if "site-packages" in path:
                # Проверяем, относится ли путь к Blender
                if "Blender" in path or "blender" in path:
                    blender_site_packages = path
                    break
        
        # Если путь не найден, используем предположительный путь
        if not blender_site_packages:
            if 'win32' in sys.platform:
                # Для Windows
                blender_site_packages = os.path.join(os.path.dirname(os.path.dirname(python_exe)), 
                                                  "python", "Lib", "site-packages")
            else:
                # Для Linux/Mac
                blender_site_packages = os.path.join(os.path.dirname(os.path.dirname(python_exe)), 
                                                  "lib", "python" + py_version_str.rsplit(".", 1)[0], 
                                                  "site-packages")
        
        # Формируем список пакетов для установки
        packages = []
        for package_name, package_info in KNOWN_PACKAGES.items():
            packages.append(f"{package_name}=={package_info['version']}")
        
        # Выводим информацию перед установкой
        message = "Устанавливаем пакеты напрямую в директорию Blender:\n"
        message += f"Python: {python_exe}\n"
        message += f"Директория: {blender_site_packages}\n\n"
        message += "Пакеты для установки:\n"
        for package in packages:
            message += f"• {package}\n"
        
        self.report({'INFO'}, "Начинаем установку пакетов...")
        show_message_to_user(message, title="Установка зависимостей", icon='INFO')
        
        # Выполняем установку всех пакетов за один раз
        try:
            cmd = [python_exe, "-m", "pip", "install", f"--target={blender_site_packages}"] + packages
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                errors='replace'
            )
            
            # Очищаем кэши
            import importlib
            importlib.invalidate_caches()
            global _dependency_status_cache
            _dependency_status_cache = None
            
            # Обновляем интерфейс
            if IN_BLENDER:
                for window in bpy.context.window_manager.windows:
                    for area in window.screen.areas:
                        area.tag_redraw()
            
            # Проверяем результат
            if result.returncode == 0:
                success_message = "Все пакеты успешно установлены!\n\n"
                success_message += "ВАЖНО: НЕОБХОДИМО ПЕРЕЗАПУСТИТЬ BLENDER!\n"
                success_message += "После установки новых пакетов требуется полный\n"
                success_message += "перезапуск Blender для их корректного определения."
                
                self.report({'INFO'}, "Установка успешно завершена")
                show_message_to_user(success_message, title="ТРЕБУЕТСЯ ПЕРЕЗАПУСК BLENDER", icon='ERROR')
                return {'FINISHED'}
            else:
                error_message = "Ошибка при установке пакетов:\n\n"
                error_message += result.stderr + "\n\n"
                error_message += "Вероятно, нужны права администратора.\n"
                error_message += "Запустите Blender от имени администратора или\n"
                error_message += "используйте кнопку 'Создать установщик' и запустите его от админа."
                
                self.report({'ERROR'}, "Ошибка установки пакетов")
                show_message_to_user(error_message, title="Ошибка установки", icon='ERROR')
                return {'CANCELLED'}
                
        except Exception as e:
            self.report({'ERROR'}, f"Ошибка: {str(e)}")
            show_message_to_user(
                f"Ошибка при установке пакетов:\n{str(e)}\n\n"
                f"Вероятно, нужны права администратора.",
                title="Ошибка установки", icon='ERROR'
            )
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

# Класс для установки одной зависимости
class IMAGEMODELLER_OT_InstallDependency(bpy.types.Operator):
    bl_idname = "im.install_dependency"
    bl_label = "Установить зависимость"
    bl_description = "Установить выбранный Python-пакет"
    bl_options = {'REGISTER', 'INTERNAL'}
    
    package_name: bpy.props.StringProperty(
        name="Package Name",
        description="Имя пакета для установки",
        default=""
    )
    
    def execute(self, context):
        if not self.package_name:
            self.report({'ERROR'}, "Не указано имя пакета")
            return {'CANCELLED'}
        
        import site
        import subprocess
        import sys
        import os
        
        # Получаем путь к Python
        python_exe = get_python_exe()
        py_version_str, _ = get_python_version()
        
        # Получаем директорию site-packages Blender
        blender_site_packages = None
        for path in site.getsitepackages():
            if "site-packages" in path:
                # Проверяем, относится ли путь к Blender
                if "Blender" in path or "blender" in path:
                    blender_site_packages = path
                    break
        
        # Если путь не найден, используем предположительный путь
        if not blender_site_packages:
            if 'win32' in sys.platform:
                # Для Windows
                blender_site_packages = os.path.join(os.path.dirname(os.path.dirname(python_exe)), 
                                                   "python", "Lib", "site-packages")
            else:
                # Для Linux/Mac
                blender_site_packages = os.path.join(os.path.dirname(os.path.dirname(python_exe)), 
                                                   "lib", "python" + py_version_str.rsplit(".", 1)[0], 
                                                   "site-packages")
        
        # Получаем версию пакета
        status = get_dependency_status()
        if self.package_name not in status:
            self.report({'ERROR'}, f"Неизвестный пакет: {self.package_name}")
            return {'CANCELLED'}
            
        required_version = status[self.package_name]['required_version']
        package_spec = f"{self.package_name}=={required_version}"
        
        # Выводим информацию
        message = f"Устанавливаем пакет напрямую в директорию Blender:\n"
        message += f"• Пакет: {package_spec}\n"
        message += f"• Python: {python_exe}\n"
        message += f"• Директория: {blender_site_packages}\n"
        
        self.report({'INFO'}, f"Установка {package_spec}...")
        show_message_to_user(message, title="Установка пакета", icon='INFO')
        
        # Выполняем установку
        try:
            result = subprocess.run(
                [python_exe, "-m", "pip", "install", f"--target={blender_site_packages}", package_spec],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                errors='replace'
            )
            
            # Очищаем кэши
            import importlib
            importlib.invalidate_caches()
            global _dependency_status_cache
            _dependency_status_cache = None
            
            # Обновляем интерфейс
            if IN_BLENDER:
                for window in bpy.context.window_manager.windows:
                    for area in window.screen.areas:
                        area.tag_redraw()
            
            # Проверяем результат
            if result.returncode == 0:
                success_message = f"Пакет {package_spec} успешно установлен!\n\n"
                success_message += "ВАЖНО: НЕОБХОДИМО ПЕРЕЗАПУСТИТЬ BLENDER!\n"
                success_message += "После установки новых пакетов требуется полный\n"
                success_message += "перезапуск Blender для их корректного определения."
                
                self.report({'INFO'}, "Установка успешно завершена")
                show_message_to_user(success_message, title="ТРЕБУЕТСЯ ПЕРЕЗАПУСК BLENDER", icon='ERROR')
                return {'FINISHED'}
            else:
                error_message = "Ошибка при установке пакета:\n\n"
                error_message += result.stderr + "\n\n"
                error_message += "Вероятно, нужны права администратора.\n"
                error_message += "Запустите Blender от имени администратора или\n"
                error_message += "используйте кнопку 'Создать установщик' и запустите его от админа."
                
                self.report({'ERROR'}, "Ошибка установки пакета")
                show_message_to_user(error_message, title="Ошибка установки", icon='ERROR')
                return {'CANCELLED'}
        
        except Exception as e:
            self.report({'ERROR'}, f"Ошибка: {str(e)}")
            show_message_to_user(
                f"Ошибка при установке пакета:\n{str(e)}\n\n"
                f"Вероятно, нужны права администратора.",
                title="Ошибка установки", icon='ERROR'
            )
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

# Класс для обновления статуса зависимостей
class IMAGEMODELLER_OT_RefreshDependencies(bpy.types.Operator):
    bl_idname = "im.refresh_dependencies"
    bl_label = "Обновить статус зависимостей"
    bl_description = "Обновить информацию о статусе установленных зависимостей"
    bl_options = {'REGISTER', 'INTERNAL'}
    
    def execute(self, context):
        # Очищаем кэш импорта
        import importlib
        importlib.invalidate_caches()
        
        # Очищаем кэш статуса зависимостей
        global _dependency_status_cache
        _dependency_status_cache = None
        
        # Принудительно обновляем статус
        status = get_dependency_status(verbose=True, force_refresh=True)
        
        # Сообщаем о результате
        installed_count = sum(1 for info in status.values() if info['installed'])
        total_count = len(status)
        
        self.report({'INFO'}, f"Обновлен статус зависимостей: {installed_count}/{total_count} установлено")
        
        # Проверка возможных проблем с установкой
        import site
        message = f"Статус зависимостей обновлен: {installed_count}/{total_count} установлено\n\n"
        
        # Добавляем подробную информацию о каждом пакете
        for package, info in status.items():
            installed_status = 'установлен' if info['installed'] else 'не установлен'
            message += f"• {package}: {installed_status}\n"
            
            # Если пакет не установлен, пробуем дополнительную диагностику
            if not info['installed']:
                # Проверяем, установлен ли пакет, но не импортируется
                try:
                    import subprocess
                    python_exe = get_python_exe()
                    result = subprocess.run(
                        [python_exe, "-m", "pip", "show", package],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        encoding='utf-8',
                        errors='replace'
                    )
                    if result.returncode == 0:
                        # Пакет установлен через pip, но не импортируется
                        location = None
                        for line in result.stdout.split('\n'):
                            if line.startswith('Location:'):
                                location = line.split(':', 1)[1].strip()
                                break
                        
                        if location:
                            message += f"   ⚠️ Пакет установлен в: {location}\n"
                            
                            # Проверяем, правильная ли это директория
                            blender_site_packages = site.getsitepackages()[0]
                            if location not in blender_site_packages and blender_site_packages not in location:
                                message += f"   ⚠️ Но должен быть в: {blender_site_packages}\n"
                                message += f"   ⚠️ Необходима переустановка в правильную директорию\n"
                except Exception as e:
                    pass  # Игнорируем ошибки при диагностике
        
        # Добавляем информацию о Python
        message += "\nИнформация о Python в Blender:\n"
        message += f"• Python интерпретатор: {sys.executable}\n"
        message += f"• Путь к site-packages: {site.getsitepackages()[0]}\n"
        
        # Показываем инструкции по ручной установке, если есть проблемы
        if installed_count < total_count:
            message += "\nДля ручной установки используйте команду:\n"
            message += f"{sys.executable} -m pip install --target=\"{site.getsitepackages()[0]}\" numpy scipy opencv-python"
        
        show_message_to_user(message, title="Статус зависимостей", icon='INFO')
        
        # Принудительно обновляем все панели интерфейса
        if IN_BLENDER:
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()
        
        return {'FINISHED'}

# Список классов для регистрации
classes = [
    IMAGEMODELLER_OT_InstallAllDependencies,
    IMAGEMODELLER_OT_InstallDependency,
    IMAGEMODELLER_OT_RefreshDependencies
]

def install_all_dependencies():
    """
    Автоматически устанавливает все необходимые зависимости.
    Эта функция вызывается после активации лицензии для обеспечения 
    полной функциональности аддона без необходимости дополнительных действий.
    """
    print("Начинаем автоматическую установку всех зависимостей...")
    
    # Получаем статус зависимостей
    dep_status = get_dependency_status(verbose=True, force_refresh=True)
    
    # Счетчики для статистики
    installed_count = 0
    failed_count = 0
    already_installed = 0
    
    # Устанавливаем каждый пакет, который не установлен
    for package_name, info in dep_status.items():
        if not info.get('installed', False):
            print(f"Устанавливаем {package_name}...")
            
            # Получаем версию пакета, если известна
            version = None
            if package_name in KNOWN_PACKAGES and 'version' in KNOWN_PACKAGES[package_name]:
                version = KNOWN_PACKAGES[package_name]['version']
            
            # Устанавливаем пакет
            success = install_package(package_name, version)
            
            if success:
                print(f"Пакет {package_name} успешно установлен")
                installed_count += 1
            else:
                print(f"Не удалось установить пакет {package_name}")
                failed_count += 1
        else:
            print(f"Пакет {package_name} уже установлен")
            already_installed += 1
    
    # Выводим общую статистику
    print(f"\nУстановка зависимостей завершена:")
    print(f"- Установлено: {installed_count}")
    print(f"- Не удалось установить: {failed_count}")
    print(f"- Уже было установлено: {already_installed}")
    print(f"- Всего зависимостей: {len(dep_status)}")
    
    # Сбрасываем кэш импортов после установки
    importlib.invalidate_caches()
    
    # Пересчитываем статус после установки
    new_status = get_dependency_status(verbose=True, force_refresh=True)
    missing = [name for name, info in new_status.items() if not info.get('installed', False)]
    
    if missing:
        print(f"ВНИМАНИЕ: После установки все еще отсутствуют зависимости: {', '.join(missing)}")
        print("Может потребоваться перезапуск Blender для корректной работы аддона.")
        return False
    else:
        print("Все зависимости успешно установлены! Аддон готов к использованию.")
        return True

def register():
    """Register Blender addon."""
    if IN_BLENDER:
        for cls in classes:
            bpy.utils.register_class(cls)
        print("Dependency manager registered")

def unregister():
    """Unregister Blender addon."""
    if IN_BLENDER:
        for cls in reversed(classes):
            bpy.utils.unregister_class(cls)
        print("Dependency manager unregistered")

def main():
    """Main function."""
    status = get_dependency_status(verbose=True)
    print("\nDependency Status:")
    for package, info in status.items():
        print(f"{package}: {'✓' if info['installed'] else '✗'}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        print(traceback.format_exc()) 