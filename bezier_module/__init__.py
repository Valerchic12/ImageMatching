"""
Модуль кривых Безье для автоматической генерации точек калибровки.
"""

import bpy
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Глобальные переменные для отслеживания статуса импорта
bezier_curves_imported = False
bezier_ui_imported = False

# Функция для безопасного импорта модулей
def import_bezier_modules():
    global bezier_curves_imported, bezier_ui_imported
    try:
        # Импортируем модули только по требованию
        from . import bezier_curves
        from . import bezier_ui
        bezier_curves_imported = True
        bezier_ui_imported = True
        return True
    except ImportError as e:
        logger.error(f"Ошибка импорта модулей кривых Безье: {e}")
        bezier_curves_imported = False
        bezier_ui_imported = False
        return False

# Функции для взаимодействия с основным аддоном
def get_bezier_points(context=None):
    """
    Получение точек калибровки с кривых Безье
    
    Args:
        context: Контекст Blender (опционально)
    
    Returns:
        list: Список точек калибровки в формате Vector
    """
    # Импортируем модули при необходимости
    if not bezier_ui_imported:
        if not import_bezier_modules():
            logger.error("Модуль bezier_ui не импортирован, невозможно получить точки")
            return []
    
    # Теперь безопасно импортируем bezier_ui
    from . import bezier_ui
    manager = bezier_ui.get_bezier_manager()
    points = []
    
    if manager:
        for curve in manager.curves:
            if curve.is_complete:
                # Получаем точки с кривой
                curve_points = [point.get_position() for point in curve.curve_points]
                points.extend(curve_points)
    
    logger.info(f"Получено {len(points)} точек калибровки с кривых Безье")
    return points


def is_bezier_mode_active(context=None):
    """
    Проверка, активен ли режим кривой Безье
    
    Args:
        context: Контекст Blender
    
    Returns:
        bool: True если активен режим кривой Безье, иначе False
    """
    if not context:
        context = bpy.context
    
    try:
        # Получаем свойства из сцены
        props = context.scene.bezier_props
        return props.placement_mode == 'BEZIER_CURVE'
    except AttributeError:
        # Если свойства не зарегистрированы, режим не может быть активным
        return False


def add_bezier_points_to_images(context, points, image_indices=None):
    """
    Добавление точек с кривой Безье к изображениям
    
    Args:
        context: Контекст Blender
        points: Список точек в формате Vector
        image_indices: Список индексов изображений, к которым нужно добавить точки (опционально)
    
    Returns:
        bool: True если точки успешно добавлены, иначе False
    """
    # Здесь должен быть код для взаимодействия с основным аддоном
    # Это заглушка, которую нужно будет заменить на реальный код
    logger.info(f"Добавление {len(points)} точек к изображениям")
    return True


def reset_bezier_state():
    """
    Сброс состояния кривых Безье
    """
    # Импортируем модуль только при необходимости
    if not bezier_ui_imported:
        if not import_bezier_modules():
            logger.error("Невозможно сбросить состояние: модуль bezier_ui не импортирован")
            return
    
    from . import bezier_ui
    bezier_ui.reset_bezier_state()
    logger.info("Состояние кривых Безье сброшено")


def reset_bezier_curves():
    """
    Полный сброс кривых Безье - удаление всех кривых из менеджера и создание новой
    """
    # Импортируем модуль только при необходимости
    if not bezier_ui_imported:
        if not import_bezier_modules():
            logger.error("Невозможно сбросить кривые: модуль bezier_ui не импортирован")
            return
    
    from . import bezier_ui
    bezier_ui.reset_bezier_curves()
    logger.info("Кривые Безье полностью сброшены")


# Функции регистрации
def register():
    """Регистрация модуля кривых Безье"""
    logger.info("Регистрация модуля кривых Безье...")
    
    # Импортируем модули только при регистрации
    if import_bezier_modules():
        # Добавьте проверку перед регистрацией
        from bpy.utils import register_class
        import bpy.types
        
        # Импортируем модули безопасно
        from . import bezier_curves
        from . import bezier_ui

        # Регистрируем модули, если они импортированы
        if hasattr(bezier_curves, "register"):
            bezier_curves.register()
            logger.info("Модуль bezier_curves зарегистрирован")
        
        if hasattr(bezier_ui, "register"):
            # Проверяем, зарегистрирован ли уже класс BezierCurveProperties
            if not hasattr(bpy.types, "BezierCurveProperties"):
                bezier_ui.register()
                logger.info("Модуль bezier_ui зарегистрирован")
            else:
                logger.info("Модуль bezier_ui уже зарегистрирован")
        
        logger.info("Модуль кривых Безье успешно зарегистрирован")
    else:
        logger.error("Не удалось импортировать модули кривых Безье, регистрация не выполнена")


def unregister():
    """Отмена регистрации модуля кривых Безье"""
    logger.info("Отмена регистрации модуля кривых Безье...")
    
    # Пытаемся импортировать модули для отмены регистрации
    try:
        # Отменяем регистрацию модулей в обратном порядке
        if bezier_ui_imported:
            from . import bezier_ui
            if hasattr(bezier_ui, "unregister"):
                bezier_ui.unregister()
                logger.info("Отменена регистрация модуля bezier_ui")
        
        if bezier_curves_imported:
            from . import bezier_curves
            if hasattr(bezier_curves, "unregister"):
                bezier_curves.unregister()
                logger.info("Отменена регистрация модуля bezier_curves")
        
        logger.info("Отмена регистрации модуля кривых Безье завершена")
    except ImportError as e:
        logger.error(f"Ошибка при отмене регистрации модулей кривых Безье: {e}")


# Если файл запущен напрямую, регистрируем модуль для тестирования
if __name__ == "__main__":
    register() 