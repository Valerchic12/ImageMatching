try:
    import bpy
except:
    print("Использует тест без Блендер")
    pass
from bpy.types import Panel, UIList

# Импортируем модуль кривых Безье
try:
    from .bezier_module import is_bezier_mode_active
    bezier_module_imported = True
except ImportError:
    bezier_module_imported = False
    print("Внимание: модуль bezier_module не найден или не может быть импортирован")


class CAMCALIB_UL_images(UIList):
    """Список загруженных изображений"""

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            placed_points_count = sum(1 for point in item.points if point.is_placed)
            row = layout.row(align=True)
            row.label(text=item.name, icon='IMAGE_DATA')
            if getattr(item, "is_mirror", False):
                row.label(text="", icon='MOD_MIRROR')
            row.separator(factor=0.5)
            row.label(text=str(placed_points_count), icon='LIGHT_POINT')


class CAMCALIB_UL_point_groups(UIList):
    """Список групп точек"""

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            if item.rejection_reason:
                row.label(text="", icon='CANCEL')
            elif any(abs(value) > 1e-8 for value in item.location_3d):
                row.label(text="", icon='CHECKMARK')
            else:
                row.label(text="", icon='LIGHT_POINT')
            row.prop(item, "name", text="", emboss=False)


class CAMCALIB_PT_main_panel(Panel):
    bl_label = "Image Matching"
    bl_idname = "CAMCALIB_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Image Matching'

    @staticmethod
    def _draw_section(layout, props, prop_name, title, icon='NONE', badge_text=None):
        box = layout.box()
        header = box.row(align=True)
        header.prop(
            props,
            prop_name,
            icon='TRIA_DOWN' if getattr(props, prop_name) else 'TRIA_RIGHT',
            icon_only=True,
            emboss=False,
        )
        header.label(text=title, icon=icon)
        if badge_text:
            header.label(text=badge_text)
        return box if getattr(props, prop_name) else None

    @staticmethod
    def _count_grouped_points(props):
        count = 0
        for image_item in props.images:
            for point in image_item.points:
                if point.is_placed and point.point_group_id >= 0:
                    count += 1
        return count

    @staticmethod
    def _count_images_with_grouped_points(props):
        count = 0
        for image_item in props.images:
            if any(point.is_placed and point.point_group_id >= 0 for point in image_item.points):
                count += 1
        return count

    def _draw_images_section(self, layout, props, image_count):
        images_box = self._draw_section(
            layout,
            props,
            "show_images_section",
            "1. Изображения",
            icon='IMAGE_DATA',
            badge_text=f"{image_count} шт" if image_count else None,
        )
        if not images_box:
            return

        if image_count == 0:
            empty_col = images_box.column(align=True)
            empty_col.label(text="Добавьте набор фотографий для разметки", icon='INFO')
            empty_col.operator("camera_calibration.load_image", text="Загрузить изображения", icon='FILE_FOLDER')
            return

        row = images_box.row()
        row.template_list(
            "CAMCALIB_UL_images",
            "",
            props,
            "images",
            props,
            "active_image_index",
            rows=min(max(image_count, 3), 6),
        )

        tools = row.column(align=True)
        tools.operator("camera_calibration.load_image", icon='ADD', text="")
        if props.active_image_index >= 0:
            op = tools.operator("camera_calibration.remove_image", icon='REMOVE', text="")
            op.index = props.active_image_index
        tools.separator()
        tools.operator("camera_calibration.remove_all_images", icon='X', text="")

    def _draw_points_section(self, layout, props, image_count, group_count, grouped_points_count):
        points_box = self._draw_section(
            layout,
            props,
            "show_point_groups_section",
            "2. Разметка",
            icon='LIGHT_POINT',
            badge_text=f"{group_count} групп" if group_count else None,
        )
        if not points_box:
            return

        editor_box = points_box.box()
        editor_box.label(text="Редактор разметки", icon='RESTRICT_VIEW_OFF')
        editor_box.label(text=f"Всего связанных точек: {grouped_points_count}")
        if props.image_editor_active:
            editor_box.label(text="Редактор уже запущен", icon='CHECKMARK')

        editor_row = editor_box.row()
        editor_row.scale_y = 1.3
        editor_row.enabled = image_count > 0
        editor_row.operator("camera_calibration.image_editor", text="Открыть редактор", icon='RESTRICT_VIEW_OFF')
        if image_count == 0:
            editor_box.label(text="Сначала загрузите изображения", icon='INFO')

        if group_count == 0:
            empty_groups = points_box.box()
            empty_groups.label(text="Создайте хотя бы одну группу для разметки", icon='INFO')
            empty_groups.operator("camera_calibration.add_point_group", text="Создать первую группу", icon='ADD')
            return

        row = points_box.row()
        row.template_list(
            "CAMCALIB_UL_point_groups",
            "",
            props,
            "point_groups",
            props,
            "active_point_group_index",
            rows=min(max(group_count, 4), 8),
        )

        tools = row.column(align=True)
        tools.operator("camera_calibration.add_point_group", icon='ADD', text="")
        if props.active_point_group_index >= 0:
            tools.operator("camera_calibration.remove_point_group", icon='REMOVE', text="")

        if props.active_point_group_index < 0 or props.active_point_group_index >= group_count:
            points_box.label(text="Выберите группу перед расстановкой точек", icon='INFO')
            return

        group = props.point_groups[props.active_point_group_index]
        active_group_box = points_box.box()
        active_group_box.label(text="Активная группа", icon='LIGHT_POINT')
        active_group_box.use_property_split = True
        active_group_box.use_property_decorate = False
        active_group_box.prop(group, "name", text="Название")

        status_row = active_group_box.row(align=True)
        if group.rejection_reason:
            status_row.alert = True
            status_row.label(text="Не реконструирована", icon='ERROR')
        elif any(abs(value) > 1e-8 for value in group.location_3d):
            status_row.label(text="Есть 3D-точка", icon='CHECKMARK')
        else:
            status_row.label(text="Пока только 2D-разметка", icon='LIGHT_POINT')

        actions = active_group_box.row(align=True)
        actions.operator("camera_calibration.select_point_group", text="Сделать активной", icon='RESTRICT_SELECT_OFF')
        actions.operator("camera_calibration.clear_active_group", text="Снять привязку", icon='X')

        if group.rejection_reason:
            reject_box = points_box.box()
            reject_box.alert = True
            reject_box.label(text="Причина исключения", icon='ERROR')
            for part in group.rejection_reason.split(" | "):
                part = part.strip()
                if part:
                    reject_box.label(text=part)

    def _draw_mode_section(self, layout, context):
        if not bezier_module_imported:
            return

        mode_box = layout.box()
        mode_box.label(text="Режим разметки", icon='INFO')

        try:
            bezier_props = context.scene.bezier_props
        except AttributeError:
            mode_box.label(text="Модуль кривых Безье недоступен", icon='ERROR')
            return

        row = mode_box.row(align=True)
        row.prop(bezier_props, "placement_mode", expand=True)

        if bezier_props.placement_mode != 'BEZIER_CURVE':
            return

        curve_box = mode_box.box()
        curve_box.label(text="Кривая Безье", icon='INFO')
        curve_box.use_property_split = True
        curve_box.use_property_decorate = False

        col = curve_box.column(align=True)
        col.prop(bezier_props, "num_control_points")
        col.prop(bezier_props, "num_curve_points")
        col.prop(bezier_props, "show_preview")

        if bezier_props.placement_status:
            status_box = curve_box.box()
            status_box.label(text="Статус", icon='INFO')
            status_box.label(text=bezier_props.placement_status)

        row = curve_box.row(align=True)
        row.scale_y = 1.2
        row.operator("camera_calibration.apply_bezier_points", text="Применить точки", icon='CHECKMARK')
        row.operator("camera_calibration.clear_bezier", text="Очистить кривую", icon='X')

    def _draw_calibration_section(self, layout, props, image_count, group_count, grouped_points_count):
        active_image_count = self._count_images_with_grouped_points(props)
        calibration_badge = None
        if props.calibration_in_progress:
            calibration_badge = "в процессе"
        elif props.is_calibrated:
            calibration_badge = "готово"

        calibration_box = self._draw_section(
            layout,
            props,
            "show_calibration_section",
            "3. Калибровка",
            icon='CAMERA_DATA',
            badge_text=calibration_badge,
        )
        if not calibration_box:
            return

        readiness_box = calibration_box.box()
        readiness_box.label(text="Готовность данных", icon='INFO')
        readiness_box.label(text=f"Фото: {image_count}")
        readiness_box.label(text=f"Фото с точками: {active_image_count}")
        readiness_box.label(text=f"Группы: {group_count}")
        readiness_box.label(text=f"Связанные наблюдения: {grouped_points_count}")

        if props.calibration_in_progress:
            progress_box = calibration_box.box()
            progress_box.label(text="Калибровка выполняется", icon='TIME')
            if props.calibration_status:
                progress_box.label(text=props.calibration_status)
            progress_box.prop(props, "calibration_progress", text="Прогресс", slider=True)
        else:
            if active_image_count < 2:
                warning = calibration_box.box()
                warning.alert = True
                warning.label(text="Нужно минимум 2 изображения с точками", icon='ERROR')
            elif group_count == 0:
                warning = calibration_box.box()
                warning.alert = True
                warning.label(text="Нужны группы точек для запуска", icon='ERROR')

        run_row = calibration_box.row()
        run_row.scale_y = 1.4
        run_row.enabled = not props.calibration_in_progress
        run_row.operator(
            "camera_calibration.calibrate",
            text="Перекалибровать" if props.is_calibrated else "Запустить калибровку",
            icon='PLAY',
        )

        advanced_row = calibration_box.row()
        advanced_row.enabled = not props.calibration_in_progress
        advanced_row.operator(
            "camera_calibration.run_advanced_calibration",
            text="Расширенный режим",
            icon='CAMERA_DATA',
        )

    def _draw_results_section(self, layout, context, props):
        if not props.is_calibrated:
            return

        result_box = layout.box()
        result_box.label(text="Результат", icon='CHECKMARK')
        result_box.label(text=f"Ошибка калибровки: {props.calibration_error:.4f}")
        result_box.label(text=f"FOV: {props.horizontal_fov:.2f}° x {props.vertical_fov:.2f}°")

        fov_row = result_box.row()
        fov_row.operator("camera_calibration.calculate_fov", text="Пересчитать FOV", icon='DRIVER_ROTATIONAL_DIFFERENCE')

        scale_box = result_box.box()
        scale_box.label(text="Масштаб сцены", icon='DRIVER_DISTANCE')
        scale_box.use_property_split = True
        scale_box.use_property_decorate = False
        scale_box.prop(props, "scale_factor", text="Коэффициент")

        scale_actions = scale_box.row(align=True)
        scale_actions.operator("camera_calibration.apply_scale", text="Применить", icon='CHECKMARK')

        estimate_box = scale_box.box()
        estimate_box.label(text="Оценка по двум точкам", icon='DRIVER_DISTANCE')
        estimate_box.use_property_split = True
        estimate_box.use_property_decorate = False
        estimate_box.prop(props, "point_id1", text="Точка 1")
        estimate_box.prop(props, "point_id2", text="Точка 2")
        estimate_box.prop(props, "real_distance", text="Реальное расстояние")
        estimate_box.operator("camera_calibration.estimate_scale_from_points", text="Оценить масштаб", icon='DRIVER_DISTANCE')

        align_box = result_box.box()
        align_box.label(text="Выравнивание сцены", icon='INFO')
        if context.active_object and context.active_object.type == 'CAMERA':
            align_box.label(text=f"Активная камера: {context.active_object.name}")
            row = align_box.row(align=True)
            op_x = row.operator("camera_calibration.align_scene_to_camera", text="Ось X")
            op_x.align_axis = "X"
            op_y = row.operator("camera_calibration.align_scene_to_camera", text="Ось Y")
            op_y.align_axis = "Y"
            op_z = row.operator("camera_calibration.align_scene_to_camera", text="Ось Z")
            op_z.align_axis = "Z"
        else:
            align_box.label(text="Выберите камеру для выравнивания", icon='ERROR')

    def _draw_settings_section(self, layout, props, image_count):
        settings_box = self._draw_section(
            layout,
            props,
            "show_settings_section",
            "4. Отображение и advanced",
            icon='PREFERENCES',
        )
        if not settings_box:
            return

        display_box = settings_box.box()
        display_box.label(text="Отображение в редакторе", icon='INFO')
        display_toggles = display_box.row(align=True)
        display_toggles.prop(props, "show_points", text="Точки", toggle=True)
        display_toggles.prop(props, "show_point_ids", text="ID", toggle=True)
        display_toggles.prop(props, "show_point_groups", text="Группы", toggle=True)

        display_form = display_box.column(align=True)
        display_form.use_property_split = True
        display_form.use_property_decorate = False
        display_form.prop(props, "show_magnifier", text="Лупа")
        display_form.prop(props, "point_size", text="Размер точек")
        display_form.prop(props, "point_brightness", text="Яркость точек")

        camera_box = settings_box.box()
        camera_box.label(text="Камера и bootstrap", icon='CAMERA_DATA')
        camera_box.use_property_split = True
        camera_box.use_property_decorate = False
        camera_box.prop(props, "sensor_preset", text="Сенсор")
        camera_box.prop(props, "sensor_width", text="Ширина сенсора")
        camera_box.prop(props, "focal_length", text="Фокусное (мм)")

        if not props.is_calibrated:
            camera_box.label(text="Фокусное будет доуточнено во время калибровки", icon='INFO')
        else:
            camera_box.label(text="Фокусное уже учтено в результате калибровки", icon='CHECKMARK')

        mirror_box = self._draw_section(
            settings_box,
            props,
            "show_mirror_section",
            "Симметрия / 360°",
            icon='MOD_MIRROR',
            badge_text="вкл" if props.use_mirror_calibration else None,
        )
        if not mirror_box:
            return

        mirror_box.prop(props, "use_mirror_calibration", text="Включить двухстороннюю калибровку")
        if not props.use_mirror_calibration:
            return

        mirror_box.use_property_split = True
        mirror_box.use_property_decorate = False
        mirror_box.prop(props, "symmetry_plane_method", text="Метод")

        if props.symmetry_plane_method == 'manual':
            mirror_box.prop(props, "symmetry_point1", text="Точка 1")
            mirror_box.prop(props, "symmetry_point2", text="Точка 2")
            mirror_box.prop(props, "symmetry_point3", text="Точка 3")

        preview_row = mirror_box.row(align=True)
        preview_row.prop(props, "mirror_preview_enabled", text="Предпросмотр")
        preview_row.prop(props, "mirror_point_color", text="")

        if props.symmetry_plane_created:
            mirror_box.prop(props, "plane_scale", text="Масштаб плоскости")
            mirror_box.prop(props, "show_symmetry_plane", text="Показывать плоскость")

        mirror_actions = mirror_box.row(align=True)
        mirror_actions.scale_y = 1.2
        if not props.symmetry_plane_created:
            mirror_actions.operator("camera_calibration.create_symmetry_plane", text="Создать плоскость", icon='MOD_MIRROR')
        else:
            mirror_actions.operator("camera_calibration.update_symmetry_plane", text="Обновить", icon='FILE_REFRESH')
            mirror_actions.operator("camera_calibration.remove_symmetry_plane", text="Удалить", icon='X')

        if props.symmetry_plane_created:
            mirror_box.label(text="Зеркальные точки будут добавлены при калибровке", icon='INFO')

        mirrored_images_box = mirror_box.box()
        mirrored_images_box.label(text="Зеркальные изображения", icon='IMAGE_DATA')
        mirrored_images_box.prop(props, "create_mirrored_images", text="Создавать зеркальные копии")

        if props.create_mirrored_images:
            if image_count > 0:
                mirrored_images_box.label(text="Исключите фронтальные и задние кадры:", icon='INFO')
                for img in props.images:
                    if not img.name.startswith("Mirror_"):
                        mirrored_images_box.prop(img, "is_mirror_excluded", text=img.name)
            mirrored_images_box.operator(
                "camera_calibration.create_mirrored_images",
                text="Создать зеркальные изображения",
                icon='MOD_MIRROR',
            )

    def draw(self, context):
        layout = self.layout
        props = context.scene.camera_calibration
        layout.use_property_decorate = False

        image_count = len(props.images)
        group_count = len(props.point_groups)
        grouped_points_count = self._count_grouped_points(props)

        self._draw_images_section(layout, props, image_count)
        self._draw_points_section(layout, props, image_count, group_count, grouped_points_count)
        self._draw_mode_section(layout, context)
        self._draw_calibration_section(layout, props, image_count, group_count, grouped_points_count)
        self._draw_results_section(layout, context, props)
        self._draw_settings_section(layout, props, image_count)


classes = (
    CAMCALIB_UL_images,
    CAMCALIB_UL_point_groups,
    CAMCALIB_PT_main_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
