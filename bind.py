# type: ignore reportInvalidTypeFormat
"""
bind to blender logic.
"""
import bpy
try:
    from .lib import dump_bones, keys_BFS, main, _
except ImportError as e:
    from lib import dump_bones, keys_BFS, main, _


if __name__ == '__main__':
    print(_('Load mocap'))


class Mocap_PropsGroup(bpy.types.PropertyGroup):
    pkl_path: bpy.props.StringProperty(
        name='pkl',
        default='./hmr4d.pkl',
        description='smplx/gvhmr ouput .pkl file, generated from mocap_wrapper. have dict keys: smpl_params_global, smpl_params_local',
        subtype='FILE_PATH',
    )


class MOCAP_PT_Panel(bpy.types.Panel):
    bl_label = _('mocap importer')
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SMPL-X'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.mocap_importer
        obj = context.object
        row = layout.row()

        row.prop(props, 'pkl_path')
        row = layout.row()
        row.operator('object.load_mocap', icon='APPEND_BLEND')
        row = layout.row()
        row.operator('object.get_bones_info', icon='BONE_DATA')


class LoadMocap_Operator(bpy.types.Operator):
    bl_idname = 'object.load_mocap'
    bl_label = _('Load mocap')
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = _('Load mocap data from pkl file.')

    def execute(self, context):
        scene = context.scene
        props = scene.mocap_importer
        pkl_path = props.pkl_path
        main(pkl_path)
        return {'FINISHED'}


class GetBonesInfo_Operator(bpy.types.Operator):
    bl_idname = 'object.get_bones_info'
    bl_label = _('print bones')
    bl_description = _('print bones info for making mapping or debugging')

    def execute(self, context):
        tree = dump_bones(context.active_object)
        print(tree)
        List = keys_BFS(tree)
        print(List)
        return {'FINISHED'}


class ReloadScriptOperator(bpy.types.Operator):
    bl_idname = 'object.reload_script'
    bl_label = 'Reload'
    bl_options = {'REGISTER'}
    bl_description = 'Reload Script. 重载脚本'

    def execute(self, context):
        # 获取当前文本编辑器
        for area in bpy.context.screen.areas:
            if area.type == 'TEXT_EDITOR':
                with bpy.context.temp_override(area=area):
                    bpy.ops.text.resolve_conflict(resolution='RELOAD')
                    bpy.ops.text.run_script()
                break
        else:
            self.report({'ERROR'}, "没有找到文本编辑器")
            return {'CANCELLED'}
        return {'FINISHED'}


def register():
    bpy.types.Scene.mocap_importer = bpy.props.PointerProperty(type=Mocap_PropsGroup)


def unregister():
    del bpy.types.Scene.mocap_importer
