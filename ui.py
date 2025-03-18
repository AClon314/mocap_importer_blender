"""bind to blender logic."""
import bpy
try:
    from .lib import get_logger, mapping_items, dump_bones, keys_BFS, main
except ImportError as e:
    print(f'ui ⚠️ {e}')
    from lib import *
Log = get_logger(__name__)
MAPPING_ITEMS = mapping_items()
def Props(context): return context.scene.mocap_importer


class Mocap_PropsGroup(bpy.types.PropertyGroup):
    pkl_path: bpy.props.StringProperty(
        name='pkl',
        default='./hmr4d.pkl',
        description='gvhmr/wilor ouput .pkl file, generated from mocap_wrapper',
        subtype='FILE_PATH',
    )  # type: ignore
    mapping: bpy.props.EnumProperty(
        name='Armature',
        items=MAPPING_ITEMS,
        default=MAPPING_ITEMS[0][0],
        description='bones struct mapping type',
    )  # type: ignore
    import_start: bpy.props.IntProperty(
        name='Start',
        default=0,
        description='start frame to import',
    )   # type: ignore
    import_end: bpy.props.IntProperty(
        name='End',
        default=250,
        description='end frame to import',
    )   # type: ignore
    ibone: bpy.props.IntProperty(
        name='bone index',
        default=22,
        description='bone index to bind, for debug',
        min=0,
        max=22,
        step=1,
    )  # type: ignore


class MOCAP_PT_Panel(bpy.types.Panel):
    bl_label = 'Import'
    bl_category = 'SMPL-X'
    bl_idname = 'MOCAP_PT_Panel'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):
        layout = self.layout
        # props = Props(context)
        props = context.scene.mocap_importer
        row = layout.row()

        row.prop(props, 'pkl_path')
        row = layout.row()
        row.prop(props, 'mapping')
        row = layout.row()
        row.operator('object.load_mocap', icon='APPEND_BLEND')
        row = layout.row()

        col = layout.column(align=True)
        col.prop(props, 'import_start')
        col.prop(props, 'import_end')


# class DEBUG_PT_Panel(bpy.types.Panel):
#     bl_parent_id = 'MOCAP_PT_Panel'
#     bl_label = 'Development'

#     def draw(self, context):
#         layout = self.layout
#         # props = Props(context)
#         props = context.scene.mocap_importer
#         row = layout.row()
#         row.prop(props, 'ibone')
#         row = layout.row()
#         row.operator('object.get_bones_info', icon='BONE_DATA')


class LoadMocap_Operator(bpy.types.Operator):
    bl_idname = 'object.load_mocap'
    bl_label = 'Load mocap'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = 'Load mocap data from pkl file.'

    def execute(self, context):
        scene = context.scene
        # props = Props(context)
        props = context.scene.mocap_importer
        pkl_path = props.pkl_path
        map_type = None if props.map_type == 'Auto detect' else props.map_type
        main(pkl_path, mapping=map_type, ibone=props.ibone + 1)
        return {'FINISHED'}


class GetBonesInfo_Operator(bpy.types.Operator):
    bl_idname = 'object.get_bones_info'
    bl_label = 'print bones'
    bl_description = 'print bones info for making mapping or debugging'

    def execute(self, context):
        tree = dump_bones(context.active_object)
        List = keys_BFS(tree)
        print('TYPE_BODY = Literal', List)
        print('BONES', tree)
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
    bpy.types.Scene.mocap_importer = bpy.props.PointerProperty(type=Mocap_PropsGroup)   # type: ignore


def unregister():
    del bpy.types.Scene.mocap_importer    # type: ignore
