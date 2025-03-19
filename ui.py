# object,mesh,scene,wm,render,anim,material,texture,light,armature,curve,text,node,image,view3d,ed
"""bind to blender logic."""
import bpy
try:
    from .lib import get_logger, mapping_items, dump_bones, keys_BFS, main
except ImportError as e:
    print(f'ui ⚠️ {e}')
    from lib import *
Log = get_logger(__name__)
MAPPING_ITEMS = mapping_items()
BL_ID = 'MOCAP_PT_Panel'
BL_CATAGORY = 'SMPL-X'
BL_SPACE = 'VIEW_3D'
BL_REGION = 'UI'
BL_CONTEXT = 'objectmode'
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
        min=0,
        description='start frame to import',
    )   # type: ignore
    import_end: bpy.props.IntProperty(
        name='End',
        default=100,
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


class DefaultPanel:
    bl_space_type = BL_SPACE
    bl_region_type = BL_REGION
    bl_category = BL_CATAGORY
    bl_context = BL_CONTEXT
    bl_options = {"DEFAULT_CLOSED"}


class ExpandedPanel(DefaultPanel):
    bl_options = {"HEADER_LAYOUT_EXPAND"}


class MOCAP_PT_Panel(ExpandedPanel, bpy.types.Panel):
    bl_label = 'Import'
    bl_idname = BL_ID

    def draw(self, context):
        layout = self.layout
        props = Props(context)
        row = layout.row()

        row.prop(props, 'pkl_path')
        row = layout.row()
        split = row.split(factor=0.75, align=True)
        split.prop(props, 'mapping')
        split.operator('armature.add_mapping', icon='ADD')
        split.operator('armature.open_mapping', icon='FILE_FOLDER')

        row = layout.row()
        row.operator('armature.load_mocap', icon='ARMATURE_DATA')

        row = layout.row(align=True)
        row.prop(props, 'import_start')
        row.prop(props, 'import_end')

        row = layout.row()


class DEBUG_PT_Panel(DefaultPanel, bpy.types.Panel):
    bl_parent_id = BL_ID
    bl_label = 'Development'

    def draw(self, context):
        layout = self.layout
        props = Props(context)
        row = layout.row()
        row.prop(props, 'ibone')
        row = layout.row()
        row.operator('armature.get_bones_info', icon='BONE_DATA')


class LoadMocap_Operator(bpy.types.Operator):
    bl_idname = 'armature.load_mocap'
    bl_label = 'Load mocap'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = 'Load mocap data from pkl file.'

    def execute(self, context):
        scene = context.scene
        props = Props(context)
        pkl_path = props.pkl_path
        mapping = None if props.mapping == 'Auto detect' else props.mapping
        main(pkl_path, mapping=mapping, ibone=props.ibone + 1)
        return {'FINISHED'}


class GetBonesInfo_Operator(bpy.types.Operator):
    bl_idname = 'armature.get_bones_info'
    bl_label = 'print bones'
    bl_description = 'print bones info for making mapping or debugging'

    def execute(self, context):
        tree = dump_bones(context.active_object)
        List = keys_BFS(tree)
        print('TYPE_BODY = Literal', List)
        print('BONES', tree)
        return {'FINISHED'}


class AddMapping_Operator(bpy.types.Operator):
    bl_idname = 'armature.add_mapping'
    bl_label = 'Add mapping'
    bl_description = 'Add mapping.py based on selected armature'

    def execute(self, context):
        self.report({'INFO'}, "Add Mapping button clicked")
        return {'FINISHED'}


class OpenMapping_Operator(bpy.types.Operator):
    bl_idname = 'armature.open_mapping'
    bl_label = 'Open mapping'
    bl_description = 'Open mapping.py based on selected armature'

    def execute(self, context):
        self.report({'INFO'}, "Open Mapping button clicked")
        return {'FINISHED'}


class ReloadScriptOperator(bpy.types.Operator):
    bl_idname = 'wm.reload_script'
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
