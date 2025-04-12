# object,mesh,scene,wm,render,anim,material,texture,light,armature,curve,text,node,image,view3d,ed
"""bind to blender logic."""
import bpy
import traceback
from typing import Callable
try:
    from .lib import DIR_MAPPING, getLogger, items_mapping, items_motions, get_bones_info, load_data, add_mapping, apply
except ImportError as e:
    print(f'ui ⚠️ {e}')
    from lib import *
Log = getLogger(__name__)
BL_ID = 'MOCAP_PT_Panel'
BL_CATAGORY = 'SMPL-X'
BL_SPACE = 'VIEW_3D'
BL_REGION = 'UI'
BL_CONTEXT = 'objectmode'
def Props(context): return context.scene.mocap_importer


def Execute(func: Callable, self, context):
    """usage
    ```python
    def execute(self, context):
        def wrap():
            ...
        return Execute(self, wrap)
    ```
    """
    try:
        func(self=self, context=context)
        return {'FINISHED'}
    except Exception as e:
        self.report({'ERROR'}, str(e))
        Log.error(traceback.format_exc())
        return {'CANCELLED'}


class Mocap_PropsGroup(bpy.types.PropertyGroup):
    input_video: bpy.props.StringProperty(
        name='Input',
        description='Video',
        default='input.mp4',
        subtype='FILE_PATH',
    )  # type: ignore
    input_npz: bpy.props.StringProperty(
        name='npz',
        description='gvhmr/wilor ouput .npz file, generated from mocap_wrapper',
        default='mocap_example.npz',
        subtype='FILE_PATH',
        update=load_data,
    )  # type: ignore
    motions: bpy.props.EnumProperty(
        name='Action',
        description='load which motion action',
        items=items_motions,
        default=0,
    )  # type: ignore
    mapping: bpy.props.EnumProperty(
        name='Armature',
        description='re-mapping to which bones struct',
        items=items_mapping,
        default=0,
    )  # type: ignore
    # import_start: bpy.props.IntProperty(
    #     name='Start',
    #     description='start frame to import',
    #     default=0,
    #     min=0,
    #     # update=update_pose,
    # )   # type: ignore
    # import_end: bpy.props.IntProperty(
    #     name='End',
    #     default=100,
    #     description='end frame to import',
    # )   # type: ignore
    ibone: bpy.props.IntProperty(
        name='bone index',
        description='bone index to bind, for debug',
        default=22,
        min=0,
        max=24,
        step=1,
    )  # type: ignore
    debug_kwargs: bpy.props.StringProperty(
        name='kwargs',
        description='kwargs for debug',
        default="quat=0",
    )   # type: ignore


class DefaultPanel:
    bl_space_type = BL_SPACE
    bl_region_type = BL_REGION
    bl_category = BL_CATAGORY
    bl_options = {"DEFAULT_CLOSED"}


class ExpandedPanel(DefaultPanel):
    bl_options = {"HEADER_LAYOUT_EXPAND"}


class IMPORT_PT_Panel(ExpandedPanel, bpy.types.Panel):
    bl_label = 'Import'
    bl_idname = BL_ID

    def draw(self, context):
        layout = self.layout
        props = Props(context)
        row = layout.row()

        row.prop(props, 'input_npz')
        row = layout.row()
        split = row.split(factor=0.75, align=True)
        split.prop(props, 'mapping')
        split.operator('armature.add_mapping', icon='ADD')
        split.operator('wm.open_dir_mapping', icon='FILE_FOLDER')

        row = layout.row()
        row.prop(props, 'motions')

        row = layout.row()
        row.operator('armature.load_mocap', icon='ARMATURE_DATA')

        # row = layout.row(align=True)
        # row.prop(props, 'import_start')
        # row.prop(props, 'import_end')

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
        row = layout.row()
        row.prop(props, 'debug_kwargs')


class RUN_PT_Panel(DefaultPanel, bpy.types.Panel):
    bl_label = 'Init'

    def draw(self, context):
        layout = self.layout
        props = Props(context)
        row = layout.row()
        row.prop(props, 'input_video')


class LoadMocap_Operator(bpy.types.Operator):
    bl_idname = 'armature.load_mocap'
    bl_label = 'Load mocap'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = 'Load mocap data from npz file.'

    def execute(self, context):
        scene = context.scene
        props = Props(context)
        input_npz = props.input_npz
        mapping = None if props.mapping == 'Auto detect' else props.mapping
        kwargs = eval(f'dict({props.debug_kwargs})')
        apply(props.motions, mapping=mapping, ibone=props.ibone + 1, **kwargs)
        return {'FINISHED'}


class GetBonesInfo_Operator(bpy.types.Operator):
    bl_idname = 'armature.get_bones_info'
    bl_label = 'print bones'
    bl_description = 'print bones info for making mapping or debugging'

    def execute(self, context):
        s = get_bones_info()
        Log.info(s, extra={'mouse': False})
        return {'FINISHED'}


class OpenMapping_Operator(bpy.types.Operator):
    bl_idname = 'wm.open_dir_mapping'
    bl_label = 'Open Folder'
    bl_description = 'Open mapping folder'

    def execute(self, context):
        # TODO: when no file manager opened, this may freeze or popop with DEFAULT style in linux
        bpy.ops.wm.path_open(filepath=DIR_MAPPING)
        return {'FINISHED'}


class AddMapping_Operator(bpy.types.Operator):
    bl_idname = 'armature.add_mapping'
    bl_label = 'Add Mapping'
    bl_description = 'Add mapping based on selected armature'
    bl_options = {'REGISTER'}

    def execute(self, context):
        try:
            add_mapping()
        except FileExistsError:
            bpy.ops.wm.open_dir_mapping()   # type: ignore
            raise
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
