# object,mesh,scene,wm,render,anim,material,texture,light,armature,curve,text,node,image,view3d,ed
"""bind to blender logic."""
import bpy
try:
    from .lib import DIR_MAPPING, _PKG_, Log, execute, items_mapping, items_motions, get_bones_info, load_data, add_mapping, apply
except ImportError as e:
    print(f'ui ⚠️ {e}')
    from lib import *
BL_ID = 'MOCAP_PT_Panel'
BL_CATAGORY = 'SMPL-X'
BL_SPACE = 'VIEW_3D'
BL_REGION = 'UI'
BL_CONTEXT = 'objectmode'
def Props(context): return context.scene.mocap_importer
def Layout(self) -> 'bpy.types.UILayout': return self.layout


class Mocap_PropsGroup(bpy.types.PropertyGroup):
    input_file: bpy.props.StringProperty(
        name='Input',
        description='gvhmr/wilor ouput .npz file, generated from mocap_wrapper; or video like .mp4',
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
    base_frame: bpy.props.IntProperty(
        name='Frame',
        description='could set -1(last) or 0(first) frame as origin location for offset calculation',
        # subtype='FACTOR',
        default=0,
        soft_min=-1,
        soft_max=0,
        step=1,
    )  # type: ignore
    clean_th: bpy.props.FloatProperty(
        name='Cleanup',
        description='Simplify F-Curves by removing closely spaced keyframes',
        translation_context='Operator',
        subtype='FACTOR',
        default=0,
        max=1,
        soft_min=0,
        soft_max=0.05,
        step=1,
        precision=3,
    )  # type: ignore
    decimate_th: bpy.props.FloatProperty(
        name='Decimate',
        description='How much the new decimated curve is allowed to deviate from the original',
        subtype='FACTOR',
        default=0.01,
        max=1,
        soft_min=0,
        soft_max=0.05,
        step=1,
        precision=3,
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


class MAIN_PT_Panel(ExpandedPanel, bpy.types.Panel):
    bl_label = ''
    _bl_label = 'Motion tracking tools'
    bl_description = _PKG_
    bl_idname = BL_ID
    def draw(self, context): ...
    def draw_header(self, context): Layout(self).label(text=self._bl_label, icon='OUTLINER_OB_ARMATURE')


class IMPORT_PT_Panel(ExpandedPanel, bpy.types.Panel):
    bl_label = 'Import'
    bl_parent_id = BL_ID

    def draw(self, context):
        layout = Layout(self)
        props = Props(context)
        col = layout.column(align=True)
        col.prop(props, 'input_file')
        col.prop(props, 'motions')

        col = layout.column()
        col.operator('armature.load_mocap', icon='ARMATURE_DATA')


class TWEAK_PT_Panel(DefaultPanel, bpy.types.Panel):
    bl_label = 'Tweak'
    bl_parent_id = BL_ID
    bl_translation_context = 'Operator'

    def draw(self, context):
        layout = Layout(self)
        props = Props(context)
        col = layout.column()
        row = layout.row(align=True)
        row.prop(props, 'mapping')
        row.operator('armature.add_mapping', icon='ADD', text='')
        row.operator('wm.open_dir_mapping', icon='FILE_FOLDER', text='')

        _row = col.row(align=True)
        _row.prop(props, 'base_frame')

        row = col.row(align=True)
        _row = row.row(align=True)
        _row.prop(props, 'clean_th')
        _row.active = props.clean_th > 0
        _row = row.row(align=True)
        _row.prop(props, 'decimate_th')
        _row.active = props.decimate_th > 0


class DEBUG_PT_Panel(DefaultPanel, bpy.types.Panel):
    bl_label = 'Development'
    bl_parent_id = BL_ID

    def draw(self, context):
        layout = Layout(self)
        props = Props(context)
        row = layout.row()
        row.operator('armature.get_bones_info', icon='BONE_DATA')
        row = layout.row()
        row.prop(props, 'debug_kwargs')


class ApplyMocap_Operator(bpy.types.Operator):
    bl_idname = 'armature.load_mocap'
    bl_label = 'Apply Mocap'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = 'Apply Mocap data from npz file.'

    @execute
    def execute(self, context):
        props = Props(context)
        mapping = None if props.mapping == 'Auto detect' else props.mapping
        kwargs = eval(f'dict({props.debug_kwargs})')
        apply(props.motions, mapping=mapping, base_frame=props.base_frame, clean_th=props.clean_th, decimate_th=props.decimate_th, **kwargs)
        return {'FINISHED'}


class GetBonesInfo_Operator(bpy.types.Operator):
    bl_idname = 'armature.get_bones_info'
    bl_label = 'print bones'
    bl_description = 'print bones info for making mapping or debugging'

    @execute
    def execute(self, context):
        s = get_bones_info()
        Log.info(s, extra={'mouse': False})
        return {'FINISHED'}


class OpenMapping_Operator(bpy.types.Operator):
    bl_idname = 'wm.open_dir_mapping'
    bl_label = 'Open Folder'
    bl_description = 'Open mapping folder'

    @execute
    def execute(self, context):
        # TODO: when no file manager opened, this may freeze or popop with DEFAULT style in linux
        bpy.ops.wm.path_open(filepath=DIR_MAPPING)
        return {'FINISHED'}


class AddMapping_Operator(bpy.types.Operator):
    bl_idname = 'armature.add_mapping'
    bl_label = 'Add Mapping'
    bl_description = 'Add mapping based on selected armature'
    bl_options = {'REGISTER'}

    @execute
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

    @execute
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
