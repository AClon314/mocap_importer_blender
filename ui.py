# object,mesh,scene,wm,render,anim,material,texture,light,armature,curve,text,node,image,view3d,ed
"""bind to blender logic."""
import bpy
from bpy_extras.io_utils import ImportHelper, ExportHelper
from .logger import _PKG_
from .b import add_mapping, get_bones_info
from .lib import DIR_MAPPING, Log, items_mapping, items_motions, load_data, apply
VIDEO_EXT = "webm,mkv,flv,flv,vob,vob,ogv,ogg,drc,gifv,webm,gifv,mng,avi,mov,qt,wmv,yuv,rm,rmvb,viv,asf,amv,mp4,m4p,m4v,mpg,mp2,mpeg,mpe,mpv,mpg,mpeg,m2v,m4v,svi,3gp,3g2,mxf,roq,nsv,flv,f4v,f4p,f4a,f4b".split(',')
BL_ID = 'MOCAP_PT_Panel'
BL_CATAGORY = 'SMPL-X'
BL_SPACE = 'VIEW_3D'
BL_REGION = 'UI'
BL_CONTEXT = 'objectmode'
def Props(context): return context.scene.mocap_importer
def Layout(self) -> 'bpy.types.UILayout': return self.layout
def Eval(self, context): return eval(self.debug_eval)
# def Eval(self, context): ...


def execute(func):
    """bpy.Operator: `self.report()`"""

    def wrap(self, context):
        setattr(Log, 'report', self.report)
        ret = func(self, context)
        delattr(Log, 'report')
        return ret
    return wrap


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
        row = col.row(align=True)
        row.prop(props, 'input_file', icon='FILE_MOVIE', text='')
        row.operator('mocap.load_file', icon='FILE_FOLDER', text='')
        col.prop(props, 'motions', icon='ACTION', text='')

        col = layout.column()
        col.operator('mocap.apply', icon='ARMATURE_DATA')


class TWEAK_PT_Panel(DefaultPanel, bpy.types.Panel):
    bl_label = 'Tweak'
    bl_parent_id = BL_ID
    bl_translation_context = 'Operator'

    def draw(self, context):
        layout = Layout(self)
        props = Props(context)
        col = layout.column()
        row = layout.row(align=True)
        row.prop(props, 'mapping', icon='OUTLINER_OB_ARMATURE', text='')
        row.operator('mocap.add_mapping', icon='ADD', text='')
        row.operator('wm.open_dir_mapping', icon='FILE_FOLDER', text='')

        _row = col.row(align=True)
        _row.prop(props, 'base_frame')

        row = col.row()
        is_clean = props.clean_th > 0
        is_dec = props.decimate_th > 0
        _row = row.row(align=True)
        _row.prop(props, 'clean_th', icon='HANDLETYPE_AUTO_CLAMP_VEC')
        _row.prop(props, 'keep_end', icon='NEXT_KEYFRAME', text='')
        _row.active = is_clean
        _row = row.row()
        _row.prop(props, 'decimate_th', icon='HANDLETYPE_ALIGNED_VEC')
        _row.active = is_dec


class DEBUG_PT_Panel(DefaultPanel, bpy.types.Panel):
    bl_label = 'Development'
    bl_parent_id = BL_ID

    def draw(self, context):
        layout = Layout(self)
        props = Props(context)
        col = layout.column()
        col.operator('armature.get_bones_info', icon='BONE_DATA')
        col.prop(props, 'debug_kwargs', text='')
        # col.prop(props, 'debug_eval', icon='CONSOLE', text='')


class ApplyMocap_Operator(bpy.types.Operator):
    bl_idname = 'mocap.apply'
    bl_label = 'Apply Mocap'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = 'Apply Mocap data from npz file.'

    @execute
    def execute(self, context):
        props = Props(context)
        mapping = None if props.mapping == 'Auto detect' else props.mapping
        kw = {k: v for k, v in props.items()}
        kwargs = eval(f'dict({props.debug_kwargs})')
        apply(props.motions, mapping=mapping, **kwargs, **kw)
        return {'FINISHED'}


class LoadFile_Operator(bpy.types.Operator, ImportHelper):
    bl_idname = 'mocap.load_file'
    bl_label = 'Load'
    bl_description = 'gvhmr/wilor ouput .npz file, generated from mocap_wrapper; or video like .mp4'
    bl_options = {'REGISTER'}

    filter_glob: bpy.props.StringProperty(
        default='*.npz;*.' + ';*.'.join(VIDEO_EXT),
        options={'HIDDEN'},
    )   # type: ignore

    def invoke(self, context, event):
        props = Props(context)
        self.filepath = props.input_file
        return super().invoke(context, event)

    def execute(self, context):
        props = Props(context)
        props.input_file = self.filepath  # Update props.input_file with the selected file path
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
    bl_idname = 'mocap.add_mapping'
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


class ModalTimerOperator(bpy.types.Operator):
    """https://blender.stackexchange.com/a/305675/146607"""
    bl_idname = "wm.modal_timer_operator"
    bl_label = "Modal Timer Operator"
    _timer = None

    def modal(self, context, event):
        [a.tag_redraw() for a in context.screen.areas]
        if self._timer.time_duration > 3:
            context.window_manager.progress = 1
            return {'FINISHED'}
        context.window_manager.progress = self._timer.time_duration / 3
        return {'PASS_THROUGH'}

    def execute(self, context):
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}


class Mocap_PropsGroup(bpy.types.PropertyGroup):
    input_file: bpy.props.StringProperty(
        name='Input',
        description=LoadFile_Operator.bl_description,
        default='mocap_example.npz',
        # subtype='FILE_PATH',
        update=load_data,
    )  # type: ignore
    motions: bpy.props.EnumProperty(
        name='Action',
        description='load which motion action',
        items=items_motions,
        default=0,
    )  # type: ignore
    keep_end: bpy.props.BoolProperty(
        name='End Frame',
        description='Keep last frame at -1 as start frame when clean up keyframes curves',
        default=True,
    )  # type: ignore
    mapping: bpy.props.EnumProperty(
        name='Mapping',
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
        name='Arguments',
        description='kwargs for debug',
        default="quat=0",
    )   # type: ignore
    # debug_eval: bpy.props.StringProperty(
    #     name='Console Execute',
    #     description='Dangerous',
    #     default="print('😄')",
    #     update=Eval,
    # )   # type: ignore


def register():
    bpy.types.Scene.mocap_importer = bpy.props.PointerProperty(type=Mocap_PropsGroup)   # type: ignore


def unregister():
    del bpy.types.Scene.mocap_importer    # type: ignore
