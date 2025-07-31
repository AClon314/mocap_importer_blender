'''
b.py means a share lib of bpy
'''
from functools import cache
import os
import re
import bpy
import numpy as np
from .lib import DIR_MAPPING, GEN, MAPPING_TEMPLATE, TYPE_MAPPING, TYPE_MAPPING_KEYS, TYPE_RUN, Log, Map, Run, MotionData, Progress, bone_to_dict, euler, get_major, get_mapping, get_similar, keys_BFS, quat, quat_rotAxis
from time import time
from numbers import Number
from contextlib import contextmanager
from typing import Callable, Generator, Literal, Sequence, get_args
try:
    from mathutils import Matrix, Vector, Quaternion, Euler
except ImportError as e:
    Log.warning(e)
TYPE_ROT = Literal['QUATERNION', 'XYZ', 'YZX', 'ZXY', 'XZY', 'YXZ', 'ZYX', 'AXIS_ANGLE']
ROT_KEY = get_args(TYPE_ROT)
TYPE_I18N = Literal[  # blender 4.3.2
    'ca_AD', 'en_US', 'es', 'fr_FR', 'ja_JP', 'sk_SK', 'ur', 'vi_VN', 'zh_HANS',
    'de_DE', 'it_IT', 'ka', 'ko_KR', 'pt_BR', 'pt_PT', 'ru_RU', 'sw', 'ta', 'tr_TR', 'uk_UA', 'zh_HANT',
    'ab', 'ar_EG', 'be', 'bg_BG', 'cs_CZ', 'da', 'el_GR', 'eo', 'eu_EU', 'fa_IR', 'fi_FI', 'ha', 'he_IL', 'hi_IN', 'hr_HR', 'hu_HU', 'id_ID', 'km', 'ky_KG', 'lt', 'ne_NP', 'nl_NL', 'pl_PL', 'ro_RO', 'sl', 'sr_RS', 'sr_RS@latin', 'sv_SE', 'th_TH'
]
_MOTIONDATA = MotionData()
MOTION_DATA = _MOTIONDATA


def apply(*who: str, mapping: TYPE_MAPPING | None = None, **kwargs):
    Log.debug(f'apply {locals()=}')
    whos, mapping = props_filter(who, mapping)
    for w in whos:
        data = MOTION_DATA(who=w)
        for r in data.runs:
            run = getattr(Run()[r], r)
            gen: Generator = run(data, mapping=mapping, **kwargs)
            GEN.append(gen)
    Log.debug(f'apply {len(GEN.queue)=} {GEN=}')


def props_filter(who: Sequence[str], mapping=None):
    '''filter from ui.py, for lib.py'''
    if mapping == 'Auto detect':
        mapping = None
    if len(who) == 1 and who[0] == 'all':
        whos = [m[0] for m in items_motions()]
        whos.remove('all')  # remove 'all' option
        whos = [m for m in whos if 'cam@' not in m]
    else:
        whos = list(who)
    return whos, mapping


@cache
def items_motions(self=None, context=None):
    Log.debug('items_motions')
    all = ['all', 'All', 'len={}. All motion data belows']
    items: list[tuple[str, str, str]] = []
    if not MOTION_DATA:
        load_data()
    tags: dict[str, list] = {}
    k_tag = {}
    ranges = {}
    for k in MOTION_DATA.keys():
        try:
            list_k = k.split(';')
            _range = get_range(list_k)
            tag = ';'.join(list_k[1:3])  # mapping(gvhmr);who(person1);start(0)
            TAG = ';'.join(list_k[4:]) + ' ' + _range
            if tag not in tags.keys():
                # Log.debug(f'{tag=}')
                tags[tag] = [TAG]
                ranges[tag] = _range
                k_tag[f'{list_k[0]};{tag}'] = tag
            else:
                tags[tag].append(TAG)
        except Exception as e:
            tags[tag] = ['❌' + k]
            Log.error(f'{locals()=}', exc_info=e, extra={'log': True})
    Log.debug(f'{tags=}\t\t{k_tag=}\t\t{ranges=}')
    tag_k = {v: k for k, v in k_tag.items()}
    for t, TAG in tags.items():
        items.append((tag_k[t], f'{t};{ranges[t]}', f'tags={len(TAG)}: {TAG}'))
    items.sort(key=lambda x: f'{len(tags[k_tag[x[0]]])}{x[0]}')
    all[-1] = all[-1].format(len(tags))
    items.insert(0, tuple(all))  # type: ignore
    return items


def get_range(keys: list[str]):
    try:
        _len = len(MOTION_DATA(*keys).value)    # type: ignore
    except Exception as e:
        return ''
    try:
        start = int(keys[3])
        _stop = f'={start + _len}' if start != 0 else ''
    except ValueError:
        _stop = ''
    _range = f'{start}+{_len}{_stop}'
    return _range


@cache
def items_mapping(self=None, context=None):
    Log.debug('items_mapping')
    items: list[tuple[str, str, str]] = [(
        'auto', 'Auto', 'Auto detect armature type, based on majority bone names.')]
    for k, m in Map().items():
        help = ''
        try:
            help = m.HELP[bpy.app.translations.locale]
        except Exception:
            Log.warning(f'No help for {k}')
            help = m.__doc__ if m.__doc__ else ''
        items.append((k, k, help))
    return items


def load_data(self=None, context=None):
    """load motion data when npz file path changed"""
    global MOTION_DATA
    file = bpy.context.scene.mocap_importer.input_file   # type: ignore
    if os.path.exists(file) and file != MOTION_DATA.npz:
        del MOTION_DATA
        MOTION_DATA = MotionData(npz=file)
        items_motions.cache_clear()


def get_bone_global_rotation(
    armature: 'bpy.types.Object',
    bone: str,
    frame: int | None = None
) -> Matrix:
    """
    获取某骨骼在某帧的全局旋转矩阵。

    Args:
        armature_name (str): 骨架对象名称。
        bone_name (str): 骨骼名称。
        frame (int): 帧号。

    Returns:
        Matrix: 骨骼的全局旋转矩阵。
    """
    if frame:
        bpy.context.scene.frame_set(frame)  # type: ignore

    # 获取骨骼的全局矩阵
    _bone = armature.pose.bones[bone]
    global_matrix = armature.matrix_world @ _bone.matrix

    # 提取旋转部分
    matrix = global_matrix.to_3x3()
    return matrix


def get_bones_global_rotation(
    armature: 'bpy.types.Object',
    bone_resort: Sequence[str] | None = None,
    Slice: slice | None = None,
) -> np.ndarray:
    """
    导出指定帧范围内骨骼全局绝对旋转关键帧为四元数数组

    Args:
        armature (bpy.types.Object): 骨架对象。
        bone_resort (Sequence[str] | None): 骨骼名称排序列表。
        Slice (slice | None): 帧范围，格式为 slice(start, stop, step)。

    Returns:
        numpy.ndarray: 形状为 (total_frames, total_bones, 4) 的四元数数组。
    """
    # 确定帧范围
    if Slice:
        frames = range(*Slice.indices(Slice.stop))
    else:
        action = armature.animation_data.action
        if not action:
            raise ValueError("Action Not found")
        start, end = map(int, action.frame_range)
        frames = range(start, end + 1)

    # 收集骨骼信息
    bones = {b.name: b for b in armature.pose.bones}
    if bone_resort:
        bones = {name: bones[name] for name in bone_resort if name in bones}
    bone_names = list(bones.keys())
    total_bones = len(bone_names)

    # 初始化数组
    rot_mode = get_bone_rotation_mode(armature)
    rot_arr = np.zeros((len(frames), total_bones, 4 if rot_mode == 'QUATERNION' else 3))

    # 导出数据
    pg = Progress(len(frames))
    # with progress_mouse(len(frames)) as update:
    for fr_i, frame in enumerate(frames):
        bpy.context.scene.frame_set(frame)
        for bone_i, bone_name in enumerate(bone_names):
            rot = get_bone_global_rotation(armature, bone_name)
            rot = rot.to_quaternion() if rot_mode == 'QUATERNION' else rot.to_euler(rot_mode)
            rot = np.array(rot)
            rot_arr[fr_i, bone_i] = rot
        pg.update()

    return rot_arr


def get_bones_relative_rotation(
    armature: 'bpy.types.Object',
    bone_resort: Sequence[str] | None = None,
    Slice: slice | None = None,
):
    """
    导出指定帧范围内骨骼**相对**旋转关键帧为四元数数组

    Args:
        Slice (slice): 帧范围，格式为slice(start, stop, step)，例如slice(1, 100, 1)

    Returns:
        numpy.ndarray: 形状为(total_frames, total_bones, 4)的四元数数组
    """
    action = armature.animation_data.action
    if not action:
        raise ValueError("Action Not found")

    # 收集骨骼信息
    bones = {b.name: b for b in armature.pose.bones}
    Log.debug(f'Bones before: {bones.keys()}')
    if bone_resort:
        Keys = bones.keys()
        bones = {b_name: bones[b_name] for b_name in bone_resort if b_name in Keys}
        Log.debug(f'Bones after: {bones.keys()}')
    total_bones = len(bones)

    # 预处理：为每个骨骼获取四元数FCurves和初始旋转值
    bone_fcurves = {}
    bone_init_quats = {}
    for name, bone in bones.items():
        data_path = f'pose.bones["{name}"].rotation_quaternion'
        fcurves = [action.fcurves.find(data_path, index=i) for i in range(4)]
        bone_fcurves[name] = fcurves
        bone_init_quats[name] = bone.rotation_quaternion

    # 准备帧范围
    if Slice:
        frames = range(*Slice.indices(Slice.stop))
    else:
        frames = range(int(action.frame_range[0]), int(action.frame_range[1]) + 1)

    # 初始化数组
    quat_array = np.zeros((len(frames), total_bones, 4))

    # 填充数据
    for fr_i, fr in enumerate(frames):
        for bone_i, bone_name in enumerate(bones.keys()):
            # 获取四元数分量
            quat = []
            for i in range(4):
                fcurve = bone_fcurves[bone_name][i]
                if fcurve:
                    value = fcurve.evaluate(fr)
                else:
                    # 使用初始旋转值
                    value = bone_init_quats[bone_name][i]
                quat.append(value)
            quat_array[fr_i, bone_i] = quat

    return quat_array


def add_keyframes(
    action: 'bpy.types.Action',
    vectors: Sequence[float] | Vector,
    frame: int,
    data_path: str,
    group='',
    interpolation: Literal['CONSTANT', 'LINEAR', 'BEZIER'] | str = 'BEZIER',
    **kwargs
):
    """
    add/override fcurve

    Args:
        frame (int): `frame_begin if is_multi_frames else at_which_frame`
        data_path (str): eg: `location` `rotation_quaternion` `rotation_euler` `pose.bones["pelvis"].location`
        group (str): fcurve group name

    Usage:
    ```python
    add_keyframes(action, rots[:, i], 1, f'pose.bones["{B}"].rotation_euler', f'{i}_{B}')
    ```
    """
    fcurves = action.fcurves
    kw: dict = dict(data_path=data_path)
    update = kwargs.pop('update', lambda: None)
    if isinstance(vectors[0], Number):
        is_multi = False
        channels = len(vectors)
    else:
        is_multi = True
        channels = len(vectors[0])  # type: ignore
    for C in range(channels):
        kw['index'] = C
        fcurve = fcurves.find(**kw)
        if not fcurve:
            fcurve = fcurves.new(**kw)

        if is_multi:
            for F in range(len(vectors)):
                keyframe = fcurve.keyframe_points.insert(frame + F, value=vectors[F][C])   # type: ignore
                keyframe.interpolation = interpolation  # type: ignore
                update() if F % channels == 0 else None
                if F % channels == 0:
                    yield
        else:
            keyframe = fcurve.keyframe_points.insert(frame, value=vectors[C])
            keyframe.interpolation = interpolation  # type: ignore
            update() if C % channels == 0 else None
            if C % channels:
                yield
        # Log.debug(f'is_multi={is_multi}, channels={channels}, shape={vectors.shape}', stack_info=False)

        if group and (not fcurve.group or fcurve.group.name != group):
            if group not in action.groups:
                action.groups.new(name=group)
            fcurve.group = action.groups[group]
    return action


def get_frame_range(action: 'bpy.types.Action'):
    """get range from F-curves, would raise ValueError"""
    if not action.fcurves:
        raise ValueError("Action has no F-Curves")
    min_frame = float('inf')
    max_frame = float('-inf')

    for fcurve in action.fcurves:
        Log.debug(f'{fcurve.data_path=} {fcurve.array_index=} {fcurve.group=}')
        if fcurve.keyframe_points:
            frames = [kp.co[0] for kp in fcurve.keyframe_points]
            min_frame = min(min_frame, min(frames))
            max_frame = max(max_frame, max(frames))
            Log.debug(f'{len(frames)=} {min_frame=} {max_frame=}')

    if min_frame != float('inf') and max_frame != float('-inf'):
        return (int(min_frame), int(max_frame))
    raise ValueError("Action has no valid frame range")


@contextmanager
def temp_override(
    area='NLA_EDITOR',
    mode: Literal['global', 'current'] = 'current'
):
    """
    Not recommended to use, all **`bpy.ops`** are **slower** than **pure data** manipulation.

    usage:
    ```python
    with temp_override(area='GRAPH_EDITOR', mode='global') as context:
        print(bpy.context == context)   # True
    ```
    """
    override = {}
    if mode == 'current':
        win = bpy.context.window
        scr = win.screen
        areas = [a for a in scr.areas if a.type == area]
        if not areas:
            raise RuntimeError(f"No area of type '{area}' found in the current screen.")
        # region = areas[0].regions[0]
        override = {
            # 'window': win,
            'screen': scr,
            'area': areas[0],
            # 'region': region,
            # 'scene': bpy.context.scene,
            # 'object': bpy.context.active_object,
        }
    elif mode == 'global':
        Break = False
        for screen in bpy.data.screens:
            for _area in screen.areas:
                if _area.type == area:
                    override = {"screen": screen, "area": _area}
                    Break = True
                    break
            if Break:
                break
    Log.debug(f'Context override: {override}')
    with bpy.context.temp_override(**override):
        yield bpy.context


@contextmanager
def bpy_action(
    obj: 'bpy.types.Object | None' = None,
    name='Action',
    nla_push=True,
):
    """
    Create a new action for object, at last push NLA and restore the old action after context.  
    Usage:```python
    with bpy_action(obj, name='MyAction', nla_push=True) as action:
        yield from add_keyframes(action, ...)
        # return add_keyframes(action, ...) # ⚠️ don't do this, or it will execute the cleanup work of `will` context
    ```

    TODO: Support Action Slot when blender 5.0 https://developer.blender.org/docs/release_notes/4.4/upgrading/slotted_actions/
    """
    old_action = track = strip = None
    start = 1
    obj = bpy.context.active_object if not obj else obj
    if not obj:
        raise ValueError('No object found')
    if not obj.animation_data:
        obj.animation_data_create()
    if not obj.animation_data:
        raise ValueError('No animation data found')
    if obj.animation_data.action:
        old_action = obj.animation_data.action
    action = obj.animation_data.action = bpy.data.actions.new(name=name)
    try:
        # Log.debug(f'action_suitable_slots={obj.animation_data.action_suitable_slots}')
        slot = action.slots.new(id_type='OBJECT', name=name)    # type: ignore
        obj.animation_data.action_slot = slot   # type: ignore
    except AttributeError:
        Log.info('skip create action slot because blender < 4.4')
    if nla_push:
        # find track that track.name==name
        tracks = [t for t in obj.animation_data.nla_tracks if t.name == name]
        if len(tracks) > 0:
            track = tracks[0]
        else:
            track = obj.animation_data.nla_tracks.new()
        track.name = name

        # append behind the last strip of current track
        strips = track.strips
        if len(strips) > 0:
            start = int(strips[-1].frame_end)
        strip = track.strips.new(name=name, start=start, action=action)
        # strip.extrapolation = 'HOLD'
        # strip.blend_type = 'REPLACE'
    yield action
    if nla_push and strip:
        Log.debug(f'Pushing NLA strip')
        # bpy.context.evaluated_depsgraph_get().update()
        # bpy.context.scene.update_tag()
        # bpy.context.view_layer.update()
        end = action.frame_range[1]
        strip.action_frame_start = start
        strip.action_frame_end = end
    if old_action and obj and obj.animation_data and obj.animation_data.action:
        obj.animation_data.action = old_action
    else:
        Log.debug('No action to restore')


@contextmanager
def progress_mouse(*Range: float, is_percent=True):
    """
    Show 4-digit progress number on mouse cursor with UI freeze.

    Args:
        is_percent: if True, convert Range into 0\\~10000 for **‱**
        Range: if Range is None and is_percent == True, fallback to 0\\~10000
        if len(Range) != 3 and is_percent == True, show **‱**, auto set step for 10 thousandths
    """
    MAX = 10000
    wm: 'bpy.types.WindowManager' = bpy.context.window_manager  # type: ignore
    R = list(Range)
    if is_percent:
        if len(Range) == 1:
            R = [0, MAX, MAX / Range[0]]
        elif len(Range) == 2:
            R = [0, MAX, MAX / (Range[1] - Range[0])]
    _step = R[2] if len(R) == 3 else 1
    R = np.arange(*R, dtype=np.float32)
    wm.progress_begin(R[0], R[-1])
    # Log.debug(f'progress🖱 {Range} → {R}', stack_info=False)
    i = 0
    v = R[0]
    last_update_time = time()
    update_interval = 0.5

    def update(Step: float | None = None, Set: float | None = None):
        """
        Args:
            Set: Any value between min and max as set in progress_mouse(Range=...)
        """
        nonlocal v, last_update_time
        if Set is not None:
            v = Set
        elif Step is None:
            v += _step
        else:
            v += Step
        current_time = time()
        if current_time - last_update_time >= update_interval:
            wm.progress_update(v)
            last_update_time = current_time
        return v
    yield update
    Log.debug(f'progress🖱 end {v}', stack_info=True)
    wm.progress_end()


def select_armature(deselect=True, exclude_name=set()):
    """Select armatures in the scene"""
    if deselect:
        bpy.ops.object.select_all(action='DESELECT')
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']
    if exclude_name:
        armatures = [obj for obj in armatures if obj.name not in exclude_name]
    for obj in armatures:
        obj.select_set(True)
    return armatures


def get_armatures(armatures: 'list[bpy.types.Object] | None' = None):
    """if None, always get active(selected) armature"""
    if not armatures:
        armatures = bpy.context.selected_objects
    if not armatures:
        if hasattr(bpy.ops.scene, 'smplx_add_gender'):
            exclude = {e.name for e in bpy.context.scene.objects}
            bpy.ops.scene.smplx_add_gender()    # type:ignore
            return select_armature(exclude_name=exclude)
        else:
            raise ValueError(f'Please select an armature or install smpl-x blender addon: {e}')
    for armature in armatures:
        if armature.type != 'ARMATURE':
            raise ValueError(f'Not an armature: {armature.name}')
    return armatures


def get_selected_bones(armature=None, context=None):
    """if None, always get active(selected) bones"""
    if not context:
        context = bpy.context
    obj = get_armatures([armature])[0]

    if context.mode == 'OBJECT':
        data = obj.data
        selected_bones = [bone for bone in data.bones if bone.select]
    elif context.mode == 'POSE':
        pose = obj.pose
        selected_bones = [bone for bone in pose.bones if bone.bone.select]
    else:
        raise ValueError("Please select an armature in OBJECT or POSE mode.")
    bones: list[str] = [bone.name for bone in selected_bones]
    len_bones = len(bones)
    len_smplx = len(Map()['smplx'].BODY)   # TODO: use smplx BONES
    if len_bones != len_smplx:
        Log.warning(f'{obj.name}: selected {len_bones}, but should be {len_smplx}')
    return bones


def bones_tree(armature: 'bpy.types.Object', whiteList: Sequence[str] | None = None):
    """bones to dict tree"""
    if armature and armature.type == 'ARMATURE':
        for bone in armature.pose.bones:
            if not bone.parent:
                return {bone.name: bone_to_dict(bone, whiteList)}
    return {}


def add_mapping(armatures: Sequence['bpy.types.Object'] | None = None, check=True):
    """
    add mapping based on selected armatures
    """
    files: list[str] = []
    if not armatures:
        armatures = get_armatures()
    for armature in armatures:
        tree = bones_tree(armature, whiteList=get_selected_bones(armature=armature))
        bones = keys_BFS(tree)
        map = {}
        for x, my in zip(Map()['smplx'].BONES, bones, strict=False):
            map[x] = my

        t = ''
        with open(MAPPING_TEMPLATE, 'r') as f:
            t = f.read()
        # fastest way but not safe, {format} to 《》
        t = re.sub(r'\{(.*)\}(?= *#)', r'《\1》', t)
        # {} to 「」
        t = re.sub(r'{', '「', t)
        t = re.sub(r'}', '」', t)
        # 《》 to {}
        t = re.sub(r'《', '{', t)
        t = re.sub(r'》', '}', t)
        t = t.format(t, armature=armature.name, type_body=bones, map=map, bones_tree=tree)
        # 「」 to {}
        t = re.sub(r'「', '{', t)
        t = re.sub(r'」', '}', t)

        filename = f'{armature.name}.py'
        file: str = os.path.join(DIR_MAPPING, filename)
        if check and os.path.exists(file):
            Log.error(f'Mapping exists: {file}')
        else:
            with open(file, 'w') as f:
                f.write(t)
        files.append(file)
    return files


def guess_obj_mapping(obj: 'bpy.types.Object') -> TYPE_MAPPING:
    bones = bones_tree(obj)
    keys = keys_BFS(bones)
    mapping = None
    max_similar = 0
    for map, mod in Map().items():
        similar = get_similar(keys, mod.BONES)
        if similar > max_similar:
            max_similar = similar
            mapping = map
    Log.info(f'Guess mapping to: {mapping} with {max_similar:.2f}')
    return mapping  # type: ignore


def get_bones(
    armature: 'bpy.types.Object',
    key: TYPE_MAPPING_KEYS = 'BONES',
    mapping: TYPE_MAPPING | None = None,
):
    """
    guess mapping[smpl,smplx]/Range_end/bone_rotation_mode[eular,quat]

    Usage:
    ```python
    data, BODY, armature, Slice = check_before_run(data, 'BODY', 'gvhmr', mapping, Slice)
    ```
    """
    mapping = None if mapping == 'auto' else mapping
    mapping = get_mapping(mapping=mapping, armature=armature)
    BONES: list[str] = getattr(Map()[mapping], key)   # type:ignore
    # Log.debug("mapping from {}".format(f'{data.mapping}→{mapping}' if data.mapping[:2] != mapping[:2] else mapping))
    return BONES


def get_slice(data: MotionData, Slice: slice):
    Log.debug(f'{data.keys()=}')
    if Slice.stop is None:
        Len = len(data('global_orient').value)   # TODO: 使用专有信息 npz['meatadata'](dtype=object) as dict
        t = list(Slice.indices(Len))
        t[1] = Len
        Slice = slice(*t)
        Log.info(f'Frame range (Slice) fallback to {Slice}')
    return Slice


def get_bone_rotation_mode(armature: 'bpy.types.Object'):
    bones_rots: list[TYPE_ROT] = [b.rotation_mode for b in armature.pose.bones]
    rot = get_major(bones_rots)
    rot = 'QUATERNION' if not rot else rot
    return rot


def get_bones_info(armature=None):
    """For debug: print bones info"""
    armatures = get_armatures()
    S = ""
    for armature in armatures:
        tree = bones_tree(armature=armature)
        List = keys_BFS(tree)
        S += f"""TYPE_BODY = Literal{List}
BONES_TREE = {tree}"""
        # for b in List:
        #     global_rot = get_bone_global_rotation(armature=armature, bone=b)
        #     S += f"\n{b}:\n{global_rot.to_quaternion()}"
        cur = bpy.context.scene.frame_current   # type: ignore
        S += '\nget_bones_global_rotation:\n' + str(get_bones_global_rotation(armature=armature, bone_resort=List, Slice=slice(cur, cur + 1)))
        S += '\nget_bones_relative_rotation:\n' + str(get_bones_relative_rotation(armature=armature, bone_resort=List, Slice=slice(cur, cur + 1)))
    return S


def data_Slice_name_transl_rotate(data: MotionData, Slice: slice, run: TYPE_RUN):
    data = data(mapping=data.mapping, run=run)  # type: ignore
    name = ';'.join([data.who, data.run])
    Slice = get_slice(data, Slice)

    transl = data('transl')
    if any(transl):
        transl = transl.value[Slice]
    else:
        transl = None
    rotate = data('global_orient').value[Slice]
    return data, Slice, name, transl, rotate


def armature_BODY(mapping: TYPE_MAPPING | None = None, key: TYPE_MAPPING_KEYS = 'BONES'):
    armature = get_armatures()[0]
    BODY = get_bones(armature, key=key, mapping=mapping)
    return armature, BODY


def decimate(
    action: 'bpy.types.Action',
    bones: Sequence[str],
    clean_th: float,
    decimate_th: float,
    rots,
    keep_end: bool = False,
    **kwargs,
):
    '''
    Would raise AttributeError if can't find GRAPH_EDITOR area.
    '''
    if clean_th <= 0 and decimate_th <= 0:
        return
    # TODO: FK转IK，旋转→位置，平滑动画
    # TODO: ⭐复制优化前动画，二分法调整threshold直到用户满意（方便后期手工曲线编辑）⭐
    # TODO: 增加armatures参数，未处理多骨架同时导入
    try:
        with temp_override(area='GRAPH_EDITOR', mode='global') as context:
            ...
    except AttributeError:
        raise RuntimeError("GRAPH_EDITOR area not found, please open a GRAPH_EDITOR area to use this function.")
    with temp_override(area='GRAPH_EDITOR', mode='global') as context:
        obj = context.active_object
        old_show = context.area.spaces[0].dopesheet.show_only_selected
        context.area.spaces[0].dopesheet.show_only_selected = True

        old_bones = [b for b in obj.pose.bones if not b.bone.select]
        for b in old_bones:
            b.bone.select = True

        exclude = [len(rots)] if keep_end else []   # TODO: remove rots args
        for fcurve in action.fcurves:
            # 过滤掉非关键通道（如位置通道可能不需要处理）
            if not any(b in fcurve.data_path for b in bones):
                continue
            for keyframe in fcurve.keyframe_points:
                if keyframe.co[0] in exclude:
                    keyframe.select_control_point = False

        if clean_th > 0:
            bpy.ops.graph.clean(threshold=clean_th, channels=False)
        if decimate_th > 0:
            bpy.ops.graph.decimate(remove_error_margin=decimate_th, mode='ERROR')

        context.area.spaces[0].dopesheet.show_only_selected = old_show
        for b in old_bones:
            b.bone.select = False


def pose_reset(
    action: 'bpy.types.Action',
    bones: Sequence[str],
    rot: TYPE_ROT = 'QUATERNION',
    frame=0,
):
    """Reset to keyframes, with bones to ZERO rotation at `frame`"""
    if rot == 'QUATERNION':
        ZERO = [1, 0, 0, 0]
        path = 'pose.bones["{}"].rotation_quaternion'
    else:
        ZERO = [0, 0, 0]
        path = 'pose.bones["{}"].rotation_euler'
    for B in bones:
        add_keyframes(action, ZERO, frame, path.format(B), B, 'CONSTANT')
    return action


def pose_apply(
    armature: 'bpy.types.Object',
    action: 'bpy.types.Action',
    bones: Sequence[str],
    pose: 'np.ndarray',
    transl: 'np.ndarray | None' = None,
    transl_base: 'np.ndarray | None' = None,
    frame=1,
    **kwargs
):
    """Apply to keyframes, with translation, pose, and shape to character using Action and F-Curves.

    Args:
        bones: `index num` mapping to `bone names`
        pose: `global_orient` + `pose` rotations, shape==`(frames, len_bones+1 , 3 or 4)`
        transl_base: if **NOT None**, transl will be **relative** to transl_base
        rot: blender rotation mode
        frame: begin frame
        clean_th: -1 to disable,suggest 0.001~0.005; **keep default bezier curve handle ⚫**, clean nearby keyframes if `current-previous > threshold`; aims to remove time noise/tiny shake
        decimate_th: -1 to disable, suggest 0.001~0.1; **will modify curve handle ⚫→🔶**, decide to decimate current frame if `error=new-old < threshold`; aims to be editable
    """
    rot = get_bone_rotation_mode(armature)
    method = str(kwargs.get('rot', 0)).lower()
    method = method[0] if len(method) > 0 else ''
    if rot == 'QUATERNION':
        if method == 'a':  # axis angle
            rots = quat_rotAxis(pose)
        elif method == 'r':  # raw
            rots = pose
        else:
            rots = quat(pose)
        path = 'pose.bones["{}"].rotation_quaternion'
    else:
        rots = euler(pose)
        path = 'pose.bones["{}"].rotation_euler'

    pg = Progress(len(bones) * len(rots))
    if transl is not None:
        if transl_base is not None:
            transl = transl - transl_base
            add_keyframes(action, transl_base, frame, f'location', 'Object Transforms')
        pg_t = Progress(len(transl))
        yield from add_keyframes(action, transl, frame, f'pose.bones["{bones[0]}"].location', bones[0], update=pg_t.update)

    bones = bones[1:] if bones[0] == 'root' else bones  # Skip root!
    # with progress_mouse(len(bones) * len(rots)) as update:
    for i, B in enumerate(bones):
        yield from add_keyframes(action, rots[:, i], frame, path.format(B), B, update=pg.update)
    pose_reset(action, bones, rot)


def transform_apply(
    obj: 'bpy.types.Object',
    action: 'bpy.types.Action',
    rotate: 'np.ndarray | None' = None,
    transl: 'np.ndarray | None' = None,
    frame: int = 1,
):
    rot = obj.rotation_mode
    path = 'rotation_quaternion' if rot == 'QUATERNION' else 'rotation_euler'
    if rotate is not None:
        pg_r = Progress(len(rotate))
    if transl is not None:
        pg_t = Progress(len(transl))
        yield from add_keyframes(action, transl, frame, 'location', 'Object Transforms', update=pg_t.update)
    if rotate is not None:
        # with progress_mouse(len(rotate)) as update:
        yield from add_keyframes(action, rotate, frame, path, 'Object Transforms', update=pg_r.update)
