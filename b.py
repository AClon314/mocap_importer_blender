import os
import re
import bpy
import numpy as np
from .lib import DIR_MAPPING, MAPPING_TEMPLATE, TYPE_MAPPING, TYPE_RUN, Log, Map, MotionData, bone_to_dict, euler, get_major, get_mapping, get_similar, keys_BFS, quat, quat_rotAxis
from time import time
from numbers import Number
from contextlib import contextmanager
from typing import Literal, Sequence, get_args
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


def bone_global_rotation_matrix(
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


def bones_rotation(
    armature: 'bpy.types.Object',
    bone_resort: Sequence[str] | None = None,
    Slice: slice | None = None,
):
    """
    导出指定帧范围内骨骼旋转关键帧为四元数数组

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
    Log.debug(f'Bones: {bones.keys()}')
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
        else:
            keyframe = fcurve.keyframe_points.insert(frame, value=vectors[C])
            keyframe.interpolation = interpolation  # type: ignore
            update() if C % channels == 0 else None
        # Log.debug(f'is_multi={is_multi}, channels={channels}, shape={vectors.shape}', stack_info=False)

        if group and (not fcurve.group or fcurve.group.name != group):
            if group not in action.groups:
                action.groups.new(name=group)
            fcurve.group = action.groups[group]
    return action


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
    Create a new action for object, at last push NLA and restore the old action after context

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
    # Log.debug(f'old_action={old_action}')
    action = obj.animation_data.action = bpy.data.actions.new(name=name)
    try:
        # Log.debug(f'action_suitable_slots={obj.animation_data.action_suitable_slots}')
        slot = action.slots.new(id_type='OBJECT', name=name)
        obj.animation_data.action_slot = slot
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
        # Log.debug(f'start={start}, strips={strips}')
        strip = track.strips.new(name=name, start=start, action=action)
        # strip.extrapolation = 'HOLD'
        # strip.blend_type = 'REPLACE'
    yield action
    if nla_push and strip:
        Len = action.frame_range[1]
        strip.action_frame_end = Len
        # strip.frame_end = start + Len
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
        Mod: reduce UI update. if Mod == 0, will auto decide Mod number.
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
    Log.debug(f'progress🖱 {Range} → {R}', stack_info=False)
    i = 0
    v = R[0]
    mod = 1
    timer = time()

    def update(Step: float | None = None, Set: float | None = None):
        """
        Args:
            Set: Any value between min and max as set in progress_mouse(Range=...)
        """
        nonlocal i, v, mod, timer
        i += 1
        if Set is not None:
            v = Set
        elif Step is None:
            v += _step
        else:
            v += Step
        if Set or i % mod < 1:
            if timer > 0:
                t = timer
                timer = time()
                if timer - t < 0.25:
                    mod *= 2
                else:
                    timer = -1
            wm.progress_update(v)
        return v
    yield update
    Log.debug(f'progress🖱 end {v}', stack_info=True)
    wm.progress_end()


def get_armatures(armatures: 'list[bpy.types.Object] | None' = None):
    """if None, always get active(selected) armature"""
    if not armatures:
        armatures = bpy.context.selected_objects
    if not armatures:
        raise ValueError('Please select an armature')
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


def check_before_run(
    data: MotionData,
    key: str,
    run: TYPE_RUN,
    mapping: TYPE_MAPPING | None = None,
    Slice=slice(0, None),
):
    """
    guess mapping[smpl,smplx]/Range_end/bone_rotation_mode[eular,quat]
    TODO: OMG, this shit💩 is too bad, need to refactor

    Usage:
    ```python
    global BODY
    data, armature, bone_rot, BODY, _Range = check_before_run('gvhmr','BODY', data, Range, mapping)
    ```
    """
    data = data(mapping=data.mapping, run=run)  # type: ignore
    armature = get_armatures()[0]

    mapping = None if mapping == 'auto' else mapping
    mapping = get_mapping(mapping=mapping, armature=armature)
    BONES = getattr(Map()[mapping], key, 'BODY')   # type:ignore
    Log.debug("mapping from {}".format(f'{data.mapping}→{mapping}' if data.mapping[:2] != mapping[:2] else mapping))

    if Slice.stop is None:
        Len = len(data('global_orient').value)   # TODO: 使用专有信息 npz['meatadata'](dtype=object) as dict
        t = list(Slice.indices(Len))
        t[1] = Len
        Slice = slice(*t)
        Log.info(f'Frame range (Slice) fallback to {Slice}')
    return data, BONES, armature, Slice


def bone_rotation_mode(armature):
    bones_rots: list[TYPE_ROT] = [b.rotation_mode for b in armature.pose.bones]
    rot = get_major(bones_rots)
    rot = 'QUATERNION' if not rot else rot
    return rot


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
    clean_th=0.002,
    decimate_th=0.005,
    keep_end=False,
    **kwargs
):
    """Apply to keyframes, with translation, pose, and shape to character using Action and F-Curves.

    Args:
        bones: `index num` mapping to `bone names`
        transl_base: if **NOT None**, transl will be **relative** to transl_base
        rot: blender rotation mode
        frame: begin frame
        clean_th: -1 to disable,suggest 0.001~0.005; **keep default bezier curve handle ⚫**, clean nearby keyframes if `current-previous > threshold`; aims to remove time noise/tiny shake
        decimate_th: -1 to disable, suggest 0.001~0.1; **will modify curve handle ⚫→🔶**, decide to decimate current frame if `error=new-old < threshold`; aims to be editable
    """
    rot = bone_rotation_mode(armature)
    method = str(kwargs.get('quat', 0))[0]
    if rot == 'QUATERNION':
        if method == 'a':  # axis
            rots = quat_rotAxis(pose)
        elif method == 'r':  # raw
            rots = pose
        else:
            rots = quat(pose)
        path = 'pose.bones["{}"].rotation_quaternion'
    else:
        rots = euler(pose)
        path = 'pose.bones["{}"].rotation_euler'

    if transl is not None:
        if transl_base is not None:
            transl = transl - transl_base
            add_keyframes(action, transl_base, frame, f'location', 'Object Transforms')
        add_keyframes(action, transl, 1, f'pose.bones["{bones[0]}"].location', bones[0])

    bones = bones[1:] if bones[0] == 'root' else bones  # Skip root!
    with progress_mouse(len(bones) * len(rots)) as update:
        for i, B in enumerate(bones):
            add_keyframes(action, rots[:, i], frame, path.format(B), B, update=update)

    is_clean = clean_th > 0
    is_decimate = decimate_th > 0
    if is_clean or is_decimate:
        # TODO: FK转IK，旋转→位置，平滑动画
        # TODO: ⭐复制优化前动画，二分法调整threshold直到用户满意（方便后期手工曲线编辑）⭐
        # TODO: 增加armatures参数，未处理多骨架同时导入
        with temp_override(area='GRAPH_EDITOR', mode='global') as context:
            obj = context.active_object
            old_show = context.area.spaces[0].dopesheet.show_only_selected
            context.area.spaces[0].dopesheet.show_only_selected = True

            old_bones = [b for b in obj.pose.bones if not b.bone.select]
            for b in old_bones:
                b.bone.select = True

            # exclude = [1] if keep_begin else []
            exclude = [len(rots)] if keep_end else []
            for fcurve in action.fcurves:
                # 过滤掉非关键通道（如位置通道可能不需要处理）
                if not any(b in fcurve.data_path for b in bones):
                    continue
                for keyframe in fcurve.keyframe_points:
                    if keyframe.co[0] in exclude:
                        keyframe.select_control_point = False

            if is_clean:
                bpy.ops.graph.clean(threshold=clean_th, channels=False)
            if is_decimate:
                bpy.ops.graph.decimate(remove_error_margin=decimate_th, mode='ERROR')

            context.area.spaces[0].dopesheet.show_only_selected = old_show
            for b in old_bones:
                b.bone.select = False
    pose_reset(action, bones, rot)
    return action
