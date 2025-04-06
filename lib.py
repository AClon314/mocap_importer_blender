"""
https://github.com/carlosedubarreto/CEB_4d_Humans/blob/main/four_d_humans_blender.py
"""
import os
import re
import bpy
import sys
import logging
import inspect
import importlib
import numpy as np
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Union, Literal, TypeVar, get_args
DIR_SELF = os.path.dirname(__file__)
DIR_MAPPING = os.path.join(DIR_SELF, 'mapping')
MAPPING_TEMPLATE = os.path.join(DIR_MAPPING, 'template.pyi')
TYPE_MAPPING = Literal['smpl', 'smplx']
TYPE_RUN = Literal['gvhmr', 'wilor']
TYPE_I18N = Literal[  # blender 4.3.2
    'ca_AD', 'en_US', 'es', 'fr_FR', 'ja_JP', 'sk_SK', 'ur', 'vi_VN', 'zh_HANS',
    'de_DE', 'it_IT', 'ka', 'ko_KR', 'pt_BR', 'pt_PT', 'ru_RU', 'sw', 'ta', 'tr_TR', 'uk_UA', 'zh_HANT',
    'ab', 'ar_EG', 'be', 'bg_BG', 'cs_CZ', 'da', 'el_GR', 'eo', 'eu_EU', 'fa_IR', 'fi_FI', 'ha', 'he_IL', 'hi_IN', 'hr_HR', 'hu_HU', 'id_ID', 'km', 'ky_KG', 'lt', 'ne_NP', 'nl_NL', 'pl_PL', 'ro_RO', 'sl', 'sr_RS', 'sr_RS@latin', 'sv_SE', 'th_TH'
]
T = TypeVar('T')
TN = np.ndarray
MOTION_DATA = None
LEVEL_PREFIX = {
    logging.DEBUG: '🐛DEBUG',
    logging.INFO: '💬 INFO',
    logging.WARNING: '⚠️  WARN',
    logging.ERROR: '❌ERROR',
    logging.CRITICAL: '⛔⛔CRITICAL',
    logging.FATAL: '☠️FATAL',
}
try:
    from warnings import deprecated
except ImportError:
    deprecated = lambda *args, **kwargs: lambda func: func


def caller_name(skips=8, frame=None):
    """skip: ['caller_name', 'format', 'emit', 'handle', 'callHandlers', '_log', 'debug', 'info', 'warning', 'error', 'critical', 'fatal']"""
    if frame is None:
        frame = inspect.currentframe()
    while frame:
        if skips >= 0:
            skips -= 1
            frame = frame.f_back
            continue
        name = frame.f_code.co_name + '()'
        parent = frame.f_back
        name_parent = ''
        if parent:
            name_parent = '←' + parent.f_code.co_name + '()'
        return name + name_parent
    return ''


def getLogger(name=__name__, level=10):
    """```python
    Log = getLogger(__name__)
    ```"""
    Log = logging.getLogger(name)
    Log.setLevel(level)
    Log.propagate = False   # disable propagate to root logger
    stream_handler = logging.StreamHandler()

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = LEVEL_PREFIX.get(record.levelno, record.levelname)
            record.msg = f'{record.msg}\t{caller_name()}'
            return super().format(record)

    stream_handler.setFormatter(
        CustomFormatter(
            '%(levelname)s\t%(asctime)s  %(message)s\t%(name)s:%(lineno)d',
            datefmt='%H:%M:%S'))
    Log.addHandler(stream_handler)
    return Log


Log = getLogger(__name__)
ID = __package__.split('.')[-1] if __package__ else __name__
try:
    from mathutils import Matrix, Vector, Quaternion, Euler
except ImportError as e:
    Log.warning(e)
TYPE_PROP = Literal['body_pose', 'hand_pose', 'global_orient', 'betas', 'transl', 'bbox']
PROP_KEY = get_args(TYPE_PROP)
TYPE_ROT = Literal['QUATERNION', 'XYZ', 'YZX', 'ZXY', 'XZY', 'YXZ', 'ZYX', 'AXIS_ANGLE']
ROT_KEY = get_args(TYPE_ROT)
def get_major(L: Sequence[T]) -> T | None: return max(L, key=L.count) if L else None


def skip_or_in(part, full, pattern=';{};'):
    """used in class `MotionData`"""
    if pattern:
        part = pattern.format(part) if part else None
        full = pattern.format(full) if full else None
    return (not part) or (not full) or (part in full)


def warn_or_return_first(L: List[T]) -> T:
    """used in class `MotionData`"""
    Len = len(L)
    if Len > 1:
        Log.warning(f'{Len} > 1 from {L}')
    return L[0]


def Mod(Dir='mapping'):
    files = os.listdir(os.path.join(DIR_SELF, Dir))
    pys = []
    mods: Dict[str, ModuleType] = {}
    for f in files:
        if f.endswith('.py') and f not in ['template.py', '__init__.py']:
            pys.append(f[:-3])
    for p in pys:
        mod = importlib.import_module(f'.{Dir}.{p}', package=__package__)
        mods.update({p: mod})
    return mods


def Map(Dir='mapping'): return Mod(Dir=Dir)
def Run(Dir='run'): return Mod(Dir=Dir)


def items_mapping(self=None, context=None):
    items: List[tuple[str, str, str]] = [(
        'auto', 'Auto',
        'Auto detect armature type, based on name (will enhanced in later version)')]
    for k, m in Map().items():
        help = ''
        try:
            help = m.HELP[bpy.app.translations.locale]
        except Exception:
            Log.warning(f'No help for {k}')
            help = m.__doc__ if m.__doc__ else ''
        items.append((k, k, help))
    return items


def items_motions(self=None, context=None):
    items: List[tuple[str, str, str]] = []
    if MOTION_DATA is None:
        load_data()
    if MOTION_DATA is not None:
        for k in MOTION_DATA.whos:
            items.append((k, k, ''))
    else:
        raise ValueError('Failed to load motion data')
    return items


def load_data(self=None, context=None):
    """load motion data when npz file path changed"""
    global MOTION_DATA
    if MOTION_DATA is not None:
        del MOTION_DATA
    file = bpy.context.scene.mocap_importer.input_npz   # type: ignore
    MOTION_DATA = MotionData(npz=file)


def add_keyframe(
    action: 'bpy.types.Action',
    data_path: str,
    at_frame: int,
    vector: Union[Sequence, Vector, Quaternion, Euler],
):
    """
    also override fcurve if exists

    Usage:
    ```python
    insert_frame(action, f'pose.bones["{BODY[i]}"].rotation_quaternion', frame, quat)
    ```
    """
    fcurves = action.fcurves
    kw = dict(data_path=data_path, index=0)
    for i, x in enumerate(vector):  # type: ignore
        kw['index'] = i
        fcurve = fcurves.find(**kw)  # type: ignore
        if not fcurve:
            fcurve = fcurves.new(**kw)  # type: ignore
        fcurve.keyframe_points.insert(at_frame, value=x)
    return action


def context_override(area='NLA_EDITOR'):
    """
    Not recommended to use, all **`bpy.ops`** are **slower** than **pure data** manipulation.

    usage:
    ```python
    with bpy.context.temp_override(**context_override()):
        bpy.ops.nla.action_pushdown(override)
    ```
    """
    win = bpy.context.window
    scr = win.screen
    areas = [a for a in scr.areas if a.type == area]
    if not areas:
        raise RuntimeError(f"No area of type '{area}' found in the current screen.")
    region = areas[0].regions[0]
    override = {
        'window': win,
        'screen': scr,
        'area': areas[0],
        'region': region,
        'scene': bpy.context.scene,
        'object': bpy.context.active_object,
    }
    Log.debug(f'Context override: {override}')
    return override


@contextmanager
def new_action(
    obj: Optional['bpy.types.Object'] = None,
    name='Action',
    nla_push=True,
):
    """
    Create a new action for object, at last push NLA and restore the old action after context

    TODO: Support Action Slot when blender 5.0 https://developer.blender.org/docs/release_notes/4.4/upgrading/slotted_actions/
    """
    old_action = track = strip = None
    start = 1
    try:
        if not obj:
            obj = bpy.context.active_object
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
            Log.debug(f'action_suitable_slots={obj.animation_data.action_suitable_slots}')
            slot = action.slots.new(id_type='OBJECT', name=name)
            obj.animation_data.action_slot = slot
        except AttributeError:
            Log.info('skip create action slot because blender < 4.4')
        if nla_push:
            # find track that track.name==name, append behind the last strip of the same track
            tracks = [t for t in obj.animation_data.nla_tracks if t.name == name]
            if len(tracks) > 0:
                track = tracks[0]
            else:
                track = obj.animation_data.nla_tracks.new()
            track.name = name

            strips = track.strips
            if len(strips) > 0:
                start = int(strips[-1].frame_end)
            # Log.debug(f'start={start}, strips={strips}')
            strip = track.strips.new(name=name, start=start, action=action)
            # strip.extrapolation = 'HOLD'
            # strip.blend_type = 'REPLACE'
        yield action
    finally:
        if nla_push and strip:
            Len = action.frame_range[1]
            strip.action_frame_end = Len
            # strip.frame_end = start + Len
        if old_action and obj and obj.animation_data and obj.animation_data.action:
            obj.animation_data.action = old_action
        else:
            Log.info('No action to restore')


class MotionData(dict):
    """
    usage:
    ```python
    data(mapping='smplx', run='gvhmr', key='trans', coord='global').values()[0]
    ```
    """

    def keys(self) -> List[str]:
        return list(super().keys())

    def values(self) -> List[np.ndarray]:
        return list(super().values())

    def __init__(self, /, *args, npz: Union[str, os.PathLike, None] = None, lazy=False, **kwargs):
        """
        Inherit from dict
        Args:
            npz (str, Path, optional): npz file path.
            lazy (bool, optional): if True, do NOT load npz file.
        """
        super().__init__(*args, **kwargs)
        if npz:
            self.npz = npz
            if not lazy:
                self.update(np.load(npz, allow_pickle=True))

    def __call__(
        self,
        mapping: Optional[TYPE_MAPPING] = None,
        run: Optional[TYPE_RUN] = None,
        who: Union[str, int, None] = None,
        prop: Optional[TYPE_PROP] = None,
        coord: Optional[Literal['global', 'incam']] = None,
    ):
        # Log.debug(f'self.__dict__={self.__dict__}')
        D = MotionData(npz=self.npz, lazy=True)
        if isinstance(who, int):
            who = f'person{who}'

        for k, v in self.items():
            is_in = [skip_or_in(args, k) for args in [mapping, run, who, prop, coord]]
            is_in = all(is_in)
            if is_in:
                D[k] = v
        return D

    def distinct(self, col_num: int):
        """
        Args:
            col_num (int): 0 for mapping, 1 for run, 2 for key, 3 for person, 4 for coord
            literal : filter keys by Literal. Defaults to None.

        """
        L: List[str] = []
        for k in self.keys():
            if isinstance(k, str):
                keys = k.split(';')
                if len(keys) > col_num:
                    col_name = keys[col_num]
                    if col_name not in L:
                        L.append(col_name)
        return L

    @property
    def mappings(self): return self.distinct(0)
    @property
    def runs_keyname(self): return self.distinct(1)
    @property
    def whos(self): return self.distinct(2)

    @property
    def props(self):
        """could return `[your_customkeys, '*_pose', 'global_orient', 'transl', 'betas']`"""
        return self.distinct(3)

    @property
    def coords(self): return self.distinct(4)

    @property
    def mapping(self): return warn_or_return_first(self.mappings)
    @property
    def run_keyname(self): return warn_or_return_first(self.runs_keyname)
    @property
    def prop(self): return self.props[0]
    @property
    def who(self): return warn_or_return_first(self.whos)
    @property
    def coord(self): return warn_or_return_first(self.coords)

    @property
    def value(self):
        """same as:
        ```python
        return self.values()[0]
        ```"""
        return warn_or_return_first(self.values())

    @property
    def keyname(self):
        """return **FULL** keyname like `smplx;gvhmr;pose;person0;global`, same as:
        ```python
        return self.keys()[0]
        ```"""
        return self.keys()[0]


def log_array(arr: Union[np.ndarray, list], name='ndarray'):
    def recursive_convert(array):
        if isinstance(array, np.ndarray):
            return array.tolist()
        elif isinstance(array, list):
            return [recursive_convert(item) for item in array]
        else:
            return array

    def array_to_str(array):
        if isinstance(array, list):
            return '\t'.join(array_to_str(item) for item in array)
        else:
            return str(array)

    if isinstance(arr, list):
        arr = np.array(arr)

    array = recursive_convert(arr.tolist())
    array = array_to_str(array)
    text = f'{name}={array}'
    Log.debug(text)
    print()
    return text


def dump_bones(armature):
    """将骨架转换为字典"""
    def bone_to_dict(bone, deep=0, deep_max=1000):
        if deep_max and deep > deep_max:
            raise ValueError(f'Bones tree too deep, {deep} > {deep_max}')
        return {child.name: bone_to_dict(child, deep + 1) for child in bone.children}

    if armature and armature.type == 'ARMATURE':
        for bone in armature.pose.bones:
            if not bone.parent:
                return {bone.name: bone_to_dict(bone)}
    return {}


def keys_BFS(
    d: dict, wrap=False,
    deep_max=1000,
):
    """
    sort keys of dict by BFS

    Parameters
    ----------
    d : dict
        dict to sort
    wrap : bool, optional
        if True, return [[key0], [k1,k2], [k3,k4,k5], ...]
        else return [key0, k1, k2, k3, k4, k5, ...]
    """
    deep = 0
    ret = []
    Q = [d]  # 初始队列包含根字典
    while Q:
        if deep_max and deep > deep_max:
            raise ValueError(f'Dict tree too deep, {deep} > {deep_max}')
        current_level = []
        next_queue = []
        for current_dict in Q:
            # 收集当前字典的所有键到当前层级
            current_level.extend(current_dict.keys())
            # 收集当前字典的所有子字典到下一层队列
            next_queue.extend(current_dict.values())
        # 如果当前层级有键，则添加到结果中
        if current_level:
            if wrap:
                ret.append(current_level)
            else:
                ret.extend(current_level)
        # 更新队列为下一层级的子字典列表
        Q = next_queue
        deep += 1
    return ret


def get_similar(list1, list2):
    """
    calc jaccard similarity of two lists
    Returns:
        float: ∈[0, 1]
    """
    set1, set2 = set(list1), set(list2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    ret = intersection / union if union != 0 else 0
    return ret


def guess_obj_mapping(obj: 'bpy.types.Object', select=True) -> Union[TYPE_MAPPING, None]:
    if obj.type != 'ARMATURE':
        return None
    bones = dump_bones(obj)
    keys = keys_BFS(bones)
    mapping = None
    max_similar = 0
    for map, mod in Map().items():
        similar = get_similar(keys, mod.BONES)
        if similar > max_similar:
            max_similar = similar
            mapping = map
    if mapping and select:
        bpy.context.view_layer.objects.active = obj
    Log.info(f'Guess mapping: {mapping} with {max_similar:.2f}')
    return mapping  # type: ignore


def get_mapping_from_selected_or_objs(mapping: Union[TYPE_MAPPING, None] = None):
    """
    import mapping module by name
    will set global variable BODY(temporary)
    """
    if mapping is None:
        # guess mapping
        active = bpy.context.active_object
        if active:
            mapping = guess_obj_mapping(active)
        else:
            for obj in bpy.data.objects:
                if obj.type == 'ARMATURE':
                    mapping = guess_obj_mapping(obj)
                    if mapping:
                        break
    if mapping is None:
        raise ValueError(f'Unknown mapping: {mapping}, try to select/add new mapping to')
    return mapping


def add_mapping(armature):
    """
    add mapping based on selected armature
    """
    if not armature:
        armature = bpy.context.active_object
    if not armature or armature.type != 'ARMATURE':
        raise ValueError('Please select an armature')
    bones_tree = dump_bones(armature)
    bones = keys_BFS(bones_tree)
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
    t = t.format(t, type_body=bones, map=map, bones_tree=bones_tree)
    # 「」 to {}
    t = re.sub(r'「', '{', t)
    t = re.sub(r'」', '}', t)

    filename = f'{armature.name}.py'
    file: str = os.path.join(DIR_MAPPING, filename)
    if os.path.exists(file):
        raise FileExistsError(f'Mapping exists: {file}')
    else:
        with open(file, 'w') as f:
            f.write(t)
    Log.info(f'Restart addon to update mapping:  {file}')


def check_before_run(
    run: TYPE_RUN,
    key: str,
    data: MotionData,
    Range: list,
    mapping: Optional[TYPE_MAPPING] = None
):
    """
    guess mapping[smpl,smplx]/Range_end/bone_rotation_mode[eular,quat]

    Usage:
    ```python
    global BODY
    data, armature, bone_rot, BODY, _Range = check_before_run('gvhmr','BODY', data, Range, mapping)
    ```
    """
    data = data(mapping=data.mapping, run=run)  # type: ignore
    is_range = len(Range) > 1

    armature = bpy.context.active_object
    if armature is None or armature.type != 'ARMATURE':
        raise ValueError('No armature found')
    bones_rots: list[TYPE_ROT] = [b.rotation_mode for b in armature.pose.bones]
    bone_rot = get_major(bones_rots)
    bone_rot = 'QUATERNION' if not bone_rot else bone_rot

    mapping = None if mapping == 'auto' else mapping
    mapping = get_mapping_from_selected_or_objs(mapping)
    BONES = getattr(Map()[mapping], key, 'BODY')   # type:ignore

    if is_range and Range[1] is None:
        Range[1] = len(data(prop='global_orient').value)    # TODO: use data.frames
        Log.info(f'range_frame[1] fallback to {Range[1]}')
    _Range = range(*Range)
    str_map = f'{data.mapping}→{mapping}' if data.mapping[:2] != mapping[:2] else mapping
    Log.info(f'mapping from {str_map}')
    return data, armature, bone_rot, BONES, _Range


def apply(who: Union[str, int], mapping: Optional[TYPE_MAPPING], **kwargs):
    global MOTION_DATA
    if MOTION_DATA is None:
        raise ValueError('Failed to load motion data')
    data = MOTION_DATA(mapping='smplx', who=who)
    for r in data.runs_keyname:
        run = getattr(Run()[r], r)
        run(data, mapping=mapping, **kwargs)


def Axis(is_torch=False): return 'dim' if is_torch else 'axis'


@deprecated('use `quat_rotAxis` instead')
def quat(xyz: TN) -> TN:
    """euler to quat
    Args:
        arr (TN): 输入张量/数组，shape为(...,3)，对应[roll, pitch, yaw]（弧度）
    Returns:
        quat: normalized [w,x,y,z], shape==(...,4)
    """
    if xyz.shape[-1] == 4:
        return xyz
    assert xyz.shape[-1] == 3, f"Last dimension should be 3, but found {xyz.shape}"
    lib = Lib(xyz)  # 自动检测库类型
    is_torch = lib.__name__ == 'torch'

    # 计算半角三角函数（支持广播）
    half_angles = 0.5 * xyz
    cos_half = lib.cos(half_angles)  # shape (...,3)
    sin_half = lib.sin(half_angles)

    # 分库处理维度解包
    if is_torch:
        cr, cp, cy = cos_half.unbind(dim=-1)
        sr, sp, sy = sin_half.unbind(dim=-1)
    else:  # NumPy处理
        cr, cp, cy = cos_half[..., 0], cos_half[..., 1], cos_half[..., 2]
        sr, sp, sy = sin_half[..., 0], sin_half[..., 1], sin_half[..., 2]

    # 并行计算四元数分量（保持维度）
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # 堆叠并归一化
    _quat = lib.stack([w, x, y, z], **{Axis(is_torch): -1})
    _quat /= Norm(_quat)
    return _quat


def euler(wxyz: TN) -> TN:
    """union quat to euler
    Args:
        quat (TN): [w,x,y,z], shape==(...,4)
    Returns:
        euler: [roll_x, pitch_y, yaw_z] in arc system, shape==(...,3)
    """
    if wxyz.shape[-1] == 3:
        return wxyz
    assert wxyz.shape[-1] == 4, f"Last dimension should be 4, but found {wxyz.shape}"
    lib = Lib(wxyz)  # 自动检测库类型
    is_torch = lib.__name__ == 'torch'
    EPSILON = 1e-12  # 数值稳定系数

    # 归一化四元数（防止输入未归一化）
    wxyz = wxyz / Norm(wxyz, dim=-1, keepdim=True)  # type: ignore

    # 解包四元数分量（支持广播）
    w, x, y, z = wxyz[..., 0], wxyz[..., 1], wxyz[..., 2], wxyz[..., 3]

    # 计算roll (x轴旋转)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = lib.arctan2(sinr_cosp, cosr_cosp + EPSILON)  # 防止除零

    # 计算pitch (y轴旋转)
    sinp = 2 * (w * y - z * x)
    pitch = lib.arcsin(sinp.clip(-1.0, 1.0))  # 限制在有效范围内

    # 计算yaw (z轴旋转)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = lib.arctan2(siny_cosp, cosy_cosp + EPSILON)

    # 堆叠结果
    _euler = lib.stack([roll, pitch, yaw], **{Axis(is_torch): -1})
    return _euler


def get_mod(mod1: ModuleType | str):
    if isinstance(mod1, str):
        _mod1 = sys.modules.get(mod1, None)
    else:
        _mod1 = mod1
    return _mod1


def Lib(arr, mod1: ModuleType | str = np, mod2: ModuleType | str = 'torch', ret_1_if=np.ndarray):
    """usage:
    ```python
    lib = Lib(arr)
    is_torch = lib.__name__ == 'torch'
    ```
    """
    _mod1 = get_mod(mod1)
    _mod2 = get_mod(mod2)
    if _mod1 and _mod2:
        mod = _mod1 if isinstance(arr, ret_1_if) else _mod2
    elif _mod1:
        mod = _mod1
    elif _mod2:
        mod = _mod2
    else:
        raise ImportError("Both libraries are not available.")
    # Log.debug(f"🔍 {mod.__name__}")
    return mod


def Norm(arr: TN, dim: int = -1, keepdim: bool = True) -> TN:
    """计算范数，支持批量输入"""
    lib = Lib(arr)
    is_torch = lib.__name__ == 'torch'
    if is_torch:
        return lib.norm(arr, dim=dim, keepdim=keepdim)
    else:
        return lib.linalg.norm(arr, axis=dim, keepdims=keepdim)


def skew_symmetric(v: TN) -> TN:
    """生成反对称矩阵，支持批量输入"""
    lib = Lib(v)
    is_torch = lib.__name__ == 'torch'
    axis = Axis(is_torch)
    axis_1 = {axis: -1}
    # 创建各分量
    zeros = lib.zeros_like(v[..., 0])  # 形状 (...)
    row0 = lib.stack([zeros, -v[..., 2], v[..., 1]], **axis_1)  # (...,3)
    row1 = lib.stack([v[..., 2], zeros, -v[..., 0]], **axis_1)
    row2 = lib.stack([-v[..., 1], v[..., 0], zeros], **axis_1)
    # 堆叠为矩阵
    if is_torch:
        return lib.stack([row0, row1, row2], dim=-2)
    else:
        return lib.stack([row0, row1, row2], axis=-2)  # (...,3,3)


def Rodrigues(rot_vec3: TN) -> TN:
    """
    支持批量处理的罗德里格斯公式

    Parameters
    ----------
    rotvec : np.ndarray
        3D rotation vector

    Returns
    -------
    np.ndarray
        3x3 rotation matrix

    _R: np.ndarray = np.eye(3) + sin * K + (1 - cos) * K @ K  # 原式
    choose (3,1) instead 3:    3 is vec, k.T == k;    (3,1) is matrix, k.T != k
    """
    if rot_vec3.shape[-1] == 4:
        return rot_vec3
    assert rot_vec3.shape[-1] == 3, f"Last dimension must be 3, but got {rot_vec3.shape}"
    lib = Lib(rot_vec3)
    is_torch = lib.__name__ == 'torch'

    # 计算旋转角度
    theta = Norm(rot_vec3, dim=-1, keepdim=True)  # (...,1)

    EPSILON = 1e-8
    mask = theta < EPSILON

    # 处理小角度情况
    K_small = skew_symmetric(rot_vec3)
    eye = lib.eye(3, dtype=rot_vec3.dtype)
    if is_torch:
        eye = eye.to(rot_vec3.device)
    R_small = eye + K_small  # 广播加法

    # 处理一般情况
    safe_theta = lib.where(mask, EPSILON * lib.ones_like(theta), theta)  # 避免除零
    k = rot_vec3 / safe_theta  # 单位向量

    K = skew_symmetric(k)
    k = k[..., None]  # 添加最后维度 (...,3,1)
    kkt = lib.matmul(k, lib.swapaxes(k, -1, -2))  # (...,3,3)

    cos_t = lib.cos(theta)[..., None]  # (...,1,1)
    sin_t = lib.sin(theta)[..., None]

    R_full = cos_t * eye + sin_t * K + (1 - cos_t) * kkt

    # 合并结果
    if is_torch:
        mask = mask.view(*mask.shape, 1, 1)
    else:
        mask = mask[..., None]

    ret = lib.where(mask, R_small, R_full)
    return ret


def RotMat_to_quat(R: TN) -> TN:
    """将3x3旋转矩阵转换为单位四元数 [w, x, y, z]，支持批量和PyTorch/NumPy"""
    if R.shape[-1] == 4:
        return R
    assert R.shape[-2:] == (3, 3), f"输入R的末两维必须为3x3，当前为{R.shape}"
    lib = Lib(R)  # 自动检测模块
    is_torch = lib.__name__ == 'torch'
    EPSILON = 1e-12  # 数值稳定系数

    # 计算迹，形状为(...)
    trace = lib.einsum('...ii->...', R)

    # 计算四个分量的平方（带数值稳定处理）
    q_sq = lib.stack([
        (trace + 1) / 4,
        (1 + 2 * R[..., 0, 0] - trace) / 4,
        (1 + 2 * R[..., 1, 1] - trace) / 4,
        (1 + 2 * R[..., 2, 2] - trace) / 4,
    ], axis=-1)

    q_sq = lib.maximum(q_sq, 0.0)  # 确保平方值非负

    # 找到最大分量的索引，形状(...)
    i = lib.argmax(q_sq, axis=-1)

    # 计算分母（带数值稳定处理）
    denoms = 4 * lib.sqrt(q_sq + EPSILON)  # 添加极小值防止sqrt(0)

    # 构造每个case的四元数分量
    cases = []
    for i_case in range(4):
        denom = denoms[..., i_case]  # 当前case的分母
        if i_case == 0:
            w = lib.sqrt(q_sq[..., 0] + EPSILON)  # 数值稳定
            x = (R[..., 2, 1] - R[..., 1, 2]) / denom
            y = (R[..., 0, 2] - R[..., 2, 0]) / denom
            z = (R[..., 1, 0] - R[..., 0, 1]) / denom
        elif i_case == 1:
            x = lib.sqrt(q_sq[..., 1] + EPSILON)
            w = (R[..., 2, 1] - R[..., 1, 2]) / denom
            y = (R[..., 0, 1] + R[..., 1, 0]) / denom
            z = (R[..., 0, 2] + R[..., 2, 0]) / denom
        elif i_case == 2:
            y = lib.sqrt(q_sq[..., 2] + EPSILON)
            w = (R[..., 0, 2] - R[..., 2, 0]) / denom
            x = (R[..., 0, 1] + R[..., 1, 0]) / denom
            z = (R[..., 1, 2] + R[..., 2, 1]) / denom
        else:  # i_case == 3
            z = lib.sqrt(q_sq[..., 3] + EPSILON)
            w = (R[..., 1, 0] - R[..., 0, 1]) / denom
            x = (R[..., 0, 2] + R[..., 2, 0]) / denom
            y = (R[..., 1, 2] + R[..., 2, 1]) / denom

        case = lib.stack([w, x, y, z], axis=-1)
        cases.append(case)

    # 合并所有情况并进行索引选择
    cases = lib.stack(cases, axis=0)
    if is_torch:
        index = i.reshape(1, *i.shape, 1).expand(1, *i.shape, 4)
        q = lib.gather(cases, dim=0, index=index).squeeze(0)
    else:
        # 构造NumPy兼容的索引
        index = i.reshape(1, *i.shape, 1)  # 添加新轴以对齐批量维度
        index = np.broadcast_to(index, (1,) + i.shape + (4,))  # 扩展至四元数维度
        q = np.take_along_axis(cases, index, axis=0).squeeze(0)  # 选择并压缩维度

    # 归一化处理（带数值稳定）
    norm = Norm(q, dim=-1, keepdim=True)
    ret = q / (norm + EPSILON)  # 防止除零
    return ret


def quat_rotAxis(arr: TN) -> TN: return RotMat_to_quat(Rodrigues(arr))


def apply_pose(
    action: 'bpy.types.Action',
    pose,
    frame: int,
    bones: Sequence[str],
    trans: Optional[Any] = None,
    bone_rot: TYPE_ROT = 'QUATERNION',
    **kwargs
):
    """Apply translation, pose, and shape to character using Action and F-Curves."""
    method = str(kwargs.get('quat', 0))[0]
    if bone_rot == 'QUATERNION':
        if method == 'a':  # axis
            rots = quat_rotAxis(pose)
        elif method == 'r':  # raw
            rots = pose
        else:
            rots = quat(pose)
    else:
        rots = euler(pose)

    start = 0
    if trans is not None:
        trans = Vector((trans[0], trans[1], trans[2]))
        add_keyframe(action, f'pose.bones["{bones[0]}"].location', frame, trans)
        start = 1

    # Insert rotation keyframes for each bone
    for i, rot in enumerate(rots, start=start):  # Skip root!
        bone_name = bones[i]
        if bone_rot == 'QUATERNION':
            add_keyframe(action, f'pose.bones["{bone_name}"].rotation_quaternion', frame, rot)
        else:
            add_keyframe(action, f'pose.bones["{bone_name}"].rotation_euler', frame, rot)
    return action


def register():
    ...


def unregister():
    ...


if __name__ == "__main__":
    # debug
    try:
        ...
    except ImportError:
        ...
