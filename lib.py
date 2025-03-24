"""
https://github.com/carlosedubarreto/CEB_4d_Humans/blob/main/four_d_humans_blender.py
"""
from logging import warn
import os
import re
import bpy
import importlib
import numpy as np
from contextlib import contextmanager
from types import ModuleType
from typing import Dict, List, Optional, Sequence, Union, Literal, TypeVar, get_args
DIR_SELF = os.path.dirname(__file__)
DIR_MAPPING = os.path.join(DIR_SELF, 'mapping')
MAPPING_TEMPLATE = os.path.join(DIR_MAPPING, 'template.pyi')
TYPE_MAPPING = Literal['smpl', 'smplx']
TYPE_RUN = Literal['gvhmr', 'wilor']
TYPE_I18N = Literal[
    'ca_AD', 'en_US', 'es', 'fr_FR', 'ja_JP', 'sk_SK', 'ur', 'vi_VN', 'zh_HANS',
    'de_DE', 'it_IT', 'ka', 'ko_KR', 'pt_BR', 'pt_PT', 'ru_RU', 'sw', 'ta', 'tr_TR', 'uk_UA', 'zh_HANT',
    'ab', 'ar_EG', 'be', 'bg_BG', 'cs_CZ', 'da', 'el_GR', 'eo', 'eu_EU', 'fa_IR', 'fi_FI', 'ha', 'he_IL', 'hi_IN', 'hr_HR', 'hu_HU', 'id_ID', 'km', 'ky_KG', 'lt', 'ne_NP', 'nl_NL', 'pl_PL', 'ro_RO', 'sl', 'sr_RS', 'sr_RS@latin', 'sv_SE', 'th_TH'
]
T = TypeVar('T')
MOTION_DATA = None


def get_logger(name=__name__, level=10):
    """```python
    Log = get_logger(__name__)
    ```"""
    import logging
    Log = logging.getLogger(name)
    Log.setLevel(level)
    Log.propagate = False   # disable propagate to root logger
    stream_handler = logging.StreamHandler()

    # 添加日志级别前缀
    level_prefix = {
        logging.DEBUG: '🐛 DEBUG',
        logging.INFO: '💬 INFO',
        logging.WARNING: '⚠️ WARNING',
        logging.ERROR: '❌ ERROR',
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_prefix.get(record.levelno, record.levelname)
            return super().format(record)

    stream_handler.setFormatter(
        CustomFormatter(
            '%(levelname)s\t%(asctime)s  %(message)s\t%(name)s:%(lineno)d',
            datefmt='%H:%M:%S'))
    Log.addHandler(stream_handler)
    return Log


Log = get_logger(__name__)
ID = __package__.split('.')[-1] if __package__ else __name__


try:
    from mathutils import Matrix, Vector, Quaternion, Euler
except ImportError as e:
    Log.warning(e)

TYPE_MotionData_KEY = Literal['pose', 'rotate', 'shape', 'trans']
MotionData_KEY = get_args(TYPE_MotionData_KEY)


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


def Mod():
    files = os.listdir(DIR_MAPPING)
    pys = []
    mods: Dict[TYPE_MAPPING, ModuleType] = {}
    for f in files:
        if f.endswith('.py') and f not in ['template.py', '__init__.py']:
            pys.append(f[:-3])
    for p in pys:
        mod = importlib.import_module(f'.mapping.{p}', package=__package__)
        mods.update({p: mod})
    return mods


def items_mapping(self=None, context=None):
    items: List[tuple[str, str, str]] = [(
        'auto', 'Auto',
        'Auto detect armature type, based on name (will enhanced in later version)')]
    for k, m in Mod().items():
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


def keyframe_add(
    action: 'bpy.types.Action',
    data_path: str,
    at_frame: int,
    vector: Union[Sequence, Vector],
):
    """
    Usage:
    ```python
    insert_frame(action, f'pose.bones["{BODY[i]}"].rotation_quaternion', frame, quat)
    ```
    """
    kw = dict(data_path=data_path, index=0)
    for i, x in enumerate(vector):
        kw['index'] = i
        fcurve = action.fcurves.find(**kw)
        if not fcurve:
            fcurve = action.fcurves.new(**kw)
        fcurve.keyframe_points.insert(at_frame, value=x)
    return action


def context_override(area='NLA_EDITOR'):
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
        'object': bpy.context.object,
    }
    Log.debug(f'Context override: {override}')
    return override


@contextmanager
def new_action(
    obj: Optional['bpy.types.Object'] = None,
    name='Action'
):
    """Create a new action for object, and restore the old action after context"""
    old_action = None
    try:
        if not obj:
            obj = bpy.context.active_object
        if not obj:
            raise ValueError('No object found')
        if not obj.animation_data:
            obj.animation_data_create()
        if obj.animation_data.action:
            old_action = obj.animation_data.action
        action = obj.animation_data.action = bpy.data.actions.new(name=name)
        yield action
    finally:
        if old_action and obj and obj.animation_data and obj.animation_data.action:
            obj.animation_data.action = old_action
        else:
            Log.warning('No action to restore')


class MotionData(dict):
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

        usage:
        ```python
        data(mapping='smplx', run='gvhmr', key='trans', coord='global').values()[0]
        ```
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
        key: Union[TYPE_MotionData_KEY, str, None] = None,
        who: Union[str, int, None] = None,
        coord: Optional[Literal['global', 'incam']] = None,
    ):
        # Log.debug(f'self.__dict__={self.__dict__}')
        D = MotionData(npz=self.npz, lazy=True)
        if isinstance(who, int):
            who = f'person{who}'

        for k, v in self.items():
            is_in = [skip_or_in(args, k) for args in [mapping, run, key, who, coord]]
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
    def keys_custom(self):
        """could return `[your_customkeys, 'pose', 'rotate', 'trans', 'shape']`"""
        return self.distinct(2)

    @property
    def whos(self): return self.distinct(3)

    @property
    def coords(self): return self.distinct(4)

    @property
    def mapping(self): return warn_or_return_first(self.mappings)
    @property
    def run_keyname(self): return warn_or_return_first(self.runs_keyname)

    @property
    def key_custom(self): return warn_or_return_first(self.keys_custom)

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
    def key(self):
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
    for map, mod in Mod().items():
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
            raise ValueError(f'Unknown mapping: {mapping}')
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
    for x, my in zip(Mod()['smplx'].BONES, bones, strict=False):
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
    file = os.path.join(DIR_MAPPING, filename)
    if os.path.exists(file):
        raise FileExistsError(f'Mapping exists: {file}')
    else:
        with open(file, 'w') as f:
            f.write(t)
    Log.info(f'Restart addon to update mapping:  {file}')


def apply_motion(person: Union[str, int], mapping: Optional[TYPE_MAPPING], **kwargs):
    from .gvhmr import gvhmr
    if MOTION_DATA is None:
        raise ValueError('Failed to load motion data')
    gvhmr(MOTION_DATA('smplx', 'gvhmr', who=person), mapping=mapping, **kwargs)


def register():
    ...


def unregister():
    ...


if __name__ == "__main__":
    # debug
    try:
        data = MotionData(npz='/home/n/document/code/GVHMR/output/demo/jumper/mocap_jumper.npz')
        print(data(mapping='smplx', run='gvhmr', key='trans', coord='global').values()[0])
        # from mapping.smplx import *
        # ret = keys_BFS(BONES_TREE)
        # print(ret)
    except ImportError:
        ...
