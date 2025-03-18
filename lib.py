"""
https://github.com/carlosedubarreto/CEB_4d_Humans/blob/main/four_d_humans_blender.py
"""
import os
from types import ModuleType
import bpy
import pickle
import importlib
import numpy as np
from typing import List, Union, Literal
DIR_SELF = os.path.dirname(__file__)
DIR_MAPPING = os.path.join(DIR_SELF, 'mapping')
TYPE_MAPPING = Literal['smpl', 'smplx']
TYPE_I18N = Literal[
    'ca_AD', 'en_US', 'es', 'fr_FR', 'ja_JP', 'sk_SK', 'ur', 'vi_VN', 'zh_HANS',
    'de_DE', 'it_IT', 'ka', 'ko_KR', 'pt_BR', 'pt_PT', 'ru_RU', 'sw', 'ta', 'tr_TR', 'uk_UA', 'zh_HANT',
    'ab', 'ar_EG', 'be', 'bg_BG', 'cs_CZ', 'da', 'el_GR', 'eo', 'eu_EU', 'fa_IR', 'fi_FI', 'ha', 'he_IL', 'hi_IN', 'hr_HR', 'hu_HU', 'id_ID', 'km', 'ky_KG', 'lt', 'ne_NP', 'nl_NL', 'pl_PL', 'ro_RO', 'sl', 'sr_RS', 'sr_RS@latin', 'sv_SE', 'th_TH'
]
high_from_floor = 1.5


def import_mapping():
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


def mapping_items():
    items: List[tuple[str, str, str]] = [(
        'Auto', 'Auto',
        'Auto detect armature type, based on name (will enhanced in later version)')]
    for k, m in MODS.items():
        items.append((k, k, m.HELP.get(bpy.app.translations.locale, f'Error: No HELP for {k}')))
    return items


def get_logger(name=__name__, level=10):
    """```python
    Log = get_logger(__name__)
    ```"""
    import logging
    Log = logging.getLogger(__name__)
    Log.setLevel(level)
    Log.propagate = False   # disable propagate to root logger
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(
            '%(levelname)s\t%(asctime)s  💬 %(message)s\t%(name)s:%(lineno)d',
            datefmt='%H:%M:%S'))
    Log.addHandler(stream_handler)
    return Log


Log = get_logger(__name__)
ID = __package__.split('.')[-1] if __package__ else __name__
MODS = import_mapping()

try:
    from mathutils import Matrix, Vector, Quaternion, Euler
    from .gvhmr import gvhmr
except ImportError as e:
    Log.warning(e)


def load_pickle(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)


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


def bone_to_dict(bone, deep=0, deep_max=1000):
    if deep_max and deep > deep_max:
        raise ValueError(f'Bones tree too deep, {deep} > {deep_max}')
    return {child.name: bone_to_dict(child, deep + 1) for child in bone.children}


def dump_bones(armature):
    """将骨架转换为字典"""
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


def guess_obj_mapping(obj: 'bpy.types.Object') -> Union[TYPE_MAPPING, None]:
    mapping: Union[TYPE_MAPPING, None] = None
    if obj.name.startswith('SMPLX-'):
        mapping = 'smplx'
    elif obj.name.startswith('Armature'):
        mapping = 'smpl'
    bpy.context.view_layer.objects.active = obj
    return mapping


def dynamic_import(mapping: Union[TYPE_MAPPING, None] = None):
    global BODY
    if mapping is None:
        # guess mapping
        active = bpy.context.active_object
        if active:
            mapping = guess_obj_mapping(active)
        else:
            for obj in bpy.data.objects:
                mapping = guess_obj_mapping(obj)
                if mapping:
                    break

    if mapping == 'smpl':
        from .mapping.smpl import BODY
    elif mapping == 'smplx':
        from .mapping.smplx import BODY
    else:
        raise ValueError(f'Unknown mapping: {mapping}')
    return BODY


def main(file, **kwargs):
    gvhmr(file, **kwargs)


if __name__ == "__main__":
    # debug
    try:
        from mapping.smplx import *
        ret = keys_BFS(BONES_TREE)
        print(ret)
    except ImportError:
        ...
