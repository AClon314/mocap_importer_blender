"""
https://github.com/carlosedubarreto/CEB_4d_Humans/blob/main/four_d_humans_blender.py
"""
import os
from typing import Union, Literal
import bpy
import pickle
import gettext
import numpy as np
DIR_SELF = os.path.dirname(__file__)
DIR_I18N = './i18n'
TYPE_MAPPING = Literal['smpl', 'smplx']
high_from_floor = 1.5


def get_toml(filename='blender_manifest.toml'):
    import toml
    with open(os.path.join(DIR_SELF, filename)) as f:
        data = toml.load(f)
    return data


def i18n():
    """usage:
    ```python
    _ = i18n()
    print(_('hello'))
    ```"""
    remapping = {
        'zh_CN': 'zh_HANS',
        'zh_TW': 'zh_HANT',
    }
    try:
        locale = bpy.app.translations.locale
    except AttributeError:
        locale = os.getenv('LANG', 'zh_HANS').split('.')[0]  # .UTF-8
        locale = remapping.get(locale, locale)
        Log.warning(f'bpy.locale not found, use {locale} instead')
    Log.debug(f'locale: {locale}')

    # this is working
    os.environ['LANGUAGE'] = locale
    gettext.bindtextdomain(ID, DIR_I18N)
    gettext.textdomain(ID)

    # this is not working
    # translation = gettext.translation(
    #     ID, DIR_I18N, [locale]
    # )
    # translation.install()
    return gettext.gettext


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
ID = get_toml()['id']
_ = i18n()

try:
    from mathutils import Matrix, Vector, Quaternion, Euler
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


def Rodrigues(rotvec: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    rotvec : np.ndarray
        3D rotation vector

    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    # L1范数：Σ|xᵢ|
    # L2范数（欧式距离）：√ (Σ|xᵢ|²)
    theta = np.linalg.norm(rotvec)  # L2 norm
    k = rotvec
    if theta > 0.:
        k = (rotvec / theta).reshape(3)    # 旋转向量的单位向量
    cos = np.cos(theta)
    sin = np.sin(theta)
    K = np.asarray([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]],
        dtype=float,
    )
    # R: np.ndarray = cos * np.eye(3) + sin * K + (1 - cos) * k.dot(k.T)
    R: np.ndarray = np.eye(3) + sin * K + (1 - cos) * K.dot(K.T)
    return R


def rodrigues_to_body_shapes(pose: np.ndarray):
    """
    Parameters
    ----------
    pose : np.ndarray
        22x3 rotation vectors
    """
    pose = np.asarray(pose).reshape(22, 3)
    rot_matrix = [Rodrigues(rot) for rot in pose]
    body_shapes = np.concatenate([
        (mat_rot - np.eye(3)).ravel() for mat_rot in rot_matrix[1:]
    ])
    ret = (rot_matrix, body_shapes)
    log_array(rot_matrix, 'body_shapes')
    return ret


def get_global_pose(global_pose, armature, frame=None):

    armature.pose.bones[BODY[0]].rotation_quaternion.w = 0.0
    armature.pose.bones[BODY[0]].rotation_quaternion.x = -1.0

    bone = armature.pose.bones[BODY[1]]
    # if frame is not None:
    #     bone.keyframe_insert('rotation_quaternion', frame=frame)

    root_orig = armature.pose.bones[BODY[0]].rotation_quaternion
    mw_orig = armature.matrix_world.to_quaternion()
    pelvis_quat = Matrix(global_pose[0]).to_quaternion()

    bone.rotation_quaternion = pelvis_quat
    bone.keyframe_insert('rotation_quaternion', frame=frame)

    pelvis_applyied = armature.pose.bones[BODY[1]].rotation_quaternion
    bpy.context.view_layer.update()

    rot_world_orig = root_orig @ pelvis_applyied @ mw_orig  # pegar a rotacao em relacao ao mundo

    return rot_world_orig


def apply_pose(
    trans,
    body_pose,
    armature: 'bpy.types.Object',
    frame=0,
    **kwargs
):
    """apply trans pose and shape to character"""

    # transform pose into rotation matrices (for pose) and pose blendshapes
    #    if self.option in [2,3]: #para WHAM ou slahmr
    #        mrots, bsh = rodrigues2bshapes(body_pose)
    #    else: #para 4d humans
    #        mrots = body_pose
    mrots, bsh = rodrigues_to_body_shapes(body_pose)
    # mrots = body_pose

    trans = Vector((trans[0], trans[1] - high_from_floor, trans[2]))

    armature.pose.bones[BODY[1]].location = trans
    armature.pose.bones[BODY[1]].keyframe_insert('location', frame=frame)

    armature.pose.bones[BODY[0]].rotation_quaternion.w = 0.0
    armature.pose.bones[BODY[0]].rotation_quaternion.x = -1.0

    for i, rot in enumerate(mrots):
        # if i < 22:  # 因为我使用的模型没有手盖
        if i < kwargs.get('ibone', 22):
            # if i == i_bone:
            bone = armature.pose.bones[BODY[i]]
            log_array(rot, f'{i}_{BODY[i]}')
            bone.rotation_quaternion = Matrix(rot).to_quaternion()

            if frame is not None:
                bone.keyframe_insert('rotation_quaternion', frame=frame)


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
    elif obj.name.startswith('basicModel_'):
        mapping = 'smpl'
    bpy.context.view_layer.objects.active = obj
    return mapping


def dynamic_import(mapping: Union[TYPE_MAPPING, None] = None):
    global BODY
    if mapping is None:
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
    results = load_pickle(file)

    armature = bpy.context.active_object
    if armature is None or armature.type != 'ARMATURE':
        bpy.ops.scene.smplx_add_gender()    # type: ignore
    dynamic_import(kwargs.get('mapping', None))
    if armature is None:
        raise ValueError('No armature found')

    frames = len(results['smpl_params_global']['transl'])
    # shape = results[character]['betas'].tolist()
    for f in range(0, frames // 4):
        print(f'{ID}: {f}/{frames}\t{f/frames*100:.3f}%', end='\r')
        bpy.context.scene.frame_set(f)
        global_trans = results['smpl_params_global']['transl'][f]
        global_orient: np.ndarray = results['smpl_params_global']['global_orient'][f]
        body_pose: np.ndarray = results['smpl_params_global']['body_pose'][f]
        body_pose = body_pose.reshape(int(len(body_pose) / 3), 3)   # (21,3)
        body_pose = np.vstack([global_orient, body_pose])  # (22,3)
        apply_pose(global_trans, body_pose, armature, f, **kwargs)
        bpy.context.view_layer.update()

    armature.pose.bones[BODY[0]].rotation_quaternion.w = 1.0
    armature.pose.bones[BODY[0]].rotation_quaternion.x = 0.0
    armature.pose.bones[BODY[0]].rotation_quaternion.y = 0.0
    armature.pose.bones[BODY[0]].rotation_quaternion.z = 0.0

    armature.pose.bones[BODY[1]].constraints.new('COPY_LOCATION')
    # armature.pose.bones[BODY[1]].constraints["Copy Location"].target = armature_ref
    armature.pose.bones[BODY[1]].constraints[0].target = armature
    armature.pose.bones[BODY[1]].constraints[0].subtarget = BODY[1]
    # armature.pose.bones[BODY[1]].constraints["Copy Location"].subtarget = BODY[1]

    armature.pose.bones[BODY[1]].constraints.new('COPY_ROTATION')
    # armature.pose.bones[BODY[1]].constraints["Copy Rotation"].target = armature_ref
    armature.pose.bones[BODY[1]].constraints[1].target = armature
    # armature.pose.bones[BODY[1]].constraints["Copy Rotation"].subtarget = BODY[1]
    armature.pose.bones[BODY[1]].constraints[1].subtarget = BODY[1]
    Log.info(f'done')


def register():
    global _
    _ = i18n()


def unregister():
    ...


if __name__ == "__main__":
    # debug
    try:
        from mapping.smplx import *
        ret = keys_BFS(BONES)
        print(ret)
    except ImportError:
        ...
