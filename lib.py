"""
https://github.com/carlosedubarreto/CEB_4d_Humans/blob/main/four_d_humans_blender.py
"""
from typing import Generator, Union
import bpy
import pickle
import numpy as np
try:
    from mathutils import Matrix, Vector, Quaternion, Euler
    from .mapping.smplx import *
except ImportError:
    from mapping.smplx import *

high_from_floor = 1.5


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


def load_pickle(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)


def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]], dtype=object)  # adicionei "",dtype=object" por que estava dando erro
    return (cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat)


def rodrigues_to_body_shapes(pose):

    rod_rots = np.asarray(pose).reshape(22, 3)
    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                              for mat_rot in mat_rots[1:]])
    return (mat_rots, bshapes)


def get_global_pose(global_pose, arm_ob, frame=None):

    arm_ob.pose.bones['m_avg_root'].rotation_quaternion.w = 0.0
    arm_ob.pose.bones['m_avg_root'].rotation_quaternion.x = -1.0

    bone = arm_ob.pose.bones['m_avg_Pelvis']
    # if frame is not None:
    #     bone.keyframe_insert('rotation_quaternion', frame=frame)

    root_orig = arm_ob.pose.bones['m_avg_root'].rotation_quaternion
    mw_orig = arm_ob.matrix_world.to_quaternion()
    pelvis_quat = Matrix(global_pose[0]).to_quaternion()

    bone.rotation_quaternion = pelvis_quat
    bone.keyframe_insert('rotation_quaternion', frame=frame)

    pelvis_applyied = arm_ob.pose.bones['m_avg_Pelvis'].rotation_quaternion
    bpy.context.view_layer.update()

    rot_world_orig = root_orig @ pelvis_applyied @ mw_orig  # pegar a rotacao em relacao ao mundo

    return rot_world_orig


def apply_trans_pose_shape(trans, body_pose, armature, frame=None):
    """apply trans pose and shape to character"""

    # transform pose into rotation matrices (for pose) and pose blendshapes
    #    if self.option in [2,3]: #para WHAM ou slahmr
    #        mrots, bsh = rodrigues2bshapes(body_pose)
    #    else: #para 4d humans
    #        mrots = body_pose
    mrots, bsh = rodrigues_to_body_shapes(body_pose)
#    mrots = body_pose

#    trans = Vector((trans[0],trans[1]-2.2,trans[2]))
    trans = Vector((trans[0], trans[1] - high_from_floor, trans[2]))

    # print('frame in apply pose:', frame)
    armature.pose.bones['pelvis'].location = trans
    armature.pose.bones['pelvis'].keyframe_insert('location', frame=frame)

    armature.pose.bones['root'].rotation_quaternion.w = 0.0
    armature.pose.bones['root'].rotation_quaternion.x = -1.0

    for ibone, mrot in enumerate(mrots):
        if ibone < 22:  # 因为我使用的模型没有手盖
            bone = armature.pose.bones[part_bones['bone_%02d' % ibone]]
            bone.rotation_quaternion = Matrix(mrot).to_quaternion()

            if frame is not None:
                bone.keyframe_insert('rotation_quaternion', frame=frame)


def bone_to_dict(bone, deep=0, deep_max=1000):
    if deep_max and deep > deep_max:
        raise ValueError(f'Bones tree too deep, {deep} > {deep_max}')
    return {child.name: bone_to_dict(child, deep + 1) for child in bone.children}


def armature_to_dict(armature):
    """将骨架转换为字典"""
    if armature and armature.type == 'ARMATURE':
        for bone in armature.pose.bones:
            if not bone.parent:
                return {bone.name: bone_to_dict(bone)}
    return {}


def dump_bones(armature=None):
    """打印当前活动骨架的骨骼树状信息（JSON格式）"""
    import json
    if armature is None:
        armature = bpy.context.active_object
    armature_dict = armature_to_dict(armature)
    print(json.dumps(armature_dict, indent=2))


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


def main(file):
    results = load_pickle(file)

    armature = bpy.context.active_object
    if armature is None or armature.type != 'ARMATURE':
        bpy.ops.scene.smplx_add_gender()    # type: ignore
    Log.info(f'armature: {armature}')

    frames = len(results['smpl_params_global']['transl'])
    Log.info(f'frames: {frames}')
    # shape = results[character]['betas'].tolist()
    for f in range(0, frames):
        bpy.context.scene.frame_set(f)
        trans = results['smpl_params_global']['transl'][f]
        global_orient = results['smpl_params_global']['global_orient'][f]
        body_pose = results['smpl_params_global']['body_pose'][f]
        body_pose_fim = body_pose.reshape(int(len(body_pose) / 3), 3)
        final_body_pose = np.vstack([global_orient, body_pose_fim])
        apply_trans_pose_shape(Vector(trans), final_body_pose, armature, f)
        bpy.context.view_layer.update()


if __name__ == "__main__":
    ret = keys_BFS(SMPLX_DICT)  # type: ignore
    print(ret)
