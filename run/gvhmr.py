import bpy
from ..lib import *
from ..b import *
# TODO: retarget 1.相对于第0帧旋转了多少(其余fk→ik, COPY_LOCATION only, 要求mesh为T-pose)   2.copy_from相对于copy_to多旋转了多少，copy_to就少旋转一些(如，rokoko可以使rest pose的不同A-pose，转换后到达唯一的位置，不要求mesh为T-pose)


def gvhmr(
    armature: bpy.types.Object,
    data: MotionData,
    mapping: TYPE_MAPPING | None = None,
    Slice=slice(0, None),
    base_frame=0,
    **kwargs
):
    """
    per person

    Args:
        data (MotionData, dict): mocap data
        Slice (tuple, optional): Frames range. Defaults to (0, Max_frames).

    Example:
    ```python
    gvhmr(data('smplx', 'gvhmr', person=0))
    ```
    """
    data, mapping, name, Slice, transl, rotate = data_mapping_name_Slice_transl_rotate(armature, data, mapping, Slice, run='gvhmr')
    transl_base = None if transl is None else transl[base_frame]
    body_pose = data('body_pose')
    if not body_pose:  # for cam@
        objs = bpy.context.selected_objects
        cam = objs[0] if objs else None
        if not cam:
            cam = bpy.ops.object.camera_add(location=(0, 0, 0))
            cam = bpy.context.object
            if not cam:
                raise RuntimeError('No active object and failed to add camera')
        with bpy_action(cam, name) as action:
            yield from transform_apply(obj=cam, action=action, rotate=rotate, transl=transl)
        return

    BODY = get_BONES(mapping=mapping, key='BODY')
    if not isinstance(armature.data, bpy.types.Armature):
        raise TypeError(f'Expected armature data type, got {type(armature.data)}')
    torso = np.array(armature.matrix_world @ get_bone_local_facing(armature.data.bones[BODY[1]]))
    Log.debug(f'{BODY[1]=}\t{torso=}')

    # rotate[0] = [1, 0, 0, 0]   # no rotation

    if mapping not in ('smpl', 'smplx') and kwargs.get('offset', None):
        rotate = pelvis_rotate_offset(rotate, torso)
    rotate = rotate.reshape(-1, 1, rotate.shape[-1])
    pose = body_pose.value[Slice]
    pose = np.concatenate([rotate, pose], axis=1)  # (frames,1+22,3 or 4)
    with bpy_action(armature, name) as action:
        yield from pose_apply(armature=armature, action=action, pose=pose, transl=transl, bones=BODY, frame=data.begin + 1, transl_base=transl_base, **kwargs)


def pelvis_rotate_offset(rotate, pelvis_before):
    # TODO: read source code of rokoko addon
    pelvis_before = pelvis_before / np.linalg.norm(pelvis_before)
    pelvis_after = np.array([0, 0, 1])  # SMPL rot axis, normalized
    rot_axis = np.cross(pelvis_before, pelvis_after)
    rot_axis_norm = np.linalg.norm(rot_axis)
    Log.debug(f'{rot_axis=}\t{rot_axis_norm=}')
    # 计算需要应用的旋转四元数来抵消偏移
    if rot_axis_norm > EPSILON:  # 避免除零错误
        rot_axis = rot_axis / rot_axis_norm
        angle = np.arccos(np.clip(np.dot(pelvis_before, pelvis_after), -1.0, 1.0))
        # 构建四元数(w, x, y, z)
        half_angle = angle / 2.0
        s = np.sin(half_angle)
        delta_rot = np.array([np.cos(half_angle), rot_axis[0] * s, rot_axis[1] * s, rot_axis[2] * s])
        Log.debug(f'{delta_rot=}\t{angle=}')
    else:
        # 如果不需要旋转或180度旋转
        if np.dot(pelvis_before, pelvis_after) > 0:
            # 已经对齐，单位四元数
            delta_rot = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            # 相反方向，绕x轴旋转180度
            delta_rot = np.array([0.0, 1.0, 0.0, 0.0])

    is_3 = False
    if rotate.shape[-1] == 3:
        is_3 = True
        rotate = quat(rotate)
    if rotate.shape[-1] == 4:
        # 将delta_rot扩展为与rotate相同的形状
        delta_rot_expanded = np.tile(delta_rot, (rotate.shape[0], 1))
        Log.debug(f'before{euler(rotate[0])=}')
        rotate = quat_multiply(delta_rot_expanded, rotate)
        Log.debug(f'after {euler(rotate[0])=}')
    if is_3:
        rotate = euler(rotate)
    return rotate


def get_bone_offset(From: bpy.types.Object, To: bpy.types.Object):
    transforms = {}
    if not isinstance(From.data, bpy.types.Armature) or not isinstance(To.data, bpy.types.Armature):
        raise TypeError(f'Expected armature type, got {type(From.data)=} and {type(To.data)=}')
    for bone in From.data.bones:
        transforms[bone.name] = (
            From.matrix_world.inverted() @ bone.head.copy(),
            From.matrix_world.inverted() @ bone.tail.copy(),
            mat3_to_vec_roll(From.matrix_world.inverted().to_3x3() @ bone.matrix.to_3x3())
        )  # Head loc, tail loc, bone roll
    return transforms


import math


def mat3_to_vec_roll(mat):
    vecmat = vec_roll_to_mat3(mat.col[1], 0)
    vecmatinv = vecmat.inverted()
    rollmat = vecmatinv @ mat
    roll = math.atan2(rollmat[0][2], rollmat[2][2])
    return roll


def vec_roll_to_mat3(vec, roll):
    target = Vector((0, 0.1, 0))
    nor = vec.normalized()
    axis = target.cross(nor)
    if axis.dot(axis) > 0.0000000001:
        axis.normalize()
        theta = target.angle(nor)
        bMatrix = Matrix.Rotation(theta, 3, axis)
    else:
        updown = 1 if target.dot(nor) > 0 else -1
        bMatrix = Matrix.Scale(updown, 3)
        bMatrix[2][2] = 1.0

    rMatrix = Matrix.Rotation(roll, 3, nor)
    mat = rMatrix @ bMatrix
    return mat
