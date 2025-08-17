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
    # torso = np.array(armature.matrix_world @ get_bone_local_facing(armature.data.bones[BODY[1]]))
    # Log.debug(f'{BODY[1]=}\t{torso=}')

    if mapping not in ('smpl', 'smplx') and kwargs.get('offset', None):
        ...
        rotate = pelvis_rotate_offset(rotate, armature, BODY[1])
    rotate = rotate.reshape(-1, 1, rotate.shape[-1])
    pose = body_pose.value[Slice]
    pose = np.concatenate([rotate, pose], axis=1)  # (frames,1+22,3 or 4)
    with bpy_action(armature, name) as action:
        yield from pose_apply(armature=armature, action=action, pose=pose, transl=transl, bones=BODY, frame=data.begin + 1, transl_base=transl_base, **kwargs)

# TODO: read source code of rokoko addon:
# 1. copy To.bones → From.bones, new_bones in From use To.bones.name
# 2. set new_bones' parent to From.bones.parent correspondently
# 3. copy From.bones.head/tail/roll → From.new_bones.head/tail/roll
# 4. set constrain on To.bones, target is From.new_bones, use COPY_ROTATION (+ COPY_LOCATION if root bones)
# 5. bake To.bones


def pelvis_rotate_offset(rotate: np.ndarray, armature: bpy.types.Object, torso_name='torso'):
    '''Q_new_pose = Q_old_pose * Q_delta^-1'''
    if not isinstance(armature.data, bpy.types.Armature):
        raise TypeError(f'Expected armature data type, got {type(armature.data)}')
    Log.error(f'{torso_name=}')
    bone_torso = armature.data.bones[torso_name]
    facing_torso = np.array(bone_torso.tail - bone_torso.head)
    facing_torso = facing_torso / np.linalg.norm(facing_torso)
    facing_pelvis = np.array([0, 0, 1])
    Log.debug(f'{facing_torso=}\t{facing_pelvis=}')
    delta_q = delta_quat(facing_torso, facing_pelvis)
    Log.debug(f'{delta_q=}')
    delta_q_1 = quat_1(delta_q)
    for i in range(rotate.shape[0]):
        rotate[i] = multi_quat(rotate[i], delta_q_1)
    return rotate


def get_bone_offset(From: bpy.types.Object, To: bpy.types.Object):
    transforms = {}
    if not isinstance(From.data, bpy.types.Armature) or not isinstance(To.data, bpy.types.Armature):
        raise TypeError(f'Expected armature type, got {type(From.data)=} and {type(To.data)=}')
    for bone in From.data.bones:
        transforms[bone.name] = (
            From.matrix_world.inverted() @ bone.head,
            From.matrix_world.inverted() @ bone.tail,
            mat3_to_vec_roll(From.matrix_world.inverted().to_3x3() @ bone.matrix.to_3x3())
        )  # Head loc, tail loc, bone roll
    for item in self.retarget_bone_list:
        bone_source = armature_source.data.edit_bones.get(item.bone_name_source)
        # Recreate target bone
        bone_new = armature_source.data.edit_bones.new(item.bone_name_target + RETARGET_ID)
        bone_new.head, bone_new.tail, bone_new.roll = transforms[item.bone_name_target]
        bone_new.parent = bone_source


import math


def mat3_to_vec_roll(M):
    vecmat = vec_roll_to_mat3(M.col[1], 0)
    vecmatinv = vecmat.inverted()
    M_roll = vecmatinv @ M
    roll = math.atan2(M_roll[0][2], M_roll[2][2])
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
