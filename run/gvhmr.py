from ..lib import *


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


def gvhmr(
    data: MotionData,
    Range=[0, None],
    mapping: Optional[TYPE_MAPPING] = None,
    **kwargs
):
    """
    per person

    Args:
        data (MotionData, dict): mocap data
        Range (tuple, optional): Frames range. Defaults to (0, Max_frames).

    Example:
    ```python
    gvhmr(data('smplx', 'gvhmr', person=0))
    ```
    """
    data, armature, rot, BODY, Slice = check_before_run('gvhmr', 'BODY', data, Range, mapping)
    transl = data(prop='transl', coord='global').value[Slice]
    rotate = data(prop='global_orient', coord='global').value[Slice]
    rotate = rotate.reshape(-1, 1, rotate.shape[-1])
    pose = data(prop='body_pose', coord='global').value[Slice]
    pose = np.concatenate([rotate, pose], axis=1)  # (frames,22,3 or 4)

    with bpy_action(armature, ';'.join([data.who, data.run])) as action:
        pose_reset(action, BODY[:23], rot)
        pose_to_keyframes(action=action, pose=pose, transl=transl, transl_base=transl[Slice.start], bones=BODY[:23], rot=rot, **kwargs)
