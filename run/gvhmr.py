from ..lib import *
Log = get_logger(__name__)


def Rodrigues(rot_vec3: np.ndarray) -> np.ndarray:
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
    theta = np.linalg.norm(rot_vec3)  # L2 norm
    k = rot_vec3
    if theta > 0.:
        k = (rot_vec3 / theta).reshape(3, 1)    # 旋转向量的单位向量, TODO why (3,1) instead 3 ?
    cos = np.cos(theta)
    sin = np.sin(theta)
    K = np.asarray([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]],
        dtype=object,
    )
    R: np.ndarray = cos * np.eye(3) + sin * K + (1 - cos) * k.dot(k.T)
    # R: np.ndarray = np.eye(3) + sin * K + (1 - cos) * K.dot(K.T)
    return R


@deprecated('draft')
def rodrigues_to_rotates(pose: np.ndarray):
    """
    Parameters
    ----------
    pose : np.ndarray
        22x3 rotation vectors
    """
    # pose = np.asarray(pose).reshape(22, 3)
    rot_matrixs = [Rodrigues(rot) for rot in pose]
    # body_shapes = np.concatenate([
    #     (rotM - np.eye(3)).ravel() for rotM in rot_matrixs[1:]
    # ])
    # ret = (rot_matrixs, body_shapes)
    # log_array(rot_matrixs, 'body_shapes')
    return rot_matrixs


@deprecated('draft')
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
    action: 'bpy.types.Action',
    pose,
    trans,
    frame: int,
    bone_rot: TYPE_ROT = 'QUATERNION',
    **kwargs
):
    """Apply translation, pose, and shape to character using Action and F-Curves."""
    if bone_rot == 'QUATERNION':
        rots = [Rodrigues(rot) for rot in pose]
    else:
        rots = pose

    trans = Vector((trans[0], trans[1], trans[2]))

    # Insert translation keyframe
    keyframe_add(action, f'pose.bones["{BODY[0]}"].location', frame, trans)

    # Insert rotation keyframes for each bone
    for i, rot in enumerate(rots, start=1):  # Skip root!
        if i <= kwargs.get('ibone', 22):
            bone_name = BODY[i]
            if bone_rot == 'QUATERNION':
                rot = Matrix(rot).to_quaternion()  # type: ignore
                keyframe_add(action, f'pose.bones["{bone_name}"].rotation_quaternion', frame, rot)
            else:
                keyframe_add(action, f'pose.bones["{bone_name}"].rotation_euler', frame, rot)
    return action


def gvhmr(
    data: MotionData,
    range_frame=(0, None),
    mapping: Optional[TYPE_MAPPING] = None,
    **kwargs
):
    """
    per person

    Args:
        data (MotionData, dict): mocap data
        frame_range (tuple, optional): Frames range. Defaults to (0, Max_frames).

    Example:
    ```python
    gvhmr(data('smplx', 'gvhmr', person=0))
    ```
    """
    is_range = len(range_frame) > 1

    armature = bpy.context.active_object
    if armature is None or armature.type != 'ARMATURE':
        raise ValueError('No armature found')
    bones_rots: list[TYPE_ROT] = [b.rotation_mode for b in armature.pose.bones]
    bone_rot = get_major(bones_rots)
    bone_rot = 'QUATERNION' if not bone_rot else bone_rot

    mapping = None if mapping == 'auto' else mapping
    mapping = get_mapping_from_selected_or_objs(mapping)
    global BODY
    BODY = Mod()[mapping].BODY   # type:ignore

    data = data(mapping=data.mapping, run='gvhmr')  # type: ignore
    translation = data(prop='transl', coord='global').value
    rotate = data(prop='global_orient', coord='global').value
    rotate = rotate.reshape(-1, 1, 3)
    pose = data(prop='body_pose', coord='global').value
    pose = pose.reshape(-1, len(pose[0]) // 3, 3)   # (frames,21,3)
    pose = np.concatenate([rotate, pose], axis=1)  # (frames,22,3)

    range_frame = list(range_frame)
    if is_range and range_frame[1] is None:
        range_frame[1] = len(translation)  # type: ignore
        Log.info(f'range_frame[1] fallback to {range_frame[1]}')
    Range = range(*range_frame)
    # shape = results[character]['betas'].tolist()
    with new_action(armature, ';'.join([data.who, data.run_keyname, data.mapping])) as action:
        for f in Range:
            print(f'gvhmr {ID}: {f}/{range_frame[1]}\t{f / range_frame[1] * 100:.3f}%', end='\r')
            apply_pose(action, pose[f], translation[f], f + 1, bone_rot=bone_rot, **kwargs)

    Log.info(f'done')
    del data
