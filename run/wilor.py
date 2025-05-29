from ..lib import *
from ..b import check_before_run, get_bones_global_rotation, pose_apply, bpy_action


def rotMat_relative(rotMats: 'np.ndarray', base_frame=0):
    """
    重新计算相对于指定帧的相对旋转

    Args:
        rotMats (np.array): 旋转矩阵数组，形状为 (frames, 3, 3)
        base_frame (int): 作为零旋转参考的帧索引

    Returns:
        np.array: 相对于参考帧的旋转矩阵数组
    """
    zero_rot = rotMats[base_frame]
    zero_rot_inv = np.linalg.inv(zero_rot)
    results = np.zeros_like(rotMats)
    for i in range(rotMats.shape[0]):
        results[i] = zero_rot_inv @ rotMats[i]
    return results


def mano_to_smplx(
    body_pose: 'np.ndarray',
    hand_pose: 'np.ndarray',
    global_orient: 'np.ndarray',
    is_left=False,
    base_frame=0,
):
    """
    hand to body pose:
    https://github.com/VincentHu19/Mano2Smpl-X/blob/main/mano2smplx.py

    Args:
        - body_pose (np.array): SMPLX's local pose (22, 3 or 4)
        - hand_pose (np.array): MANO's local pose (15, 3 or 4)
        - global_orient (np.array): **hand**'s global orientation (?, 3 or 4)
        - is_left (bool): whether the hand is left or right
        - base_frame (int): the frame to use as a reference for relative rotation

    ---
    Returns
    ---
        - wrist_pose (np.array): SMPLX's local pose (3 or 4), re-calculate based on `body_write` & hans's `global_orient`
        - hand_pose (np.array): SMPLX's local pose (15, 3 or 4), mirrored if is_left
    """
    Log.debug(f'{body_pose.shape=}')
    Log.debug(f'{global_orient.shape=}')

    M = np.diag([-1, 1, 1])  # Preparing for the left hand switch
    lib = Lib(body_pose)
    is_torch = lib.__name__ == 'torch'
    global_orient = rotMat(global_orient)
    if is_left:
        idx_elbow = 19  # +1 offset due to root bone
        global_orient = M @ global_orient @ M  # mirror switch
        hand_pose = rotMat(hand_pose)
        hand_pose = M @ hand_pose @ M
        hand_pose = rotMat_to_quat(hand_pose)
        Log.debug(f'{hand_pose.shape=}')
    else:
        idx_elbow = 20
    body_elbow = body_pose[:, idx_elbow]
    body_elbow = rotMat(body_elbow)

    if body_elbow.shape[0] < global_orient.shape[0]:
        # body < hand, padding body from the last frame
        pad_size = global_orient.shape[0] - body_elbow.shape[0]
        body_elbow = np.pad(body_elbow, ((0, pad_size), (0, 0), (0, 0)), mode='edge')
    else:
        # body > hand, clip body
        body_elbow = body_elbow[:global_orient.shape[0]]

    wrist_pose = np.linalg.inv(body_elbow) @ global_orient
    wrist_pose = rotMat_relative(wrist_pose, base_frame=base_frame)
    wrist_pose = rotMat_to_quat(wrist_pose)
    Log.debug(f'{wrist_pose.shape=}')
    return wrist_pose, hand_pose


def wrist_hand(wrist: 'np.ndarray', hand: 'np.ndarray', BODY: list[str], HAND: list[str], is_left=False):
    if hand.shape[-1] == 3:
        wrist = euler(wrist)
    elif hand.shape[-1] == 4:
        wrist = quat(wrist)
    wrist = np.expand_dims(wrist, axis=1)  # (frames,1,3|4)
    pose = np.concatenate([wrist, hand], axis=1)  # (frames,16,3|4)
    prefix = 'left_' if is_left else 'right_'
    name_wrist: str = BODY[21] if is_left else BODY[22]  # +1 offset
    bones_names = [name_wrist] + [prefix + bone for bone in HAND]
    Log.debug(f'{pose=}')
    return pose, bones_names


def wilor(
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
        Range (tuple, optional): Frames range. Defaults to (0, Max_frames).

    Example:
    ```python
    wilor(data('smplx', 'wilor', person=0))
    ```
    """
    data, HAND, armature, Slice = check_before_run(data, 'HANDS', 'wilor', mapping, Slice)

    BODY: list[str] = getattr(Map()['smplx'], 'BODY')   # type:ignore
    body_pose = get_bones_global_rotation(armature=armature, bone_resort=BODY, Slice=Slice)
    is_left = True if 'L' in data.who else False

    hand_pose = data('hand_pose').value[Slice]
    rotate = data('global_orient').value[Slice]

    wrist_rotate, hand_pose = mano_to_smplx(
        body_pose, hand_pose, rotate,
        is_left=is_left, base_frame=base_frame)
    hand_pose, HAND = wrist_hand(wrist_rotate, hand_pose, BODY, HAND, is_left=is_left)

    with bpy_action(armature, ';'.join([data.who, data.run])) as action:
        pose_apply(armature=armature, action=action, pose=hand_pose, bones=HAND, **kwargs)
