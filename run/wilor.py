from ..lib import *
from ..b import check_before_run, get_bones_global_rotation, pose_apply, bpy_action


def mano_to_smplx(
    body_pose: 'np.ndarray',
    hand_pose: 'np.ndarray',
    global_orient: 'np.ndarray',
    is_left=False,
    **kwargs,
):
    """
    hand to body pose:
    https://github.com/VincentHu19/Mano2Smpl-X/blob/main/mano2smplx.py

    Args:
        - body_pose (np.array): SMPLX's local pose (22, 3 or 4)
        - hand_pose (np.array): MANO's local pose (15, 3 or 4)
        - global_orient (np.array): **hand**'s global orientation (?, 3 or 4)
        - is_left (bool): whether the hand is left or right

    ---
    Returns
    ---
        - wrist_pose (np.array): SMPLX's local pose (3 or 4), re-calculate based on `body_write` & hans's `global_orient`
        - hand_pose (np.array): SMPLX's local pose (15, 3 or 4), mirrored if is_left
    """
    Log.debug(f'{body_pose.shape=}')
    Log.debug(f'{hand_pose.shape=}')
    Log.debug(f'{global_orient.shape=}')

    M = np.diag([-1, 1, 1])  # Preparing for the left hand switch
    lib = Lib(body_pose)
    is_torch = lib.__name__ == 'torch'
    hand_pose = rotMat(hand_pose)
    global_orient = rotMat(global_orient)
    if is_left:
        idx_elbow = 19  # +1 offset due to root bone
        global_orient = M @ global_orient @ M  # mirror switch
        hand_pose = M @ hand_pose @ M
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
    print(f'{wrist_pose=}\n{body_elbow.shape=}\n{body_elbow[1:]}\n{np.linalg.inv(body_elbow[1:])=}\n{global_orient.shape=}')

    wrist_pose = np.ones_like(wrist_pose)  # TODO DEBUG

    wrist_pose = rotMat_to_quat(wrist_pose)
    hand_pose = rotMat_to_quat(hand_pose)

    Log.debug(f'wrist_pose: {wrist_pose.shape}')
    Log.debug(f'hand_pose: {hand_pose.shape}')

    return wrist_pose, hand_pose


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
    # pose_body = data('body_pose', mapping='smplx', run='gvhmr').value
    data, HAND, armature, Slice = check_before_run(data, 'HANDS', 'wilor', mapping, Slice)

    BODY: list[str] = getattr(Map()['smplx'], 'BODY')   # type:ignore
    body_pose = get_bones_global_rotation(armature=armature, bone_resort=BODY, Slice=Slice)

    is_left = True if 'L' in data.who else False
    prefix = 'left_' if is_left else 'right_'
    name_wrist: str = BODY[21] if is_left else BODY[22]  # +1 offset
    HAND = [name_wrist] + [prefix + bone for bone in HAND]

    hand_pose = data('hand_pose').value[Slice]
    rotate = data('global_orient').value[Slice]
    rotate = quat(rotate)

    wrist_rotate, hand_pose = mano_to_smplx(body_pose, hand_pose, rotate, is_left=is_left, **kwargs)
    wrist_rotate = wrist_rotate.reshape(-1, 1, wrist_rotate.shape[-1])
    hand_pose = np.concatenate([wrist_rotate, hand_pose], axis=1)  # (frames,16,3|4)
    Log.debug(f'hand_pose final: {hand_pose.shape}')

    with bpy_action(armature, ';'.join([data.who, data.run])) as action:
        pose_apply(armature=armature, action=action, pose=hand_pose, bones=HAND, **kwargs)
