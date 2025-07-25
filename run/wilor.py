from ..lib import *
from ..b import *


def wrist_hand(
    wrist: 'np.ndarray', hand: 'np.ndarray',
    BODY: list[str], HAND: list[str], is_left=False
):
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
    # get hand data
    data, Slice, name, transl, rotate = data_Slice_name_transl_rotate(data, Slice, run='wilor')
    armature, HAND = armature_BODY(mapping, key='HANDS')
    hand_pose = data('hand_pose').value[Slice]
    rotate = data('global_orient').value[Slice]

    # get body data
    BODY: list[str] = getattr(Map()['smplx'], 'BODY')   # type:ignore
    body_pose = get_bones_global_rotation(armature=armature, bone_resort=BODY, Slice=Slice)
    is_left = True if 'L' in data.who else False

    wrist_rotate, hand_pose = mano_to_smplx(
        body_pose, hand_pose, rotate,
        base_frame=base_frame, is_left=is_left)
    hand_pose, HAND = wrist_hand(wrist_rotate, hand_pose, BODY, HAND, is_left=is_left)
    with bpy_action(armature, name) as action:
        return pose_apply(armature=armature, action=action, pose=hand_pose, bones=HAND, frame=data.begin + 1, **kwargs)
