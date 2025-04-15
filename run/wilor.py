from ..lib import *


def wilor(
    data: MotionData,
    mapping: Optional[TYPE_MAPPING] = None,
    Range=[0, None],
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
    data, armature, rot, HAND, Slice, base_frame = check_before_run('wilor', 'HANDS', data, mapping, Range, base_frame)

    # rotate = data(prop='global_orient').value
    # rotate = rotate.reshape(-1, 1, rotate.shape[-1])
    pose = data(prop='hand_pose').value
    # pose = np.concatenate([rotate, pose], axis=1)  # (frames,22,3|4)

    with bpy_action(armature, ';'.join([data.who, data.run])) as action:
        pose_reset(action, HAND, rot)
        pose_apply(action=action, pose=pose, bones=HAND, rot=rot, **kwargs)
