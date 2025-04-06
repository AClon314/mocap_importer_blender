from ..lib import *
Log = getLogger(__name__)


def wilor(
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
    wilor(data('smplx', 'wilor', person=0))
    ```
    """
    data, armature, bone_rot, HAND, _Range = check_before_run('wilor', 'HANDS', data, Range, mapping)

    # rotate = data(prop='global_orient').value
    # rotate = rotate.reshape(-1, 1, rotate.shape[-1])
    pose = data(prop='hand_pose').value
    # pose = np.concatenate([rotate, pose], axis=1)  # (frames,22,3|4)

    with new_action(armature, ';'.join([data.who, data.run_keyname])) as action:
        for f in _Range:
            # print(f'wilor {ID}: {f}/{range_frame[1]}\t{f / range_frame[1] * 100:.3f}%', end='\r')
            apply_pose(action=action, pose=pose[f], frame=f + 1, bones=HAND, bone_rot=bone_rot, **kwargs)

    Log.info(f'done')
