from ..lib import *
from ..b import check_before_run, pose_apply, bpy_action


def gvhmr(
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
    gvhmr(data('smplx', 'gvhmr', person=0))
    ```
    """
    data, BODY, armature, Slice = check_before_run(data, 'BODY', 'gvhmr', mapping, Slice)

    transl = data('transl', 'global').value[Slice]
    rotate = data('global_orient', 'global').value[Slice]
    rotate = rotate.reshape(-1, 1, rotate.shape[-1])
    pose = data('body_pose', 'global').value[Slice]
    pose = np.concatenate([rotate, pose], axis=1)  # (frames,22,3 or 4)

    with bpy_action(armature, ';'.join([data.who, data.run])) as action:
        pose_apply(armature=armature, action=action, pose=pose, transl=transl, transl_base=transl[base_frame], bones=BODY, **kwargs)
