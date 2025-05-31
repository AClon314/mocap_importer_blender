from ..lib import *
from ..b import init_0, init_1, pose_apply, bpy_action, transform_apply


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
    data, Slice, name, transl, rotate = init_0(data, Slice, run='gvhmr')
    if transl:
        kw = dict(transl_base=transl[base_frame])
    else:
        kw = {}
    body_pose = data('body_pose')
    if not body_pose:
        obj = bpy.context.selected_objects[0]
        with bpy_action(obj, name) as action:
            transform_apply(obj=obj, action=action, rotate=rotate, transl=transl)
        return

    armature, BODY = init_1(mapping, key='BODY')
    rotate = rotate.reshape(-1, 1, rotate.shape[-1])
    pose = body_pose.value[Slice]
    pose = np.concatenate([rotate, pose], axis=1)  # (frames,22,3 or 4)
    with bpy_action(armature, name) as action:
        pose_apply(armature=armature, action=action, pose=pose, transl=transl, bones=BODY, frame=data.begin + 1, **kw, **kwargs)
