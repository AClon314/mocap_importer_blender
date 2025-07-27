import bpy
from ..lib import *
from ..b import *


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
        Slice (tuple, optional): Frames range. Defaults to (0, Max_frames).

    Example:
    ```python
    gvhmr(data('smplx', 'gvhmr', person=0))
    ```
    """
    data, Slice, name, transl, rotate = data_Slice_name_transl_rotate(data, Slice, run='gvhmr')
    if transl is not None:
        kw = dict(transl_base=transl[base_frame])
    else:
        kw = {}
    body_pose = data('body_pose')
    if not body_pose:
        obj = bpy.context.selected_objects[0]
        with bpy_action(obj, name) as action:
            return transform_apply(obj=obj, action=action, rotate=rotate, transl=transl)

    armature, BODY = armature_BODY(mapping, key='BODY')
    rotate = rotate.reshape(-1, 1, rotate.shape[-1])
    pose = body_pose.value[Slice]
    pose = np.concatenate([rotate, pose], axis=1)  # (frames,22,3 or 4)
    with bpy_action(armature, name) as action:
        return pose_apply(armature=armature, action=action, pose=pose, transl=transl, bones=BODY, frame=data.begin + 1, **kw, **kwargs)
