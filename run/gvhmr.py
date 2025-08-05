import bpy
from ..lib import *
from ..b import *


def gvhmr(
    armature: bpy.types.Object,
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
    data, mapping, name, Slice, transl, rotate = data_mapping_name_Slice_transl_rotate(armature, data, mapping, Slice, run='gvhmr')
    transl_base = None if transl is None else transl[base_frame]
    body_pose = data('body_pose')
    if not body_pose:  # for cam@
        objs = bpy.context.selected_objects
        cam = objs[0] if objs else None
        if not cam:
            cam = bpy.ops.object.camera_add(location=(0, 0, 0))
            cam = bpy.context.object
            if not cam:
                raise RuntimeError('No active object and failed to add camera')
        with bpy_action(cam, name) as action:
            yield from transform_apply(obj=cam, action=action, rotate=rotate, transl=transl)
        return

    BODY = get_BONES(mapping=mapping, key='BODY')
    rotate = rotate.reshape(-1, 1, rotate.shape[-1])
    pose = body_pose.value[Slice]
    pose = np.concatenate([rotate, pose], axis=1)  # (frames,22,3 or 4)
    Log.debug(f'{transl_base=}')
    with bpy_action(armature, name) as action:
        yield from pose_apply(armature=armature, action=action, pose=pose, transl=transl, bones=BODY, frame=data.begin + 1, transl_base=transl_base, **kwargs)
