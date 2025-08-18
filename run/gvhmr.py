import bpy
from ..lib import *
from ..b import *


def gvhmr(
    armature: bpy.types.Object,
    data: MotionData,
    mapping: TYPE_MAPPING | None = None,
    Slice=slice(0, None),
    base_frame=0,
    **kwargs,
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
    data, mapping, name, Slice, transl, rotate = data_mapping_name_Slice_transl_rotate(
        armature, data, mapping, Slice, run="gvhmr"
    )
    transl_base = None if transl is None else transl[base_frame]
    body_pose = data("body_pose")
    if not body_pose:  # for cam@
        objs = bpy.context.selected_objects
        cam = objs[0] if objs else None
        if not cam:
            cam = bpy.ops.object.camera_add(location=(0, 0, 0))
            cam = bpy.context.object
            if not cam:
                raise RuntimeError("No active object and failed to add camera")
        with bpy_action(cam, name) as action:
            yield from transform_apply(
                obj=cam, action=action, rotate=rotate, transl=transl
            )
        return

    BODY = get_BONES(mapping=mapping, key="BODY")
    if not isinstance(armature.data, bpy.types.Armature):
        raise TypeError(f"Expected armature data type, got {type(armature.data)}")

    if mapping not in ("smpl", "smplx") and not kwargs.get("offset", None):
        rotate = pelvis_rotate_offset(rotate, armature, BODY[1])
    rotate = rotate.reshape(-1, 1, rotate.shape[-1])
    pose = body_pose.value[Slice]
    pose = np.concatenate([rotate, pose], axis=1)  # (frames,1+22,3 or 4)
    with bpy_action(armature, name) as action:
        yield from pose_apply(
            armature=armature,
            action=action,
            pose=pose,
            transl=transl,
            bones=BODY,
            frame=data.begin + 1,
            transl_base=transl_base,
            **kwargs,
        )


def pelvis_rotate_offset(
    rotate: np.ndarray, armature: bpy.types.Object, torso_name="torso"
):
    """Q_new_pose = Q_old_pose * Q_delta^-1"""
    if not isinstance(armature.data, bpy.types.Armature):
        raise TypeError(f"Expected armature data type, got {type(armature.data)}")
    Log.error(f"{torso_name=}")
    bone_torso = armature.data.bones[torso_name]
    facing_torso = np.array(bone_torso.tail - bone_torso.head)
    facing_torso = facing_torso / np.linalg.norm(facing_torso)
    facing_pelvis = np.array([0, 0, 1])
    Log.debug(f"{facing_torso=}\t{facing_pelvis=}")
    rotate = change_coord(rotate, facing_pelvis, facing_torso)
    return rotate
