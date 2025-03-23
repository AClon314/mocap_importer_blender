from optparse import Option
from .lib import *
Log = get_logger(__name__)


def Rodrigues(rot_vec3: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    rotvec : np.ndarray
        3D rotation vector

    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    # L1范数：Σ|xᵢ|
    # L2范数（欧式距离）：√ (Σ|xᵢ|²)
    theta = np.linalg.norm(rot_vec3)  # L2 norm
    k = rot_vec3
    if theta > 0.:
        k = (rot_vec3 / theta).reshape(3, 1)    # 旋转向量的单位向量, TODO why (3,1) instead 3 ?
    cos = np.cos(theta)
    sin = np.sin(theta)
    K = np.asarray([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]],
        dtype=object,
    )
    R: np.ndarray = cos * np.eye(3) + sin * K + (1 - cos) * k.dot(k.T)
    # R: np.ndarray = np.eye(3) + sin * K + (1 - cos) * K.dot(K.T)
    return R


def rodrigues_to_rotates(pose: np.ndarray):
    """
    Parameters
    ----------
    pose : np.ndarray
        22x3 rotation vectors
    """
    # pose = np.asarray(pose).reshape(22, 3)
    rot_matrixs = [Rodrigues(rot) for rot in pose]
    # body_shapes = np.concatenate([
    #     (rotM - np.eye(3)).ravel() for rotM in rot_matrixs[1:]
    # ])
    # ret = (rot_matrixs, body_shapes)
    # log_array(rot_matrixs, 'body_shapes')
    return rot_matrixs


def get_global_pose(global_pose, armature, frame=None):

    armature.pose.bones[BODY[0]].rotation_quaternion.w = 0.0
    armature.pose.bones[BODY[0]].rotation_quaternion.x = -1.0

    bone = armature.pose.bones[BODY[1]]
    # if frame is not None:
    #     bone.keyframe_insert('rotation_quaternion', frame=frame)

    root_orig = armature.pose.bones[BODY[0]].rotation_quaternion
    mw_orig = armature.matrix_world.to_quaternion()
    pelvis_quat = Matrix(global_pose[0]).to_quaternion()

    bone.rotation_quaternion = pelvis_quat
    bone.keyframe_insert('rotation_quaternion', frame=frame)

    pelvis_applyied = armature.pose.bones[BODY[1]].rotation_quaternion
    bpy.context.view_layer.update()

    rot_world_orig = root_orig @ pelvis_applyied @ mw_orig  # pegar a rotacao em relacao ao mundo

    return rot_world_orig


def apply_pose(
    trans,
    pose,
    armature: 'bpy.types.Object',
    frame=0,
    **kwargs
):
    """apply trans pose and shape to character"""
    root = pose[0]

    rots = [Rodrigues(rot) for rot in pose]

    trans = Vector((trans[0], trans[1], trans[2]))

    armature.pose.bones[BODY[0]].location = trans
    armature.pose.bones[BODY[0]].keyframe_insert('location', frame=frame)

    # armature.pose.bones[BODY[0]].rotation_euler = Euler((root[0], root[1], root[2]), 'XYZ')  # -pi, pi
    # armature.pose.bones[BODY[0]].keyframe_insert('rotation_euler', frame=frame)

    for i, rot in enumerate(rots, start=1):    # skip root!
        if i <= kwargs.get('ibone', 24):
            bone = armature.pose.bones[BODY[i]]
            # log_array(rot, f'{i}_{BODY[i]}')
            # -1, 1
            bone.rotation_quaternion = Matrix(rot).to_quaternion()  # type: ignore
            # bone.rotation_quaternion = Quaternion()

            if frame is not None:
                bone.keyframe_insert('rotation_quaternion', frame=frame)


def per_frame(data: MotionData, to_armature, at_frame: int, **kwargs):
    translation = data(key='trans', coord='global').value[at_frame]  # ['smpl_params_global']['transl'][at_frame]
    rotate: np.ndarray = data(key='rotate', coord='global').value[at_frame]  # ['smpl_params_global']['global_orient'][at_frame]
    pose: np.ndarray = data(key='pose', coord='global').value[at_frame]  # ['smpl_params_global']['body_pose'][at_frame]
    pose = pose.reshape(int(len(pose) / 3), 3)   # (21,3)
    pose = np.vstack([rotate, pose])  # (22,3)
    apply_pose(translation, pose, to_armature, at_frame, **kwargs)


def gvhmr(
    data: MotionData,
    frame_range=(0, None),
    mapping: Optional[TYPE_MAPPING] = None,
    **kwargs
):
    """
    per person

    Args:
        data (MotionData, dict): mocap data
        frame_range (tuple, optional): Frames range. Defaults to (0, Max_frames).

    Example:
    ```python
    gvhmr(data('smplx', 'gvhmr', person=0))
    ```
    """
    is_range = len(frame_range) > 1

    armature = bpy.context.active_object
    if armature is None or armature.type != 'ARMATURE':
        bpy.ops.scene.smplx_add_gender()    # type: ignore
    mapping = None if mapping == 'auto' else mapping
    mapping = get_mapping_from_selected_or_objs(mapping)

    # mapping = data.mappings
    # if len(mapping) == 0:
    #     raise ValueError('Mapping must be one')
    # elif len(mapping) > 1:
    #     Log.warning(f'Mapping must be one: {mapping}, fallback to {mapping[0]}')
    # mapping = mapping[0]

    global BODY
    BODY = Mod()[mapping].BODY   # type:ignore

    armature = bpy.context.active_object
    if armature is None:
        raise ValueError('No armature found')

    frame_range = list(frame_range)
    if is_range and frame_range[1] is None:
        frame_range[1] = len(data(mapping='smplx', run='gvhmr', key='trans', coord='global').value)  # type: ignore
        Log.warning(f'Range[1] is None, set to {frame_range[1]}')
    # shape = results[character]['betas'].tolist()
    for f in range(*frame_range):
        print(f'gvhmr {ID}: {f}/{frame_range[1]}\t{f/frame_range[1]*100:.3f}%', end='\r')
        if is_range:
            bpy.context.scene.frame_set(f)
        per_frame(data, armature, f, **kwargs)
        bpy.context.view_layer.update()

    Log.info(f'done')
    del data
