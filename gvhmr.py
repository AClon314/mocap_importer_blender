from .lib import *
Log = get_logger(__name__)


def Rodrigues(rotvec: np.ndarray) -> np.ndarray:
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
    theta = np.linalg.norm(rotvec)  # L2 norm
    k = rotvec
    if theta > 0.:
        k = (rotvec / theta).reshape(3, 1)    # 旋转向量的单位向量, TODO why (3,1) instead 3 ?
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


def rodrigues_to_body_shapes(pose: np.ndarray):
    """
    Parameters
    ----------
    pose : np.ndarray
        22x3 rotation vectors
    """
    pose = np.asarray(pose).reshape(22, 3)
    rot_matrixs = [Rodrigues(rot) for rot in pose]
    body_shapes = np.concatenate([
        (rotM - np.eye(3)).ravel() for rotM in rot_matrixs[1:]
    ])
    ret = (rot_matrixs, body_shapes)
    # log_array(rot_matrixs, 'body_shapes')
    return ret


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
    body_pose,
    armature: 'bpy.types.Object',
    frame=0,
    **kwargs
):
    """apply trans pose and shape to character"""

    mrots, bsh = rodrigues_to_body_shapes(body_pose)
    # mrots = body_pose

    trans = Vector((trans[0], trans[1] - high_from_floor, trans[2]))

    armature.pose.bones[BODY[1]].location = trans
    armature.pose.bones[BODY[1]].keyframe_insert('location', frame=frame)

    # armature.pose.bones[BODY[0]].rotation_quaternion.w = 1.0
    # armature.pose.bones[BODY[0]].rotation_quaternion.x = 0.0

    for i, rot in enumerate(mrots, start=1):    # skip root!
        if i < kwargs.get('ibone', 22):
            bone = armature.pose.bones[BODY[i]]
            # log_array(rot, f'{i}_{BODY[i]}')
            bone.rotation_quaternion = Matrix(rot).to_quaternion()  # type: ignore

            if frame is not None:
                bone.keyframe_insert('rotation_quaternion', frame=frame)


def per_frame(From_data, to_armature, at_frame, **kwargs):
    global_translation = From_data['smpl_params_global']['transl'][at_frame]
    global_orient: np.ndarray = From_data['smpl_params_global']['global_orient'][at_frame]
    body_pose: np.ndarray = From_data['smpl_params_global']['body_pose'][at_frame]
    body_pose = body_pose.reshape(int(len(body_pose) / 3), 3)   # (21,3)
    body_pose = np.vstack([global_orient, body_pose])  # (22,3)
    apply_pose(global_translation, body_pose, to_armature, at_frame, **kwargs)
    Log.info(f'done')


def gvhmr(file, Range=(0, None), mapping=None, **kwargs):
    is_range = len(Range) > 1
    # data = load_pickle(file)
    data = load_npz(file)

    armature = bpy.context.active_object
    if armature is None or armature.type != 'ARMATURE':
        bpy.ops.scene.smplx_add_gender()    # type: ignore
    mapping = None if mapping and mapping.lower() == 'auto' else mapping
    mapping = get_mapping_from_selected_or_objs(mapping)
    global BODY
    BODY = Mod()[mapping].BODY   # type:ignore

    armature = bpy.context.active_object
    if armature is None:
        raise ValueError('No armature found')

    Range = list(Range)
    if is_range and Range[1] is None:
        Range[1] = len(data['smpl_params_global']['transl'])
        Log.warning(f'Range[1] is None, set to {Range[1]}')
    # shape = results[character]['betas'].tolist()
    for f in range(*Range):
        print(f'gvhmr {ID}: {f}/{Range[1]}\t{f/Range[1]*100:.3f}%', end='\r')
        if is_range:
            bpy.context.scene.frame_set(f)
        per_frame(data, armature, f, **kwargs)
        bpy.context.view_layer.update()

    Log.info(f'done')
