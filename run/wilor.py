from ..lib import *
from ..b import check_before_run, pose_apply, bpy_action


def compute_global_rotation(pose_axis_anges, joint_idx):
    """
    calculating joints' global rotation
    Args:
        pose_axis_anges (np.array): SMPLX's local pose (22,3)
    Returns:
        np.array: (3, 3)
    """
    global_rotation = np.eye(3)
    parents = [
        -1, 0, 0, 0, 1,
        2, 3, 4, 5, 6, 7,
        8, 9, 9, 9, 12,
        13, 14, 16, 17, 18,
        19]
    while joint_idx != -1:
        joint_rotation = Rodrigues(pose_axis_anges[joint_idx])
        global_rotation = joint_rotation @ global_rotation
        joint_idx = parents[joint_idx]
    return global_rotation


def get_bone_global_rotation(armature, bone_name, frame):
    """
    获取某骨骼在某帧的全局旋转矩阵。

    Args:
        armature_name (str): 骨架对象名称。
        bone_name (str): 骨骼名称。
        frame (int): 帧号。

    Returns:
        Matrix: 骨骼的全局旋转矩阵。
    """
    bpy.context.scene.frame_set(frame)  # type: ignore

    # 获取骨骼的全局矩阵
    bone = armature.pose.bones[bone_name]
    global_matrix = armature.matrix_world @ bone.matrix

    # 提取旋转部分
    global_rotation = global_matrix.to_3x3()
    return global_rotation


def mano_to_smplx(smplx_body_gvhmr, mano_hand_hamer):
    """https://github.com/VincentHu19/Mano2Smpl-X/blob/main/mano2smplx.py"""
    # M = np.diag([-1, 1, 1])  # Preparing for the left hand switch
    M = np.diag([1, 1, 1])

    lib = Lib(smplx_body_gvhmr["global_orient"])
    is_torch = lib.__name__ == 'torch'
    # Assuming that your data are stored in gvhmr_smplx_params and hamer_mano_params
    # full_body_pose = lib.concatenate((smplx_body_gvhmr["global_orient"], smplx_body_gvhmr["body_pose"].reshape(21, 3)), **{Axis(is_torch): 0})     # gvhmr_smplx_params["global_orient"]: (3, 3)
    # left_elbow_global_rot = compute_global_rotation(full_body_pose, 18)  # left elbow IDX: 18
    # right_elbow_global_rot = compute_global_rotation(full_body_pose, 19)  # left elbow IDX: 19
    left_elbow_global_rot = smplx_body_gvhmr[:, 18]
    right_elbow_global_rot = smplx_body_gvhmr[:, 19]

    left_wrist_global_rot = mano_hand_hamer["global_orient"][0].cpu().numpy()  # hamer_mano_params["global_orient"]: (2, 3, 3)
    left_wrist_global_rot = M @ left_wrist_global_rot @ M  # mirror switch
    left_wrist_pose = np.linalg.inv(left_elbow_global_rot) @ left_wrist_global_rot

    right_wrist_global_rot = mano_hand_hamer["global_orient"][1].cpu().numpy()
    right_wrist_pose = np.linalg.inv(right_elbow_global_rot) @ right_wrist_global_rot

    left_wrist_pose_vec = euler(RotMat_to_quat(left_wrist_pose))
    right_wrist_pose_vec = euler(RotMat_to_quat(right_wrist_pose))

    left_hand_pose = np.ones(45)
    right_hand_pose = np.ones(45)
    for i in range(15):
        left_finger_pose = M @ mano_hand_hamer["hand_pose"][0][i].cpu().numpy() @ M  # hamer_mano_params["hand_pose"]: (2, 15, 3, 3)
        left_finger_pose_vec = euler(RotMat_to_quat(left_finger_pose))
        left_hand_pose[i * 3: i * 3 + 3] = left_finger_pose_vec

        right_finger_pose = mano_hand_hamer["hand_pose"][1][i].cpu().numpy()
        right_finger_pose_vec = euler(RotMat_to_quat(right_finger_pose))
        right_hand_pose[i * 3: i * 3 + 3] = right_finger_pose_vec

    # smplx_body_gvhmr["body_pose"][57: 60] = left_wrist_pose_vec
    # smplx_body_gvhmr["body_pose"][60: 63] = right_wrist_pose_vec
    # smplx_body_gvhmr["left_hand_pose"] = left_hand_pose
    # smplx_body_gvhmr["right_hand_pose"] = right_hand_pose

    return left_wrist_pose_vec, right_wrist_pose_vec, left_hand_pose, right_hand_pose


def wilor(
    data: MotionData,
    mapping: TYPE_MAPPING | None = None,
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
    # pose_body = data('body_pose', mapping='smplx', run='gvhmr').value
    # pose_body =
    data, HAND, armature, rot, Slice = check_before_run(data, 'HANDS', 'wilor', mapping, Range)
    # rotate = data(prop='global_orient').value
    # rotate = rotate.reshape(-1, 1, rotate.shape[-1])
    pose = data('hand_pose').value[Slice]
    # pose = np.concatenate([rotate, pose], axis=1)  # (frames,22,3|4)

    _, _, pose, r = mano_to_smplx(pose_body, pose)

    with bpy_action(armature, ';'.join([data.who, data.run])) as action:
        pose_apply(action=action, pose=pose, bones=HAND, rot=rot, **kwargs)
