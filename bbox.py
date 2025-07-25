from .b import *


def bbox(
    who: str,
    video: str | None = None,
    Slice: slice | None = None,
    **kwargs,
):
    """
    Args:
        who (str): 数据标识符
        video (str): 视频文件路径
        Slice (slice): 帧范围切片
        frame (int): 开始帧数

    Returns:
        bpy.types.Object: 生成的bbox物体
    """
    # get motion data
    _data = get_motion_data(who)('bbox')
    if Slice:
        _data = _data[Slice]
    # shape: (总帧数, 4)，格式：[x, y, X, Y]（xy=左上，XY=右下）
    data = _data.value
    total_frames = data.shape[0]

    if (video_plane := bpy.context.active_object) and video_plane.type == 'MESH' and video_plane.name.startswith('video:'):
        _w, _h = get_video_plain_wh(video_plane)
        if not (_w and _h):
            raise ValueError(f"视频平面 {video_plane.name} 没有找到视频宽高信息，请检查视频材质。")
    else:
        video_plane, _w, _h = add_video_plain(who, video, **kwargs)
    ratio = _h / _w  # 视频宽高比（高/宽）

    # 创建bbox平面（尺寸1x1，后续通过缩放匹配实际大小）
    HEIGHT = 0.01
    bpy.ops.mesh.primitive_plane_add(size=1, align='WORLD', location=(0, 0, HEIGHT))  # Z轴偏移避免遮挡
    bbox_obj = bpy.context.active_object
    if not bbox_obj:
        raise RuntimeError("Failed to create bbox object")
    bbox_obj.select_set(False)
    bbox_obj.name = f"bbox:{who}"
    bbox_obj.show_name = True
    bbox_obj.display_type = 'BOUNDS'  # 仅显示边界框
    bbox_obj.parent = video_plane

    # 预计算所有帧的位置和缩放数据
    locations = np.zeros((total_frames, 3))
    scales = np.zeros((total_frames, 3))
    for frame_idx in range(total_frames):
        x, y, X, Y = data[frame_idx]
        # 像素坐标转归一化坐标（视频宽高范围[0, video_w]和[0, video_h]）
        norm_x = x / _w          # X轴归一化（0~1）
        norm_y = 1 - (y / _h)    # Y轴反转（图像Y向下→3D Y向上）
        norm_X = X / _w
        norm_Y = 1 - (Y / _h)

        # 计算bbox中心坐标（在视频平面内）
        center_x = (norm_x + norm_X) / 2 - 0.5  # 转换为[-0.5, 0.5]范围（视频平面宽度1）
        center_y = ((norm_y + norm_Y) / 2 - 0.5) * ratio  # 先转换为[-0.5, 0.5]，再缩放到[-ratio/2, ratio/2]

        # 计算bbox缩放比例（相对于视频平面）
        scale_x = (norm_X - norm_x)   # 宽度占视频宽度的比例
        scale_y = (norm_Y - norm_y) * ratio   # 高度占视频高度的比例（已适配宽高比）

        # 存储位置和缩放数据
        locations[frame_idx] = [center_x, center_y, HEIGHT]
        scales[frame_idx] = [scale_x, scale_y, 1]
        yield

    with bpy_action(bbox_obj, name=f"bbox:{who}", nla_push=False) as action:
        yield from add_keyframes(action, locations, _data.begin + 1, "location", "Object Transforms")
        yield from add_keyframes(action, scales, _data.begin + 1, "scale", "Object Transforms")
    video_plane.select_set(True)
    bpy.context.view_layer.objects.active = video_plane


def get_video_plain_wh(video_plane):
    _w, _h = None, None
    if video_plane.data.materials:
        material = video_plane.data.materials[0]
        if material and material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    if node.image.source == 'MOVIE':
                        _w, _h = node.image.size
                        break
    return _w, _h


def add_video_plain(who, video, **kwargs):
    if not video and (npz := kwargs.get('npz', None)):
        file = str(npz).rstrip('.mocap.npz')
        Dir, filename = os.path.split(file)
        # 在Dir下查找以filename.***的文件
        for f in os.listdir(Dir):
            if f.startswith(filename):
                video = os.path.join(Dir, f)
                break
    if not (video and os.path.exists(video)):
        raise FileNotFoundError(f"视频文件不存在：{video}")
    # 获取视频宽高
    img = bpy.data.images.load(video)
    img.source = 'MOVIE'
    _w, _h = img.size
    video_frames = img.frame_duration
    del img

    # 创建平面并调整尺寸（宽度1，高度=宽高比）
    bpy.ops.mesh.primitive_plane_add(size=1, align='WORLD', location=(0, 0, 0))
    video_plane = bpy.context.active_object
    if not video_plane:
        raise RuntimeError("Failed to create video plane object")
    video_plane.name = f"video:{os.path.basename(video)}"
    video_plane.scale = (1, _h / _w, 1)  # 调整高度保持宽高比
    bpy.ops.object.transform_apply(scale=True)  # 应用缩放

    # 创建视频材质（使用 principled BSDF 节点）
    video_mat = bpy.data.materials.new(name=f"video:{who}")
    video_mat.use_nodes = True
    nodes = video_mat.node_tree.nodes  # type:ignore
    links = video_mat.node_tree.links  # type:ignore

    # 清除默认节点并添加视频纹理
    for node in nodes:
        nodes.remove(node)
    tex_node = nodes.new('ShaderNodeTexImage')
    tex_node.image = bpy.data.images.load(video)  # type:ignore
    tex_node.image.source = 'MOVIE'  # type:ignore
    tex_node.image_user.frame_duration = video_frames  # type:ignore
    tex_node.image_user.use_auto_refresh = True  # 自动刷新视频帧 # type:ignore
    tex_node.image_user.use_cyclic = True  # 循环播放 # type:ignore

    # 连接材质节点
    bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    links.new(tex_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
    video_plane.data.materials.append(video_mat)  # type:ignore
    return video_plane, int(_w), int(_h)
