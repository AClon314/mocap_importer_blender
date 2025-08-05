import numpy as np
from . import b
from .b import *
from .lib import copy_args, GEN
HEIGHT = 0.01


def gen_bbox(
    who: str,
    video: str | None = None,
    **kwargs,
):
    """
    Args:
        who (str): 数据标识符
        video (str): 视频文件路径
        frame (int): 开始帧数
    """
    _data = b.MOTION_DATA('bbox', mapping='smplx', who=who)
    if not _data:
        return
    Log.debug(f'{who=} {_data.keys()=}')
    # shape: (总帧数, 4)，格式：[x, y, X, Y]（xy=左上，XY=右下）
    data = _data.value
    total_frames = data.shape[0]

    video_plane, _w, _h = get_or_add_video_plane(who, video, _data.npz)
    ratio = _h / _w

    bbox_obj = add_bbox_plane(who, video_plane)

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

    start = 0 if _data.Slice.start is None else _data.Slice.start
    pg = Progress(total_frames * 2)
    with bpy_action(bbox_obj, name=f"bbox:{who}", nla_push=False) as action:
        yield from add_keyframes(action, locations, start + 1, "location", "Object Transforms", update=pg.update)
        yield from add_keyframes(action, scales, start + 1, "scale", "Object Transforms", update=pg.update)
    video_plane.select_set(True)
    bpy.context.view_layer.objects.active = video_plane


@copy_args(gen_bbox)  # type: ignore
def bbox(*who: str, **kwargs):
    whos, _ = props_filter(who=who)
    for w in whos:
        GEN.append(gen_bbox(w, **kwargs))


def add_bbox_plane(who: str, video_plane: bpy.types.Object):
    '''create bbox **mesh** plane, size 1m x 1m. Use re-size animation to match actual size later.'''
    bpy.ops.mesh.primitive_plane_add(size=1, align='WORLD', location=(0, 0, HEIGHT))  # Z轴偏移避免遮挡
    bbox_obj = bpy.context.active_object
    if not bbox_obj:
        raise RuntimeError("Failed to create bbox object")
    bbox_obj.select_set(False)
    bbox_obj.name = f"bbox:{who}"
    bbox_obj.show_name = True
    bbox_obj.display_type = 'BOUNDS'  # 仅显示边界框
    bbox_obj.parent = video_plane
    return bbox_obj


def get_or_add_video_plane(who: str, video: str | None, npz: str | None = None):
    if (video_plane := bpy.context.active_object) and video_plane.type == 'MESH' and video_plane.name.startswith('video:'):
        _w, _h = get_wh_by_node(video_plane)
    else:
        video_plane, _w, _h = add_video_plane(who, video, npz=npz)
    return video_plane, _w, _h


def get_wh_by_node(video_plane: bpy.types.Object) -> tuple[int, int]:
    _w, _h = 0, 0
    data = video_plane.data
    if not isinstance(data, bpy.types.Mesh):
        raise TypeError(f"Expected a Mesh object, got {type(data)}")
    material = data.materials[0]
    if material and material.node_tree:
        for node in material.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                if node.image.source == 'MOVIE':
                    _w, _h = node.image.size
                    break
    return _w, _h


def add_video_plane(who: str, video: str | None = None, npz: str | None = None):
    if npz and not video:
        file = str(npz).rstrip('.mocap.npz')
        Dir, filename = os.path.split(file)
        # 在Dir下查找以filename.***的文件
        for f in os.listdir(Dir):
            if f.startswith(filename):
                video = os.path.join(Dir, f)
                break
    if not (video and os.path.exists(video)):
        raise FileNotFoundError(f"视频文件不存在：{video}")

    _w, _h, video_frames = get_wh_by_load(video)

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


def get_wh_by_load(video: str):
    img = bpy.data.images.load(video)
    img.source = 'MOVIE'
    _w, _h = img.size
    video_frames = img.frame_duration
    del img
    return _w, _h, video_frames
