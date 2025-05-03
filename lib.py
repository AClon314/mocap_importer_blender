"""
https://github.com/carlosedubarreto/CEB_4d_Humans/blob/main/four_d_humans_blender.py
"""
import os
import bpy
import sys
import importlib
import numpy as np
from .logger import Log
from types import ModuleType
from typing import Dict, List, Sequence, Literal, TypeVar, get_args
DIR_SELF = os.path.dirname(__file__)
DIR_MAPPING = os.path.join(DIR_SELF, 'mapping')
MAPPING_TEMPLATE = os.path.join(DIR_MAPPING, 'template.pyi')
TYPE_MAPPING = Literal['smpl', 'smplx']
TYPE_RUN = Literal['gvhmr', 'wilor']
T = TypeVar('T')
TN = np.ndarray
MOTION_DATA = None
TYPE_PROP = Literal['body_pose', 'hand_pose', 'global_orient', 'betas', 'transl', 'bbox']
PROP_KEY = get_args(TYPE_PROP)
def get_major(L: Sequence[T]) -> T | None: return max(L, key=L.count) if L else None
def Map(Dir='mapping') -> Dict[TYPE_MAPPING | str, ModuleType]: return Mod(Dir=Dir)
def Run(Dir='run') -> Dict[TYPE_RUN | str, ModuleType]: return Mod(Dir=Dir)
def Axis(is_torch=False): return 'dim' if is_torch else 'axis'


def in_or_skip(part, full, pattern=''):
    """used in class `MotionData`"""
    if pattern:
        part = pattern.format(part) if part else None
        full = pattern.format(full) if full else None
    return (not part) or (not full) or (part in full)


def warn_or_return_first(L: List[T]) -> T:
    """used in class `MotionData`"""
    Len = len(L)
    if Len > 1:
        Log.warning(f'{Len} > 1', extra={'report': False}, stack_info=True)
    return L[0]


def Mod(Dir='mapping'):
    files = os.listdir(os.path.join(DIR_SELF, Dir))
    pys = []
    mods: Dict[str, ModuleType] = {}
    for f in files:
        if f.endswith('.py') and f not in ['template.py', '__init__.py']:
            pys.append(f[:-3])
    for p in pys:
        mod = importlib.import_module(f'.{Dir}.{p}', package=__package__)
        mods.update({p: mod})
    return mods


def items_mapping(self=None, context=None):
    items: List[tuple[str, str, str]] = [(
        'auto', 'Auto',
        'Auto detect armature type, based on name (will enhanced in later version)')]
    for k, m in Map().items():
        help = ''
        try:
            help = m.HELP[bpy.app.translations.locale]
        except Exception:
            Log.warning(f'No help for {k}')
            help = m.__doc__ if m.__doc__ else ''
        items.append((k, k, help))
    return items


def items_motions(self=None, context=None):
    """TODO: this func will trigger when redraw, so frequently"""
    items: List[tuple[str, str, str]] = []
    if MOTION_DATA is None:
        load_data()
    if MOTION_DATA is not None:
        for k in MOTION_DATA.whos:
            items.append((k, k, ''))
    return items


def load_data(self=None, context=None):
    """load motion data when npz file path changed"""
    global MOTION_DATA
    if MOTION_DATA is not None:
        del MOTION_DATA
    file = bpy.context.scene.mocap_importer.input_file   # type: ignore
    MOTION_DATA = MotionData(npz=file)


class MotionData(dict):
    """
    usage:
    ```python
    # __call__ is filter
    data(mapping='smplx', run='gvhmr', prop='trans', coord='global').values()[0]
    ```
    """

    def keys(self) -> List[str]:
        return list(super().keys())

    def values(self) -> List[np.ndarray]:
        return list(super().values())

    def __init__(self, /, *args, npz: str | os.PathLike | None = None, lazy=False, **kwargs):
        """
        Inherit from dict
        Args:
            npz (str, Path, optional): npz file path.
            lazy (bool, optional): if True, do NOT load npz file.
        """
        super().__init__(*args, **kwargs)
        if npz:
            self.npz = npz
            if not lazy:
                self.update(np.load(npz, allow_pickle=True))

    def __call__(
        self,
        *prop: TYPE_PROP | Literal['global', 'incam'],
        mapping: TYPE_MAPPING | None = None,
        run: TYPE_RUN | None = None,
        who: str | int | None = None,
        Range=lambda frame: 0 < frame < np.inf,
        # coord: Optional[Literal['global', 'incam']] = None,
    ):
        # Log.debug(f'self.__dict__={self.__dict__}')
        D = MotionData(npz=self.npz, lazy=True)
        if isinstance(who, int):
            who = f'person{who}'

        for k, v in self.items():
            # TODO: Range (int)
            is_in = [in_or_skip(args, k, ';{};') for args in [mapping, run, who, *prop]]
            is_in = all(is_in)
            if is_in:
                D[k] = v
        return D

    def distinct(self, col_num: int):
        """
        Args:
            col_num (int): 0 for mapping, 1 for run, 2 for key, 3 for person, 4 for coord
            literal : filter keys by Literal. Defaults to None.

        """
        L: List[str] = []
        for k in self.keys():
            keys = k.split(';')
            col_name = keys[col_num]
            if col_name not in L:
                L.append(col_name)
        return L

    @property
    def mappings(self): return self.distinct(0)
    @property
    def runs(self): return self.distinct(1)
    @property
    def whos(self): return self.distinct(2)
    @property
    def begins(self): return self.distinct(3)

    def props(self, col=0):
        """
        Returns:
            `['*_pose', 'global_orient', 'transl', 'betas', your_customkeys]`"""
        return self.distinct(col + 4)

    # @property
    # def coords(self): return self.distinct(4)

    @property
    def mapping(self): return warn_or_return_first(self.mappings)
    @property
    def run(self): return warn_or_return_first(self.runs)
    @property
    def who(self): return warn_or_return_first(self.whos)
    def prop(self, col=0): return self.props(col)[0]

    @property
    def value(self):
        """same as:
        ```python
        return self.values()[0]
        ```"""
        return warn_or_return_first(self.values())

    @property
    def keyname(self):
        """return **FULL** keyname like `smplx;gvhmr;pose;person0;global`, same as:
        ```python
        return self.keys()[0]
        ```"""
        return self.keys()[0]


def log_array(arr: np.ndarray | list, name='ndarray'):
    def recursive_convert(array):
        if isinstance(array, np.ndarray):
            return array.tolist()
        elif isinstance(array, list):
            return [recursive_convert(item) for item in array]
        else:
            return array

    def array_to_str(array):
        if isinstance(array, list):
            return '\t'.join(array_to_str(item) for item in array)
        else:
            return str(array)

    if isinstance(arr, list):
        arr = np.array(arr)

    array = recursive_convert(arr.tolist())
    array = array_to_str(array)
    text = f'{name}={array}'
    Log.debug(text)
    print()
    return text


def bone_to_dict(bone, whiteList: Sequence[str] | None = None):
    """bone to dict, Recursive calls to this function form a tree"""
    # if deep_max and deep > deep_max:
    #     raise ValueError(f'Bones tree too deep, {deep} > {deep_max}')
    return {child.name: bone_to_dict(child) for child in bone.children if in_or_skip(child.name, whiteList)}


def keys_BFS(
    d: dict, wrap=False,
    deep_max=1000,
):
    """
    sort keys of dict by BFS

    Parameters
    ----------
    d : dict
        dict to sort
    wrap : bool, optional
        if True, return [[key0], [k1,k2], [k3,k4,k5], ...]
        else return [key0, k1, k2, k3, k4, k5, ...]
    """
    deep = 0
    ret = []
    Q = [d]  # 初始队列包含根字典
    while Q:
        if deep_max and deep > deep_max:
            raise ValueError(f'Dict tree too deep, {deep} > {deep_max}')
        current_level = []
        next_queue = []
        for current_dict in Q:
            # 收集当前字典的所有键到当前层级
            current_level.extend(current_dict.keys())
            # 收集当前字典的所有子字典到下一层队列
            next_queue.extend(current_dict.values())
        # 如果当前层级有键，则添加到结果中
        if current_level:
            if wrap:
                ret.append(current_level)
            else:
                ret.extend(current_level)
        # 更新队列为下一层级的子字典列表
        Q = next_queue
        deep += 1
    return ret


def get_bones_info(armature=None):
    """For debug: print bones info"""
    from .b import bones_tree, get_armatures, bone_global_rotation_matrix
    armatures = get_armatures()
    S = ""
    for armature in armatures:
        tree = bones_tree(armature=armature)
        List = keys_BFS(tree)
        S += f"""TYPE_BODY = Literal{List}
BONES_TREE = {tree}"""
        for b in List:
            global_rot = bone_global_rotation_matrix(armature=armature, bone=b)
            S += f"\n{b}: {global_rot}"
    return S


def get_similar(list1, list2):
    """
    calc jaccard similarity of two lists
    Returns:
        float: ∈[0, 1]
    """
    set1, set2 = set(list1), set(list2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    ret = intersection / union if union != 0 else 0
    return ret


def get_mapping(mapping: TYPE_MAPPING | None = None, armature=None):
    """
    import mapping module by name
    will set global variable BODY(temporary)
    """
    from .b import get_armatures, guess_obj_mapping
    if mapping is None:
        armature = get_armatures(armatures=[armature])[0]
        mapping = guess_obj_mapping(armature)
    if mapping is None:
        raise ValueError(f'Unknown mapping: {mapping}, try to select/add new mapping to')
    return mapping


def apply(who: str | int, mapping: TYPE_MAPPING | None, **kwargs):
    global MOTION_DATA
    if MOTION_DATA is None:
        raise ValueError('Failed to load motion data')
    data = MOTION_DATA(mapping='smplx', who=who)
    for r in data.runs:
        run = getattr(Run()[r], r)
        run(data, mapping=mapping, **kwargs)


def quat(xyz: TN) -> TN:
    """euler to quat
    Args:
        arr (TN): 输入张量/数组，shape为(...,3)，对应[roll, pitch, yaw]（弧度）
    Returns:
        quat: normalized [w,x,y,z], shape==(...,4)
    """
    if xyz.shape[-1] == 4:
        return xyz
    assert xyz.shape[-1] == 3, f"Last dimension should be 3, but found {xyz.shape}"
    lib = Lib(xyz)  # 自动检测库类型
    is_torch = lib.__name__ == 'torch'

    # 计算半角三角函数（支持广播）
    half_angles = 0.5 * xyz
    cos_half = lib.cos(half_angles)  # shape (...,3)
    sin_half = lib.sin(half_angles)

    # 分库处理维度解包
    if is_torch:
        cr, cp, cy = cos_half.unbind(dim=-1)
        sr, sp, sy = sin_half.unbind(dim=-1)
    else:  # NumPy处理
        cr, cp, cy = cos_half[..., 0], cos_half[..., 1], cos_half[..., 2]
        sr, sp, sy = sin_half[..., 0], sin_half[..., 1], sin_half[..., 2]

    # 并行计算四元数分量（保持维度）
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # 堆叠并归一化
    _quat = lib.stack([w, x, y, z], **{Axis(is_torch): -1})
    _quat /= Norm(_quat)
    return _quat


def euler(wxyz: TN) -> TN:
    """union quat to euler
    Args:
        quat (TN): [w,x,y,z], shape==(...,4)
    Returns:
        euler: [roll_x, pitch_y, yaw_z] in arc system, shape==(...,3)
    """
    if wxyz.shape[-1] == 3:
        return wxyz
    assert wxyz.shape[-1] == 4, f"Last dimension should be 4, but found {wxyz.shape}"
    lib = Lib(wxyz)  # 自动检测库类型
    is_torch = lib.__name__ == 'torch'
    EPSILON = 1e-12  # 数值稳定系数

    # 归一化四元数（防止输入未归一化）
    wxyz = wxyz / Norm(wxyz, dim=-1, keepdim=True)  # type: ignore

    # 解包四元数分量（支持广播）
    w, x, y, z = wxyz[..., 0], wxyz[..., 1], wxyz[..., 2], wxyz[..., 3]

    # 计算roll (x轴旋转)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = lib.arctan2(sinr_cosp, cosr_cosp + EPSILON)  # 防止除零

    # 计算pitch (y轴旋转)
    sinp = 2 * (w * y - z * x)
    pitch = lib.arcsin(sinp.clip(-1.0, 1.0))  # 限制在有效范围内

    # 计算yaw (z轴旋转)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = lib.arctan2(siny_cosp, cosy_cosp + EPSILON)

    # 堆叠结果
    _euler = lib.stack([roll, pitch, yaw], **{Axis(is_torch): -1})
    return _euler


def get_mod(mod1: ModuleType | str):
    if isinstance(mod1, str):
        _mod1 = sys.modules.get(mod1, None)
    else:
        _mod1 = mod1
    return _mod1


def Lib(arr, mod1: ModuleType | str = np, mod2: ModuleType | str = 'torch', ret_1_if=np.ndarray):
    """usage:
    ```python
    lib = Lib(arr)
    is_torch = lib.__name__ == 'torch'
    ```
    """
    _mod1 = get_mod(mod1)
    _mod2 = get_mod(mod2)
    if _mod1 and _mod2:
        mod = _mod1 if isinstance(arr, ret_1_if) else _mod2
    elif _mod1:
        mod = _mod1
    elif _mod2:
        mod = _mod2
    else:
        raise ImportError("Both libraries are not available.")
    # Log.debug(f"🔍 {mod.__name__}")
    return mod


def Norm(arr: TN, dim: int = -1, keepdim: bool = True) -> TN:
    """计算范数，支持批量输入"""
    lib = Lib(arr)
    is_torch = lib.__name__ == 'torch'
    if is_torch:
        return lib.norm(arr, dim=dim, keepdim=keepdim)
    else:
        return lib.linalg.norm(arr, axis=dim, keepdims=keepdim)


def skew_symmetric(v: TN) -> TN:
    """生成反对称矩阵，支持批量输入"""
    lib = Lib(v)
    is_torch = lib.__name__ == 'torch'
    axis = Axis(is_torch)
    axis_1 = {axis: -1}
    # 创建各分量
    zeros = lib.zeros_like(v[..., 0])  # 形状 (...)
    row0 = lib.stack([zeros, -v[..., 2], v[..., 1]], **axis_1)  # (...,3)
    row1 = lib.stack([v[..., 2], zeros, -v[..., 0]], **axis_1)
    row2 = lib.stack([-v[..., 1], v[..., 0], zeros], **axis_1)
    # 堆叠为矩阵
    if is_torch:
        return lib.stack([row0, row1, row2], dim=-2)
    else:
        return lib.stack([row0, row1, row2], axis=-2)  # (...,3,3)


def Rodrigues(rot_vec3: TN) -> TN:
    """
    支持批量处理的罗德里格斯公式

    Parameters
    ----------
    rotvec : np.ndarray
        3D rotation vector

    Returns
    -------
    np.ndarray
        3x3 rotation matrix

    _R: np.ndarray = np.eye(3) + sin * K + (1 - cos) * K @ K  # 原式
    choose (3,1) instead 3:    3 is vec, k.T == k;    (3,1) is matrix, k.T != k
    """
    if rot_vec3.shape[-1] == 4:
        return rot_vec3
    assert rot_vec3.shape[-1] == 3, f"Last dimension must be 3, but got {rot_vec3.shape}"
    lib = Lib(rot_vec3)
    is_torch = lib.__name__ == 'torch'

    # 计算旋转角度
    theta = Norm(rot_vec3, dim=-1, keepdim=True)  # (...,1)

    EPSILON = 1e-8
    mask = theta < EPSILON

    # 处理小角度情况
    K_small = skew_symmetric(rot_vec3)
    eye = lib.eye(3, dtype=rot_vec3.dtype)
    if is_torch:
        eye = eye.to(rot_vec3.device)
    R_small = eye + K_small  # 广播加法

    # 处理一般情况
    safe_theta = lib.where(mask, EPSILON * lib.ones_like(theta), theta)  # 避免除零
    k = rot_vec3 / safe_theta  # 单位向量

    K = skew_symmetric(k)
    k = k[..., None]  # 添加最后维度 (...,3,1)
    kkt = lib.matmul(k, lib.swapaxes(k, -1, -2))  # (...,3,3)

    cos_t = lib.cos(theta)[..., None]  # (...,1,1)
    sin_t = lib.sin(theta)[..., None]

    R_full = cos_t * eye + sin_t * K + (1 - cos_t) * kkt

    # 合并结果
    if is_torch:
        mask = mask.view(*mask.shape, 1, 1)
    else:
        mask = mask[..., None]

    ret = lib.where(mask, R_small, R_full)
    return ret


def RotMat_to_quat(R: TN) -> TN:
    """将3x3旋转矩阵转换为单位四元数 [w, x, y, z]，支持批量和PyTorch/NumPy"""
    if R.shape[-1] == 4:
        return R
    assert R.shape[-2:] == (3, 3), f"输入R的末两维必须为3x3，当前为{R.shape}"
    lib = Lib(R)  # 自动检测模块
    is_torch = lib.__name__ == 'torch'
    EPSILON = 1e-12  # 数值稳定系数

    # 计算迹，形状为(...)
    trace = lib.einsum('...ii->...', R)

    # 计算四个分量的平方（带数值稳定处理）
    q_sq = lib.stack([
        (trace + 1) / 4,
        (1 + 2 * R[..., 0, 0] - trace) / 4,
        (1 + 2 * R[..., 1, 1] - trace) / 4,
        (1 + 2 * R[..., 2, 2] - trace) / 4,
    ], axis=-1)

    q_sq = lib.maximum(q_sq, 0.0)  # 确保平方值非负

    # 找到最大分量的索引，形状(...)
    i = lib.argmax(q_sq, axis=-1)

    # 计算分母（带数值稳定处理）
    denoms = 4 * lib.sqrt(q_sq + EPSILON)  # 添加极小值防止sqrt(0)

    # 构造每个case的四元数分量
    cases = []
    for i_case in range(4):
        denom = denoms[..., i_case]  # 当前case的分母
        if i_case == 0:
            w = lib.sqrt(q_sq[..., 0] + EPSILON)  # 数值稳定
            x = (R[..., 2, 1] - R[..., 1, 2]) / denom
            y = (R[..., 0, 2] - R[..., 2, 0]) / denom
            z = (R[..., 1, 0] - R[..., 0, 1]) / denom
        elif i_case == 1:
            x = lib.sqrt(q_sq[..., 1] + EPSILON)
            w = (R[..., 2, 1] - R[..., 1, 2]) / denom
            y = (R[..., 0, 1] + R[..., 1, 0]) / denom
            z = (R[..., 0, 2] + R[..., 2, 0]) / denom
        elif i_case == 2:
            y = lib.sqrt(q_sq[..., 2] + EPSILON)
            w = (R[..., 0, 2] - R[..., 2, 0]) / denom
            x = (R[..., 0, 1] + R[..., 1, 0]) / denom
            z = (R[..., 1, 2] + R[..., 2, 1]) / denom
        else:  # i_case == 3
            z = lib.sqrt(q_sq[..., 3] + EPSILON)
            w = (R[..., 1, 0] - R[..., 0, 1]) / denom
            x = (R[..., 0, 2] + R[..., 2, 0]) / denom
            y = (R[..., 1, 2] + R[..., 2, 1]) / denom

        case = lib.stack([w, x, y, z], axis=-1)
        cases.append(case)

    # 合并所有情况并进行索引选择
    cases = lib.stack(cases, axis=0)
    if is_torch:
        index = i.reshape(1, *i.shape, 1).expand(1, *i.shape, 4)
        q = lib.gather(cases, dim=0, index=index).squeeze(0)
    else:
        # 构造NumPy兼容的索引
        index = i.reshape(1, *i.shape, 1)  # 添加新轴以对齐批量维度
        index = np.broadcast_to(index, (1,) + i.shape + (4,))  # 扩展至四元数维度
        q = np.take_along_axis(cases, index, axis=0).squeeze(0)  # 选择并压缩维度

    # 归一化处理（带数值稳定）
    norm = Norm(q, dim=-1, keepdim=True)
    ret = q / (norm + EPSILON)  # 防止除零
    return ret


def quat_rotAxis(arr: TN) -> TN: return RotMat_to_quat(Rodrigues(arr))


def register():
    ...


def unregister():
    ...


if __name__ == "__main__":
    # debug
    try:
        ...
    except ImportError:
        ...
