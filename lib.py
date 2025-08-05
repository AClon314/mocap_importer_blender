"""
lib.py is a share lib that not rely on bpy module, include basic data class and calculate functions.
"""
import os
import sys
import importlib
import itertools
import numpy as np
from time import time
from functools import cache
from collections import UserDict
from .logger import Log
from types import ModuleType
from typing import Callable, Dict, Generator, Iterable, ParamSpec, Sequence, Literal, TypeVar, cast, get_args
INF = float('inf')
BATCH_SIZE = 1
DIR_SELF = os.path.dirname(__file__)
DIR_MAPPING = os.path.join(DIR_SELF, 'mapping')
MAPPING_TEMPLATE = os.path.join(DIR_MAPPING, 'template.pyi')
TYPE_MAPPING = Literal['smpl', 'smplx', 'rigify']
TYPE_MAPPING_KEYS = Literal['BONES', 'BODY', 'HANDS', 'HEAD']
TYPE_RUN = Literal['gvhmr', 'wilor']
_PS = ParamSpec("_PS")
_TV = TypeVar("_TV")
TYPE_PROP = Literal['body_pose', 'hand_pose', 'global_orient', 'betas', 'transl', 'bbox']
PROP_KEY = get_args(TYPE_PROP)
def get_major(L: Sequence[_TV]) -> _TV | None: return max(L, key=L.count) if L else None
def Axis(is_torch=False): return 'dim' if is_torch else 'axis'
@cache
def Map(Dir='mapping') -> Dict[TYPE_MAPPING, ModuleType]: return Mod(Dir=Dir)   # type: ignore
@cache
def Run(Dir='run') -> Dict[TYPE_RUN, ModuleType]: return Mod(Dir=Dir)   # type: ignore
# def cache(func: Callable[_PS, _TV]): return copy_args(func)(functools.cache(func))


def copy_args(func: Callable[_PS, _TV]):
    """Decorator does nothing and returning the casted original function"""
    def return_func(func: Callable[..., _TV]) -> Callable[_PS, _TV]:
        return cast(Callable[_PS, _TV], func)
    return return_func


def in_or_skip(part, full, pattern=''):
    """Check if `part` is in `full`, or if `part` is None/empty or `full` is None/empty`, return True. optionally Format-string with `pattern`."""
    if pattern:
        part = pattern.format(part) if part else None
        full = pattern.format(full) if full else None
    return (not part) or (not full) or (part in full)


def warn_or_return_first(L: list[_TV]) -> _TV:
    """Warn if more than one item in list, return the first item."""
    Len = len(L)
    if Len > 1:
        Log.warning(f'{Len} > 1', extra={'report': True, 'mouse': False})
    return L[0]


def format_sec(seconds: float):
    if seconds == INF:
        return '⏳'
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    return f"{s}s" if m == 0 else f"{m}:{s}"


def Mod(Dir='mapping'):
    files = os.listdir(os.path.join(DIR_SELF, Dir))
    pys = []
    mods: Dict[str, ModuleType] = {}
    for f in files:
        if f.endswith('.py') and f not in ['template.py', '__init__.py']:
            pys.append(f[:-3])
    for p in pys:
        mod = importlib.import_module(f'.{Dir}.{p}', package=__package__)
        mods[p] = mod
    return mods


def gen_calc():
    '''
    Intensive computing tasks that can be paused. return True when tasks finished.
    '''
    if Progress.PAUSE():
        return
    global BATCH_SIZE
    tick_prev = time()
    while True:
        Log.debug(f'gen_calc {len(Progress.selves)=} {len(GEN.queue)=} {BATCH_SIZE=}')
        try:
            for _ in range(BATCH_SIZE):
                next(GEN.chain)
        except StopIteration:
            return True
        now = time()
        delta = now - tick_prev - Progress.update_interval
        tick_prev = now
        if delta < Progress.update_interval // 2:
            BATCH_SIZE *= 2
        elif delta > Progress.update_interval:
            BATCH_SIZE = max(1, BATCH_SIZE // 2)
        else:
            break


class Generators:
    queue: list[Iterable] = []
    cache = None
    def debug(self): Log.debug(f'Gens: {len(self.queue)=}\t{self.__dict__=}')
    def insert(self, at=0, *gen: Generator): self.queue[at:at] = gen; self.cache = None
    def append(self, *gen: Generator): self.queue.extend(gen); self.cache = None
    def pop(self, at=-1): self.queue.pop(at); self.cache = None
    def remove(self, *gen: Iterable): [self.queue.remove(g) for g in gen]; self.cache = None
    def clear(self): self.queue.clear(); self.cache = None

    @property
    def chain(self):
        if self.cache:
            return self.cache
        self.cache = itertools.chain(*self.queue)
        self.queue = [self.cache]
        self.debug()
        return self.cache


GEN = Generators()


class Progress():
    selves: list['Progress'] = []
    update_interval = 0.5
    @property
    def _dur_change(self): return time() - self._tick_change
    @property
    def time(self): return time() - self.tick_start
    @property
    def active_time(self): return self._dur_run if self.pause else self._dur_run + self._dur_change
    @property
    def len(self): return self.Range.stop - self.Range.start
    @property
    def done(self): return self._current - self.Range.start
    @property
    def percent(self): return self.done / (self.len) if self.len != 0 else INF
    @property
    def rate(self): return self.done / self.active_time if self.active_time != 0 else INF
    @property
    def eta(self): return (self.Range.stop - self._current) / self.rate if self.rate != 0 else INF
    @property
    def out_range(self): return self._current < self.Range.start or self._current >= self.Range.stop
    def status(self): return f'{self.percent:.0%},{format_sec(self.eta)},{self.rate:.1f}{self.unit}/s'

    @classmethod
    def TIME(cls): return sum(s.time for s in cls.selves)
    @classmethod
    def ACTIVE_TIME(cls): return sum(s.active_time for s in cls.selves)
    @classmethod
    def LEN(cls): return sum(s.len for s in cls.selves)
    @classmethod
    def DONE(cls): return sum(s.done for s in cls.selves)
    @classmethod
    def PERCENT(cls): return cls.DONE() / cls.LEN() if cls.LEN() != 0 else INF
    @classmethod
    def RATE(cls): return cls.DONE() / cls.ACTIVE_TIME() if cls.ACTIVE_TIME() != 0 else INF
    @classmethod
    def ETA(cls): return sum(s.eta for s in cls.selves)
    @classmethod
    def STATUS(cls): return f'{cls.PERCENT():.0%},{format_sec(cls.ETA())},{cls.RATE():.0f}/s' if cls.LEN() > 0 else ''

    @classmethod
    def PAUSE(cls, set: bool | None = None):
        if set is None:
            return any(s.pause for s in cls.selves)
        for s in cls.selves:
            s.pause = set
        return set

    @property
    def current(self): return self._current
    @property
    def pause(self): return self._pause
    @current.setter
    def current(self, value: int): self.update(set=value)

    @pause.setter
    def pause(self, b: bool):
        self._pause = b
        if b:
            self._dur_run += self._dur_change
        self._tick_change = time()

    def __init__(self, *Range: int, unit: str = 'frame', msg='', pause=False):
        self.Range = range(*Range) if Range else range(100)
        self.msg = msg
        self.unit = unit
        self._current = self.Range.start
        self.tick_start = self.tick_update = self._tick_change = time()
        self._dur_run = 0
        self._pause = pause
        self.__class__.selves.append(self)

    def update(self, step: int | None = None, set: int | None = None):
        """
        Args:
            step: step to increase, if None, use `self.step`
            set: Any value between min and max as set in Range=...
        """
        if self.out_range:
            self.__class__.selves.remove(self) if self in self.__class__.selves else None
            return self._current
        if set is not None:
            self._current = set
        elif step is None:
            self._current += self.Range.step
        else:
            self._current += step
        now = time()
        # if now - self.tick_update >= self.update_interval:
        #     # wm.progress_update(self.current)
        #     self.tick_update = now
        if self._current >= self.Range.stop:
            self.pause = True
        return self._current


class MotionData(UserDict):
    """
    usage:
    ```python
    # __call__ is filter
    data(mapping='smplx', run='gvhmr', prop='trans', coord='global').values()[0]
    ```
    """

    def keys(self) -> list[str]: return list(super().keys())
    def values(self) -> list[np.ndarray]: return list(super().values())
    def __bool__(self): return bool(self.keys())

    def __init__(self, /, *args, npz: str | None = None, lazy=False, **kwargs):
        """
        Inherit from dict
        Args:
            npz (str, Path, optional): npz file path.
            lazy (bool, optional): if True, do NOT load npz file.
        """
        super().__init__(*args, **kwargs)
        self.Slice = slice(None)
        self.npz = npz
        if not lazy and npz and os.path.exists(npz):
            self.update(np.load(npz, allow_pickle=True))

    def __call__(
        self,
        *prop: TYPE_PROP,
        mapping: TYPE_MAPPING | None = None,
        run: TYPE_RUN | None = None,
        who: str | int | None = None,
        Slice: slice | None = None,
    ):
        # Log.debug(f'self.__dict__={self.__dict__}')
        MD = MotionData(npz=self.npz, lazy=True)
        if isinstance(who, int):
            who = f'person{who}'
        if Slice:
            self.Slice = Slice

        for k, v in self.items():
            is_in = [in_or_skip(args, k, ';{};') for args in [mapping, run, who, *prop]]
            is_in = all(is_in)
            if is_in:
                MD[k] = v
        return MD

    def distinct(self, col_num: int):
        """
        Args:
            col_num (int): 0 for mapping, 1 for run, 2 for key, 3 for person, 4 for coord
            literal : filter keys by Literal. Defaults to None.

        """
        L: list[str] = []
        for k in self.keys():
            keys = k.split(';')
            col_name = keys[col_num]
            if col_name not in L:
                L.append(col_name)
        return L

    @property
    def mappings(self) -> list[TYPE_MAPPING]: return self.distinct(0)  # type: ignore
    @property
    def runs(self) -> list[TYPE_RUN]: return self.distinct(1)   # type: ignore
    @property
    def whos(self): return self.distinct(2)
    @property
    def begins(self): return [int(x) for x in self.distinct(3)]

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
    @property
    def begin(self): return warn_or_return_first(self.begins)
    def prop(self, col=0): return self.props(col)[0]

    @property
    def value(self):
        """same as:
        ```python
        return self.values()[0]
        ```"""
        v = warn_or_return_first(self.values())
        try:
            return v[self.Slice]
        except Exception as e:
            # Log.warning(e, exc_info=e)
            return v

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


def bone_to_dict(bone, whitelist: Sequence[str] | None = None):
    """bone to dict, Recursive calls to this function form a tree

    Args:
        whitelist (Sequence[str], optional): list of bone names to include. Defaults to None.
    """
    return {child.name: bone_to_dict(child) for child in bone.children if in_or_skip(child.name, whitelist)}


def keys_BFS(d: dict, wrap=False, whitelist: Sequence[str] | None = None):
    """
    sort keys of dict by BFS (Breadth-First Search) algorithm.

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
        current_level = []
        next_queue = []
        for current_dict in Q:
            current_level.extend(current_dict.keys())  # 收集当前字典的所有键到当前层级
            next_queue.extend(current_dict.values())  # 收集当前字典的所有子字典到下一层队列
        current_level = [k for k in current_level if in_or_skip(k, whitelist)]
        ret.append(current_level) if wrap else ret.extend(current_level)
        Q = next_queue  # 更新队列为下一层级的子字典列表
        deep += 1
    return ret


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


def quat(xyz: np.ndarray) -> np.ndarray:
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


def euler(wxyz: np.ndarray) -> np.ndarray:
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


def Norm(arr: np.ndarray, dim: int = -1, keepdim: bool = True) -> np.ndarray:
    """计算范数，支持批量输入"""
    lib = Lib(arr)
    is_torch = lib.__name__ == 'torch'
    if is_torch:
        return lib.norm(arr, dim=dim, keepdim=keepdim)
    else:
        return lib.linalg.norm(arr, axis=dim, keepdims=keepdim)


def skew_symmetric(v: np.ndarray) -> np.ndarray:
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


def Rodrigues(rot_vec3: np.ndarray) -> np.ndarray:
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


def rotMat_to_quat(R: np.ndarray) -> np.ndarray:
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


def quat_rotAxis(arr: np.ndarray) -> np.ndarray: return rotMat_to_quat(Rodrigues(arr))


def quat_to_rotMat(quats):
    original_shape = quats.shape
    N = np.prod(original_shape[:-1])  # 所有维度的乘积，除最后一个维度
    arr = quats.reshape(N, 4)  # 转换为 (N, 4)

    # 提取四元数分量
    w, x, y, z = arr.T  # 每个分量形状为 (N,)

    # 构建旋转矩阵
    R = np.array([
        [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2]
    ])  # R.shape == (3, 3, N)

    R = R.transpose(2, 0, 1)  # (N, 3, 3)
    return R.reshape(*original_shape[:-1], 3, 3)  # (..., 3, 3)


def euler_to_rotMat(eulers):
    original_shape = eulers.shape
    N = np.prod(original_shape[:-1])  # 所有维度的乘积，除最后一个维度
    arr = eulers.reshape(N, 3)  # 转换为 (N, 3)

    roll, pitch, yaw = arr.T  # 每个角度形状为 (N,)

    cos_r = np.cos(roll)
    sin_r = np.sin(roll)
    cos_p = np.cos(pitch)
    sin_p = np.sin(pitch)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)

    R = np.array([
        [cos_y * cos_p,
         cos_y * sin_p * sin_r - sin_y * cos_r,
         cos_y * sin_p * cos_r + sin_y * sin_r],
        [sin_y * cos_p,
         sin_y * sin_p * sin_r + cos_y * cos_r,
         sin_y * sin_p * cos_r - cos_y * sin_r],
        [-sin_p,
         cos_p * sin_r,
         cos_p * cos_r]
    ])  # R.shape == (3, 3, N)

    R = R.transpose(2, 0, 1)  # (N, 3, 3)
    return R.reshape(*original_shape[:-1], 3, 3)  # (..., 3, 3)


def rotMat(arr: np.ndarray):
    """quat/euler to rotation matrix"""
    if arr.shape[-1] == 4:
        return quat_to_rotMat(arr)
    elif arr.shape[-1] == 3:
        return euler_to_rotMat(arr)
    else:
        raise ValueError(f"Last dimension must be 3 or 4, but got {arr.shape[-1]}")


def mano_to_smplx(
    body_pose: 'np.ndarray',
    hand_pose: 'np.ndarray',
    global_orient: 'np.ndarray',
    base_frame=0,
    is_left=False,
    left_fix=False,
):
    """
    hand to body pose:
    https://github.com/VincentHu19/Mano2Smpl-X/blob/main/mano2smplx.py

    Args:
        - body_pose (np.array): SMPLX's local pose (22, 3 or 4)
        - hand_pose (np.array): MANO's local pose (15, 3 or 4)
        - global_orient (np.array): **hand**'s global orientation (?, 3 or 4)
        - base_frame (int): the frame to use as a reference for relative rotation
        - is_left (bool): whether the hand is left or right
        - left_fix (bool): whether to mirror the left hand pose, for hamer

    ---
    Returns
    ---
        - wrist_pose (np.array): SMPLX's local pose (3 or 4), re-calculate based on `body_write` & hans's `global_orient`
        - hand_pose (np.array): SMPLX's local pose (15, 3 or 4), mirrored if is_left
    """
    Log.debug(f'{body_pose.shape=}')
    Log.debug(f'{global_orient.shape=}')

    M = np.diag([-1, 1, 1])  # Preparing for the left hand switch
    global_orient = rotMat(global_orient)
    if is_left:
        idx_elbow = 19  # +1 offset due to root bone
        if left_fix:
            global_orient = M @ global_orient @ M  # mirror
            hand_pose = rotMat(hand_pose)
            hand_pose = M @ hand_pose @ M
            hand_pose = rotMat_to_quat(hand_pose)
    else:
        idx_elbow = 20
    body_elbow = body_pose[:, idx_elbow]
    body_elbow = rotMat(body_elbow)

    if body_elbow.shape[0] < global_orient.shape[0]:
        # body < hand, padding body from the last frame
        pad_size = global_orient.shape[0] - body_elbow.shape[0]
        body_elbow = np.pad(body_elbow, ((0, pad_size), (0, 0), (0, 0)), mode='edge')
    else:
        # body > hand, clip body
        body_elbow = body_elbow[:global_orient.shape[0]]

    wrist_pose = np.linalg.inv(body_elbow) @ global_orient
    wrist_pose = rotMat_relative(wrist_pose, base_frame=base_frame)
    wrist_pose = rotMat_to_quat(wrist_pose)
    Log.debug(f'{wrist_pose.shape=}')
    return wrist_pose, hand_pose


def rotMat_relative(rotMats: 'np.ndarray', base_frame=0):
    """
    重新计算相对于指定帧的相对旋转

    Args:
        rotMats (np.array): 旋转矩阵数组，形状为 (frames, 3, 3)
        base_frame (int): 作为零旋转参考的帧索引

    Returns:
        np.array: 相对于参考帧的旋转矩阵数组
    """
    zero_rot = rotMats[base_frame]
    zero_rot_inv = np.linalg.inv(zero_rot)
    results = np.zeros_like(rotMats)
    for i in range(rotMats.shape[0]):
        results[i] = zero_rot_inv @ rotMats[i]
    return results
