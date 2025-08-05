"""update version of smpl. With more 3 head bones and 2*5*3=30 hands bones"""
from typing import Dict, Literal, get_args
...
...
...
...
...
TYPE_BODY = Literal[
    'root',
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',  # head
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]   # len = 25
TYPE_HEAD = Literal['jaw', 'left_eye_smplhf', 'right_eye_smplhf']
TYPE_HANDS = Literal[
    'index', 'middle', 'pinky', 'ring', 'thumb'  # https://github.com/otaheri/MANO/blob/master/mano/joints_info.py#L29C2-L43C15
]
TYPE_BONES = Literal[TYPE_BODY, TYPE_HEAD, TYPE_HANDS]

BODY = list(get_args(TYPE_BODY))
HEAD = list(get_args(TYPE_HEAD))
HANDS = list(get_args(TYPE_HANDS))
HANDS = [f'{lr}_{bone}{i}' for lr in ['left', 'right'] for bone in HANDS for i in range(1, 4)]
BONES = BODY + HEAD + HANDS

from ..b import TYPE_I18N
HELP: Dict[TYPE_I18N, str | None] = {
    'en_US': __doc__,
    'zh_HANS': """SMPL 的更新版本。增加了 3 块头骨和 2*5*3=30 块手骨""",
}
