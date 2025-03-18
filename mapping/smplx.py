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
    'left_hand',    # unsed in gvhmr
    'right_hand',
]
TYPE_HEAD = Literal['jaw', 'left_eye_smplhf', 'right_eye_smplhf']
TYPE_HANDS = Literal[
    'left_index1', 'left_middle1', 'left_pinky1', 'left_ring1', 'left_thumb1',
    'right_index1', 'right_middle1', 'right_pinky1', 'right_ring1', 'right_thumb1',
    'left_index2', 'left_middle2', 'left_pinky2', 'left_ring2', 'left_thumb2',
    'right_index2', 'right_middle2', 'right_pinky2', 'right_ring2', 'right_thumb2',
    'left_index3', 'left_middle3', 'left_pinky3', 'left_ring3', 'left_thumb3',
    'right_index3', 'right_middle3', 'right_pinky3', 'right_ring3', 'right_thumb3',
]
TYPE_BONES = Literal[TYPE_BODY, TYPE_HEAD, TYPE_HANDS]

BODY = get_args(TYPE_BONES)
HEAD = get_args(TYPE_HEAD)
HANDS = get_args(TYPE_HANDS)
BONES = BODY + HEAD + HANDS

MAP: Dict[TYPE_BONES, TYPE_BONES] = {k: k for k in BODY}

BONES_TREE = {
    'root': {'pelvis': {'left_hip': {'left_knee': {'left_ankle': {'left_foot': {}}}},
                        'right_hip': {'right_knee': {'right_ankle': {'right_foot': {}}}},
                        'spine1': {'spine2': {'spine3': {'neck': {'head': {'jaw': {},
                                                                           'left_eye_smplhf': {},
                                                                           'right_eye_smplhf': {}}},
                                                         'left_collar': {'left_shoulder': {'left_elbow': {'left_wrist': {'left_index1': {'left_index2': {'left_index3': {}}},
                                                                                                                         'left_middle1': {'left_middle2': {'left_middle3': {}}},
                                                                                                                         'left_pinky1': {'left_pinky2': {'left_pinky3': {}}},
                                                                                                                         'left_ring1': {'left_ring2': {'left_ring3': {}}},
                                                                                                                         'left_thumb1': {'left_thumb2': {'left_thumb3': {}}}}}}},
                                                         'right_collar': {'right_shoulder': {'right_elbow': {'right_wrist': {'right_index1': {'right_index2': {'right_index3': {}}},
                                                                                                                             'right_middle1': {'right_middle2': {'right_middle3': {}}},
                                                                                                                             'right_pinky1': {'right_pinky2': {'right_pinky3': {}}},
                                                                                                                             'right_ring1': {'right_ring2': {'right_ring3': {}}},
                                                                                                                             'right_thumb1': {'right_thumb2': {'right_thumb3': {}}}}}}}}}}}}
}

from .. import lib
HELP: Dict[lib.TYPE_I18N, str | None] = {
    'en_US': __doc__,
    'zh_HANS': """SMPL 的更新版本。增加了 3 块头骨和 2*5*3=30 块手骨""",
}
