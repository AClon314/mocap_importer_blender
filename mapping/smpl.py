"""smpl bones struct with 24 body bones"""
from .smplx import TYPE_BONES as T_BONES, BODY as X_BODY  # BONES as X_BONES
from typing import Literal, Dict, get_args
...
...
...
...
TYPE_BODY = Literal[
    'm_avg_root',
    'm_avg_Pelvis',
    'm_avg_L_Hip',
    'm_avg_R_Hip',
    'm_avg_Spine1',
    'm_avg_L_Knee',
    'm_avg_R_Knee',
    'm_avg_Spine2',
    'm_avg_L_Ankle',
    'm_avg_R_Ankle',
    'm_avg_Spine3',
    'm_avg_L_Foot',
    'm_avg_R_Foot',
    'm_avg_Neck',
    'm_avg_L_Collar',
    'm_avg_R_Collar',
    'm_avg_Head',
    'm_avg_L_Shoulder',
    'm_avg_R_Shoulder',
    'm_avg_L_Elbow',
    'm_avg_R_Elbow',
    'm_avg_L_Wrist',
    'm_avg_R_Wrist',
    'm_avg_L_Hand',
    'm_avg_R_Hand',
]
# TYPE_HEAD = Literal[
#     ...
# ]
# TYPE_HANDS = Literal[
#     ...
# ]
# TYPE_BONES = Literal[TYPE_BODY, TYPE_HEAD, TYPE_HANDS]
TYPE_BONES = TYPE_BODY

BODY = get_args(TYPE_BODY)
# HEAD = get_args(TYPE_HEAD)
# HANDS = get_args(TYPE_HANDS)
# BONES = BODY + HEAD + HANDS
BONES = BODY

MAP: Dict[T_BONES, TYPE_BONES] = {k: v for k, v in zip(X_BODY, BODY)}
MAP_R = {v: k for k, v in MAP.items()}
BONES_TREE = {
    'm_avg_root': {'m_avg_Pelvis': {'m_avg_L_Hip': {'m_avg_L_Knee': {'m_avg_L_Ankle': {'m_avg_L_Foot': {}}}},
                                    'm_avg_R_Hip': {'m_avg_R_Knee': {'m_avg_R_Ankle': {'m_avg_R_Foot': {}}}},
                                    'm_avg_Spine1': {'m_avg_Spine2': {'m_avg_Spine3': {'m_avg_Neck': {'m_avg_Head': {}},
                                                                                       'm_avg_L_Collar': {'m_avg_L_Shoulder': {'m_avg_L_Elbow': {'m_avg_L_Wrist': {'m_avg_L_Hand': {}}}}},
                                                                                       'm_avg_R_Collar': {'m_avg_R_Shoulder': {'m_avg_R_Elbow': {'m_avg_R_Wrist': {'m_avg_R_Hand': {}}}}}}}}}}
}

from ..lib import TYPE_I18N
HELP: Dict[TYPE_I18N, str | None] = {
    'en_US': __doc__,
    'zh_HANS': """有24根骨骼的SMPL骨骼结构""",
}
