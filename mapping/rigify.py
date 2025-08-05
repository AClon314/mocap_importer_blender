"""Only FK mapping, need T-pose, and the armature is after generation from rigify blender addon.\n⚠️: You have to manually add bone `spine.004` in metarig to generate `spine_fk.004` in rig"""
from typing import Literal, Dict, get_args
...
...
...
...
...
TYPE_BODY = Literal[
    'root',
    'torso',
    'thigh_fk.L',
    'thigh_fk.R',
    'spine_fk.002',  # 001 control below, 002 control above
    'shin_fk.L',
    'shin_fk.R',
    'spine_fk.003',
    'foot_fk.L',
    'foot_fk.R',
    'spine_fk.004',  # add 1 more manually
    'toe_fk.L',
    'toe_fk.R',
    'neck',
    'shoulder.L',
    'shoulder.R',
    'head',
    'upper_arm_fk.L',
    'upper_arm_fk.R',
    'forearm_fk.L',
    'forearm_fk.R',
    'hand_fk.L',
    'hand_fk.R',
]   # breast
TYPE_HEAD = Literal['jaw_master', 'eye_master.L', 'eye_master.R']
TYPE_HANDS = Literal[
    'f_index', 'f_middle', 'f_pinky', 'f_ring', 'thumb',
]
TYPE_BONES = Literal[TYPE_BODY, TYPE_HEAD, TYPE_HANDS]

BODY = list(get_args(TYPE_BODY))
HEAD = list(get_args(TYPE_HEAD))
HANDS = list(get_args(TYPE_HANDS))
HANDS = [bone + '.L' for bone in HANDS] + [bone + '.R' for bone in HANDS]
BONES = BODY + HEAD + HANDS

from ..b import TYPE_I18N
HELP: Dict[TYPE_I18N, str | None] = {
    'en_US': __doc__,
    'zh_HANS': """仅FK映射，需要T形姿势，用于rigify生成后的骨架\n⚠️ 请手动在metarig添加`spine.004`，以在rig骨架中生成`spine_fk.004`""",
}
