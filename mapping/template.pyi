"""generate from {armature} #, after editing you can contribute by https://github.com/AClon314/mocap-wrapper/issues/new?template=feature_request.md ♥️"""
from typing import Literal, Dict, get_args
TYPE_BODY = Literal{type_body}   # type: ignore
TYPE_HEAD = Literal[
    ...
]
TYPE_HANDS = Literal[
    ...
]
TYPE_BONES = Literal[TYPE_BODY, TYPE_HEAD, TYPE_HANDS]

BODY = get_args(TYPE_BODY)
HEAD = get_args(TYPE_HEAD)
HANDS = get_args(TYPE_HANDS)
BONES = BODY + HEAD + HANDS

from .smplx import TYPE_BONES as T_BONES
MAP: Dict[T_BONES, TYPE_BODY] = {map}   # type: ignore
MAP_R = {v: k for k, v in MAP.items()}
BONES_TREE = {bones_tree}  # type: ignore
from ..lib import TYPE_I18N
HELP: Dict[TYPE_I18N, str | None] = {
    'en_US': __doc__,
    'zh_HANS': """从`添加映射/Add Mapping`生成""",
}
