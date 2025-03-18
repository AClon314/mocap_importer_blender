"""generate from `get bones info` operator"""
from typing import Literal, Dict, get_args
TYPE_BODY = Literal{}   # type: ignore
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
MAP: Dict[T_BONES, TYPE_BODY] = {}   # type: ignore
BONES_TREE = {}
from ..lib import TYPE_I18N
HELP: Dict[TYPE_I18N, str | None] = {
    'en_US': __doc__,
    'zh_HANS': """从`get bones info`操作生成""",
}
