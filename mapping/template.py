"""generate from `get bones info` operator"""
from .smplx import TYPE_BONES as T_BONES
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

MAP: Dict[T_BONES, TYPE_BODY] = {}   # type: ignore
BONES_TREE = {}
