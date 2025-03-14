from typing import Literal, get_args


...
...
...

TYPE_BONES_BODY = Literal[
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
    'right_shoulder',  # 'jaw', 'left_eye_smplhf', 'right_eye_smplhf',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]
TYPE_BONES_HANDS = Literal[
    'left_index1', 'left_middle1', 'left_pinky1', 'left_ring1', 'left_thumb1',
    'right_index1', 'right_middle1', 'right_pinky1', 'right_ring1', 'right_thumb1',
    'left_index2', 'left_middle2', 'left_pinky2', 'left_ring2', 'left_thumb2',
    'right_index2', 'right_middle2', 'right_pinky2', 'right_ring2', 'right_thumb2',
    'left_index3', 'left_middle3', 'left_pinky3', 'left_ring3', 'left_thumb3',
    'right_index3', 'right_middle3', 'right_pinky3', 'right_ring3', 'right_thumb3',
]

SMPLX_BODY = get_args(TYPE_BONES_BODY)
SMPLX_HANDS = get_args(TYPE_BONES_HANDS)

SMPLX_DICT = {
    "root": {
        "pelvis": {
            "left_hip": {
                "left_knee": {
                    "left_ankle": {
                        "left_foot": {}
                    }
                }
            },
            "right_hip": {
                "right_knee": {
                    "right_ankle": {
                        "right_foot": {}
                    }
                }
            },
            "spine1": {
                "spine2": {
                    "spine3": {
                        "neck": {
                            "head": {
                                "jaw": {},
                                "left_eye_smplhf": {},
                                "right_eye_smplhf": {}
                            }
                        },
                        "left_collar": {
                            "left_shoulder": {
                                "left_elbow": {
                                    "left_wrist": {
                                        "left_index1": {
                                            "left_index2": {
                                                "left_index3": {}
                                            }
                                        },
                                        "left_middle1": {
                                            "left_middle2": {
                                                "left_middle3": {}
                                            }
                                        },
                                        "left_pinky1": {
                                            "left_pinky2": {
                                                "left_pinky3": {}
                                            }
                                        },
                                        "left_ring1": {
                                            "left_ring2": {
                                                "left_ring3": {}
                                            }
                                        },
                                        "left_thumb1": {
                                            "left_thumb2": {
                                                "left_thumb3": {}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "right_collar": {
                            "right_shoulder": {
                                "right_elbow": {
                                    "right_wrist": {
                                        "right_index1": {
                                            "right_index2": {
                                                "right_index3": {}
                                            }
                                        },
                                        "right_middle1": {
                                            "right_middle2": {
                                                "right_middle3": {}
                                            }
                                        },
                                        "right_pinky1": {
                                            "right_pinky2": {
                                                "right_pinky3": {}
                                            }
                                        },
                                        "right_ring1": {
                                            "right_ring2": {
                                                "right_ring3": {}
                                            }
                                        },
                                        "right_thumb1": {
                                            "right_thumb2": {
                                                "right_thumb3": {}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
