import os, sys, numpy as np, pytest

OLD = np.array([0, 0, 1])  # smplx pelvis definition
NEW = np.array([0, 1, 0])  # rigify torso edit facing vector
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib import change_coord, euler, quat, quat_1, _raw_multi_quat, delta_quat, Log


@pytest.mark.parametrize(
    "a, b",
    [
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0])),
        *[(np.random.random(4), np.random.random(4)) for _ in range(100)],
    ],
)
def test_quat_calc(a, b):
    """make sure `multi_quat` & `quat_1` is right"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    a_b = _raw_multi_quat(a, b)
    _b = _raw_multi_quat(quat_1(a), a_b)
    Log.debug(f"{locals()=}")
    assert np.allclose(b, _b)


def test_pelvis_torso():
    """make sure NEW = dt * OLD * dt^-1, dt=delta_quat(From=OLD,To=NEW)"""
    # TODO: learn math 💪
    q_old = quat(OLD)
    delta = delta_quat(OLD, NEW)
    q_new = _raw_multi_quat(delta, _raw_multi_quat(q_old, quat_1(delta)))
    Log.debug(f"{locals()=}")
    assert np.allclose(NEW, euler(q_new), atol=1e-6)


@pytest.mark.parametrize(
    "rotate",
    [
        # np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64),
        np.array([np.random.random(4) for _ in range(100)]),
    ],
)
def test_rotate_offset(rotate):
    """make sure numpy broadcast is ok"""
    # make rotate last dim normalize
    rotate[:, -1] /= np.linalg.norm(rotate[:, -1])
    new_rotate = change_coord(rotate, OLD, NEW)
    Log.debug(f"{rotate.shape=}")
    assert not np.allclose(rotate, new_rotate, atol=1e-6)
