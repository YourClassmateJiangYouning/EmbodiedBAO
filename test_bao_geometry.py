"""Pure-Python regression checks for EmbodiedBAO coordinate conventions."""

from __future__ import annotations

import unittest

import numpy as np

from environment import (
    BAOEnv,
    HAND_LOCAL_REACH,
    HAND_LOCAL_REST,
    MOVE_STEP,
    ROBOT_START_POS,
    _forward_vector,
    _isaac_to_user_pos,
    _user_to_isaac_pos,
)


GROUND_OFFSET = 1.0442


class _FakeRoot:
    def __init__(self) -> None:
        self.isaac = np.zeros(3, dtype=float)

    def get_world_poses(self) -> tuple:
        return np.array([self.isaac]), np.zeros((1, 4))

    def set_world_poses(self, positions: np.ndarray, orientations: np.ndarray) -> None:
        self.isaac = np.asarray(positions[0], dtype=float).copy()


def _make_env() -> BAOEnv:
    env = object.__new__(BAOEnv)
    env.task_dict = {}
    env.robot_root = _FakeRoot()
    env._robot_ground_offset = GROUND_OFFSET
    env._robot_yaw = 0.0
    return env


class CoordinateRegressionTest(unittest.TestCase):
    def test_user_isaac_round_trip(self) -> None:
        user = np.array([1.5, 1.2, -0.5], dtype=float)
        np.testing.assert_allclose(_isaac_to_user_pos(_user_to_isaac_pos(user)), user)

    def test_ground_offset_is_added_only_once(self) -> None:
        env = _make_env()
        env._set_robot_pose(ROBOT_START_POS, 0.0)
        ys = []
        for _ in range(10):
            root = env._root_position()
            ys.append(float(root[1]))
            env._set_robot_pose(
                root + _forward_vector(np.radians(env._robot_yaw)) * MOVE_STEP,
                env._robot_yaw,
            )
        np.testing.assert_allclose(ys, np.zeros(10), atol=1e-9)

    def test_analytic_hand_heights_are_ground_relative(self) -> None:
        env = _make_env()
        env._set_robot_pose(ROBOT_START_POS, 0.0)
        env.reaching = False
        self.assertAlmostEqual(float(env._analytic_hand_position()[1]), HAND_LOCAL_REST[1])
        env.reaching = True
        self.assertAlmostEqual(
            float(env._analytic_hand_position()[1]), HAND_LOCAL_REACH[1]
        )


if __name__ == "__main__":
    unittest.main()
