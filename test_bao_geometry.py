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
    _check_wall_collision,
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
    env._channel_width = 0.38
    env._articulation_ok = False
    env.hand_xform = None
    env.reaching = False
    env.raised_arm = None
    env._camera_yaw_offset = 0.0
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
            env._set_robot_pose(root + np.array([MOVE_STEP, 0.0, 0.0]), env._robot_yaw)
        np.testing.assert_allclose(ys, np.zeros(10), atol=1e-9)

    def test_translations_follow_robot_facing(self) -> None:
        env = _make_env()
        env._set_robot_pose(ROBOT_START_POS, 90.0)
        self.assertAlmostEqual(float(env._robot_yaw), 90.0)

        env._apply_action("forward")
        pos = env._root_position()
        self.assertAlmostEqual(pos[0], 1.50)
        self.assertAlmostEqual(pos[2], -MOVE_STEP)

        env._apply_action("backward")
        pos = env._root_position()
        self.assertAlmostEqual(pos[0], 1.50)
        self.assertAlmostEqual(pos[2], 0.0)

        env._apply_action("left")
        pos = env._root_position()
        self.assertAlmostEqual(pos[0], ROBOT_START_POS[0] - MOVE_STEP)
        self.assertAlmostEqual(pos[2], 0.0)

        env._apply_action("right")
        pos = env._root_position()
        self.assertAlmostEqual(pos[0], ROBOT_START_POS[0])
        self.assertAlmostEqual(env.get_torso_rotation(), 90.0)

    def test_collision_bool_and_position_api(self) -> None:
        env = _make_env()
        env._set_robot_pose(np.array([1.95, 0.0, 0.0]), 0.0)
        self.assertTrue(env.check_collision_with_wall())
        point = env.get_collision_position()
        self.assertIsNotNone(point)
        self.assertEqual(len(point), 3)

    def test_channel_width_changes_analytic_collision(self) -> None:
        root = np.array([1.95, 0.0, 0.0], dtype=float)
        self.assertIsNotNone(
            _check_wall_collision(root, 0.0, channel_width=0.38)
        )
        self.assertIsNone(
            _check_wall_collision(root, 0.0, channel_width=0.60)
        )

    def test_analytic_hand_heights_are_ground_relative(self) -> None:
        env = _make_env()
        env._set_robot_pose(ROBOT_START_POS, 0.0)
        env.reaching = False
        self.assertAlmostEqual(float(env._analytic_hand_position()[1]), HAND_LOCAL_REST[1])
        env.reaching = True
        self.assertAlmostEqual(
            float(env._analytic_hand_position()[1]), HAND_LOCAL_REACH[1]
        )

    def test_reach_follows_body_forward(self) -> None:
        env = _make_env()
        env._set_robot_pose(np.array([1.96, 0.0, 0.0]), 0.0)
        env.reaching = True
        hand = env.get_hand_position()
        self.assertAlmostEqual(float(hand[0]), 2.40)
        self.assertAlmostEqual(float(hand[1]), 1.20)
        self.assertAlmostEqual(float(hand[2]), 0.24)

        env._set_robot_pose(np.array([1.96, 0.0, 0.0]), 90.0)
        env.reaching = True
        hand = env.get_hand_position()
        self.assertAlmostEqual(float(hand[0]), 2.20)
        self.assertAlmostEqual(float(hand[1]), 1.20)
        self.assertAlmostEqual(float(hand[2]), -0.44)

    def test_side_arm_can_touch_ball_while_sideways(self) -> None:
        env = _make_env()
        env._set_robot_pose(np.array([1.96, 0.0, 0.0]), 90.0)
        env._apply_action("raise_right_arm")
        hand = env.get_hand_position()
        self.assertAlmostEqual(float(hand[0]), 2.40)
        self.assertAlmostEqual(float(hand[1]), 1.20)
        self.assertAlmostEqual(float(hand[2]), 0.0)
        self.assertTrue(env.check_success())

        env._set_robot_pose(np.array([1.96, 0.0, 0.0]), -90.0)
        env._apply_action("raise_left_arm")
        hand = env.get_hand_position()
        self.assertAlmostEqual(float(hand[0]), 2.40)
        self.assertAlmostEqual(float(hand[2]), 0.0)
        self.assertTrue(env.check_success())

    def test_camera_look_actions_change_yaw_offset(self) -> None:
        env = _make_env()
        env._apply_action("look_left")
        self.assertAlmostEqual(env._camera_yaw_offset, 30.0)
        env._apply_action("look_right")
        self.assertAlmostEqual(env._camera_yaw_offset, 0.0)


if __name__ == "__main__":
    unittest.main()
