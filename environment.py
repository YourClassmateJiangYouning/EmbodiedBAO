"""EmbodiedBAO environment for NVIDIA Isaac Sim.

The benchmark recreates the psychological "Body-as-Obstacle" (BAO) task in a
4m x 4m room.  World coordinates follow the user-specified convention:

    x : forward axis (scene spans x in [0, 4]; the robot starts at x < 2)
    y : up axis (the ground plane is y = 0)
    z : lateral axis (scene spans z in [-2, 2])

The transparent acrylic wall sits on the plane x = 2.0 and spans the whole
scene.  A vertical channel of width 0.38 m is centred at z = 0 and runs from
the ground to the top of the wall, so the only way to reach the rear half of
the scene is to pass sideways through the channel (shoulder width 0.57 m does
not fit frontally, while torso thickness 0.22 m does).

The environment follows MirrorBench's interaction pattern: the robot is moved
kinematically (world-pose teleports), every action is gated by an analytic
collision check against the wall, and the head camera is a 1024x1024 RGB
camera that follows the robot root.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.prims import XFormPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.sensors.camera import Camera
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade
    import carb

    _HAS_ISAAC_SIM = True
except Exception:  # pragma: no cover - exercised only outside Isaac Sim
    _HAS_ISAAC_SIM = False


# ---------------------------------------------------------------------------
# Scene constants (metres / degrees)
# ---------------------------------------------------------------------------

SCENE_SIZE = 4.0
GROUND_THICKNESS = 0.02

WALL_X = 2.0
WALL_HEIGHT = 2.0
WALL_THICKNESS = 0.02
CHANNEL_WIDTH = 0.38
CHANNEL_HALF_WIDTH = CHANNEL_WIDTH / 2.0
PANEL_WIDTH = (SCENE_SIZE - CHANNEL_WIDTH) / 2.0  # 1.81 m per panel

ROBOT_START_POS = np.array([1.5, 0.0, 0.0], dtype=float)
ROBOT_START_YAW_DEG = 0.0

TARGET_POS = np.array([2.4, 1.737, 0.0], dtype=float)
TARGET_RADIUS = 0.04  # 8 cm diameter
SUCCESS_DISTANCE = 0.03

MOVE_STEP = 0.05  # 5 cm
TURN_STEP_DEG = 15.0

# H1 kinematic constants (used for analytic collision checks and fallback).
ROBOT_SHOULDER_WIDTH = 0.57
ROBOT_TORSO_THICKNESS = 0.22
ROBOT_BODY_CENTER_Y = 0.9
ROBOT_BODY_HALF_HEIGHT = 0.9
ROBOT_HEAD_HEIGHT = 1.55
ARM_REACH = 0.34
MODEL_YAW_OFFSET_DEG = 0.0
ARM_HANG_SHOULDER_PITCH_RAD = 0.0
ARM_HANG_ELBOW_PITCH_RAD = 1.57

# Analytic end-effector offsets in the robot frame (x forward, y up, z right).
HAND_LOCAL_REST = np.array([0.10, 0.95, 0.24], dtype=float)
HAND_LOCAL_REACH = np.array([0.44, 1.73, 0.24], dtype=float)  # +34 cm forward

REACH_SHOULDER_PITCH_RAD = -1.35
REACH_ELBOW_PITCH_RAD = 0.0

ACTIONS = [
    "forward",
    "backward",
    "left",
    "right",
    "turn_left",
    "turn_right",
    "reach",
    "retreat",
]


# ---------------------------------------------------------------------------
# Pure geometry helpers (unit-testable without Isaac Sim)
# ---------------------------------------------------------------------------


def _rotate_xz(vec: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Rotate a 3D vector around the up axis (+y) by yaw_rad."""
    c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array([x * c + z * s, y, -x * s + z * c], dtype=float)


def _forward_vector(yaw_rad: float) -> np.ndarray:
    """Unit vector pointing in the robot's facing direction."""
    return np.array([np.cos(yaw_rad), 0.0, -np.sin(yaw_rad)], dtype=float)


def _user_to_isaac_pos(pos: np.ndarray) -> np.ndarray:
    """Map a user-frame position (y up) to Isaac Sim (z up): swap y and z."""
    p = np.asarray(pos, dtype=float)
    return np.array([p[0], p[2], p[1]], dtype=float)


def _isaac_to_user_pos(pos: np.ndarray) -> np.ndarray:
    """Map an Isaac Sim position (z up) back to the user frame (y up)."""
    p = np.asarray(pos, dtype=float)
    return np.array([p[0], p[2], p[1]], dtype=float)


def _user_to_isaac_scale(scale: np.ndarray) -> np.ndarray:
    """Swap the y/z components of a scale vector for Isaac Sim."""
    s = np.asarray(scale, dtype=float)
    return np.array([s[0], s[2], s[1]], dtype=float)


def _panel_boxes() -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return (center, half-extents) of the two wall panels."""
    z_center = PANEL_WIDTH / 2.0 + CHANNEL_HALF_WIDTH
    half = np.array(
        [WALL_THICKNESS / 2.0, WALL_HEIGHT / 2.0, PANEL_WIDTH / 2.0], dtype=float
    )
    return [
        (np.array([WALL_X, WALL_HEIGHT / 2.0, -z_center], dtype=float), half.copy()),
        (np.array([WALL_X, WALL_HEIGHT / 2.0, z_center], dtype=float), half.copy()),
    ]


def _robot_body_aabb(
    root_pos: np.ndarray, yaw_rad: float
) -> Tuple[np.ndarray, np.ndarray]:
    """World-space AABB of the robot body box (torso + shoulders)."""
    c, s = abs(float(np.cos(yaw_rad))), abs(float(np.sin(yaw_rad)))
    hx = ROBOT_TORSO_THICKNESS / 2.0
    hz = ROBOT_SHOULDER_WIDTH / 2.0
    center = np.asarray(root_pos, dtype=float) + _rotate_xz(
        np.array([0.0, ROBOT_BODY_CENTER_Y, 0.0]), yaw_rad
    )
    half = np.array([hx * c + hz * s, ROBOT_BODY_HALF_HEIGHT, hx * s + hz * c])
    return center, half


def _aabb_overlap(
    c1: np.ndarray, h1: np.ndarray, c2: np.ndarray, h2: np.ndarray
) -> bool:
    return bool(np.all(np.abs(np.asarray(c1) - np.asarray(c2)) <= h1 + h2 + 1e-9))


def _check_wall_collision(
    root_pos: np.ndarray,
    yaw_rad: float,
    hand_pos: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Check the robot body (and optionally the hand) against the wall panels.

    Returns None when there is no collision, otherwise a dict with the
    colliding part ("torso", "shoulder" or "hand") and a contact point.
    """
    body_center, body_half = _robot_body_aabb(root_pos, yaw_rad)
    for panel_center, panel_half in _panel_boxes():
        if _aabb_overlap(body_center, body_half, panel_center, panel_half):
            point = np.clip(body_center, panel_center - panel_half, panel_center + panel_half)
            part = "shoulder" if body_center[1] > 1.2 else "torso"
            return {
                "part": part,
                "point": point.tolist(),
                "body_center": body_center.tolist(),
            }
    if hand_pos is not None:
        hand = np.asarray(hand_pos, dtype=float)
        for panel_center, panel_half in _panel_boxes():
            if _aabb_overlap(hand, np.zeros(3), panel_center, panel_half + 0.03):
                return {"part": "hand", "point": hand.tolist()}
    return None


@dataclass
class StepResult:
    """Result of one environment action."""

    rgb: Optional[np.ndarray]
    legal: bool
    feedback: str
    distance: float
    success: bool
    collision: Optional[Dict[str, Any]] = None
    state: Optional[Dict[str, Any]] = None


class BAOEnv:
    """Isaac Sim environment for the Body-as-Obstacle task."""

    def __init__(self, sim_app: Any = None, task_dict: Optional[Dict[str, Any]] = None) -> None:
        if not _HAS_ISAAC_SIM:
            raise RuntimeError(
                "Isaac Sim could not be imported. Run this script with "
                "%ISAACSIM_ROOT%\\python.bat (Windows) or $ISAACSIM_ROOT/python.sh (Linux)."
            )
        self.sim_app = sim_app
        self.task_dict = task_dict or {}

        settings = carb.settings.get_settings()
        settings.set(
            "/rtx/rendermode",
            str(self.task_dict.get("rendermode", "RaytracedLighting")),
        )
        settings.set("/rtx/pathtracing/spp", int(self.task_dict.get("spp", 16)))

        self.world = World(stage_units_in_meters=1.0)
        self.stage = self.world.stage

        self.robot_prim_path = "/World/H1"
        self.robot_usd_path: str = ""
        self.robot_root: Optional[XFormPrim] = None
        self.hand_xform: Optional[XFormPrim] = None
        self.hand_prim_path: Optional[str] = None
        self.head_xform: Optional[XFormPrim] = None
        self.head_prim_path: Optional[str] = None
        self.head_visual_path: Optional[str] = None
        self.eye_camera: Optional[Camera] = None

        self._articulation: Any = None
        self._articulation_ok = False
        self._articulation_error = ""
        self._reach_joint_indices: Optional[np.ndarray] = None
        self._robot_yaw = ROBOT_START_YAW_DEG
        self._robot_ground_offset = 0.0
        self.reaching = False

        self._create_ground()
        self._create_wall()
        self._create_target()
        self._create_lights()
        self._create_camera()
        self._load_robot()
        self._create_eye_camera()
        self._update_camera()
        self._update_eye_camera()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _create_ground(self) -> None:
        FixedCuboid(
            prim_path="/World/Ground",
            name="ground",
            position=_user_to_isaac_pos(
                np.array([2.0, -GROUND_THICKNESS / 2.0, 0.0])
            ),
            size=1.0,
            scale=_user_to_isaac_scale(
                np.array([SCENE_SIZE, GROUND_THICKNESS, SCENE_SIZE])
            ),
        )

    def _create_wall(self) -> None:
        z_center = PANEL_WIDTH / 2.0 + CHANNEL_HALF_WIDTH
        for i, sign in enumerate((-1.0, 1.0)):
            FixedCuboid(
                prim_path=f"/World/WallPanel_{i}",
                name=f"wall_panel_{i}",
                position=_user_to_isaac_pos(
                    np.array([WALL_X, WALL_HEIGHT / 2.0, sign * z_center])
                ),
                size=1.0,
                scale=_user_to_isaac_scale(
                    np.array([WALL_THICKNESS, WALL_HEIGHT, PANEL_WIDTH])
                ),
            )
            UsdGeom.Gprim(
                self.stage.GetPrimAtPath(f"/World/WallPanel_{i}")
            ).CreateDoubleSidedAttr(True)
            self._create_and_bind_glass_material(
                f"/World/WallPanel_{i}", f"/World/Looks/GlassMaterial_{i}"
            )

        for sign in (-1.0, 1.0):
            edge_id = 0 if sign < 0 else 1
            FixedCuboid(
                prim_path=f"/World/ChannelEdge_{edge_id}",
                name=f"channel_edge_{edge_id}",
                position=_user_to_isaac_pos(
                    np.array([WALL_X, WALL_HEIGHT / 2.0, sign * CHANNEL_HALF_WIDTH])
                ),
                size=1.0,
                scale=_user_to_isaac_scale(
                    np.array([0.01, 0.01, WALL_HEIGHT])
                ),
            )
            self._create_and_bind_material(
                f"/World/ChannelEdge_{edge_id}",
                f"/World/Looks/ChannelEdgeMaterial_{edge_id}",
                color=[0.75, 0.78, 0.82],
                metallic=0.0,
                roughness=0.4,
            )

    def _create_target(self) -> None:
        prim_path = "/World/TargetBall"
        sphere = UsdGeom.Sphere.Define(self.stage, prim_path)
        sphere.GetRadiusAttr().Set(TARGET_RADIUS)
        r = TARGET_RADIUS
        sphere.GetExtentAttr().Set([(-r, -r, -r), (r, r, r)])
        prim = sphere.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdGeom.Xformable(prim).AddTranslateOp().Set(
            Gf.Vec3d(*_user_to_isaac_pos(TARGET_POS))
        )
        self._create_and_bind_material(
            prim_path,
            "/World/Looks/TargetMaterial",
            color=[0.0, 1.0, 0.0],
            metallic=0.0,
            roughness=0.3,
        )

    def _create_camera(self) -> None:
        resolution = tuple(
            int(v) for v in self.task_dict.get("camera_resolution", (1024, 1024))
        )
        self.camera = Camera(
            prim_path="/World/Camera",
            translation=_user_to_isaac_pos(
                np.array([ROBOT_START_POS[0], ROBOT_HEAD_HEIGHT, ROBOT_START_POS[2]])
            ),
            frequency=20,
            resolution=resolution,
        )
        self.camera.set_focal_length(float(self.task_dict.get("camera_focal", 2.5)))

    def _create_eye_camera(self) -> None:
        """Robot eye camera mounted at the H1 head d435 module."""
        resolution = tuple(
            int(v) for v in self.task_dict.get("camera_resolution", (1024, 1024))
        )
        self.eye_camera = Camera(
            prim_path="/World/RobotEyeCamera",
            translation=_user_to_isaac_pos(
                np.array([ROBOT_START_POS[0], ROBOT_HEAD_HEIGHT, ROBOT_START_POS[2]])
            ),
            frequency=20,
            resolution=resolution,
        )
        self.eye_camera.set_focal_length(float(self.task_dict.get("camera_focal", 2.5)))

    def _create_lights(self) -> None:
        """Add scene lights; without them the camera images are black."""
        distant = UsdLux.DistantLight.Define(self.stage, "/World/DistantLight")
        distant.GetIntensityAttr().Set(1000.0)
        distant.AddRotateXYZOp().Set(Gf.Vec3d(-60.0, 0.0, 0.0))

        dome = UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
        dome.GetIntensityAttr().Set(200.0)

    def _load_robot(self) -> None:
        usd_path = self._resolve_robot_usd_path()
        add_reference_to_stage(usd_path=usd_path, prim_path=self.robot_prim_path)
        self.robot_usd_path = usd_path
        if not self.task_dict.get("robot_physics", False):
            self._disable_robot_physics()
        self.robot_root = XFormPrim(prim_paths_expr=self.robot_prim_path)
        self._robot_ground_offset = self._compute_robot_ground_offset()
        self._set_robot_pose(ROBOT_START_POS, ROBOT_START_YAW_DEG)
        self._find_hand_prim()
        self._find_head_camera_link()

    def _compute_robot_ground_offset(self) -> float:
        """Raise the robot so its lowest mesh point sits on the ground (z=0)."""
        try:
            from pxr import Usd, UsdGeom

            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            min_z = float("inf")
            for prim in self.stage.Traverse():
                path = str(prim.GetPath())
                if not path.startswith(self.robot_prim_path):
                    continue
                bound = cache.ComputeWorldBound(prim)
                range3d = bound.ComputeAlignedRange()
                lo = range3d.GetMin()
                hi = range3d.GetMax()
                if not all(
                    math.isfinite(v)
                    for v in (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])
                ):
                    continue
                min_z = min(min_z, lo[2])
            if min_z == float("inf"):
                return 0.0
            offset = float(-min_z)
            print(f"[BAOEnv] robot ground offset = {offset:.4f} m")
            return offset
        except Exception as exc:
            print(f"[BAOEnv] ground offset detection failed, using 0.0: {exc}")
            return 0.0

    def _resolve_robot_usd_path(self) -> str:
        candidates: List[str] = []
        override = self.task_dict.get("robot_usd_path") or os.environ.get("EMBODIEDBAO_H1_USD")
        if override:
            candidates.append(str(override))

        assets_root = ""
        try:
            assets_root = get_assets_root_path() or ""
        except Exception:
            assets_root = ""
        if assets_root:
            candidates.append(
                os.path.join(assets_root, "Isaac", "Robots", "Unitree", "H1", "h1.usd")
            )
            candidates.append(
                os.path.join(assets_root, "Isaac", "Robots", "Unitree", "H1", "h1_with_hands.usd")
            )

        isaac_lab_root = os.environ.get("ISAACLAB_ASSETS_DIR", "")
        if isaac_lab_root:
            candidates.append(
                os.path.join(isaac_lab_root, "Isaac", "Robots", "Unitree", "H1", "h1.usd")
            )
            candidates.append(
                os.path.join(isaac_lab_root, "Isaac", "Robots", "Unitree", "H1", "h1_with_hands.usd")
            )

        isaacsim_root = os.environ.get("ISAACSIM_ROOT", "")
        if isaacsim_root:
            candidates.append(
                os.path.join(isaacsim_root, "assets", "Isaac", "Robots", "Unitree", "H1", "h1.usd")
            )
        candidates.append(os.path.join(os.getcwd(), "assets", "H1", "h1.usd"))
        candidates.append(os.path.join(os.getcwd(), "assets", "H1", "h1_with_hands.usd"))
        candidates.append("omniverse://localhost/Isaac/Robots/Unitree/H1/h1.usd")
        candidates.append("omniverse://localhost/Isaac/Robots/Unitree/H1/h1.usda")
        for path in self._discover_h1_assets():
            if path not in candidates:
                candidates.append(path)

        seen = set()
        for path in candidates:
            if not path or path in seen:
                continue
            seen.add(path)
            if os.path.isfile(path):
                return path
        for path in seen:
            if path.startswith("omniverse://") or path.startswith("http"):
                return path
        raise FileNotFoundError(
            "Could not locate a Unitree H1 USD asset. Set EMBODIEDBAO_H1_USD to the "
            "USD path, place h1.usd under assets/H1/, or install Isaac Sim/Isaac Lab "
            "assets."
        )

    def _discover_h1_assets(self, max_depth: int = 10) -> List[str]:
        """Search common Isaac Lab / local asset locations for H1 USD files."""
        roots: List[str] = []
        for env_name in ("EMBODIEDBAO_H1_USD_DIR", "ISAACLAB_ASSETS_DIR"):
            root = os.environ.get(env_name, "")
            if root and os.path.isdir(root):
                roots.append(root)
        try:
            import isaaclab_assets

            module_dir = os.path.dirname(os.path.abspath(isaaclab_assets.__file__))
            if module_dir not in roots and os.path.isdir(module_dir):
                roots.append(module_dir)
        except Exception:
            pass
        home = os.path.expanduser("~")
        roots.extend(
            [
                os.path.join(home, "isaaclab"),
                os.path.join(home, ".local", "share", "ov", "pkg"),
            ]
        )
        isaacsim_root = os.environ.get("ISAACSIM_ROOT", "")
        if isaacsim_root:
            roots.append(os.path.join(isaacsim_root, "assets"))
        roots.append(os.path.join(os.getcwd(), "assets"))
        roots = [root for root in roots if root and os.path.isdir(root)]

        found: List[str] = []
        for root in roots:
            for dirpath, dirnames, filenames in os.walk(root):
                depth = dirpath[len(root):].count(os.sep)
                if depth >= max_depth:
                    dirnames[:] = []
                    continue
                for name in filenames:
                    lower = name.lower()
                    if not lower.startswith("h1") or not lower.endswith(
                        (".usd", ".usda")
                    ):
                        continue
                    path = os.path.join(dirpath, name)
                    if path not in found:
                        found.append(path)
        found.sort(
            key=lambda path: (
                0 if "unitree" in path.lower() else 1,
                path.lower(),
            )
        )
        return found

    def _find_hand_prim(self) -> None:
        scored: List[Tuple[int, str]] = []
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(self.robot_prim_path):
                continue
            name = prim.GetName().lower()
            if "right_hand" in name or name in ("r_hand", "right_hand"):
                scored.append((2, path))
            elif "hand" in name:
                scored.append((1, path))
        if not scored:
            return
        scored.sort(key=lambda item: -item[0])
        self.hand_prim_path = scored[0][1]
        try:
            self.hand_xform = XFormPrim(prim_paths_expr=self.hand_prim_path)
        except Exception as exc:  # pragma: no cover - runtime Isaac Sim path
            print(f"[BAOEnv] Could not wrap hand prim {self.hand_prim_path}: {exc}")
            self.hand_xform = None

    def _find_head_camera_link(self) -> None:
        """Locate the H1 head camera module (prefer the visual lens mesh)."""
        scored: List[Tuple[int, str]] = []
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(self.robot_prim_path):
                continue
            name = prim.GetName().lower()
            is_visual = "/visuals/" in path.lower()
            if "d435" in name and "rgb" in name:
                scored.append((3 if is_visual else 2, path))
            elif "d435" in name and "imager" in name:
                scored.append((2 if is_visual else 1, path))
            elif "d435" in name and is_visual:
                scored.append((1, path))
        if not scored:
            return
        scored.sort(key=lambda item: -item[0])
        self.head_prim_path = scored[0][1]
        if "/visuals/" in self.head_prim_path.lower():
            self.head_visual_path = self.head_prim_path
        try:
            self.head_xform = XFormPrim(prim_paths_expr=self.head_prim_path)
        except Exception as exc:
            print(f"[BAOEnv] Could not wrap head camera link {self.head_prim_path}: {exc}")
            self.head_xform = None

    def _disable_robot_collisions(self) -> None:
        """Kinematic mode: keep the articulated body but drop its colliders."""
        for sub_prim in self.stage.Traverse():
            path = str(sub_prim.GetPath())
            if not path.startswith(self.robot_prim_path):
                continue
            if sub_prim.HasAPI(UsdPhysics.CollisionAPI):
                sub_prim.RemoveAPI(UsdPhysics.CollisionAPI)
            if sub_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                sub_prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)

    def _disable_robot_physics(self) -> None:
        """Freeze the robot in its authored pose.

        The robot stays movable kinematically and wall contact is still
        detected by the analytic collision gate, but the physics engine can
        never knock the body over.
        """
        for sub_prim in self.stage.Traverse():
            path = str(sub_prim.GetPath())
            if not path.startswith(self.robot_prim_path):
                continue
            for api in (
                UsdPhysics.RigidBodyAPI,
                UsdPhysics.CollisionAPI,
                PhysxSchema.PhysxRigidBodyAPI,
                PhysxSchema.PhysxCollisionAPI,
                PhysxSchema.PhysxArticulationAPI,
                UsdPhysics.ArticulationRootAPI,
            ):
                try:
                    if sub_prim.HasAPI(api):
                        sub_prim.RemoveAPI(api)
                except Exception:
                    pass
            if sub_prim.IsA(UsdPhysics.Joint):
                try:
                    sub_prim.SetActive(False)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------

    def _create_and_bind_glass_material(self, prim_path: str, mat_path: str) -> None:
        # OmniGlass.mdl is not reliably available in every Isaac Sim build;
        # default to a translucent PreviewSurface so the wall is always see-through.
        if not self.task_dict.get("use_omni_glass", False):
            self._create_and_bind_material(
                prim_path,
                mat_path,
                color=[0.72, 0.88, 0.92],
                metallic=0.0,
                roughness=0.02,
                opacity=0.08,
            )
            return
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim {prim_path} not found to bind glass material.")
        sdf_path = Sdf.Path(mat_path)
        mat = UsdShade.Material.Define(self.stage, sdf_path)
        shader = UsdShade.Shader.Define(self.stage, sdf_path.AppendChild("OmniGlass"))
        shader.CreateIdAttr("OmniGlass.mdl")
        shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.85, 0.92, 0.95)
        )
        shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.45)
        shader.CreateInput("glass_reflection", Sdf.ValueTypeNames.Float).Set(0.9)
        shader.CreateInput("glass_refraction", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("glass_roughness", Sdf.ValueTypeNames.Float).Set(0.02)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(prim).Bind(mat)

    def _create_and_bind_material(
        self,
        prim_path: str,
        mat_path: str,
        color: List[float],
        metallic: float = 1.0,
        roughness: float = 0.0,
        opacity: float = 1.0,
    ) -> None:
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim {prim_path} not found to bind material.")
        sdf_path = Sdf.Path(mat_path)
        mat = UsdShade.Material.Define(self.stage, sdf_path)
        shader = UsdShade.Shader.Define(self.stage, sdf_path.AppendChild("PreviewSurface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(prim).Bind(mat)

    # ------------------------------------------------------------------
    # Robot pose / articulation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _yaw_quat(yaw_deg: float) -> np.ndarray:
        """Quaternion (w, x, y, z) for yaw around Isaac Sim's +z (up)."""
        half = float(np.radians(yaw_deg)) / 2.0
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])

    def _set_robot_pose(self, position: np.ndarray, yaw_deg: float) -> None:
        if self.robot_root is None:
            return
        pos = np.asarray(position, dtype=float).copy()
        pos[1] += float(self.task_dict.get("robot_root_y_offset", 0.0))
        yaw_offset = float(self.task_dict.get("robot_yaw_offset", MODEL_YAW_OFFSET_DEG))
        quat = self._yaw_quat(yaw_deg + yaw_offset)
        pos_isaac = _user_to_isaac_pos(pos)
        pos_isaac[2] += self._robot_ground_offset
        self.robot_root.set_world_poses(
            positions=np.array([pos_isaac]),
            orientations=np.array([quat]),
        )
        self._robot_yaw = float(yaw_deg)

    def _root_position(self) -> np.ndarray:
        if self.robot_root is None:
            return ROBOT_START_POS.copy()
        pos = self.robot_root.get_world_poses()[0][0]
        return _isaac_to_user_pos(np.asarray(pos, dtype=float))

    def _init_robot_controller(self) -> bool:
        if not self.task_dict.get("robot_physics", False):
            self._articulation_ok = False
            return False
        try:
            if self._articulation is None:
                import_errors = []
                articulation_class = None
                for module_name in (
                    "isaacsim.core.api.articulations",
                    "isaacsim.core.articulations",
                    "omni.isaac.core.articulations",
                ):
                    try:
                        articulation_class = getattr(
                            __import__(module_name, fromlist=["Articulation"]),
                            "Articulation",
                        )
                        break
                    except Exception as exc:
                        import_errors.append(f"{module_name}: {exc}")
                if articulation_class is None:
                    try:
                        from isaacsim.core.prims import SingleArticulation

                        articulation_class = SingleArticulation
                    except Exception as exc:
                        import_errors.append(
                            f"isaacsim.core.prims.SingleArticulation: {exc}"
                        )
                    try:
                        from isaacsim.core.prims import Articulation

                        articulation_class = Articulation
                    except Exception as exc:
                        import_errors.append(
                            f"isaacsim.core.prims.Articulation: {exc}"
                        )
                if articulation_class is None:
                    raise ImportError("; ".join(import_errors))
                Articulation = articulation_class

                try:
                    self._articulation = Articulation(
                        prim_path=self.robot_prim_path
                    )
                except TypeError:
                    self._articulation = Articulation(
                        prim_paths_expr=self.robot_prim_path
                    )
                self._articulation.initialize()
            self._articulation.post_reset()
            if self._reach_joint_indices is None:
                self._reach_joint_indices = self._find_reach_joint_indices()
            self._articulation_ok = self._reach_joint_indices is not None
        except Exception as exc:
            available = []
            if self._articulation is not None:
                available = [
                    n for n in dir(self._articulation) if not n.startswith("_")
                ][:60]
            self._articulation_error = f"{exc} | available={available}"
            print(f"[BAOEnv] Articulation init failed, using kinematic hand fallback: {exc}")
            self._articulation = None
            self._articulation_ok = False
            self._reach_joint_indices = None
        return self._articulation_ok

    def _articulation_dof_names(self) -> List[str]:
        art = self._articulation
        for name in ("get_dof_names", "get_joint_names"):
            fn = getattr(art, name, None)
            if callable(fn):
                return [str(n) for n in fn()]
        for name in ("dof_names", "joint_names"):
            val = getattr(art, name, None)
            if val is not None:
                return [str(n) for n in val]
        raise AttributeError("no dof-names accessor on articulation")

    def _articulation_set_targets(
        self, positions: np.ndarray, joint_indices: np.ndarray
    ) -> None:
        art = self._articulation
        pos = np.asarray(positions, dtype=float)
        idx = np.asarray(joint_indices, dtype=int)
        for name in (
            "set_joint_targets",
            "set_dof_targets",
            "set_joint_positions_targets",
            "set_dof_positions_targets",
        ):
            fn = getattr(art, name, None)
            if not callable(fn):
                continue
            try:
                fn(positions=pos, joint_indices=idx)
                return
            except TypeError:
                try:
                    fn(pos, idx)
                    return
                except TypeError:
                    continue
        controller_fn = getattr(art, "get_articulation_controller", None)
        if callable(controller_fn):
            try:
                controller = controller_fn()
                for name in ("set_joint_target_positions", "set_joint_positions"):
                    fn = getattr(controller, name, None)
                    if not callable(fn):
                        continue
                    try:
                        fn(positions=pos, joint_indices=idx)
                        return
                    except TypeError:
                        fn(pos, idx)
                        return
            except Exception:
                pass
        set_pos = getattr(art, "set_joint_positions", None)
        if callable(set_pos):
            try:
                set_pos(positions=pos, joint_indices=idx)
                return
            except TypeError:
                try:
                    set_pos(pos, idx)
                    return
                except TypeError:
                    pass
        apply_action = getattr(art, "apply_action", None)
        if callable(apply_action):
            try:
                apply_action({"joint_positions": pos, "joint_indices": idx})
                return
            except Exception:
                pass
        raise AttributeError("no joint-target setter on articulation")

    def _articulation_joint_positions(self) -> np.ndarray:
        art = self._articulation
        for name in ("get_joint_positions", "get_dof_positions"):
            fn = getattr(art, name, None)
            if callable(fn):
                return np.asarray(fn(), dtype=float).reshape(-1)
        for name in ("joint_positions", "dof_positions"):
            val = getattr(art, name, None)
            if val is not None:
                return np.asarray(val, dtype=float).reshape(-1)
        return np.zeros(0)

    def _find_reach_joint_indices(self) -> Optional[np.ndarray]:
        names = self._articulation_dof_names()

        def pick(keyword_groups: List[List[str]]) -> Optional[int]:
            for keywords in keyword_groups:
                for i, name in enumerate(names):
                    lower = name.lower()
                    if all(k in lower for k in keywords):
                        return i
            return None

        right_shoulder = pick([["right", "shoulder_pitch"], ["shoulder_pitch"]])
        right_elbow = pick([["right", "elbow"], ["elbow"]])
        if right_shoulder is None or right_elbow is None:
            return None
        return np.array([right_shoulder, right_elbow], dtype=int)

    def _set_reaching(self, reaching: bool) -> None:
        if self._articulation_ok and self._reach_joint_indices is not None:
            if reaching:
                targets = np.array([REACH_SHOULDER_PITCH_RAD, REACH_ELBOW_PITCH_RAD])
            else:
                targets = np.zeros(len(self._reach_joint_indices))
            self._articulation_set_targets(
                targets, self._reach_joint_indices
            )
        self.reaching = reaching

    def _set_standing_joint_targets(self) -> None:
        """Pose the robot: upright hips/knees and arms hanging down."""
        if not self._articulation_ok or self._articulation is None:
            return
        try:
            names = self._articulation_dof_names()
            indices: List[int] = []
            positions: List[float] = []
            hang_shoulder = float(
                self.task_dict.get(
                    "arm_hang_shoulder_pitch_rad", ARM_HANG_SHOULDER_PITCH_RAD
                )
            )
            hang_elbow = float(
                self.task_dict.get("arm_hang_elbow_pitch_rad", ARM_HANG_ELBOW_PITCH_RAD)
            )
            for i, name in enumerate(names):
                lower = name.lower()
                if any(key in lower for key in ("hip", "knee")):
                    indices.append(i)
                    positions.append(0.0)
                elif "shoulder_pitch" in lower:
                    indices.append(i)
                    positions.append(hang_shoulder)
                elif "elbow" in lower:
                    indices.append(i)
                    positions.append(hang_elbow)
            if indices:
                self._articulation_set_targets(
                    np.asarray(positions, dtype=float),
                    np.array(indices, dtype=int),
                )
                try:
                    values = self._articulation_joint_positions()
                    print(
                        "[BAOEnv] arm pose:",
                        [
                            (names[i], round(float(values[i]), 3))
                            for i in indices
                            if i < len(values)
                        ],
                    )
                except Exception:
                    pass
        except Exception as exc:
            available = (
                [n for n in dir(self._articulation) if not n.startswith("_")][:60]
                if self._articulation is not None
                else []
            )
            print(
                f"[BAOEnv] standing joint init skipped: {exc} | available={available}"
            )

    def _analytic_hand_position(self) -> np.ndarray:
        root = self._root_position()
        local = HAND_LOCAL_REACH if self.reaching else HAND_LOCAL_REST
        yaw_offset = float(self.task_dict.get("robot_yaw_offset", MODEL_YAW_OFFSET_DEG))
        return root + _rotate_xz(local, np.radians(self._robot_yaw + yaw_offset))

    def _update_camera(self) -> None:
        pos = self._root_position()
        forward = _forward_vector(np.radians(self._robot_yaw))
        # Keep the observation camera at head height and always look at the
        # green ball so it stays centered in the frame.
        forward_offset = float(self.task_dict.get("camera_forward_offset", 0.42))
        camera_height = float(self.task_dict.get("camera_height", 1.1))
        eye = np.array([pos[0], camera_height, pos[2]]) + forward * forward_offset
        target = np.asarray(TARGET_POS, dtype=float).copy()
        eye_isaac = _user_to_isaac_pos(eye)
        target_isaac = _user_to_isaac_pos(target)
        try:
            set_camera_view(
                eye=eye_isaac.tolist(),
                target=target_isaac.tolist(),
                up=[0.0, 0.0, 1.0],
                camera_prim_path="/World/Camera",
            )
        except TypeError:
            set_camera_view(
                eye=eye_isaac.tolist(),
                target=target_isaac.tolist(),
                camera_prim_path="/World/Camera",
            )

    def _update_eye_camera(self) -> None:
        """Aim the robot eye camera from the robot body at ball height."""
        if self.eye_camera is None:
            return
        head = self._head_camera_position()
        eye = head
        target = np.asarray(TARGET_POS, dtype=float).copy()
        eye_isaac = _user_to_isaac_pos(eye)
        target_isaac = _user_to_isaac_pos(target)
        try:
            set_camera_view(
                eye=eye_isaac.tolist(),
                target=target_isaac.tolist(),
                up=[0.0, 0.0, 1.0],
                camera_prim_path="/World/RobotEyeCamera",
            )
        except TypeError:
            set_camera_view(
                eye=eye_isaac.tolist(),
                target=target_isaac.tolist(),
                camera_prim_path="/World/RobotEyeCamera",
            )

    def _head_camera_position(self) -> np.ndarray:
        """Robot eye anchor: on the body, at the green ball height."""
        root = self._root_position()
        forward = _forward_vector(np.radians(self._robot_yaw))
        height = float(self.task_dict.get("camera_height", TARGET_POS[1]))
        offset = float(self.task_dict.get("eye_forward_offset", 0.45))
        return np.array([root[0], height, root[2]], dtype=float) + forward * offset

    # ------------------------------------------------------------------
    # Public environment interface
    # ------------------------------------------------------------------

    def reset_scene(self) -> Tuple[np.ndarray, float]:
        """Reset the robot and target, then return (rgb, distance)."""
        self.world.reset()
        self.camera.initialize()
        if self.eye_camera is not None:
            self.eye_camera.initialize()
        self._init_robot_controller()
        self._set_robot_pose(ROBOT_START_POS, ROBOT_START_YAW_DEG)
        self.reaching = False
        if self._articulation is not None:
            try:
                self._articulation.post_reset()
            except Exception:
                pass
        self._set_standing_joint_targets()
        self._update_camera()
        self._update_eye_camera()
        for _ in range(int(self.task_dict.get("reset_steps", 30))):
            self.world.step(render=True)
        rgb = (
            self.eye_camera.get_rgb()
            if self.eye_camera is not None
            else self.camera.get_rgb()
        )
        return rgb, self.get_distance_to_target()

    def reset(self) -> Tuple[np.ndarray, float]:
        """Alias for reset_scene (MirrorBench compatibility)."""
        return self.reset_scene()

    def get_camera_image(self) -> np.ndarray:
        if self.eye_camera is not None:
            return self.eye_camera.get_rgb()
        return self.camera.get_rgb()

    def get_robot_state(self) -> Dict[str, Any]:
        pos = self._root_position()
        joints: Dict[str, float] = {}
        if self._articulation_ok and self._articulation is not None:
            try:
                names = self._articulation_dof_names()
                values = self._articulation_joint_positions()
                joints = {name: float(v) for name, v in zip(names, values)}
            except Exception:
                joints = {}
        return {
            "position": pos.tolist(),
            "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": self._robot_yaw},
            "joint_angles": joints,
            "end_effector_position": self.get_hand_position().tolist(),
            "reaching": self.reaching,
        }

    def get_hand_position(self) -> np.ndarray:
        if self._articulation_ok and self.hand_xform is not None:
            try:
                pos = self.hand_xform.get_world_poses()[0][0]
                return _isaac_to_user_pos(np.asarray(pos, dtype=float))
            except Exception:
                pass
        return self._analytic_hand_position()

    def get_distance_to_target(self) -> float:
        hand = np.asarray(self.get_hand_position(), dtype=float)
        return float(np.linalg.norm(hand - TARGET_POS) - TARGET_RADIUS)

    def get_body_obstacle_status(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Return (stuck, collision_info) for torso attempts through the channel."""
        root = self._root_position()
        near_wall = abs(root[0] - WALL_X) <= ROBOT_TORSO_THICKNESS / 2.0 + MOVE_STEP + 0.05
        at_channel = abs(root[2]) <= CHANNEL_HALF_WIDTH + 0.30
        probe = root + np.array([MOVE_STEP, 0.0, 0.0])
        collision = _check_wall_collision(probe, np.radians(self._robot_yaw), hand_pos=None)
        stuck = bool(near_wall and at_channel and collision is not None)
        return stuck, collision

    def check_success(self) -> bool:
        return bool(self.get_distance_to_target() < SUCCESS_DISTANCE)

    def check_collision_with_wall(self) -> Optional[Dict[str, Any]]:
        root = self._root_position()
        hand = self.get_hand_position()
        return _check_wall_collision(root, np.radians(self._robot_yaw), hand_pos=hand)

    def execute_action(self, action: str, n_steps: Optional[int] = None) -> StepResult:
        if n_steps is None:
            n_steps = int(self.task_dict.get("action_steps", 30))
        action = str(action).strip().lower()
        if action not in ACTIONS:
            return StepResult(
                rgb=self.get_camera_image(),
                legal=False,
                feedback=f"unknown action: {action}",
                distance=self.get_distance_to_target(),
                success=self.check_success(),
            )

        legal, feedback, collision = self._apply_action(action)
        self._update_camera()
        self._update_eye_camera()
        for _ in range(int(n_steps)):
            self.world.step(render=True)

        distance = self.get_distance_to_target()
        return StepResult(
            rgb=self.get_camera_image(),
            legal=legal,
            feedback=feedback,
            distance=distance,
            success=self.check_success(),
            collision=collision,
            state=self.get_robot_state(),
        )

    def step_wait(self, n_steps: int = 10000) -> None:
        for _ in range(n_steps):
            self.world.step(render=True)

    def close(self) -> None:
        if self.sim_app is not None:
            self.sim_app.close()

    # ------------------------------------------------------------------
    # Internal action execution
    # ------------------------------------------------------------------

    def _apply_action(self, action: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        root = self._root_position()
        yaw = np.radians(self._robot_yaw)

        if action in ("forward", "backward", "left", "right"):
            sign = 1.0 if action in ("forward", "right") else -1.0
            if action in ("forward", "backward"):
                delta = _forward_vector(yaw) * (sign * MOVE_STEP)
            else:
                delta = _rotate_xz(np.array([0.0, 0.0, sign * MOVE_STEP]), yaw)
            target = root + delta
            collision = _check_wall_collision(target, yaw, hand_pos=None)
            if collision is not None:
                return False, f"blocked by transparent wall ({collision['part']})", collision
            self._set_robot_pose(target, self._robot_yaw)
            return True, "executed", None

        if action in ("turn_left", "turn_right"):
            new_yaw = self._robot_yaw + (
                TURN_STEP_DEG if action == "turn_left" else -TURN_STEP_DEG
            )
            collision = _check_wall_collision(root, np.radians(new_yaw), hand_pos=None)
            if collision is not None:
                return False, "cannot turn: body would collide with transparent wall", collision
            self._set_robot_pose(root, new_yaw)
            return True, "executed", None

        if action in ("reach", "retreat"):
            self._set_reaching(action == "reach")
            return True, "executed", None

        return False, "invalid action", None


def setup_scene(sim_app: Any = None, task_dict: Optional[Dict[str, Any]] = None) -> BAOEnv:
    """Convenience factory used by main.py: create and return a BAOEnv."""
    return BAOEnv(sim_app=sim_app, task_dict=task_dict)


def _smoke_test() -> None:
    """Minimal headless smoke test (run with the Isaac Sim python)."""
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    env = BAOEnv(simulation_app, task_dict={"headless": True})
    rgb, dist = env.reset_scene()
    print(f"[smoke] reset rgb={rgb.shape} dist={dist:.4f}")
    actions = ["forward"] * 10 + ["turn_right"] * 6 + ["left"] * 8 + ["turn_left"] * 6
    for action in actions:
        result = env.execute_action(action)
        print(
            f"[smoke] {action:10s} legal={result.legal} "
            f"feedback={result.feedback} dist={result.distance:.4f}"
        )
    env.close()


if __name__ == "__main__":
    _smoke_test()
