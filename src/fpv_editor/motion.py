"""Camera motion between consecutive frames: rotation rates, flow speed and approach.

FPV tricks are fast camera rotations (flips peak near 900 deg/s and last 0.1-0.4 s), so
they are only visible when every frame is analysed. Dense optical flow (DIS) is sampled
on a grid of textured points outside the OSD, the points are turned into viewing rays
through an equidistant fisheye model, and a robust rotation-only fit (2-point RANSAC,
then Kabsch) gives the camera's pitch, yaw and roll rates. An affine fit of the same
points gives the expansion rate (divergence): positive while the camera closes in on
something, which is what gap and obstacle passes look like.

The lens field of view is unknown for most DVR footage; 120 degrees fits the analog
flights tested. Degree values scale with it, so later stages compare rates within a
flight or use generous thresholds rather than exact angles.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
GrayFrame = NDArray[np.uint8]
Mask = NDArray[np.bool_]

# Per-pair result columns of `MotionEstimator.__call__`.
PITCH, YAW, ROLL, CONFIDENCE, SPEED, DIVERGENCE = range(6)
FIELDS = 6


@dataclass(frozen=True)
class MotionConfig:
    hfov_deg: float = 120.0  # horizontal field of view of an equidistant fisheye lens
    grid_step: int = 8  # flow sample spacing at 240 px wide (scales with width)
    texture: float = 12.0  # mean |gradient| a cell needs before its flow is trusted
    inlier_px: float = 2.5  # rotation-fit inlier threshold at 240 px wide
    ransac_iterations: int = 100
    min_points: int = 12


class FisheyeCamera:
    """Equidistant fisheye: pixel distance from the centre is proportional to the angle."""

    def __init__(self, width: int, height: int, hfov_deg: float) -> None:
        self.focal = (width / 2) / (np.radians(hfov_deg) / 2)
        self.cx, self.cy = (width - 1) / 2, (height - 1) / 2

    def rays(self, points: NDArray[np.float32]) -> FloatArray:
        x = (points[:, 0] - self.cx) / self.focal
        y = (points[:, 1] - self.cy) / self.focal
        angle = np.hypot(x, y)
        scale = np.where(angle > 1e-9, np.sin(angle) / np.maximum(angle, 1e-9), 1.0)
        return np.stack([x * scale, y * scale, np.cos(angle)], axis=1)


class MotionEstimator:
    """Measures the motion between two consecutive grey frames of one size.

    ``usable`` marks pixels whose flow may be used (False on the OSD, props and borders).
    """

    def __init__(
        self, width: int, height: int, usable: Mask, fps: float, config: MotionConfig
    ) -> None:
        self.config = config
        self.width = width
        self.fps = fps
        self.camera = FisheyeCamera(width, height, config.hfov_deg)
        self.scale = width / 240.0
        self.step = max(4, round(config.grid_step * self.scale))
        self.flow = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        # Sample points whose whole cell is usable.
        cell_ok = cv2.erode(usable.astype(np.uint8), np.ones((self.step, self.step), np.uint8))
        ys, xs = np.mgrid[self.step // 2 : height : self.step, self.step // 2 : width : self.step]
        keep = cell_ok[ys, xs] > 0
        self.xs, self.ys = xs[keep], ys[keep]
        self.grid = np.stack([self.xs, self.ys], axis=1).astype(np.float32)
        self.rng = np.random.default_rng(0)

    def __call__(self, previous: GrayFrame, current: GrayFrame) -> FloatArray:
        """pitch, yaw, roll (deg/s), confidence (0..1), speed (widths/s), divergence (1/s)."""
        result = np.full(FIELDS, np.nan)
        result[CONFIDENCE] = 0.0
        if len(self.grid) < self.config.min_points:
            return result
        # No initial flow: passing a zero array changes DIS results; the stubs lack None.
        flow = self.flow.calc(previous, current, None)  # type: ignore[call-overload]
        gx = cv2.Sobel(previous, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(previous, cv2.CV_32F, 0, 1, ksize=3)
        texture = cv2.blur(np.abs(gx) + np.abs(gy), (self.step, self.step))
        textured = texture[self.ys, self.xs] > self.config.texture
        moved = flow[self.ys, self.xs]
        result[SPEED] = float(np.median(np.hypot(moved[:, 0], moved[:, 1]))) / self.width * self.fps
        start = self.grid[textured]
        end = start + moved[textured]
        height, width = previous.shape
        inside = (end[:, 0] >= 0) & (end[:, 0] < width) & (end[:, 1] >= 0) & (end[:, 1] < height)
        start, end = start[inside], end[inside].astype(np.float32)
        if len(start) < self.config.min_points:
            return result
        affine, _ = cv2.estimateAffine2D(
            start, end, method=cv2.RANSAC, ransacReprojThreshold=1.5 * self.scale
        )
        if affine is not None:
            result[DIVERGENCE] = (affine[0, 0] + affine[1, 1] - 2.0) * self.fps
        rays0, rays1 = self.camera.rays(start), self.camera.rays(end)
        threshold = self.config.inlier_px * self.scale / self.camera.focal
        fit = _solve_rotation(rays0, rays1, threshold, self.config.ransac_iterations, self.rng)
        if fit is None:
            return result
        rotation, inliers, rms = fit
        rotvec, _ = cv2.Rodrigues(rotation)
        # The scene turned by `rotation`, so the camera turned by its inverse. Axes are the
        # camera's x (right), y (down) and z (forward): + pitch is nose up, + yaw is a turn
        # to the right and + roll is a roll to the right.
        result[PITCH : ROLL + 1] = np.degrees(-rotvec.ravel() * self.fps)
        result[CONFIDENCE] = _confidence(len(start), inliers, rms * self.camera.focal / self.scale)
        return result


def _cross(a: FloatArray, b: FloatArray) -> FloatArray:
    """Row-wise cross product of (n, 3) arrays (``np.cross`` has a large per-call cost)."""
    a0, a1, a2 = a[:, 0], a[:, 1], a[:, 2]
    b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]
    return np.stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], axis=-1)


def _triads(a: FloatArray, b: FloatArray) -> FloatArray:
    second = _cross(a, b)
    second /= np.sqrt(np.einsum("ij,ij->i", second, second))[:, None] + 1e-12
    third = _cross(a, second)
    return np.stack([a, second, third], axis=-1)


def _kabsch(a: FloatArray, b: FloatArray) -> FloatArray:
    """Rotation R minimising sum |R a - b|^2."""
    u, _, vt = np.linalg.svd(a.T @ b)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    return np.asarray(vt.T @ np.diag([1.0, 1.0, sign]) @ u.T, dtype=np.float64)


def _solve_rotation(
    a: FloatArray, b: FloatArray, threshold: float, iterations: int, rng: np.random.Generator
) -> tuple[FloatArray, int, float] | None:
    """Robust rotation with b ~ R a: (R, inlier count, rms residual in radians)."""
    count = len(a)
    scored = rng.choice(count, 150, replace=False) if count > 150 else np.arange(count)
    # Draw point pairs in batches of 20 (the order the fit has always used), then score
    # every candidate at once; the first candidate with the most hits wins.
    draws = [
        (rng.integers(0, count, 20), rng.integers(0, count, 20))
        for _ in range(max(1, iterations // 20))
    ]
    i = np.concatenate([d[0] for d in draws])
    j = np.concatenate([d[1] for d in draws])
    distinct = (i != j) & (np.einsum("ij,ij->i", a[i], a[j]) < np.cos(np.radians(2.0)))
    i, j = i[distinct], j[distinct]
    if len(i) == 0:
        return None
    candidates = _triads(b[i], b[j]) @ np.transpose(_triads(a[i], a[j]), (0, 2, 1))
    offsets = candidates @ a[scored].T - b[scored].T[None]  # (candidate, axis, point)
    hits = ((offsets * offsets).sum(axis=1) < threshold**2).sum(axis=1)
    rotation = candidates[int(np.argmax(hits))]
    for _ in range(3):
        inliers = np.linalg.norm(a @ rotation.T - b, axis=1) < threshold
        if inliers.sum() < 3:
            return None
        rotation = _kabsch(a[inliers], b[inliers])
    residual = np.linalg.norm(a[inliers] @ rotation.T - b[inliers], axis=1)
    return rotation, int(inliers.sum()), float(np.sqrt(np.mean(residual**2)))


def _confidence(points: int, inliers: int, rms_px: float) -> float:
    """0..1: enough inliers, a high inlier share and a small residual (px at 240 wide)."""
    enough = np.clip((inliers - 8) / 32.0, 0, 1)
    share = np.clip((inliers / points - 0.3) / 0.4, 0, 1)
    tight = np.clip(1.5 - rms_px / 1.5, 0, 1)
    return float(enough * share * tight)
