"""Small, coordinate-preserving repairs for CVAT polygon and polyline labels.

The repairer first removes repeated vertices while preserving the first
occurrence order.  It then only changes point connectivity (2-opt) or removes
a local loop, and uses a pixel mask to decide whether the change is small
enough to accept automatically.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

ShapeKind = Literal["polygon", "polyline"]
_EPSILON = 1e-9


@dataclass(frozen=True)
class RepairResult:
    """Outcome of attempting to repair one shape."""

    points: np.ndarray
    changed: bool
    diff_pixels: int | None
    needs_review: bool
    intersections: tuple[tuple[int, int], ...]


def repair_shape(
    points: Sequence[Sequence[float]] | np.ndarray,
    shape_kind: ShapeKind,
    image_size: tuple[int, int],
    threshold: int,
    *,
    polyline_width: int = 1,
) -> RepairResult:
    """Repair self-intersections while preserving retained vertex coordinates.

    ``image_size`` is ``(width, height)``.  A shape with no self-intersection
    is returned with its duplicate vertices removed.  When every legal
    candidate changes more pixels than ``threshold``, the normalized points
    are returned and ``needs_review`` is set so the caller can keep the
    annotation in a manual-review queue.
    """
    if shape_kind not in {"polygon", "polyline"}:
        raise ValueError(f"Unsupported shape kind: {shape_kind!r}")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {image_size}")
    if threshold < 0:
        raise ValueError(
            f"Self-intersection threshold must be non-negative, got {threshold}"
        )
    if polyline_width < 1:
        raise ValueError(
            f"Polyline rasterization width must be positive, got {polyline_width}"
        )

    original = deduplicate_points(points)
    closed = shape_kind == "polygon"
    intersections = tuple(find_self_intersections(original, closed=closed))
    if not intersections:
        return RepairResult(original, False, 0, False, intersections)

    best: tuple[tuple[int, int, int, tuple[float, ...]], np.ndarray] | None = None
    states = [original]
    seen = {_points_key(original)}
    # A few intersecting edges normally need one 2-opt.  The bounded search
    # also handles annotations containing several independent crossings
    # without allowing combinatorial growth on a malformed large contour.
    max_depth = min(max(len(original), 1), 8)
    max_states = 128
    for depth in range(max_depth + 1):
        next_states: list[np.ndarray] = []
        for state in states:
            if len(next_states) >= max_states:
                break
            state_intersections = find_self_intersections(state, closed=closed)
            if not state_intersections:
                diff = _pixel_diff(
                    original,
                    state,
                    shape_kind,
                    width,
                    height,
                    polyline_width,
                )
                key = (
                    diff,
                    abs(len(state) - len(original)),
                    depth,
                    _points_key(state),
                )
                if best is None or key < best[0]:
                    best = (key, state)
                continue
            if depth >= max_depth:
                continue
            # Expanding every current crossing gives the search a chance to
            # choose a different local reconnection when the first one leaves
            # another crossing behind.
            for edge_i, edge_j in state_intersections:
                for candidate in _candidate_points(
                    state,
                    edge_i,
                    edge_j,
                    closed=closed,
                ):
                    if not _valid_shape(
                        candidate,
                        closed=closed,
                        require_simple=False,
                    ):
                        continue
                    key = _points_key(candidate)
                    if key in seen:
                        continue
                    seen.add(key)
                    next_states.append(candidate)
                    if len(next_states) >= max_states:
                        break
                if len(next_states) >= max_states:
                    break
        if best is not None and best[0][0] == 0:
            break
        states = next_states
        if not states:
            break

    if best is None:
        return RepairResult(
            original,
            False,
            None,
            True,
            intersections,
        )

    diff, candidate = best[0][0], best[1]
    if diff > threshold:
        return RepairResult(original, False, diff, True, intersections)
    return RepairResult(
        candidate,
        not np.array_equal(original, candidate),
        diff,
        False,
        intersections,
    )


def repair_self_intersections(
    points: Sequence[Sequence[float]] | np.ndarray,
    shape_kind: ShapeKind,
    image_size: tuple[int, int],
    threshold: int,
    *,
    polyline_width: int = 1,
) -> np.ndarray:
    """Return repaired points, or normalized points when review is needed."""
    return repair_shape(
        points,
        shape_kind,
        image_size,
        threshold,
        polyline_width=polyline_width,
    ).points


def deduplicate_points(
    points: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Remove repeated vertices while preserving the first occurrence order."""
    array = _as_points(points)
    if len(array) < 2:
        return array

    seen: set[tuple[float, float]] = set()
    unique: list[np.ndarray] = []
    for point in array:
        key = (float(point[0]), float(point[1]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(point)
    return np.asarray(unique, dtype=np.float64)


def find_self_intersections(
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    closed: bool,
) -> list[tuple[int, int]]:
    """Return non-adjacent segment indexes that intersect."""
    array = _as_points(points)
    edge_count = len(array) if closed else max(len(array) - 1, 0)
    intersections: list[tuple[int, int]] = []
    for first in range(edge_count):
        first_end = (first + 1) % len(array) if closed else first + 1
        for second in range(first + 1, edge_count):
            # Consecutive segments share a vertex by definition.  The first
            # and last edges of a closed contour are consecutive as well.
            if second == first + 1:
                continue
            if closed and first == 0 and second == edge_count - 1:
                continue
            second_end = (second + 1) % len(array) if closed else second + 1
            if _segments_intersect(
                array[first],
                array[first_end],
                array[second],
                array[second_end],
            ):
                intersections.append((first, second))
    return intersections


def rasterize_polygon_nonzero(
    points: Sequence[Sequence[float]] | np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize a polygon using the SVG/CVAT non-zero winding rule.

    Pixels are sampled at their centers.  The scanline implementation avoids
    turning a self-intersecting source into a new contour; it is used only for
    comparing the original and candidate labels.
    """
    array = _as_points(points)
    mask = np.zeros((height, width), dtype=bool)
    if len(array) < 3:
        return mask

    min_y = max(0, math.floor(float(array[:, 1].min() - 0.5)))
    max_y = min(height - 1, math.ceil(float(array[:, 1].max() - 0.5)))
    if min_y > max_y:
        return mask

    for row in range(min_y, max_y + 1):
        y = row + 0.5
        crossings: list[tuple[float, int]] = []
        for first, second in zip(array, np.roll(array, -1, axis=0), strict=True):
            x1, y1 = float(first[0]), float(first[1])
            x2, y2 = float(second[0]), float(second[1])
            if abs(y2 - y1) <= _EPSILON:
                continue
            if y1 <= y < y2:
                direction = 1
            elif y2 <= y < y1:
                direction = -1
            else:
                continue
            x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            crossings.append((x, direction))

        crossings.sort(key=lambda item: item[0])
        winding = 0
        previous_x: float | None = None
        index = 0
        while index < len(crossings):
            x = crossings[index][0]
            if previous_x is not None and winding:
                start = max(0, math.ceil(previous_x - 0.5))
                stop = min(width, math.ceil(x - 0.5))
                if stop > start:
                    mask[row, start:stop] = True
            while index < len(crossings) and abs(crossings[index][0] - x) <= _EPSILON:
                winding += crossings[index][1]
                index += 1
            previous_x = x
    return mask


def rasterize_polyline(
    points: Sequence[Sequence[float]] | np.ndarray,
    width: int,
    height: int,
    line_width: int,
) -> np.ndarray:
    """Rasterize a centerline with the same integer rounding as CVAT export."""
    array = _as_points(points)
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(array) < 2:
        return mask.astype(bool)
    rounded = np.rint(array).astype(np.int32)
    cv2.polylines(
        mask,
        [rounded],
        isClosed=False,
        color=1,
        thickness=line_width,
    )
    return mask.astype(bool)


def _candidate_points(
    points: np.ndarray,
    first_edge: int,
    second_edge: int,
    *,
    closed: bool,
) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    if closed:
        count = len(points)
        offset = (second_edge - first_edge) % count
        if offset < 2 or offset > count - 2:
            return candidates
        rotated = np.concatenate((points[first_edge:], points[:first_edge]))
        intersection = _line_intersection(
            rotated[0],
            rotated[1],
            rotated[offset],
            rotated[(offset + 1) % count],
        )
        candidates.append(
            np.concatenate(
                (rotated[:1], rotated[1 : offset + 1][::-1], rotated[offset + 1 :])
            )
        )
        if intersection is not None:
            point = intersection.reshape(1, 2)
            # Remove the forward local loop, with and without preserving its
            # exact crossing point as a new vertex.
            candidates.append(np.concatenate((rotated[:1], rotated[offset + 1 :])))
            candidates.append(
                np.concatenate((rotated[:1], point, rotated[offset + 1 :]))
            )
            # The other cyclic path can be the tiny loop (for example when a
            # contour starts inside the bad section).
            candidates.append(np.concatenate((rotated[1 : offset + 1], rotated[:1])))
            candidates.append(
                np.concatenate((rotated[1 : offset + 1], point, rotated[:1]))
            )
    else:
        if second_edge <= first_edge + 1:
            return candidates
        intersection = _line_intersection(
            points[first_edge],
            points[first_edge + 1],
            points[second_edge],
            points[second_edge + 1],
        )
        candidates.append(
            np.concatenate(
                (
                    points[: first_edge + 1],
                    points[first_edge + 1 : second_edge + 1][::-1],
                    points[second_edge + 1 :],
                )
            )
        )
        if intersection is not None:
            point = intersection.reshape(1, 2)
            candidates.append(
                np.concatenate(
                    (points[: first_edge + 1], point, points[second_edge + 1 :])
                )
            )
            candidates.append(
                np.concatenate((points[: first_edge + 1], points[second_edge + 1 :]))
            )

    unique: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for candidate in candidates:
        key = _points_key(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _valid_shape(
    points: np.ndarray,
    *,
    closed: bool,
    require_simple: bool = True,
) -> bool:
    if not np.isfinite(points).all():
        return False
    if closed:
        if len(points) < 3 or abs(_signed_area(points)) <= _EPSILON:
            return False
    elif len(points) < 2:
        return False
    if closed:
        if any(
            np.linalg.norm(first - second) <= _EPSILON
            for first, second in zip(
                points,
                np.roll(points, -1, axis=0),
                strict=True,
            )
        ):
            return False
    elif any(
        np.linalg.norm(first - second) <= _EPSILON
        for first, second in zip(points[:-1], points[1:], strict=True)
    ):
        return False
    return not require_simple or not find_self_intersections(points, closed=closed)


def _pixel_diff(
    original: np.ndarray,
    candidate: np.ndarray,
    shape_kind: ShapeKind,
    width: int,
    height: int,
    polyline_width: int,
) -> int:
    if shape_kind == "polygon":
        first = rasterize_polygon_nonzero(original, width, height)
        second = rasterize_polygon_nonzero(candidate, width, height)
    else:
        first = rasterize_polyline(original, width, height, polyline_width)
        second = rasterize_polyline(candidate, width, height, polyline_width)
    return int(np.count_nonzero(np.logical_xor(first, second)))


def _as_points(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Points must have shape (N, 2), got {array.shape}")
    return array.copy()


def _points_key(points: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in np.round(points, decimals=9).reshape(-1))


def _signed_area(points: np.ndarray) -> float:
    next_points = np.roll(points, -1, axis=0)
    return float(
        0.5
        * np.sum(points[:, 0] * next_points[:, 1] - next_points[:, 0] * points[:, 1])
    )


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _orientation(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    return _cross(second - first, third - first)


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    scale = max(
        1.0,
        float(
            np.max(
                np.abs(np.vstack((first_start, first_end, second_start, second_end)))
            )
        ),
    )
    epsilon = _EPSILON * scale * scale
    first_orientation = _orientation(first_start, first_end, second_start)
    second_orientation = _orientation(first_start, first_end, second_end)
    third_orientation = _orientation(second_start, second_end, first_start)
    fourth_orientation = _orientation(second_start, second_end, first_end)

    if (
        (first_orientation > epsilon and second_orientation < -epsilon)
        or (first_orientation < -epsilon and second_orientation > epsilon)
    ) and (
        (third_orientation > epsilon and fourth_orientation < -epsilon)
        or (third_orientation < -epsilon and fourth_orientation > epsilon)
    ):
        return True
    return any(
        abs(orientation) <= epsilon and _on_segment(start, end, point, epsilon)
        for orientation, start, end, point in (
            (first_orientation, first_start, first_end, second_start),
            (second_orientation, first_start, first_end, second_end),
            (third_orientation, second_start, second_end, first_start),
            (fourth_orientation, second_start, second_end, first_end),
        )
    )


def _on_segment(
    start: np.ndarray,
    end: np.ndarray,
    point: np.ndarray,
    epsilon: float,
) -> bool:
    return bool(
        min(start[0], end[0]) - epsilon <= point[0] <= max(start[0], end[0]) + epsilon
        and min(start[1], end[1]) - epsilon
        <= point[1]
        <= max(start[1], end[1]) + epsilon
    )


def _line_intersection(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> np.ndarray | None:
    first_vector = first_end - first_start
    second_vector = second_end - second_start
    denominator = _cross(first_vector, second_vector)
    scale = max(
        1.0,
        float(
            np.max(
                np.abs(np.vstack((first_start, first_end, second_start, second_end)))
            )
        ),
    )
    if abs(denominator) <= _EPSILON * scale * scale:
        return None
    parameter = _cross(second_start - first_start, second_vector) / denominator
    point = first_start + parameter * first_vector
    return point if np.isfinite(point).all() else None
