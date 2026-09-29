"""BoxSet: the spatial primitives everything else is built on.

The raycast fuzz test is the safety net for the whole geometry layer — if
batched slab math is wrong, beams lie, the occupancy map fills with garbage,
and nothing downstream is debuggable. The reference below is written out
long-hand on purpose: it handles rays parallel to a slab with an explicit
branch rather than the signed-epsilon trick the real implementation uses, so
agreement between them means something.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.geometry import BoxSet


def ref_ray_box(origin, direction, lower, upper):
    """First entry distance for one ray into one box, or None."""
    tmin, tmax = -np.inf, np.inf
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                return None
            continue
        t1 = (lower[axis] - origin[axis]) / direction[axis]
        t2 = (upper[axis] - origin[axis]) / direction[axis]
        tmin = max(tmin, min(t1, t2))
        tmax = min(tmax, max(t1, t2))
    if tmax < max(tmin, 0.0):
        return None
    return max(tmin, 0.0)


def ref_raycast(boxes, origin, direction, max_range):
    d = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return max_range
    d = d / norm
    best = max_range
    for lo, hi in zip(boxes.lower, boxes.upper):
        t = ref_ray_box(origin, d, lo, hi)
        if t is not None:
            best = min(best, t)
    return best


@pytest.fixture
def two_boxes():
    return BoxSet(
        lower=[[2.0, 0.0, 0.0], [5.0, 5.0, 0.0]],
        upper=[[3.0, 4.0, 3.0], [6.0, 6.0, 1.0]],
        tags=["wall", "debris"],
    )


# --------------------------------------------------------------------- rays

def test_ray_hits_box_at_known_distance(two_boxes):
    d = two_boxes.raycast_batch([0.0, 1.0, 1.0], [[1.0, 0.0, 0.0]], 10.0)
    assert d[0] == pytest.approx(2.0)


def test_ray_misses_returns_max_range(two_boxes):
    d = two_boxes.raycast_batch([0.0, 1.0, 1.0], [[0.0, -1.0, 0.0]], 10.0)
    assert d[0] == pytest.approx(10.0)


def test_ray_beyond_max_range_is_clamped(two_boxes):
    d = two_boxes.raycast_batch([0.0, 1.0, 1.0], [[1.0, 0.0, 0.0]], 1.5)
    assert d[0] == pytest.approx(1.5)


def test_ray_starting_inside_a_box_returns_zero(two_boxes):
    d = two_boxes.raycast_batch([2.5, 1.0, 1.0], [[1.0, 0.0, 0.0]], 10.0)
    assert d[0] == 0.0


def test_ray_parallel_to_slab_does_not_spuriously_hit(two_boxes):
    # Travelling along +x at z=5, above the 3 m-tall box: must not register.
    d = two_boxes.raycast_batch([0.0, 1.0, 5.0], [[1.0, 0.0, 0.0]], 10.0)
    assert d[0] == pytest.approx(10.0)


def test_zero_length_direction_returns_max_range(two_boxes):
    d = two_boxes.raycast_batch([0.0, 1.0, 1.0], [[0.0, 0.0, 0.0]], 7.0)
    assert d[0] == pytest.approx(7.0)


def test_unnormalised_direction_matches_normalised(two_boxes):
    a = two_boxes.raycast_batch([0.0, 1.0, 1.0], [[1.0, 0.0, 0.0]], 10.0)
    b = two_boxes.raycast_batch([0.0, 1.0, 1.0], [[17.0, 0.0, 0.0]], 10.0)
    assert a[0] == pytest.approx(b[0])


def test_raycast_batch_matches_reference_on_random_rays():
    rng = np.random.default_rng(7)
    lower = rng.uniform(0.0, 18.0, (40, 3))
    boxes = BoxSet(lower=lower, upper=lower + rng.uniform(0.2, 2.5, (40, 3)))

    origins = rng.uniform(-2.0, 22.0, (250, 3))
    dirs = rng.normal(size=(250, 40, 3))
    for origin, ray_block in zip(origins, dirs):
        got = boxes.raycast_batch(origin, ray_block, 15.0)
        want = [ref_raycast(boxes, origin, d, 15.0) for d in ray_block]
        assert np.allclose(got, want, atol=1e-9), f"mismatch from origin {origin}"


def test_empty_set_returns_max_range():
    d = BoxSet.empty().raycast_batch([0, 0, 0], [[1, 0, 0], [0, 1, 0]], 4.0)
    assert np.allclose(d, 4.0)


# --------------------------------------------------------------- broadphase

def test_candidates_is_a_superset_of_every_hit():
    """The broadphase may over-report. It must never under-report."""
    rng = np.random.default_rng(11)
    lower = rng.uniform(0.0, 30.0, (120, 3))
    boxes = BoxSet(lower=lower, upper=lower + rng.uniform(0.2, 1.5, (120, 3)))
    boxes.build_grid(2.0)

    max_range = 6.0
    for origin in rng.uniform(0.0, 30.0, (60, 3)):
        near = boxes.candidates(origin, max_range)
        dirs = rng.normal(size=(24, 3))
        full = boxes.raycast_batch(origin, dirs, max_range)
        narrowed = boxes.raycast_batch(origin, dirs, max_range, idx=near)
        assert np.allclose(full, narrowed), f"broadphase dropped a hit at {origin}"


def test_grid_and_ungridded_queries_agree():
    rng = np.random.default_rng(3)
    lower = rng.uniform(0.0, 12.0, (30, 3))
    plain = BoxSet(lower=lower, upper=lower + 1.0)
    gridded = BoxSet(lower=lower, upper=lower + 1.0)
    gridded.build_grid(1.5)

    dirs = rng.normal(size=(16, 3))
    for origin in rng.uniform(0.0, 12.0, (20, 3)):
        a = plain.raycast_batch(origin, dirs, 8.0)
        b = gridded.raycast_batch(origin, dirs, 8.0, idx=gridded.candidates(origin, 8.0))
        assert np.allclose(a, b)


def test_build_grid_on_empty_set_is_harmless():
    empty = BoxSet.empty()
    empty.build_grid(2.0)
    assert len(empty.candidates([0, 0, 0], 5.0)) == 0


# ------------------------------------------------------------- penetrations

def test_sphere_clear_of_boxes_has_no_penetration(two_boxes):
    assert two_boxes.penetrations([0.0, 1.0, 1.0], 0.14) == []


def test_sphere_overlapping_one_face_pushes_straight_out(two_boxes):
    hits = two_boxes.penetrations([1.9, 1.0, 1.0], 0.2)
    assert len(hits) == 1
    normal, depth, index = hits[0]
    assert index == 0
    assert np.allclose(normal, [-1.0, 0.0, 0.0])
    assert depth == pytest.approx(0.1)


def test_concave_corner_reports_both_boxes():
    """The reason callers must resolve every contact, not just the deepest."""
    corner = BoxSet(
        lower=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        upper=[[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]],
    )
    hits = corner.penetrations([0.95, 0.95, 0.5], 0.2)
    assert sorted(h[2] for h in hits) == [0, 1]


def test_zero_radius_point_inside_a_box_still_penetrates(two_boxes):
    """`contains(p, 0.0)` depends on this — `dist2 < radius**2` alone is 0 < 0."""
    hits = two_boxes.penetrations([2.5, 2.0, 1.5], 0.0)
    assert len(hits) == 1
    assert hits[0][2] == 0


def test_zero_radius_point_outside_every_box_is_clear(two_boxes):
    assert two_boxes.penetrations([8.0, 2.0, 1.5], 0.0) == []


def test_sphere_centre_inside_box_escapes_through_nearest_face(two_boxes):
    # 0.1 m past the x=3.0 face, deep inside on every other axis.
    hits = two_boxes.penetrations([2.9, 2.0, 1.5], 0.14)
    assert len(hits) == 1
    normal, depth, _ = hits[0]
    assert np.allclose(normal, [1.0, 0.0, 0.0])
    assert depth == pytest.approx(0.1 + 0.14)


# ----------------------------------------------------------------- support

def test_support_z_finds_the_box_underneath(two_boxes):
    assert two_boxes.support_z([5.5, 5.5, 1.3], 0.14) == pytest.approx(1.0)


def test_support_z_ignores_surfaces_above(two_boxes):
    # Standing at the foot of the 3 m wall: its top is not what holds you up.
    assert two_boxes.support_z([2.5, 2.0, 0.2], 0.14) == -np.inf


def test_support_z_is_negative_infinity_over_open_floor(two_boxes):
    assert two_boxes.support_z([8.0, 2.0, 0.5], 0.14) == -np.inf


def test_support_z_counts_an_overhanging_sphere():
    ledge = BoxSet(lower=[[0.0, 0.0, 0.0]], upper=[[1.0, 1.0, 0.5]])
    assert ledge.support_z([1.1, 0.5, 0.7], 0.2) == pytest.approx(0.5)


# --------------------------------------------------------------- rasterise

def test_rasterize_marks_only_covered_cells():
    boxes = BoxSet(lower=[[1.0, 1.0, 0.0]], upper=[[2.0, 3.0, 3.0]])
    mask = boxes.rasterize(0.5, 0.0, 3.0, ([0.0, 0.0], [4.0, 4.0]))
    assert mask.shape == (8, 8)
    assert mask[2:4, 2:6].all()
    assert mask.sum() == 8


def test_rasterize_excludes_boxes_on_another_floor():
    upstairs = BoxSet(lower=[[1.0, 1.0, 3.0]], upper=[[2.0, 2.0, 6.0]])
    mask = upstairs.rasterize(0.5, 0.0, 3.0, ([0.0, 0.0], [4.0, 4.0]))
    assert not mask.any()
