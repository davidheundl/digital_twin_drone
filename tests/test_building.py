"""Building: the shell-plus-solids geometry the world collides against.

The empty-building cases pin down that a building with no interior is still
exactly the old bare room — that equivalence is what lets the default config,
the protocol's `room.size`, and the original physics tests stay unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.building import Building, Feature
from sim.geometry import BoxSet

ROOM_CFG = {
    "size": [10.0, 8.0, 3.0],
    "wall_restitution": 0.25,
    "wall_friction": 0.4,
    "crash_speed": 3.0,
}


@pytest.fixture
def empty():
    return Building.empty_room(ROOM_CFG)


@pytest.fixture
def partitioned():
    """A 10x8x3 room split by a wall at x=5 with a gap (doorway) at y in [3,5]."""
    solids = BoxSet(
        lower=[[4.9, 0.0, 0.0], [4.9, 5.0, 0.0]],
        upper=[[5.1, 3.0, 3.0], [5.1, 8.0, 3.0]],
        tags=["wall", "wall"],
    )
    return Building(shell_size=[10.0, 8.0, 3.0], restitution=0.25, friction=0.4,
                    crash_speed=3.0, solids=solids)


# ------------------------------------------------- equivalence with the old room

def test_empty_building_contains_like_the_old_room(empty):
    assert empty.contains(np.array([5.0, 4.0, 1.5]), 0.14)
    assert not empty.contains(np.array([0.05, 4.0, 1.5]), 0.14)
    assert not empty.contains(np.array([5.0, 4.0, 3.5]), 0.0)


def test_empty_building_raycast_hits_the_shell(empty):
    # From the centre, +x wall is 5 m away.
    assert empty.raycast(np.array([5.0, 4.0, 1.5]), np.array([1.0, 0.0, 0.0]), 12.0) \
        == pytest.approx(5.0)
    # Ceiling is 1.5 m up.
    assert empty.raycast(np.array([5.0, 4.0, 1.5]), np.array([0.0, 0.0, 1.0]), 12.0) \
        == pytest.approx(1.5)


def test_empty_building_raycast_clamps_to_max_range(empty):
    assert empty.raycast(np.array([5.0, 4.0, 1.5]), np.array([1.0, 0.0, 0.0]), 2.0) \
        == pytest.approx(2.0)


def test_empty_building_has_no_contacts_in_free_space(empty):
    assert empty.contacts(np.array([5.0, 4.0, 1.5]), 0.14) == []


def test_empty_building_ground_is_the_shell_floor(empty):
    assert empty.ground_z([5.0, 4.0]) == pytest.approx(0.0)


# --------------------------------------------------------------- shell contacts

def test_sphere_through_the_floor_is_pushed_up(empty):
    hits = empty.contacts(np.array([5.0, 4.0, 0.10]), 0.14)
    assert len(hits) == 1
    normal, depth = hits[0]
    assert np.allclose(normal, [0.0, 0.0, 1.0])
    assert depth == pytest.approx(0.04)


def test_shell_corner_reports_every_violated_face(empty):
    """Three faces at once — the case a single deepest-contact fix gets wrong."""
    hits = empty.contacts(np.array([0.05, 0.05, 0.05]), 0.14)
    assert len(hits) == 3
    normals = sorted(tuple(n) for n, _ in hits)
    assert normals == [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]


# --------------------------------------------------------------- interior walls

def test_interior_wall_blocks_a_beam(partitioned):
    # Looking +x from x=2: the partition at x=4.9 is 2.9 m away, not the far wall.
    d = partitioned.raycast(np.array([2.0, 1.0, 1.5]), np.array([1.0, 0.0, 0.0]), 12.0)
    assert d == pytest.approx(2.9)


def test_beam_passes_through_the_doorway(partitioned):
    # At y=4 the partition has a gap, so the beam reaches the far shell wall.
    d = partitioned.raycast(np.array([2.0, 4.0, 1.5]), np.array([1.0, 0.0, 0.0]), 12.0)
    assert d == pytest.approx(8.0)


def test_contains_rejects_a_point_inside_a_wall(partitioned):
    assert not partitioned.contains(np.array([5.0, 1.0, 1.5]), 0.0)
    assert partitioned.contains(np.array([5.0, 4.0, 1.5]), 0.14)


def test_sphere_against_an_interior_wall_is_pushed_clear(partitioned):
    hits = partitioned.contacts(np.array([4.8, 1.0, 1.5]), 0.14)
    assert len(hits) == 1
    normal, depth = hits[0]
    assert np.allclose(normal, [-1.0, 0.0, 0.0])
    assert depth == pytest.approx(0.04)


def test_scalar_and_batched_raycast_agree(partitioned):
    rng = np.random.default_rng(5)
    origin = np.array([3.0, 4.0, 1.5])
    dirs = rng.normal(size=(24, 3))
    batched = partitioned.raycast_batch(origin, dirs, 9.0)
    scalar = [partitioned.raycast(origin, d, 9.0) for d in dirs]
    assert np.allclose(batched, scalar)


def test_broadphase_narrowing_does_not_change_the_answer(partitioned):
    rng = np.random.default_rng(9)
    origin = np.array([3.0, 4.0, 1.5])
    dirs = rng.normal(size=(24, 3))
    near = partitioned.solids.candidates(origin, 9.0)
    assert np.allclose(partitioned.raycast_batch(origin, dirs, 9.0),
                       partitioned.raycast_batch(origin, dirs, 9.0, idx=near))


# ------------------------------------------------------------------- support

def test_support_z_rests_on_debris_not_the_floor():
    debris = BoxSet(lower=[[2.0, 2.0, 0.0]], upper=[[3.0, 3.0, 0.6]], tags=["debris"])
    b = Building(shell_size=[10.0, 8.0, 3.0], restitution=0.25, friction=0.4,
                 crash_speed=3.0, solids=debris)
    assert b.support_z(np.array([2.5, 2.5, 0.75]), 0.14) == pytest.approx(0.6)
    assert b.support_z(np.array([7.0, 2.5, 0.2]), 0.14) == pytest.approx(0.0)


def test_ground_z_reports_the_debris_top():
    debris = BoxSet(lower=[[2.0, 2.0, 0.0]], upper=[[3.0, 3.0, 0.6]], tags=["debris"])
    b = Building(shell_size=[10.0, 8.0, 3.0], restitution=0.25, friction=0.4,
                 crash_speed=3.0, solids=debris)
    assert b.ground_z([2.5, 2.5]) == pytest.approx(0.6)
    assert b.ground_z([7.0, 2.5]) == pytest.approx(0.0)


# ------------------------------------------------------------------ features

def test_features_are_queryable_by_kind():
    feats = [
        Feature("victim", "v1", np.array([2.0, 3.0, 0.3])),
        Feature("victim", "v2", np.array([8.0, 1.0, 0.3])),
        Feature("anchor", "a1", np.array([0.5, 0.5, 2.8])),
    ]
    b = Building(shell_size=[10.0, 8.0, 3.0], restitution=0.25, friction=0.4,
                 crash_speed=3.0, features=feats)
    assert [f.id for f in b.features_of("victim")] == ["v1", "v2"]
    assert len(b.features_of("anchor")) == 1
    assert b.features_of("decoy") == []


def test_drone_settles_on_debris_not_the_shell_floor():
    """The `room.lower[2]`-is-the-floor assumption, exercised through the world."""
    from sim.config import load_config
    from sim.world import World

    world = World(load_config())
    spawn = world.params.initial_position[:2]
    debris = BoxSet(lower=[[spawn[0] - 1.0, spawn[1] - 1.0, 0.0]],
                    upper=[[spawn[0] + 1.0, spawn[1] + 1.0, 0.7]], tags=["debris"])
    world.building = Building(shell_size=world.building.size, restitution=0.25, friction=0.4,
                          crash_speed=3.0, solids=debris)
    world.sensors.room = world.building
    world.reset()

    assert world.state.position[2] == pytest.approx(0.7 + world.params.body_radius)

    for _ in range(2000):
        world.step(0.001)
    assert world.state.position[2] == pytest.approx(0.7 + world.params.body_radius)
    assert not world.airborne


def test_features_are_not_geometry():
    """A victim must never block a beam or stop the drone."""
    feats = [Feature("victim", "v1", np.array([5.0, 4.0, 1.5]))]
    b = Building(shell_size=[10.0, 8.0, 3.0], restitution=0.25, friction=0.4,
                 crash_speed=3.0, features=feats)
    assert b.raycast(np.array([1.0, 4.0, 1.5]), np.array([1.0, 0.0, 0.0]), 12.0) \
        == pytest.approx(9.0)
    assert b.contacts(np.array([5.0, 4.0, 1.5]), 0.14) == []
