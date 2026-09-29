"""Beam layout and the batched raycast that feeds it.

The length-invariant test matters more than it looks: telemetry's `beams` and
hello's `beam_angles_deg` / `beam_elevations_deg` are zipped together by every
consumer. If they ever drift apart, readings get attributed to the wrong
direction and the map fills with plausible-looking garbage.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.building import Building
from sim.config import load_config
from sim.geometry import BoxSet
from sim.world import World
from server.protocol import hello_message
from sim.harness import StubRuntime


@pytest.fixture
def world():
    return World(load_config(overrides={"sensors": {"noise_enabled": False}}))


def boot(world):
    world.power_on()
    for _ in range(600):
        world.step(0.001)
    return world


# ------------------------------------------------------------------- layout

def test_default_layout_has_the_expected_rings(world):
    # 16 horizontal + 4 up + 4 down + 1 zenith
    assert world.sensors.beam_count == 25


def test_every_beam_direction_is_a_unit_vector(world):
    norms = np.linalg.norm(world.sensors.beam_dirs_body, axis=1)
    assert np.allclose(norms, 1.0)


def test_zenith_beam_points_straight_up(world):
    assert np.allclose(world.sensors.beam_dirs_body[-1], [0.0, 0.0, 1.0], atol=1e-12)


def test_first_beam_points_forward(world):
    assert np.allclose(world.sensors.beam_dirs_body[0], [1.0, 0.0, 0.0], atol=1e-12)


def test_layout_includes_off_horizontal_beams(world):
    elevations = set(np.round(world.sensors.beam_elevation_deg, 3))
    assert elevations == {0.0, 30.0, -30.0, 90.0}


def test_direction_arrays_all_have_one_entry_per_beam(world):
    n = world.sensors.beam_count
    assert len(world.sensors.beam_azimuth_deg) == n
    assert len(world.sensors.beam_elevation_deg) == n
    assert world.sensors.beam_dirs_body.shape == (n, 3)


# ---------------------------------------------------------------- invariant

def test_hello_direction_arrays_match_the_telemetry_beam_count(world):
    boot(world)
    hello = hello_message(world, StubRuntime(world.cfg))
    beams = world.snapshot()["sensors"]["beams"]

    assert hello["sensors"]["beam_count"] == len(beams)
    assert len(hello["sensors"]["beam_angles_deg"]) == len(beams)
    assert len(hello["sensors"]["beam_elevations_deg"]) == len(beams)


# ------------------------------------------------------------------ reading

def test_batched_beams_match_per_ray_raycasts(world):
    boot(world)
    state = world.state
    from sim.vecmath import quat_to_matrix
    R = quat_to_matrix(state.orientation)

    beams = world.snapshot()["sensors"]["beams"]
    max_range = float(world.cfg["sensors"]["beams"]["max_range"])
    for i, direction_body in enumerate(world.sensors.beam_dirs_body):
        want = world.room.raycast(state.position, R @ direction_body, max_range)
        want = min(max(0.0, want - world.params.body_radius), max_range)
        assert beams[i] == pytest.approx(want, abs=1e-3), f"beam {i} disagrees"


def test_beams_are_clamped_to_max_range(world):
    boot(world)
    beams = world.snapshot()["sensors"]["beams"]
    max_range = float(world.cfg["sensors"]["beams"]["max_range"])
    assert all(0.0 <= b <= max_range + 1e-6 for b in beams)


def test_zenith_beam_measures_the_ceiling(world):
    boot(world)
    snap = world.snapshot()
    height = snap["truth"]["position"][2]
    ceiling = float(world.room.size[2])
    expected = ceiling - height - world.params.body_radius
    assert snap["sensors"]["beams"][-1] == pytest.approx(expected, abs=1e-3)


def test_interior_wall_shortens_the_forward_beam(world):
    """Same drone, same pose — a wall 2 m ahead must show up on beam 0."""
    boot(world)
    before = world.snapshot()["sensors"]["beams"][0]

    x, y = world.state.position[:2]
    wall = BoxSet(lower=[[x + 2.0, y - 2.0, 0.0]], upper=[[x + 2.2, y + 2.0, 3.0]])
    world.building = Building(shell_size=world.building.size, restitution=0.25, friction=0.4,
                          crash_speed=3.0, solids=wall)
    world.sensors.room = world.building
    world.sensors._candidate_idx = None

    after = world.snapshot()["sensors"]["beams"][0]
    assert after == pytest.approx(2.0 - world.params.body_radius, abs=1e-3)
    assert after < before


# --------------------------------------------------------------- broadphase

def test_candidate_cache_refreshes_when_the_drone_moves():
    cfg = load_config(overrides={"sensors": {"noise_enabled": False}})
    world = World(cfg)
    wall = BoxSet(lower=[[8.0, 0.0, 0.0]], upper=[[8.2, 8.0, 3.0]])
    world.building = Building(shell_size=world.building.size, restitution=0.25, friction=0.4,
                          crash_speed=3.0, solids=wall, broadphase_cell=2.0)
    world.sensors.room = world.building
    boot(world)

    # Far from the wall, then teleported next to it: the cached candidate set
    # must not survive the jump.
    world.snapshot()
    world.teleport(position=[7.0, 4.0, 1.5])
    forward = world.snapshot()["sensors"]["beams"][0]
    assert forward == pytest.approx(1.0 - world.params.body_radius, abs=1e-3)
