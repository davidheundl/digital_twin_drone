"""Invariants the physics must never violate."""

import numpy as np
import pytest

from sim import World, load_config
from sim.drone import GRAVITY, derivative
from sim.vecmath import quat_to_euler

DT = 0.001


@pytest.fixture
def world():
    return World(load_config(overrides={"sensors": {"noise_enabled": False}}))


def fly(world, seconds, motors=None):
    if motors is not None:
        world.set_motors(motors)
    for _ in range(int(seconds / DT)):
        world.step(DT)


def boot_and_arm(world):
    world.power_on()
    fly(world, 0.6)
    assert world.arm()[0], "should be armable after boot"


def test_hover_throttle_exactly_cancels_gravity(world):
    """The advertised hover throttle must produce zero net acceleration."""
    state = world.state.copy()
    state.thrust = np.full(4, world.params.weight / 4)
    d = derivative(world.params, state, np.full(4, world.params.hover_throttle))
    assert np.allclose(d[3:6], 0.0, atol=1e-9)
    assert np.allclose(d[10:13], 0.0, atol=1e-9)


def test_sits_still_when_powered_off(world):
    start = world.state.position.copy()
    fly(world, 3.0)
    assert world.mode == "OFF"
    assert np.allclose(world.state.position, start, atol=1e-9)
    assert np.allclose(world.state.velocity, 0.0, atol=1e-9)


def test_motors_do_nothing_unless_armed(world):
    world.power_on()
    fly(world, 0.6)
    assert world.mode == "IDLE"
    fly(world, 1.0, motors=[1.0, 1.0, 1.0, 1.0])
    assert np.allclose(world.state.velocity, 0.0, atol=1e-9)
    assert world.state.position[2] == pytest.approx(world.params.body_radius)


def test_full_throttle_climbs(world):
    boot_and_arm(world)
    fly(world, 1.0, motors=[0.8] * 4)
    assert world.state.position[2] > 1.0
    assert world.airborne


def test_free_fall_matches_gravity(world):
    """With motors idle in mid-air the drone must fall at g (minus drag)."""
    boot_and_arm(world)
    world.teleport(position=[5.0, 4.0, 2.5])
    fly(world, 0.25, motors=[0.0] * 4)
    expected = 0.5 * abs(GRAVITY[2]) * 0.25 ** 2
    drop = 2.5 - world.state.position[2]
    assert drop == pytest.approx(expected, rel=0.05)


def test_differential_thrust_rolls_right(world):
    """More thrust on the right-hand motors (m0, m3) must roll right."""
    boot_and_arm(world)
    world.teleport(position=[5.0, 4.0, 2.0])
    h = world.params.hover_throttle
    fly(world, 0.2, motors=[h * 1.4, h * 0.6, h * 0.6, h * 1.4])
    roll = quat_to_euler(world.state.orientation)[0]
    assert roll < -0.05, "right-heavy thrust should produce negative (right) roll"


def test_yaw_from_spin_imbalance(world):
    boot_and_arm(world)
    world.teleport(position=[5.0, 4.0, 2.0])
    h = world.params.hover_throttle
    # m0/m1 are CW, m2/m3 are CCW; favouring the CW pair yaws one way.
    fly(world, 0.4, motors=[h * 1.3, h * 1.3, h * 0.7, h * 0.7])
    assert abs(quat_to_euler(world.state.orientation)[2]) > 0.02


@pytest.mark.parametrize("target", [
    [0.2, 4.0, 1.5], [9.8, 4.0, 1.5], [5.0, 0.2, 1.5], [5.0, 7.8, 1.5], [5.0, 4.0, 2.9],
])
def test_never_escapes_the_room(world, target):
    """Fired hard at every wall, the drone must stay inside the box."""
    boot_and_arm(world)
    world.teleport(position=[5.0, 4.0, 1.5])
    direction = np.array(target) - np.array([5.0, 4.0, 1.5])
    direction = direction / np.linalg.norm(direction)
    world.state.velocity = direction * 12.0
    fly(world, 3.0, motors=[0.0] * 4)
    radius = world.params.body_radius
    assert np.all(world.state.position >= -1e-6)
    assert np.all(world.state.position <= world.room.size + 1e-6)
    assert world.room.contains(world.state.position, radius * 0.999)


def test_hard_impact_crashes(world):
    boot_and_arm(world)
    world.teleport(position=[5.0, 4.0, 2.5])
    world.state.velocity = np.array([0.0, 0.0, -8.0])
    fly(world, 1.0, motors=[0.0] * 4)
    assert world.mode == "CRASHED"
    assert not world.arm()[0]


def test_gentle_landing_does_not_crash(world):
    boot_and_arm(world)
    world.teleport(position=[5.0, 4.0, 0.5])
    world.state.velocity = np.array([0.0, 0.0, -0.5])
    fly(world, 2.0, motors=[0.0] * 4)
    assert world.mode == "ARMED"
    assert not world.airborne


def test_battery_drains_under_load(world):
    boot_and_arm(world)
    before = world.battery.percent
    fly(world, 2.0, motors=[0.6] * 4)
    assert world.battery.percent < before


def test_state_never_becomes_nan(world):
    """Adversarial input must not be able to blow up the integrator."""
    boot_and_arm(world)
    rng = np.random.default_rng(7)
    for _ in range(300):
        world.set_motors(rng.uniform(-5.0, 5.0, 4))  # out-of-range on purpose
        for _ in range(10):
            world.step(DT)
    vector = world.state.to_vector()
    assert np.all(np.isfinite(vector))
    assert world.room.contains(world.state.position, 0.0)
