"""Command validation and rejection behaviour - the boundary your controller hits."""

import numpy as np
import pytest

from sim import World, load_config
from server.protocol import CommandError, dispatch, parse_message


class FakeRuntime:
    """Minimal stand-in for server.runtime.Runtime."""
    def __init__(self):
        self.paused = False
        self.time_scale = 1.0
        self.commands = 0
        self.steps_requested = 0

    def note_command(self):
        self.commands += 1

    def request_steps(self, n):
        self.steps_requested += n


@pytest.fixture
def world():
    return World(load_config())


@pytest.fixture
def runtime():
    return FakeRuntime()


def send(world, runtime, **msg):
    return dispatch(parse_message(msg), world, runtime)


def test_unknown_command_rejected():
    with pytest.raises(CommandError):
        parse_message({"type": "fly_to_the_moon"})


def test_missing_type_rejected():
    with pytest.raises(CommandError):
        parse_message({"motors": [0, 0, 0, 0]})


def test_non_object_message_rejected():
    with pytest.raises(CommandError):
        parse_message(["not", "an", "object"])


@pytest.mark.parametrize("bad_motors", [
    [1, 2, 3],           # wrong length
    [1, 2, 3, 4, 5],      # wrong length
    ["a", "b", "c", "d"],  # wrong type
    [float("nan"), 0, 0, 0],  # non-finite
    "not a list",
])
def test_set_motors_rejects_malformed_input(world, runtime, bad_motors):
    with pytest.raises(CommandError):
        send(world, runtime, type="set_motors", motors=bad_motors)


def test_set_motors_clamps_out_of_range(world, runtime):
    ok, reason, _ = send(world, runtime, type="set_motors", motors=[5.0, -3.0, 0.5, 1.0])
    assert ok
    assert np.allclose(world.motor_cmd, [1.0, 0.0, 0.5, 1.0])


def test_arm_rejected_before_power_on(world, runtime):
    ok, reason, _ = send(world, runtime, type="arm")
    assert not ok
    assert "not powered" in reason


def test_arm_rejected_during_boot(world, runtime):
    send(world, runtime, type="power_on")
    ok, reason, _ = send(world, runtime, type="arm")
    assert not ok
    assert "booting" in reason


def test_double_arm_rejected(world, runtime):
    world.power_on()
    for _ in range(600):
        world.step(0.001)
    ok1, _, _ = send(world, runtime, type="arm")
    ok2, reason2, _ = send(world, runtime, type="arm")
    assert ok1 and not ok2
    assert "already armed" in reason2


def test_teleport_outside_room_rejected(world, runtime):
    ok, reason, _ = send(world, runtime, type="teleport", position=[999, 0, 0])
    assert not ok
    assert "outside" in reason


def test_teleport_bad_shape_rejected(world, runtime):
    with pytest.raises(CommandError):
        send(world, runtime, type="teleport", position=[1, 2])


def test_step_bounds_enforced(world, runtime):
    with pytest.raises(CommandError):
        send(world, runtime, type="step", steps=0)
    with pytest.raises(CommandError):
        send(world, runtime, type="step", steps=1_000_000)
    ok, _, _ = send(world, runtime, type="step", steps=50)
    assert ok
    assert runtime.steps_requested == 50


def test_time_scale_bounds_enforced(world, runtime):
    with pytest.raises(CommandError):
        send(world, runtime, type="set_time_scale", scale=0)
    with pytest.raises(CommandError):
        send(world, runtime, type="set_time_scale", scale=5000)
    ok, _, _ = send(world, runtime, type="set_time_scale", scale=4.0)
    assert ok
    assert runtime.time_scale == 4.0


def test_get_state_returns_snapshot(world, runtime):
    ok, reason, extra = send(world, runtime, type="get_state")
    assert ok
    assert "state" in extra
    assert extra["state"]["mode"] == "OFF"


def test_ping_pong(world, runtime):
    ok, reason, _ = send(world, runtime, type="ping")
    assert ok and reason == "pong"
