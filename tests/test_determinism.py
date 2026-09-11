"""Same seed + same commands must produce a bit-identical trace.

This is what makes lockstep mode useful for debugging: a run that crashes your
controller can be replayed exactly.
"""

import numpy as np

from sim import World, load_config


def run_episode(seed: int, motor_sequence):
    cfg = load_config(overrides={"sensors": {"seed": seed, "noise_enabled": True}})
    world = World(cfg)
    world.power_on()
    for _ in range(600):
        world.step(0.001)
    world.arm()

    trace = []
    for motors in motor_sequence:
        world.set_motors(motors)
        for _ in range(5):
            world.step(0.001)
        trace.append(world.snapshot())
    return trace


def _make_sequence():
    rng = np.random.default_rng(42)
    h = 0.4
    return [np.clip(h + rng.normal(0, 0.05, 4), 0, 1) for _ in range(200)]


def test_same_seed_same_commands_bit_identical():
    sequence = _make_sequence()
    trace_a = run_episode(seed=99, motor_sequence=sequence)
    trace_b = run_episode(seed=99, motor_sequence=sequence)

    assert len(trace_a) == len(trace_b)
    for frame_a, frame_b in zip(trace_a, trace_b):
        assert frame_a["truth"] == frame_b["truth"]
        assert frame_a["sensors"] == frame_b["sensors"]
        assert frame_a["battery"] == frame_b["battery"]


def test_different_seed_different_sensor_noise():
    sequence = _make_sequence()
    trace_a = run_episode(seed=1, motor_sequence=sequence)
    trace_b = run_episode(seed=2, motor_sequence=sequence)

    # Ground truth is seed-independent (noise doesn't feed back into physics
    # here since the reference loop isn't driving off sensors)...
    assert trace_a[-1]["truth"] == trace_b[-1]["truth"]
    # ...but raw sensor readings must differ because biases are seeded.
    assert trace_a[0]["sensors"]["accel"] != trace_b[0]["sensors"]["accel"]
