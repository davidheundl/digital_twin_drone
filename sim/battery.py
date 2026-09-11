"""Simple battery model: coulomb counting with load-dependent voltage sag."""

from __future__ import annotations


class Battery:
    def __init__(self, cfg: dict):
        self.capacity_mah = float(cfg["capacity_mah"])
        self.nominal_voltage = float(cfg["nominal_voltage"])
        self.internal_resistance = float(cfg["internal_resistance"])
        self.idle_current = float(cfg["idle_current"])
        self.thrust_current_coeff = float(cfg["thrust_current_coeff"])
        self.warn_percent = float(cfg["warn_percent"])
        self.critical_percent = float(cfg["critical_percent"])
        self.reset()

    def reset(self) -> None:
        self.charge_mah = self.capacity_mah
        self.current = 0.0
        self._warned = False
        self._critical = False

    @property
    def percent(self) -> float:
        return max(0.0, 100.0 * self.charge_mah / self.capacity_mah)

    @property
    def empty(self) -> bool:
        return self.charge_mah <= 0.0

    @property
    def voltage(self) -> float:
        # Open-circuit voltage falls off mildly with depletion, then sags under load.
        soc = self.percent / 100.0
        open_circuit = self.nominal_voltage * (0.88 + 0.12 * soc)
        return max(0.0, open_circuit - self.current * self.internal_resistance)

    def update(self, dt: float, powered: bool, total_thrust_n: float) -> list[str]:
        """Advance the battery. Returns a list of newly-triggered event names."""
        if not powered:
            self.current = 0.0
            return []

        self.current = self.idle_current + self.thrust_current_coeff * max(0.0, total_thrust_n) ** 1.5
        self.charge_mah -= self.current * (dt / 3600.0) * 1000.0
        self.charge_mah = max(0.0, self.charge_mah)

        events: list[str] = []
        if not self._warned and self.percent <= self.warn_percent:
            self._warned = True
            events.append("battery_low")
        if not self._critical and self.percent <= self.critical_percent:
            self._critical = True
            events.append("battery_critical")
        if self.empty:
            events.append("battery_empty")
        return events
