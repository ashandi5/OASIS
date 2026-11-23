"""Automated checks for the OASIS RK4 thermal simulator."""

from __future__ import annotations

import math
import sys
import unittest
from copy import deepcopy
from pathlib import Path


CODE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIRECTORY))

import OASIS_Thermal_Simulator as oasis  # noqa: E402


class TestRK4(unittest.TestCase):
    def test_exponential_decay(self) -> None:
        """RK4 should closely reproduce y(t) = exp(-0.2t)."""
        state = [1.0]
        time_s = 0.0
        step_s = 0.1
        for _ in range(100):
            state = oasis.rk4_step(
                lambda _time, values: [-0.2 * values[0]],
                time_s,
                state,
                step_s,
            )
            time_s += step_s

        expected = math.exp(-0.2 * time_s)
        self.assertAlmostEqual(state[0], expected, places=7)

    def test_short_simulation_has_finite_temperatures(self) -> None:
        config = deepcopy(oasis.DEFAULT_CONFIG)
        config["simulation"]["duration_s"] = 10.0
        config["simulation"]["time_step_s"] = 1.0
        rows = oasis.simulate(config)

        self.assertEqual(len(rows), 11)
        temperature_columns = (
            "gpu_temperature_c",
            "secondary_temperature_c",
            "primary_temperature_c",
            "room_temperature_c",
        )
        for row in rows:
            for column in temperature_columns:
                self.assertTrue(math.isfinite(row[column]))

    def test_flow_sizing_is_positive(self) -> None:
        sizing = oasis.calculate_flow_sizing(oasis.DEFAULT_CONFIG)
        self.assertGreater(sizing["design_volume_flow_m3_s"], 0.0)
        self.assertGreater(sizing["design_volume_flow_gpm"], 0.0)


if __name__ == "__main__":
    unittest.main()
