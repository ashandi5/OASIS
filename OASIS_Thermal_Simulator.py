#!/usr/bin/env python3
"""OASIS transient thermal model for a liquid-cooled GPU data center.

The model solves four coupled thermal nodes with classical RK4:
  1. GPU/cold-plate node
  2. Secondary IT coolant loop
  3. Primary facility-water loop
  4. Data-hall room air

Run with built-in demonstration values:
    python OASIS_Thermal_Simulator.py

Write an editable JSON configuration:
    python OASIS_Thermal_Simulator.py --write-example-config oasis_config.json

Run the edited configuration:
    python OASIS_Thermal_Simulator.py --config oasis_config.json

Outputs: CSV time series, JSON summary, and a PNG graph when matplotlib is
installed. All calculations use SI units unless a column explicitly says otherwise.
The built-in equipment values are placeholders and must be replaced with selected
manufacturer data before treating the results as a design prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


Vector = list[float]
Derivative = Callable[[float, Sequence[float]], Vector]


DEFAULT_CONFIG = {
    "project": {
        "name": "OASIS GPU Data Center",
        "notes": "Concept demonstration values; replace with Revit schedules and equipment data."
    },
    "simulation": {
        "duration_s": 7200.0,
        "time_step_s": 1.0,
        "initial_gpu_temperature_c": 42.0,
        "initial_secondary_temperature_c": 30.0,
        "initial_primary_temperature_c": 36.0,
        "initial_room_temperature_c": 24.0,
        "maximum_gpu_temperature_c": 85.0,
        "maximum_secondary_temperature_c": 55.0
    },
    "data_center": {
        "rack_count": 20,
        "rated_heat_per_rack_w": 50000.0,
        "liquid_capture_fraction": 0.85
    },
    "thermal": {
        "gpu_capacitance_j_per_k": 3000000.0,
        "secondary_capacitance_j_per_k": 4200000.0,
        "primary_capacitance_j_per_k": 21000000.0,
        "room_capacitance_j_per_k": 15000000.0,
        "gpu_to_secondary_conductance_w_per_k": 100000.0,
        "secondary_to_primary_conductance_w_per_k": 90000.0,
        "primary_to_ambient_conductance_w_per_k": 70000.0,
        "room_cooling_conductance_w_per_k": 18000.0,
        "maximum_heat_rejection_w": 1200000.0
    },
    "coolant": {
        "density_kg_per_m3": 997.0,
        "specific_heat_j_per_kg_k": 4180.0,
        "design_delta_t_k": 10.0,
        "design_flow_margin_fraction": 0.15
    },
    "pump": {
        "design_volumetric_flow_m3_s": 0.028,
        "design_pressure_drop_pa": 180000.0,
        "pump_efficiency": 0.78,
        "motor_efficiency": 0.92
    },
    "facility_energy": {
        "cooling_cop": 6.0,
        "ups_efficiency": 0.97,
        "fixed_other_power_w": 40000.0
    },
    "profiles": {
        "rack_load_fraction": [
            [0.0, 0.25], [600.0, 0.70], [1800.0, 1.00],
            [4200.0, 0.65], [6000.0, 0.25]
        ],
        "ambient_temperature_c": [[0.0, 43.0], [3600.0, 46.0]],
        "room_supply_temperature_c": [[0.0, 22.0]],
        "coolant_flow_fraction": [
            [0.0, 1.0], [3000.0, 0.55], [3900.0, 1.0]
        ],
        "heat_rejection_capacity_fraction": [
            [0.0, 1.0], [4800.0, 0.50], [5400.0, 1.0]
        ]
    }
}


@dataclass(frozen=True)
class ModelParameters:
    rack_count: int
    rated_heat_per_rack_w: float
    liquid_capture_fraction: float
    gpu_capacitance: float
    secondary_capacitance: float
    primary_capacitance: float
    room_capacitance: float
    g_gpu_secondary: float
    g_secondary_primary: float
    g_primary_ambient: float
    g_room: float
    max_heat_rejection_w: float
    design_flow_m3_s: float
    design_pressure_drop_pa: float
    pump_efficiency: float
    motor_efficiency: float
    cooling_cop: float
    ups_efficiency: float
    fixed_other_power_w: float


def step_profile(points: Iterable[Sequence[float]]) -> Callable[[float], float]:
    """Create a piecewise-constant profile from [[start_time_s, value], ...]."""
    data = sorted((float(t), float(v)) for t, v in points)
    if not data:
        raise ValueError("Every profile needs at least one [time_s, value] point.")

    def evaluate(t: float) -> float:
        value = data[0][1]
        for start, candidate in data:
            if t < start:
                break
            value = candidate
        return value

    return evaluate


def rk4_step(f: Derivative, t: float, y: Sequence[float], dt: float) -> Vector:
    """Perform one classical fourth-order Runge-Kutta step."""
    k1 = f(t, y)
    y2 = [value + 0.5 * dt * slope for value, slope in zip(y, k1)]
    k2 = f(t + 0.5 * dt, y2)
    y3 = [value + 0.5 * dt * slope for value, slope in zip(y, k2)]
    k3 = f(t + 0.5 * dt, y3)
    y4 = [value + dt * slope for value, slope in zip(y, k3)]
    k4 = f(t + dt, y4)
    return [
        value + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0
        for value, a, b, c, d in zip(y, k1, k2, k3, k4)
    ]


def parameters_from_config(config: dict) -> ModelParameters:
    dc = config["data_center"]
    th = config["thermal"]
    pump = config["pump"]
    energy = config["facility_energy"]
    params = ModelParameters(
        rack_count=int(dc["rack_count"]),
        rated_heat_per_rack_w=float(dc["rated_heat_per_rack_w"]),
        liquid_capture_fraction=float(dc["liquid_capture_fraction"]),
        gpu_capacitance=float(th["gpu_capacitance_j_per_k"]),
        secondary_capacitance=float(th["secondary_capacitance_j_per_k"]),
        primary_capacitance=float(th["primary_capacitance_j_per_k"]),
        room_capacitance=float(th["room_capacitance_j_per_k"]),
        g_gpu_secondary=float(th["gpu_to_secondary_conductance_w_per_k"]),
        g_secondary_primary=float(th["secondary_to_primary_conductance_w_per_k"]),
        g_primary_ambient=float(th["primary_to_ambient_conductance_w_per_k"]),
        g_room=float(th["room_cooling_conductance_w_per_k"]),
        max_heat_rejection_w=float(th["maximum_heat_rejection_w"]),
        design_flow_m3_s=float(pump["design_volumetric_flow_m3_s"]),
        design_pressure_drop_pa=float(pump["design_pressure_drop_pa"]),
        pump_efficiency=float(pump["pump_efficiency"]),
        motor_efficiency=float(pump["motor_efficiency"]),
        cooling_cop=float(energy["cooling_cop"]),
        ups_efficiency=float(energy["ups_efficiency"]),
        fixed_other_power_w=float(energy["fixed_other_power_w"]),
    )
    validate_parameters(params)
    return params


def validate_parameters(p: ModelParameters) -> None:
    if p.rack_count <= 0:
        raise ValueError("rack_count must be positive")
    if not 0.0 <= p.liquid_capture_fraction <= 1.0:
        raise ValueError("liquid_capture_fraction must be between 0 and 1")
    if not 0.0 < p.pump_efficiency <= 1.0:
        raise ValueError("pump_efficiency must be between 0 and 1")
    if not 0.0 < p.motor_efficiency <= 1.0:
        raise ValueError("motor_efficiency must be between 0 and 1")
    if not 0.0 < p.ups_efficiency <= 1.0:
        raise ValueError("ups_efficiency must be between 0 and 1")
    positive = {
        "rated_heat_per_rack_w": p.rated_heat_per_rack_w,
        "gpu_capacitance": p.gpu_capacitance,
        "secondary_capacitance": p.secondary_capacitance,
        "primary_capacitance": p.primary_capacitance,
        "room_capacitance": p.room_capacitance,
        "g_gpu_secondary": p.g_gpu_secondary,
        "g_secondary_primary": p.g_secondary_primary,
        "g_primary_ambient": p.g_primary_ambient,
        "g_room": p.g_room,
        "max_heat_rejection_w": p.max_heat_rejection_w,
        "design_flow_m3_s": p.design_flow_m3_s,
        "design_pressure_drop_pa": p.design_pressure_drop_pa,
        "cooling_cop": p.cooling_cop,
    }
    for name, value in positive.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive; received {value}")


def build_model(config: dict) -> tuple[Derivative, Callable[[float, Sequence[float]], dict]]:
    """Return the ODE derivative and an instantaneous diagnostic calculator."""
    p = parameters_from_config(config)
    profiles = config["profiles"]
    rack_load = step_profile(profiles["rack_load_fraction"])
    ambient = step_profile(profiles["ambient_temperature_c"])
    room_supply = step_profile(profiles["room_supply_temperature_c"])
    flow_fraction = step_profile(profiles["coolant_flow_fraction"])
    capacity_fraction = step_profile(profiles["heat_rejection_capacity_fraction"])

    def diagnostics(t: float, state: Sequence[float]) -> dict:
        gpu_c, secondary_c, primary_c, room_c = state
        load = max(rack_load(t), 0.0)
        flow = max(flow_fraction(t), 0.0)
        capacity = max(capacity_fraction(t), 0.0)
        it_heat_w = p.rack_count * p.rated_heat_per_rack_w * load
        liquid_heat_w = p.liquid_capture_fraction * it_heat_w
        room_heat_w = (1.0 - p.liquid_capture_fraction) * it_heat_w

        # A simple flow-sensitive conductance approximation. Replace this relation
        # with manufacturer curves when they are available.
        flow_heat_factor = max(flow, 0.02) ** 0.8
        g_gs = p.g_gpu_secondary * flow_heat_factor
        g_sp = p.g_secondary_primary * flow_heat_factor
        q_gpu_secondary = g_gs * (gpu_c - secondary_c)
        q_secondary_primary = g_sp * (secondary_c - primary_c)
        q_ua = max(p.g_primary_ambient * (primary_c - ambient(t)), 0.0)
        q_rejection_limit = p.max_heat_rejection_w * capacity
        q_rejected = min(q_ua, q_rejection_limit)
        q_room_removed = p.g_room * (room_c - room_supply(t))

        actual_flow_m3_s = p.design_flow_m3_s * flow
        # Affinity-law approximation: pressure loss scales approximately with flow^2.
        actual_pressure_pa = p.design_pressure_drop_pa * flow**2
        pump_power_w = (
            actual_pressure_pa * actual_flow_m3_s
            / (p.pump_efficiency * p.motor_efficiency)
        )
        heat_removal_for_power_w = max(q_rejected, 0.0) + max(q_room_removed, 0.0)
        cooling_electric_w = heat_removal_for_power_w / p.cooling_cop
        ups_loss_w = it_heat_w * (1.0 / p.ups_efficiency - 1.0)
        facility_power_w = (
            it_heat_w + pump_power_w + cooling_electric_w
            + ups_loss_w + p.fixed_other_power_w
        )
        pue = facility_power_w / it_heat_w if it_heat_w > 0.0 else math.nan

        return {
            "load_fraction": load,
            "flow_fraction": flow,
            "capacity_fraction": capacity,
            "ambient_temperature_c": ambient(t),
            "room_supply_temperature_c": room_supply(t),
            "it_heat_w": it_heat_w,
            "liquid_heat_w": liquid_heat_w,
            "room_heat_w": room_heat_w,
            "q_gpu_to_secondary_w": q_gpu_secondary,
            "q_secondary_to_primary_w": q_secondary_primary,
            "q_rejected_w": q_rejected,
            "q_room_removed_w": q_room_removed,
            "pump_power_w": pump_power_w,
            "cooling_electric_w": cooling_electric_w,
            "ups_loss_w": ups_loss_w,
            "facility_power_w": facility_power_w,
            "pue": pue,
        }

    def derivative(t: float, state: Sequence[float]) -> Vector:
        d = diagnostics(t, state)
        d_gpu = (d["liquid_heat_w"] - d["q_gpu_to_secondary_w"]) / p.gpu_capacitance
        d_secondary = (
            d["q_gpu_to_secondary_w"] - d["q_secondary_to_primary_w"]
        ) / p.secondary_capacitance
        d_primary = (
            d["q_secondary_to_primary_w"] - d["q_rejected_w"]
        ) / p.primary_capacitance
        d_room = (
            d["room_heat_w"] - d["q_room_removed_w"]
        ) / p.room_capacitance
        return [d_gpu, d_secondary, d_primary, d_room]

    return derivative, diagnostics


def simulate(config: dict) -> list[dict[str, float]]:
    sim = config["simulation"]
    duration = float(sim["duration_s"])
    dt = float(sim["time_step_s"])
    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration_s and time_step_s must be positive")

    state = [
        float(sim["initial_gpu_temperature_c"]),
        float(sim["initial_secondary_temperature_c"]),
        float(sim["initial_primary_temperature_c"]),
        float(sim["initial_room_temperature_c"]),
    ]
    derivative, diagnostics = build_model(config)
    rows: list[dict[str, float]] = []
    t = 0.0
    while t <= duration + 1e-9:
        d = diagnostics(t, state)
        rows.append({
            "time_s": t,
            "gpu_temperature_c": state[0],
            "secondary_temperature_c": state[1],
            "primary_temperature_c": state[2],
            "room_temperature_c": state[3],
            **d,
        })
        if t >= duration:
            break
        step = min(dt, duration - t)
        state = rk4_step(derivative, t, state, step)
        if not all(math.isfinite(value) for value in state):
            raise RuntimeError("Simulation became non-finite. Reduce the time step or check inputs.")
        t += step
    return rows


def calculate_flow_sizing(config: dict) -> dict[str, float]:
    dc = config["data_center"]
    coolant = config["coolant"]
    design_heat_w = float(dc["rack_count"]) * float(dc["rated_heat_per_rack_w"])
    cp = float(coolant["specific_heat_j_per_kg_k"])
    delta_t = float(coolant["design_delta_t_k"])
    density = float(coolant["density_kg_per_m3"])
    margin = float(coolant["design_flow_margin_fraction"])
    mass_flow = design_heat_w / (cp * delta_t)
    volume_flow = mass_flow / density
    design_volume_flow = volume_flow * (1.0 + margin)
    return {
        "design_heat_w": design_heat_w,
        "mass_flow_kg_s": mass_flow,
        "volume_flow_m3_s": volume_flow,
        "design_volume_flow_m3_s": design_volume_flow,
        "design_volume_flow_gpm": design_volume_flow * 15850.323,
    }


def make_summary(config: dict, rows: list[dict[str, float]]) -> dict:
    sim = config["simulation"]
    max_gpu = max(rows, key=lambda row: row["gpu_temperature_c"])
    max_secondary = max(rows, key=lambda row: row["secondary_temperature_c"])
    gpu_limit = float(sim["maximum_gpu_temperature_c"])
    secondary_limit = float(sim["maximum_secondary_temperature_c"])
    violations = [row for row in rows if row["gpu_temperature_c"] > gpu_limit]
    first_violation = violations[0]["time_s"] if violations else None
    finite_pue = [row["pue"] for row in rows if math.isfinite(row["pue"])]
    return {
        "project": config["project"]["name"],
        "status": "PASS" if not violations and max_secondary["secondary_temperature_c"] <= secondary_limit else "FAIL",
        "peak_gpu_temperature_c": max_gpu["gpu_temperature_c"],
        "peak_gpu_time_s": max_gpu["time_s"],
        "gpu_limit_c": gpu_limit,
        "first_gpu_limit_violation_s": first_violation,
        "peak_secondary_temperature_c": max_secondary["secondary_temperature_c"],
        "secondary_limit_c": secondary_limit,
        "average_pue": sum(finite_pue) / len(finite_pue),
        "peak_facility_power_w": max(row["facility_power_w"] for row in rows),
        "flow_sizing": calculate_flow_sizing(config),
    }


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, float]], config: dict) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    time_min = [row["time_s"] / 60.0 for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(time_min, [r["gpu_temperature_c"] for r in rows], label="GPU")
    axes[0].plot(time_min, [r["secondary_temperature_c"] for r in rows], label="Secondary coolant")
    axes[0].plot(time_min, [r["primary_temperature_c"] for r in rows], label="Primary loop")
    axes[0].plot(time_min, [r["room_temperature_c"] for r in rows], label="Room")
    axes[0].plot(time_min, [r["ambient_temperature_c"] for r in rows], "--", label="Ambient")
    axes[0].axhline(float(config["simulation"]["maximum_gpu_temperature_c"]), color="crimson", linestyle=":", label="GPU limit")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=3)

    axes[1].step(time_min, [r["it_heat_w"] / 1000.0 for r in rows], where="post", label="IT heat")
    axes[1].plot(time_min, [r["q_rejected_w"] / 1000.0 for r in rows], label="Outdoor rejection")
    axes[1].plot(time_min, [r["q_room_removed_w"] / 1000.0 for r in rows], label="Room cooling")
    axes[1].set_ylabel("Thermal power (kW)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].step(time_min, [r["flow_fraction"] for r in rows], where="post", label="Coolant flow fraction")
    axes[2].step(time_min, [r["capacity_fraction"] for r in rows], where="post", label="Heat-rejection capacity fraction")
    axes[2].plot(time_min, [r["pue"] for r in rows], label="PUE")
    axes[2].set_xlabel("Time (minutes)")
    axes[2].set_ylabel("Fraction / PUE")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.suptitle(config["project"]["name"] + " RK4 Thermal Simulation")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def verify_rk4() -> None:
    """Check RK4 against the analytical solution y(t)=y0*exp(-kt)."""
    k = 0.2
    y = [10.0]
    t = 0.0
    dt = 0.1
    for _ in range(100):
        y = rk4_step(lambda _t, state: [-k * state[0]], t, y, dt)
        t += dt
    exact = 10.0 * math.exp(-k * t)
    if abs(y[0] - exact) > 1e-7:
        raise RuntimeError(f"RK4 verification failed: numerical={y[0]}, exact={exact}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional JSON configuration")
    parser.add_argument("--write-example-config", type=Path, help="Write the built-in configuration and exit")
    parser.add_argument("--output-prefix", default="oasis_results", help="Prefix for CSV, JSON, and PNG outputs")
    args = parser.parse_args()

    if args.write_example_config:
        args.write_example_config.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.write_example_config}")
        return

    config = deepcopy(DEFAULT_CONFIG)
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8"))

    verify_rk4()
    rows = simulate(config)
    summary = make_summary(config, rows)
    prefix = Path(args.output_prefix)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    png_path = prefix.with_suffix(".png")
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plotted = write_plot(png_path, rows, config)

    print(f"{summary['status']}: peak GPU temperature = {summary['peak_gpu_temperature_c']:.2f} °C")
    print(f"Design flow = {summary['flow_sizing']['design_volume_flow_gpm']:.1f} gpm")
    print(f"Average estimated PUE = {summary['average_pue']:.3f}")
    print(f"Wrote {csv_path} and {json_path}" + (f" and {png_path}" if plotted else ""))


if __name__ == "__main__":
    main()
