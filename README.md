# OASIS Thermal Simulator

This folder contains the computational portion of the OASIS data-center cooling project. The Revit model defines the physical building and equipment layout. This code models the transient heat transfer, coolant-loop behavior, pump energy, cooling energy, and estimated PUE.

## Contents

| Path | Purpose |
|---|---|
| `OASIS_Thermal_Simulator.py` | Four-node thermal model and fourth-order Runge-Kutta solver |
| `OASIS_Thermal_Config.json` | Editable data-center, coolant, equipment, and load assumptions |
| `docs/OASIS_Thermal_Model.tex` | LaTeX derivation of the governing equations |
| `tests/test_solver.py` | Automated checks for RK4 accuracy, simulation output, and flow sizing |
| `examples/oasis_demo.json` | Summary from the demonstration scenario |
| `examples/oasis_demo.png` | Plot from the demonstration scenario |
| `requirements.txt` | Optional plotting dependency |

## Model overview

The model tracks four temperatures:

1. GPU and cold-plate thermal mass
2. Secondary IT coolant loop
3. Primary facility-water loop
4. Data-hall air

The governing energy balance for each node has the form

`C dT/dt = heat entering - heat leaving`.

The coupled differential equations are integrated with classical RK4. The supplied configuration intentionally includes a reduced-flow event and a reduced heat-rejection event so the design can be tested under abnormal operating conditions.

## Install

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The solver itself uses only the Python standard library. Matplotlib is used to create the graph.

## Run

From inside this `code` folder:

```bash
python OASIS_Thermal_Simulator.py --config OASIS_Thermal_Config.json --output-prefix results/oasis_results
```

Create the output directory first if needed:

```bash
mkdir results
```

The run creates:

- `oasis_results.csv`: complete time history
- `oasis_results.json`: design summary and pass/fail result
- `oasis_results.png`: temperature, heat-flow, and PUE graphs

## Test

```bash
python -m unittest discover -s tests -v
```

## Connecting the model to Revit

Replace the placeholder values in `OASIS_Thermal_Config.json` using Revit schedules and selected equipment data:

- rack count and heat per rack
- liquid cooling capture fraction
- coolant density and specific heat
- pipe-loop design flow and pressure drop
- pump and motor efficiencies
- heat-exchanger conductance and maximum rejection capacity
- room-air cooling capacity
- outdoor design temperature

The code does not modify the Revit model directly. Revit supplies geometry, equipment counts, pipe sizes, and schedule data. This solver then evaluates whether that proposed system remains within temperature limits over time.

## Important limitation

The included numerical values are concept-level placeholders, not stamped engineering design values. Validate all inputs against manufacturer data, local codes, ASHRAE guidance, and a qualified engineer before making real equipment or construction decisions.
