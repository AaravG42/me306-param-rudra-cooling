# Bypass recirculation in a hybrid-cooled data centre: PARAM Rudra, IIT Bombay

Code for our ME 306 course project at IIT Bombay. It is a steady-state energy and exergy model of the
PARAM Rudra cooling system, from the processor cold plates to the cooling tower, used to optimise the
operating set-points and to find out how much energy bypass recirculation saves in the enclosed aisle.

## Layout

| Path | Contents |
|---|---|
| `model/params.py` | Shared constants: C-DAC design data, dead state, climate, limits, baseline set-points |
| `model/INTERFACE.md` | Interface between modules (function names, inputs, outputs, units) |
| `model/s1_liquid.py` | Coolant properties, microchannel cold plate, liquid loop, cold-plate exergy |
| `model/s2_air.py` | Node heat budget, chassis air temperatures, server fan curves and control |
| `model/s3_aisle.py` | Aisle mass and energy balances with leakage and recirculation, SHI/RHI/RTI, leakage network |
| `model/s4_irc.py` | In-row cooler coil (epsilon-NTU), fans, chilled-water solver, rear-door heat exchanger |
| `model/s5_plant.py` | Chiller, cooling tower, CDU, coupling of all modules, energy and exergy balances |
| `model/s5_optimize.py`, `s5_steps.py`, `s5_annual.py` | Optimiser, step-by-step savings, monthly energy |
| `model/sim_airpath.py` | Traditional vs recirculating aisle, part load, raised floor, leakage, air-only reference |
| `model/sim_maps.py` | Two-variable optimisation maps |
| `model/sim_workloads.py` | Benchmark workloads (thermal virus, LINPACK/DGEMM, STREAM, sporadic use) |
| `model/sim_gaps.py` | Component studies: coolant properties, cold-plate maps, fans, board measures, rear door, sensitivity |
| `model/sim_flow_network.py` | Pipe-network simulation of coolant distribution across 619 nodes |
| `model/sim_hourly.py` | Hour-by-hour Mumbai year (TMYx 2009-2023), cooling tower vs dry cooler |
| `model/results/` | Saved outputs (JSON) used for the report |
| `model/weather/` | Mumbai TMYx 2009-2023 weather file (climate.onebuilding.org) |
| `figscripts/make_figures.py` | Produces every data figure in the report into `figures/` |

## Running

```
pip install -r requirements.txt
cd model
python3 s5_plant.py            # baseline and optimised operating point
python3 sim_airpath.py         # main recirculation study
python3 sim_hourly.py          # annual simulation
cd ..
python3 figscripts/make_figures.py
```

A full run of all scripts takes about ten minutes on two cores.

## Team

Aarav Gupta, Farhan Al Zayed, Karunakaram Srikrishna Mourya, Shibam Sundar Mahakud, Tanishk Nath Goswami.
