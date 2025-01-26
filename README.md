# Power Plant Energy Optimization

This project optimizes the energy allocation across various power plants based on a given total energy demand using Particle Swarm Optimization [PSO](https://en.wikipedia.org/wiki/Particle_swarm_optimization). It aims to minimize the production cost, considering each plant's efficiency, fuel costs, CO2 emissions, and other constraints.

PSO was the chosen solution since this is an optmization problem and PSO is a good meta heuristic for such problems, also because it was an algorithm that I studied when doing my master degree.

This is a proposed solution for the code challenge described in [README CHALLENGE.](README_CHALLENGE.md)

## Features

- Optimizes energy distribution for a set of power plants to meet a specified energy load.
- Considers various plant types including gas-fired, turbojet, and wind turbines.
- Uses Particle Swarm Optimization (PSO) to find the optimal energy allocation.
- Allows dynamic input via a REST API.

## Possible Inprovements

- Optmize the solution to maybe ignore wind power plants when %wind is 0.
- Fine tune the PSO algorithm to find best hyper parameters for the problem. That could be done using a simple grid search for the parameters n_particles, n_iterations, w_min, w_max, c1, c2.
- Try other meta-heuristics to solve the problem.
- Better error handling and logging.
- Define unit tests.
- Better modularization of the pso_powerplant code.
- Create a Dockerfile for the solution.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/nascimentocrafael/powerplant-coding-challenge
cd powerplant-coding-challenge/src
```

2. Create a virtual environment:
```bash
python3 -m venv ../env/powerplants
..\env\powerplants\Scripts\activate # On Windows
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Start the Flask server:
```bash
python api.py
```

5. The application will be running at http://127.0.0.1:8888/.

6. Dependencies

- Flask: A micro web framework for Python.
- NumPy: For numerical operations.
- random: For random number generation (used in PSO for velocity and position updates).

## API Documentation
/ (http://127.0.0.1:8888/) (POST)

This endpoint optimizes the energy allocation for a set of power plants based on the requested energy load.

### Request Body

- load (float): The total energy load that needs to be met (in MWh).
- fuels (dict): A dictionary containing the fuel parameters:
    - gas(euro/MWh) (float): The cost of gas per MWh.
    - kerosine(euro/MWh) (float): The cost of kerosine per MWh.
    - co2(euro/ton) (float): The CO2 cost per ton.
    - wind(%) (float): The percentage of wind energy used in wind turbines.
- powerplants (array of dicts): A list of power plants with the following fields:
    - name (string): The name of the power plant.
    - type (string): The type of power plant (gasfired, turbojet, or windturbine).
    - efficiency (float): The efficiency of the plant (between 0 and 1).
    - pmin (float): The minimum production capacity of the plant (in MWh).
    - pmax (float): The maximum production capacity of the plant (in MWh).
- n_particles (optional, int): The number of particles for Particle Swarm Optimization. Default is 50.

### Example Request
```json
{
    "load": 1000,
    "fuels": {
        "gas(euro/MWh)": 50,
        "kerosine(euro/MWh)": 100,
        "co2(euro/ton)": 20,
        "wind(%)": 100
    },
    "powerplants": [
        {
        "name": "Plant 1",
        "type": "gasfired",
        "efficiency": 0.9,
        "pmin": 50,
        "pmax": 200
        },
        {
        "name": "Plant 2",
        "type": "windturbine",
        "efficiency": 1.0,
        "pmin": 0,
        "pmax": 150
        }
    ]
}
```

### Example Response
```json
[
    {
        "name": "Plant 1",
        "p": 600.0
    },
    {
        "name": "Plant 2",
        "p": 400.0
    }
]
```