from flask import Flask, request, jsonify
import pso_powerplant as pso
from powerplant import PowerPlant
    
# Flask setup
app = Flask(__name__)

def initialize_power_plants_from_input(fuel_params: dict, power_plants_params: list[dict]):
    """
    Initializes a list of PowerPlant instances based on fuel parameters and power plant data.

    This function takes the fuel parameters and the power plant configuration, and creates `PowerPlant` objects.
    It calculates the relevant cost parameters (fuel cost, CO2 cost) for each plant and adjusts the power output
    of wind turbines based on the specified wind percentage.

    Args:
        fuel_params (dict): A dictionary containing fuel-related data, such as fuel cost per MWh and CO2 cost per ton.
                            Expected keys include:
                            - 'gas(euro/MWh)': Fuel cost for gas-fired plants.
                            - 'kerosine(euro/MWh)': Fuel cost for turbojet plants.
                            - 'co2(euro/ton)': CO2 cost for gas-fired plants.
                            - 'wind(%)': The wind energy percentage used for wind turbine plants.
        power_plants_params (list[dict]): A list of dictionaries, where each dictionary contains the parameters for a power plant.
                                     Expected keys include:
                                     - 'name': The name of the power plant.
                                     - 'type': The type of plant ('gasfired', 'turbojet', or 'windturbine').
                                     - 'efficiency': The efficiency of the plant.
                                     - 'pmin': The minimum power output (in MW).
                                     - 'pmax': The maximum power output (in MW).

    Returns:
        list[PowerPlant]: A list of `PowerPlant` instances initialized with the provided parameters.

    Raises:
        ValueError: If an unsupported plant type is encountered in the input data.

    Notes:
        - For 'gasfired' plants, fuel cost is taken from 'gas(euro/MWh)', and CO2 cost is calculated based on 'co2(euro/ton)'.
        - For 'turbojet' plants, fuel cost is taken from 'kerosine(euro/MWh)', and CO2 cost is set to 0.
        - For 'windturbine' plants, fuel and CO2 costs are set to 0, and the power output is scaled according to the wind percentage.
    """
    # Initialize power plants based on the input
    power_plants = []

    for plant_data in power_plants_params:
        pmax = plant_data['pmax']
        wind_perc = fuel_params.get('wind(%)', 0) / 100
        if plant_data['type'] == 'gasfired':
            fuel_cost = fuel_params['gas(euro/MWh)']
            co2_cost = fuel_params['co2(euro/ton)'] * 0.3
        elif plant_data['type'] == 'turbojet':
            fuel_cost = fuel_params['kerosine(euro/MWh)']
            co2_cost = 0
        elif plant_data['type'] == 'windturbine':
            fuel_cost = 0  # Wind energy has no fuel cost
            co2_cost = 0  # Wind energy has no CO2 cost
            pmax *= wind_perc
        else:
            raise ValueError(f"Unsupported plant type: {plant_data['type']}")

        power_plants.append(PowerPlant(
            name=plant_data['name'],
            plant_type=plant_data['type'],
            efficiency=plant_data['efficiency'],
            pmin=plant_data['pmin'],
            pmax=pmax,
            fuel_cost=fuel_cost,
            co2_cost=co2_cost,
            wind_percentage=wind_perc
        ))

    return power_plants

@app.route('/', methods=['POST'])
def optimize_energy():
    """
    Optimizes the energy production allocation across power plants based on the requested energy load.

    This endpoint accepts a JSON payload containing the total energy load, fuel parameters, and power plant data.
    It uses Particle Swarm Optimization (PSO) to determine the optimal energy allocation for each power plant, 
    considering their efficiency, fuel costs, CO2 emissions, and other constraints.

    The response contains the optimal energy allocation for each power plant in the system.

    Args:
        request (flask.Request): The incoming HTTP request containing a JSON payload with the following fields:
            - 'load' (float): The total energy load that needs to be met.
            - 'fuels' (dict): A dictionary containing the fuel parameters such as fuel costs and CO2 costs.
            - 'powerplants' (list[dict]): A list of dictionaries with data about each power plant (e.g., name, type, efficiency, etc.).
            - 'n_particles' (optional, int): The number of particles to use in the PSO optimization (default is 50).

    Returns:
        flask.Response: A JSON response containing the optimal energy allocation for each power plant. The response is a 
                        list of dictionaries, where each dictionary contains:
            - 'name' (str): The name of the power plant.
            - 'p' (float): The optimized energy production allocated to the power plant.

    Raises:
        KeyError: If any required field is missing from the input JSON (e.g., 'load', 'fuels', or 'powerplants').
                  A 400 error is returned with a descriptive message.

    Example:
        Request:
            POST http://127.0.0.1:8888
            {
                "load": 1000,
                "fuels": {
                    "gas(euro/MWh)": 50,
                    "kerosine(euro/MWh)": 100,
                    "co2(euro/ton)": 20,
                    "wind(%)": 100
                },
                "powerplants": [
                    {"name": "Plant 1", "type": "gasfired", "efficiency": 0.9, "pmin": 50, "pmax": 200},
                    {"name": "Plant 2", "type": "windturbine", "efficiency": 1.0, "pmin": 0, "pmax": 150}
                ]
            }

        Response:
            [
                {"name": "Plant 1", "p": 600.0},
                {"name": "Plant 2", "p": 400.0}
            ]
    """
    try:
        # Parse the incoming JSON data
        data = request.get_json()
        total_energy = data['load']
        fuel_params = data['fuels']
        plant_data = data['powerplants']
        n_particles = data.get('n_particles', 50)  # Number of particles for PSO
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter: {str(e)}'}), 400

    # Initialize PowerPlants from input data
    power_plants = initialize_power_plants_from_input(fuel_params, plant_data)

    # Run PSO
    best_solution, _ = pso.run(power_plants, total_energy, n_particles)

    # Prepare the response data in the requested format
    response = []

    # Sort power plants by merit order
    plant_merit_order = pso.sort_by_merit_order(power_plants)

    for i, _ in enumerate(best_solution):
        name = plant_merit_order[i][1].name
        energy = best_solution[plant_merit_order[i][0]]
        response.append({"name": name, "p": energy})

    return jsonify(response)

# Run the Flask application on port 8888
if __name__ == '__main__':
    app.run(debug=True, port=8888)
