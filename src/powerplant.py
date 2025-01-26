class PowerPlant:
    """
    Represents a power plant with specific attributes related to its operation, cost, and environmental impact.

    This class is used to model a power plant with various properties, including its efficiency, 
    fuel costs, minimum and maximum energy production limits, and CO2 emissions. The class is 
    useful in simulations or optimizations related to energy allocation and cost minimization.

    Attributes:
        name (str): The name of the power plant.
        plant_type (str): The type of the power plant (e.g., coal, solar, gas, etc.).
        efficiency (float): The efficiency of the power plant in converting fuel to energy (0-1).
        pmin (float): The minimum energy production capacity (in MW) of the plant.
        pmax (float): The maximum energy production capacity (in MW) of the plant.
        cost_per_mwh (float): The cost per megawatt-hour (MWh) of energy produced.
        co2_cost (float): The CO2 cost per MWh of energy produced by the plant.
        wind_percentage (float): The percentage of energy generated by the plant that comes from wind (if applicable).

    Methods:
        __str__(): Returns a string representation of the power plant's attributes.
    """

    def __init__(self, name, plant_type, efficiency, pmin, pmax, fuel_cost, co2_cost, wind_percentage):
        """
        Initializes a new PowerPlant instance.

        Args:
            name (str): The name of the power plant.
            plant_type (str): The type of the power plant (e.g., coal, solar, gas, etc.).
            efficiency (float): The efficiency of the power plant (a value between 0 and 1).
            pmin (float): The minimum energy production capacity (in MW).
            pmax (float): The maximum energy production capacity (in MW).
            fuel_cost (float): The cost per MWh of energy produced by the plant.
            co2_cost (float): The CO2 cost per MWh of energy produced.
            wind_percentage (float): The percentage of energy generated by wind (if applicable).

        Initializes the instance variables with the provided values.
        """
        self.name = name
        self.type = plant_type
        self.efficiency = efficiency
        self.pmin = pmin
        self.pmax = pmax
        self.cost_per_mwh = fuel_cost
        self.co2_cost = co2_cost
        self.wind_percentage = wind_percentage

    def __str__(self):
        """
        Returns a string representation of the power plant's attributes.

        Returns:
            str: A formatted string containing the name, type, efficiency, minimum and maximum production, 
                 cost per MWh, and CO2 cost of the power plant.
        """
        return f"{self.name} {self.type} {self.efficiency} {self.pmin} {self.pmax} {self.cost_per_mwh} {self.co2_cost}"
