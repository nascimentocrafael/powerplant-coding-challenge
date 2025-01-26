import numpy as np
import random
from powerplant import PowerPlant

def sort_by_merit_order(power_plants: list[PowerPlant]):
    """
    Sorts a list of power plants based on their merit order.

    The merit order is calculated using the formula:
    `cost_per_mwh / efficiency + co2_cost`, where:
      - `cost_per_mwh`: The cost per megawatt-hour of the power plant.
      - `efficiency`: The efficiency of the power plant.
      - `co2_cost`: The cost of CO2 emissions for the power plant.

    Args:
        power_plants (list[PowerPlant]): A list of `PowerPlant` objects to be sorted.

    Returns:
        list[tuple[int, PowerPlant]]: A list of tuples containing the index and the corresponding
            `PowerPlant`, sorted by merit order.
    """
    # Sort power plants by merit order
    plant_merit_order = sorted(
        enumerate(power_plants),
        key=lambda x: x[1].cost_per_mwh / x[1].efficiency + x[1].co2_cost
    )
    return plant_merit_order

def clip_round_particles(power_plants: list[PowerPlant], particle: np.ndarray):
    """
    Clips and rounds a list of particle values to ensure they are within valid operating ranges 
    for a list of power plants.

    The function performs the following:
    - Clips the `particle` values to ensure they fall within the range defined by each power plant's 
      `pmin` and `pmax` attributes. The `pmin` value can be optionally set to zero, based on a random 
      choice between 0 and 1.
    - Rounds the clipped values to one decimal place.

    Args:
        power_plants (list[PowerPlant]): A list of `PowerPlant` objects with defined `pmin` and `pmax` ranges.
        particle (np.ndarray): Particle values to be clipped and rounded.

    Returns:
        np.ndarray: The clipped and rounded particle values.
    """
    # Clip the values again to ensure it's within the valid range
    particle = np.clip(particle,
                        [plant.pmin * random.choice([0, 1]) for plant in power_plants] # To also start with 0 when pmin > 0
                        , [plant.pmax for plant in power_plants])
    particle = np.round(particle * 10) / 10.0
    return particle

def fitness_function(particle: np.ndarray, power_plants: list[PowerPlant], total_energy: float):
    """
    Calculates the fitness value for a given particle in an energy allocation problem.

    The fitness function evaluates the cost of allocating energy production among power plants
    to meet a specified total energy demand while respecting individual plant constraints such as 
    minimum and maximum production levels. It considers production costs, CO2 costs, and efficiency, 
    penalizing solutions that fail to meet the exact energy requirement.

    Args:
        particle (np.ndarray): The energy allocation proposal for each power plant.
        power_plants (list[PowerPlant]): A list of `PowerPlant` objects with attributes such as `pmin`, `pmax`, 
                                         `efficiency`, `cost_per_mwh`, and `co2_cost`.
        total_energy (float): The total energy demand that needs to be satisfied.

    Returns:
        float: The total cost of the energy allocation if feasible. Returns `float('inf')` for infeasible solutions 
               (e.g., if the total production does not match the energy demand).
    """
    # Sort power plants by merit order
    plant_merit_order = sort_by_merit_order(power_plants)
    total_cost = 0
    total_production = 0
    energy_allocations = [0] * len(power_plants)

    # Iterate through plants in merit order
    for rank, (plant_idx, plant) in enumerate(plant_merit_order):
        remaining_energy_needed = total_energy - total_production

        # Calculate contribution
        max_contribution = min(remaining_energy_needed, particle[plant_idx], plant.pmax)
        contribution = max(max_contribution, plant.pmin)

        # Adjust for exact matching on the last plant
        if total_production + contribution > total_energy:
            contribution = total_energy - total_production

        # Update energy allocation and costs
        energy_allocations[plant_idx] = contribution
        effective_production = contribution / plant.efficiency
        production_cost = effective_production * plant.cost_per_mwh
        co2_cost = effective_production * plant.co2_cost if plant.co2_cost > 0 else 0

        total_cost += production_cost + co2_cost
        total_production += contribution

    # Penalize infeasible solutions
    if total_production != total_energy:
        return float('inf')  # Overproduction or underproduction

    return total_cost

def initialize_particles_with_merit_order(power_plants: list[PowerPlant], total_energy: float, n_particles: int):
    """
    Initializes a set of particles for an energy allocation problem based on the merit order of power plants.

    The function creates `n_particles` particle arrays, each representing an allocation of energy among the given
    power plants to satisfy a total energy demand. The allocation prioritizes power plants with lower costs and 
    higher efficiency (merit order) and ensures feasibility within each plant's minimum (`pmin`) and maximum (`pmax`) 
    constraints.

    The process includes:
    - Allocating energy to power plants in order of merit until the energy demand is met.
    - Distributing any remaining energy proportionally across plants with remaining capacity.
    - Clipping and rounding the final allocations to ensure they are within valid ranges.

    Args:
        power_plants (list[PowerPlant]): A list of `PowerPlant` objects with attributes like `pmin`, `pmax`, 
                                         `efficiency`, `cost_per_mwh`, and `co2_cost`.
        total_energy (float): The total energy demand that needs to be allocated across the power plants.
        n_particles (int): The number of particles (allocations) to generate.

    Returns:
        np.ndarray: A 2D array of shape `(n_particles, len(power_plants))` where each row represents an energy allocation 
                    across all power plants.
    """    
    particles = np.zeros((n_particles, len(power_plants)))

    # Sort power plants by merit order (lowest cost first, considering efficiency and CO2 cost)
    sorted_plants = sort_by_merit_order(power_plants)

    for i in range(n_particles):
        remaining_energy_needed = total_energy
        particle_allocation = np.zeros(len(power_plants))
        # First, allocate energy according to merit order (start from the most efficient plants)
        for idx, plant in sorted_plants:
            # Allocate energy proportionally to each plant in merit order
            max_possible = min(remaining_energy_needed, plant.pmax)
            allocated_energy = np.random.uniform(0, max_possible)  # Random allocation within min and max
            particle_allocation[idx] = allocated_energy
            remaining_energy_needed -= allocated_energy

            # If the remaining energy required is fulfilled, no need to allocate more
            if remaining_energy_needed <= 0:
                break

        # If there's still remaining energy to allocate after the initial pass, distribute it proportionally
        if remaining_energy_needed > 0:
            # List remaining capacities
            remaining_capacity = [plant.pmax - particle_allocation[idx] for idx, plant in sorted_plants]
            remaining_capacity_sum = np.sum(remaining_capacity)

            if remaining_capacity_sum > 0:
                for idx, plant in sorted_plants:
                    if remaining_capacity[idx] > 0:
                        # Distribute the remaining energy proportionally
                        additional_energy = remaining_energy_needed * (remaining_capacity[idx] / remaining_capacity_sum)
                        particle_allocation[idx] += additional_energy
                        remaining_energy_needed -= additional_energy
                    if remaining_energy_needed <= 0:
                        break

        # Clip the values to ensure they are within the valid range for each power plant
        particle_allocation = clip_round_particles(power_plants, particle_allocation)

        particles[i] = particle_allocation

    return particles

def initialize_velocities(v_max: float, particles: np.ndarray):
    """
    Initializes velocities for a set of particles in a particle swarm optimization problem.

    Each velocity is randomly generated within the range `[-v_max, v_max]` for every particle 
    dimension, ensuring that particles can move in either direction within the defined limits.

    Args:
        v_max (float): The maximum absolute value for the velocity in any dimension.
        particles (np.ndarray): A 2D array representing the particles, where each row is a particle 
                                 and each column is a dimension of the search space.

    Returns:
        np.ndarray: A 2D array of the same shape as `particles`, containing the initialized velocities 
                    for each particle and dimension.
    """    
    velocities = np.random.uniform(-v_max, v_max, particles.shape)
    return velocities

def update_position(particle: np.ndarray, velocity: np.ndarray
                    , power_plants: list[PowerPlant], total_energy: float, v_max: float):
    """
    Updates the position of a particle in a particle swarm optimization problem.

    The function adjusts the particle's position by adding the velocity and ensures that the updated position:
    - Respects the minimum (`pmin`) and maximum (`pmax`) constraints of each power plant.
    - Satisfies the total energy requirement (`total_energy`) by scaling down excess energy if the updated particle 
      produces more energy than required.
    - Prioritizes adjustments based on the merit order (cheapest power plants first).

    Args:
        particle (np.ndarray): The current position of the particle, representing energy allocations for each power plant.
        velocity (np.ndarray): The current velocity of the particle, representing changes to the energy allocations.
        power_plants (list[PowerPlant]): A list of `PowerPlant` objects with attributes like `pmin`, `pmax`, 
                                         `efficiency`, `cost_per_mwh`, and `co2_cost`.
        total_energy (float): The required total energy production to meet demand.
        v_max (float): The maximum velocity, used for scaling or limiting velocity adjustments.

    Returns:
        np.ndarray: The updated particle position, adjusted to satisfy constraints and total energy requirements.
    """    
    # Update particle position
    new_particle = particle + velocity

    # Clip to ensure that the new particle position stays within pmin and pmax for each power plant
    new_particle = clip_round_particles(power_plants, new_particle)

    # Calculate the total energy produced by the updated particle
    total_production = np.sum(new_particle)

    # If the total energy exceeds the required total energy, adjust the particle's production
    if total_production > total_energy:
        # Excess energy, so we need to scale down the energy
        excess_energy = total_production - total_energy

        # Sort plants based on merit order (cheapest first, considering efficiency and CO2 cost)
        merit_order = sort_by_merit_order(power_plants)
        
        # Scale down the energy from the plants, starting from the least costly (cheapest) plants
        for idx, plant in merit_order:
            # Calculate the available energy that can be reduced for each plant
            max_possible_reduction = new_particle[idx] - plant.pmin
            if max_possible_reduction > 0:
                reduction = min(max_possible_reduction, excess_energy)
                new_particle[idx] -= reduction
                excess_energy -= reduction
            if excess_energy <= 0:
                break

        # If there is still excess energy, distribute it proportionally to the plants that have capacity left
        if excess_energy > 0:
            remaining_capacity = [plant.pmax - new_particle[idx] for idx, plant in merit_order]
            remaining_capacity_sum = np.sum(remaining_capacity)

            if remaining_capacity_sum > 0:
                for idx, plant in merit_order:
                    if remaining_capacity[idx] > 0:
                        additional_energy = excess_energy * (remaining_capacity[idx] / remaining_capacity_sum)
                        new_particle[idx] += additional_energy
                        excess_energy -= additional_energy
                    if excess_energy <= 0:
                        break

    # If the total energy is below the required energy, we do nothing here since it's already valid.
    
    # Clip the values again to ensure it's within the valid range
    new_particle = clip_round_particles(power_plants, new_particle)

    return new_particle

def update_velocity(velocity: np.ndarray, particle: np.ndarray,
                     v_max: float, w: float, c1: float, c2: float
                     , personal_best_position: np.ndarray, global_best_position: np.ndarray):
    """
    Updates the velocity of a particle in a particle swarm optimization problem.

    The velocity is updated using the standard PSO formula:
    - The current velocity is scaled by an inertia weight (`w`).
    - A cognitive component attracts the particle towards its personal best position, weighted by `c1` and a random factor (`r1`).
    - A social component attracts the particle towards the global best position, weighted by `c2` and a random factor (`r2`).
    - The updated velocity is clamped to ensure it stays within the range `[-v_max, v_max]`.

    Args:
        velocity (np.ndarray): The current velocity of the particle.
        particle (np.ndarray): The current position of the particle.
        v_max (float): The maximum absolute velocity allowed in any dimension.
        w (float): The inertia weight, controlling the influence of the previous velocity.
        c1 (float): The cognitive coefficient, controlling the attraction to the personal best position.
        c2 (float): The social coefficient, controlling the attraction to the global best position.
        personal_best_position (np.ndarray): The particle's best position found so far.
        global_best_position (np.ndarray): The best position found by the swarm so far.

    Returns:
        np.ndarray: The updated velocity of the particle.
    """   
    r1, r2 = np.random.rand(), np.random.rand()
    velocity = (
        w * velocity
        + c1 * r1 * (personal_best_position - particle)
        + c2 * r2 * (global_best_position - particle)
    )
    velocity = np.clip(velocity, -v_max, v_max)  # Clamp velocity
    return velocity

def run(power_plants: list[PowerPlant], total_energy: float
        , n_particles: int=100, n_iterations: int=3000
        , w_min: float=0.4, w_max: float=0.9
        , c1: float=1.5, c2: float=1.5):
    """
    Runs a Particle Swarm Optimization (PSO) algorithm to optimize energy allocation among power plants.

    The function seeks to minimize the cost of producing a specified total energy demand (`total_energy`)
    by allocating energy across a set of power plants, while respecting each plant's constraints (e.g., 
    `pmin` and `pmax`). The PSO algorithm iteratively adjusts particle velocities and positions based on 
    personal and global best solutions to find the optimal allocation.

    Args:
        power_plants (list[PowerPlant]): A list of `PowerPlant` objects with attributes like `pmin`, `pmax`,
                                         `efficiency`, `cost_per_mwh`, and `co2_cost`.
        total_energy (float): The total energy demand that needs to be produced.
        n_particles (int, optional): The number of particles in the swarm. Defaults to 100.
        n_iterations (int, optional): The maximum number of iterations for the PSO loop. Defaults to 3000.
        w_min (float, optional): The minimum inertia weight for the velocity update. Defaults to 0.4.
        w_max (float, optional): The maximum inertia weight for the velocity update. Defaults to 0.9.
        c1 (float, optional): The cognitive coefficient, controlling attraction to personal best positions. Defaults to 1.5.
        c2 (float, optional): The social coefficient, controlling attraction to the global best position. Defaults to 1.5.

    Returns:
        tuple[np.ndarray, float]: 
            - The best particle position (`np.ndarray`), representing the optimal energy allocation.
            - The corresponding fitness score (`float`), representing the total cost of the allocation.

    Notes:
        - The algorithm uses a random seed for reproducibility.
        - The velocities are initialized within a defined range based on the power plants' constraints.
        - The inertia weight decays linearly over iterations to balance exploration and exploitation.

    Example:
        ```python
        best_position, best_score = run(power_plants, total_energy=500, n_particles=50, n_iterations=1000)
        print("Optimal Allocation:", best_position)
        print("Minimum Cost:", best_score)
        ```
    """
    np.random.seed(55)
    random.seed(55)

    # Velocity bounds
    v_max = 0.1 * (max([p.pmax for p in power_plants]) - min([p.pmin for p in power_plants]))

    # Initialize particles and velocities
    particles = initialize_particles_with_merit_order(power_plants, total_energy, n_particles)
    velocities = initialize_velocities(v_max, particles)

    # Initialize bests
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([fitness_function(p, power_plants, total_energy) for p in particles])
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    # PSO Loop
    for iteration in range(n_iterations):
        w = w_max - (w_max - w_min) * (iteration / n_iterations)  # Inertia weight decay
        for i in range(n_particles):
            # Update velocity
            velocities[i] = update_velocity(velocities[i]
                                            , particles[i]
                                            , v_max
                                            , w
                                            , c1
                                            , c2
                                            , personal_best_positions[i]
                                            , global_best_position)

            # Update position
            particles[i] = update_position(particles[i]
                                           , velocities[i]
                                           , power_plants
                                           , total_energy
                                           , v_max)

            # Evaluate fitness
            fitness = fitness_function(particles[i], power_plants, total_energy)

            # Update personal and global bests
            if fitness < personal_best_scores[i]:
                personal_best_positions[i] = particles[i].copy()
                personal_best_scores[i] = fitness

            if fitness < global_best_score:
                global_best_position = particles[i].copy()
                global_best_score = fitness
                # print(global_best_position)

    return global_best_position, global_best_score
