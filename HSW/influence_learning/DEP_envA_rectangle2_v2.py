import numpy as np
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from CombinedPlaceTebcNeurons2 import CombinedPlaceTebcNeurons
from trial_marker2 import determine_cs_us

def simulate_envA(agent, position_data, balance_distribution, responsive_distribution, tebc_responsive_neurons, cell_types):
    # Number of neurons
    N = 80

    # Define place cell parameters for EnvA
    place_cells_params_envA = {
        "n": N,  # Number of place cells
        "description": "gaussian",  # Adjust as needed for EnvA
        "widths": 0.20,  # Adjust as needed for EnvA
        "place_cell_centres": None,  # Adjust as needed for EnvA
        "wall_geometry": "geodesic",  # Adjust as needed for EnvA
        "min_fr": 0,  # Minimum firing rate
        "max_fr": 12,  # Maximum firing rate
        "save_history": True  # Save history for plotting
    }

    # Import trajectory into the agent

    times = position_data[0]  # Timestamps
    positions = position_data[1:3].T  # Positions (x, y)
    unique_times, indices = np.unique(times, return_index=True)
    unique_positions = positions[indices]
    agent.import_trajectory(times=unique_times, positions=unique_positions)


    # Create CombinedPlaceTebcNeurons instance for EnvA
    combined_neurons = CombinedPlaceTebcNeurons(agent, N, balance_distribution, responsive_distribution, place_cells_params_envA, tebc_responsive_neurons, cell_types)
    firing_rates = np.zeros((N, position_data.shape[1]))
    combined_neurons.calculate_smoothed_velocity(position_data)


    # Initialize last CS and US times
    last_CS_time = None
    last_US_time = None


    for index in range(unique_positions.shape[0]):
        # Current timestamp
        current_time = unique_times[index]

        # Update the agent's position
        if index >= len(agent.history['pos']):
            print(f"Index out of range: {index} >= {len(agent.history['pos'])}")
        break
        agent_position = agent.history['pos'][index]

        # Update the neuron states without saving to history
        combined_neurons.update(save_history_override=True)
        print(combined_neurons[1920:1930,0])

        # Determine if CS or US is present and update times
        trial_marker = position_data[3, index]
        cs_present, us_present = determine_cs_us(trial_marker)
        # ... [rest of the logic for updating CS and US times]

        # Update neuron states based on the current environment
        combined_neurons.update_state(agent_position, time_since_CS, time_since_US, index)

        # Manually save history after updating neuron states
        current_time = agent.t
        if len(combined_neurons.history["t"]) == 0 or current_time > combined_neurons.history["t"][-1]:
            combined_neurons.history["t"].append(current_time)
            combined_neurons.history["firingrate"].append(list(combined_neurons.firing_rates))
        else:
            print(f"Attempted to save duplicate history in envA file at time: {current_time}")



    return firing_rates, agent, combined_neurons
