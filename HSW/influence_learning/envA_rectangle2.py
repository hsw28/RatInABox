import numpy as np
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import Neurons, PlaceCells
from trial_marker2 import determine_cs_us
from TEBCcells import TEBC


def simulate_envA(agent, position_data, balance_distribution, responsive_distribution, tebc_responsive_neurons, cell_types):
    # Number of neurons
    N = 80

    # Define place cell parameters for EnvA
    PCs = PlaceCells(
        agent,
        params={
            "n": N,  # Number of place cells
            "description": "gaussian",  # Adjust as needed for EnvA
            "widths": 0.20,  # Adjust as needed for EnvA
            "place_cell_centres": None,  # Adjust as needed for EnvA
            "wall_geometry": "geodesic",  # Adjust as needed for EnvA
            "min_fr": 0,  # Minimum firing rate
            "max_fr": 12,  # Maximum firing rate
            "save_history": True  # Save history for plotting #JUST CHANGED
            }
        )

    # Import trajectory into the agent

    '''
    times = position_data[0]  # Timestamps
    positions = position_data[1:3].T  # Positions (x, y)
    unique_times, indices = np.unique(times, return_index=True)
    unique_positions = positions[indices]
    agent.import_trajectory(times=unique_times, positions=unique_positions)
    '''

    # Create instances for EnvA
    #combined_neurons = CombinedPlaceTebcNeurons(agent, N, balance_distribution, responsive_distribution, place_cells_params_envA, tebc_responsive_neurons, cell_types)
    eyeblink_neurons = TEBC(agent, N, balance_distribution, responsive_distribution, PCs.params, tebc_responsive_neurons, cell_types)

    firing_rates = np.zeros((N, position_data.shape[1]))
    eyeblink_neurons.calculate_smoothed_velocity(position_data)

    # Initialize last CS and US times
    last_CS_time = None
    last_US_time = None


    # Simulation loop

    times = position_data[0,:]

    for index in range(len(times)):
        # Current timestamp
        current_time = times[index]

        # Update the agent
        agent.update()

        # Determine if CS or US is present
        trial_marker = position_data[3, index]
        cs_present, us_present = determine_cs_us(trial_marker)

        # Update last CS/US time if necessary
        if cs_present and (last_CS_time is None or times[index] > last_CS_time):
            last_CS_time = times[index]
        if us_present and (last_US_time is None or times[index] > last_US_time):
            last_US_time = times[index]

        # Calculate time since CS and US
        time_since_CS = times[index] - last_CS_time if last_CS_time is not None else -1
        time_since_US = times[index] - last_US_time if last_US_time is not None else -1

        # Retrieve the agent's current position from the history
        agent_position = agent.history['pos'][index]

        PCs.update()

        place_firing = (1 - eyeblink_neurons.balance_distribution)*(PCs.history['firingrate'])
        place_firing_recent = place_firing[-1]


        tebc_firing = eyeblink_neurons.update_my_state(time_since_CS, index)

        # Store firing rates

        firing_rates[:, index] = tebc_firing+place_firing_recent

    # Return the firing rates for further analysis


    return eyeblink_neurons, firing_rates, agent
