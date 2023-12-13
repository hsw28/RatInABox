import numpy as np
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import Neurons, PlaceCells
from trial_marker2 import determine_cs_us
from TEBCcells import TEBC
import cProfile
import pstats
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def simulate_envA(agent, position_data, balance_distribution, responsive_distribution, tebc_responsive_neurons, percent_place_cells, cell_types):
    N = 80  # Number of neurons

    # Define place cell parameters for EnvA
    PCs = PlaceCells(agent, params={
        "n": N,
        "description": "gaussian",
        "widths": 0.20,
        "place_cell_centres": None,
        "wall_geometry": "geodesic",
        "min_fr": 0,
        "max_fr": 1, #treating this as a percent
        "save_history": True
        #"noise_std":0.15
    })

    if isinstance(percent_place_cells, list):
        percent_place_cells = float(percent_place_cells[0])
    percent_to_zero_out = (1 - percent_place_cells)
    num_elements_to_zero_out = int(N * percent_to_zero_out)

    # Randomly select indices to zero out
    indices_to_zero_out = random.sample(range(N), num_elements_to_zero_out)
    eyeblink_neurons = TEBC(agent, N, balance_distribution, responsive_distribution, PCs.params, tebc_responsive_neurons, cell_types)

    firing_rates = np.zeros((N, position_data.shape[1]))
    spikes = np.zeros((N, position_data.shape[1]))

    eyeblink_neurons.calculate_smoothed_velocity(position_data)

    last_CS_time = None
    last_US_time = None

    times = position_data[0, :]
    trial_markers = position_data[3, :]

    for index, (current_time, trial_marker) in enumerate(zip(times, trial_markers)):
        agent.update()

        #figuring out TEBC firing
        cs_present, us_present = determine_cs_us(trial_marker)

        if cs_present:
            last_CS_time = current_time if last_CS_time is None else max(last_CS_time, current_time)

        time_since_CS = current_time - last_CS_time if last_CS_time is not None else -1
        tebc_firing = eyeblink_neurons.update_my_state(time_since_CS, index)


        #figuring out place cell firing
        PCs.update()

        #velocity contribution modeled from:    #Spatial and Behavioral Correlates of Hippocampal Neuronal Activity
                                                #Sustained activation of hippocampal pyramidal cells by ‘space clamping’ in a running wheel

        vel = eyeblink_neurons.smoothed_velocity[index];
        if vel < 0.02:
            place_response = 0
        else:
            FR = np.array(PCs.history['firingrate'][-1])
            coefficients = [-3.26092478e-04, 1.74074978e-02, 8.36619150e-02, 1.16059441]
            firing_rate_function = np.poly1d(coefficients)
            FR_mod = firing_rate_function(vel*100)
            place_firing = FR*(FR_mod/7.5)
            place_firing[indices_to_zero_out] = 0
            if eyeblink_neurons.balance_distribution[0] != 100:
                place_firing = (1 - eyeblink_neurons.balance_distribution) * place_firing

        #combine
        firing_rates[:, index] = tebc_firing + place_firing #this is per 1/7.5 seconds
        cell_spikes = np.random.uniform(0, 1, size=(N,)) < (tebc_firing + place_firing)
        spikes[:, index] = cell_spikes

    spikes = spikes.astype(int)
    return spikes, eyeblink_neurons, firing_rates, agent
