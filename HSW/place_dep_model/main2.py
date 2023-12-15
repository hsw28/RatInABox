import numpy as np
import scipy.io
import argparse
import itertools
import scipy.io
import scipy.stats as stats
from envA_rectangle2 import simulate_envA
from envB_oval2 import simulate_envB
from trial_marker2 import determine_cs_us
from learningTransfer2 import assess_learning_transfer
from actualVexpected2 import compare_actual_expected_firing
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from assign_tebc_types_and_responsiveness import assign_tebc_types_and_responsiveness
import os
import ratinabox
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs')
from cond_decoding_AvsB import cond_decoding_AvsB
from cebra import CEBRA
import cProfile
import pstats
import random



"""
Simulation Script for Neuronal Firing Rate Analysis
Note: advise using a conda environment:
    conda create -n ratinabox python=3.9
    conda activate ratinabox
    conda install numpy
    conda install scipy
    conda install matplotlib
    export PYTHONPATH="${PYTHONPATH}:/Users/Hannah/Programming/RatInABox"
    pip install shapely

For cebra in env:
    conda install pytorch::pytorch torchvision torchaudio -c pytorch <-- or other, look at https://pytorch.org/
    pip install cebra


          - adjust the BALANCE PARAMETER to adjust how much each cell incorporates spatial vs tEBC data
          - adjust responsive_values that specifies the percentage of neurons that are responsive to tEBC signals.

Usage:
    python main.py [--balance_values BALANCE_VALUES] [--balance_dist BALANCE_DIST] [--balance_std BALANCE_STD]
                   [--responsive_values RESPONSIVE_VALUES] [--responsive_type RESPONSIVE_TYPE]

Arguments:
    --balance_values  : Comma-separated list of balance values or means for Gaussian distribution.
                        Example: --balance_values 0.3,0.5,0.7
                        If not provided, a default value of 0.5 is used.
    --balance_dist    : Specifies the type of distribution for the balance factor.
                        Options are 'fixed' and 'gaussian'.
                        Default is 'fixed'.
                        Additional options is 'additive' wherein place and  tebc get cumulatily added. this only makes
                        sense with a balace value of 1
    --balance_std     : Standard deviation for the Gaussian distribution of the balance factor.
                        Only used if --balance_dist is set to 'gaussian'.
                        Default value is 0.1.
    --responsive_values: Comma-separated list of responsive rates or probabilities for distributions.
                         Example: --responsive_values 0.4,0.6,0.8
                         If not provided, a default value of 0.5 is used.
    --responsive_type : Type of distribution for the responsive rate.
                        Options are 'fixed', 'binomial', 'normal', 'poisson'.
                        Default is 'fixed'.
    --percent_place_cells: what percent of place cells you want

Examples:
    python main2.py --balance_values 0.3,0.5,0.7 --balance_dist gaussian --balance_std 0.1 --responsive_values 0.4,0.6,0.8 --responsive_type binomial --percent_place_cells .7

    python main2.py --balance_values 0.3,0.5 --balance_dist gaussian --balance_std 0.5 --responsive_values 0.4,0.6 --responsive_type binomial --percent_place_cells .7

    python main2.py --balance_values 0.3 --balance_dist gaussian --balance_std 0.1 --responsive_values 0.4 --responsive_type binomial --percent_place_cells .7

    python main2.py --balance_values 0.5 --balance_dist fixed --responsive_values 0.5 --responsive_type fixed --percent_place_cells .7

    python main2.py --balance_values 0.5,0.7 --balance_dist fixed --responsive_values 0.5 --responsive_type fixed --percent_place_cells .7

    python main2.py --balance_values 0,.25,.5,.75,1 --balance_dist fixed --responsive_values .25,.5,.75,1 --responsive_type fixed --percent_place_cells 1,.85,.7,.55


Description:
    The script conducts simulations to evaluate how different configurations of balance factors and responsive rates affect neuronal firing patterns. Balance can be set as a fixed value or as a mean for a Gaussian distribution. The responsive rate determines the proportion of neurons responsive to tEBC signals and can be set as a fixed value or sampled from specified distributions.

    The script loads position data from a MATLAB file, performs simulations in two environments, and assesses learning transfer and spatial coding accuracy. The script supports a grid search over multiple balance and responsive rate values, allowing a comprehensive analysis of various parameter combinations. Results are printed to the console.

Requirements:
    - Ensure all necessary modules and custom classes are correctly imported and configured.
    - Replace 'path_to_your_matlab_file.mat' with the actual path to your MATLAB file.
    - Adjust environment settings and neuron parameters as needed in the script.
"""

# Function to process the list-like arguments
def parse_list(arg_value):
    if ',' in arg_value:
        return [float(item) for item in arg_value.split(',')]
    else:
        return float(arg_value)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Simulation Script for Neuronal Firing Rate Analysis')
parser.add_argument('--balance_values', type=str, help='List of balance values or means for Gaussian distribution')
parser.add_argument('--balance_dist', choices=['fixed', 'gaussian', 'additive'], default='fixed', help='Distribution type for balance')
parser.add_argument('--balance_std', type=float, default=0.1, help='Standard deviation for Gaussian balance distribution')
parser.add_argument('--responsive_values', type=str, help='List of responsive rates or probabilities for distributions')
parser.add_argument('--responsive_type', choices=['fixed', 'binomial', 'normal', 'poisson'], default='fixed', help='Type of distribution for responsive rate')
parser.add_argument('--percent_place_cells', type=str, required=True, help='Percentage of place cells (single value or comma-separated list)')
args = parser.parse_args()

# Process the arguments
#balance_values = args.balance_values if args.balance_values else [0.5]
#responsive_values = args.responsive_values if args.responsive_values else [0.5]
balance_values = parse_list(args.balance_values)
responsive_values = parse_list(args.responsive_values)
percent_place_cells_values = parse_list(args.percent_place_cells)


save_directory = '/Users/Hannah/Programming/data_eyeblink/rat314/ratinabox_data/results'
ratinabox.figure_directory = save_directory
os.makedirs(save_directory, exist_ok=True)

# Construct the filename
results_filename = f"grid_search_results-balance-{args.balance_values}-{args.balance_dist}-std-{args.balance_std}-response-{args.responsive_values}-{args.responsive_type}-PCs-{args.percent_place_cells}.txt"
results_filepath = os.path.join(save_directory, results_filename)

def parse_list(arg_value):
    if isinstance(arg_value, list):
        return [float(item) for item in arg_value]
    else:
        return [float(item) for item in arg_value.split(',')]


def get_distribution_values(dist_type, params, size):
    if dist_type == 'fixed':
        return np.full(size, params[0])
    elif dist_type == 'gaussian':
        mean, std = params
        return np.clip(stats.norm(mean, std).rvs(size=size), 0, 1)
    elif dist_type == 'binomial':
        p = params[0]
        return np.random.binomial(1, p, size=size)
    elif dist_type == 'normal':
        mean, std = params
        return np.clip(stats.norm(mean, std).rvs(size=size), 0, 1)
    elif dist_type == 'poisson':
        lam = params[0]
        return np.clip(stats.poisson(lam).rvs(size=size), 0, 1)
    elif dist_type == 'additive':
        return np.full(size, 100)




# Load MATLAB file and extract position data
matlab_file_path = '/Users/Hannah/Programming/data_eyeblink/rat314/ratinabox_data/pos314.mat'  # Replace with your MATLAB file path
data = scipy.io.loadmat(matlab_file_path)
position_data_envA = data['envA314_522']  # Adjust variable name as needed
position_data_envB = data['envB314_524']  # Adjust variable name as needed

position_data_envA[1:3] = position_data_envA[1:3]/100
positions = position_data_envA[1:3].T
max_x = np.max(positions[:, 0])
max_y = np.max(positions[:, 1])
min_x = np.min(positions[:, 0])
min_y = np.min(positions[:, 1])

# Create environments for EnvA and EnvB
envA_params = {
    'boundary': [[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]],
    'boundary_conditions': 'solid'
}



envA = Environment(params=envA_params)

position_data_envB[1:3] = position_data_envB[1:3]/100
positions = position_data_envB[1:3].T
max_x = np.max(positions[:, 0])
max_y = np.max(positions[:, 1])
min_x = np.min(positions[:, 0])
min_y = np.min(positions[:, 1])


envB_params = {
    'boundary': [[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]],
    'boundary_conditions': 'solid'
}
envB = Environment(params=envB_params)

# Set parameters
num_neurons = 80
balance_values = parse_list(args.balance_values) if args.balance_values else [0.5]
responsive_values = parse_list(args.responsive_values) if args.responsive_values else [0.5]
percent_place_cells = parse_list(args.percent_place_cells) if args.percent_place_cells else [0.7]


balance_zero_done = False
responsive_zero_done = False


agentA = Agent(envA)
times = position_data_envA[0]  # Timestamps
positions = position_data_envA[1:3].T  # Positions (x, y)
unique_times, indices = np.unique(times, return_index=True)
unique_positions = positions[indices]
agentA.import_trajectory(times=unique_times, positions=unique_positions)


agentB = Agent(envB)
times = position_data_envB[0]  # Timestamps
positions = position_data_envB[1:3].T  # Positions (x, y)
unique_times, indices = np.unique(times, return_index=True)
unique_positions = positions[indices]
agentB.import_trajectory(times=unique_times, positions=unique_positions)

# Perform grid search over balance and responsive rates
with open(results_filepath, "w") as results_file:
    for balance_value, responsive_val, percent_place_cell in itertools.product(balance_values, responsive_values, percent_place_cells):
        # Use balance_value, responsive_val, and percent_place_cell in your simulation
        # Skip redundant zero value iterations
        print(balance_value)
        print(responsive_val)
        print(percent_place_cell)

        if balance_value == 0:
            if balance_zero_done and len(balance_values) > 1:
                continue
            balance_zero_done = True
        if responsive_val == 0:
            if responsive_zero_done and len(responsive_values) > 1:
                continue
            responsive_zero_done = True


        balance_distribution = get_distribution_values(args.balance_dist, [balance_value, args.balance_std], num_neurons)
        responsive_distribution = get_distribution_values(args.responsive_type, [responsive_val], num_neurons)

        # Simulate in Environment A
        tebc_responsive_neurons, cell_types = assign_tebc_types_and_responsiveness(num_neurons, responsive_distribution)

        # Profile the function
        #cProfile.runctx('simulate_envA(agentA, position_data_envA, balance_distribution, responsive_distribution, tebc_responsive_neurons, cell_types)', globals(), locals(), 'profile_stats.prof')
        #p = pstats.Stats('profile_stats.prof')
        #p.sort_stats('cumulative').print_stats(10)

        # Now run the function normally to capture its output
        spikesA, eyeblink_neuronsA, response_envA, agentA = simulate_envA(agentA, position_data_envA, balance_distribution, responsive_distribution, tebc_responsive_neurons, percent_place_cells_values, cell_types)
        # also want a percent of place cells metric


        balance_distribution_envA = eyeblink_neuronsA.balance_distribution
        tebc_responsive_rates_envA = eyeblink_neuronsA.tebc_responsive_neurons

        # Simulate in Environment B using the parameters from Environment A
        spikesB, eyeblink_neuronsB, response_envB, agentB = simulate_envB(agentB, position_data_envB, balance_distribution_envA, tebc_responsive_rates_envA, tebc_responsive_neurons, percent_place_cells_values, cell_types)



        ###PLOTTING
        '''
        ratinabox.autosave_plots = True
        ratinabox.stylize_plots()
        plt.show()
        agentA.plot_trajectory()
        plt.show()
        agentA.plot_position_heatmap()
        plt.show()
        agentA.plot_histogram_of_speeds()
        plt.show()
        agentB.plot_histogram_of_speeds()
        plt.show()
        combined_neuronsA.plot_rate_timeseries()
        plt.show()
        combined_neuronsA.plot_rate_map()
        plt.show()
        combined_neuronsA.plot_place_cell_locations()
        plt.show()
        '''




        #####save
        # Construct the full file paths
        filename_envA = f"response_envA_balance_{balance_value}_{args.balance_dist}_responsive_{responsive_val}_{args.responsive_type}_perPCs_{percent_place_cell}.npy"
        filename_envB = f"response_envB_balance_{balance_value}_{args.balance_dist}_responsive_{responsive_val}_{args.responsive_type}_perPCs_{percent_place_cell}.npy"
        full_path_envA = os.path.join(save_directory, filename_envA)
        full_path_envB = os.path.join(save_directory, filename_envB)
        # Save the response arrays to files


        np.save(full_path_envA, spikesA)

        np.save(full_path_envB, spikesB)
        ######

        # Assess learning transfer and other metrics
        #organize to run in cebra
        response_envA = np.transpose(spikesA)
        response_envB = np.transpose(spikesB)


        envA_eyeblink = position_data_envA[3].T
        response_envA = response_envA[envA_eyeblink > 0,:]
        envA_eyeblink = envA_eyeblink[envA_eyeblink > 0]
        envA_eyeblink = np.where(envA_eyeblink <= 5, 1, 2)

        envB_eyeblink = position_data_envB[3].T
        response_envB = response_envB[envB_eyeblink > 0,:]
        envB_eyeblink = envB_eyeblink[envB_eyeblink > 0]
        envB_eyeblink = np.where(envB_eyeblink <= 5, 1, 2)


        #run cebra decoding
        fract_control_all, fract_test_all = cond_decoding_AvsB(response_envA, envA_eyeblink, response_envB, envB_eyeblink)

        # Construct the identifier for this iteration
        identifier = f"{balance_value}_{args.balance_dist}_responsive_{responsive_val}_{args.responsive_type}_PCs_{args.percent_place_cells}.npy"

        #Append the results to the file
        results_file.write(f"Parameters: {identifier}\n")
        results_file.write(f"fract_control_all: {fract_control_all}\n")
        results_file.write(f"fract_test_all: {fract_test_all}\n")
        results_file.write("\n")  # Add a newline for readability


        # Print confirmation

        print(f"Saved results to {full_path_envA} and {full_path_envB}")
