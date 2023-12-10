import numpy as np
import scipy.io
import argparse
import itertools
import scipy.io
import scipy.stats as stats
from envA_rectangle2 import simulate_envA
from envB_oval2 import simulate_envB
from CombinedPlaceTebcNeurons2 import CombinedPlaceTebcNeurons
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
    --balance_std     : Standard deviation for the Gaussian distribution of the balance factor.
                        Only used if --balance_dist is set to 'gaussian'.
                        Default value is 0.1.
    --responsive_values: Comma-separated list of responsive rates or probabilities for distributions.
                         Example: --responsive_values 0.4,0.6,0.8
                         If not provided, a default value of 0.5 is used.
    --responsive_type : Type of distribution for the responsive rate.
                        Options are 'fixed', 'binomial', 'normal', 'poisson'.
                        Default is 'fixed'.

Examples:
    python main2.py --balance_values 0.3,0.5,0.7 --balance_dist gaussian --balance_std 0.1 --responsive_values 0.4,0.6,0.8 --responsive_type binomial

    python main2.py --balance_values 0.3,0.5 --balance_dist gaussian --balance_std 0.5 --responsive_values 0.4,0.6 --responsive_type binomial

    python main2.py --balance_values 0.3 --balance_dist gaussian --balance_std 0.1 --responsive_values 0.4 --responsive_type binomial

    python main2.py --balance_values 0.5 --balance_dist fixed --responsive_values 0.5 --responsive_type fixed

    python main2.py --balance_values 0.5,0.7 --balance_dist fixed --responsive_values 0.5 --responsive_type fixed

    python main2.py --balance_values 0,.25,.5,.75,1 --balance_dist fixed --responsive_values .25,.5,.75,1 --responsive_type fixed


Description:
    The script conducts simulations to evaluate how different configurations of balance factors and responsive rates affect neuronal firing patterns. Balance can be set as a fixed value or as a mean for a Gaussian distribution. The responsive rate determines the proportion of neurons responsive to tEBC signals and can be set as a fixed value or sampled from specified distributions.

    The script loads position data from a MATLAB file, performs simulations in two environments, and assesses learning transfer and spatial coding accuracy. The script supports a grid search over multiple balance and responsive rate values, allowing a comprehensive analysis of various parameter combinations. Results are printed to the console.

Requirements:
    - Ensure all necessary modules and custom classes are correctly imported and configured.
    - Replace 'path_to_your_matlab_file.mat' with the actual path to your MATLAB file.
    - Adjust environment settings and neuron parameters as needed in the script.
"""

save_directory='/Users/Hannah/Programming/data_eyeblink/rat314/ratinabox_data/results'
ratinabox.figure_directory = '/Users/Hannah/Programming/data_eyeblink/rat314/ratinabox_data/results'
# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

#naming our file
parser = argparse.ArgumentParser()
parser.add_argument('--balance_values', type=str, required=True)
parser.add_argument('--balance_dist', type=str, required=True)
parser.add_argument('--balance_std', type=float, required=False)
parser.add_argument('--responsive_values', type=str, required=True)
parser.add_argument('--responsive_type', type=str, required=True)
# Parse the arguments
args = parser.parse_args()
# Function to process the list-like arguments
def process_list_arg(arg):
    return ','.join(arg.split(','))
# Process the balance_values and responsive_values
balance_values_str = process_list_arg(args.balance_values)
responsive_values_str = process_list_arg(args.responsive_values)
# Construct the filename
results_filename = f"grid_search_results-balance-{balance_values_str}-{args.balance_dist}-std-{args.balance_std}-response-{responsive_values_str}-{args.responsive_type}.txt"
results_filepath = os.path.join(save_directory, results_filename)



def parse_list(arg_value):
    if isinstance(arg_value, list):
        return [float(item) for item in arg_value]
    else:
        return [float(item) for item in arg_value.split(',')]


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Simulation Script for Neuronal Firing Rate Analysis')
parser.add_argument('--balance_values', type=parse_list, help='List of balance values or means for Gaussian distribution')
parser.add_argument('--balance_dist', choices=['fixed', 'gaussian'], default='fixed', help='Distribution type for balance')
parser.add_argument('--balance_std', type=float, default=0.1, help='Standard deviation for Gaussian balance distribution')
parser.add_argument('--responsive_values', type=parse_list, help='List of responsive rates or probabilities for distributions')
parser.add_argument('--responsive_type', choices=['fixed', 'binomial', 'normal', 'poisson'], default='fixed', help='Type of distribution for responsive rate')
args = parser.parse_args()


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

# Load MATLAB file and extract position data
matlab_file_path = '/Users/Hannah/Programming/data_eyeblink/rat314/ratinabox_data/pos314.mat'  # Replace with your MATLAB file path
data = scipy.io.loadmat(matlab_file_path)
position_data_envA = data['envA314_522']  # Adjust variable name as needed
position_data_envB = data['envB314_524']  # Adjust variable name as needed

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
    for balance_value, responsive_val in itertools.product(balance_values, responsive_values):
        # Skip redundant zero value iterations
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
        cProfile.runctx('simulate_envA(agentA, position_data_envA, balance_distribution, responsive_distribution, tebc_responsive_neurons, cell_types)', globals(), locals(), 'profile_stats.prof')
        p = pstats.Stats('profile_stats.prof')
        p.sort_stats('cumulative').print_stats(10)

        # Now run the function normally to capture its output
        eyeblink_neurons, response_envA, agentA = simulate_envA(agentA, position_data_envA, balance_distribution, responsive_distribution, tebc_responsive_neurons, cell_types)



        balance_distribution_envA = eyeblink_neurons.balance_distribution
        tebc_responsive_rates_envA = eyeblink_neurons.tebc_responsive_neurons

        # Simulate in Environment B using the parameters from Environment A
        #response_envB, agentB, combined_neuronsB = simulate_envB(agentB, position_data_envB, balance_distribution_envA, tebc_responsive_rates_envA, tebc_responsive_neurons, cell_types)



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
        filename_envA = f"response_envA_balance_{balance_value}_{args.balance_dist}_responsive_{responsive_val}_{args.responsive_type}.npy"
        filename_envB = f"response_envB_balance_{balance_value}_{args.balance_dist}_responsive_{responsive_val}_{args.responsive_type}.npy"
        full_path_envA = os.path.join(save_directory, filename_envA)
        full_path_envB = os.path.join(save_directory, filename_envB)
        # Save the response arrays to files


        np.save(full_path_envA, response_envA)
        '''
        np.save(full_path_envB, combined_neuronsB.history['firingrate'])
        ######

        # Assess learning transfer and other metrics
        #organize to run in cebra
        response_envA = np.transpose(response_envA)
        response_envB = np.transpose(response_envB)


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
        identifier = f"{balance_value}_{args.balance_dist}_responsive_{responsive_val}_{args.responsive_type}"

        # Append the results to the file
        results_file.write(f"Parameters: {identifier}\n")
        results_file.write(f"fract_control_all: {fract_control_all}\n")
        results_file.write(f"fract_test_all: {fract_test_all}\n")
        results_file.write("\n")  # Add a newline for readability


        # Print confirmation
        '''
        print(f"Saved results to {full_path_envA} and {full_path_envB}")