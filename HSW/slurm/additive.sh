#!/bin/bash
#SBATCH --account=p32072
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=matlab_job_result_A21A22.out ## standard out and standard error goes to this file


module purge
module load python-miniconda3/4.12.0
source activate ratinabox

export PYTHONPATH="${PYTHONPATH}:~/Programming/RatInABox/"

python ~/Programming/RatInABox/HSW/additive_model/main2.py --balance_values 0.5 --balance_dist fixed --responsive_values 0.5 --responsive_type fixed --percent_place_cells .7
