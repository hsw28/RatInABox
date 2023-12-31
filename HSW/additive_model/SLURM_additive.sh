#!/bin/bash
#SBATCH --account=p32072
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=243GB
#SBATCH --time=48:00:00
#SBATCH --output=SLURM_errors.out ## standard out and standard error goes to this file


printenv

module purge
eval "$(conda shell.bash hook)"
source activate ratinabox

export PYTHONPATH="${PYTHONPATH}:/home/hsw967/Programming/RatInABox"
export PYTHONPATH="${PYTHONPATH}:/home/hsw967/Programming/Hannahs-CEBRAs"


#python /home/hsw967/Programming/RatInABox/HSW/additive_model/main2.py --balance_values 1 --balance_dist additive --responsive_values .5 --responsive_type fixed --percent_place_cells .5 --num_iters 1 --optional_param work
python /home/hsw967/Programming/RatInABox/HSW/additive_model/main2.py --balance_values 1 --balance_dist additive --responsive_values 0,.2,.4,.6,.8,1 --responsive_type fixed --percent_place_cells .4,.6,.8,1 --num_iters 5 --optional_param work
#python /home/hsw967/Programming/RatInABox/HSW/additive_model/test.py
