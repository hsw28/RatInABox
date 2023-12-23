#!/bin/bash
#SBATCH --account=p32072
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=additive_errors.out ## standard out and standard error goes to this file


module purge
eval "$(conda shell.bash hook)"
source activate ratinabox

export PYTHONPATH="${PYTHONPATH}:/home/hsw967/Programming/RatInABox"
export PYTHONPATH="${PYTHONPATH}:/home/hsw967/Programming/Hannahs-CEBRAs"


python /home/hsw967/Programming/RatInABox/HSW/additive_model/main2.py --balance_values 0.5 --balance_dist fixed --responsive_values 0.5 --responsive_type fixed --percent_place_cells .7 work[[]]
