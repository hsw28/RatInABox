#!/bin/bash
#SBATCH --account=p32072
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-9 ## number of jobs to run "in parallel"
#SBATCH --mem=220GB
#SBATCH --time=24:00:00
#SBATCH --job-name="sample_job_\${SLURM_ARRAY_TASK_ID}" ## use the task id in the name of the job
#SBATCH --output=DM_SLURM_out.%A_%a.out ## use the jobid (A) and the specific job index (a) to name your log file
#SBATCH --mail-type=ALL ## you can receive e-mail alerts from SLURM when your job begins and when your job finishes (completed, failed, etc)
#SBATCH --mail-user=hsw@northwestern.edu  ## your email


module purge
eval "$(conda shell.bash hook)"
source activate ratinabox

export PYTHONPATH="${PYTHONPATH}:/home/hsw967/Programming/RatInABox"
export PYTHONPATH="${PYTHONPATH}:/home/hsw967/Programming/Hannahs-CEBRAs"

# Read input arguments from file
IFS=$'\n' read -d '' -r -a input_args < input_args.txt

# Run the Python script with the argument corresponding to the SLURM_ARRAY_TASK_ID
python /home/hsw967/Programming/RatInABox/HSW/dependent_model/main2.py --balance_values 1 --balance_dist additive --responsive_values 0,.2,.4,.6,.8,1 --responsive_type fixed --percent_place_cells .2,.4,.6,.8,1 --num_iters 5 --optional_param work --filename ${input_args[$SLURM_ARRAY_TASK_ID]}
